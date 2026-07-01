"""Train a Waymo latent-space world model on frozen vector-tokenizer latents.

V0 design:

- freeze a trained Waymo vector tokenizer;
- encode full Waymo scenes to continuous latent z;
- train DreamerV4-style shortcut/flow dynamics in packed z space;
- evaluate by rolling out from the first observed 11 Waymo steps and decoding
  the full 80-step future.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler, random_split

WAYMO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WAYMO_ROOT.parent
CORE_ROOT = WAYMO_ROOT / "core"
DREAMER_ROOT = REPO_ROOT / "dreamer4"
for path in (REPO_ROOT, CORE_ROOT, DREAMER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from vector_tokenizer_decoder import (
        VectorBlockCausalTokenizerDecoder,
        VectorDecoderOutput,
        vector_tokenizer_reconstruction_loss,
    )
    from waymo_vector_dataset import WaymoVectorDataset
except ModuleNotFoundError:
    from waymo.core.vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from waymo.core.vector_tokenizer_decoder import (
        VectorBlockCausalTokenizerDecoder,
        VectorDecoderOutput,
        vector_tokenizer_reconstruction_loss,
    )
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset

from model import Dynamics, pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck


def seed_everything(seed: int) -> None:
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


def cleanup_distributed(ddp: bool, device: torch.device) -> None:
    if not ddp:
        return
    if device.type == "cuda":
        dist.barrier(device_ids=[int(device.index if device.index is not None else torch.cuda.current_device())])
    else:
        dist.barrier()
    dist.destroy_process_group()


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed_everything(info.seed)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_splits(dataset: WaymoVectorDataset, val_fraction: float, seed: int) -> tuple[Iterable, Iterable | None]:
    if val_fraction <= 0.0 or len(dataset) < 2:
        return dataset, None
    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_val = min(n_val, len(dataset) - 1)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def _slice_agents_time(agents: torch.Tensor, agent_mask: torch.Tensor, start: int, end: int) -> torch.Tensor:
    k = int(agent_mask.shape[-1])
    if agents.shape[1] == k:
        return agents[:, :, start:end]
    return agents[:, start:end]


def slice_time_window(
    batch: Dict[str, Any],
    time_window: int,
    *,
    random_start: bool = False,
) -> Dict[str, Any]:
    if time_window <= 0:
        return batch
    total_steps = int(batch["lights"].shape[1])
    if total_steps <= time_window:
        start = 0
    elif random_start:
        start = int(torch.randint(0, total_steps - int(time_window) + 1, (1,)).item())
    else:
        start = 0
    end = min(total_steps, start + int(time_window))

    out = dict(batch)
    out["agents"] = _slice_agents_time(batch["agents"], batch["agent_mask"], start, end)
    out["lights"] = batch["lights"][:, start:end]
    out["light_mask"] = batch["light_mask"][:, start:end]
    return out


def slice_future_batch(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    out = dict(batch)
    out["agents"] = _slice_agents_time(batch["agents"], batch["agent_mask"], start, end)
    out["lights"] = batch["lights"][:, start:end]
    out["light_mask"] = batch["light_mask"][:, start:end]
    return out


def tensor_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().float().item()) for k, v in metrics.items()}


def metric_order(metrics: Dict[str, Any]) -> list[str]:
    preferred = [
        "loss_total",
        "latent_mse_future",
        "flow_mse",
        "bootstrap_mse",
        "loss_emp",
        "loss_self",
        "sigma_mean",
        "agent_xy_mae_m",
        "agent_fde_mae_m",
        "focus_agent_xy_mae_m",
        "focus_agent_fde_m",
        "agent_speed_mae_mps",
        "agent_vxvy_mae_mps",
        "agent_yaw_mae_deg",
        "agent_valid_acc",
        "light_state_acc",
        "light_valid_acc",
    ]
    return [name for name in preferred if name in metrics] + sorted(set(metrics) - set(preferred))


def format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in metric_order(metrics))


def reduce_metric_dict(metrics: Dict[str, torch.Tensor], device: torch.device, ddp: bool) -> Dict[str, float]:
    names = metric_order(metrics)
    values = torch.tensor([float(metrics[name].detach().float().item()) for name in names], device=device)
    if ddp:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {name: float(value.item()) for name, value in zip(names, values)}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    model = model.module if hasattr(model, "module") else model
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _as_args_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, argparse.Namespace):
        return vars(value)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _arg(args: Dict[str, Any], name: str, default: Any) -> Any:
    return args[name] if name in args and args[name] is not None else default


def _infer_tokenizer_shapes(state: Dict[str, torch.Tensor]) -> tuple[int, int]:
    try:
        n_agents = int(state["decoder.agent_queries"].shape[0])
        n_lights = int(state["decoder.light_queries"].shape[0])
    except KeyError as exc:
        raise KeyError("Checkpoint does not look like a Waymo vector tokenizer checkpoint.") from exc
    return n_agents, n_lights


class FrozenWaymoVectorTokenizer(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def build_waymo_vector_tokenizer_from_args(args: Dict[str, Any], n_agents: int, n_lights: int) -> FrozenWaymoVectorTokenizer:
    encoder_kwargs = dict(
        d_model=int(_arg(args, "d_model", 256)),
        n_heads=int(_arg(args, "n_heads", 4)),
        depth=int(_arg(args, "depth", 4)),
        n_latents=int(_arg(args, "n_latents", 64)),
        d_bottleneck=int(_arg(args, "d_bottleneck", 64)),
        hidden_dim=int(_arg(args, "hidden_dim", 128)),
        dropout=float(_arg(args, "dropout", 0.0)),
        mlp_ratio=float(_arg(args, "mlp_ratio", 4.0)),
        time_every=int(_arg(args, "time_every", 1)),
        scale_pos_embeds=bool(_arg(args, "scale_pos_embeds", True)),
        bottleneck_output=str(_arg(args, "bottleneck_output", "tanh")),
    )
    encoder_variant = str(_arg(args, "encoder_variant", "repeat_map"))
    if encoder_variant == "repeat_map":
        encoder = VectorBlockCausalEncoder(**encoder_kwargs)
    elif encoder_variant == "static_map_query":
        encoder = VectorStaticMapQueryEncoder(
            **encoder_kwargs,
            map_depth=int(_arg(args, "map_depth", 2)),
            map_cross_every=int(_arg(args, "map_cross_every", 1)),
            map_query_tokens=str(_arg(args, "map_query_tokens", "latent_agent")),
        )
    else:
        raise ValueError(f"Unknown encoder_variant in tokenizer checkpoint: {encoder_variant!r}")

    agent_xy_loss = str(_arg(args, "agent_xy_loss", "smooth_l1"))
    decoder = VectorBlockCausalTokenizerDecoder(
        d_bottleneck=int(_arg(args, "d_bottleneck", 64)),
        d_model=int(_arg(args, "d_model", 256)),
        n_heads=int(_arg(args, "n_heads", 4)),
        depth=int(_arg(args, "decoder_depth", _arg(args, "depth", 4))),
        n_latents=int(_arg(args, "n_latents", 64)),
        n_agents=n_agents,
        n_lights=n_lights,
        dropout=float(_arg(args, "dropout", 0.0)),
        mlp_ratio=float(_arg(args, "mlp_ratio", 4.0)),
        time_every=int(_arg(args, "time_every", 1)),
        scale_pos_embeds=bool(_arg(args, "scale_pos_embeds", True)),
        use_agent_tokens=bool(_arg(args, "decoder_use_agent_tokens", False)),
        agent_token_mode=str(_arg(args, "decoder_agent_token_mode", "none")),
        predict_agent_xy_gmm=agent_xy_loss == "gmm",
    )
    return FrozenWaymoVectorTokenizer(encoder=encoder, decoder=decoder)


@torch.no_grad()
def load_frozen_waymo_vector_tokenizer(ckpt_path: str, device: torch.device) -> tuple[FrozenWaymoVectorTokenizer, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tok_args = _as_args_dict(ckpt.get("args", {}))
    state = ckpt["model"]
    n_agents, n_lights = _infer_tokenizer_shapes(state)
    tokenizer = build_waymo_vector_tokenizer_from_args(tok_args, n_agents=n_agents, n_lights=n_lights)
    tokenizer.load_state_dict(state, strict=True)
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)
    return tokenizer, tok_args


@torch.no_grad()
def encode_batch_z(tokenizer: FrozenWaymoVectorTokenizer, batch: Dict[str, Any]) -> torch.Tensor:
    out = tokenizer.encoder(
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    return out.z


def _emax_from_kmax(k_max: int) -> int:
    emax = int(round(math.log2(k_max)))
    assert (1 << emax) == k_max, "k_max must be a power of two"
    return emax


def _is_pow2(n: int) -> bool:
    return (n > 0) and ((n & (n - 1)) == 0)


def _sample_step_excluding_dmin(device: torch.device, bsz: int, time_steps: int, k_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    emax = _emax_from_kmax(k_max)
    step_idx = torch.randint(low=0, high=max(1, emax), size=(bsz, time_steps), device=device, dtype=torch.long)
    d = 1.0 / (1 << step_idx).to(torch.float32)
    return d, step_idx


def _sample_tau_for_step(
    device: torch.device,
    bsz: int,
    time_steps: int,
    k_max: int,
    step_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = (1 << step_idx).to(torch.long)
    u = torch.rand((bsz, time_steps), device=device, dtype=torch.float32)
    j_idx = torch.floor(u * k.to(torch.float32)).to(torch.long)
    tau = j_idx.to(torch.float32) / k.to(torch.float32)
    scale = torch.div(torch.tensor(k_max, device=device), k, rounding_mode="floor")
    tau_idx = j_idx * scale
    return tau, tau_idx


def dynamics_pretrain_loss(
    dynamics: torch.nn.Module,
    *,
    z1: torch.Tensor,
    k_max: int,
    b_self: int,
    step: int,
    bootstrap_start: int,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = z1.device
    bsz, time_steps = z1.shape[:2]
    b_self = max(0, min(int(b_self), bsz - 1))
    b_emp = bsz - b_self
    emax = _emax_from_kmax(k_max)

    step_idx_emp = torch.full((b_emp, time_steps), emax, device=device, dtype=torch.long)
    if b_self > 0:
        d_self, step_idx_self = _sample_step_excluding_dmin(device, b_self, time_steps, k_max)
        step_idx_full = torch.cat([step_idx_emp, step_idx_self], dim=0)
    else:
        d_self = torch.zeros((0, time_steps), device=device, dtype=torch.float32)
        step_idx_self = torch.zeros((0, time_steps), device=device, dtype=torch.long)
        step_idx_full = step_idx_emp

    sigma_full, sigma_idx_full = _sample_tau_for_step(device, bsz, time_steps, k_max, step_idx_full)
    sigma_emp = sigma_full[:b_emp]
    sigma_self = sigma_full[b_emp:]
    sigma_idx_self = sigma_idx_full[b_emp:]

    z0_full = torch.randn_like(z1)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
    z_tilde_self = z_tilde_full[b_emp:]

    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1

    z1_hat_full, _ = dynamics(None, step_idx_full, sigma_idx_full, z_tilde_full, act_mask=None, agent_tokens=None)
    z1_hat_emp = z1_hat_full[:b_emp]
    z1_hat_self = z1_hat_full[b_emp:]

    flow_per = (z1_hat_emp.float() - z1[:b_emp].float()).pow(2).mean(dim=(2, 3))
    loss_emp = (flow_per * w_emp).mean()

    boot_mse = torch.zeros((), device=device, dtype=torch.float32)
    loss_self = torch.zeros((), device=device, dtype=torch.float32)
    if b_self > 0 and step >= bootstrap_start:
        d_half = d_self / 2.0
        step_idx_half = step_idx_self + 1
        sigma_plus = sigma_self + d_half
        sigma_idx_plus = sigma_idx_self + (torch.tensor(k_max, device=device, dtype=torch.float32) * d_half).to(torch.long)

        z1_hat_half1, _ = dynamics(None, step_idx_half, sigma_idx_self, z_tilde_self, act_mask=None, agent_tokens=None)
        b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None]

        z1_hat_half2, _ = dynamics(None, step_idx_half, sigma_idx_plus, z_prime.to(z_tilde_self.dtype), act_mask=None, agent_tokens=None)
        b_doubleprime = (z1_hat_half2.float() - z_prime.float()) / (1.0 - sigma_plus).clamp_min(1e-6)[..., None, None]

        vhat_sigma = (z1_hat_self.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        vbar_target = ((b_prime + b_doubleprime) / 2.0).detach()
        boot_per = (1.0 - sigma_self).pow(2) * (vhat_sigma - vbar_target).pow(2).mean(dim=(2, 3))
        loss_self = (boot_per * w_self).mean()
        boot_mse = boot_per.mean()

    loss = ((loss_emp * b_emp) + (loss_self * b_self)) / bsz
    return loss, {
        "loss_total": loss.detach(),
        "flow_mse": flow_per.mean().detach(),
        "bootstrap_mse": boot_mse.detach(),
        "loss_emp": loss_emp.detach(),
        "loss_self": loss_self.detach(),
        "sigma_mean": sigma_full.mean().detach(),
    }


def make_tau_schedule(*, k_max: int, schedule: str, d: Optional[float] = None) -> Dict[str, Any]:
    assert _is_pow2(k_max), "k_max must be a power of two"
    if schedule == "finest":
        k = k_max
    elif schedule == "shortcut":
        assert d is not None, "shortcut schedule requires eval_d"
        inv = int(round(1.0 / float(d)))
        assert _is_pow2(inv), "eval_d must be 1/(power of two)"
        assert inv <= k_max, "eval_d must be >= 1/k_max"
        assert (k_max % inv) == 0, "k_max must be divisible by 1/eval_d"
        k = inv
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    e = int(round(math.log2(k)))
    scale = k_max // k
    tau = [i / k for i in range(k)] + [1.0]
    tau_idx = [i * scale for i in range(k)] + [k_max]
    return dict(K=k, e=e, scale=scale, tau=tau, tau_idx=tau_idx, dt=1.0 / k, schedule=schedule, d=1.0 / k)


@torch.no_grad()
def sample_one_timestep_packed(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,
    k_max: int,
    sched: Dict[str, Any],
    max_rollout_window: int,
) -> torch.Tensor:
    if max_rollout_window > 0:
        past_keep = max(1, int(max_rollout_window) - 1)
        past_packed = past_packed[:, -past_keep:]

    device = past_packed.device
    dtype = past_packed.dtype
    bsz, past_t, n_spatial, d_spatial = past_packed.shape
    k = int(sched["K"])
    e = int(sched["e"])
    tau = sched["tau"]
    tau_idx = sched["tau_idx"]
    dt = float(sched["dt"])
    emax = _emax_from_kmax(k_max)

    z = torch.randn((bsz, 1, n_spatial, d_spatial), device=device, dtype=dtype)
    step_idxs = torch.full((bsz, past_t + 1), emax, device=device, dtype=torch.long)
    step_idxs[:, -1] = e
    signal_idxs = torch.full((bsz, past_t + 1), k_max - 1, device=device, dtype=torch.long)

    for i in range(k):
        tau_i = float(tau[i])
        signal_idxs[:, -1] = int(tau_idx[i])
        packed_seq = torch.cat([past_packed, z], dim=1)
        x1_hat_full, _ = dyn(None, step_idxs, signal_idxs, packed_seq, act_mask=None, agent_tokens=None)
        x1_hat = x1_hat_full[:, -1:, :, :]
        denom = max(1e-4, 1.0 - tau_i)
        b = (x1_hat.float() - z.float()) / denom
        z = (z.float() + b * dt).to(dtype)
    return z[:, 0]


@torch.no_grad()
def sample_autoregressive_packed_sequence(
    dyn: Dynamics,
    *,
    z_gt_packed: torch.Tensor,
    ctx_length: int,
    horizon: int,
    k_max: int,
    sched: Dict[str, Any],
    max_rollout_window: int,
) -> torch.Tensor:
    total = int(z_gt_packed.shape[1])
    ctx_length = max(1, min(int(ctx_length), total - 1))
    horizon = min(int(horizon), total - ctx_length)
    outs = [z_gt_packed[:, t] for t in range(ctx_length)]
    for _ in range(horizon):
        past = torch.stack(outs, dim=1)
        z_next = sample_one_timestep_packed(
            dyn,
            past_packed=past,
            k_max=k_max,
            sched=sched,
            max_rollout_window=max_rollout_window,
        )
        outs.append(z_next)
    return torch.stack(outs, dim=1)


def slice_decoder_output(pred: VectorDecoderOutput, start: int, end: int) -> VectorDecoderOutput:
    return replace(
        pred,
        agent_continuous=pred.agent_continuous[:, start:end],
        agent_valid_logits=pred.agent_valid_logits[:, start:end],
        light_state_logits=pred.light_state_logits[:, start:end],
        light_valid_logits=pred.light_valid_logits[:, start:end],
        agent_tokens=pred.agent_tokens[:, start:end],
        light_tokens=pred.light_tokens[:, start:end],
        token_mask=pred.token_mask[:, start:end],
        agent_xy_gmm=None if pred.agent_xy_gmm is None else pred.agent_xy_gmm[:, start:end],
    )


def reconstruction_metrics(
    pred: VectorDecoderOutput,
    batch: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    _, metrics = vector_tokenizer_reconstruction_loss(
        pred,
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
        agent_xy_weight=args.agent_xy_weight,
        agent_vel_weight=args.agent_vel_weight,
        agent_yaw_weight=args.agent_yaw_weight,
        agent_valid_weight=args.agent_valid_weight,
        light_state_weight=args.light_state_weight,
        light_valid_weight=args.light_valid_weight,
        agent_delta_xy_weight=args.agent_delta_xy_weight,
        agent_fde_xy_weight=args.agent_fde_xy_weight,
        agent_kinematic_xy_weight=args.agent_kinematic_xy_weight,
        agent_speed_yaw_kinematic_weight=args.agent_speed_yaw_kinematic_weight,
        kinematic_dt=args.kinematic_dt,
        focus_agent_weight=args.focus_agent_weight,
        agent_xy_loss=args.agent_xy_loss,
        agent_xy_parameterization=args.agent_xy_parameterization,
    )
    return metrics


@torch.no_grad()
def evaluate(
    dyn: torch.nn.Module,
    tokenizer: FrozenWaymoVectorTokenizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    ddp: bool,
) -> Dict[str, float]:
    was_training = dyn.training
    dyn.eval()
    sched = make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    totals: Dict[str, float] = {}
    count = 0

    for batch in loader:
        batch = slice_time_window(move_batch(batch, device), args.eval_seq_len, random_start=False)
        z_gt = encode_batch_z(tokenizer, batch)
        z_gt_packed = pack_bottleneck_to_spatial(z_gt, n_spatial=args.n_spatial, k=args.packing_factor)
        z_pred_packed = sample_autoregressive_packed_sequence(
            unwrap_model(dyn),
            z_gt_packed=z_gt_packed,
            ctx_length=args.eval_ctx,
            horizon=args.eval_horizon,
            k_max=args.k_max,
            sched=sched,
            max_rollout_window=args.max_rollout_window,
        )
        z_pred = unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
        decoded = tokenizer.decoder(
            z_pred,
            agent_mask=batch["agent_mask"],
            light_mask=batch["light_mask"][:, : z_pred.shape[1]],
        )

        score_start = min(int(args.eval_ctx), int(z_pred.shape[1]) - 1)
        score_end = int(z_pred.shape[1])
        decoded_future = slice_decoder_output(decoded, score_start, score_end)
        batch_future = slice_future_batch(batch, score_start, score_end)
        metrics = reconstruction_metrics(decoded_future, batch_future, args)
        metrics["latent_mse_future"] = (
            z_pred_packed[:, score_start:score_end].float() - z_gt_packed[:, score_start:score_end].float()
        ).pow(2).mean()
        values = tensor_metrics(metrics)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if args.eval_max_batches > 0 and count >= args.eval_max_batches:
            break

    names = metric_order(totals)
    packed = torch.tensor([float(count)] + [totals.get(name, 0.0) for name in names], device=device, dtype=torch.float64)
    if ddp:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(packed[0].item()))
    if was_training:
        dyn.train()
    return {name: float(packed[i + 1].item() / total_count) for i, name in enumerate(names)}


def save_ckpt(path: Path, *, dyn: torch.nn.Module, opt: torch.optim.Optimizer, scaler: GradScaler, args: argparse.Namespace, step: int, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "step": int(step),
            "epoch": int(epoch),
            "dynamics": unwrap_model(dyn).state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
        },
        tmp,
    )
    tmp.replace(path)


def load_ckpt(path: Path, *, dyn: torch.nn.Module, opt: torch.optim.Optimizer, scaler: GradScaler) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    unwrap_model(dyn).load_state_dict(ckpt["dynamics"], strict=True)
    opt.load_state_dict(ckpt["opt"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def train(args: argparse.Namespace) -> None:
    ddp, rank, world_size, local_rank = init_distributed()
    if ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed + rank)

    dataset = WaymoVectorDataset(args.data_dir)
    if args.val_data_dir is not None:
        train_ds: Iterable = dataset
        val_ds: Iterable | None = WaymoVectorDataset(args.val_data_dir)
    else:
        train_ds, val_ds = make_splits(dataset, args.val_fraction, args.seed)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
        collate_fn=_collate,
    )

    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=worker_init_fn,
            collate_fn=_collate,
        )

    tokenizer, tok_args = load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
    d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor != 0:
        raise ValueError(f"n_latents={n_latents} must be divisible by packing_factor={args.packing_factor}")
    args.n_spatial = n_latents // args.packing_factor
    args.d_spatial = d_bottleneck * args.packing_factor

    dyn = Dynamics(
        d_model=args.d_model_dyn,
        d_bottleneck=d_bottleneck,
        d_spatial=args.d_spatial,
        n_spatial=args.n_spatial,
        n_register=args.n_register,
        n_agent=0,
        n_heads=args.n_heads,
        depth=args.dyn_depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        space_mode="wm_agent_isolated",
        scale_pos_embeds=args.scale_pos_embeds,
    ).to(device)

    if args.compile:
        dyn = torch.compile(dyn)
    if ddp:
        dyn = torch.nn.parallel.DistributedDataParallel(
            dyn,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    opt = torch.optim.AdamW(dyn.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": torch.float32}[args.amp_dtype]
    scaler = GradScaler(device="cuda", enabled=(use_amp and amp_dtype == torch.float16))

    step = 0
    start_epoch = 0
    if args.resume is not None:
        step, start_epoch = load_ckpt(Path(args.resume), dyn=dyn, opt=opt, scaler=scaler)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is not installed; run without --wandb or install wandb.") from exc
        if is_rank0():
            wandb_run = wandb
            wandb_run.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity, config=vars(args))

    if is_rank0():
        dyn_params = sum(p.numel() for p in unwrap_model(dyn).parameters() if p.requires_grad)
        tok_params = sum(p.numel() for p in tokenizer.parameters())
        print(
            f"device={device} ddp={ddp} world_size={world_size} amp={args.amp_dtype} "
            f"train={len(train_ds)} val={0 if val_ds is None else len(val_ds)}"
        )
        print(
            f"tokenizer={args.tokenizer_ckpt} n_latents={n_latents} d_bottleneck={d_bottleneck} "
            f"packing={args.packing_factor} n_spatial={args.n_spatial} d_spatial={args.d_spatial}"
        )
        print(
            f"dynamics d_model={args.d_model_dyn} depth={args.dyn_depth} heads={args.n_heads} "
            f"seq_len={args.seq_len} max_rollout_window={args.max_rollout_window} "
            f"eval_ctx={args.eval_ctx} eval_horizon={args.eval_horizon}"
        )
        print(f"parameters dynamics={dyn_params:,} frozen_tokenizer={tok_params:,}")

    t0 = time.time()
    latest = Path(args.ckpt_dir) / "latest.pt"
    stop = False
    last_epoch = start_epoch
    grad_accum = max(1, int(args.grad_accum))

    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            last_epoch = epoch + 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            dyn.train()

            for batch_idx, batch in enumerate(train_loader):
                if step >= args.max_steps:
                    stop = True
                    break
                batch = slice_time_window(move_batch(batch, device), args.seq_len, random_start=args.random_time_window_start)

                with torch.no_grad():
                    z = encode_batch_z(tokenizer, batch)
                    z_packed = pack_bottleneck_to_spatial(z, n_spatial=args.n_spatial, k=args.packing_factor)

                b_self = int(round(z_packed.shape[0] * args.self_fraction))
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss, metrics = dynamics_pretrain_loss(
                        dyn,
                        z1=z_packed,
                        k_max=args.k_max,
                        b_self=b_self,
                        step=step,
                        bootstrap_start=args.bootstrap_start,
                    )
                    loss = loss / grad_accum

                scaler.scale(loss).backward()
                should_step = ((batch_idx + 1) % grad_accum == 0)
                if should_step:
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(dyn.parameters(), args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1

                    should_log = step == 1 or (args.log_every > 0 and step % args.log_every == 0)
                    if should_log:
                        values = reduce_metric_dict(metrics, device, ddp)
                        if is_rank0():
                            elapsed = max(1e-6, time.time() - t0)
                            print(f"step={step} epoch={epoch + 1} {format_metrics(values)} steps_per_sec={step / elapsed:.3f}")
                        if is_rank0() and wandb_run is not None:
                            wandb_run.log({f"train/{k}": v for k, v in values.items()}, step=step)

                    if val_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                        val_metrics = evaluate(dyn, tokenizer, val_loader, device, args, ddp=ddp)
                        if is_rank0():
                            print(f"eval step={step} {format_metrics(val_metrics)}")
                        if is_rank0() and wandb_run is not None:
                            wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)

                    if is_rank0() and args.save_every > 0 and step % args.save_every == 0:
                        save_ckpt(Path(args.ckpt_dir) / f"step_{step:08d}.pt", dyn=dyn, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch)
                        save_ckpt(latest, dyn=dyn, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch)

            if stop:
                break
            if is_rank0():
                save_ckpt(latest, dyn=dyn, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch + 1)

        if stop:
            break

    if is_rank0():
        save_ckpt(Path(args.ckpt_dir) / f"final_step_{step:08d}.pt", dyn=dyn, opt=opt, scaler=scaler, args=args, step=step, epoch=last_epoch)
    if is_rank0() and wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(ddp, device)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Waymo latent-space world model.")
    p.add_argument("--data_dir", type=str, nargs="+", required=True)
    p.add_argument("--val_data_dir", type=str, nargs="+", default=None)
    p.add_argument("--tokenizer_ckpt", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, default="/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/waymo_world_model")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--seq_len", type=int, default=100, help="Training time window. 100 keeps a full 91-step Waymo scene.")
    p.add_argument("--random_time_window_start", action="store_true")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--d_model_dyn", type=int, default=512)
    p.add_argument("--dyn_depth", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=4)
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--n_register", type=int, default=8)
    p.add_argument("--scale_pos_embeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action="store_true")

    p.add_argument("--k_max", type=int, default=64)
    p.add_argument("--bootstrap_start", type=int, default=0)
    p.add_argument("--self_fraction", type=float, default=0.857142857)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp_dtype", choices=["fp16", "bf16", "none"], default="bf16")

    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_max_batches", type=int, default=8)
    p.add_argument("--eval_seq_len", type=int, default=100)
    p.add_argument("--eval_ctx", type=int, default=11)
    p.add_argument("--eval_horizon", type=int, default=80)
    p.add_argument("--max_rollout_window", type=int, default=100)
    p.add_argument("--eval_schedule", choices=["finest", "shortcut"], default="shortcut")
    p.add_argument("--eval_d", type=float, default=0.25)

    p.add_argument("--agent_xy_weight", type=float, default=1.0)
    p.add_argument("--agent_xy_loss", choices=["smooth_l1", "gmm"], default="smooth_l1")
    p.add_argument("--agent_xy_parameterization", choices=["absolute", "delta"], default="absolute")
    p.add_argument("--agent_vel_weight", type=float, default=0.5)
    p.add_argument("--agent_yaw_weight", type=float, default=0.5)
    p.add_argument("--agent_valid_weight", type=float, default=0.2)
    p.add_argument("--light_state_weight", type=float, default=0.5)
    p.add_argument("--light_valid_weight", type=float, default=0.1)
    p.add_argument("--agent_delta_xy_weight", type=float, default=0.0)
    p.add_argument("--agent_fde_xy_weight", type=float, default=0.0)
    p.add_argument("--agent_kinematic_xy_weight", type=float, default=0.0)
    p.add_argument("--agent_speed_yaw_kinematic_weight", type=float, default=0.0)
    p.add_argument("--kinematic_dt", type=float, default=0.1)
    p.add_argument("--focus_agent_weight", type=float, default=1.0)

    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="waymo-world-model")
    p.add_argument("--wandb_run_name", type=str, default="waymo_latent_dynamics_v0")
    p.add_argument("--wandb_entity", type=str, default=None)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
