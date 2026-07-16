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
    from focus_agent_tokenizer import FocusAgentTokenizer, FocusTokenizerOutput, focus_tokenizer_loss
    from vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from vector_tokenizer_decoder import (
        VectorBlockCausalTokenizerDecoder,
        VectorDecoderOutput,
        vector_tokenizer_reconstruction_loss,
    )
    from waymo_vector_dataset import WaymoVectorDataset
except ModuleNotFoundError:
    from waymo.core.focus_agent_tokenizer import FocusAgentTokenizer, FocusTokenizerOutput, focus_tokenizer_loss
    from waymo.core.vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from waymo.core.vector_tokenizer_decoder import (
        VectorBlockCausalTokenizerDecoder,
        VectorDecoderOutput,
        vector_tokenizer_reconstruction_loss,
    )
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset

from model import Dynamics, FocusFiLMDynamics, pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck


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


def agents_to_btkf(agents: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    k = int(agent_mask.shape[-1])
    if agents.shape[1] == k:
        return agents.transpose(1, 2).contiguous()
    return agents


def wrap_angle_rad(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


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


def _select_action_slots(batch: Dict[str, Any], source: str) -> torch.Tensor:
    agents = batch["agents"]
    agent_mask = batch["agent_mask"]
    bsz = int(agents.shape[0])
    device = agents.device
    if source == "focus":
        return torch.zeros((bsz,), device=device, dtype=torch.long)
    if source != "sdc":
        raise ValueError(f"Unknown ego_action_source={source!r}; expected 'sdc' or 'focus'.")
    if "agent_src_indices" not in batch or "original_sdc_src_index" not in batch:
        return torch.zeros((bsz,), device=device, dtype=torch.long)
    src = batch["agent_src_indices"].to(device=device)
    sdc = batch["original_sdc_src_index"].to(device=device).view(bsz, 1)
    matches = (src == sdc) & agent_mask.to(device=device, dtype=torch.bool)
    has_match = matches.any(dim=1)
    return torch.where(has_match, matches.float().argmax(dim=1).to(torch.long), torch.zeros_like(has_match, dtype=torch.long))


def build_ego_action_features(
    batch: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not args.use_ego_actions:
        return None, None, None
    agents_btkf = agents_to_btkf(batch["agents"], batch["agent_mask"])
    bsz, time_steps = agents_btkf.shape[:2]
    device = agents_btkf.device
    slots = _select_action_slots(batch, args.ego_action_source)
    gather = slots.view(bsz, 1, 1, 1).expand(-1, time_steps, 1, agents_btkf.shape[-1])
    ego = agents_btkf.gather(dim=2, index=gather).squeeze(2)

    valid = ego[..., 5] > 0.5
    prev_xy = torch.cat([ego[:, :1, 0:2], ego[:, :-1, 0:2]], dim=1)
    prev_yaw = torch.cat([ego[:, :1, 6], ego[:, :-1, 6]], dim=1)
    prev_valid = torch.cat([valid[:, :1], valid[:, :-1]], dim=1)
    step_valid = valid & prev_valid

    delta_xy = ego[..., 0:2] - prev_xy
    delta_yaw = wrap_angle_rad(ego[..., 6] - prev_yaw).unsqueeze(-1)
    speed = ego[..., 2:3]
    vxvy = ego[..., 3:5]
    if args.ego_action_normalization == "scaled":
        delta_xy = delta_xy / float(args.ego_action_xy_scale)
        delta_yaw = delta_yaw / float(args.ego_action_yaw_scale)
        speed = speed / float(args.ego_action_speed_scale)
        vxvy = vxvy / float(args.ego_action_speed_scale)
    valid_f = valid.to(dtype=ego.dtype).unsqueeze(-1)

    actions = torch.zeros((bsz, time_steps, 16), device=device, dtype=ego.dtype)
    actions[..., 0:2] = delta_xy
    actions[..., 2:3] = delta_yaw
    actions[..., 3:4] = speed
    actions[..., 4:6] = vxvy
    actions[..., 6:7] = valid_f
    actions[:, 0, 0:3] = 0.0
    actions = actions * valid_f

    mask = torch.zeros_like(actions)
    mask[..., 0:7] = valid_f
    mask[..., 0:3] = mask[..., 0:3] * step_valid.to(dtype=ego.dtype).unsqueeze(-1)
    return actions, mask, slots


def build_agent_loss_weight_multiplier(
    batch: Dict[str, Any],
    args: argparse.Namespace,
    *,
    action_slots: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if args.agent_far_weight >= 0.999:
        return None
    agents_btkf = agents_to_btkf(batch["agents"], batch["agent_mask"])
    bsz, time_steps, n_agents = agents_btkf.shape[:3]
    device = agents_btkf.device
    source_slots = action_slots
    if source_slots is None:
        source_slots = _select_action_slots(batch, args.agent_distance_source)
    gather = source_slots.view(bsz, 1, 1, 1).expand(-1, time_steps, 1, agents_btkf.shape[-1])
    source = agents_btkf.gather(dim=2, index=gather).squeeze(2)
    dist = (agents_btkf[..., 0:2] - source[:, :, None, 0:2]).norm(dim=-1)
    radius = max(1e-6, float(args.agent_near_radius_m))
    near = torch.exp(-((dist / radius) ** 2))
    weights = float(args.agent_far_weight) + (1.0 - float(args.agent_far_weight)) * near
    weights = weights * batch["agent_mask"].to(device=device, dtype=weights.dtype)[:, None, :]
    weights[..., 0] = torch.maximum(weights[..., 0], torch.ones_like(weights[..., 0]))
    if "agent_tracks_to_predict" in batch:
        ttp = batch["agent_tracks_to_predict"].to(device=device, dtype=torch.bool)
        weights = torch.where(ttp[:, None, :], torch.ones_like(weights), weights)
    if source_slots is not None:
        src_mask = torch.nn.functional.one_hot(source_slots.clamp_min(0), num_classes=n_agents).to(device=device, dtype=torch.bool)
        weights = torch.where(src_mask[:, None, :], torch.ones_like(weights), weights)
    return weights


def tensor_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().float().item()) for k, v in metrics.items()}


def metric_order(metrics: Dict[str, Any]) -> list[str]:
    preferred = [
        "loss_total",
        "latent_mse_future",
        "tf_onestep_mse",
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


def freeze_unused_action_mlp(model: torch.nn.Module) -> int:
    """Freeze action MLP weights for latent pretraining runs that pass actions=None."""
    action_encoder = getattr(unwrap_model(model), "action_encoder", None)
    if action_encoder is None:
        return 0
    frozen = 0
    for name in ("fc1", "fc2"):
        layer = getattr(action_encoder, name, None)
        if layer is None:
            continue
        for param in layer.parameters():
            param.requires_grad_(False)
            frozen += param.numel()
    return frozen


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


class FrozenWaymoFocusTokenizer(torch.nn.Module):
    def __init__(self, model: FocusAgentTokenizer):
        super().__init__()
        self.model = model
        self.n_latents = 1
        self.d_bottleneck = model.d_model if model.representation_type == "agent_token" else model.d_latent


def build_waymo_focus_tokenizer_from_args(args: Dict[str, Any], representation: str) -> FrozenWaymoFocusTokenizer:
    d_latent = int(_arg(args, "d_latent", 16 if representation == "latent_z16" else 64))
    model = FocusAgentTokenizer(
        representation=representation,
        d_model=int(_arg(args, "d_model", 256)),
        d_latent=d_latent,
        hidden_dim=int(_arg(args, "hidden_dim", 128)),
        n_heads=int(_arg(args, "n_heads", 4)),
        depth=int(_arg(args, "depth", 4)),
        decoder_depth=int(_arg(args, "decoder_depth", 2)),
        map_depth=int(_arg(args, "map_depth", 2)),
        dropout=float(_arg(args, "dropout", 0.05)),
        mlp_ratio=float(_arg(args, "mlp_ratio", 4.0)),
        scale_pos_embeds=bool(_arg(args, "scale_pos_embeds", True)),
    )
    return FrozenWaymoFocusTokenizer(model)


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
        attend_map=bool(_arg(args, "decoder_attend_map", False)),
        map_cross_every=int(_arg(args, "decoder_map_cross_every", 1)),
        map_query_tokens=str(_arg(args, "decoder_map_query_tokens", "all")),
        predict_agent_xy_gmm=agent_xy_loss == "gmm",
    )
    return FrozenWaymoVectorTokenizer(encoder=encoder, decoder=decoder)


@torch.no_grad()
def load_frozen_waymo_vector_tokenizer(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tok_args = _as_args_dict(ckpt.get("args", {}))
    state = ckpt["model"]
    if ckpt.get("format") == "waymo_focus_tokenizer_v1":
        representation = str(ckpt.get("representation", _arg(tok_args, "representation", "agent_token")))
        tokenizer = build_waymo_focus_tokenizer_from_args(tok_args, representation)
        tokenizer.model.load_state_dict(state, strict=True)
        tok_args.update(
            tokenizer_kind="focus",
            representation=representation,
            n_latents=tokenizer.n_latents,
            d_bottleneck=tokenizer.d_bottleneck,
        )
    else:
        n_agents, n_lights = _infer_tokenizer_shapes(state)
        tokenizer = build_waymo_vector_tokenizer_from_args(tok_args, n_agents=n_agents, n_lights=n_lights)
        tokenizer.load_state_dict(state, strict=True)
        tok_args.update(tokenizer_kind="vector")
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)
    return tokenizer, tok_args


@torch.no_grad()
def encode_batch_dynamics_inputs(
    tokenizer: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    return_map: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if isinstance(tokenizer, FrozenWaymoFocusTokenizer):
        representation, map_tokens, map_mask = tokenizer.model.encode(
            agents=batch["agents"],
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_mask=batch["map_mask"],
        )
        return representation, (map_tokens if return_map else None), (map_mask if return_map else None)
    out = tokenizer.encoder(
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    map_tokens = out.map_tokens
    if return_map and map_tokens.dim() == 4:
        map_tokens = map_tokens[:, 0]
    return out.z, (map_tokens if return_map else None), (out.map_token_mask if return_map else None)


@torch.no_grad()
def encode_batch_z(tokenizer: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    z, _, _ = encode_batch_dynamics_inputs(tokenizer, batch, return_map=False)
    return z


def tokenizer_map_memory_dim(tokenizer: torch.nn.Module) -> int:
    if isinstance(tokenizer, FrozenWaymoFocusTokenizer):
        return int(tokenizer.model.d_model)
    return int(tokenizer.encoder.d_model)


@torch.no_grad()
def decoder_map_kwargs(tokenizer: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(tokenizer, FrozenWaymoFocusTokenizer):
        return {}
    if not getattr(tokenizer.decoder, "attend_map", False):
        return {}
    map_tokens, map_mask = tokenizer.encoder.encode_static_map(batch["map_polylines"], batch["map_mask"])
    return {"encoder_map_tokens": map_tokens, "encoder_map_mask": map_mask}


def decode_batch_z(tokenizer: torch.nn.Module, z: torch.Tensor, batch: Dict[str, Any]) -> Any:
    if isinstance(tokenizer, FrozenWaymoFocusTokenizer):
        with torch.no_grad():
            map_tokens, map_mask = tokenizer.model.encode_static_map(batch["map_polylines"], batch["map_mask"])
        continuous, valid_logits = tokenizer.model.decode(z, map_tokens=map_tokens, map_mask=map_mask)
        return FocusTokenizerOutput(
            representation=z,
            agent_continuous=continuous,
            agent_valid_logits=valid_logits,
            map_tokens=map_tokens,
            map_mask=map_mask,
        )
    return tokenizer.decoder(
        z,
        agent_mask=batch["agent_mask"],
        light_mask=batch["light_mask"][:, : z.shape[1]],
        **decoder_map_kwargs(tokenizer, batch),
    )


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
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    k_max: int,
    b_self: int,
    step: int,
    bootstrap_start: int,
    map_tokens: Optional[torch.Tensor] = None,
    map_mask: Optional[torch.Tensor] = None,
    return_pred: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
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
    actions_self = None if actions is None else actions[b_emp:]
    act_mask_self = None if act_mask is None or act_mask.dim() < 3 else act_mask[b_emp:]
    map_tokens_self = None if map_tokens is None else map_tokens[b_emp:]
    map_mask_self = None if map_mask is None else map_mask[b_emp:]

    z0_full = torch.randn_like(z1)
    z_tilde_full = (1.0 - sigma_full)[..., None, None] * z0_full + sigma_full[..., None, None] * z1
    z_tilde_self = z_tilde_full[b_emp:]

    w_emp = 0.9 * sigma_emp + 0.1
    w_self = 0.9 * sigma_self + 0.1

    z1_hat_full, _ = dynamics(
        actions,
        step_idx_full,
        sigma_idx_full,
        z_tilde_full,
        act_mask=act_mask,
        agent_tokens=None,
        map_tokens=map_tokens,
        map_mask=map_mask,
    )
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

        z1_hat_half1, _ = dynamics(
            actions_self,
            step_idx_half,
            sigma_idx_self,
            z_tilde_self,
            act_mask=act_mask_self,
            agent_tokens=None,
            map_tokens=map_tokens_self,
            map_mask=map_mask_self,
        )
        b_prime = (z1_hat_half1.float() - z_tilde_self.float()) / (1.0 - sigma_self).clamp_min(1e-6)[..., None, None]
        z_prime = z_tilde_self.float() + b_prime * d_half[..., None, None]

        z1_hat_half2, _ = dynamics(
            actions_self,
            step_idx_half,
            sigma_idx_plus,
            z_prime.to(z_tilde_self.dtype),
            act_mask=act_mask_self,
            agent_tokens=None,
            map_tokens=map_tokens_self,
            map_mask=map_mask_self,
        )
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
    }, (z1_hat_full if return_pred else None)


def _flatten_time_windows(x: torch.Tensor, window: int) -> torch.Tensor:
    """Return sliding time windows as (B * N, window, ...)."""
    chunks = [x[:, start : start + window] for start in range(int(x.shape[1]) - window + 1)]
    return torch.cat(chunks, dim=0)


def tf_onestep_loss(
    dynamics: torch.nn.Module,
    *,
    z1: torch.Tensor,
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    k_max: int,
    context: int,
    map_tokens: Optional[torch.Tensor] = None,
    map_mask: Optional[torch.Tensor] = None,
    return_pred: bool = False,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Teacher-forced next-z baseline.

    Each training example uses real z[t:t+context] as context, predicts only
    z[t+context], and then the next window is again formed from real z.
    """
    device = z1.device
    bsz, time_steps = z1.shape[:2]
    context = int(context)
    if context < 1:
        raise ValueError(f"tf_context must be >= 1, got {context}")
    if time_steps <= context:
        raise ValueError(f"Need seq_len > tf_context for tf_onestep, got seq_len={time_steps} context={context}")

    n_windows = time_steps - context
    past = _flatten_time_windows(z1[:, : time_steps - 1], context)
    target = z1[:, context:].transpose(0, 1).reshape(bsz * n_windows, *z1.shape[2:])
    noise = torch.randn_like(target).unsqueeze(1)
    packed_seq = torch.cat([past, noise], dim=1)

    emax = _emax_from_kmax(k_max)
    step_idxs = torch.full((bsz * n_windows, context + 1), emax, device=device, dtype=torch.long)
    signal_idxs = torch.full((bsz * n_windows, context + 1), k_max - 1, device=device, dtype=torch.long)
    step_idxs[:, -1] = 0
    signal_idxs[:, -1] = 0

    actions_seq = None
    if actions is not None:
        actions_seq = _flatten_time_windows(actions, context + 1)
    act_mask_seq = None
    if act_mask is not None and act_mask.dim() >= 3:
        act_mask_seq = _flatten_time_windows(act_mask, context + 1)

    map_tokens_seq = None if map_tokens is None else map_tokens.repeat(n_windows, 1, 1)
    map_mask_seq = None if map_mask is None else map_mask.repeat(n_windows, 1)

    pred_full, _ = dynamics(
        actions_seq,
        step_idxs,
        signal_idxs,
        packed_seq,
        act_mask=act_mask_seq,
        agent_tokens=None,
        map_tokens=map_tokens_seq,
        map_mask=map_mask_seq,
    )
    pred = pred_full[:, -1]
    per = (pred.float() - target.float()).pow(2).mean(dim=(1, 2))
    loss = per.mean()

    pred_seq = None
    if return_pred:
        pred_seq = pred.view(n_windows, bsz, *z1.shape[2:]).transpose(0, 1).contiguous()
    return loss, {
        "loss_total": loss.detach(),
        "tf_onestep_mse": per.mean().detach(),
    }, pred_seq


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
    actions_seq: Optional[torch.Tensor],
    act_mask_seq: Optional[torch.Tensor],
    map_tokens: Optional[torch.Tensor],
    map_mask: Optional[torch.Tensor],
    k_max: int,
    sched: Dict[str, Any],
    max_rollout_window: int,
) -> torch.Tensor:
    if max_rollout_window > 0:
        past_keep = max(1, int(max_rollout_window) - 1)
        past_packed = past_packed[:, -past_keep:]
        if actions_seq is not None:
            actions_seq = actions_seq[:, -(past_keep + 1) :]
        if act_mask_seq is not None and act_mask_seq.dim() == 3:
            act_mask_seq = act_mask_seq[:, -(past_keep + 1) :]

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
        x1_hat_full, _ = dyn(
            actions_seq,
            step_idxs,
            signal_idxs,
            packed_seq,
            act_mask=act_mask_seq,
            agent_tokens=None,
            map_tokens=map_tokens,
            map_mask=map_mask,
        )
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
    actions: Optional[torch.Tensor],
    act_mask: Optional[torch.Tensor],
    map_tokens: Optional[torch.Tensor],
    map_mask: Optional[torch.Tensor],
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
        next_t = past.shape[1]
        actions_seq = None if actions is None else actions[:, : next_t + 1]
        act_mask_seq = None if act_mask is None else act_mask[:, : next_t + 1]
        z_next = sample_one_timestep_packed(
            dyn,
            past_packed=past,
            actions_seq=actions_seq,
            act_mask_seq=act_mask_seq,
            map_tokens=map_tokens,
            map_mask=map_mask,
            k_max=k_max,
            sched=sched,
            max_rollout_window=max_rollout_window,
        )
        outs.append(z_next)
    return torch.stack(outs, dim=1)


def slice_decoder_output(pred: Any, start: int, end: int) -> Any:
    if isinstance(pred, FocusTokenizerOutput):
        return replace(
            pred,
            representation=pred.representation[:, start:end],
            agent_continuous=pred.agent_continuous[:, start:end],
            agent_valid_logits=pred.agent_valid_logits[:, start:end],
        )
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


def tokenizer_reconstruction_loss(
    tokenizer: torch.nn.Module,
    pred: Any,
    batch: Dict[str, Any],
    args: argparse.Namespace,
    *,
    agent_loss_weight_multiplier: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(tokenizer, FrozenWaymoFocusTokenizer):
        loss, focus_metrics = focus_tokenizer_loss(
            pred,
            agents=batch["agents"],
            agent_mask=batch["agent_mask"],
            xy_weight=args.agent_xy_weight,
            velocity_weight=args.agent_vel_weight,
            yaw_weight=args.agent_yaw_weight,
            valid_weight=args.agent_valid_weight,
            delta_xy_weight=args.agent_delta_xy_weight,
            kinematic_xy_weight=args.agent_kinematic_xy_weight,
            speed_yaw_kinematic_weight=args.agent_speed_yaw_kinematic_weight,
            kinematic_dt=args.kinematic_dt,
        )
        zero = loss.detach() * 0.0
        metrics = {
            "loss_total": focus_metrics["loss_total"],
            "loss_agent_xy": focus_metrics["loss_xy"],
            "loss_agent_vel": focus_metrics["loss_velocity"],
            "loss_agent_yaw": focus_metrics["loss_yaw"],
            "loss_agent_valid": focus_metrics["loss_valid"],
            "loss_agent_delta_xy": focus_metrics["loss_delta_xy"],
            "loss_agent_fde_xy": zero,
            "loss_agent_kinematic_xy": focus_metrics["loss_kinematic_xy"],
            "loss_agent_speed_yaw_kinematic": focus_metrics["loss_speed_yaw_kinematic"],
            "loss_light_state": zero,
            "loss_light_valid": zero,
            "agent_xy_mae_m": focus_metrics["focus_xy_mae_m"],
            "agent_delta_xy_mae_m": focus_metrics["loss_delta_xy"],
            "agent_kinematic_xy_mae_m": focus_metrics["loss_kinematic_xy"],
            "agent_speed_yaw_kinematic_mae_m": focus_metrics["loss_speed_yaw_kinematic"],
            "agent_fde_mae_m": focus_metrics["focus_fde_m"],
            "focus_agent_xy_mae_m": focus_metrics["focus_xy_mae_m"],
            "focus_agent_fde_m": focus_metrics["focus_fde_m"],
            "agent_speed_mae_mps": focus_metrics["focus_speed_mae_mps"],
            "agent_vxvy_mae_mps": focus_metrics["focus_vxvy_mae_mps"],
            "agent_yaw_mae_deg": focus_metrics["focus_yaw_mae_deg"],
            "agent_valid_acc": focus_metrics["focus_valid_acc"],
            "light_state_acc": zero,
            "light_valid_acc": zero,
            "representation_rms": focus_metrics["representation_rms"],
        }
        return loss, metrics

    return vector_tokenizer_reconstruction_loss(
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
        agent_loss_weight_multiplier=agent_loss_weight_multiplier,
    )


def reconstruction_metrics(
    tokenizer: torch.nn.Module,
    pred: Any,
    batch: Dict[str, Any],
    args: argparse.Namespace,
    *,
    agent_loss_weight_multiplier: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    _, metrics = tokenizer_reconstruction_loss(
        tokenizer,
        pred,
        batch,
        args,
        agent_loss_weight_multiplier=agent_loss_weight_multiplier,
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
        actions, act_mask, action_slots = build_ego_action_features(batch, args)
        z_gt, map_tokens, map_mask = encode_batch_dynamics_inputs(
            tokenizer,
            batch,
            return_map=args.dynamics_attend_map,
        )
        z_gt_packed = pack_bottleneck_to_spatial(z_gt, n_spatial=args.n_spatial, k=args.packing_factor)
        z_pred_packed = sample_autoregressive_packed_sequence(
            unwrap_model(dyn),
            z_gt_packed=z_gt_packed,
            actions=actions,
            act_mask=act_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            ctx_length=args.eval_ctx,
            horizon=args.eval_horizon,
            k_max=args.k_max,
            sched=sched,
            max_rollout_window=args.max_rollout_window,
        )
        z_pred = unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
        # Decode at the original sequence length when evaluating very short horizons.
        # The decoder is causal in time, so scored rollout frames cannot attend to
        # appended future GT latents, but this avoids short-sequence SDPA edge cases.
        z_decode = z_pred
        if z_pred.shape[1] < z_gt.shape[1]:
            z_decode = torch.cat([z_pred, z_gt[:, z_pred.shape[1] :]], dim=1)
        decoded = decode_batch_z(tokenizer, z_decode, batch)

        score_start = min(int(args.eval_ctx), int(z_pred.shape[1]) - 1)
        score_end = int(z_pred.shape[1])
        decoded_future = slice_decoder_output(decoded, score_start, score_end)
        batch_future = slice_future_batch(batch, score_start, score_end)
        future_weight = build_agent_loss_weight_multiplier(batch_future, args, action_slots=action_slots)
        metrics = reconstruction_metrics(
            tokenizer,
            decoded_future,
            batch_future,
            args,
            agent_loss_weight_multiplier=future_weight,
        )
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
            "args": vars(args),
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
    if isinstance(tokenizer, FrozenWaymoFocusTokenizer):
        n_latents = tokenizer.n_latents
        d_bottleneck = tokenizer.d_bottleneck
    else:
        n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
        d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor != 0:
        raise ValueError(f"n_latents={n_latents} must be divisible by packing_factor={args.packing_factor}")
    args.n_spatial = n_latents // args.packing_factor
    args.d_spatial = d_bottleneck * args.packing_factor

    if args.dynamics_attend_map and args.dynamics_variant != "standard":
        raise ValueError("Map conditioning is currently supported only for dynamics_variant=standard")
    if args.dynamics_attend_map and args.map_cross_every < 1:
        raise ValueError(f"map_cross_every must be >= 1 when map conditioning is enabled; got {args.map_cross_every}")

    if args.dynamics_variant == "focus_film":
        dyn = FocusFiLMDynamics(
            d_model=args.d_model_dyn,
            d_bottleneck=d_bottleneck,
            d_spatial=args.d_spatial,
            n_spatial=args.n_spatial,
            n_register=args.n_register,
            n_heads=args.n_heads,
            depth=args.dyn_depth,
            k_max=args.k_max,
            dropout=args.dropout,
            mlp_ratio=args.mlp_ratio,
            scale_pos_embeds=args.scale_pos_embeds,
        ).to(device)
    else:
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
            action_clamp_inputs=args.ego_action_clamp,
            map_memory_dim=tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None,
            map_cross_every=args.map_cross_every if args.dynamics_attend_map else 0,
        ).to(device)
    frozen_action_mlp_params = 0 if args.use_ego_actions else freeze_unused_action_mlp(dyn)

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
            f"tokenizer={args.tokenizer_ckpt} kind={tok_args.get('tokenizer_kind', 'vector')} "
            f"n_latents={n_latents} d_bottleneck={d_bottleneck} "
            f"packing={args.packing_factor} n_spatial={args.n_spatial} d_spatial={args.d_spatial}"
        )
        print(
            f"dynamics variant={args.dynamics_variant} d_model={args.d_model_dyn} "
            f"depth={args.dyn_depth} heads={args.n_heads} registers={args.n_register} "
            f"attend_map={args.dynamics_attend_map} map_cross_every={args.map_cross_every} "
            f"seq_len={args.seq_len} max_rollout_window={args.max_rollout_window} "
            f"eval_ctx={args.eval_ctx} eval_horizon={args.eval_horizon}"
        )
        print(f"train_objective={args.train_objective} tf_context={args.tf_context}")
        print(
            f"ego_actions={args.use_ego_actions} source={args.ego_action_source} "
            f"normalization={args.ego_action_normalization} clamp={args.ego_action_clamp} "
            f"agent_far_weight={args.agent_far_weight} near_radius_m={args.agent_near_radius_m} "
            f"train_decoded_loss_weight={args.train_decoded_loss_weight}"
        )
        print(
            f"parameters dynamics={dyn_params:,} frozen_action_mlp={frozen_action_mlp_params:,} "
            f"frozen_tokenizer={tok_params:,}"
        )

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
                actions, act_mask, action_slots = build_ego_action_features(batch, args)
                agent_weight = build_agent_loss_weight_multiplier(batch, args, action_slots=action_slots)

                with torch.no_grad():
                    z, map_tokens, map_mask = encode_batch_dynamics_inputs(
                        tokenizer,
                        batch,
                        return_map=args.dynamics_attend_map,
                    )
                    z_packed = pack_bottleneck_to_spatial(z, n_spatial=args.n_spatial, k=args.packing_factor)

                b_self = int(round(z_packed.shape[0] * args.self_fraction))
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    if args.train_objective == "shortcut":
                        loss, metrics, z_hat_packed = dynamics_pretrain_loss(
                            dyn,
                            z1=z_packed,
                            actions=actions,
                            act_mask=act_mask,
                            k_max=args.k_max,
                            b_self=b_self,
                            step=step,
                            bootstrap_start=args.bootstrap_start,
                            map_tokens=map_tokens,
                            map_mask=map_mask,
                            return_pred=args.train_decoded_loss_weight > 0.0,
                        )
                        decoded_batch = batch
                        decoded_agent_weight = agent_weight
                    elif args.train_objective == "tf_onestep":
                        loss, metrics, z_hat_packed = tf_onestep_loss(
                            dyn,
                            z1=z_packed,
                            actions=actions,
                            act_mask=act_mask,
                            k_max=args.k_max,
                            context=args.tf_context,
                            map_tokens=map_tokens,
                            map_mask=map_mask,
                            return_pred=args.train_decoded_loss_weight > 0.0,
                        )
                        decoded_batch = slice_future_batch(batch, int(args.tf_context), int(z_packed.shape[1]))
                        decoded_agent_weight = build_agent_loss_weight_multiplier(decoded_batch, args, action_slots=action_slots)
                    else:
                        raise ValueError(f"Unknown train_objective={args.train_objective!r}")
                    if args.train_decoded_loss_weight > 0.0:
                        z_hat = unpack_spatial_to_bottleneck(z_hat_packed, k=args.packing_factor)
                        decoded_train = decode_batch_z(tokenizer, z_hat, decoded_batch)
                        decoded_loss, decoded_metrics = tokenizer_reconstruction_loss(
                            tokenizer,
                            decoded_train,
                            decoded_batch,
                            args,
                            agent_loss_weight_multiplier=decoded_agent_weight,
                        )
                        loss = loss + float(args.train_decoded_loss_weight) * decoded_loss
                        metrics["loss_decoded_total"] = decoded_loss.detach()
                        metrics["decoded_agent_xy_mae_m"] = decoded_metrics["agent_xy_mae_m"].detach()
                        metrics["decoded_focus_agent_fde_m"] = decoded_metrics["focus_agent_fde_m"].detach()
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
            if is_rank0() and args.save_latest_each_epoch:
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
    p.add_argument("--dynamics_variant", choices=["standard", "focus_film"], default="standard")
    p.add_argument("--dyn_depth", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=4)
    p.add_argument("--dynamics_attend_map", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--map_cross_every", type=int, default=1)
    p.add_argument("--packing_factor", type=int, default=2)
    p.add_argument("--n_register", type=int, default=8)
    p.add_argument("--scale_pos_embeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action="store_true")

    p.add_argument("--k_max", type=int, default=64)
    p.add_argument("--bootstrap_start", type=int, default=0)
    p.add_argument("--self_fraction", type=float, default=0.857142857)
    p.add_argument("--train_objective", choices=["shortcut", "tf_onestep"], default="shortcut")
    p.add_argument("--tf_context", type=int, default=10)

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
    p.add_argument("--use_ego_actions", action="store_true")
    p.add_argument("--ego_action_source", choices=["sdc", "focus"], default="focus")
    p.add_argument("--ego_action_normalization", choices=["scaled", "raw"], default="scaled")
    p.add_argument("--ego_action_clamp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ego_action_xy_scale", type=float, default=5.0)
    p.add_argument("--ego_action_yaw_scale", type=float, default=math.pi)
    p.add_argument("--ego_action_speed_scale", type=float, default=30.0)
    p.add_argument("--agent_far_weight", type=float, default=1.0)
    p.add_argument("--agent_near_radius_m", type=float, default=50.0)
    p.add_argument("--agent_distance_source", choices=["sdc", "focus"], default="focus")
    p.add_argument("--train_decoded_loss_weight", type=float, default=0.0)

    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument(
        "--save_latest_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save latest.pt after every epoch. Disable for tiny datasets where epochs contain very few steps.",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="waymo-world-model")
    p.add_argument("--wandb_run_name", type=str, default="waymo_latent_dynamics_v0")
    p.add_argument("--wandb_entity", type=str, default=None)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
