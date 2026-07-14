"""Train the Waymo vector tokenizer on filtered NPZ scenes.

This trains Decoder 1 with either Encoder A or Encoder B:

- Encoder A input: selected agents, repeated static map tokens, traffic lights
- Encoder B input: selected agents/lights as dynamic tokens, static map memory queried by agents/latents
- bottleneck: z with shape (B,T,n_latents,d_bottleneck)
- decoder input: z plus learned agent/light query tokens
- targets: selected-agent state reconstruction and traffic-light state/validity
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler, random_split

WAYMO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WAYMO_ROOT.parent
CORE_ROOT = WAYMO_ROOT / "core"
INTERACTIVE_ROOT = WAYMO_ROOT / "interactive_probe"
for path in (REPO_ROOT, CORE_ROOT, INTERACTIVE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from labels import InteractiveLabelConfig, build_scene_interactive_labels
    from vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from vector_tokenizer_decoder import (
        TokenizerInteractionAuxHead,
        VectorBlockCausalTokenizerDecoder,
        VectorTokenizer,
        _slice_time_window,
        vector_tokenizer_reconstruction_loss,
    )
    from waymo_vector_dataset import WaymoVectorDataset
except ModuleNotFoundError:
    from waymo.interactive_probe.labels import InteractiveLabelConfig, build_scene_interactive_labels
    from waymo.core.vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from waymo.core.vector_tokenizer_decoder import (
        TokenizerInteractionAuxHead,
        VectorBlockCausalTokenizerDecoder,
        VectorTokenizer,
        _slice_time_window,
        vector_tokenizer_reconstruction_loss,
    )
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset


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
    if ddp:
        if device.type == "cuda":
            dist.barrier(device_ids=[int(device.index if device.index is not None else torch.cuda.current_device())])
        else:
            dist.barrier()
        dist.destroy_process_group()


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed_everything(info.seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_splits(dataset: WaymoVectorDataset, val_fraction: float, seed: int) -> tuple[Iterable, Iterable | None]:
    if val_fraction <= 0.0 or len(dataset) < 2:
        return dataset, None
    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_val = min(n_val, len(dataset) - 1)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds


def slice_time_window(
    batch: Dict[str, torch.Tensor],
    time_window: int,
    *,
    random_start: bool = False,
    required_future_steps: int = 0,
) -> Dict[str, torch.Tensor]:
    out = dict(batch)
    if time_window <= 0:
        out["_time_start"] = torch.zeros((), device=batch["lights"].device, dtype=torch.long)
        return out
    total_steps = int(batch["lights"].shape[1])
    if total_steps <= time_window:
        out = _slice_time_window(batch, time_window)
        out["_time_start"] = torch.zeros((), device=batch["lights"].device, dtype=torch.long)
        return out
    if random_start:
        max_start = max(0, total_steps - int(time_window) - int(required_future_steps))
        start = int(torch.randint(0, max_start + 1, (1,)).item())
    else:
        start = 0
    end = start + int(time_window)

    k = batch["agent_mask"].shape[-1]
    if batch["agents"].shape[1] == k:
        out["agents"] = batch["agents"][:, :, start:end]
    else:
        out["agents"] = batch["agents"][:, start:end]
    out["lights"] = batch["lights"][:, start:end]
    out["light_mask"] = batch["light_mask"][:, start:end]
    out["_time_start"] = torch.tensor(start, device=batch["lights"].device, dtype=torch.long)
    return out


def build_model(args: argparse.Namespace, n_agents: int, n_lights: int) -> VectorTokenizer:
    encoder_kwargs = dict(
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        n_latents=args.n_latents,
        d_bottleneck=args.d_bottleneck,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        scale_pos_embeds=args.scale_pos_embeds,
        bottleneck_output=args.bottleneck_output,
    )
    if args.encoder_variant == "repeat_map":
        encoder = VectorBlockCausalEncoder(**encoder_kwargs)
    elif args.encoder_variant == "static_map_query":
        encoder = VectorStaticMapQueryEncoder(
            **encoder_kwargs,
            map_depth=args.map_depth,
            map_cross_every=args.map_cross_every,
            map_query_tokens=args.map_query_tokens,
        )
    else:
        raise ValueError(f"Unknown encoder_variant: {args.encoder_variant}")
    decoder = VectorBlockCausalTokenizerDecoder(
        d_bottleneck=args.d_bottleneck,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.decoder_depth,
        n_latents=args.n_latents,
        n_agents=n_agents,
        n_lights=n_lights,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        scale_pos_embeds=args.scale_pos_embeds,
        use_agent_tokens=args.decoder_use_agent_tokens,
        agent_token_mode=args.decoder_agent_token_mode,
        attend_map=args.decoder_attend_map,
        map_cross_every=args.decoder_map_cross_every,
        map_query_tokens=args.decoder_map_query_tokens,
        predict_agent_xy_gmm=args.agent_xy_loss == "gmm",
    )
    interaction_aux = None
    if args.interaction_aux_weight > 0.0:
        interaction_aux = TokenizerInteractionAuxHead(
            d_bottleneck=args.d_bottleneck,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_latents=args.n_latents,
            n_agents=n_agents,
            dropout=args.dropout,
            scale_pos_embeds=args.scale_pos_embeds,
        )
    model = VectorTokenizer(encoder=encoder, decoder=decoder, interaction_aux=interaction_aux)
    if interaction_aux is not None and args.interaction_aux_init_decoder_queries:
        model.init_interaction_aux_from_decoder_queries()
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    model = model.module if hasattr(model, "module") else model
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def compute_loss(out, batch: Dict[str, torch.Tensor], args: argparse.Namespace):
    return vector_tokenizer_reconstruction_loss(
        out.decoder,
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


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def compute_interaction_aux_loss(
    out,
    full_batch: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    inter = getattr(out, "interaction", None)
    if inter is None:
        zero = out.encoder.z.sum() * 0.0
        return zero, {"loss_interaction_aux": zero.detach()}

    rel_logits_all = inter.relevance_logits
    bsz, t_window, k = rel_logits_all.shape
    local_step = int(args.interaction_query_step)
    if local_step < 0:
        local_step = t_window + local_step
    if local_step < 0 or local_step >= t_window:
        raise ValueError(f"interaction_query_step={args.interaction_query_step} outside T={t_window}")
    start = int(batch.get("_time_start", torch.zeros((), device=rel_logits_all.device)).detach().cpu().item())
    query_step = start + local_step

    agents_np = full_batch["agents"].detach().cpu().numpy()
    mask_np = full_batch["agent_mask"].detach().cpu().numpy()
    device = rel_logits_all.device
    rel_target = torch.zeros((bsz, k), device=device, dtype=rel_logits_all.dtype)
    rel_mask = torch.zeros((bsz, k), device=device, dtype=rel_logits_all.dtype)
    type_target = torch.zeros((bsz, k), device=device, dtype=torch.long)
    type_mask = torch.zeros((bsz, k), device=device, dtype=rel_logits_all.dtype)
    resp_bin_target = torch.zeros((bsz, k, 3), device=device, dtype=rel_logits_all.dtype)
    resp_bin_mask = torch.zeros((bsz, k, 3), device=device, dtype=rel_logits_all.dtype)
    resp_reg_target = torch.zeros((bsz, k, 1), device=device, dtype=rel_logits_all.dtype)
    resp_reg_mask = torch.zeros((bsz, k, 1), device=device, dtype=rel_logits_all.dtype)

    cfg = InteractiveLabelConfig(future_steps=int(args.interaction_future_steps))
    for b in range(bsz):
        labels = build_scene_interactive_labels(
            agents_np[b],
            mask_np[b],
            query_step=query_step,
            focus_index=int(args.interaction_focus_index),
            cfg=cfg,
        )
        if labels.candidate_index.size == 0:
            continue
        keep = labels.candidate_index < k
        idx_np = labels.candidate_index[keep]
        if idx_np.size == 0:
            continue
        rows = np.flatnonzero(keep)
        idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
        rel_target[b, idx] = torch.as_tensor(labels.relevance_targets[rows, 0], device=device, dtype=rel_logits_all.dtype)
        rel_mask[b, idx] = torch.as_tensor(labels.relevance_masks[rows, 0], device=device, dtype=rel_logits_all.dtype)
        type_target[b, idx] = torch.as_tensor(labels.type_targets[rows], device=device, dtype=torch.long)
        type_mask[b, idx] = torch.as_tensor(labels.type_masks[rows], device=device, dtype=rel_logits_all.dtype)
        resp_bin_target[b, idx] = torch.as_tensor(labels.response_bin_targets[rows], device=device, dtype=rel_logits_all.dtype)
        resp_bin_mask[b, idx] = torch.as_tensor(labels.response_bin_masks[rows], device=device, dtype=rel_logits_all.dtype)
        resp_reg_target[b, idx] = torch.as_tensor(labels.response_reg_targets[rows], device=device, dtype=rel_logits_all.dtype)
        resp_reg_mask[b, idx] = torch.as_tensor(labels.response_reg_masks[rows], device=device, dtype=rel_logits_all.dtype)

    if k > int(args.interaction_focus_index):
        rel_mask[:, int(args.interaction_focus_index)] = 0.0
        type_mask[:, int(args.interaction_focus_index)] = 0.0
        resp_bin_mask[:, int(args.interaction_focus_index)] = 0.0
        resp_reg_mask[:, int(args.interaction_focus_index)] = 0.0

    rel_logits = inter.relevance_logits[:, local_step]
    type_logits = inter.type_logits[:, local_step]
    resp_bin_logits = inter.response_bin_logits[:, local_step]
    resp_reg = inter.response_regression[:, local_step]

    rel_loss = _masked_average(
        torch.nn.functional.binary_cross_entropy_with_logits(rel_logits, rel_target, reduction="none"),
        rel_mask,
    )
    type_loss_raw = torch.nn.functional.cross_entropy(
        type_logits.reshape(-1, type_logits.shape[-1]),
        type_target.reshape(-1),
        reduction="none",
    ).view(bsz, k)
    type_loss = _masked_average(type_loss_raw, type_mask)
    resp_bin_loss = _masked_average(
        torch.nn.functional.binary_cross_entropy_with_logits(resp_bin_logits, resp_bin_target, reduction="none"),
        resp_bin_mask,
    )
    resp_reg_loss = _masked_average(
        torch.nn.functional.smooth_l1_loss(resp_reg, resp_reg_target, reduction="none"),
        resp_reg_mask,
    )

    aux_loss = (
        float(args.interaction_relevance_weight) * rel_loss
        + float(args.interaction_type_weight) * type_loss
        + float(args.interaction_response_bin_weight) * resp_bin_loss
        + float(args.interaction_response_reg_weight) * resp_reg_loss
    )

    rel_pred = (torch.sigmoid(rel_logits) >= 0.5).to(rel_target.dtype)
    rel_acc = _masked_average((rel_pred == rel_target).to(rel_target.dtype), rel_mask)
    type_pred = type_logits.argmax(dim=-1)
    type_acc = _masked_average((type_pred == type_target).to(type_mask.dtype), type_mask)
    resp_bin_pred = (torch.sigmoid(resp_bin_logits) >= 0.5).to(resp_bin_target.dtype)
    resp_bin_acc = _masked_average((resp_bin_pred == resp_bin_target).to(resp_bin_target.dtype), resp_bin_mask)

    metrics = {
        "loss_interaction_aux": aux_loss.detach(),
        "loss_interaction_relevance": rel_loss.detach(),
        "loss_interaction_type": type_loss.detach(),
        "loss_interaction_response_bin": resp_bin_loss.detach(),
        "loss_interaction_response_reg": resp_reg_loss.detach(),
        "interaction_relevance_acc": rel_acc.detach(),
        "interaction_type_acc": type_acc.detach(),
        "interaction_response_bin_acc": resp_bin_acc.detach(),
        "interaction_relevance_count": rel_mask.sum().detach(),
        "interaction_type_count": type_mask.sum().detach(),
        "interaction_response_bin_count": resp_bin_mask.sum().detach(),
        "interaction_response_reg_count": resp_reg_mask.sum().detach(),
    }
    return aux_loss, metrics


def compute_total_loss(
    out,
    batch: Dict[str, torch.Tensor],
    full_batch: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    recon_loss, metrics = compute_loss(out, batch, args)
    metrics = dict(metrics)
    metrics["loss_reconstruction"] = recon_loss.detach()
    total = recon_loss
    if args.interaction_aux_weight > 0.0:
        aux_loss, aux_metrics = compute_interaction_aux_loss(out, full_batch, batch, args)
        total = total + float(args.interaction_aux_weight) * aux_loss
        metrics.update(aux_metrics)
    metrics["loss_total"] = total.detach()
    return total, metrics


def metric_values(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().float().item()) for k, v in metrics.items()}


def format_metrics(metrics: Dict[str, float]) -> str:
    names = [
        "loss_total",
        "loss_agent_xy",
        "loss_agent_vel",
        "loss_agent_yaw",
        "loss_agent_valid",
        "loss_agent_delta_xy",
        "loss_agent_fde_xy",
        "loss_agent_kinematic_xy",
        "loss_agent_speed_yaw_kinematic",
        "loss_light_state",
        "loss_light_valid",
        "loss_reconstruction",
        "loss_interaction_aux",
        "loss_interaction_relevance",
        "loss_interaction_type",
        "loss_interaction_response_bin",
        "loss_interaction_response_reg",
        "agent_xy_mae_m",
        "agent_delta_xy_mae_m",
        "agent_fde_mae_m",
        "agent_kinematic_xy_mae_m",
        "agent_speed_yaw_kinematic_mae_m",
        "focus_agent_xy_mae_m",
        "focus_agent_fde_m",
        "agent_speed_mae_mps",
        "agent_vxvy_mae_mps",
        "agent_yaw_mae_deg",
        "agent_valid_acc",
        "light_state_acc",
        "light_valid_acc",
        "interaction_relevance_acc",
        "interaction_type_acc",
        "interaction_response_bin_acc",
        "interaction_relevance_count",
        "interaction_type_count",
        "interaction_response_bin_count",
        "interaction_response_reg_count",
    ]
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in names if name in metrics)


def ordered_metric_names(metrics: Dict[str, torch.Tensor] | Dict[str, float]) -> list[str]:
    preferred = [
        "loss_agent_valid",
        "loss_agent_vel",
        "loss_agent_xy",
        "loss_agent_yaw",
        "loss_agent_delta_xy",
        "loss_agent_fde_xy",
        "loss_agent_kinematic_xy",
        "loss_agent_speed_yaw_kinematic",
        "loss_light_state",
        "loss_light_valid",
        "loss_reconstruction",
        "loss_interaction_aux",
        "loss_interaction_relevance",
        "loss_interaction_type",
        "loss_interaction_response_bin",
        "loss_interaction_response_reg",
        "loss_total",
        "agent_xy_mae_m",
        "agent_delta_xy_mae_m",
        "agent_fde_mae_m",
        "agent_kinematic_xy_mae_m",
        "agent_speed_yaw_kinematic_mae_m",
        "focus_agent_xy_mae_m",
        "focus_agent_fde_m",
        "agent_speed_mae_mps",
        "agent_vxvy_mae_mps",
        "agent_yaw_mae_deg",
        "agent_valid_acc",
        "light_state_acc",
        "light_valid_acc",
        "interaction_relevance_acc",
        "interaction_type_acc",
        "interaction_response_bin_acc",
        "interaction_relevance_count",
        "interaction_type_count",
        "interaction_response_bin_count",
        "interaction_response_reg_count",
    ]
    return [name for name in preferred if name in metrics] + sorted(set(metrics) - set(preferred))


def average_metrics(metrics: Dict[str, torch.Tensor], device: torch.device, ddp: bool) -> Dict[str, float]:
    names = ordered_metric_names(metrics)
    values = torch.tensor([float(metrics[name].detach().float().item()) for name in names], device=device)
    if ddp:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {name: float(value.item()) for name, value in zip(names, values)}


@torch.no_grad()
def evaluate(
    model: VectorTokenizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    ddp: bool = False,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        full_batch = move_batch(batch, device)
        batch = slice_time_window(
            full_batch,
            args.time_window,
            random_start=args.eval_random_time_window_start,
            required_future_steps=args.interaction_future_steps if args.interaction_aux_weight > 0.0 else 0,
        )
        out = model(
            agents=batch["agents"],
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_mask=batch["map_mask"],
            lights=batch["lights"],
            light_mask=batch["light_mask"],
        )
        _, metrics = compute_total_loss(out, batch, full_batch, args)
        values = metric_values(metrics)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    names = ordered_metric_names(totals)
    packed = torch.tensor([float(count)] + [totals.get(name, 0.0) for name in names], device=device, dtype=torch.float64)
    if ddp:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(packed[0].item()))
    if was_training:
        model.train()
    return {name: float(packed[i + 1].item() / total_count) for i, name in enumerate(names)}


def save_ckpt(
    path: Path,
    *,
    model: VectorTokenizer,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "step": int(step),
            "epoch": int(epoch),
        },
        tmp,
    )
    tmp.replace(path)


def load_ckpt(path: Path, model: VectorTokenizer, opt: torch.optim.Optimizer, scaler: GradScaler) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_agent_xy_loss = getattr(ckpt_args, "agent_xy_loss", "smooth_l1")
    elif isinstance(ckpt_args, dict):
        ckpt_agent_xy_loss = ckpt_args.get("agent_xy_loss", "smooth_l1")
    else:
        ckpt_agent_xy_loss = "smooth_l1"
    model_to_load = unwrap_model(model)
    model_changed = False
    loaded_optional_interaction_aux = False
    try:
        model_to_load.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError:
        incompatible = model_to_load.load_state_dict(ckpt["model"], strict=False)
        allowed_gmm_keys = {
            "decoder.agent_xy_gmm_head.weight",
            "decoder.agent_xy_gmm_head.bias",
        }
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        missing_allowed = missing.issubset(allowed_gmm_keys) or all(
            key in allowed_gmm_keys or key.startswith("interaction_aux.") for key in missing
        )
        unexpected_allowed = unexpected.issubset(allowed_gmm_keys)
        if (
            not missing_allowed
            or not unexpected_allowed
            or ckpt_agent_xy_loss == "gmm"
        ):
            raise
        loaded_optional_interaction_aux = bool(any(key.startswith("interaction_aux.") for key in missing))
        model_changed = True
        reason = "optional interaction aux head differs" if loaded_optional_interaction_aux else "optional xy GMM head differs"
        print(f"Loaded {path} without optimizer state because {reason}.")
    if loaded_optional_interaction_aux and hasattr(model_to_load, "init_interaction_aux_from_decoder_queries"):
        model_to_load.init_interaction_aux_from_decoder_queries()
        print("Initialized interaction aux slot queries from loaded decoder.agent_queries.")
    if not model_changed:
        opt.load_state_dict(ckpt["opt"])
    if not model_changed and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def train(args: argparse.Namespace) -> None:
    ddp, rank, world_size, local_rank = init_distributed()
    if ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed + rank)
    use_amp = (device.type == "cuda") and (not args.no_amp)

    dataset = WaymoVectorDataset(args.data_dir)
    sample = dataset[0]
    n_agents = int(sample["agent_mask"].shape[-1])
    n_lights = int(sample["lights"].shape[1])
    if args.val_data_dir is not None:
        train_ds = dataset
        val_ds = WaymoVectorDataset(args.val_data_dir)
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
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
        collate_fn=_collate,
    )
    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=worker_init_fn,
            collate_fn=_collate,
        )

    model = build_model(args, n_agents=n_agents, n_lights=n_lights).to(device)
    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device="cuda", enabled=use_amp)

    step = 0
    start_epoch = 0
    if args.resume:
        step, start_epoch = load_ckpt(Path(args.resume), model, opt, scaler)

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
        param_count = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        print(
            f"device={device} amp={use_amp} ddp={ddp} world_size={world_size} "
            f"train={len(train_ds)} val={0 if val_ds is None else len(val_ds)}"
        )
        print(
            f"n_agents={n_agents} n_lights={n_lights} n_latents={args.n_latents} "
            f"d_bottleneck={args.d_bottleneck} decoder_agent_token_mode={args.decoder_agent_token_mode} "
            f"decoder_attend_map={args.decoder_attend_map} decoder_map_query_tokens={args.decoder_map_query_tokens}"
        )
        print(
            f"interaction_aux_weight={args.interaction_aux_weight} "
            f"interaction_query_step={args.interaction_query_step} "
            f"interaction_future_steps={args.interaction_future_steps}"
        )
        print(
            f"encoder_variant={args.encoder_variant} depth={args.depth} decoder_depth={args.decoder_depth} "
            f"map_depth={args.map_depth} map_cross_every={args.map_cross_every} map_query_tokens={args.map_query_tokens}"
        )
        print(f"learnable parameters: {param_count:,}")

    t0 = time.time()
    latest = Path(args.ckpt_dir) / "latest.pt"
    stop = False
    last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            full_batch = move_batch(batch, device)
            batch = slice_time_window(
                full_batch,
                args.time_window,
                random_start=args.random_time_window_start,
                required_future_steps=args.interaction_future_steps if args.interaction_aux_weight > 0.0 else 0,
            )
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                out = model(
                    agents=batch["agents"],
                    agent_mask=batch["agent_mask"],
                    map_polylines=batch["map_polylines"],
                    map_mask=batch["map_mask"],
                    lights=batch["lights"],
                    light_mask=batch["light_mask"],
                )
                loss, metrics = compute_total_loss(out, batch, full_batch, args)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            step += 1

            should_log = step % args.log_every == 0 or step == 1
            if ddp and (should_log or args.wandb):
                values = average_metrics(metrics, device, ddp=True)
            else:
                values = metric_values(metrics)
            if is_rank0() and should_log:
                elapsed = max(1e-6, time.time() - t0)
                print(f"step={step} epoch={epoch + 1} {format_metrics(values)} steps_per_sec={step / elapsed:.3f}")
            if is_rank0() and wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in values.items()}, step=step)

            if val_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, device, args, ddp=ddp)
                if is_rank0():
                    print(f"eval step={step} {format_metrics(val_metrics)}")
                if is_rank0() and wandb_run is not None:
                    wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)

            if is_rank0() and args.save_every > 0 and step % args.save_every == 0:
                save_ckpt(Path(args.ckpt_dir) / f"step_{step:08d}.pt", model=model, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch)
                save_ckpt(latest, model=model, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch)

            if args.max_steps > 0 and step >= args.max_steps:
                stop = True
                break

        if is_rank0():
            save_ckpt(latest, model=model, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch + 1)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args, ddp=ddp)
            if is_rank0():
                print(f"epoch={epoch + 1} val {format_metrics(val_metrics)}")
            if is_rank0() and wandb_run is not None:
                wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
        if stop:
            break

    if is_rank0():
        save_ckpt(Path(args.ckpt_dir) / f"final_step_{step:08d}.pt", model=model, opt=opt, scaler=scaler, args=args, step=step, epoch=last_epoch)
    if is_rank0() and wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(ddp, device)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Waymo vector tokenizer.")
    p.add_argument("--data_dir", type=str, nargs="+", default=["/p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset"])
    p.add_argument("--val_data_dir", type=str, nargs="+", default=None)
    p.add_argument("--ckpt_dir", type=str, default="/p/yufeng/tri30/dreamer4/waymo/checkpoints/vector_tokenizer")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--time_window", type=int, default=32)
    p.add_argument("--random_time_window_start", action="store_true", help="Sample a random contiguous time window during training.")
    p.add_argument("--eval_random_time_window_start", action="store_true", help="Also sample random windows during validation/eval logging.")
    p.add_argument("--val_fraction", type=float, default=0.1)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--decoder_depth", type=int, default=3)
    p.add_argument("--n_latents", type=int, default=8)
    p.add_argument("--d_bottleneck", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)
    p.add_argument("--scale_pos_embeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--encoder_variant", choices=["repeat_map", "static_map_query"], default="repeat_map")
    p.add_argument("--map_depth", type=int, default=2)
    p.add_argument("--map_cross_every", type=int, default=1)
    p.add_argument("--map_query_tokens", choices=["latent", "agent", "latent_agent", "all"], default="latent_agent")
    p.add_argument("--bottleneck_output", choices=["tanh", "layernorm"], default="tanh")
    p.add_argument("--decoder_use_agent_tokens", action="store_true")
    p.add_argument("--decoder_agent_token_mode", choices=["none", "all", "focus"], default="none")
    p.add_argument("--decoder_attend_map", action="store_true")
    p.add_argument("--decoder_map_cross_every", type=int, default=1)
    p.add_argument(
        "--decoder_map_query_tokens",
        choices=["latent", "agent", "light", "latent_agent", "agent_light", "all"],
        default="all",
    )

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

    p.add_argument("--interaction_aux_weight", type=float, default=0.0)
    p.add_argument("--interaction_relevance_weight", type=float, default=1.0)
    p.add_argument("--interaction_type_weight", type=float, default=1.0)
    p.add_argument("--interaction_response_bin_weight", type=float, default=1.0)
    p.add_argument("--interaction_response_reg_weight", type=float, default=0.2)
    p.add_argument("--interaction_query_step", type=int, default=-1)
    p.add_argument("--interaction_future_steps", type=int, default=50)
    p.add_argument("--interaction_focus_index", type=int, default=0)
    p.add_argument("--interaction_aux_init_decoder_queries", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=500)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="waymo-vector-tokenizer")
    p.add_argument("--wandb_run_name", type=str, default="baseline")
    p.add_argument("--wandb_entity", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.decoder_use_agent_tokens and args.decoder_agent_token_mode == "none":
        args.decoder_agent_token_mode = "all"
    if args.decoder_agent_token_mode != "none":
        args.decoder_use_agent_tokens = True
    train(args)


if __name__ == "__main__":
    main()
