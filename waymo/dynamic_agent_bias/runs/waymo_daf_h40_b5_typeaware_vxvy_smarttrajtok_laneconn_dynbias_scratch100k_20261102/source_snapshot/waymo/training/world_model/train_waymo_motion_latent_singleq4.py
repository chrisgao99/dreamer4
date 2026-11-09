#!/usr/bin/env python3
"""Train single-q Motion Latent with a four-step latent shortcut sampler."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from dreamer4.model import pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck
from waymo.core.vector_tokenizer_encoder import _collate
from waymo.core.waymo_vector_dataset import WaymoVectorDataset
from waymo.training.world_model import train_waymo_motion_latent_v1 as v1
from waymo.training.world_model import train_waymo_world_model as wm
from waymo.training.world_model.motion_latent_singleq4 import MotionLatentSingleQ4
from waymo.training.world_model.motion_latent_v1 import (
    LightweightAgentSemanticReader,
    latent_q_consistency_loss,
    motion_residual_loss,
    motion_targets,
)


CHECKPOINT_FORMAT = "waymo_motion_latent_singleq4_v1"


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    best_val_score: float | None = None,
    val_metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "epoch": int(epoch),
        "args": vars(args),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if best_val_score is not None:
        payload["best_val_score"] = float(best_val_score)
    if val_metrics is not None:
        payload["val_metrics"] = {name: float(value) for name, value in val_metrics.items()}
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unexpected checkpoint format in {path}: {checkpoint.get('format')}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_val_score", float("inf"))),
    )


def load_model_weights(path: str, *, model: torch.nn.Module) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unexpected checkpoint format in {path}: {checkpoint.get('format')}")
    model.load_state_dict(checkpoint["model"], strict=True)


def q_ground_truth_loss(
    q_next: torch.Tensor,
    q_gt_next: torch.Tensor,
    agent_mask: torch.Tensor,
    action_slots: torch.Tensor,
    *,
    validity_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct physical-state loss after hard integration.

    The controlled focus agent is excluded because its state is overwritten by
    the supplied action exactly.  Other agents are normalized in physical
    units before applying Smooth L1.
    """
    bsz, n_agents = q_next.shape[:2]
    controlled = torch.zeros((bsz, n_agents), dtype=torch.bool, device=q_next.device)
    controlled[torch.arange(bsz, device=q_next.device), action_slots.long()] = True
    static = agent_mask.to(torch.bool) & ~controlled
    # Continuous state has a label only when GT is valid.  Invalid GT slots
    # still contribute to the validity-classification term below.
    state_mask = static & (q_gt_next[..., 5] > 0.5)

    yaw_pred = q_next[..., 6]
    yaw_gt = q_gt_next[..., 6]
    pred = torch.cat(
        [
            q_next[..., 0:2] / 5.0,
            q_next[..., 2:5] / 5.0,
            torch.sin(yaw_pred)[..., None],
            torch.cos(yaw_pred)[..., None],
        ],
        dim=-1,
    )
    target = torch.cat(
        [
            q_gt_next[..., 0:2] / 5.0,
            q_gt_next[..., 2:5] / 5.0,
            torch.sin(yaw_gt)[..., None],
            torch.cos(yaw_gt)[..., None],
        ],
        dim=-1,
    )
    per = F.smooth_l1_loss(pred.float(), target.float(), beta=0.1, reduction="none")
    denom = state_mask.sum().clamp_min(1).to(per.dtype)
    continuous = (per * state_mask[..., None]).sum() / (denom * per.shape[-1])
    if static.any():
        valid_prob = q_next[..., 5].float()[static].clamp(1e-5, 1.0 - 1e-5)
        valid_target = q_gt_next[..., 5].float()[static]
        valid = -(
            valid_target * valid_prob.log()
            + (1.0 - valid_target) * (1.0 - valid_prob).log()
        ).mean()
    else:
        valid = q_next[..., 5].float().sum() * 0.0
    loss = continuous + float(validity_weight) * valid

    xy_error = torch.linalg.vector_norm(q_next[..., 0:2] - q_gt_next[..., 0:2], dim=-1)
    velocity_error = torch.linalg.vector_norm(q_next[..., 3:5] - q_gt_next[..., 3:5], dim=-1)
    yaw_error = torch.atan2(torch.sin(yaw_pred - yaw_gt), torch.cos(yaw_pred - yaw_gt)).abs()
    return loss, {
        "q_gt_total": loss.detach(),
        "q_gt_continuous": continuous.detach(),
        "q_gt_valid": valid.detach(),
        "q_xy_error_m": (xy_error * state_mask).sum().detach() / denom,
        "q_velocity_error_mps": (velocity_error * state_mask).sum().detach() / denom,
        "q_yaw_error_rad": (yaw_error * state_mask).sum().detach() / denom,
    }


def prediction_context(
    z_history: list[torch.Tensor],
    q_history: list[torch.Tensor],
    time_history: list[int],
    *,
    target_time: int,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    max_context: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = min(int(max_context), len(z_history))
    past_packed = torch.stack(z_history[-keep:], dim=1)
    past_q = torch.stack(q_history[-keep:], dim=1)
    past_times = time_history[-keep:]
    indices = torch.tensor(past_times + [int(target_time)], device=actions.device, dtype=torch.long)
    actions_sequence = actions.index_select(1, indices)
    act_mask_sequence = act_mask.index_select(1, indices)
    return past_packed, past_q, actions_sequence, act_mask_sequence


def single_transition_loss(
    model: MotionLatentSingleQ4,
    reader: Optional[LightweightAgentSemanticReader],
    *,
    z_history: list[torch.Tensor],
    q_history: list[torch.Tensor],
    time_history: list[int],
    target_time: int,
    z_gt_packed: torch.Tensor,
    q_gt: torch.Tensor,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    action_slots: torch.Tensor,
    agent_mask: torch.Tensor,
    map_tokens: Optional[torch.Tensor],
    map_mask: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    past_packed, past_q, actions_sequence, act_mask_sequence = prediction_context(
        z_history,
        q_history,
        time_history,
        target_time=target_time,
        actions=actions,
        act_mask=act_mask,
        max_context=args.max_context,
    )
    q_current = q_history[-1]
    motion_raw, q_next = model.predict_single_q(
        past_packed=past_packed,
        past_q=past_q,
        q_current=q_current,
        actions_sequence=actions_sequence,
        act_mask_sequence=act_mask_sequence,
        action_slots=action_slots,
        agent_mask=agent_mask,
        map_tokens=map_tokens,
        map_mask=map_mask,
        kinematic_dt=args.kinematic_dt,
    )

    zero = motion_raw.float().sum() * 0.0
    motion_loss = zero
    motion_metrics: dict[str, torch.Tensor] = {}
    if float(args.motion_weight) != 0.0:
        target_motion = motion_targets(q_current, q_gt[:, target_time], dt=args.kinematic_dt)
        motion_loss, motion_metrics = motion_residual_loss(
            motion_raw,
            target_motion,
            q_current,
            q_gt[:, target_time],
            agent_mask,
            action_slots,
            validity_weight=args.motion_validity_weight,
        )

    q_gt_loss = zero
    if float(args.q_gt_weight) != 0.0:
        q_gt_loss, q_metrics = q_ground_truth_loss(
            q_next,
            q_gt[:, target_time],
            agent_mask,
            action_slots,
            validity_weight=args.q_gt_validity_weight,
        )
    else:
        # Keep physical diagnostics in no-GT runs without creating a GT-loss
        # gradient path.
        with torch.no_grad():
            _, q_metrics = q_ground_truth_loss(
                q_next.detach(),
                q_gt[:, target_time],
                agent_mask,
                action_slots,
                validity_weight=args.q_gt_validity_weight,
            )

    q_condition = q_next.detach() if args.detach_q_condition else q_next
    z_target = z_gt_packed[:, target_time]
    z_tau = torch.randn_like(z_target)
    endpoint_losses: list[torch.Tensor] = []
    endpoint_metrics: dict[str, torch.Tensor] = {}
    shortcut_steps = int(args.shortcut_steps)
    if shortcut_steps != 4:
        raise ValueError(f"singleq4 requires exactly four shortcut steps, got {shortcut_steps}")
    if args.k_max % shortcut_steps:
        raise ValueError("k_max must be divisible by shortcut_steps")
    solver_dt = 1.0 / shortcut_steps
    signal_scale = args.k_max // shortcut_steps
    for solver_index in range(shortcut_steps):
        tau = solver_index / shortcut_steps
        z_endpoint = model.predict_latent_endpoint(
            past_packed=past_packed,
            past_q=past_q,
            z_tau=z_tau,
            q_condition=q_condition,
            actions_sequence=actions_sequence,
            act_mask_sequence=act_mask_sequence,
            agent_mask=agent_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            tau_index=solver_index * signal_scale,
        )
        endpoint_loss = F.mse_loss(z_endpoint.float(), z_target.float())
        endpoint_losses.append(endpoint_loss)
        endpoint_metrics[f"latent_mse_tau{solver_index}"] = endpoint_loss.detach()
        velocity = (z_endpoint.float() - z_tau.float()) / max(1e-4, 1.0 - tau)
        z_tau = (z_tau.float() + solver_dt * velocity).to(z_endpoint.dtype)

    z_next = z_tau
    latent_loss = torch.stack(endpoint_losses).mean()
    consistency_loss = zero
    consistency_metrics: dict[str, torch.Tensor] = {}
    if float(args.consistency_weight) != 0.0:
        if reader is None:
            raise ValueError("A semantic reader is required when consistency_weight is nonzero")
        z_next_unpacked = unpack_spatial_to_bottleneck(
            z_next[:, None], k=args.packing_factor
        )
        semantic_pred = reader(z_next_unpacked, agent_mask=agent_mask)
        consistency_loss, consistency_metrics = latent_q_consistency_loss(
            semantic_pred, q_next, agent_mask
        )
    loss = (
        latent_loss
        + args.motion_weight * motion_loss
        + args.consistency_weight * consistency_loss
        + args.q_gt_weight * q_gt_loss
    )
    metrics = {
        "loss_total": loss.detach(),
        "latent_mse": latent_loss.detach(),
        "motion_weight": torch.tensor(float(args.motion_weight), device=loss.device),
        "consistency_weight": torch.tensor(float(args.consistency_weight), device=loss.device),
        "q_gt_weight": torch.tensor(args.q_gt_weight, device=loss.device),
        "single_q_integrations": torch.tensor(1.0, device=loss.device),
        "latent_shortcut_steps": torch.tensor(float(shortcut_steps), device=loss.device),
        **endpoint_metrics,
        **motion_metrics,
        **q_metrics,
    }
    if motion_metrics:
        metrics["motion_loss"] = motion_loss.detach()
    if consistency_metrics:
        metrics.update(
            consistency_loss=consistency_loss.detach(),
            consistency_continuous=consistency_metrics["loss_continuous"],
            consistency_valid=consistency_metrics["loss_valid"],
        )
    return loss, metrics, z_next, q_next


def average_metrics(
    sums: dict[str, torch.Tensor], values: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    for name, value in values.items():
        detached = value.detach()
        sums[name] = sums.get(name, torch.zeros_like(detached)) + detached
    return sums


def build_validation_loader(
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader | None:
    if not args.val_data_dir and not args.val_manifest:
        return None
    if not args.val_data_dir or not args.val_manifest:
        raise ValueError("--val_data_dir and --val_manifest must be supplied together")
    val_set = WaymoVectorDataset(args.val_data_dir)
    manifest_path = Path(args.val_manifest)
    payload = json.loads(manifest_path.read_text())
    samples = payload.get("samples", [])
    if len(samples) < int(args.val_subset_size):
        raise ValueError(
            f"Validation manifest has {len(samples)} samples, need {args.val_subset_size}"
        )
    indices = [int(item["dataset_index"]) for item in samples[: int(args.val_subset_size)]]
    if any(index < 0 or index >= len(val_set) for index in indices):
        raise ValueError("Validation manifest contains an out-of-range dataset index")
    print(
        f"validation dataset={len(val_set)} subset={len(indices)} manifest={manifest_path} "
        f"batch={args.val_batch_size} horizon={args.best_eval_horizon or args.rollout_end} "
        f"best_metric={args.best_metric}",
        flush=True,
    )
    return DataLoader(
        Subset(val_set, indices),
        batch_size=int(args.val_batch_size),
        shuffle=False,
        num_workers=int(args.val_num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=int(args.val_num_workers) > 0,
        worker_init_fn=wm.worker_init_fn,
        collate_fn=_collate,
    )


@torch.no_grad()
def rollout_validation_sequence(
    model: MotionLatentSingleQ4,
    *,
    z_gt_packed: torch.Tensor,
    q_gt: torch.Tensor,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    action_slots: torch.Tensor,
    agent_mask: torch.Tensor,
    map_tokens: Optional[torch.Tensor],
    map_mask: Optional[torch.Tensor],
    horizon: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_history = [z_gt_packed[:, 0]]
    q_history = [q_gt[:, 0]]
    time_history = [0]
    z_outputs = list(z_history)
    q_outputs = list(q_history)
    shortcut_steps = int(args.shortcut_steps)
    solver_dt = 1.0 / shortcut_steps
    signal_scale = int(args.k_max) // shortcut_steps
    for target_time in range(1, int(horizon) + 1):
        past_packed, past_q, actions_sequence, act_mask_sequence = prediction_context(
            z_history,
            q_history,
            time_history,
            target_time=target_time,
            actions=actions,
            act_mask=act_mask,
            max_context=args.max_context,
        )
        _, q_next = model.predict_single_q(
            past_packed=past_packed,
            past_q=past_q,
            q_current=q_history[-1],
            actions_sequence=actions_sequence,
            act_mask_sequence=act_mask_sequence,
            action_slots=action_slots,
            agent_mask=agent_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            kinematic_dt=args.kinematic_dt,
        )
        z_tau = torch.randn_like(z_gt_packed[:, target_time])
        for solver_index in range(shortcut_steps):
            tau = solver_index / shortcut_steps
            z_endpoint = model.predict_latent_endpoint(
                past_packed=past_packed,
                past_q=past_q,
                z_tau=z_tau,
                q_condition=q_next,
                actions_sequence=actions_sequence,
                act_mask_sequence=act_mask_sequence,
                agent_mask=agent_mask,
                map_tokens=map_tokens,
                map_mask=map_mask,
                tau_index=solver_index * signal_scale,
            )
            velocity = (z_endpoint.float() - z_tau.float()) / max(1e-4, 1.0 - tau)
            z_tau = (z_tau.float() + solver_dt * velocity).to(z_endpoint.dtype)
        z_next = z_tau
        z_outputs.append(z_next)
        q_outputs.append(q_next)
        z_history.append(z_next)
        q_history.append(q_next)
        time_history.append(target_time)
        if len(z_history) > int(args.max_context):
            z_history = z_history[-int(args.max_context) :]
            q_history = q_history[-int(args.max_context) :]
            time_history = time_history[-int(args.max_context) :]
    return torch.stack(z_outputs, dim=1), torch.stack(q_outputs, dim=1)


@torch.no_grad()
def evaluate_validation_rollout(
    model: MotionLatentSingleQ4,
    tokenizer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    n_spatial: int,
) -> dict[str, float]:
    horizon = int(args.best_eval_horizon or args.rollout_end)
    if horizon < 1 or horizon > 90:
        raise ValueError(f"best_eval_horizon must be in [1, 90], got {horizon}")
    was_training = model.training
    model.eval()
    totals = {
        "decoder_xy": 0.0,
        "decoder_velocity": 0.0,
        "decoder_yaw": 0.0,
        "decoder_valid_correct": 0.0,
        "q_xy": 0.0,
        "q_velocity": 0.0,
        "q_yaw": 0.0,
        "q_valid_correct": 0.0,
        "state_count": 0.0,
        "slot_count": 0.0,
        "decoder_fde": 0.0,
        "q_fde": 0.0,
        "fde_count": 0.0,
    }
    cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(args.val_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.val_seed))
        for batch_index, batch in enumerate(loader, start=1):
            batch = wm.slice_time_window(
                wm.move_batch(batch, device), horizon + 1, random_start=False
            )
            q_gt = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
            actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
            if actions is None or act_mask is None:
                raise ValueError("Validation requires ego actions")
            z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
                tokenizer, batch, args, return_map=True
            )
            z_gt_packed = pack_bottleneck_to_spatial(
                z_gt, n_spatial=n_spatial, k=args.packing_factor
            )
            z_pred_packed, q_pred = rollout_validation_sequence(
                model,
                z_gt_packed=z_gt_packed,
                q_gt=q_gt,
                actions=actions,
                act_mask=act_mask,
                action_slots=action_slots,
                agent_mask=batch["agent_mask"],
                map_tokens=map_tokens,
                map_mask=map_mask,
                horizon=horizon,
                args=args,
            )
            z_pred = unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
            decoded = wm.decode_batch_z_for_world_model(tokenizer, z_pred, batch, args)

            target = q_gt[:, 1 : horizon + 1].float()
            q_future = q_pred[:, 1 : horizon + 1].float()
            decoder_cont = decoded.agent_continuous[:, 1 : horizon + 1].float()
            slot_mask = batch["agent_mask"].to(torch.bool)[:, None].expand_as(target[..., 5])
            state_mask = slot_mask & (target[..., 5] > 0.5)
            state_count = float(state_mask.sum().item())
            slot_count = float(slot_mask.sum().item())

            decoder_xy = torch.linalg.vector_norm(decoder_cont[..., 0:2] - target[..., 0:2], dim=-1)
            decoder_velocity = torch.linalg.vector_norm(
                decoder_cont[..., 3:5] - target[..., 3:5], dim=-1
            )
            decoder_yaw_pred = torch.atan2(decoder_cont[..., 5], decoder_cont[..., 6])
            decoder_yaw_delta = decoder_yaw_pred - target[..., 6]
            decoder_yaw = torch.atan2(
                torch.sin(decoder_yaw_delta), torch.cos(decoder_yaw_delta)
            ).abs()
            q_xy = torch.linalg.vector_norm(q_future[..., 0:2] - target[..., 0:2], dim=-1)
            q_velocity = torch.linalg.vector_norm(q_future[..., 3:5] - target[..., 3:5], dim=-1)
            q_yaw_delta = q_future[..., 6] - target[..., 6]
            q_yaw = torch.atan2(torch.sin(q_yaw_delta), torch.cos(q_yaw_delta)).abs()

            totals["decoder_xy"] += float(decoder_xy[state_mask].sum().item())
            totals["decoder_velocity"] += float(decoder_velocity[state_mask].sum().item())
            totals["decoder_yaw"] += float(decoder_yaw[state_mask].sum().item())
            totals["q_xy"] += float(q_xy[state_mask].sum().item())
            totals["q_velocity"] += float(q_velocity[state_mask].sum().item())
            totals["q_yaw"] += float(q_yaw[state_mask].sum().item())
            decoder_valid = decoded.agent_valid_logits[:, 1 : horizon + 1] > 0.0
            q_valid = q_future[..., 5] > 0.5
            target_valid = target[..., 5] > 0.5
            totals["decoder_valid_correct"] += float(
                (decoder_valid == target_valid)[slot_mask].sum().item()
            )
            totals["q_valid_correct"] += float((q_valid == target_valid)[slot_mask].sum().item())
            totals["state_count"] += state_count
            totals["slot_count"] += slot_count
            final_mask = state_mask[:, -1]
            totals["decoder_fde"] += float(decoder_xy[:, -1][final_mask].sum().item())
            totals["q_fde"] += float(q_xy[:, -1][final_mask].sum().item())
            totals["fde_count"] += float(final_mask.sum().item())
            if batch_index == 1 or batch_index % 4 == 0 or batch_index == len(loader):
                print(f"validation progress {batch_index}/{len(loader)}", flush=True)
    if was_training:
        model.train()
    state_denom = max(1.0, totals["state_count"])
    slot_denom = max(1.0, totals["slot_count"])
    fde_denom = max(1.0, totals["fde_count"])
    return {
        "decoder_xy_mae_m": totals["decoder_xy"] / state_denom,
        "decoder_velocity_mae_mps": totals["decoder_velocity"] / state_denom,
        "decoder_yaw_mae_rad": totals["decoder_yaw"] / state_denom,
        "decoder_valid_acc": totals["decoder_valid_correct"] / slot_denom,
        "decoder_xy_fde_m": totals["decoder_fde"] / fde_denom,
        "q_xy_mae_m": totals["q_xy"] / state_denom,
        "q_velocity_mae_mps": totals["q_velocity"] / state_denom,
        "q_yaw_mae_rad": totals["q_yaw"] / state_denom,
        "q_valid_acc": totals["q_valid_correct"] / slot_denom,
        "q_xy_fde_m": totals["q_fde"] / fde_denom,
        "eval_horizon": float(horizon),
        "eval_samples": float(len(loader.dataset)),
    }


def train(args: argparse.Namespace) -> None:
    if int(args.shortcut_steps) != 4:
        raise ValueError("This trainer is intentionally fixed to four latent shortcut steps")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(args.seed)

    train_set = WaymoVectorDataset(args.data_dir)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=wm.worker_init_fn,
        collate_fn=_collate,
    )
    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    if not hasattr(tokenizer, "decoder"):
        raise ValueError("Single-q Motion Latent requires the vector tokenizer")
    reader: Optional[LightweightAgentSemanticReader] = None
    if float(args.consistency_weight) != 0.0:
        if not args.semantic_reader_ckpt:
            raise ValueError(
                "--semantic_reader_ckpt is required when consistency_weight is nonzero"
            )
        reader = v1.load_semantic_reader(
            args.semantic_reader_ckpt,
            tokenizer=tokenizer,
            tok_args=tok_args,
            device=device,
        )
    n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
    d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor:
        raise ValueError("n_latents must be divisible by packing_factor")
    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    model = MotionLatentSingleQ4(
        d_model=args.d_model,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=args.n_register,
        n_agents=tokenizer.decoder.n_agents,
        n_heads=args.n_heads,
        depth=args.depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        scale_pos_embeds=True,
        action_clamp_inputs=False,
        map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer),
        map_cross_every=args.map_cross_every,
    ).to(device)
    if args.init_ckpt and args.resume:
        raise ValueError("--init_ckpt and --resume are mutually exclusive")
    if args.init_ckpt:
        load_model_weights(args.init_ckpt, model=model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": torch.float32}[args.amp_dtype]
    scaler = GradScaler(device="cuda", enabled=use_amp and amp_dtype == torch.float16)
    step, start_epoch, best_val_score = 0, 0, float("inf")
    if args.resume:
        step, start_epoch, best_val_score = load_checkpoint(
            args.resume, model=model, optimizer=optimizer, scaler=scaler
        )

    val_loader = build_validation_loader(args, device)
    if args.best_only and val_loader is None:
        raise ValueError("--best_only requires --val_data_dir and --val_manifest")

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    reader_params = 0 if reader is None else sum(parameter.numel() for parameter in reader.parameters())
    print(
        f"motion_latent_singleq4 device={device} train={len(train_set)} "
        f"batch={args.batch_size} model_params={trainable:,} "
        f"reader={'disabled' if reader is None else 'enabled'} frozen_reader_params={reader_params:,}"
    )
    print(
        f"max_steps={args.max_steps} train_mode={args.train_mode} rollout_end={args.rollout_end} "
        f"max_context={args.max_context} shortcut_steps=4 physical_integrations_per_frame=1 "
        f"q_gt_weight={args.q_gt_weight} detach_q_condition={args.detach_q_condition} "
        f"tokenizer_chunk_window={args.tokenizer_chunk_window} "
        f"tokenizer_chunk_stride={args.tokenizer_chunk_stride} init_ckpt={args.init_ckpt}"
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    start_time = time.time()
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break
            batch = wm.slice_time_window(
                wm.move_batch(batch, device), args.rollout_end + 1, random_start=False
            )
            q_gt = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
            if q_gt.shape[1] < args.rollout_end + 1:
                raise ValueError(
                    f"Scene has only {q_gt.shape[1]} frames, need {args.rollout_end + 1}"
                )
            actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
            if actions is None or act_mask is None:
                raise ValueError("Single-q Motion Latent requires ego actions")
            with torch.no_grad():
                z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
                    tokenizer, batch, args, return_map=True
                )
                z_gt_packed = pack_bottleneck_to_spatial(
                    z_gt, n_spatial=n_spatial, k=args.packing_factor
                )

            if args.train_mode == "online_step":
                context, ctx1_probability = v1.choose_context(step, device, False)
                z_history = [z_gt_packed[:, index].detach() for index in range(context)]
                q_history = [q_gt[:, index].detach() for index in range(context)]
                time_history = list(range(context))
                for target_time in range(context, args.rollout_end + 1):
                    if step >= args.max_steps:
                        break
                    with autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        loss, metrics, z_next, q_next = single_transition_loss(
                            model,
                            reader,
                            z_history=z_history,
                            q_history=q_history,
                            time_history=time_history,
                            target_time=target_time,
                            z_gt_packed=z_gt_packed,
                            q_gt=q_gt,
                            actions=actions,
                            act_mask=act_mask,
                            action_slots=action_slots,
                            agent_mask=batch["agent_mask"],
                            map_tokens=map_tokens,
                            map_mask=map_mask,
                            args=args,
                        )
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    z_history.append(z_next.detach())
                    q_history.append(q_next.detach())
                    time_history.append(target_time)
                    if len(z_history) > args.max_context:
                        z_history = z_history[-args.max_context :]
                        q_history = q_history[-args.max_context :]
                        time_history = time_history[-args.max_context :]
                    metrics.update(
                        {
                            "ctx": torch.tensor(float(context), device=device),
                            "ctx1_probability": torch.tensor(ctx1_probability, device=device),
                            "rollout_target_time": torch.tensor(float(target_time), device=device),
                        }
                    )
                    log_step(
                        step,
                        epoch,
                        metrics,
                        start_time=start_time,
                        log_every=args.log_every,
                        wandb_run=wandb_run,
                    )
                    best_val_score = maybe_save(
                        step,
                        epoch,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scaler=scaler,
                        args=args,
                        ckpt_dir=ckpt_dir,
                        latest=latest,
                        val_loader=val_loader,
                        device=device,
                        n_spatial=n_spatial,
                        best_val_score=best_val_score,
                        wandb_run=wandb_run,
                    )
                continue

            # Memory-bounded on-policy rollout: predicted state is fed into
            # later physical frames, while backward is completed one frame at
            # a time so H90 never retains 450 backbone graphs concurrently.
            z_history = [z_gt_packed[:, 0].detach()]
            q_history = [q_gt[:, 0].detach()]
            time_history = [0]
            metric_sums: dict[str, torch.Tensor] = {}
            optimizer.zero_grad(set_to_none=True)
            horizon = int(args.rollout_end)
            for target_time in range(1, horizon + 1):
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss, metrics, z_next, q_next = single_transition_loss(
                        model,
                        reader,
                        z_history=z_history,
                        q_history=q_history,
                        time_history=time_history,
                        target_time=target_time,
                        z_gt_packed=z_gt_packed,
                        q_gt=q_gt,
                        actions=actions,
                        act_mask=act_mask,
                        action_slots=action_slots,
                        agent_mask=batch["agent_mask"],
                        map_tokens=map_tokens,
                        map_mask=map_mask,
                        args=args,
                    )
                scaler.scale(loss / horizon).backward()
                average_metrics(metric_sums, metrics)
                z_history.append(z_next.detach())
                q_history.append(q_next.detach())
                time_history.append(target_time)
                if len(z_history) > args.max_context:
                    z_history = z_history[-args.max_context :]
                    q_history = q_history[-args.max_context :]
                    time_history = time_history[-args.max_context :]
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = {name: value / horizon for name, value in metric_sums.items()}
            metrics.update(
                {
                    "ctx": torch.tensor(1.0, device=device),
                    "rollout_target_time": torch.tensor(float(horizon), device=device),
                }
            )
            log_step(
                step,
                epoch,
                metrics,
                start_time=start_time,
                log_every=args.log_every,
                wandb_run=wandb_run,
            )
            best_val_score = maybe_save(
                step,
                epoch,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                ckpt_dir=ckpt_dir,
                latest=latest,
                val_loader=val_loader,
                device=device,
                n_spatial=n_spatial,
                best_val_score=best_val_score,
                wandb_run=wandb_run,
            )
        epoch += 1

    if args.best_only:
        if int(args.save_every) <= 0 or step % int(args.save_every):
            best_val_score = maybe_save(
                step,
                epoch,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                ckpt_dir=ckpt_dir,
                latest=latest,
                val_loader=val_loader,
                device=device,
                n_spatial=n_spatial,
                best_val_score=best_val_score,
                wandb_run=wandb_run,
                force=True,
            )
        if not (ckpt_dir / "best.pt").is_file():
            raise RuntimeError("best-only training completed without producing best.pt")
        print(
            f"training complete step={step}; best {args.best_metric}={best_val_score:.6f} "
            f"checkpoint={ckpt_dir / 'best.pt'}",
            flush=True,
        )
    elif not args.skip_final_save:
        save_checkpoint(
            ckpt_dir / f"final_step_{step:08d}.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            step=step,
            epoch=epoch,
        )
        save_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            step=step,
            epoch=epoch,
        )
    if wandb_run is not None:
        wandb_run.finish()


def log_step(
    step: int,
    epoch: int,
    metrics: dict[str, torch.Tensor],
    *,
    start_time: float,
    log_every: int,
    wandb_run: Any,
) -> None:
    if step != 1 and (log_every <= 0 or step % log_every):
        return
    values = wm.tensor_metrics(metrics)
    elapsed = max(1e-6, time.time() - start_time)
    print(
        f"step={step} epoch={epoch + 1} {wm.format_metrics(values)} "
        f"steps_per_sec={step / elapsed:.3f}",
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.log({f"train/{name}": value for name, value in values.items()}, step=step)


def maybe_save(
    step: int,
    epoch: int,
    *,
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    ckpt_dir: Path,
    latest: Path,
    val_loader: DataLoader | None,
    device: torch.device,
    n_spatial: int,
    best_val_score: float,
    wandb_run: Any,
    force: bool = False,
) -> float:
    if not force and (args.save_every <= 0 or step % args.save_every):
        return best_val_score
    if val_loader is not None:
        val_metrics = evaluate_validation_rollout(
            model,
            tokenizer,
            val_loader,
            device,
            args,
            n_spatial=n_spatial,
        )
        if args.best_metric not in val_metrics:
            raise ValueError(f"Unknown best metric: {args.best_metric}")
        score = float(val_metrics[args.best_metric])
        formatted = " ".join(f"{name}={value:.6f}" for name, value in val_metrics.items())
        print(f"validation step={step} {formatted}", flush=True)
        if wandb_run is not None:
            wandb_run.log({f"val/{name}": value for name, value in val_metrics.items()}, step=step)
        if score < best_val_score:
            previous = best_val_score
            best_val_score = score
            save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                step=step,
                epoch=epoch,
                best_val_score=best_val_score,
                val_metrics=val_metrics,
            )
            previous_text = "inf" if not math.isfinite(previous) else f"{previous:.6f}"
            print(
                f"best improved: {args.best_metric} {previous_text} -> {best_val_score:.6f}; "
                f"saved {ckpt_dir / 'best.pt'}",
                flush=True,
            )
        else:
            print(
                f"best unchanged: {args.best_metric}={score:.6f} >= {best_val_score:.6f}; "
                "checkpoint not saved",
                flush=True,
            )
        if args.best_only:
            return best_val_score
    save_checkpoint(
        ckpt_dir / f"step_{step:08d}.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        step=step,
        epoch=epoch,
    )
    save_checkpoint(
        latest,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        step=step,
        epoch=epoch,
    )
    return best_val_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", nargs="+", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--semantic_reader_ckpt", default=None)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init_ckpt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=600_000)
    parser.add_argument("--train_mode", choices=("online_step", "rollout_stream"), default="online_step")
    parser.add_argument("--rollout_end", type=int, default=90)
    parser.add_argument("--max_context", type=int, default=10)
    parser.add_argument("--tokenizer_chunk_window", type=int, default=32)
    parser.add_argument("--tokenizer_chunk_stride", type=int, default=30)
    parser.add_argument("--shortcut_steps", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--time_every", type=int, default=1)
    parser.add_argument("--map_cross_every", type=int, default=1)
    parser.add_argument("--packing_factor", type=int, default=2)
    parser.add_argument("--n_register", type=int, default=8)
    parser.add_argument("--k_max", type=int, default=64)
    parser.add_argument("--kinematic_dt", type=float, default=0.1)
    parser.add_argument("--motion_weight", type=float, default=1.0)
    parser.add_argument("--motion_validity_weight", type=float, default=0.2)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--q_gt_weight", type=float, default=0.0)
    parser.add_argument("--q_gt_validity_weight", type=float, default=0.2)
    parser.add_argument("--detach_q_condition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--val_data_dir", nargs="+", default=None)
    parser.add_argument("--val_manifest", default=None)
    parser.add_argument("--val_subset_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--val_num_workers", type=int, default=4)
    parser.add_argument("--val_seed", type=int, default=0)
    parser.add_argument("--best_eval_horizon", type=int, default=0)
    parser.add_argument(
        "--best_metric",
        choices=("decoder_xy_mae_m", "q_xy_mae_m"),
        default="decoder_xy_mae_m",
    )
    parser.add_argument("--best_only", action="store_true")
    parser.add_argument("--skip_final_save", action="store_true")
    parser.add_argument("--use_ego_actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ego_action_source", choices=("focus", "sdc"), default="focus")
    parser.add_argument("--ego_action_normalization", choices=("raw", "scaled"), default="raw")
    parser.add_argument("--ego_action_xy_scale", type=float, default=5.0)
    parser.add_argument("--ego_action_yaw_scale", type=float, default=math.pi)
    parser.add_argument("--ego_action_speed_scale", type=float, default=30.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="waymo-world-model")
    parser.add_argument("--wandb_run_name", default="waymo_motion_latent_singleq4")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
