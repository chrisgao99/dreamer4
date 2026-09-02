#!/usr/bin/env python3
"""Train the tokenizer-free explicit-agent Waymo action-flow world model."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

WAYMO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT / "core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.core.waymo_vector_dataset import WaymoVectorDataset
from waymo.training.world_model.direct_action_flow import (
    ActionNormalizer,
    ActionTargets,
    DirectActionFlowModel,
    agents_to_bntf,
    execute_holonomic_actions,
    flow_matching_loss,
    gather_agent_window,
    inverse_holonomic_actions,
    rollout_receding_horizon,
    select_window_anchors,
    wrap_angle_rad,
)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed_everything(info.seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def collate_vector_batch(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack tensor fields; paths and scenario strings are not training inputs."""
    return {
        key: torch.stack([item[key] for item in items], dim=0)
        for key, value in items[0].items()
        if torch.is_tensor(value)
    }


class AgentStatsDataset(Dataset):
    """Load only agent arrays while computing one-time action statistics."""

    def __init__(self, data_dir: str, max_files: int = 0) -> None:
        paths = sorted(glob.glob(str(Path(data_dir) / "*.npz")))
        if int(max_files) > 0 and len(paths) > int(max_files):
            # Even coverage is deterministic and avoids biasing the statistics
            # toward one lexicographic part of the dataset.
            indices = np.linspace(0, len(paths) - 1, int(max_files), dtype=np.int64)
            paths = [paths[int(index)] for index in indices]
        self.paths = paths
        if not self.paths:
            raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as data:
            return (
                torch.from_numpy(data["agents"]).float(),
                torch.from_numpy(data["agent_mask"]).bool(),
            )


def _stats_collate(
    items: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.stack([item[0] for item in items]), torch.stack([item[1] for item in items])


def compute_action_statistics(
    data_dir: str,
    *,
    history_length: int,
    num_types: int,
    batch_size: int,
    num_workers: int,
    max_files: int,
) -> dict[str, Any]:
    """Compute local-action moments over all valid future adjacent pairs."""
    dataset = AgentStatsDataset(data_dir, max_files=max_files)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=False,
        persistent_workers=(int(num_workers) > 0),
        collate_fn=_stats_collate,
        worker_init_fn=worker_init_fn,
    )
    count = torch.zeros(num_types, dtype=torch.float64)
    total = torch.zeros(num_types, 3, dtype=torch.float64)
    total_sq = torch.zeros(num_types, 3, dtype=torch.float64)
    first_action_index = int(history_length) - 1
    started = time.monotonic()
    for batch_index, (agents, agent_mask) in enumerate(loader):
        agents = agents_to_bntf(agents, agent_mask)
        current = agents[:, :, first_action_index:-1]
        nxt = agents[:, :, first_action_index + 1 :]
        valid = (
            agent_mask[:, :, None]
            & (current[..., 5] > 0.5)
            & (nxt[..., 5] > 0.5)
        )
        delta_x = nxt[..., 0] - current[..., 0]
        delta_y = nxt[..., 1] - current[..., 1]
        yaw = current[..., 6]
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        action = torch.stack(
            (
                c * delta_x + s * delta_y,
                -s * delta_x + c * delta_y,
                wrap_angle_rad(nxt[..., 6] - yaw),
            ),
            dim=-1,
        ).double()
        agent_type = current[..., 7].round().long().clamp(0, num_types - 1)
        for type_index in range(num_types):
            selected = valid & (agent_type == type_index)
            values = action[selected]
            if values.numel() == 0:
                continue
            count[type_index] += values.shape[0]
            total[type_index] += values.sum(dim=0)
            total_sq[type_index] += values.square().sum(dim=0)
        if (batch_index + 1) % 100 == 0:
            print(
                f"action-stats batches={batch_index + 1}/{len(loader)} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    global_count = count.sum().clamp_min(1.0)
    global_total = total.sum(dim=0)
    global_total_sq = total_sq.sum(dim=0)
    global_mean = global_total / global_count
    global_var = (global_total_sq / global_count - global_mean.square()).clamp_min(0.0)
    global_std = global_var.sqrt()
    mean = torch.empty_like(total)
    std = torch.empty_like(total)
    # Prevent rare/near-static channels from producing extreme normalized values.
    minimum_std = torch.tensor((0.02, 0.02, 0.002), dtype=torch.float64)
    for type_index in range(num_types):
        if count[type_index] > 0:
            mean[type_index] = total[type_index] / count[type_index]
            variance = (
                total_sq[type_index] / count[type_index] - mean[type_index].square()
            ).clamp_min(0.0)
            std[type_index] = variance.sqrt()
        else:
            mean[type_index] = global_mean
            std[type_index] = global_std
    std = torch.maximum(std, minimum_std)
    return {
        "version": 1,
        "source_data_dir": str(Path(data_dir).resolve()),
        "first_action_index": first_action_index,
        "num_files": len(dataset),
        "num_types": int(num_types),
        "action_order": ["a_longitudinal_m", "a_lateral_m", "delta_yaw_rad"],
        "count": [int(value) for value in count.tolist()],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "global_mean": global_mean.tolist(),
        "global_std": global_std.tolist(),
    }


def load_or_compute_action_statistics(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.action_stats_path)
    if path.is_file():
        payload = json.loads(path.read_text())
        print(f"Loaded action statistics: {path}", flush=True)
        return payload
    print(f"Computing action statistics from {args.data_dir}", flush=True)
    payload = compute_action_statistics(
        args.data_dir,
        history_length=args.history_length,
        num_types=args.num_agent_types,
        batch_size=args.stats_batch_size,
        num_workers=args.num_workers,
        max_files=args.stats_max_files,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(f"Saved action statistics: {path}", flush=True)
    return payload


def build_normalizer(payload: dict[str, Any], device: torch.device) -> ActionNormalizer:
    return ActionNormalizer(
        torch.tensor(payload["mean"], dtype=torch.float32),
        torch.tensor(payload["std"], dtype=torch.float32),
    ).to(device)


@dataclass(frozen=True)
class PreparedBatch:
    history: torch.Tensor
    targets: ActionTargets
    normalized_actions: torch.Tensor
    current_lights: torch.Tensor
    current_light_mask: torch.Tensor
    anchors: torch.Tensor


def gather_current_lights(
    lights: torch.Tensor, light_mask: torch.Tensor, anchors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = torch.arange(int(lights.shape[0]), device=lights.device)
    return lights[batch, anchors], light_mask[batch, anchors]


def prepare_batch(
    batch: dict[str, Any],
    normalizer: ActionNormalizer,
    args: argparse.Namespace,
    *,
    random_start: bool,
) -> PreparedBatch:
    agents = agents_to_bntf(batch["agents"], batch["agent_mask"])
    anchors = select_window_anchors(
        agents,
        batch["agent_mask"],
        history_length=args.history_length,
        horizon=args.horizon,
        random_start=random_start,
    )
    history, future = gather_agent_window(
        agents,
        anchors,
        history_length=args.history_length,
        horizon=args.horizon,
    )
    targets = inverse_holonomic_actions(history, future, batch["agent_mask"])
    normalized = normalizer.normalize(targets.actions, targets.agent_type)
    normalized = normalized * targets.valid[..., None].to(normalized.dtype)
    current_lights, current_light_mask = gather_current_lights(
        batch["lights"], batch["light_mask"], anchors
    )
    return PreparedBatch(
        history=history,
        targets=targets,
        normalized_actions=normalized,
        current_lights=current_lights,
        current_light_mask=current_light_mask,
        anchors=anchors,
    )


def scene_kwargs(batch: dict[str, Any], prepared: PreparedBatch) -> dict[str, torch.Tensor]:
    return {
        "history": prepared.history,
        "agent_mask": batch["agent_mask"],
        "map_polylines": batch["map_polylines"],
        "map_mask": batch["map_mask"],
        "current_lights": prepared.current_lights,
        "current_light_mask": prepared.current_light_mask,
    }


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source = model.state_dict()
        target = self.model.state_dict()
        for name, value in target.items():
            incoming = source[name].detach()
            if value.is_floating_point():
                value.lerp_(incoming.to(value.dtype), 1.0 - self.decay)
            else:
                value.copy_(incoming)


def create_model(args: argparse.Namespace) -> DirectActionFlowModel:
    return DirectActionFlowModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        history_length=args.history_length,
        horizon=args.horizon,
        chunk_size=args.commitment,
        history_depth=args.history_depth,
        map_depth=args.map_depth,
        scene_depth=args.scene_depth,
        action_depth=args.action_depth,
        step_refiner_depth=args.step_refiner_depth,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        position_scale_m=args.position_scale_m,
    )


def lr_multiplier(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return max(1e-8, float(step + 1) / float(max(1, args.warmup_steps)))
    progress = min(
        1.0,
        float(step - args.warmup_steps)
        / float(max(1, args.max_steps - args.warmup_steps)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def save_checkpoint(
    path: Path,
    *,
    model: DirectActionFlowModel,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    stats: dict[str, Any],
    step: int,
    epoch: int,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "ema_model": ema.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
        "action_stats": stats,
        "step": int(step),
        "epoch": int(epoch),
        "best_val": float(best_val),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    model: DirectActionFlowModel,
    ema: ModelEMA,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema.model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]), strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_val", math.inf)),
    )


@torch.no_grad()
def evaluate_flow(
    model: DirectActionFlowModel,
    loader: DataLoader,
    normalizer: ActionNormalizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch_index, raw_batch in enumerate(loader):
        if args.eval_batches > 0 and batch_index >= args.eval_batches:
            break
        batch = move_batch(raw_batch, device)
        prepared = prepare_batch(batch, normalizer, args, random_start=False)
        scene = model.encode_scene(**scene_kwargs(batch, prepared))
        loss, metrics = flow_matching_loss(
            model,
            scene,
            prepared.normalized_actions,
            prepared.targets.valid,
        )
        predicted_pose = execute_holonomic_actions(
            prepared.targets.current_pose,
            prepared.targets.actions,
            prepared.targets.valid,
        )
        pose_xy_error = torch.linalg.vector_norm(
            predicted_pose[..., 0:2] - prepared.targets.future_pose[..., 0:2], dim=-1
        )
        valid = prepared.targets.valid
        roundtrip = (pose_xy_error * valid).sum() / valid.sum().clamp_min(1)
        values = {
            "val_flow_loss": loss,
            "val_normalized_endpoint_mae": metrics["normalized_endpoint_mae"],
            "val_inverse_execute_xy_error_m": roundtrip,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        batches += 1
    if batches == 0:
        raise RuntimeError("Validation loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def evaluate_samples(
    model: DirectActionFlowModel,
    loader: DataLoader,
    normalizer: ActionNormalizer,
    device: torch.device,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, float]:
    model.eval()
    ade_sum = 0.0
    minade_sum = 0.0
    diversity_sum = 0.0
    scenes = 0
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for batch_index, raw_batch in enumerate(loader):
        if args.sample_eval_batches > 0 and batch_index >= args.sample_eval_batches:
            break
        batch = move_batch(raw_batch, device)
        prepared = prepare_batch(batch, normalizer, args, random_start=False)
        scene = model.encode_scene(**scene_kwargs(batch, prepared))
        model_mask = scene.agent_mask[:, :, None].expand(
            -1, -1, args.horizon
        ).clone()
        focus = prepared.normalized_actions[:, 0]
        candidate_poses = []
        for _ in range(args.eval_num_rollouts):
            normalized = model.sample_normalized_actions(
                scene,
                model_mask,
                focus,
                solver_steps=args.eval_solver_steps,
                generator=generator,
            )
            metric = normalizer.denormalize(normalized, prepared.targets.agent_type)
            candidate_poses.append(
                execute_holonomic_actions(
                    prepared.targets.current_pose, metric, model_mask
                )
            )
        poses = torch.stack(candidate_poses, dim=1)  # (B,R,N,H,3)
        target_xy = prepared.targets.future_pose[..., 0:2]
        valid = prepared.targets.valid.clone()
        valid[:, 0] = False
        distance = torch.linalg.vector_norm(
            poses[..., 0:2] - target_xy[:, None], dim=-1
        )
        denom = valid.sum(dim=(1, 2)).clamp_min(1)
        candidate_ade = (distance * valid[:, None]).sum(dim=(2, 3)) / denom[:, None]

        if args.eval_num_rollouts > 1:
            pair = torch.linalg.vector_norm(
                poses[:, :, None, ..., 0:2] - poses[:, None, :, ..., 0:2], dim=-1
            )
            pair_mean = (pair * valid[:, None, None]).sum(dim=(3, 4)) / denom[
                :, None, None
            ]
            upper = torch.triu(
                torch.ones(
                    args.eval_num_rollouts,
                    args.eval_num_rollouts,
                    dtype=torch.bool,
                    device=device,
                ),
                diagonal=1,
            )
            diversity = pair_mean[:, upper].mean(dim=1)
        else:
            diversity = torch.zeros_like(candidate_ade[:, 0])
        batch_size = int(poses.shape[0])
        ade_sum += float(candidate_ade.mean(dim=1).sum())
        minade_sum += float(candidate_ade.min(dim=1).values.sum())
        diversity_sum += float(diversity.sum())
        scenes += batch_size
    if scenes == 0:
        return {}
    return {
        "sample_mean_ade_m": ade_sum / scenes,
        "sample_minade_m": minade_sum / scenes,
        "sample_pairwise_trajectory_distance_m": diversity_sum / scenes,
        "sample_num_rollouts": float(args.eval_num_rollouts),
    }


@torch.no_grad()
def evaluate_receding_rollout(
    model: DirectActionFlowModel,
    loader: DataLoader,
    normalizer: ActionNormalizer,
    device: torch.device,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict[str, float]:
    """Small diagnostic of the actual generate-H/execute-B closed loop."""
    model.eval()
    error_sum = 0.0
    scenes = 0
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for batch_index, raw_batch in enumerate(loader):
        if args.receding_eval_batches > 0 and batch_index >= args.receding_eval_batches:
            break
        batch = move_batch(raw_batch, device)
        agents = agents_to_bntf(batch["agents"], batch["agent_mask"])
        anchor = args.history_length - 1
        available = min(args.receding_eval_horizon, int(agents.shape[2]) - anchor - 1)
        if available <= 0:
            continue
        anchors = torch.full(
            (int(agents.shape[0]),), anchor, device=device, dtype=torch.long
        )
        history, future = gather_agent_window(
            agents,
            anchors,
            history_length=args.history_length,
            horizon=available,
        )
        targets = inverse_holonomic_actions(history, future, batch["agent_mask"])
        light_sequence = batch["lights"][:, anchor : anchor + available]
        light_mask_sequence = batch["light_mask"][:, anchor : anchor + available]
        poses = rollout_receding_horizon(
            model,
            normalizer,
            initial_history=history,
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_mask=batch["map_mask"],
            current_light_sequence=light_sequence,
            current_light_mask_sequence=light_mask_sequence,
            focus_action_sequence=targets.actions[:, 0],
            focus_action_valid=targets.valid[:, 0],
            rollout_steps=available,
            commitment=args.commitment,
            solver_steps=args.eval_solver_steps,
            generator=generator,
        )
        valid = targets.valid.clone()
        valid[:, 0] = False
        distance = torch.linalg.vector_norm(
            poses[..., 0:2] - targets.future_pose[..., 0:2], dim=-1
        )
        denom = valid.sum(dim=(1, 2)).clamp_min(1)
        scene_ade = (distance * valid).sum(dim=(1, 2)) / denom
        error_sum += float(scene_ade.sum())
        scenes += int(scene_ade.shape[0])
    if scenes == 0:
        return {}
    return {"receding_nonfocus_ade_m": error_sum / scenes}


def make_loader(
    dataset: WaymoVectorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=bool(shuffle),
        persistent_workers=(int(num_workers) > 0),
        worker_init_fn=worker_init_fn,
        collate_fn=collate_vector_batch,
    )


def train(args: argparse.Namespace) -> None:
    if args.horizon % args.commitment:
        raise ValueError("--horizon must be divisible by --commitment")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)

    stats = load_or_compute_action_statistics(args)
    normalizer = build_normalizer(stats, device)
    train_dataset = WaymoVectorDataset(args.data_dir)
    val_dataset = WaymoVectorDataset(args.val_data_dir)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    model = create_model(args).to(device)
    ema = ModelEMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "none": torch.float32,
    }[args.amp_dtype]
    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    step = 0
    epoch = 0
    best_val = math.inf
    if args.resume:
        step, epoch, best_val = load_checkpoint(
            Path(args.resume),
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
        )
        print(f"Resumed {args.resume}: step={step} epoch={epoch} best_val={best_val:.6f}")
    # Set the learning rate used by the first optimizer update (and restore the
    # analytically defined schedule after resume) before entering the loop.
    initial_lr_multiplier = lr_multiplier(step, args)
    for group in optimizer.param_groups:
        group["lr"] = args.lr * initial_lr_multiplier

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ckpt_dir / "metrics.jsonl"
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"device={device} amp={args.amp_dtype} train={len(train_dataset)} val={len(val_dataset)} "
        f"parameters={trainable:,}",
        flush=True,
    )
    print(
        f"explicit_agent_action_flow L={args.history_length} H={args.horizon} "
        f"B={args.commitment} d={args.d_model} heads={args.n_heads} "
        f"history/map/scene/action/refiner_depth="
        f"{args.history_depth}/{args.map_depth}/{args.scene_depth}/"
        f"{args.action_depth}/{args.step_refiner_depth}",
        flush=True,
    )
    print(
        "agent_state=x,y,yaw; valid=mask; type=static_condition; "
        "light=current_only; focus=known_H_step_action; generated=nonfocus",
        flush=True,
    )

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config=vars(args),
            )
        except Exception as exc:
            print(f"wandb disabled after initialization error: {exc}", flush=True)

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    rolling: dict[str, float] = {}
    rolling_count = 0
    last_log_time = time.monotonic()
    stop = False
    while step < args.max_steps and not stop:
        epoch += 1
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            prepared = prepare_batch(batch, normalizer, args, random_start=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                scene = model.encode_scene(**scene_kwargs(batch, prepared))
                loss, metrics = flow_matching_loss(
                    model,
                    scene,
                    prepared.normalized_actions,
                    prepared.targets.valid,
                )
                scaled_loss = loss / float(args.grad_accum_steps)
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            rolling_count += 1
            for name, value in metrics.items():
                rolling[name] = rolling.get(name, 0.0) + float(value)
            if micro_step % args.grad_accum_steps:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            multiplier = lr_multiplier(step, args)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            ema.update(model)

            if step % args.log_every == 0:
                elapsed = max(1e-6, time.monotonic() - last_log_time)
                record = {
                    "step": step,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm),
                    "steps_per_second": args.log_every / elapsed,
                    **{
                        f"train_{name}": value / max(1, rolling_count)
                        for name, value in rolling.items()
                    },
                }
                print(" ".join(f"{k}={v:.6g}" for k, v in record.items()), flush=True)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
                rolling.clear()
                rolling_count = 0
                last_log_time = time.monotonic()

            should_eval = step % args.eval_every == 0 or step == args.max_steps
            if should_eval:
                validation = evaluate_flow(ema.model, val_loader, normalizer, device, args)
                if step % args.sample_eval_every == 0 or step == args.max_steps:
                    validation.update(
                        evaluate_samples(
                            ema.model,
                            val_loader,
                            normalizer,
                            device,
                            args,
                            seed=args.eval_seed,
                        )
                    )
                if (
                    args.receding_eval_every > 0
                    and (step % args.receding_eval_every == 0 or step == args.max_steps)
                ):
                    validation.update(
                        evaluate_receding_rollout(
                            ema.model,
                            val_loader,
                            normalizer,
                            device,
                            args,
                            seed=args.eval_seed + 1,
                        )
                    )
                record = {"step": step, "epoch": epoch, **validation}
                print("validation " + " ".join(f"{k}={v:.6g}" for k, v in record.items()), flush=True)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if wandb_run is not None:
                    wandb_run.log(validation, step=step)
                model.train()

                if validation["val_flow_loss"] < best_val:
                    best_val = validation["val_flow_loss"]
                    save_checkpoint(
                        ckpt_dir / "best.pt",
                        model=model,
                        ema=ema,
                        optimizer=optimizer,
                        scaler=scaler,
                        args=args,
                        stats=stats,
                        step=step,
                        epoch=epoch,
                        best_val=best_val,
                    )
                    print(f"new best val_flow_loss={best_val:.6f} at step={step}", flush=True)

            if step % args.save_every == 0 or step == args.max_steps:
                numbered = ckpt_dir / f"step_{step:09d}.pt"
                save_checkpoint(
                    numbered,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    stats=stats,
                    step=step,
                    epoch=epoch,
                    best_val=best_val,
                )
                shutil.copyfile(numbered, ckpt_dir / "latest.pt")
                print(f"saved {numbered}", flush=True)
            if step >= args.max_steps:
                stop = True
                break

    final_path = ckpt_dir / f"final_step_{step:09d}.pt"
    save_checkpoint(
        final_path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        stats=stats,
        step=step,
        epoch=epoch,
        best_val=best_val,
    )
    print(f"training complete: {final_path}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--val_data_dir", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--action_stats_path", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--history_length", type=int, default=11)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--commitment", type=int, default=5)
    parser.add_argument("--position_scale_m", type=float, default=100.0)
    parser.add_argument("--num_agent_types", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--history_depth", type=int, default=2)
    parser.add_argument("--map_depth", type=int, default=2)
    parser.add_argument("--scene_depth", type=int, default=4)
    parser.add_argument("--action_depth", type=int, default=8)
    parser.add_argument("--step_refiner_depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stats_batch_size", type=int, default=64)
    parser.add_argument(
        "--stats_max_files",
        type=int,
        default=8192,
        help="Deterministic training-file subset used for one-time action moments; 0 uses all files.",
    )
    parser.add_argument("--max_steps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=5_000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")

    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--eval_batches", type=int, default=32)
    parser.add_argument("--sample_eval_every", type=int, default=10_000)
    parser.add_argument("--sample_eval_batches", type=int, default=4)
    parser.add_argument("--eval_num_rollouts", type=int, default=4)
    parser.add_argument("--eval_solver_steps", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=12345)
    parser.add_argument("--receding_eval_every", type=int, default=50_000)
    parser.add_argument("--receding_eval_batches", type=int, default=1)
    parser.add_argument("--receding_eval_horizon", type=int, default=80)

    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", default="waymo-world-model")
    parser.add_argument("--wandb_run_name", default="waymo_direct_action_flow_v1")
    parser.add_argument("--wandb_entity", default=None)
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
