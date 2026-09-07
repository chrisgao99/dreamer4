#!/usr/bin/env python3
"""Evaluate a DirectActionFlow checkpoint with a full receding-horizon rollout."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT / "core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.core.waymo_vector_dataset import WaymoVectorDataset
from waymo.training.world_model.direct_action_flow import (
    ActionNormalizer,
    DirectActionFlowModel,
    agents_to_bntf,
    gather_agent_window,
    inverse_holonomic_actions,
    rollout_receding_horizon,
)
from waymo.training.world_model.train_waymo_direct_action_flow import (
    build_normalizer,
    create_model,
    make_loader,
    move_batch,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_data_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_batches", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rollout_steps", type=int, default=80)
    parser.add_argument("--solver_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12346)
    parser.add_argument("--log_every", type=int, default=8)
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def load_model_and_normalizer(
    checkpoint_path: Path,
    device: torch.device,
    weights: str,
) -> tuple[DirectActionFlowModel, ActionNormalizer, argparse.Namespace, int]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if "args" not in checkpoint or "action_stats" not in checkpoint:
        raise KeyError("Checkpoint must contain args and action_stats")

    train_args = argparse.Namespace(**checkpoint["args"])
    model = create_model(train_args).to(device)
    state_key = "ema_model" if weights == "ema" else "model"
    if state_key not in checkpoint:
        if weights == "ema" and "model" in checkpoint:
            state_key = "model"
            print("Checkpoint has no EMA weights; falling back to model weights", flush=True)
        else:
            raise KeyError(f"Checkpoint has no {state_key!r} state dict")
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.eval()
    normalizer = build_normalizer(checkpoint["action_stats"], device)
    checkpoint_step = int(checkpoint.get("step", -1))
    del checkpoint
    gc.collect()
    return model, normalizer, train_args, checkpoint_step


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    require_positive("--eval_batch_size", args.eval_batch_size)
    require_positive("--eval_max_batches", args.eval_max_batches)
    require_positive("--rollout_steps", args.rollout_steps)
    require_positive("--solver_steps", args.solver_steps)
    require_positive("--log_every", args.log_every)

    checkpoint_path = Path(args.checkpoint).resolve()
    val_data_dir = Path(args.val_data_dir).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not val_data_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_data_dir}")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested, but CUDA is unavailable")
        if device.index is not None:
            torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)

    model, normalizer, train_args, checkpoint_step = load_model_and_normalizer(
        checkpoint_path,
        device,
        args.weights,
    )
    if int(train_args.history_length) != 11:
        raise ValueError(
            "This evaluation requires an 11-frame context, but the checkpoint "
            f"was trained with history_length={train_args.history_length}"
        )
    if int(args.rollout_steps) % int(train_args.commitment):
        raise ValueError(
            f"rollout_steps={args.rollout_steps} must be divisible by checkpoint "
            f"commitment={train_args.commitment}"
        )

    dataset = WaymoVectorDataset(str(val_data_dir))
    loader = make_loader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(int(args.seed))

    scene_ade_sum = 0.0
    nonempty_scene_ade_sum = 0.0
    point_distance_sum = 0.0
    valid_point_count = 0
    scene_count = 0
    nonempty_scene_count = 0
    empty_scene_count = 0
    batches = 0
    started = time.monotonic()

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= int(args.eval_max_batches):
            break
        batch = move_batch(raw_batch, device)
        agents = agents_to_bntf(batch["agents"], batch["agent_mask"])
        anchor = int(train_args.history_length) - 1
        available = int(agents.shape[2]) - anchor - 1
        if available < int(args.rollout_steps):
            raise ValueError(
                f"Batch {batch_index} has only {available} future steps after an "
                f"11-frame context; {args.rollout_steps} are required"
            )

        history, future = gather_agent_window(
            agents,
            torch.full(
                (int(agents.shape[0]),),
                anchor,
                dtype=torch.long,
                device=device,
            ),
            history_length=int(train_args.history_length),
            horizon=int(args.rollout_steps),
        )
        targets = inverse_holonomic_actions(
            history,
            future,
            batch["agent_mask"],
            max_displacement_m=float(train_args.physical_max_displacement_m),
            max_yaw_delta_rad=float(train_args.physical_max_yaw_delta_rad),
        )
        poses = rollout_receding_horizon(
            model,
            normalizer,
            initial_history=history,
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_mask=batch["map_mask"],
            current_light_sequence=batch["lights"][
                :, anchor : anchor + int(args.rollout_steps)
            ],
            current_light_mask_sequence=batch["light_mask"][
                :, anchor : anchor + int(args.rollout_steps)
            ],
            focus_action_sequence=targets.actions[:, 0],
            focus_action_valid=targets.valid[:, 0],
            rollout_steps=int(args.rollout_steps),
            commitment=int(train_args.commitment),
            solver_steps=int(args.solver_steps),
            generator=generator,
        )

        valid = targets.valid.clone()
        valid[:, 0] = False
        distance = torch.linalg.vector_norm(
            poses[..., 0:2] - targets.future_pose[..., 0:2],
            dim=-1,
        )
        per_scene_valid = valid.sum(dim=(1, 2))
        per_scene_distance = (distance * valid).sum(dim=(1, 2))
        per_scene_ade = per_scene_distance / per_scene_valid.clamp_min(1)
        nonempty = per_scene_valid > 0

        # scene_mean_ade_m matches the training-time receding_nonfocus_ade_m:
        # compute ADE within each scene, then average scenes equally.
        scene_ade_sum += float(per_scene_ade.sum())
        nonempty_scene_ade_sum += float(per_scene_ade[nonempty].sum())
        point_distance_sum += float(per_scene_distance.sum())
        valid_point_count += int(per_scene_valid.sum())
        batch_scenes = int(agents.shape[0])
        batch_nonempty = int(nonempty.sum())
        scene_count += batch_scenes
        nonempty_scene_count += batch_nonempty
        empty_scene_count += batch_scenes - batch_nonempty
        batches += 1

        if batches % int(args.log_every) == 0 or batches == int(args.eval_max_batches):
            elapsed = time.monotonic() - started
            print(
                f"progress batches={batches}/{args.eval_max_batches} "
                f"scenes={scene_count} "
                f"scene_mean_ade_m={scene_ade_sum / max(1, scene_count):.6f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    if batches == 0 or scene_count == 0:
        raise RuntimeError("Validation loader produced no evaluation batches")
    if batches != int(args.eval_max_batches):
        raise RuntimeError(
            f"Requested {args.eval_max_batches} batches, but only {batches} were available"
        )
    if valid_point_count == 0:
        raise RuntimeError("No valid non-focus trajectory points were evaluated")

    elapsed = time.monotonic() - started
    return {
        "metric": "nonfocus_xy_ade_over_full_receding_rollout",
        "scene_mean_ade_m": scene_ade_sum / scene_count,
        "nonempty_scene_mean_ade_m": nonempty_scene_ade_sum
        / max(1, nonempty_scene_count),
        "point_weighted_ade_m": point_distance_sum / valid_point_count,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "weights": args.weights,
        "val_data_dir": str(val_data_dir),
        "history_context_steps": int(train_args.history_length),
        "model_plan_horizon": int(train_args.horizon),
        "commitment_steps": int(train_args.commitment),
        "rollout_steps": int(args.rollout_steps),
        "solver_steps": int(args.solver_steps),
        "seed": int(args.seed),
        "eval_batches": batches,
        "eval_batch_size": int(args.eval_batch_size),
        "scene_count": scene_count,
        "nonempty_scene_count": nonempty_scene_count,
        "empty_scene_count": empty_scene_count,
        "valid_nonfocus_agent_time_points": valid_point_count,
        "physical_max_displacement_m": float(
            train_args.physical_max_displacement_m
        ),
        "physical_max_yaw_delta_rad": float(
            train_args.physical_max_yaw_delta_rad
        ),
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, output_path)
    print("result " + " ".join(f"{key}={value}" for key, value in result.items()), flush=True)
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
