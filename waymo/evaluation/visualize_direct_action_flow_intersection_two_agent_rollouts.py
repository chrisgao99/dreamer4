#!/usr/bin/env python3
"""Reproduce the fixed intersection-two-agent plots with DirectActionFlow."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT / "core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.core.waymo_vector_dataset import WaymoVectorDataset  # noqa: E402
from waymo.evaluation.eval_waymo_direct_action_flow_multisample import (  # noqa: E402
    load_model_and_normalizer,
)
from waymo.evaluation.visualize_h90_intersection_two_agent_rollouts import (  # noqa: E402
    _make_contact_sheet,
    _plot_scene,
    _square_bounds,
)
from waymo.training.world_model.direct_action_flow import (  # noqa: E402
    agents_to_bntf,
    gather_agent_window,
    inverse_holonomic_actions,
    rollout_receding_horizon,
)
from waymo.training.world_model.train_waymo_direct_action_flow import (  # noqa: E402
    make_loader,
    move_batch,
    seed_everything,
)


def _repeat_batch(value: torch.Tensor, repeats: int) -> torch.Tensor:
    return value.repeat_interleave(int(repeats), dim=0)


def _reference_bounds(
    reference_dir: Path,
    record: dict[str, Any],
    *,
    scene_index: int,
    context_frames: int,
    horizon: int,
    margin_m: float,
) -> tuple[float, float, float, float]:
    bundle = reference_dir / (
        f"scene_{scene_index:02d}_{record['scenario_id']}_trajectories.npz"
    )
    if not bundle.is_file():
        raise FileNotFoundError(f"Missing h90 reference trajectories: {bundle}")
    with np.load(bundle, allow_pickle=False) as source:
        ego_xy = source["ego_gt_xy"]
        ego_valid = source["ego_gt_valid"].astype(bool)
        target_gt_xy = source["target_gt_xy"]
        target_valid = source["target_gt_valid"].astype(bool)
        target_rollout_xy = source["target_rollout_xy"]
    start = int(context_frames) - 1
    end = int(context_frames) + int(horizon)
    return _square_bounds(
        [
            ego_xy[start:end][ego_valid[start:end]],
            target_gt_xy[start:end][target_valid[start:end]],
            target_rollout_xy[:, start:end][:, target_valid[start:end]],
        ],
        margin=float(margin_m),
    )


def _outside_fraction(
    pred_xy: np.ndarray,
    valid: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> float:
    xmin, xmax, ymin, ymax = bounds
    points = pred_xy[:, valid]
    outside = (
        (points[..., 0] < xmin)
        | (points[..., 0] > xmax)
        | (points[..., 1] < ymin)
        | (points[..., 1] > ymax)
    )
    return float(outside.mean()) if outside.size else 0.0


@torch.inference_mode()
def visualize(args: argparse.Namespace) -> None:
    if int(args.num_rollouts) != 10:
        raise ValueError("This comparison expects exactly 10 rollouts per scene")
    if int(args.rollout_steps) != 80:
        raise ValueError("This comparison expects an 80-step rollout")
    if int(args.num_scenes) != 10:
        raise ValueError("This comparison expects the same ten selected scenes")

    checkpoint_path = Path(args.checkpoint).resolve()
    selected_manifest_path = Path(args.selected_manifest).resolve()
    reference_dir = Path(args.reference_h90_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (checkpoint_path, selected_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not reference_dir.is_dir():
        raise FileNotFoundError(reference_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(selected_manifest_path.read_text())
    records = list(selection["samples"])
    if len(records) != int(args.num_scenes):
        raise ValueError(
            f"Expected {args.num_scenes} selected records, found {len(records)}"
        )
    if len({str(record["scenario_id"]) for record in records}) != len(records):
        raise ValueError("Selected manifest contains duplicate raw scenarios")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.index is not None:
            torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    seed_everything(int(args.seed))
    model, normalizer, train_args, checkpoint_step = load_model_and_normalizer(
        checkpoint_path,
        device,
        args.weights,
    )
    if int(train_args.history_length) != 11:
        raise ValueError(
            f"Expected history_length=11, checkpoint has {train_args.history_length}"
        )
    if int(train_args.horizon) != 15:
        raise ValueError(f"Expected native horizon=15, checkpoint has {train_args.horizon}")
    commitment = (
        int(args.commitment_steps)
        if int(args.commitment_steps) > 0
        else int(train_args.commitment)
    )
    if commitment < 1 or commitment > int(train_args.horizon):
        raise ValueError(f"Invalid commitment={commitment}")

    dataset = WaymoVectorDataset(args.val_data_dir)
    subset = Subset(dataset, [int(record["dataset_index"]) for record in records])
    loader = make_loader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        device=device,
    )

    setattr(args, "eval_ctx", int(train_args.history_length))
    setattr(args, "horizon", int(args.rollout_steps))
    setattr(args, "eval_num_rollouts", int(args.num_rollouts))
    reference_bounds_image_paths: list[Path] = []
    full_bounds_image_paths: list[Path] = []
    result_rows: list[dict[str, Any]] = []
    repeats = int(args.num_rollouts)
    anchor = int(train_args.history_length) - 1
    for scene_index, (record, raw_batch) in enumerate(zip(records, loader)):
        batch = move_batch(raw_batch, device)
        agents = agents_to_bntf(batch["agents"], batch["agent_mask"])
        history, future = gather_agent_window(
            agents,
            torch.full((1,), anchor, dtype=torch.long, device=device),
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
        if not bool(targets.valid[0, 0].all()):
            raise RuntimeError(
                f"Recorded focus plan is not valid for all 80 steps in {record['scenario_id']}"
            )

        scene_seed = int(args.seed) + int(record["sample_order"])
        generator = torch.Generator(device=device).manual_seed(scene_seed)
        poses = rollout_receding_horizon(
            model,
            normalizer,
            initial_history=_repeat_batch(history, repeats),
            agent_mask=_repeat_batch(batch["agent_mask"], repeats),
            map_polylines=_repeat_batch(batch["map_polylines"], repeats),
            map_mask=_repeat_batch(batch["map_mask"], repeats),
            current_light_sequence=_repeat_batch(
                batch["lights"][:, anchor : anchor + int(args.rollout_steps)],
                repeats,
            ),
            current_light_mask_sequence=_repeat_batch(
                batch["light_mask"][:, anchor : anchor + int(args.rollout_steps)],
                repeats,
            ),
            focus_action_sequence=_repeat_batch(targets.actions[:, 0], repeats),
            focus_action_valid=_repeat_batch(targets.valid[:, 0], repeats),
            rollout_steps=int(args.rollout_steps),
            commitment=commitment,
            solver_steps=int(args.solver_steps),
            generator=generator,
        )
        focus_gt_xy = targets.future_pose[0, 0, :, 0:2]
        focus_replay_error = torch.linalg.vector_norm(
            poses[:, 0, :, 0:2] - focus_gt_xy[None],
            dim=-1,
        )
        focus_replay_max_error_m = float(focus_replay_error.max())
        if focus_replay_max_error_m > 1e-3:
            raise RuntimeError(
                f"Controlled focus failed to replay its recorded plan in {record['scenario_id']}: "
                f"max error={focus_replay_max_error_m:.6f}m"
            )
        gt_tkf = agents[0].permute(1, 0, 2).detach().float().cpu().numpy()
        context_xy = gt_tkf[: int(train_args.history_length), :, 0:2]
        future_xy = poses[..., 0:2].permute(0, 2, 1, 3).detach().float().cpu().numpy()
        pred_xy = np.concatenate(
            (np.repeat(context_xy[None], repeats, axis=0), future_xy),
            axis=1,
        )
        if pred_xy.shape != (repeats, 91, int(agents.shape[1]), 2):
            raise RuntimeError(f"Unexpected prediction shape {pred_xy.shape}")

        agent_ids = batch["agent_ids"][0].detach().cpu().numpy()
        target_slot = int(record["target_slot"])
        actual_target_id = int(agent_ids[target_slot])
        if actual_target_id != int(record["target_track_id"]):
            raise RuntimeError(
                f"Target mismatch in {record['scenario_id']}: expected "
                f"{record['target_track_id']}, loaded {actual_target_id}"
            )
        bounds = _reference_bounds(
            reference_dir,
            record,
            scene_index=scene_index,
            context_frames=int(train_args.history_length),
            horizon=int(args.rollout_steps),
            margin_m=float(args.margin_m),
        )
        valid = gt_tkf[anchor:, target_slot, 5] > 0.5
        target_prediction = pred_xy[:, anchor:, target_slot]
        outside_fraction = _outside_fraction(target_prediction, valid, bounds)
        image_path = output_dir / f"scene_{scene_index:02d}_{record['scenario_id']}.png"
        metrics = _plot_scene(
            args=args,
            record=record,
            gt_tkf=gt_tkf,
            pred_xy=pred_xy,
            map_polylines=batch["map_polylines"][0].detach().float().cpu().numpy(),
            map_mask=batch["map_mask"][0].detach().cpu().numpy().astype(bool),
            agent_ids=agent_ids,
            output_path=image_path,
            bounds_override=bounds,
        )
        full_bounds_image_path = output_dir / "full_bounds" / (
            f"scene_{scene_index:02d}_{record['scenario_id']}.png"
        )
        full_bounds_metrics = _plot_scene(
            args=args,
            record=record,
            gt_tkf=gt_tkf,
            pred_xy=pred_xy,
            map_polylines=batch["map_polylines"][0].detach().float().cpu().numpy(),
            map_mask=batch["map_mask"][0].detach().cpu().numpy().astype(bool),
            agent_ids=agent_ids,
            output_path=full_bounds_image_path,
        )
        for key, value in metrics.items():
            if not np.allclose(
                np.asarray(value),
                np.asarray(full_bounds_metrics[key]),
                rtol=0.0,
                atol=1e-7,
            ):
                raise RuntimeError(f"Plot bounds unexpectedly changed metric {key}")
        trajectory_path = output_dir / (
            f"scene_{scene_index:02d}_{record['scenario_id']}_trajectories.npz"
        )
        np.savez_compressed(
            trajectory_path,
            ego_gt_xy=gt_tkf[:, 0, 0:2],
            ego_gt_valid=gt_tkf[:, 0, 5] > 0.5,
            target_gt_xy=gt_tkf[:, target_slot, 0:2],
            target_gt_valid=gt_tkf[:, target_slot, 5] > 0.5,
            target_rollout_xy=pred_xy[:, :, target_slot],
            context_frames=np.asarray(int(train_args.history_length)),
            rollout_steps=np.asarray(int(args.rollout_steps)),
            native_plan_horizon=np.asarray(int(train_args.horizon)),
            commitment_steps=np.asarray(commitment),
            solver_steps=np.asarray(int(args.solver_steps)),
            scene_seed=np.asarray(scene_seed),
            h90_reference_bounds=np.asarray(bounds),
        )
        row = {
            **record,
            "scene_rank": scene_index,
            "scene_seed": scene_seed,
            "image": str(full_bounds_image_path),
            "full_bounds_image": str(full_bounds_image_path),
            "same_h90_bounds_image": str(image_path),
            "trajectory_npz": str(trajectory_path),
            "focus_replay_max_error_m": focus_replay_max_error_m,
            "fraction_target_rollout_points_outside_h90_plot_bounds": outside_fraction,
            **metrics,
        }
        result_rows.append(row)
        reference_bounds_image_paths.append(image_path)
        full_bounds_image_paths.append(full_bounds_image_path)
        print(
            f"generated {scene_index + 1}/{len(records)} scenario={record['scenario_id']} "
            f"target={actual_target_id} mean_ADE={metrics['mean_ade_m']:.3f}m "
            f"pairwise={metrics['mean_pairwise_trajectory_distance_m']:.3f}m "
            f"outside_reference_bounds={outside_fraction:.4f}",
            flush=True,
        )

    reference_bounds_contact_sheet = output_dir / "contact_sheet_intersection10_n10.png"
    full_bounds_contact_sheet = output_dir / "contact_sheet_intersection10_n10_full_bounds.png"
    _make_contact_sheet(reference_bounds_image_paths, reference_bounds_contact_sheet)
    _make_contact_sheet(full_bounds_image_paths, full_bounds_contact_sheet)
    payload = {
        "eval_ckpt": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "weights": str(args.weights),
        "model_label": str(args.model_label),
        "protocol": (
            f"ctx11_native_h15_commit{commitment}_receding_h80_"
            f"solver{int(args.solver_steps)}_recorded_focus_plan_target_agent_n10"
        ),
        "provided_ego_plan": "recorded ground-truth focus trajectory",
        "ego_draw_count_per_figure": 1,
        "target_ground_truth_draw_count_per_figure": 1,
        "target_rollout_count_per_figure": repeats,
        "other_agent_trajectories_drawn": 0,
        "selected_manifest": str(selected_manifest_path),
        "plot_bounds_reference": str(reference_dir),
        "contact_sheet": str(full_bounds_contact_sheet),
        "full_bounds_contact_sheet": str(full_bounds_contact_sheet),
        "same_h90_bounds_contact_sheet": str(reference_bounds_contact_sheet),
        "samples": result_rows,
    }
    metrics_path = output_dir / "rollout_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote_metrics={metrics_path}", flush=True)
    print(f"wrote_full_bounds_contact_sheet={full_bounds_contact_sheet}", flush=True)
    print(f"wrote_same_h90_bounds_contact_sheet={reference_bounds_contact_sheet}", flush=True)
    del model, normalizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_data_dir", required=True)
    parser.add_argument("--selected_manifest", required=True)
    parser.add_argument("--reference_h90_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_scenes", type=int, default=10)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--rollout_steps", type=int, default=80)
    parser.add_argument("--commitment_steps", type=int, default=0)
    parser.add_argument("--solver_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--margin_m", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--model_label", default="DirectActionFlow step 90k (EMA)")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
