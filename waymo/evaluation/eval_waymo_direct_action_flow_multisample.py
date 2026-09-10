#!/usr/bin/env python3
"""Evaluate K full DirectActionFlow rollouts and retain ADE/CPD details.

Follow the checkpoint's focus-conditioning mode. Conditioned checkpoints use
logged focus actions and score nonfocus agents; generate-all checkpoints receive
no future focus actions and score all agents, including focus.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
from waymo.training.world_model.multisample_validation import (
    FLOW_ERD_CPD_TYPE_IDS,
    FLOW_ERD_CPD_TYPE_NAMES,
    flow_erd_cpd_metrics,
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
    parser.add_argument("--rollout_ade_csv", required=True)
    parser.add_argument("--scene_metrics_csv", required=True)
    parser.add_argument("--details_npz", required=True)
    parser.add_argument("--start_batch", type=int, default=0, help="Skip completed batches, retaining original dataset indices and seeds.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--focus_mode", choices=("checkpoint", "conditioned", "generate_all"),
        default="checkpoint",
        help="Assert the checkpoint focus mode; checkpoint infers it automatically.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_batches", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_rollouts", type=int, default=32)
    parser.add_argument("--rollout_steps", type=int, default=80)
    parser.add_argument(
        "--commitment_steps",
        type=int,
        default=0,
        help="Executed steps per replan; 0 uses the checkpoint training commitment.",
    )
    parser.add_argument("--solver_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12346)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument(
        "--cpd_type_scales",
        type=float,
        nargs=3,
        metavar=("VEHICLE", "PEDESTRIAN", "CYCLIST"),
        required=True,
    )
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


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def decoded_scenario_id(path: str) -> str:
    """Return a stable identifier without reopening every validation NPZ."""
    name = Path(path).stem
    return name.split("__focus_", maxsplit=1)[0]


def resolve_focus_conditioning(train_args: argparse.Namespace, mode: str) -> bool:
    # Older conditioned checkpoints predate this training argument.
    conditioned = bool(getattr(train_args, "condition_focus_actions", True))
    if mode != "checkpoint" and conditioned != (mode == "conditioned"):
        raise ValueError(f"Requested focus_mode={mode} conflicts with checkpoint "
                         f"condition_focus_actions={conditioned}")
    return conditioned


def summarize_agent_scopes(
    candidate_agent_ade: np.ndarray, valid_steps: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Aggregate each scope with valid-point weights, then equal scene weights.

    Best joint rollout is selected independently for each scope. Scenes without
    any valid points in a scope are omitted, not assigned zero error.
    """
    result = {}
    for scope in ("all", "nonfocus", "focus"):
        weights = valid_steps.copy()
        if scope == "nonfocus":
            weights[:, 0] = 0
        elif scope == "focus":
            weights[:, 1:] = 0
        denominator = weights.sum(axis=1)
        scene_valid = denominator > 0
        if not scene_valid.any():
            result[scope] = {"scene_count": 0, "mean_ade_m": None, "minade_m": None}
            continue
        candidate = (
            np.nan_to_num(candidate_agent_ade, nan=0.0) * weights[:, None]
        ).sum(axis=2)[scene_valid] / denominator[scene_valid, None]
        result[scope] = {
            "scene_count": int(scene_valid.sum()),
            "valid_agent_time_points_per_rollout": int(weights.sum()),
            "mean_ade_m": float(candidate.mean()),
            "minade_m": float(candidate.min(axis=1).mean()),
        }
    return result


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "eval_batch_size",
        "eval_max_batches",
        "num_workers",
        "num_rollouts",
        "rollout_steps",
        "solver_steps",
        "log_every",
    ):
        value = int(getattr(args, name))
        if name == "num_workers":
            if value < 0:
                raise ValueError(f"--{name} must be >= 0, got {value}")
        else:
            require_positive(f"--{name}", value)
    if int(args.commitment_steps) < 0:
        raise ValueError(
            f"--commitment_steps must be >= 0, got {args.commitment_steps}"
        )
    if int(args.num_rollouts) < 2:
        raise ValueError("--num_rollouts must be >= 2 to compute CPD")
    if any((not np.isfinite(scale)) or scale <= 0.0 for scale in args.cpd_type_scales):
        raise ValueError("--cpd_type_scales values must be finite and positive")

    if not 0 <= args.start_batch < args.eval_max_batches:
        raise ValueError("start_batch must be in [0, eval_max_batches)")

    checkpoint_path = Path(args.checkpoint).resolve()
    val_data_dir = Path(args.val_data_dir).resolve()
    output_json = Path(args.output_json).resolve()
    rollout_ade_csv = Path(args.rollout_ade_csv).resolve()
    scene_metrics_csv = Path(args.scene_metrics_csv).resolve()
    details_npz = Path(args.details_npz).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not val_data_dir.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {val_data_dir}")
    for output_path in (output_json, rollout_ade_csv, scene_metrics_csv, details_npz):
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
    condition_focus_actions = resolve_focus_conditioning(train_args, args.focus_mode)
    scope = "nonfocus" if condition_focus_actions else "all_agent"
    scene_ade_column = f"{scope}_scene_ade_m"
    valid_points_column = (
        "valid_nonfocus_agent_time_points" if condition_focus_actions
        else "valid_all_agent_time_points"
    )
    valid_agents_column = "valid_nonfocus_agents" if condition_focus_actions else "valid_all_agents"
    print(f"checkpoint_step={checkpoint_step} condition_focus_actions={condition_focus_actions} "
          f"metric_scope={scope}", flush=True)
    if int(train_args.history_length) != 11:
        raise ValueError(
            "This evaluation requires an 11-frame context, but the checkpoint "
            f"was trained with history_length={train_args.history_length}"
        )
    commitment_steps = (
        int(args.commitment_steps)
        if int(args.commitment_steps) > 0
        else int(train_args.commitment)
    )
    if commitment_steps > int(train_args.horizon):
        raise ValueError(
            f"commitment_steps={commitment_steps} exceeds the model plan horizon "
            f"{train_args.horizon}"
        )

    dataset = WaymoVectorDataset(str(val_data_dir))
    loader = make_loader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    candidate_scene_ade_parts: list[np.ndarray] = []
    candidate_agent_ade_parts: list[np.ndarray] = []
    agent_valid_steps_parts: list[np.ndarray] = []
    agent_ids_parts: list[np.ndarray] = []
    cpd_type_mse_parts: list[np.ndarray] = []
    cpd_type_count_parts: list[np.ndarray] = []
    cpd_type_present_parts: list[np.ndarray] = []
    scene_cpd_parts: list[np.ndarray] = []
    scene_cpd_unscaled_parts: list[np.ndarray] = []
    scene_cpd_valid_parts: list[np.ndarray] = []
    all_dataset_indices: list[int] = []
    all_paths: list[str] = []
    all_scenario_ids: list[str] = []
    pair_indices: np.ndarray | None = None

    scene_ade_sum = 0.0
    scene_minade_sum = 0.0
    cpd_sum = 0.0
    cpd_valid_scenes = 0
    valid_point_count = 0
    scene_count = 0
    batches = 0
    started = time.monotonic()

    with rollout_ade_csv.open("w", newline="") as rollout_handle, scene_metrics_csv.open(
        "w", newline=""
    ) as scene_handle:
        rollout_writer = csv.DictWriter(
            rollout_handle,
            fieldnames=(
                "dataset_index",
                "batch_index",
                "scene_in_batch",
                "scenario_id",
                "npz_path",
                "rollout_index",
                "rollout_seed",
                scene_ade_column,
                valid_points_column,
            ),
        )
        scene_writer = csv.DictWriter(
            scene_handle,
            fieldnames=(
                "dataset_index",
                "batch_index",
                "scene_in_batch",
                "scenario_id",
                "npz_path",
                "mean_ade_over_rollouts_m",
                "joint_minade_over_rollouts_m",
                "per_agent_minade_m",
                "flow_erd_cpd",
                "flow_erd_cpd_unscaled",
                valid_agents_column,
                valid_points_column,
            ),
        )
        rollout_writer.writeheader()
        scene_writer.writeheader()

        for batch_index, raw_batch in enumerate(loader):
            if batch_index >= int(args.eval_max_batches):
                break
            if batch_index < args.start_batch:
                continue
            batch = move_batch(raw_batch, device)
            agents = agents_to_bntf(batch["agents"], batch["agent_mask"])
            batch_size = int(agents.shape[0])
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
                    (batch_size,),
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

            candidate_poses = []
            rollout_seeds = []
            for rollout_index in range(int(args.num_rollouts)):
                rollout_seed = (
                    int(args.seed)
                    + batch_index * int(args.num_rollouts)
                    + rollout_index
                )
                rollout_seeds.append(rollout_seed)
                generator = torch.Generator(device=device).manual_seed(rollout_seed)
                candidate_poses.append(
                    rollout_receding_horizon(
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
                        focus_action_sequence=(targets.actions[:, 0] if condition_focus_actions else None),
                        focus_action_valid=(targets.valid[:, 0] if condition_focus_actions else None),
                        rollout_steps=int(args.rollout_steps),
                        commitment=commitment_steps,
                        solver_steps=int(args.solver_steps),
                        generator=generator,
                    )
                )
            # (B,R,N,H,3), with each R a complete joint rollout.
            poses = torch.stack(candidate_poses, dim=1)

            valid = targets.valid.clone()
            if condition_focus_actions:
                valid[:, 0] = False
            distance = torch.linalg.vector_norm(
                poses[..., 0:2] - targets.future_pose[:, None, ..., 0:2],
                dim=-1,
            )
            per_agent_valid_steps = valid.sum(dim=-1)
            per_agent_ade = (distance * valid[:, None]).sum(dim=-1) / per_agent_valid_steps[
                :, None
            ].clamp_min(1)
            valid_agents = per_agent_valid_steps > 0
            per_agent_ade = torch.where(
                valid_agents[:, None],
                per_agent_ade,
                torch.full_like(per_agent_ade, torch.nan),
            )
            per_scene_valid_points = valid.sum(dim=(1, 2))
            candidate_scene_ade = (distance * valid[:, None]).sum(dim=(2, 3)) / (
                per_scene_valid_points[:, None].clamp_min(1)
            )
            scene_mean_ade = candidate_scene_ade.mean(dim=1)
            scene_minade = candidate_scene_ade.min(dim=1).values
            per_agent_minade = torch.nan_to_num(
                per_agent_ade.min(dim=1).values,
                nan=0.0,
            )
            scene_per_agent_minade = (
                per_agent_minade * valid_agents
            ).sum(dim=1) / valid_agents.sum(dim=1).clamp_min(1)

            # CPD consumes one context pose plus the H generated future poses.
            context_xy = history[:, :, -1, 0:2][:, None, None].expand(
                -1, int(args.num_rollouts), -1, -1, -1
            )
            predicted_xy = torch.cat(
                (context_xy, poses[..., 0:2].permute(0, 1, 3, 2, 4)),
                dim=2,
            )
            cpd_metadata = torch.cat(
                (
                    history[:, :, -1:].permute(0, 2, 1, 3),
                    future.permute(0, 2, 1, 3),
                ),
                dim=1,
            )
            _, cpd_components = flow_erd_cpd_metrics(
                predicted_xy,
                cpd_metadata,
                future_start=1,
                type_scales=args.cpd_type_scales,
                exclude_focus=condition_focus_actions,
            )
            scene_cpd = cpd_components["scene_cpd"]
            scene_cpd_unscaled = cpd_components["scene_cpd_unscaled"]
            scene_cpd_valid = cpd_components["scene_valid"]

            start_index = batch_index * int(args.eval_batch_size)
            batch_paths = dataset.paths[start_index : start_index + batch_size]
            for scene_in_batch, npz_path in enumerate(batch_paths):
                dataset_index = start_index + scene_in_batch
                scenario_id = decoded_scenario_id(npz_path)
                common = {
                    "dataset_index": dataset_index,
                    "batch_index": batch_index,
                    "scene_in_batch": scene_in_batch,
                    "scenario_id": scenario_id,
                    "npz_path": npz_path,
                }
                for rollout_index, rollout_seed in enumerate(rollout_seeds):
                    rollout_writer.writerow(
                        {
                            **common,
                            "rollout_index": rollout_index,
                            "rollout_seed": rollout_seed,
                            scene_ade_column: float(
                                candidate_scene_ade[scene_in_batch, rollout_index]
                            ),
                            valid_points_column: int(
                                per_scene_valid_points[scene_in_batch]
                            ),
                        }
                    )
                scene_writer.writerow(
                    {
                        **common,
                        "mean_ade_over_rollouts_m": float(scene_mean_ade[scene_in_batch]),
                        "joint_minade_over_rollouts_m": float(scene_minade[scene_in_batch]),
                        "per_agent_minade_m": float(scene_per_agent_minade[scene_in_batch]),
                        "flow_erd_cpd": float(scene_cpd[scene_in_batch]),
                        "flow_erd_cpd_unscaled": float(
                            scene_cpd_unscaled[scene_in_batch]
                        ),
                        valid_agents_column: int(valid_agents[scene_in_batch].sum()),
                        valid_points_column: int(
                            per_scene_valid_points[scene_in_batch]
                        ),
                    }
                )
                all_dataset_indices.append(dataset_index)
                all_paths.append(npz_path)
                all_scenario_ids.append(scenario_id)
            rollout_handle.flush()
            scene_handle.flush()

            candidate_scene_ade_parts.append(candidate_scene_ade.cpu().numpy())
            candidate_agent_ade_parts.append(per_agent_ade.cpu().numpy())
            agent_valid_steps_parts.append(per_agent_valid_steps.cpu().numpy())
            agent_ids_parts.append(batch["agent_ids"].cpu().numpy())
            cpd_type_mse_parts.append(cpd_components["type_mse"].cpu().numpy())
            cpd_type_count_parts.append(cpd_components["type_count"].cpu().numpy())
            cpd_type_present_parts.append(cpd_components["type_present"].cpu().numpy())
            scene_cpd_parts.append(scene_cpd.cpu().numpy())
            scene_cpd_unscaled_parts.append(scene_cpd_unscaled.cpu().numpy())
            scene_cpd_valid_parts.append(scene_cpd_valid.cpu().numpy())
            current_pair_indices = cpd_components["pair_indices"].cpu().numpy()
            if pair_indices is None:
                pair_indices = current_pair_indices
            elif not np.array_equal(pair_indices, current_pair_indices):
                raise RuntimeError("CPD pair ordering changed between batches")

            scene_ade_sum += float(scene_mean_ade.sum())
            scene_minade_sum += float(scene_minade.sum())
            valid_cpd_float = scene_cpd_valid.to(scene_cpd.dtype)
            cpd_sum += float((scene_cpd * valid_cpd_float).sum())
            cpd_valid_scenes += int(scene_cpd_valid.sum())
            valid_point_count += int(per_scene_valid_points.sum())
            scene_count += batch_size
            batches += 1

            if batches % int(args.log_every) == 0 or batches == int(args.eval_max_batches):
                elapsed = time.monotonic() - started
                print(
                    f"progress batches={batches + args.start_batch}/{args.eval_max_batches} "
                    f"resumed_prefix_batches={args.start_batch} "
                    f"scenes={scene_count} rollouts_per_scene={args.num_rollouts} "
                    f"mean_ade_m={scene_ade_sum / max(1, scene_count):.6f} "
                    f"minade_m={scene_minade_sum / max(1, scene_count):.6f} "
                    f"cpd={cpd_sum / max(1, cpd_valid_scenes):.6f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    if batches == 0 or scene_count == 0:
        raise RuntimeError("Validation loader produced no evaluation batches")
    if batches != int(args.eval_max_batches) - args.start_batch:
        raise RuntimeError(
            f"Requested {args.eval_max_batches} batches, but only {batches} were available"
        )
    if valid_point_count == 0:
        raise RuntimeError("No valid generated-agent trajectory points were evaluated")
    if cpd_valid_scenes == 0:
        raise RuntimeError("No scenes contained a valid generated-agent CPD roster")
    assert pair_indices is not None

    candidate_scene_ade_all = np.concatenate(candidate_scene_ade_parts, axis=0)
    candidate_agent_ade_all = np.concatenate(candidate_agent_ade_parts, axis=0)
    agent_valid_steps_all = np.concatenate(agent_valid_steps_parts, axis=0)
    agent_ids_all = np.concatenate(agent_ids_parts, axis=0)
    scene_cpd_all = np.concatenate(scene_cpd_parts, axis=0)
    scene_cpd_unscaled_all = np.concatenate(scene_cpd_unscaled_parts, axis=0)
    scene_cpd_valid_all = np.concatenate(scene_cpd_valid_parts, axis=0)

    valid_agent_all = agent_valid_steps_all > 0
    per_agent_minade_all = np.where(
        valid_agent_all[:, None, :], candidate_agent_ade_all, np.inf
    ).min(axis=1)
    per_agent_minade_m = float(per_agent_minade_all[valid_agent_all].mean())
    mean_ade_by_rollout = candidate_scene_ade_all.mean(axis=0)
    elapsed = time.monotonic() - started
    result = {
        "metric_scope": (
            "generated_nonfocus_agents; focus_future_is_conditioned_and_excluded"
            if condition_focus_actions else
            "generated_all_agents; focus_future_is_not_conditioned; focus_is_included"
        ),
        "condition_focus_actions": condition_focus_actions,
        "cpd_exclude_focus": condition_focus_actions,
        "ade_definition": (
            f"Each candidate ADE averages xy error over valid {scope} agent-time points "
            "within one scene. mean_ade_m averages scenes and rollouts equally; minade_m "
            "takes the best joint rollout per scene before averaging scenes."
        ),
        "mean_ade_m": float(candidate_scene_ade_all.mean()),
        "minade_m": float(candidate_scene_ade_all.min(axis=1).mean()),
        "per_agent_minade_m": per_agent_minade_m,
        "first_rollout_ade_m": float(candidate_scene_ade_all[:, 0].mean()),
        "mean_ade_by_rollout_m": mean_ade_by_rollout.tolist(),
        "flow_erd_cpd": float(scene_cpd_all[scene_cpd_valid_all].mean()),
        "flow_erd_cpd_unscaled": float(
            scene_cpd_unscaled_all[scene_cpd_valid_all].mean()
        ),
        "flow_erd_cpd_valid_scene_fraction": float(scene_cpd_valid_all.mean()),
        "flow_erd_cpd_type_order": list(FLOW_ERD_CPD_TYPE_NAMES),
        "flow_erd_cpd_type_ids": list(FLOW_ERD_CPD_TYPE_IDS),
        "flow_erd_cpd_type_scales_m": [float(value) for value in args.cpd_type_scales],
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "weights": args.weights,
        "val_data_dir": str(val_data_dir),
        "history_context_steps": int(train_args.history_length),
        "model_plan_horizon": int(train_args.horizon),
        "model_training_commitment_steps": int(train_args.commitment),
        "rollout_commitment_steps": commitment_steps,
        "rollout_steps": int(args.rollout_steps),
        "num_rollouts": int(args.num_rollouts),
        "solver_steps": int(args.solver_steps),
        "seed": int(args.seed),
        "rollout_seed_formula": "seed + batch_index * num_rollouts + rollout_index",
        "eval_batches": batches,
        "start_batch": args.start_batch,
        "eval_batch_size": int(args.eval_batch_size),
        "scene_count": scene_count,
        f"{valid_points_column}_per_rollout": valid_point_count,
        "rollout_ade_csv": str(rollout_ade_csv),
        "scene_metrics_csv": str(scene_metrics_csv),
        "details_npz": str(details_npz),
        "elapsed_seconds": elapsed,
    }
    if not condition_focus_actions:
        result["ade_by_agent_scope"] = summarize_agent_scopes(
            candidate_agent_ade_all, agent_valid_steps_all,
        )

    atomic_write_npz(
        details_npz,
        candidate_scene_ade_m=candidate_scene_ade_all,
        candidate_agent_ade_m=candidate_agent_ade_all,
        agent_valid_steps=agent_valid_steps_all,
        agent_ids=agent_ids_all,
        dataset_index=np.asarray(all_dataset_indices, dtype=np.int64),
        scenario_id=np.asarray(all_scenario_ids, dtype=str),
        npz_path=np.asarray(all_paths, dtype=str),
        cpd_type_mse=np.concatenate(cpd_type_mse_parts, axis=0),
        cpd_type_count=np.concatenate(cpd_type_count_parts, axis=0),
        cpd_type_present=np.concatenate(cpd_type_present_parts, axis=0),
        cpd_pair_indices=pair_indices,
        scene_cpd=scene_cpd_all,
        scene_cpd_unscaled=scene_cpd_unscaled_all,
        scene_cpd_valid=scene_cpd_valid_all,
        cpd_type_ids=np.asarray(FLOW_ERD_CPD_TYPE_IDS, dtype=np.int64),
        cpd_type_names=np.asarray(FLOW_ERD_CPD_TYPE_NAMES, dtype=str),
        cpd_type_scales_m=np.asarray(args.cpd_type_scales, dtype=np.float64),
        condition_focus_actions=np.asarray(condition_focus_actions),
        cpd_exclude_focus=np.asarray(condition_focus_actions),
    )
    atomic_write_json(output_json, result)
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print("result " + " ".join(f"{key}={value}" for key, value in result.items()), flush=True)
    print(f"saved {Path(args.output_json).resolve()}", flush=True)


if __name__ == "__main__":
    main()
