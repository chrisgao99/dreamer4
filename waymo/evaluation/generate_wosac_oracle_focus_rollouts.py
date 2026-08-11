"""Generate local WOSAC rollouts from the legacy Waymo world model.

Protocol used by this evaluator:

* original WOMD indices 1..10 are the ten context frames;
* original indices 11..90 are generated autoregressively (H80);
* the focus agent supplies oracle future actions and its emitted trajectory is
  replaced by the logged future before saving;
* only the fixed 32 NPZ slots are emitted (unselected WOMD agents are not
  filled in);
* multiple stochastic rollouts of one focus-view are batched together.

The saved NPZ files are intentionally TensorFlow-independent.  They are scored
by ``score_wosac_oracle_focus_rollouts.py`` in a separate environment so that
TensorFlow cannot reserve memory on the inference GPU.
"""

from __future__ import annotations

import time

PROCESS_START = time.perf_counter()

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.core.vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo.evaluation import eval_waymo_world_model_horizons as base_eval  # noqa: E402
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


ORIGINAL_STEPS = 91
SEQUENCE_START = 1
EVAL_STEPS = 90
CONTEXT_STEPS = 10
FUTURE_STEPS = 80


class ManifestNpzDataset(wm.WaymoVectorDataset):
    """WaymoVectorDataset preserving the manifest order exactly."""

    def __init__(self, paths: list[str]):
        self.paths = [str(Path(path)) for path in paths]
        if not self.paths:
            raise ValueError("The selected manifest contains no NPZ paths")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_manifest_rows(
    path: Path,
    max_views: int,
    *,
    scenario_id: str | None = None,
    focus_track_id: int | None = None,
) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        all_rows = [dict(row) for row in csv.DictReader(handle)]
    rows = [row for row in all_rows if _as_bool(row.get("eligible_ooi_pair", True))]
    if scenario_id:
        rows = [row for row in rows if row.get("scenario_id") == scenario_id]
    if focus_track_id is not None:
        rows = [row for row in rows if int(row.get("focus_track_id", -1)) == int(focus_track_id)]
    if max_views > 0:
        rows = rows[:max_views]
    if not rows:
        raise ValueError(f"No eligible rows found in {path}")
    required = ("scenario_id", "focus_track_id", "npz_path", "scenario_pb_path")
    for row_index, row in enumerate(rows):
        for key in required:
            if not row.get(key):
                raise ValueError(f"Manifest row {row_index} is missing {key!r}")
        for key in ("npz_path", "scenario_pb_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"Manifest row {row_index} missing {key}: {row[key]}")
    return rows


def repeat_batch(batch: dict[str, Any], count: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.shape[0] != 1:
                raise ValueError(f"Expected scene batch size 1 for {key}, got {tuple(value.shape)}")
            out[key] = value.repeat_interleave(count, dim=0)
        else:
            out[key] = value
    return out


def repeat_optional(value: torch.Tensor | None, count: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape[0] != 1:
        raise ValueError(f"Expected leading dimension 1, got {tuple(value.shape)}")
    return value.repeat_interleave(count, dim=0)


def local_to_world_xy(local_xy: np.ndarray, origin_xy: np.ndarray, heading: float) -> np.ndarray:
    c = np.float32(math.cos(heading))
    s = np.float32(math.sin(heading))
    world = np.empty_like(local_xy, dtype=np.float32)
    world[..., 0] = c * local_xy[..., 0] - s * local_xy[..., 1] + origin_xy[0]
    world[..., 1] = s * local_xy[..., 0] + c * local_xy[..., 1] + origin_xy[1]
    return world


def wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)


def make_output_stem(view_index: int, row: dict[str, str]) -> str:
    return f"{view_index:05d}_{row['scenario_id']}_focus_{int(row['focus_track_id'])}"


def proxy_metrics(
    pred_xy: np.ndarray,
    gt_agents: np.ndarray,
    current_valid: np.ndarray,
) -> dict[str, float]:
    """Small model-space diagnostics; these are not the WOSAC metrics."""
    # pred_xy: [R, 80, K, 2], gt_agents: [80, K, F].  Focus slot 0 is
    # deliberately omitted because it is replaced by the oracle trajectory.
    score_slots = current_valid.copy()
    score_slots[0] = False
    gt_valid = gt_agents[..., 5] > 0.5
    valid = gt_valid & score_slots[None, :]
    distance = np.linalg.norm(pred_xy - gt_agents[None, ..., 0:2], axis=-1)
    denom = max(1, int(valid.sum()))
    ade_per_rollout = (distance * valid[None]).sum(axis=(1, 2)) / denom
    return {
        "nonfocus_ade_mean_m": float(ade_per_rollout.mean()),
        "nonfocus_ade_min_m": float(ade_per_rollout.min()),
        "num_scored_nonfocus_slots": int(score_slots.sum()),
        "num_scored_nonfocus_states": int(valid.sum()),
    }


@torch.inference_mode()
def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.eval_schedule != "shortcut":
        raise ValueError("This evaluator currently requires --eval_schedule shortcut")
    if int(args.eval_ctx) != CONTEXT_STEPS:
        raise ValueError(f"eval_ctx must be {CONTEXT_STEPS}")
    if int(args.eval_seq_len) != ORIGINAL_STEPS:
        raise ValueError(f"eval_seq_len must be {ORIGINAL_STEPS}")
    if int(args.eval_horizon) != FUTURE_STEPS:
        raise ValueError(f"eval_horizon must be {FUTURE_STEPS}")
    if int(args.num_rollouts) != 32:
        raise ValueError("Local WOSAC scoring requires exactly 32 rollouts")
    if not 1 <= int(args.rollout_batch_size) <= int(args.num_rollouts):
        raise ValueError("rollout_batch_size must be in [1, num_rollouts]")
    tokenizer_chunk_window = int(args.tokenizer_chunk_window)
    tokenizer_chunk_stride = int(args.tokenizer_chunk_stride)
    if not 1 <= tokenizer_chunk_window <= 32:
        raise ValueError(
            "WOSAC rollout generation requires tokenizer chunks in [1, 32] timesteps; "
            f"got tokenizer_chunk_window={tokenizer_chunk_window}"
        )
    if not 1 <= tokenizer_chunk_stride <= tokenizer_chunk_window:
        raise ValueError(
            "tokenizer_chunk_stride must be in [1, tokenizer_chunk_window]; "
            f"got stride={tokenizer_chunk_stride}, window={tokenizer_chunk_window}"
        )
    tokenizer_chunk_ranges = wm.tokenizer_time_chunk_ranges(
        ORIGINAL_STEPS,
        tokenizer_chunk_window,
        tokenizer_chunk_stride,
    )
    if max(end - start for start, end in tokenizer_chunk_ranges) > 32:
        raise AssertionError(f"Tokenizer chunk longer than 32 timesteps: {tokenizer_chunk_ranges}")

    manifest_path = Path(args.manifest).resolve()
    rows = load_manifest_rows(
        manifest_path,
        int(args.max_views),
        scenario_id=args.scenario_id,
        focus_track_id=args.focus_track_id,
    )
    if args.preflight_only:
        print(
            f"preflight_ok selected_views={len(rows)} manifest={manifest_path} "
            f"first_npz={rows[0]['npz_path']}",
            flush=True,
        )
        return {"preflight_only": True, "selected_views": len(rows)}

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise RuntimeError("Rollout generation is intended to run on CUDA; pass --device cuda")
    wm.seed_everything(args.seed)

    dataset = ManifestNpzDataset([row["npz_path"] for row in rows])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=wm.worker_init_fn,
        collate_fn=wm._collate,
    )

    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    if isinstance(tokenizer, wm.FrozenWaymoFocusTokenizer):
        raise ValueError("This local WOSAC evaluator expects the 32-slot vector tokenizer")
    n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
    d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor:
        raise ValueError(f"n_latents={n_latents} is not divisible by packing_factor={args.packing_factor}")
    args.n_spatial = n_latents // args.packing_factor
    args.d_spatial = d_bottleneck * args.packing_factor

    ckpt = torch.load(args.eval_ckpt, map_location="cpu")
    if ckpt.get("format") == base_eval.MOTION_LATENT_V1_FORMAT:
        raise ValueError("Expected the legacy world-model checkpoint, not MotionLatent V1")
    dyn = base_eval.build_dynamics(
        args,
        d_bottleneck,
        device,
        map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None,
    )
    base_eval.load_dynamics_state(dyn, args.eval_ckpt, ckpt=ckpt)
    dyn.eval()

    schedule = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    output_dir = Path(args.output_dir).resolve()
    rollout_dir = output_dir / "rollouts"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = output_dir / "rollout_manifest.jsonl"

    setup_seconds = time.perf_counter() - PROCESS_START
    print(
        f"generation_protocol=oracle_focus_local_32slot context_original_indices=1..10 "
        f"future_original_indices=11..90 rollouts={args.num_rollouts} "
        f"rollout_batch_size={args.rollout_batch_size} solver_steps_per_frame={schedule['K']}",
        flush=True,
    )
    print(
        f"checkpoint={args.eval_ckpt} ckpt_step={int(ckpt.get('step', -1))} "
        f"views={len(rows)} setup_seconds={setup_seconds:.2f}",
        flush=True,
    )
    print(
        f"tokenizer_encode_decode_chunk_window={tokenizer_chunk_window} "
        f"stride={tokenizer_chunk_stride} ranges={tokenizer_chunk_ranges} max_chunk_steps=32",
        flush=True,
    )

    generation_start = time.perf_counter()
    view_records: list[dict[str, Any]] = []
    proxy_totals: dict[str, float] = {}
    with manifest_out.open("w") as manifest_handle:
        for view_index, (row, raw_batch) in enumerate(zip(rows, loader), start=1):
            view_start = time.perf_counter()
            batch_full = wm.move_batch(raw_batch, device)
            if int(batch_full["lights"].shape[1]) != ORIGINAL_STEPS:
                raise ValueError(
                    f"{row['npz_path']} has {batch_full['lights'].shape[1]} frames; expected {ORIGINAL_STEPS}"
                )
            if int(batch_full["agent_ids"][0, 0].item()) != int(row["focus_track_id"]):
                raise ValueError(
                    f"Focus mismatch for {row['npz_path']}: slot0={int(batch_full['agent_ids'][0, 0])} "
                    f"manifest={row['focus_track_id']}"
                )

            actions_full, act_mask_full, _ = wm.build_ego_action_features(batch_full, args)
            z_gt_full, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
                tokenizer,
                batch_full,
                args,
                return_map=args.dynamics_attend_map,
            )
            z_gt = z_gt_full[:, SEQUENCE_START:ORIGINAL_STEPS]
            actions = None if actions_full is None else actions_full[:, SEQUENCE_START:ORIGINAL_STEPS]
            act_mask = None if act_mask_full is None else act_mask_full[:, SEQUENCE_START:ORIGINAL_STEPS]
            z_gt_packed = wm.pack_bottleneck_to_spatial(
                z_gt,
                n_spatial=args.n_spatial,
                k=args.packing_factor,
            )

            batch_agents = wm.agents_to_btkf(batch_full["agents"], batch_full["agent_mask"])
            gt_future = batch_agents[0, 11:91].detach().float().cpu().numpy()
            current_valid = (
                batch_full["agent_mask"][0]
                & (batch_agents[0, 10, :, 5] > 0.5)
            ).detach().cpu().numpy().astype(bool)
            if not bool(current_valid[0]):
                raise ValueError(f"Focus slot is not current-valid: {row['npz_path']}")

            xy_parts: list[np.ndarray] = []
            yaw_parts: list[np.ndarray] = []
            valid_probability_parts: list[np.ndarray] = []
            remaining = int(args.num_rollouts)
            while remaining > 0:
                microbatch = min(int(args.rollout_batch_size), remaining)
                repeated_full = repeat_batch(batch_full, microbatch)
                z_pred_packed = wm.sample_autoregressive_packed_sequence(
                    wm.unwrap_model(dyn),
                    z_gt_packed=z_gt_packed.repeat_interleave(microbatch, dim=0),
                    actions=repeat_optional(actions, microbatch),
                    act_mask=repeat_optional(act_mask, microbatch),
                    map_tokens=repeat_optional(map_tokens, microbatch),
                    map_mask=repeat_optional(map_mask, microbatch),
                    ctx_length=CONTEXT_STEPS,
                    horizon=FUTURE_STEPS,
                    k_max=args.k_max,
                    sched=schedule,
                    max_rollout_window=args.max_rollout_window,
                )
                z_pred = wm.unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)

                # Decode at the original 91-frame alignment.  Prepending the
                # dropped frame preserves the tokenizer's [0,32), [30,62),
                # [59,91) chunk boundaries.
                z_decode_full = torch.cat(
                    [z_gt_full[:, :1].repeat_interleave(microbatch, dim=0), z_pred],
                    dim=1,
                )
                decoded = wm.decode_batch_z_for_world_model(tokenizer, z_decode_full, repeated_full, args)
                anchor_xy = wm.agents_to_btkf(
                    repeated_full["agents"], repeated_full["agent_mask"]
                )[:, 0, :, 0:2]
                decoded_xy = decoder_agent_xy(
                    decoded,
                    agent_xy_loss=args.agent_xy_loss,
                    agent_xy_parameterization=args.agent_xy_parameterization,
                    anchor_xy=anchor_xy,
                )[:, 11:91]
                decoded_yaw = torch.atan2(
                    decoded.agent_continuous[:, 11:91, :, 5],
                    decoded.agent_continuous[:, 11:91, :, 6],
                )
                decoded_valid_probability = torch.sigmoid(decoded.agent_valid_logits[:, 11:91])

                xy_parts.append(decoded_xy.detach().float().cpu().numpy())
                yaw_parts.append(decoded_yaw.detach().float().cpu().numpy())
                valid_probability_parts.append(
                    decoded_valid_probability.detach().float().cpu().numpy()
                )
                remaining -= microbatch
                del repeated_full, z_pred_packed, z_pred, z_decode_full, decoded

            pred_xy_local = np.concatenate(xy_parts, axis=0).astype(np.float32)
            pred_yaw_local = np.concatenate(yaw_parts, axis=0).astype(np.float32)
            pred_valid_probability = np.concatenate(valid_probability_parts, axis=0).astype(np.float32)
            if pred_xy_local.shape[0] != int(args.num_rollouts):
                raise RuntimeError(f"Generated {pred_xy_local.shape[0]} rollouts, expected {args.num_rollouts}")

            # Oracle focus means both its conditioning actions and the emitted
            # trajectory are exactly logged.  Slot 0 is guaranteed to be focus.
            pred_xy_local[:, :, 0] = gt_future[None, :, 0, 0:2]
            pred_yaw_local[:, :, 0] = gt_future[None, :, 0, 6]
            pred_valid_probability[:, :, 0] = gt_future[None, :, 0, 5]
            if not np.isfinite(pred_xy_local).all() or not np.isfinite(pred_yaw_local).all():
                raise ValueError(f"Non-finite rollout for {row['scenario_id']} focus={row['focus_track_id']}")

            origin_xy = batch_full["ego_origin_xy"][0].detach().float().cpu().numpy()
            frame_heading = float(batch_full["ego_heading"][0].detach().float().item())
            pred_xy_world = local_to_world_xy(pred_xy_local, origin_xy, frame_heading)
            pred_yaw_world = wrap_angle_np(pred_yaw_local + np.float32(frame_heading))
            diagnostics = proxy_metrics(pred_xy_local, gt_future, current_valid)
            for key, value in diagnostics.items():
                if isinstance(value, float):
                    proxy_totals[key] = proxy_totals.get(key, 0.0) + value

            output_path = rollout_dir / f"{make_output_stem(view_index - 1, row)}.npz"
            np.savez_compressed(
                output_path,
                center_x=pred_xy_world[..., 0],
                center_y=pred_xy_world[..., 1],
                heading=pred_yaw_world,
                valid_probability=pred_valid_probability,
                agent_ids=batch_full["agent_ids"][0].detach().cpu().numpy().astype(np.int64),
                current_valid=current_valid,
                focus_track_id=np.asarray(int(row["focus_track_id"]), dtype=np.int64),
                partner_track_id=np.asarray(int(row["partner_track_id"]), dtype=np.int64),
                scenario_id=np.asarray(row["scenario_id"]),
                scenario_pb_path=np.asarray(row["scenario_pb_path"]),
                source_npz_path=np.asarray(row["npz_path"]),
                oracle_focus=np.asarray(True),
                original_context_indices=np.arange(1, 11, dtype=np.int64),
                original_future_indices=np.arange(11, 91, dtype=np.int64),
            )
            view_seconds = time.perf_counter() - view_start
            view_record = {
                "view_index": view_index - 1,
                "scenario_id": row["scenario_id"],
                "focus_track_id": int(row["focus_track_id"]),
                "partner_track_id": int(row["partner_track_id"]),
                "num_selected_slots": int(batch_full["agent_mask"][0].sum().item()),
                "num_selected_current_valid": int(current_valid.sum()),
                "rollout_path": str(output_path),
                "view_seconds": view_seconds,
                "proxy_metrics": diagnostics,
            }
            view_records.append(view_record)
            manifest_handle.write(json.dumps(view_record, sort_keys=True) + "\n")
            manifest_handle.flush()

            elapsed = time.perf_counter() - generation_start
            per_view = elapsed / view_index
            eta_minutes = per_view * (len(rows) - view_index) / 60.0
            print(
                f"rollout_progress={view_index}/{len(rows)} view_seconds={view_seconds:.2f} "
                f"mean_seconds_per_view={per_view:.2f} remaining_minutes={eta_minutes:.1f} "
                f"scenario={row['scenario_id']} focus={row['focus_track_id']}",
                flush=True,
            )

    generation_seconds = time.perf_counter() - generation_start
    per_view_seconds = generation_seconds / len(rows)
    projected_generation_seconds = setup_seconds + per_view_seconds * int(args.total_views_for_eta)
    proxy_mean = {
        key: value / len(rows)
        for key, value in proxy_totals.items()
    }
    summary = {
        "protocol": "oracle_focus_local_32slot",
        "official_wosac_submission_compliant": False,
        "missing_unselected_agents_filled": False,
        "interactions_scope": "selected_current_valid_agents",
        "focus_trajectory": "logged_oracle",
        "manifest": str(manifest_path),
        "selected_views": len(rows),
        "total_views_for_eta": int(args.total_views_for_eta),
        "num_rollouts_per_view": int(args.num_rollouts),
        "rollout_batch_size": int(args.rollout_batch_size),
        "solver_steps_per_frame": int(schedule["K"]),
        "eval_d": float(schedule["d"]),
        "original_context_indices_zero_based": list(range(1, 11)),
        "original_future_indices_zero_based": list(range(11, 91)),
        "setup_seconds": setup_seconds,
        "generation_seconds": generation_seconds,
        "seconds_per_view": per_view_seconds,
        "projected_total_seconds": projected_generation_seconds,
        "projected_total_hours": projected_generation_seconds / 3600.0,
        "proxy_metrics_mean": proxy_mean,
        "checkpoint": str(Path(args.eval_ckpt).resolve()),
        "checkpoint_step": int(ckpt.get("step", -1)),
        "tokenizer_checkpoint": str(Path(args.tokenizer_ckpt).resolve()),
        "tokenizer_encode_decode_chunk_window": tokenizer_chunk_window,
        "tokenizer_encode_decode_chunk_stride": tokenizer_chunk_stride,
        "tokenizer_encode_decode_chunk_ranges_zero_based_end_exclusive": [
            [start, end] for start, end in tokenizer_chunk_ranges
        ],
        "tokenizer_max_chunk_timesteps_enforced": 32,
        "rollout_manifest": str(manifest_out),
        "cuda_max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    output_json = Path(args.output_json or (output_dir / "generation_summary.json"))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"generation_complete views={len(rows)} generation_seconds={generation_seconds:.2f} "
        f"seconds_per_view={per_view_seconds:.2f} "
        f"projected_{args.total_views_for_eta}_hours={projected_generation_seconds / 3600.0:.2f} "
        f"max_cuda_memory_gib={summary['cuda_max_memory_allocated_gib']:.2f}",
        flush=True,
    )
    print(f"wrote_generation_summary={output_json}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = base_eval.add_eval_args(wm.build_argparser())
    parser.description = "Generate 32 local-WOSAC oracle-focus rollouts per manifest view."
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--scenario_id", type=str, default=None)
    parser.add_argument("--focus_track_id", type=int, default=None)
    parser.add_argument("--max_views", type=int, default=50)
    parser.add_argument("--num_rollouts", type=int, default=32)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--total_views_for_eta", type=int, default=5000)
    parser.add_argument("--preflight_only", action="store_true")
    return parser


if __name__ == "__main__":
    generate(build_parser().parse_args())
