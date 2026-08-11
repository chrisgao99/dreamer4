"""Evaluate horizon prefixes after per-scene best-of-N rollout selection.

The intended protocol uses original Waymo frames 2..11 as ten ground-truth
context frames, rolls out original frames 12..91 six times, and selects the
candidate with the lowest all-agent FDE at H80 for each validation scene.
Metrics at H10/H30/H50/H80 are then computed from prefixes of that one selected
trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.evaluation import eval_waymo_world_model_horizons as base_eval  # noqa: E402
from waymo.evaluation.eval_waymo_motion_latent_shared_rollout_horizons import (  # noqa: E402
    load_or_create_subset_manifest,
)
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


SELECTION_METRIC = "agent_fde_mae_m"


def slice_time_range(batch: dict[str, Any], start: int, length: int) -> dict[str, Any]:
    """Take an explicit time range while preserving static scene fields."""
    total_steps = int(batch["lights"].shape[1])
    start = int(start)
    length = int(length)
    if start < 0:
        raise ValueError(f"sequence_start must be non-negative, got {start}")
    if length <= 0:
        raise ValueError(f"eval_seq_len must be positive, got {length}")
    end = start + length
    if end > total_steps:
        raise ValueError(
            f"Requested time range [{start}, {end}) exceeds the available {total_steps} frames"
        )
    return wm.slice_future_batch(batch, start, end)


def rollout_horizon_metrics(
    tokenizer: torch.nn.Module,
    decoded: Any,
    z_pred_packed: torch.Tensor,
    z_gt_packed: torch.Tensor,
    batch: dict[str, Any],
    args: argparse.Namespace,
    action_slots: torch.Tensor | None,
    horizon: int,
) -> dict[str, torch.Tensor]:
    score_start = int(args.eval_ctx)
    score_end = score_start + int(horizon)
    decoded_future = wm.slice_decoder_output(decoded, score_start, score_end)
    batch_future = wm.slice_future_batch(batch, score_start, score_end)
    future_weight = wm.build_agent_loss_weight_multiplier(
        batch_future,
        args,
        action_slots=action_slots,
    )
    metrics = wm.reconstruction_metrics(
        tokenizer,
        decoded_future,
        batch_future,
        args,
        agent_loss_weight_multiplier=future_weight,
    )
    metrics["latent_mse_future"] = (
        z_pred_packed[:, score_start:score_end].float()
        - z_gt_packed[:, score_start:score_end].float()
    ).pow(2).mean()
    return metrics


@torch.no_grad()
def evaluate_best_of_n_rollouts(
    dyn: torch.nn.Module,
    tokenizer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    sample_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    dyn.eval()
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    max_horizon = max(horizons)
    schedule = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    totals: dict[int, dict[str, float]] = {horizon: {} for horizon in horizons}
    counts = {horizon: 0 for horizon in horizons}
    selections: list[dict[str, Any]] = []

    # Model construction consumes RNG. Reset immediately before evaluation so
    # candidate sampling is deterministic for a fixed seed and manifest order.
    wm.seed_everything(args.seed)

    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = slice_time_range(
            wm.move_batch(raw_batch, device),
            args.sequence_start,
            args.eval_seq_len,
        )
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            tokenizer,
            batch,
            args,
            return_map=args.dynamics_attend_map,
        )
        required = int(args.eval_ctx) + max_horizon
        if z_gt.shape[1] < required:
            raise ValueError(f"Need at least {required} frames for rollout, got {z_gt.shape[1]}")
        z_gt_packed = wm.pack_bottleneck_to_spatial(
            z_gt,
            n_spatial=args.n_spatial,
            k=args.packing_factor,
        )

        best_fde = math.inf
        best_candidate_index = -1
        best_decoded = None
        best_z_pred_packed = None
        candidate_fdes: list[float] = []

        for candidate_index in range(int(args.num_rollouts)):
            z_pred_packed = wm.sample_autoregressive_packed_sequence(
                wm.unwrap_model(dyn),
                z_gt_packed=z_gt_packed,
                actions=actions,
                act_mask=act_mask,
                map_tokens=map_tokens,
                map_mask=map_mask,
                ctx_length=args.eval_ctx,
                horizon=max_horizon,
                k_max=args.k_max,
                sched=schedule,
                max_rollout_window=args.max_rollout_window,
            )
            z_pred = wm.unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
            z_decode = z_pred
            if z_pred.shape[1] < z_gt.shape[1]:
                z_decode = torch.cat([z_pred, z_gt[:, z_pred.shape[1] :]], dim=1)
            decoded = wm.decode_batch_z_for_world_model(tokenizer, z_decode, batch, args)
            selection_metrics = rollout_horizon_metrics(
                tokenizer,
                decoded,
                z_pred_packed,
                z_gt_packed,
                batch,
                args,
                action_slots,
                args.selection_horizon,
            )
            candidate_fde = float(selection_metrics[SELECTION_METRIC].detach().float().item())
            if not math.isfinite(candidate_fde):
                raise ValueError(
                    f"Non-finite {SELECTION_METRIC} for batch {batch_index}, candidate {candidate_index + 1}"
                )
            candidate_fdes.append(candidate_fde)
            if candidate_fde < best_fde:
                best_fde = candidate_fde
                best_candidate_index = candidate_index
                best_decoded = decoded
                best_z_pred_packed = z_pred_packed

        if best_decoded is None or best_z_pred_packed is None:
            raise RuntimeError(f"No rollout candidate selected for batch {batch_index}")

        for horizon in horizons:
            metrics = rollout_horizon_metrics(
                tokenizer,
                best_decoded,
                best_z_pred_packed,
                z_gt_packed,
                batch,
                args,
                action_slots,
                horizon,
            )
            values = wm.tensor_metrics(metrics)
            for name, value in values.items():
                totals[horizon][name] = totals[horizon].get(name, 0.0) + value
            counts[horizon] += 1

        sample_record = dict(sample_records[batch_index - 1])
        selections.append(
            {
                "sample_order": int(sample_record.get("sample_order", batch_index - 1)),
                "dataset_index": int(sample_record["dataset_index"]),
                "filename": sample_record.get("filename"),
                "scenario_id": sample_record.get("scenario_id"),
                "selected_rollout_index_zero_based": int(best_candidate_index),
                "selected_rollout_number_one_based": int(best_candidate_index + 1),
                "selected_agent_fde_mae_m_h80": float(best_fde),
                "candidate_agent_fde_mae_m_h80": candidate_fdes,
            }
        )

        if batch_index == 1 or batch_index % 8 == 0 or batch_index == len(loader):
            print(
                f"best-of-{args.num_rollouts} rollout progress {batch_index}/{len(loader)} "
                f"selected={best_candidate_index + 1} {SELECTION_METRIC}@h{args.selection_horizon}={best_fde:.4f}",
                flush=True,
            )

    results = {
        f"h{horizon}": {
            name: float(total / max(1, counts[horizon]))
            for name, total in totals[horizon].items()
        }
        for horizon in horizons
    }
    return results, selections


def main(args: argparse.Namespace) -> None:
    if args.eval_batch_size != 1:
        raise ValueError("Per-scene best-of-N selection requires --eval_batch_size 1")
    if args.eval_schedule != "shortcut":
        raise ValueError("World Model best-of-N evaluation requires --eval_schedule shortcut")
    if int(args.sequence_start) != 1:
        raise ValueError("This protocol requires --sequence_start 1 (drop original frame 1)")
    if int(args.eval_seq_len) != 90:
        raise ValueError("This protocol requires --eval_seq_len 90 (original frames 2..91)")
    if int(args.eval_ctx) != 10:
        raise ValueError("This protocol requires --eval_ctx 10 (original frames 2..11)")
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    if horizons != [10, 30, 50, 80]:
        raise ValueError("This protocol requires --horizons 10 30 50 80")
    if int(args.selection_horizon) != 80:
        raise ValueError("This protocol requires --selection_horizon 80")
    if int(args.num_rollouts) != 6:
        raise ValueError("This protocol requires --num_rollouts 6")
    if int(args.eval_ctx) + max(horizons) != int(args.eval_seq_len):
        raise ValueError("eval_ctx + max(horizons) must exactly equal eval_seq_len")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(args.seed)
    eval_ds = wm.WaymoVectorDataset(args.val_data_dir)
    subset_indices, subset_payload = load_or_create_subset_manifest(
        eval_ds,
        path=Path(args.subset_manifest),
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
    )
    eval_loader = DataLoader(
        Subset(eval_ds, subset_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=wm.worker_init_fn,
        collate_fn=wm._collate,
    )

    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    if isinstance(tokenizer, wm.FrozenWaymoFocusTokenizer):
        n_latents = tokenizer.n_latents
        d_bottleneck = tokenizer.d_bottleneck
    else:
        n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
        d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor != 0:
        raise ValueError(f"n_latents={n_latents} must be divisible by packing_factor={args.packing_factor}")
    args.n_spatial = n_latents // args.packing_factor
    args.d_spatial = d_bottleneck * args.packing_factor

    ckpt = torch.load(args.eval_ckpt, map_location="cpu")
    if ckpt.get("format") == base_eval.MOTION_LATENT_V1_FORMAT:
        raise ValueError("Expected a legacy World Model checkpoint, got MotionLatent V1")
    dyn = base_eval.build_dynamics(
        args,
        d_bottleneck,
        device,
        map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None,
    )
    base_eval.load_dynamics_state(dyn, args.eval_ckpt, ckpt=ckpt)
    dyn.eval()

    schedule = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    solver_steps = int(schedule["K"])
    rollout_mode = f"legacy_world_model_shortcut_k{solver_steps}_best_of_{args.num_rollouts}"

    print(f"eval_ckpt={args.eval_ckpt}", flush=True)
    print(f"ckpt_step={int(ckpt.get('step', -1))} ckpt_epoch={int(ckpt.get('epoch', -1))}", flush=True)
    print(
        f"rollout_mode={rollout_mode} shared_rollout_horizon={max(horizons)} "
        f"solver_steps_per_frame={solver_steps} eval_d={float(schedule['d']):g}",
        flush=True,
    )
    print(
        f"val_size={len(eval_ds)} subset_size={len(subset_indices)} subset_seed={args.subset_seed} "
        f"subset_manifest={args.subset_manifest}",
        flush=True,
    )
    print(
        f"original_frames=context=2..11 rollout=12..91 sequence_start_zero_based={args.sequence_start} "
        f"eval_seq_len={args.eval_seq_len} eval_ctx={args.eval_ctx}",
        flush=True,
    )
    print(
        f"num_rollouts={args.num_rollouts} selection_metric={SELECTION_METRIC} "
        f"selection_horizon={args.selection_horizon}",
        flush=True,
    )
    print(
        f"tokenizer_chunk_window={args.tokenizer_chunk_window} "
        f"tokenizer_chunk_stride={args.tokenizer_chunk_stride}",
        flush=True,
    )
    print(f"horizons={' '.join(str(horizon) for horizon in horizons)}", flush=True)

    results, selections = evaluate_best_of_n_rollouts(
        dyn,
        tokenizer,
        eval_loader,
        device,
        args,
        subset_payload["samples"],
    )
    for horizon in sorted(results, key=lambda name: int(name[1:])):
        print(f"eval horizon={horizon[1:]} {wm.format_metrics(results[horizon])}", flush=True)

    output = {
        "eval_ckpt": args.eval_ckpt,
        "checkpoint_format": ckpt.get("format", "legacy"),
        "ckpt_step": int(ckpt.get("step", -1)),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "rollout_mode": rollout_mode,
        "solver_steps_per_predicted_frame": solver_steps,
        "eval_schedule": str(args.eval_schedule),
        "eval_d": float(schedule["d"]),
        "num_rollouts_per_scene": int(args.num_rollouts),
        "selection_metric": SELECTION_METRIC,
        "selection_horizon": int(args.selection_horizon),
        "selection_scope": "per_scene",
        "shared_rollout_horizon": max(horizons),
        "metrics_are_prefixes_of_selected_rollout": True,
        "original_context_frames_one_based": list(range(2, 12)),
        "original_rollout_frames_one_based": [12, 91],
        "sequence_start_zero_based": int(args.sequence_start),
        "eval_seq_len": int(args.eval_seq_len),
        "eval_ctx": int(args.eval_ctx),
        "horizons": horizons,
        "val_size": len(eval_ds),
        "subset_size": len(subset_indices),
        "subset_seed": int(args.subset_seed),
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "subset_selection": subset_payload["selection"],
        "tokenizer_chunk_window": int(args.tokenizer_chunk_window),
        "tokenizer_chunk_stride": int(args.tokenizer_chunk_stride),
        "metrics": results,
        "per_scene_selections": selections,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote metrics: {output_path}", flush=True)


if __name__ == "__main__":
    parser = base_eval.add_eval_args(wm.build_argparser())
    parser.description = "Evaluate per-scene best-of-N legacy World Model rollout prefixes."
    parser.add_argument("--subset_manifest", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--sequence_start", type=int, default=1)
    parser.add_argument("--num_rollouts", type=int, default=6)
    parser.add_argument("--selection_horizon", type=int, default=80)
    main(parser.parse_args())
