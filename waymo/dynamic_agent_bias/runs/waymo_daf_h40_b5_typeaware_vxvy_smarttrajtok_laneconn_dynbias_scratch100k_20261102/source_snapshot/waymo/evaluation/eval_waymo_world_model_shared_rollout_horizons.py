"""Evaluate legacy Waymo World Model prefixes from one shared shortcut rollout."""

from __future__ import annotations

import argparse
import json
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
from waymo.core.vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo.evaluation.eval_waymo_motion_latent_shared_rollout_horizons import (  # noqa: E402
    load_or_create_subset_manifest,
)
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


@torch.no_grad()
def evaluate_shared_rollout(
    dyn: torch.nn.Module,
    tokenizer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    dyn.eval()
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    max_horizon = max(horizons)
    schedule = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)

    totals: dict[int, dict[str, float]] = {horizon: {} for horizon in horizons}
    counts = {horizon: 0 for horizon in horizons}
    per_sample_metrics: list[dict[str, Any]] = []
    continuity_sums = {
        "pred_step_m": torch.zeros(args.eval_seq_len, dtype=torch.float64),
        "delta_error_m": torch.zeros(args.eval_seq_len, dtype=torch.float64),
        "second_difference_m": torch.zeros(args.eval_seq_len, dtype=torch.float64),
    }
    continuity_counts = {
        name: torch.zeros(args.eval_seq_len, dtype=torch.float64)
        for name in continuity_sums
    }

    # Model construction consumes RNG. Reset here so both checkpoints receive
    # identical noise in the identical manifest sample order.
    wm.seed_everything(args.seed)

    for batch_index, batch in enumerate(loader, start=1):
        batch = wm.slice_time_window(wm.move_batch(batch, device), args.eval_seq_len, random_start=False)
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            tokenizer,
            batch,
            args,
            return_map=args.dynamics_attend_map,
        )
        required = int(args.eval_ctx) + max_horizon
        if z_gt.shape[1] < required:
            raise ValueError(f"Need at least {required} frames for shared rollout, got {z_gt.shape[1]}")
        z_gt_packed = wm.pack_bottleneck_to_spatial(
            z_gt,
            n_spatial=args.n_spatial,
            k=args.packing_factor,
        )

        # Exactly one autoregressive ctx=1 -> H90 rollout. Shorter-horizon
        # metrics are prefixes of this same sampled trajectory.
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

        gt_agents = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
        anchor_xy = gt_agents[:, 0, :, 0:2] if args.agent_xy_parameterization == "delta" else None
        pred_xy = decoder_agent_xy(
            decoded,
            agent_xy_loss=args.agent_xy_loss,
            agent_xy_parameterization=args.agent_xy_parameterization,
            anchor_xy=anchor_xy,
        )
        gt_xy = gt_agents[..., 0:2]
        gt_valid = (gt_agents[..., 5] > 0.5) & batch["agent_mask"][:, None, :]
        transition_valid = gt_valid[:, 1:] & gt_valid[:, :-1]
        pred_delta = pred_xy[:, 1:] - pred_xy[:, :-1]
        gt_delta = gt_xy[:, 1:] - gt_xy[:, :-1]
        pred_step = torch.linalg.vector_norm(pred_delta, dim=-1)
        delta_error = torch.linalg.vector_norm(pred_delta - gt_delta, dim=-1)
        valid_count = transition_valid.sum(dim=(0, 2)).detach().double().cpu()
        for name, value in (("pred_step_m", pred_step), ("delta_error_m", delta_error)):
            continuity_sums[name][1:] += (
                value * transition_valid
            ).sum(dim=(0, 2)).detach().double().cpu()
            continuity_counts[name][1:] += valid_count

        acceleration_valid = transition_valid[:, 1:] & transition_valid[:, :-1]
        second_difference = torch.linalg.vector_norm(
            pred_xy[:, 2:] - 2.0 * pred_xy[:, 1:-1] + pred_xy[:, :-2],
            dim=-1,
        )
        continuity_sums["second_difference_m"][2:] += (
            second_difference * acceleration_valid
        ).sum(dim=(0, 2)).detach().double().cpu()
        continuity_counts["second_difference_m"][2:] += (
            acceleration_valid.sum(dim=(0, 2)).detach().double().cpu()
        )

        score_start = int(args.eval_ctx)
        sample_metric_row: dict[str, Any] = {"sample_order": batch_index - 1, "metrics": {}}
        for horizon in horizons:
            score_end = score_start + horizon
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
            values = wm.tensor_metrics(metrics)
            sample_metric_row["metrics"][f"h{horizon}"] = values
            for name, value in values.items():
                totals[horizon][name] = totals[horizon].get(name, 0.0) + value
            counts[horizon] += 1
        per_sample_metrics.append(sample_metric_row)

        if batch_index == 1 or batch_index % 16 == 0 or batch_index == len(loader):
            print(f"shared rollout progress {batch_index}/{len(loader)}", flush=True)
        if args.eval_max_batches > 0 and batch_index >= int(args.eval_max_batches):
            break

    horizon_results = {
        f"h{horizon}": {
            name: float(total / max(1, counts[horizon]))
            for name, total in totals[horizon].items()
        }
        for horizon in horizons
    }
    continuity = {}
    for name in continuity_sums:
        values = []
        for total, count in zip(continuity_sums[name].tolist(), continuity_counts[name].tolist()):
            values.append(None if count <= 0 else float(total / count))
        continuity[f"mean_{name}_by_endpoint_frame"] = values
    return {
        "metrics": horizon_results,
        "continuity": continuity,
        "per_sample_metrics": per_sample_metrics,
    }


def main(args: argparse.Namespace) -> None:
    if args.eval_batch_size != 1:
        raise ValueError("This recorded 128-batch protocol requires --eval_batch_size 1")
    if args.eval_schedule != "shortcut":
        raise ValueError("World Model shared-rollout evaluation requires --eval_schedule shortcut")
    if int(args.eval_ctx) < 1:
        raise ValueError("World Model shared-rollout evaluation requires --eval_ctx >= 1")
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError("World Model shared-rollout evaluation requires positive horizons")
    max_horizon = max(horizons)
    if int(args.eval_seq_len) < int(args.eval_ctx) + max_horizon:
        raise ValueError(
            "World Model shared-rollout evaluation requires "
            "eval_seq_len >= eval_ctx + max(horizons)"
        )

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
    rollout_mode = f"legacy_world_model_shortcut_k{solver_steps}"

    print(f"eval_ckpt={args.eval_ckpt}", flush=True)
    print(f"ckpt_step={int(ckpt.get('step', -1))} ckpt_epoch={int(ckpt.get('epoch', -1))}", flush=True)
    print(
        f"rollout_mode={rollout_mode} shared_rollout_horizon={max_horizon} "
        f"solver_steps_per_frame={solver_steps} eval_d={float(schedule['d']):g}",
        flush=True,
    )
    print(
        f"val_size={len(eval_ds)} subset_size={len(subset_indices)} subset_seed={args.subset_seed} "
        f"subset_manifest={args.subset_manifest}",
        flush=True,
    )
    encode_stride = args.tokenizer_encode_chunk_stride or args.tokenizer_chunk_stride
    decode_stride = args.tokenizer_decode_chunk_stride or args.tokenizer_chunk_stride
    print(
        f"tokenizer_chunk_window={args.tokenizer_chunk_window} "
        f"encode_stride={encode_stride} encode_stitch={args.tokenizer_encode_stitch_mode} "
        f"decode_stride={decode_stride} decode_stitch={args.tokenizer_decode_stitch_mode}",
        flush=True,
    )
    print(f"eval_ctx={args.eval_ctx} horizons={' '.join(str(h) for h in args.horizons)}", flush=True)

    evaluation = evaluate_shared_rollout(dyn, tokenizer, eval_loader, device, args)
    results = evaluation["metrics"]
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
        "shared_rollout_horizon": max_horizon,
        "metrics_are_prefixes_of_same_rollout": True,
        "eval_ctx": int(args.eval_ctx),
        "horizons": sorted(set(int(horizon) for horizon in args.horizons)),
        "val_size": len(eval_ds),
        "subset_size": len(subset_indices),
        "subset_seed": int(args.subset_seed),
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "subset_selection": subset_payload["selection"],
        "tokenizer_chunk_window": int(args.tokenizer_chunk_window),
        "tokenizer_chunk_stride": int(args.tokenizer_chunk_stride),
        "tokenizer_encode_chunk_stride": int(encode_stride),
        "tokenizer_decode_chunk_stride": int(decode_stride),
        "tokenizer_encode_stitch_mode": str(args.tokenizer_encode_stitch_mode),
        "tokenizer_decode_stitch_mode": str(args.tokenizer_decode_stitch_mode),
        "metrics": results,
        "continuity": evaluation["continuity"],
        "per_sample_metrics": evaluation["per_sample_metrics"],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote metrics: {output_path}", flush=True)


if __name__ == "__main__":
    parser = base_eval.add_eval_args(wm.build_argparser())
    parser.description = "Evaluate legacy World Model prefixes from one shared shortcut rollout."
    parser.add_argument("--subset_manifest", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--subset_seed", type=int, default=0)
    main(parser.parse_args())
