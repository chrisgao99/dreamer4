"""Evaluate MotionLatent V1 horizon prefixes from one shared direct-D1 rollout.

Each selected validation sample is rolled out once from ctx=1 through H90.
Metrics for shorter horizons are computed from prefixes of that same sampled
trajectory, so checkpoint and horizon comparisons use identical samples and
identical per-frame sampling semantics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.evaluation import eval_waymo_world_model_horizons as base_eval  # noqa: E402
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item() if array.size == 1 else value)


def _sample_record(dataset: wm.WaymoVectorDataset, sample_order: int, dataset_index: int) -> dict[str, Any]:
    path = dataset.paths[dataset_index]
    record: dict[str, Any] = {
        "sample_order": int(sample_order),
        "dataset_index": int(dataset_index),
        "path": str(path),
        "filename": Path(path).name,
    }
    with np.load(path, allow_pickle=False) as data:
        for key in ("scenario_id", "focus_src_index", "focus_track_id"):
            if key in data:
                record[key] = _scalar_string(data[key])
    return record


def load_or_create_subset_manifest(
    dataset: wm.WaymoVectorDataset,
    *,
    path: Path,
    subset_size: int,
    subset_seed: int,
) -> tuple[list[int], dict[str, Any]]:
    if path.is_file():
        payload = json.loads(path.read_text())
        if int(payload["val_size"]) != len(dataset):
            raise ValueError(
                f"Subset manifest val_size={payload['val_size']} does not match current val size={len(dataset)}"
            )
        if int(payload["subset_size"]) != int(subset_size):
            raise ValueError(
                f"Subset manifest size={payload['subset_size']} does not match requested size={subset_size}"
            )
        if int(payload["subset_seed"]) != int(subset_seed):
            raise ValueError(
                f"Subset manifest seed={payload['subset_seed']} does not match requested seed={subset_seed}"
            )
        indices = [int(record["dataset_index"]) for record in payload["samples"]]
        if len(indices) != subset_size or len(set(indices)) != subset_size:
            raise ValueError("Subset manifest must contain the requested number of unique dataset indices")
        if any(index < 0 or index >= len(dataset) for index in indices):
            raise ValueError("Subset manifest contains an out-of-range dataset index")
        return indices, payload

    if subset_size <= 0 or subset_size > len(dataset):
        raise ValueError(f"subset_size must be in [1, {len(dataset)}], got {subset_size}")
    indices = random.Random(int(subset_seed)).sample(range(len(dataset)), int(subset_size))
    payload = {
        "dataset_root": str(Path(dataset.paths[0]).parent),
        "selection": "random_without_replacement",
        "subset_seed": int(subset_seed),
        "subset_size": int(subset_size),
        "val_size": int(len(dataset)),
        "samples": [
            _sample_record(dataset, sample_order, dataset_index)
            for sample_order, dataset_index in enumerate(indices)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return indices, payload


@torch.no_grad()
def evaluate_shared_rollout(
    dyn: torch.nn.Module,
    tokenizer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt: dict[str, Any],
) -> dict[str, dict[str, float]]:
    dyn.eval()
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    max_horizon = max(horizons)
    model_args = ckpt.get("args", {})
    max_context = int(model_args.get("max_context", args.max_rollout_window))
    k_max = int(model_args.get("k_max", args.k_max))
    kinematic_dt = float(model_args.get("kinematic_dt", args.kinematic_dt))
    totals: dict[int, dict[str, float]] = {horizon: {} for horizon in horizons}
    counts = {horizon: 0 for horizon in horizons}

    # Reset after model construction/loading so every checkpoint sees the same
    # random noise sequence for the same manifest-ordered validation samples.
    wm.seed_everything(args.seed)

    for batch_index, batch in enumerate(loader, start=1):
        batch = wm.slice_time_window(wm.move_batch(batch, device), args.eval_seq_len, random_start=False)
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        if actions is None or act_mask is None:
            raise ValueError("MotionLatent V1 evaluation requires --use_ego_actions")
        q_gt = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            tokenizer,
            batch,
            args,
            return_map=True,
        )
        required = int(args.eval_ctx) + max_horizon
        if z_gt.shape[1] < required:
            raise ValueError(f"Need at least {required} frames for shared rollout, got {z_gt.shape[1]}")
        z_gt_packed = wm.pack_bottleneck_to_spatial(z_gt, n_spatial=args.n_spatial, k=args.packing_factor)

        # Exactly one direct-D1 rollout from ctx=1 through the largest horizon.
        z_pred_packed = base_eval.sample_motion_latent_v1_sequence(
            wm.unwrap_model(dyn),
            z_gt_packed=z_gt_packed,
            q_gt=q_gt,
            actions=actions,
            act_mask=act_mask,
            action_slots=action_slots,
            agent_mask=batch["agent_mask"],
            map_tokens=map_tokens,
            map_mask=map_mask,
            ctx_length=args.eval_ctx,
            horizon=max_horizon,
            max_context=max_context,
            k_max=k_max,
            kinematic_dt=kinematic_dt,
        )
        z_pred = wm.unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
        z_decode = z_pred
        if z_pred.shape[1] < z_gt.shape[1]:
            z_decode = torch.cat([z_pred, z_gt[:, z_pred.shape[1] :]], dim=1)
        decoded = wm.decode_batch_z_for_world_model(tokenizer, z_decode, batch, args)

        score_start = int(args.eval_ctx)
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
            for name, value in values.items():
                totals[horizon][name] = totals[horizon].get(name, 0.0) + value
            counts[horizon] += 1

        if batch_index == 1 or batch_index % 16 == 0 or batch_index == len(loader):
            print(f"shared rollout progress {batch_index}/{len(loader)}", flush=True)

    results: dict[str, dict[str, float]] = {}
    for horizon in horizons:
        count = max(1, counts[horizon])
        results[f"h{horizon}"] = {
            name: float(total / count)
            for name, total in totals[horizon].items()
        }
    return results


def main(args: argparse.Namespace) -> None:
    if args.eval_batch_size != 1:
        raise ValueError("This recorded 128-batch protocol requires --eval_batch_size 1")
    if float(args.eval_d) != 1.0:
        raise ValueError("MotionLatent shared-rollout evaluation must use --eval_d 1.0 (direct D1)")
    if int(args.eval_ctx) != 1:
        raise ValueError("This protocol requires --eval_ctx 1")
    if sorted(set(int(horizon) for horizon in args.horizons)) != [10, 30, 50, 80, 90]:
        raise ValueError("This protocol requires --horizons 10 30 50 80 90")
    if int(args.eval_seq_len) != 91:
        raise ValueError("This protocol requires --eval_seq_len 91 (ctx=1 plus H90)")

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
    n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
    d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor != 0:
        raise ValueError(f"n_latents={n_latents} must be divisible by packing_factor={args.packing_factor}")
    args.n_spatial = n_latents // args.packing_factor
    args.d_spatial = d_bottleneck * args.packing_factor

    ckpt = torch.load(args.eval_ckpt, map_location="cpu")
    if ckpt.get("format") != base_eval.MOTION_LATENT_V1_FORMAT:
        raise ValueError(f"Expected MotionLatent V1 checkpoint, got format={ckpt.get('format', 'legacy')}")
    dyn = base_eval.build_motion_latent_v1_dynamics(
        ckpt,
        tokenizer,
        tok_args,
        d_bottleneck,
        args.n_spatial,
        args.d_spatial,
        device,
    )
    base_eval.load_dynamics_state(dyn, args.eval_ckpt, ckpt=ckpt)
    dyn.eval()

    print(f"eval_ckpt={args.eval_ckpt}", flush=True)
    print(f"ckpt_step={int(ckpt.get('step', -1))} ckpt_epoch={int(ckpt.get('epoch', -1))}", flush=True)
    print(
        f"rollout_mode=motion_latent_v1_direct_d1 "
        f"shared_rollout_horizon={max(int(horizon) for horizon in args.horizons)} samples_per_frame=1",
        flush=True,
    )
    print(
        f"val_size={len(eval_ds)} subset_size={len(subset_indices)} subset_seed={args.subset_seed} "
        f"subset_manifest={args.subset_manifest}",
        flush=True,
    )
    print(
        f"tokenizer_chunk_window={args.tokenizer_chunk_window} "
        f"tokenizer_chunk_stride={args.tokenizer_chunk_stride}",
        flush=True,
    )
    print(f"eval_ctx=1 horizons={' '.join(str(h) for h in args.horizons)}", flush=True)

    results = evaluate_shared_rollout(dyn, tokenizer, eval_loader, device, args, ckpt)
    for horizon in sorted(results, key=lambda name: int(name[1:])):
        print(f"eval horizon={horizon[1:]} {wm.format_metrics(results[horizon])}", flush=True)

    output = {
        "eval_ckpt": args.eval_ckpt,
        "checkpoint_format": ckpt.get("format"),
        "ckpt_step": int(ckpt.get("step", -1)),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "rollout_mode": "motion_latent_v1_direct_d1",
        "samples_per_predicted_frame": 1,
        "shared_rollout_horizon": max(int(horizon) for horizon in args.horizons),
        "metrics_are_prefixes_of_same_rollout": True,
        "eval_ctx": 1,
        "horizons": sorted(set(int(horizon) for horizon in args.horizons)),
        "val_size": len(eval_ds),
        "subset_size": len(subset_indices),
        "subset_seed": int(args.subset_seed),
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "subset_selection": subset_payload["selection"],
        "tokenizer_chunk_window": int(args.tokenizer_chunk_window),
        "tokenizer_chunk_stride": int(args.tokenizer_chunk_stride),
        "metrics": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote metrics: {output_path}", flush=True)


if __name__ == "__main__":
    parser = base_eval.add_eval_args(wm.build_argparser())
    parser.description = "Evaluate MotionLatent horizon prefixes from one shared direct-D1 H90 rollout."
    parser.add_argument("--subset_manifest", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--subset_seed", type=int, default=0)
    main(parser.parse_args())
