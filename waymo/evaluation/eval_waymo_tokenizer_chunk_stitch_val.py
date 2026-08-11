"""Paired tokenizer-only reconstruction ablation for chunk stitching.

This evaluates GT -> tokenizer encode -> tokenizer decode without a world model.
Every variant sees the same samples in the same order.  Tokenizer calls are
bounded by ``tokenizer_chunk_window`` (32 for the intended protocol).
"""

from __future__ import annotations

import argparse
import copy
import csv
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

from waymo.core.vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo.evaluation.eval_waymo_motion_latent_shared_rollout_horizons import (  # noqa: E402
    load_or_create_subset_manifest,
)
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


VARIANTS = (
    ("stride30_keep_first", 30, "keep_first"),
    ("stride16_center", 16, "center_select"),
    ("stride16_keep_first", 16, "keep_first"),
)

KEY_METRICS = (
    "agent_xy_mae_m",
    "agent_fde_mae_m",
    "focus_agent_xy_mae_m",
    "focus_agent_fde_m",
    "agent_delta_xy_mae_m",
    "agent_kinematic_xy_mae_m",
    "agent_speed_yaw_kinematic_mae_m",
    "agent_speed_mae_mps",
    "agent_vxvy_mae_mps",
    "agent_yaw_mae_deg",
    "agent_valid_acc",
    "light_state_acc",
    "light_valid_acc",
    "loss_total",
)


def _selected_segments(total_steps: int, window: int, stride: int, stitch: str) -> list[list[int]]:
    ranges = wm.tokenizer_time_chunk_ranges(total_steps, window, stride)
    if stitch == "center_select":
        return [
            [part_index, global_start, global_end, local_start, local_end]
            for part_index, global_start, global_end, local_start, local_end in wm.tokenizer_center_select_segments(
                ranges, total_steps
            )
        ]
    if stitch != "keep_first":
        raise ValueError(f"Unknown stitch mode: {stitch}")
    kept_until = 0
    segments: list[list[int]] = []
    for part_index, (start, end) in enumerate(ranges):
        discard = max(0, kept_until - start)
        if discard < end - start:
            segments.append([part_index, start + discard, end, discard, end - start])
            kept_until = end
    if kept_until != total_steps:
        raise ValueError(f"Chunks cover through {kept_until}, expected {total_steps}")
    return segments


def _metric_windows(total_steps: int, score_start: int, horizons: list[int]) -> dict[str, tuple[int, int]]:
    windows = {"full91": (0, total_steps)}
    for horizon in sorted(set(horizons)):
        end = score_start + horizon
        if end > total_steps:
            raise ValueError(
                f"score_start={score_start} + horizon={horizon} exceeds total_steps={total_steps}"
            )
        windows[f"future_h{horizon}"] = (score_start, end)
    return windows


def _add_continuity(
    decoded: Any,
    batch: dict[str, Any],
    args: argparse.Namespace,
    sums: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
) -> None:
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
        sums[name][1:] += (value * transition_valid).sum(dim=(0, 2)).detach().double().cpu()
        counts[name][1:] += valid_count

    acceleration_valid = transition_valid[:, 1:] & transition_valid[:, :-1]
    second_difference = torch.linalg.vector_norm(
        pred_xy[:, 2:] - 2.0 * pred_xy[:, 1:-1] + pred_xy[:, :-2], dim=-1
    )
    sums["second_difference_m"][2:] += (
        second_difference * acceleration_valid
    ).sum(dim=(0, 2)).detach().double().cpu()
    counts["second_difference_m"][2:] += acceleration_valid.sum(dim=(0, 2)).detach().double().cpu()


def _mean_curve(total: torch.Tensor, count: torch.Tensor) -> list[float | None]:
    return [None if n <= 0 else float(value / n) for value, n in zip(total.tolist(), count.tolist())]


def _normal_ci(values: list[float]) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [mean, mean]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half_width = 1.96 * math.sqrt(variance / len(values))
    return [mean - half_width, mean + half_width]


def _paired_comparisons(
    per_sample: list[dict[str, Any]],
    aggregate: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    baseline_name = VARIANTS[0][0]
    output: dict[str, Any] = {}
    for variant_name, _, _ in VARIANTS[1:]:
        output[variant_name] = {}
        for window_name in aggregate[baseline_name]:
            output[variant_name][window_name] = {}
            common_metrics = sorted(
                set(aggregate[baseline_name][window_name]) & set(aggregate[variant_name][window_name])
            )
            for metric in common_metrics:
                base = [row["variants"][baseline_name][window_name][metric] for row in per_sample]
                value = [row["variants"][variant_name][window_name][metric] for row in per_sample]
                deltas = [right - left for left, right in zip(base, value)]
                baseline_mean = aggregate[baseline_name][window_name][metric]
                variant_mean = aggregate[variant_name][window_name][metric]
                higher_is_better = metric.endswith("_acc")
                if abs(baseline_mean) > 1e-12:
                    sign = 1.0 if higher_is_better else -1.0
                    relative_improvement = sign * (variant_mean - baseline_mean) / abs(baseline_mean) * 100.0
                else:
                    relative_improvement = 0.0
                wins = [right > left if higher_is_better else right < left for left, right in zip(base, value)]
                output[variant_name][window_name][metric] = {
                    "baseline_mean": baseline_mean,
                    "variant_mean": variant_mean,
                    "delta_variant_minus_baseline": sum(deltas) / len(deltas),
                    "delta_95pct_normal_ci": _normal_ci(deltas),
                    "paired_win_rate": sum(wins) / len(wins),
                    "relative_improvement_percent": relative_improvement,
                    "higher_is_better": higher_is_better,
                    "num_samples": len(deltas),
                }
    return output


def _write_csv(path: Path, paired: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "window",
                "metric",
                "baseline_mean",
                "variant_mean",
                "delta_variant_minus_baseline",
                "ci_low",
                "ci_high",
                "relative_improvement_percent",
                "paired_win_rate",
                "num_samples",
            ]
        )
        for variant_name, windows in paired.items():
            for window_name, metrics in windows.items():
                for metric, values in metrics.items():
                    writer.writerow(
                        [
                            variant_name,
                            window_name,
                            metric,
                            values["baseline_mean"],
                            values["variant_mean"],
                            values["delta_variant_minus_baseline"],
                            values["delta_95pct_normal_ci"][0],
                            values["delta_95pct_normal_ci"][1],
                            values["relative_improvement_percent"],
                            values["paired_win_rate"],
                            values["num_samples"],
                        ]
                    )


def _write_key_summary(path: Path, aggregate: dict[str, Any], paired: dict[str, Any]) -> None:
    lines = [
        "Tokenizer-only GT -> encode -> decode paired reconstruction ablation",
        "Positive relative improvement means better; lower is better except *_acc.",
        "",
    ]
    for window_name in ("full91", "future_h90"):
        lines.append(f"[{window_name}]")
        for metric in KEY_METRICS:
            if metric not in aggregate[VARIANTS[0][0]][window_name]:
                continue
            baseline = aggregate[VARIANTS[0][0]][window_name][metric]
            lines.append(f"{metric}: baseline={baseline:.9g}")
            for variant_name, _, _ in VARIANTS[1:]:
                values = paired[variant_name][window_name][metric]
                low, high = values["delta_95pct_normal_ci"]
                lines.append(
                    f"  {variant_name}={values['variant_mean']:.9g} "
                    f"improvement={values['relative_improvement_percent']:+.3f}% "
                    f"win_rate={values['paired_win_rate']:.3f} delta_ci=[{low:.9g}, {high:.9g}]"
                )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if int(args.eval_batch_size) != 1:
        raise ValueError("Paired per-sample tokenizer evaluation requires --eval_batch_size 1")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(args.seed)

    dataset = wm.WaymoVectorDataset(args.val_data_dir)
    subset_indices, subset_payload = load_or_create_subset_manifest(
        dataset,
        path=Path(args.subset_manifest),
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
    )
    loader = DataLoader(
        Subset(dataset, subset_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=wm.worker_init_fn,
        collate_fn=wm._collate,
    )

    tokenizer, tokenizer_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    tokenizer.eval()
    # Reproduce the tokenizer checkpoint's own reconstruction objective and
    # position parameterization while retaining the world-model metric names.
    tokenizer_metric_args = (
        "agent_xy_weight",
        "agent_xy_loss",
        "agent_xy_parameterization",
        "agent_vel_weight",
        "agent_yaw_weight",
        "agent_valid_weight",
        "light_state_weight",
        "light_valid_weight",
        "agent_delta_xy_weight",
        "agent_fde_xy_weight",
        "agent_kinematic_xy_weight",
        "agent_speed_yaw_kinematic_weight",
        "kinematic_dt",
        "focus_agent_weight",
    )
    for name in tokenizer_metric_args:
        if name in tokenizer_args:
            setattr(args, name, tokenizer_args[name])

    total_steps = int(args.eval_seq_len)
    windows = _metric_windows(total_steps, int(args.score_start), list(args.score_horizons))
    variant_args: dict[str, argparse.Namespace] = {}
    variant_configs: dict[str, Any] = {}
    for variant_name, stride, stitch in VARIANTS:
        current = copy.copy(args)
        current.tokenizer_encode_chunk_stride = stride
        current.tokenizer_decode_chunk_stride = stride
        current.tokenizer_encode_stitch_mode = stitch
        current.tokenizer_decode_stitch_mode = stitch
        variant_args[variant_name] = current
        ranges = wm.tokenizer_time_chunk_ranges(total_steps, int(args.tokenizer_chunk_window), stride)
        segments = _selected_segments(total_steps, int(args.tokenizer_chunk_window), stride, stitch)
        variant_configs[variant_name] = {
            "encode_stride": stride,
            "decode_stride": stride,
            "encode_stitch": stitch,
            "decode_stitch": stitch,
            "chunk_ranges": [list(item) for item in ranges],
            "selected_segments": segments,
            "seam_endpoint_frames": [segment[1] for segment in segments[1:]],
        }

    totals: dict[str, dict[str, dict[str, float]]] = {
        name: {window_name: {} for window_name in windows} for name, _, _ in VARIANTS
    }
    per_sample: list[dict[str, Any]] = []
    continuity_sums = {
        name: {
            key: torch.zeros(total_steps, dtype=torch.float64)
            for key in ("pred_step_m", "delta_error_m", "second_difference_m")
        }
        for name, _, _ in VARIANTS
    }
    continuity_counts = {
        name: {key: torch.zeros(total_steps, dtype=torch.float64) for key in continuity_sums[name]}
        for name, _, _ in VARIANTS
    }

    print(
        f"tokenizer_ckpt={args.tokenizer_ckpt}\n"
        f"val_size={len(dataset)} subset_size={len(subset_indices)} manifest={args.subset_manifest}\n"
        f"protocol=tokenizer_only_gt_encode_decode max_chunk={args.tokenizer_chunk_window} "
        f"eval_seq_len={total_steps} score_start={args.score_start}",
        flush=True,
    )
    for variant_name, config in variant_configs.items():
        print(f"variant={variant_name} config={json.dumps(config, separators=(',', ':'))}", flush=True)

    evaluated = 0
    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = wm.slice_time_window(wm.move_batch(raw_batch, device), total_steps, random_start=False)
        sample_row: dict[str, Any] = {"sample_order": batch_index - 1, "variants": {}}
        for variant_name, _, _ in VARIANTS:
            current = variant_args[variant_name]
            z, _, _ = wm.encode_batch_dynamics_inputs_for_world_model(tokenizer, batch, current, return_map=False)
            decoded = wm.decode_batch_z_for_world_model(tokenizer, z, batch, current)
            _add_continuity(
                decoded,
                batch,
                current,
                continuity_sums[variant_name],
                continuity_counts[variant_name],
            )
            sample_row["variants"][variant_name] = {}
            for window_name, (start, end) in windows.items():
                decoded_window = wm.slice_decoder_output(decoded, start, end)
                batch_window = wm.slice_future_batch(batch, start, end)
                values = wm.tensor_metrics(wm.reconstruction_metrics(tokenizer, decoded_window, batch_window, current))
                sample_row["variants"][variant_name][window_name] = values
                for metric, value in values.items():
                    totals[variant_name][window_name][metric] = (
                        totals[variant_name][window_name].get(metric, 0.0) + value
                    )
        per_sample.append(sample_row)
        evaluated += 1
        if batch_index == 1 or batch_index % int(args.progress_every) == 0 or batch_index == len(loader):
            print(f"tokenizer reconstruction progress {batch_index}/{len(loader)}", flush=True)
        if args.eval_max_batches > 0 and batch_index >= int(args.eval_max_batches):
            break

    aggregate = {
        variant_name: {
            window_name: {metric: total / evaluated for metric, total in metrics.items()}
            for window_name, metrics in window_totals.items()
        }
        for variant_name, window_totals in totals.items()
    }
    continuity = {
        variant_name: {
            f"mean_{key}_by_endpoint_frame": _mean_curve(
                continuity_sums[variant_name][key], continuity_counts[variant_name][key]
            )
            for key in continuity_sums[variant_name]
        }
        for variant_name, _, _ in VARIANTS
    }
    paired = _paired_comparisons(per_sample, aggregate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol": "paired_fixed_val128_tokenizer_only_gt_encode_decode",
        "tokenizer_ckpt": str(Path(args.tokenizer_ckpt).resolve()),
        "val_size": len(dataset),
        "subset_size": evaluated,
        "subset_seed": int(args.subset_seed),
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "subset_selection": subset_payload["selection"],
        "eval_seq_len": total_steps,
        "score_start": int(args.score_start),
        "score_windows": {name: list(bounds) for name, bounds in windows.items()},
        "tokenizer_chunk_window": int(args.tokenizer_chunk_window),
        "tokenizer_metric_args": {name: getattr(args, name) for name in tokenizer_metric_args},
        "variants": variant_configs,
        "aggregate_metrics": aggregate,
        "continuity": continuity,
        "paired_metric_comparisons": paired,
        "per_sample_metrics": per_sample,
    }
    json_path = output_dir / "tokenizer_only_reconstruction_val128.json"
    csv_path = output_dir / "paired_metric_comparison.csv"
    summary_path = output_dir / "key_metrics.txt"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _write_csv(csv_path, paired)
    _write_key_summary(summary_path, aggregate, paired)

    for window_name in ("full91", "future_h90"):
        print(f"[{window_name}]", flush=True)
        for variant_name, _, _ in VARIANTS:
            selected = {
                metric: aggregate[variant_name][window_name][metric]
                for metric in KEY_METRICS
                if metric in aggregate[variant_name][window_name]
            }
            print(f"{variant_name}: {wm.format_metrics(selected)}", flush=True)
    print(f"wrote_json={json_path}\nwrote_csv={csv_path}\nwrote_summary={summary_path}", flush=True)


if __name__ == "__main__":
    parser = wm.build_argparser()
    parser.description = "Paired tokenizer-only reconstruction comparison for three chunk stitching strategies."
    parser.add_argument("--subset_manifest", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--score_start", type=int, default=1)
    parser.add_argument("--score_horizons", type=int, nargs="*", default=[10, 30, 50, 80, 90])
    parser.add_argument("--progress_every", type=int, default=16)
    parser.add_argument("--output_dir", type=str, required=True)
    main(parser.parse_args())
