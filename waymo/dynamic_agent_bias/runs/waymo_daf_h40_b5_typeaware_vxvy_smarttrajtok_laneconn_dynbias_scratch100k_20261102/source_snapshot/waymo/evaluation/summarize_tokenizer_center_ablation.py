#!/usr/bin/env python3
"""Summarize paired val128 tokenizer stitching ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


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
    "latent_mse_future",
)


def chunk_ranges(total: int, window: int, stride: int) -> list[tuple[int, int]]:
    if total <= window:
        return [(0, total)]
    starts = list(range(0, total - window + 1, stride))
    final_start = total - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, start + window) for start in starts]


def selected_segments(ranges: list[tuple[int, int]], total: int, mode: str) -> list[tuple[int, int, int]]:
    if mode == "keep_first":
        kept_until = 0
        segments = []
        for part_index, (start, end) in enumerate(ranges):
            global_start = max(start, kept_until)
            if global_start < end:
                segments.append((part_index, global_start, end))
                kept_until = end
        return segments
    if mode != "center_select":
        raise ValueError(mode)
    owners = []
    for timestep in range(total):
        candidates = []
        for index, (start, end) in enumerate(ranges):
            if start <= timestep < end:
                candidates.append((abs(timestep - (start + end - 1) / 2.0), -start, index))
        owners.append(min(candidates)[2])
    segments = []
    start = 0
    for timestep in range(1, total + 1):
        if timestep == total or owners[timestep] != owners[start]:
            segments.append((owners[start], start, timestep))
            start = timestep
    return segments


def continuity_summary(payload: dict[str, Any]) -> dict[str, Any]:
    continuity = payload["continuity"]
    total = int(payload["eval_ctx"]) + int(payload["shared_rollout_horizon"])
    window = int(payload["tokenizer_chunk_window"])
    stride = int(payload["tokenizer_decode_chunk_stride"])
    mode = str(payload["tokenizer_decode_stitch_mode"])
    ranges = chunk_ranges(total, window, stride)
    segments = selected_segments(ranges, total, mode)
    seams = [start for _, start, _ in segments[1:]]
    output: dict[str, Any] = {
        "decode_ranges": ranges,
        "selected_segments": segments,
        "seam_endpoint_frames": seams,
    }
    for name, raw_values in continuity.items():
        values = np.asarray([np.nan if value is None else value for value in raw_values], dtype=np.float64)
        finite = values[np.isfinite(values)]
        seam_rows = []
        for seam in seams:
            neighbors = [values[index] for index in (seam - 1, seam + 1) if 0 <= index < len(values) and np.isfinite(values[index])]
            neighbor_mean = float(np.mean(neighbors)) if neighbors else float("nan")
            seam_value = float(values[seam])
            seam_rows.append({
                "endpoint_frame": seam,
                "value": seam_value,
                "neighbor_mean": neighbor_mean,
                "excess_ratio": seam_value / neighbor_mean if neighbor_mean > 0 else None,
            })
        output[name] = {
            "mean": float(finite.mean()),
            "p95": float(np.percentile(finite, 95)),
            "max": float(finite.max()),
            "seams": seam_rows,
            "mean_seam_excess_ratio": float(np.mean([row["excess_ratio"] for row in seam_rows if row["excess_ratio"] is not None])),
            "max_seam_excess_ratio": float(np.max([row["excess_ratio"] for row in seam_rows if row["excess_ratio"] is not None])),
        }
    return output


def paired_metric(
    baseline: dict[str, Any],
    variant: dict[str, Any],
    horizon: str,
    metric: str,
) -> dict[str, Any]:
    base = np.asarray([row["metrics"][horizon][metric] for row in baseline["per_sample_metrics"]], dtype=np.float64)
    value = np.asarray([row["metrics"][horizon][metric] for row in variant["per_sample_metrics"]], dtype=np.float64)
    delta = value - base
    lower_is_better = not metric.endswith("_acc")
    improvement = -delta if lower_is_better else delta
    sem = float(delta.std(ddof=1) / np.sqrt(delta.size)) if delta.size > 1 else 0.0
    baseline_mean = float(base.mean())
    variant_mean = float(value.mean())
    return {
        "baseline_mean": baseline_mean,
        "variant_mean": variant_mean,
        "delta_variant_minus_baseline": float(delta.mean()),
        "delta_95pct_normal_ci": [float(delta.mean() - 1.96 * sem), float(delta.mean() + 1.96 * sem)],
        "relative_improvement_percent": float(improvement.mean() / max(abs(baseline_mean), 1e-12) * 100.0),
        "paired_win_rate": float((improvement > 0).mean()),
        "lower_is_better": lower_is_better,
        "num_samples": int(delta.size),
    }


def main(args: argparse.Namespace) -> None:
    paths = {
        "baseline": Path(args.baseline).resolve(),
        "decoder_center": Path(args.decoder_center).resolve(),
        "encoder_decoder_center": Path(args.encoder_decoder_center).resolve(),
    }
    payloads = {name: json.loads(path.read_text()) for name, path in paths.items()}
    baseline = payloads["baseline"]
    horizons = [f"h{value}" for value in baseline["horizons"]]
    metrics = sorted(baseline["metrics"][horizons[0]])
    comparisons: dict[str, Any] = {}
    csv_rows = []
    for variant_name in ("decoder_center", "encoder_decoder_center"):
        comparisons[variant_name] = {}
        for horizon in horizons:
            comparisons[variant_name][horizon] = {}
            for metric in metrics:
                row = paired_metric(baseline, payloads[variant_name], horizon, metric)
                comparisons[variant_name][horizon][metric] = row
                csv_rows.append({"variant": variant_name, "horizon": horizon, "metric": metric, **row})

    output = {
        "protocol": "paired_fixed_val128_tokenizer_center_select_ablation",
        "inputs": {name: str(path) for name, path in paths.items()},
        "checkpoint_step": int(baseline["ckpt_step"]),
        "subset_manifest": baseline["subset_manifest"],
        "variants": {
            name: {
                "encode_stride": int(payload["tokenizer_encode_chunk_stride"]),
                "encode_stitch": payload["tokenizer_encode_stitch_mode"],
                "decode_stride": int(payload["tokenizer_decode_chunk_stride"]),
                "decode_stitch": payload["tokenizer_decode_stitch_mode"],
                "continuity": continuity_summary(payload),
            }
            for name, payload in payloads.items()
        },
        "paired_metric_comparisons": comparisons,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    csv_path = output_dir / "paired_metric_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    key_path = output_dir / "h90_key_metrics.txt"
    lines = []
    for metric in KEY_METRICS:
        lines.append(metric)
        lines.append(f"  baseline={baseline['metrics']['h90'][metric]:.8g}")
        for variant_name in ("decoder_center", "encoder_decoder_center"):
            row = comparisons[variant_name]["h90"][metric]
            lines.append(
                f"  {variant_name}={row['variant_mean']:.8g} "
                f"improvement={row['relative_improvement_percent']:+.3f}% "
                f"win_rate={row['paired_win_rate']:.3f} "
                f"delta_ci={row['delta_95pct_normal_ci']}"
            )
    key_path.write_text("\n".join(lines) + "\n")
    print(key_path.read_text(), flush=True)
    print(f"wrote_summary={summary_path}", flush=True)
    print(f"wrote_csv={csv_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--decoder_center", required=True)
    parser.add_argument("--encoder_decoder_center", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
