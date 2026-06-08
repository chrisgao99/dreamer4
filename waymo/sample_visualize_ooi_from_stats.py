"""Sample OOI-labeled scenarios from the raw stats CSV and render previews.

The stats CSV produced by `analyze_waymo_ooi_raw.py` stores the TFRecord path
and record index for each scenario. This script uses those pointers to fetch
the raw scenario, convert it through the vector filter, and render MP4/PNG
previews for visual label inspection.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from visualize_waymo_vector_npz import render_video
    from waymo_vector_filter import WaymoVectorConfig, filter_scenario, iter_tfrecord_examples
except ModuleNotFoundError:
    from waymo.visualize_waymo_vector_npz import render_video
    from waymo.waymo_vector_filter import WaymoVectorConfig, filter_scenario, iter_tfrecord_examples


DEFAULT_LABELS = [
    "close_interaction_5m",
    "close_interaction_10m",
    "crossing_conflict_like",
    "following_or_leading",
    "cut_in_or_merge_like",
    "intersection_turn_or_large_turn",
    "lane_change_like",
    "stopped_or_yield_like",
    "dense_interaction",
]


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "unknown"


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _read_matching_rows(csv_path: str, label: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _truthy(row.get("has_ooi", "False")):
                continue
            labels = set(x for x in row.get("interaction_labels", "").split(";") if x)
            if label == "any_ooi" or label in labels:
                rows.append(row)
    return rows


def _fetch_record(tfrecord_path: str, record_index: int) -> Dict[str, np.ndarray]:
    for idx, data in enumerate(iter_tfrecord_examples(tfrecord_path, max_records=record_index + 1)):
        if idx == record_index:
            return data
    raise IndexError(f"Record index {record_index} not found in {tfrecord_path}")


def _row_metadata(row: Dict[str, str], label: str) -> Dict[str, np.ndarray]:
    keys = [
        "tfrecord_path",
        "record_index",
        "interaction_labels",
        "ooi_src_indices",
        "ooi_track_ids",
        "ooi_types",
        "focus_src_index",
        "focus_track_id",
        "focus_type",
        "focus_is_sdc",
        "focus_rule",
    ]
    out = {f"stats_{key}": np.asarray(row.get(key, "")) for key in keys}
    out["sample_label"] = np.asarray(label)
    return out


def _render_row(row: Dict[str, str], label: str, sample_idx: int, args: argparse.Namespace) -> Dict[str, str]:
    tfrecord_path = row["tfrecord_path"]
    record_index = int(row["record_index"])
    scenario_id = _safe_name(row.get("scenario_id", f"record_{record_index}"))
    label_dir = Path(args.output_dir) / _safe_name(label)
    label_dir.mkdir(parents=True, exist_ok=True)

    data = _fetch_record(tfrecord_path, record_index)
    item = filter_scenario(data, cfg=WaymoVectorConfig(
        num_agents=args.num_agents,
        agent_distance_threshold=args.agent_distance_threshold,
        map_distance_threshold=args.map_distance_threshold,
        max_map_polylines=args.max_map_polylines,
        max_points_per_polyline=args.max_points_per_polyline,
        use_all_timesteps_for_selection=not args.history_only_selection,
        normalize_to_ego=True,
        prioritize_objects_of_interest=True,
    ))
    item.update(_row_metadata(row, label))

    stem = f"{sample_idx:02d}_{scenario_id}"
    npz_path = label_dir / f"{stem}.npz"
    mp4_path = label_dir / f"{stem}.mp4"
    preview_path = label_dir / f"{stem}_preview.png"
    np.savez_compressed(npz_path, **item)

    render_video(
        npz_path=str(npz_path),
        output_path=str(mp4_path),
        fps=args.fps,
        width=args.width,
        height=args.height,
        trail=args.trail,
        margin_m=args.margin_m,
        start=args.start,
        end=args.end,
        preview_png=str(preview_path),
        preview_frames=args.preview_frames,
        preview_cols=args.preview_cols,
        gif_output=None,
    )

    result = dict(row)
    result.update({
        "sample_label": label,
        "sample_index": str(sample_idx),
        "npz_path": str(npz_path),
        "mp4_path": str(mp4_path),
        "preview_png": str(preview_path),
    })
    return result


def main() -> None:
    args = build_argparser().parse_args()
    rng = random.Random(args.seed)
    labels = args.labels or DEFAULT_LABELS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected: List[Dict[str, str]] = []
    used_scenarios: set[str] = set()
    for label in labels:
        rows = _read_matching_rows(args.stats_csv, label)
        rng.shuffle(rows)
        count = 0
        print(f"{label}: candidates={len(rows)}")
        for row in rows:
            scenario_id = row.get("scenario_id", "")
            if not args.allow_duplicates and scenario_id in used_scenarios:
                continue
            selected.append(_render_row(row, label, count, args))
            used_scenarios.add(scenario_id)
            count += 1
            print(f"  rendered {label} #{count}: {scenario_id}")
            if count >= args.samples_per_label:
                break
        if count < args.samples_per_label:
            print(f"  warning: only rendered {count}/{args.samples_per_label} for {label}")

    manifest_path = out_dir / "selected_samples.csv"
    if selected:
        fieldnames = sorted(set().union(*(row.keys() for row in selected)))
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
    print(f"manifest: {manifest_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sample and visualize OOI scenarios from stats CSV.")
    p.add_argument(
        "--stats_csv",
        type=str,
        default="/p/yufeng/tri30/dreamer4/waymo/reports/ooi_raw_stats_train/waymo_ooi_scenario_stats.csv",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/p/yufeng/tri30/dreamer4/waymo/reports/ooi_visual_samples",
    )
    p.add_argument("--labels", type=str, nargs="*", default=None)
    p.add_argument("--samples_per_label", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_duplicates", action="store_true")

    p.add_argument("--num_agents", type=int, default=32)
    p.add_argument("--agent_distance_threshold", type=float, default=80.0)
    p.add_argument("--map_distance_threshold", type=float, default=100.0)
    p.add_argument("--max_map_polylines", type=int, default=256)
    p.add_argument("--max_points_per_polyline", type=int, default=20)
    p.add_argument("--history_only_selection", action="store_true")

    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--trail", type=int, default=15)
    p.add_argument("--margin_m", type=float, default=15.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--preview_frames", type=int, default=16)
    p.add_argument("--preview_cols", type=int, default=4)
    return p


if __name__ == "__main__":
    main()
