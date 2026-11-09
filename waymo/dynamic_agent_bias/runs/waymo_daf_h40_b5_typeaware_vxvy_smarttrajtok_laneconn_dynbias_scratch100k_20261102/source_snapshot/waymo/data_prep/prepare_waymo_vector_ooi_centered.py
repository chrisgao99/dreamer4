"""Prepare OOI-centered Waymo vector-tokenizer NPZ datasets.

This script reads the scenario-level stats CSV from `analyze_waymo_ooi_raw.py`,
randomly samples OOI scenarios, splits by scenario, and expands each OOI track
into one focus-centered training sample.

For a two-OOI scenario, this writes two NPZ files:

- scenario_focus_A.npz with slot 0 = OOI A and slot 1 = OOI B
- scenario_focus_B.npz with slot 0 = OOI B and slot 1 = OOI A
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
CORE_ROOT = WAYMO_ROOT / "core"
for path in (REPO_ROOT, CORE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from waymo_vector_filter import (
        WaymoVectorConfig,
        filter_scenario_around_focus,
        iter_tfrecord_examples,
    )
except ModuleNotFoundError:
    from waymo.core.waymo_vector_filter import (
        WaymoVectorConfig,
        filter_scenario_around_focus,
        iter_tfrecord_examples,
    )


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(x) for x in value.split(";") if x != ""]


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))


def read_ooi_rows(stats_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(stats_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _truthy(row.get("has_ooi", "False")):
                continue
            if not _truthy(row.get("focus_available", "False")):
                continue
            ooi_src = _parse_int_list(row.get("ooi_src_indices", ""))
            if not ooi_src:
                continue
            if int(row.get("num_ooi_current_valid", "0")) < len(ooi_src):
                continue
            row["_ooi_src_indices"] = ooi_src
            row["_num_focus_samples"] = len(ooi_src)
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable OOI rows found in {stats_csv}")
    return rows


def read_excluded_samples(manifest_paths: List[str], level: str) -> set:
    excluded = set()
    for manifest_path in manifest_paths:
        if not manifest_path:
            continue
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Exclude manifest not found: {path}")
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario_id = row.get("scenario_id", "")
                if level == "scenario":
                    excluded.add((scenario_id,))
                else:
                    excluded.add((scenario_id, str(row.get("focus_src_index", ""))))
    return excluded


def filter_excluded_rows(rows: List[Dict[str, str]], excluded: set, level: str) -> List[Dict[str, str]]:
    if not excluded:
        return rows
    filtered: List[Dict[str, str]] = []
    for row in rows:
        scenario_id = row["scenario_id"]
        if level == "scenario":
            if (scenario_id,) not in excluded:
                filtered.append(row)
            continue

        keep_ooi = [idx for idx in row["_ooi_src_indices"] if (scenario_id, str(int(idx))) not in excluded]
        if not keep_ooi:
            continue
        kept = dict(row)
        kept["_ooi_src_indices"] = keep_ooi
        kept["_num_focus_samples"] = len(keep_ooi)
        filtered.append(kept)
    return filtered


def sample_rows(rows: List[Dict[str, str]], num_focus_samples: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    if num_focus_samples <= 0:
        return rows
    selected: List[Dict[str, str]] = []
    total = 0
    for row in rows:
        selected.append(row)
        total += int(row["_num_focus_samples"])
        if total >= num_focus_samples:
            break
    return selected


def split_rows(rows: List[Dict[str, str]], val_fraction: float, seed: int) -> Dict[str, List[Dict[str, str]]]:
    rng = random.Random(seed + 17)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_fraction)) if val_fraction > 0 else 0
    n_val = min(max(0, n_val), max(0, len(rows) - 1))
    return {
        "train": rows[n_val:],
        "val": rows[:n_val],
    }


def _fetch_records_for_tfrecord(tfrecord_path: str, indices: Iterable[int]) -> Dict[int, Dict[str, np.ndarray]]:
    wanted = sorted(set(int(i) for i in indices))
    if not wanted:
        return {}
    max_idx = wanted[-1]
    wanted_set = set(wanted)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for idx, data in enumerate(iter_tfrecord_examples(tfrecord_path, max_records=max_idx + 1)):
        if idx in wanted_set:
            out[idx] = data
            if len(out) == len(wanted_set):
                break
    missing = wanted_set - set(out)
    if missing:
        raise IndexError(f"Missing records {sorted(missing)} in {tfrecord_path}")
    return out


def _write_sample(
    *,
    data: Dict[str, np.ndarray],
    row: Dict[str, str],
    focus_src_index: int,
    split: str,
    output_dir: Path,
    cfg: WaymoVectorConfig,
) -> Dict[str, str]:
    ooi_src_indices = list(row["_ooi_src_indices"])
    ooi_track_ids = _parse_int_list(row.get("ooi_track_ids", ""))
    focus_track_id_from_row = -1
    if focus_src_index in ooi_src_indices:
        focus_pos = ooi_src_indices.index(focus_src_index)
        if focus_pos < len(ooi_track_ids):
            focus_track_id_from_row = int(ooi_track_ids[focus_pos])

    scenario_id = _safe_name(row["scenario_id"])
    stem = f"{scenario_id}_focus_{focus_track_id_from_row}_src{int(focus_src_index)}"
    out_path = output_dir / split / f"{stem}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        try:
            with np.load(out_path, allow_pickle=False):
                pass
            return {
                "split": split,
                "scenario_id": row["scenario_id"],
                "tfrecord_path": row["tfrecord_path"],
                "record_index": row["record_index"],
                "focus_src_index": str(int(focus_src_index)),
                "focus_track_id": str(focus_track_id_from_row),
                "ooi_src_indices": row.get("ooi_src_indices", ""),
                "ooi_track_ids": row.get("ooi_track_ids", ""),
                "interaction_labels": row.get("interaction_labels", ""),
                "npz_path": str(out_path),
            }
        except Exception:
            out_path.unlink(missing_ok=True)

    item = filter_scenario_around_focus(
        data,
        focus_src_index=focus_src_index,
        cfg=cfg,
        priority_src_indices=ooi_src_indices,
        map_crop_src_indices=ooi_src_indices,
    )
    focus_track_id = int(item["focus_track_id"])

    item.update(
        {
            "sample_split": np.asarray(split),
            "source_tfrecord_path": np.asarray(row["tfrecord_path"]),
            "source_record_index": np.asarray(int(row["record_index"]), dtype=np.int64),
            "stats_interaction_labels": np.asarray(row.get("interaction_labels", "")),
            "stats_focus_rule": np.asarray(row.get("focus_rule", "")),
        }
    )
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **item)
    tmp_path.replace(out_path)
    return {
        "split": split,
        "scenario_id": row["scenario_id"],
        "tfrecord_path": row["tfrecord_path"],
        "record_index": row["record_index"],
        "focus_src_index": str(int(focus_src_index)),
        "focus_track_id": str(focus_track_id),
        "ooi_src_indices": row.get("ooi_src_indices", ""),
        "ooi_track_ids": row.get("ooi_track_ids", ""),
        "interaction_labels": row.get("interaction_labels", ""),
        "npz_path": str(out_path),
    }


def write_manifest(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = [
        "split",
        "scenario_id",
        "tfrecord_path",
        "record_index",
        "focus_src_index",
        "focus_track_id",
        "ooi_src_indices",
        "ooi_track_ids",
        "interaction_labels",
        "npz_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_ooi_rows(args.stats_csv)
    raw_rows = len(rows)
    raw_focus_samples = int(sum(int(r["_num_focus_samples"]) for r in rows))
    excluded = read_excluded_samples(args.exclude_manifest, args.exclude_level)
    rows = filter_excluded_rows(rows, excluded, args.exclude_level)
    available_focus_samples = int(sum(int(r["_num_focus_samples"]) for r in rows))
    if not rows:
        raise ValueError("No OOI rows left after applying exclude manifests")
    selected = sample_rows(rows, args.num_focus_samples, args.seed)
    splits = split_rows(selected, args.val_fraction, args.seed)

    cfg = WaymoVectorConfig(
        num_agents=args.num_agents,
        agent_distance_threshold=args.agent_distance_threshold,
        map_distance_threshold=args.map_distance_threshold,
        max_map_polylines=args.max_map_polylines,
        max_points_per_polyline=args.max_points_per_polyline,
        use_all_timesteps_for_selection=args.use_all_timesteps_for_selection,
        normalize_to_ego=True,
        prioritize_objects_of_interest=True,
    )

    for split in ("train", "val"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    summary = {
        "stats_csv": args.stats_csv,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "exclude_manifest": args.exclude_manifest,
        "exclude_level": args.exclude_level,
        "excluded_entries": len(excluded),
        "raw_scenarios": raw_rows,
        "raw_focus_samples": raw_focus_samples,
        "available_scenarios_after_exclude": len(rows),
        "available_focus_samples_after_exclude": available_focus_samples,
        "requested_num_focus_samples": args.num_focus_samples,
        "selected_scenarios": len(selected),
        "selected_focus_samples": int(sum(int(r["_num_focus_samples"]) for r in selected)),
        "train_scenarios": len(splits["train"]),
        "val_scenarios": len(splits["val"]),
        "train_focus_samples_expected": int(sum(int(r["_num_focus_samples"]) for r in splits["train"])),
        "val_focus_samples_expected": int(sum(int(r["_num_focus_samples"]) for r in splits["val"])),
        "config": cfg.__dict__,
    }
    (output_dir / "prepare_config.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    manifest_rows: List[Dict[str, str]] = []
    t0 = time.time()
    for split in ("train", "val"):
        split_rows_list = splits[split]
        by_tfrecord: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in split_rows_list:
            by_tfrecord[row["tfrecord_path"]].append(row)

        for file_idx, (tfrecord_path, tf_rows) in enumerate(sorted(by_tfrecord.items()), start=1):
            record_indices = [int(row["record_index"]) for row in tf_rows]
            records = _fetch_records_for_tfrecord(tfrecord_path, record_indices)
            print(f"[{split}] {file_idx}/{len(by_tfrecord)} {tfrecord_path} rows={len(tf_rows)}", flush=True)
            for row in tf_rows:
                data = records[int(row["record_index"])]
                for focus_src_index in row["_ooi_src_indices"]:
                    manifest_rows.append(
                        _write_sample(
                            data=data,
                            row=row,
                            focus_src_index=int(focus_src_index),
                            split=split,
                            output_dir=output_dir,
                            cfg=cfg,
                        )
                    )
                    if args.log_every > 0 and len(manifest_rows) % args.log_every == 0:
                        elapsed = max(1e-6, time.time() - t0)
                        print(f"  written={len(manifest_rows)} samples ({len(manifest_rows) / elapsed:.2f}/s)", flush=True)

    write_manifest(manifest_rows, output_dir / "manifest.csv")
    actual_summary = dict(summary)
    actual_summary.update(
        {
            "written_focus_samples": len(manifest_rows),
            "train_focus_samples": sum(1 for row in manifest_rows if row["split"] == "train"),
            "val_focus_samples": sum(1 for row in manifest_rows if row["split"] == "val"),
        }
    )
    (output_dir / "prepare_summary.json").write_text(json.dumps(actual_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(actual_summary, indent=2, sort_keys=True))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare OOI-centered Waymo vector-tokenizer NPZ files.")
    p.add_argument(
        "--stats_csv",
        type=str,
        default="/p/yufeng/tri30/dreamer4/waymo/evaluation/reports/ooi_raw_stats_train/waymo_ooi_scenario_stats.csv",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_ooi_centered_50k",
    )
    p.add_argument("--num_focus_samples", type=int, default=50_000, help="0 means use all OOI focus samples.")
    p.add_argument(
        "--exclude_manifest",
        type=str,
        nargs="*",
        default=[],
        help="Existing OOI-centered manifest.csv files to exclude from this extraction.",
    )
    p.add_argument(
        "--exclude_level",
        choices=["sample", "scenario"],
        default="sample",
        help="Exclude exact (scenario, focus_src_index) samples or whole scenarios from exclude manifests.",
    )
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=1000)

    p.add_argument("--num_agents", type=int, default=32)
    p.add_argument("--agent_distance_threshold", type=float, default=80.0)
    p.add_argument("--map_distance_threshold", type=float, default=100.0)
    p.add_argument("--max_map_polylines", type=int, default=256)
    p.add_argument("--max_points_per_polyline", type=int, default=20)
    p.add_argument("--use_all_timesteps_for_selection", action="store_true")
    return p


def main() -> None:
    prepare(build_argparser().parse_args())


if __name__ == "__main__":
    main()
