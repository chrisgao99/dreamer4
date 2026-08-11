"""Build the v2 lossless 91-step interaction-pair dataset.

The builder reuses the existing 50k OOI-centred trajectories and reads only
length/width arrays from their referenced raw TFExamples.  Original two-OOI
pairs are unconditional.  Non-OOI pairs are mined by physical path-contact
interval and ranked per focus with a soft PET score.  By default all qualifying
pairs are retained; an optional positive per-focus cap can be used for smaller
experiments.  Outputs are sharded so a large run never needs to hold every
trajectory in RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .common import ensure_agent_time_layout
    from .full_pair_samples import FULL_PAIR_FEATURE_NAMES, FullPairConfig, FullPairSample, extract_full_pair_sample
except ImportError:
    from common import ensure_agent_time_layout  # type: ignore
    from full_pair_samples import FULL_PAIR_FEATURE_NAMES, FullPairConfig, FullPairSample, extract_full_pair_sample  # type: ignore

try:
    from waymo.core.waymo_vector_filter import _temporal_agent_feature, iter_tfrecord_examples
except ModuleNotFoundError:
    from core.waymo_vector_filter import _temporal_agent_feature, iter_tfrecord_examples  # type: ignore


CSV_FIELDS = (
    "sample_index",
    "split",
    "scenario_id",
    "source_path",
    "first_agent_id",
    "second_agent_id",
    "first_agent_type",
    "second_agent_type",
    "is_original_ooi_pair",
    "event_mode",
    "primary_step_first",
    "primary_step_second",
    "interval_start_first",
    "interval_end_first",
    "interval_start_second",
    "interval_end_second",
    "zone_pet_steps",
    "zone_pet_s",
    "center_pet_steps",
    "center_pet_s",
    "min_clearance_m",
    "num_contact_components",
    "primary_component_cells",
    "relevance_score",
    "valid_steps_first",
    "valid_steps_second",
    "first_length_m",
    "first_width_m",
    "second_length_m",
    "second_width_m",
    "shard",
    "shard_row",
)


def _parse_ids(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(";") if part)


def read_manifest(path: Path, max_focus_files: int) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_focus_files > 0:
        rows = rows[:max_focus_files]
    if not rows:
        raise ValueError(f"No focus files in {path}")
    return rows


def _default_size(agent_type: int) -> tuple[float, float]:
    return {
        1: (4.8, 2.0),
        2: (0.8, 0.8),
        3: (1.8, 0.7),
    }.get(int(agent_type), (1.0, 1.0))


def selected_agent_sizes(
    raw_data: dict[str, np.ndarray],
    agent_src_indices: np.ndarray,
    agents: np.ndarray,
    agent_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    length = _temporal_agent_feature(raw_data, "length")
    width = _temporal_agent_feature(raw_data, "width")
    result = np.zeros((len(agent_src_indices), agents.shape[1], 2), dtype=np.float32)
    fallbacks = 0
    for selected_index in np.flatnonzero(agent_mask):
        selected_index = int(selected_index)
        source_index = int(agent_src_indices[selected_index])
        result[selected_index, :, 0] = length[source_index]
        result[selected_index, :, 1] = width[source_index]
        valid = agents[selected_index, :, 5] > 0.5
        usable = valid & (result[selected_index, :, 0] > 0.1) & (result[selected_index, :, 1] > 0.1)
        if bool(usable.any()):
            median = np.median(result[selected_index, usable], axis=0)
        else:
            valid_indices = np.flatnonzero(valid)
            agent_type = int(round(float(agents[selected_index, valid_indices[0], 7]))) if len(valid_indices) else 0
            median = np.asarray(_default_size(agent_type), dtype=np.float32)
            fallbacks += 1
        missing = valid & ~usable
        result[selected_index, missing] = median
    return result, fallbacks


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split: str,
        shard_size: int,
        dt: float,
        *,
        resume: bool = False,
    ):
        self.output_dir = output_dir
        self.split = split
        self.shard_size = int(shard_size)
        self.dt = float(dt)
        self.buffer: list[tuple[FullPairSample, str, str]] = []
        self.shard_index = 0
        self.total = 0
        self.shards: list[dict[str, object]] = []
        self.csv_path = output_dir / f"{split}_samples.csv"
        self.existing_rows: list[dict[str, str]] = []
        if resume:
            self._load_existing_output()
        self.csv_handle = self.csv_path.open("a" if resume else "w", newline="")
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=CSV_FIELDS)
        if not resume:
            self.csv_writer.writeheader()

    def _load_existing_output(self) -> None:
        shard_paths = sorted(self.output_dir.glob(f"{self.split}_samples_*.npz"))
        if not shard_paths or not self.csv_path.exists():
            raise FileNotFoundError(
                f"Cannot resume split={self.split}: both completed shards and {self.csv_path} are required"
            )
        with self.csv_path.open(newline="") as handle:
            self.existing_rows = list(csv.DictReader(handle))
        expected_total = 0
        for shard_index, path in enumerate(shard_paths):
            expected_name = f"{self.split}_samples_{shard_index:05d}.npz"
            if path.name != expected_name:
                raise ValueError(
                    f"Cannot resume non-contiguous shards: expected {expected_name}, found {path.name}"
                )
            with np.load(path, allow_pickle=False) as data:
                count = len(data["trajectory"])
            self.shards.append({"path": str(path), "samples": count, "start_index": expected_total})
            expected_total += count
        if expected_total != len(self.existing_rows):
            raise ValueError(
                f"Cannot resume split={self.split}: shards contain {expected_total} samples but CSV has "
                f"{len(self.existing_rows)} rows"
            )
        for expected_index, row in enumerate(self.existing_rows):
            if int(row["sample_index"]) != expected_index or row["split"] != self.split:
                raise ValueError(
                    f"Cannot resume split={self.split}: invalid CSV row at index {expected_index}"
                )
        self.total = expected_total
        self.shard_index = len(shard_paths)
        print(
            f"[{self.split}:resume] loaded shards={self.shard_index} samples={self.total}",
            flush=True,
        )

    def add(self, sample: FullPairSample, scenario_id: str, source_path: str) -> None:
        self.buffer.append((sample, scenario_id, source_path))
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_name = f"{self.split}_samples_{self.shard_index:05d}.npz"
        path = self.output_dir / shard_name
        samples = [item[0] for item in self.buffer]
        arrays = {
            "trajectory": np.stack([sample.trajectory for sample in samples]).astype(np.float32),
            "valid_mask": np.stack([sample.valid_mask for sample in samples]).astype(bool),
            "agent_size_m": np.stack([sample.agent_size_m for sample in samples]).astype(np.float32),
            "scenario_id": np.asarray([item[1] for item in self.buffer]),
            "source_path": np.asarray([item[2] for item in self.buffer]),
            "first_agent_id": np.asarray([sample.first_agent_id for sample in samples], dtype=np.int64),
            "second_agent_id": np.asarray([sample.second_agent_id for sample in samples], dtype=np.int64),
            "first_agent_type": np.asarray([sample.first_agent_type for sample in samples], dtype=np.int8),
            "second_agent_type": np.asarray([sample.second_agent_type for sample in samples], dtype=np.int8),
            "is_original_ooi_pair": np.asarray([sample.is_original_ooi_pair for sample in samples], dtype=bool),
            "event_mode": np.asarray([sample.event_mode for sample in samples]),
            "primary_step_first": np.asarray([sample.primary_step_first for sample in samples], dtype=np.float32),
            "primary_step_second": np.asarray([sample.primary_step_second for sample in samples], dtype=np.float32),
            "interaction_interval": np.asarray([
                [sample.interval_start_first, sample.interval_end_first,
                 sample.interval_start_second, sample.interval_end_second]
                for sample in samples
            ], dtype=np.float32),
            "zone_pet_steps": np.asarray([sample.zone_pet_steps for sample in samples], dtype=np.float32),
            "center_pet_steps": np.asarray([sample.center_pet_steps for sample in samples], dtype=np.float32),
            "min_clearance_m": np.asarray([sample.min_clearance_m for sample in samples], dtype=np.float32),
            "num_contact_components": np.asarray([sample.num_contact_components for sample in samples], dtype=np.int16),
            "primary_component_cells": np.asarray([sample.primary_component_cells for sample in samples], dtype=np.int32),
            "relevance_score": np.asarray([sample.relevance_score for sample in samples], dtype=np.float32),
            "conflict_xy": np.stack([sample.conflict_xy for sample in samples]).astype(np.float32),
            "feature_names": np.asarray(FULL_PAIR_FEATURE_NAMES),
            "time_s": (np.arange(samples[0].trajectory.shape[1], dtype=np.float32) * self.dt),
        }
        tmp_path = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(path)
        for shard_row, (sample, scenario_id, source_path) in enumerate(self.buffer):
            size = sample.agent_size_m
            self.csv_writer.writerow({
                "sample_index": self.total + shard_row,
                "split": self.split,
                "scenario_id": scenario_id,
                "source_path": source_path,
                "first_agent_id": sample.first_agent_id,
                "second_agent_id": sample.second_agent_id,
                "first_agent_type": sample.first_agent_type,
                "second_agent_type": sample.second_agent_type,
                "is_original_ooi_pair": sample.is_original_ooi_pair,
                "event_mode": sample.event_mode,
                "primary_step_first": sample.primary_step_first,
                "primary_step_second": sample.primary_step_second,
                "interval_start_first": sample.interval_start_first,
                "interval_end_first": sample.interval_end_first,
                "interval_start_second": sample.interval_start_second,
                "interval_end_second": sample.interval_end_second,
                "zone_pet_steps": sample.zone_pet_steps,
                "zone_pet_s": sample.zone_pet_steps * self.dt,
                "center_pet_steps": sample.center_pet_steps,
                "center_pet_s": sample.center_pet_steps * self.dt,
                "min_clearance_m": sample.min_clearance_m,
                "num_contact_components": sample.num_contact_components,
                "primary_component_cells": sample.primary_component_cells,
                "relevance_score": sample.relevance_score,
                "valid_steps_first": int(sample.valid_mask[0].sum()),
                "valid_steps_second": int(sample.valid_mask[1].sum()),
                "first_length_m": float(size[0, 0]),
                "first_width_m": float(size[0, 1]),
                "second_length_m": float(size[1, 0]),
                "second_width_m": float(size[1, 1]),
                "shard": shard_name,
                "shard_row": shard_row,
            })
        count = len(self.buffer)
        self.shards.append({"path": str(path), "samples": count, "start_index": self.total})
        self.total += count
        self.shard_index += 1
        self.buffer.clear()
        self.csv_handle.flush()
        print(f"[{self.split}] wrote {shard_name} samples={count} total={self.total}", flush=True)

    def close(self) -> None:
        self.flush()
        self.csv_handle.close()


def _expected_two_ooi_keys(rows: Iterable[dict[str, str]]) -> set[tuple[str, str, int, int]]:
    result = set()
    for row in rows:
        ids = _parse_ids(row["ooi_track_ids"])
        if len(ids) == 2:
            low, high = sorted(ids)
            result.add((row["split"], row["scenario_id"], low, high))
    return result


def _pair_key(split: str, scenario_id: str, first_id: int, second_id: int) -> tuple[str, str, int, int]:
    low, high = sorted((int(first_id), int(second_id)))
    return split, scenario_id, low, high


def load_resume_state(
    rows: list[dict[str, str]],
    raw_items: list[tuple[str, dict[int, list[dict[str, str]]]]],
    writers: dict[str, ShardWriter],
) -> tuple[
    set[tuple[str, str, int, int]],
    set[tuple[str, str, int, int]],
    Counter,
    int,
    dict[str, object],
]:
    """Restore durable state and choose a conservative raw-file restart point."""
    manifest_by_source = {row["npz_path"]: row for row in rows}
    raw_position = {path: index for index, (path, _) in enumerate(raw_items)}
    seen: set[tuple[str, str, int, int]] = set()
    retained_ooi: set[tuple[str, str, int, int]] = set()
    stats = Counter()
    last_durable_raw_by_split: dict[str, int] = {}

    for split, writer in writers.items():
        if not writer.existing_rows:
            last_durable_raw_by_split[split] = 0
            continue
        durable_positions = []
        for row in writer.existing_rows:
            key = _pair_key(split, row["scenario_id"], int(row["first_agent_id"]), int(row["second_agent_id"]))
            if key in seen:
                raise ValueError(f"Cannot resume: duplicate durable pair key {key}")
            seen.add(key)
            is_ooi = row["is_original_ooi_pair"].lower() == "true"
            if is_ooi:
                retained_ooi.add(key)
                stats["retained_ooi_pairs"] += 1
            else:
                stats["retained_mined_pairs"] += 1
            stats[f"event_mode_{row['event_mode']}"] += 1
            source = row["source_path"]
            if source not in manifest_by_source:
                raise ValueError(f"Cannot resume: durable source is absent from manifest: {source}")
            durable_positions.append(raw_position[manifest_by_source[source]["tfrecord_path"]])
        last_durable_raw_by_split[split] = max(durable_positions)

    # Restart at the earliest split-specific last flush. Samples produced after
    # that point may have existed only in a lost in-memory buffer. Durable pair
    # keys suppress duplicates while the overlap is recomputed.
    restart_index = min(last_durable_raw_by_split.values())
    skipped_focus = sum(
        len(focus_rows)
        for _, wanted in raw_items[:restart_index]
        for focus_rows in wanted.values()
    )
    stats["processed_focus_files"] = skipped_focus
    metadata: dict[str, object] = {
        "enabled": True,
        "durable_samples": sum(writer.total for writer in writers.values()),
        "last_durable_raw_file_by_split_1_based": {
            split: index + 1 for split, index in last_durable_raw_by_split.items()
        },
        "restart_raw_file_1_based": restart_index + 1,
        "skipped_focus_files_before_restart": skipped_focus,
        "overlap_is_recomputed_and_deduplicated": True,
        "stats_note": (
            "retained counts and event-mode counts include durable output; extraction diagnostics "
            "such as candidate/non-contact counts cover only the resumed suffix"
        ),
    }
    print(
        f"[resume] durable={metadata['durable_samples']} restart_raw={restart_index + 1}/{len(raw_items)} "
        f"skipped_focus={skipped_focus}",
        flush=True,
    )
    return seen, retained_ooi, stats, restart_index, metadata


def process_focus_file(
    row: dict[str, str],
    raw_data: dict[str, np.ndarray],
    cfg: FullPairConfig,
    seen: set[tuple[str, str, int, int]],
    retained_ooi: set[tuple[str, str, int, int]],
    writer: ShardWriter,
    stats: Counter,
) -> None:
    with np.load(row["npz_path"], allow_pickle=False) as data:
        agent_mask = np.asarray(data["agent_mask"], dtype=bool)
        agents = ensure_agent_time_layout(np.asarray(data["agents"], dtype=np.float32), agent_mask)
        agent_ids = np.asarray(data["agent_ids"], dtype=np.int64)
        agent_src_indices = np.asarray(data["agent_src_indices"], dtype=np.int64)
        sizes, fallback_count = selected_agent_sizes(raw_data, agent_src_indices, agents, agent_mask)
        stats["dimension_fallback_agents"] += fallback_count
        if not bool(agent_mask[0]):
            stats["invalid_focus_files"] += 1
            return
        ooi_ids = set(_parse_ids(row["ooi_track_ids"]))
        focus_id = int(agent_ids[0])
        mined: list[FullPairSample] = []
        mandatory: list[FullPairSample] = []
        for candidate_index in np.flatnonzero(agent_mask):
            candidate_index = int(candidate_index)
            if candidate_index == 0:
                continue
            candidate_id = int(agent_ids[candidate_index])
            key = _pair_key(row["split"], row["scenario_id"], focus_id, candidate_id)
            is_ooi_pair = len(ooi_ids) == 2 and {focus_id, candidate_id} == ooi_ids
            stats["candidate_focus_pairs"] += 1
            sample = extract_full_pair_sample(
                agents[0],
                agents[candidate_index],
                sizes[0],
                sizes[candidate_index],
                index_a=0,
                index_b=candidate_index,
                agent_id_a=focus_id,
                agent_id_b=candidate_id,
                is_original_ooi_pair=is_ooi_pair,
                cfg=cfg,
            )
            if sample is None:
                stats["non_ooi_without_physical_contact"] += 1
                continue
            if is_ooi_pair:
                mandatory.append(sample)
            else:
                stats["non_ooi_with_physical_contact"] += 1
                mined.append(sample)

        mined.sort(key=lambda sample: (-sample.relevance_score, sample.zone_pet_steps, sample.min_clearance_m,
                                       sample.first_agent_id, sample.second_agent_id))
        if cfg.non_ooi_top_k_per_focus > 0:
            selected_mined = mined[: cfg.non_ooi_top_k_per_focus]
        else:
            selected_mined = mined
        selected = mandatory + selected_mined
        stats["non_ooi_removed_by_top_k"] += len(mined) - len(selected_mined)
        for sample in selected:
            key = _pair_key(row["split"], row["scenario_id"], sample.first_agent_id, sample.second_agent_id)
            if key in seen:
                stats["duplicate_unordered_pairs"] += 1
                continue
            seen.add(key)
            if sample.is_original_ooi_pair:
                retained_ooi.add(key)
                stats["retained_ooi_pairs"] += 1
            else:
                stats["retained_mined_pairs"] += 1
            stats[f"event_mode_{sample.event_mode}"] += 1
            writer.add(sample, row["scenario_id"], row["npz_path"])


def build(args: argparse.Namespace) -> None:
    started = time.time()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    existing_shards = list(output_dir.glob("*_samples_*.npz")) if output_dir.exists() else []
    if existing_shards and not args.resume:
        raise FileExistsError(f"Output already contains shards; choose a new directory: {output_dir}")
    if args.resume and not existing_shards:
        raise FileNotFoundError(f"--resume requires existing completed shards under {output_dir}")
    if args.resume and (output_dir / "summary.json").exists():
        raise FileExistsError(f"Dataset already has summary.json and is complete: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(data_root / "manifest.csv", args.max_focus_files)
    cfg = FullPairConfig(
        dt=args.dt,
        contact_buffer_m=args.contact_buffer_m,
        pet_soft_scale_s=args.pet_soft_scale_s,
        non_ooi_top_k_per_focus=args.non_ooi_top_k_per_focus,
    )
    writers = {
        split: ShardWriter(output_dir, split, args.shard_size, cfg.dt, resume=args.resume)
        for split in sorted({row["split"] for row in rows})
    }
    by_tfrecord: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_tfrecord[row["tfrecord_path"]][int(row["record_index"])].append(row)
    raw_items = list(sorted(by_tfrecord.items()))
    expected_ooi = _expected_two_ooi_keys(rows)
    resume_metadata: dict[str, object] = {"enabled": False}
    restart_index = 0
    if args.resume:
        seen, retained_ooi, stats, restart_index, resume_metadata = load_resume_state(
            rows, raw_items, writers
        )
    else:
        retained_ooi = set()
        seen = set()
        stats = Counter()
    failures: list[dict[str, object]] = []

    for file_number, (tfrecord_path, wanted) in enumerate(raw_items, start=1):
        if file_number - 1 < restart_index:
            continue
        max_record = max(wanted)
        found: set[int] = set()
        try:
            for record_index, raw_data in enumerate(iter_tfrecord_examples(tfrecord_path, max_records=max_record + 1)):
                focus_rows = wanted.get(record_index)
                if not focus_rows:
                    continue
                found.add(record_index)
                for row in focus_rows:
                    try:
                        process_focus_file(
                            row, raw_data, cfg, seen, retained_ooi, writers[row["split"]], stats
                        )
                        stats["processed_focus_files"] += 1
                    except Exception as exc:
                        failures.append({"npz_path": row["npz_path"], "error": repr(exc)})
                        print(f"warning: failed {row['npz_path']}: {exc!r}", flush=True)
            missing_records = sorted(set(wanted) - found)
            if missing_records:
                failures.append({"tfrecord_path": tfrecord_path, "missing_record_indices": missing_records})
        except Exception as exc:
            failures.append({"tfrecord_path": tfrecord_path, "error": repr(exc)})
            print(f"warning: failed TFRecord {tfrecord_path}: {exc!r}", flush=True)
        elapsed = time.time() - started
        print(
            f"[raw {file_number}/{len(by_tfrecord)}] focus={stats['processed_focus_files']}/{len(rows)} "
            f"ooi={stats['retained_ooi_pairs']}/{len(expected_ooi)} mined={stats['retained_mined_pairs']} "
            f"failures={len(failures)} elapsed={elapsed:.1f}s",
            flush=True,
        )

    for writer in writers.values():
        writer.close()
    missing_ooi = sorted(expected_ooi - retained_ooi)
    summary = {
        "version": "full_91_physical_contact_intervals_v2",
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "focus_files": len(rows),
        "raw_tfrecords": len(by_tfrecord),
        "config": asdict(cfg),
        "feature_names": list(FULL_PAIR_FEATURE_NAMES),
        "trajectory_shape_per_sample": [2, 91, 6],
        "valid_mask_shape_per_sample": [2, 91],
        "expected_original_two_ooi_pairs": len(expected_ooi),
        "retained_original_two_ooi_pairs": len(retained_ooi),
        "missing_original_two_ooi_pairs": len(missing_ooi),
        "missing_original_two_ooi_preview": missing_ooi[:50],
        "stats": dict(stats),
        "failures": failures[:100],
        "num_failures": len(failures),
        "resume": resume_metadata,
        "splits": {
            split: {"samples": writer.total, "csv": str(writer.csv_path), "shards": writer.shards}
            for split, writer in writers.items()
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if failures:
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if missing_ooi:
        raise RuntimeError(f"OOI completeness invariant failed: missing {len(missing_ooi)} original pairs")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default="/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_ooi_centered_50k",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_full_pairs_50k_v2",
    )
    parser.add_argument("--max_focus_files", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--contact_buffer_m", type=float, default=1.0)
    parser.add_argument("--pet_soft_scale_s", type=float, default=3.0)
    parser.add_argument(
        "--non_ooi_top_k_per_focus",
        type=int,
        default=0,
        help="Maximum mined non-OOI pairs per focus; <=0 retains all qualifying pairs (default: 0).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted output from its durable shards and conservatively recompute overlap.",
    )
    return parser


def main() -> None:
    build(build_argparser().parse_args())


if __name__ == "__main__":
    main()
