"""Build event-aligned matched contrastive samples before tokenizer training."""

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
from scipy.spatial import cKDTree

try:
    from .pair_samples import (
        HISTORY_FEATURE_NAMES,
        RELATION_NAMES,
        RESPONSE_NAMES,
        SampleConfig,
        build_scene_samples,
    )
except ImportError:
    from pair_samples import (  # type: ignore
        HISTORY_FEATURE_NAMES,
        RELATION_NAMES,
        RESPONSE_NAMES,
        SampleConfig,
        build_scene_samples,
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not parsed or any(step <= 0 for step in parsed):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integer steps")
    return parsed


def _reservoir_manifest_paths(
    manifest_path: Path,
    *,
    split: str,
    limit: int,
    seed: int,
    selection: str,
) -> tuple[list[str], int]:
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    seen = 0
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") != split:
                continue
            path = row.get("npz_path", "")
            if not path:
                continue
            seen += 1
            if limit <= 0:
                selected.append(path)
            elif selection == "first":
                if len(selected) < limit:
                    selected.append(path)
                else:
                    break
            elif len(selected) < limit:
                selected.append(path)
            else:
                replacement = int(rng.integers(0, seen))
                if replacement < limit:
                    selected[replacement] = path
    # Sorting does not change the random subset and makes I/O/progress reproducible.
    return sorted(selected), seen


def select_input_paths(args: argparse.Namespace) -> tuple[list[str], dict[str, object]]:
    root = Path(args.data_root)
    manifest = root / "manifest.csv"
    if manifest.exists():
        paths, available = _reservoir_manifest_paths(
            manifest,
            split=args.split,
            limit=args.max_focus_samples,
            seed=args.seed,
            selection=args.selection,
        )
        return paths, {
            "source": "manifest",
            "manifest": str(manifest),
            "available_focus_samples_seen": available,
            "selection": args.selection,
        }

    split_root = root / args.split if (root / args.split).is_dir() else root
    all_paths = sorted(str(path) for path in split_root.glob("*.npz"))
    available = len(all_paths)
    if args.max_focus_samples > 0 and len(all_paths) > args.max_focus_samples:
        if args.selection == "random":
            rng = np.random.default_rng(args.seed)
            indices = np.sort(rng.choice(len(all_paths), size=args.max_focus_samples, replace=False))
            all_paths = [all_paths[int(i)] for i in indices]
        else:
            all_paths = all_paths[: args.max_focus_samples]
    return all_paths, {
        "source": "directory_glob",
        "directory": str(split_root),
        "available_focus_samples_seen": available,
        "selection": args.selection,
    }


def _normalise_histories(histories: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = histories[eligible] if bool(eligible.any()) else histories
    flat = fit.reshape(-1, fit.shape[-1])
    median = np.median(flat, axis=0).astype(np.float32)
    q25, q75 = np.percentile(flat, [25.0, 75.0], axis=0)
    scale = np.maximum((q75 - q25).astype(np.float32), 1e-3)
    normalised = np.clip((histories - median[None, None]) / scale[None, None], -8.0, 8.0)
    # The history is aligned in event-relative time; recent observations receive
    # somewhat larger weight without warping the time axis.
    time_weight = np.linspace(0.5, 1.5, histories.shape[1], dtype=np.float32)
    weighted = normalised * np.sqrt(time_weight[None, :, None])
    # Duplicate the query state once so the endpoint is a strict part of matching.
    vectors = np.concatenate([weighted.reshape(len(histories), -1), normalised[:, -1]], axis=1)
    return vectors.astype(np.float32), median, scale


def _different_scene_neighbours(
    neighbour_indices: np.ndarray,
    neighbour_distances: np.ndarray,
    anchor: int,
    scene_ids: np.ndarray,
) -> Iterable[tuple[int, float]]:
    for other, distance in zip(np.atleast_1d(neighbour_indices), np.atleast_1d(neighbour_distances)):
        other = int(other)
        if other == anchor or scene_ids[other] == scene_ids[anchor]:
            continue
        yield other, float(distance)


def build_matches(
    vectors: np.ndarray,
    *,
    eligible: np.ndarray,
    scene_ids: np.ndarray,
    lead_steps: np.ndarray,
    relation: np.ndarray,
    response: np.ndarray,
    focus_type: np.ndarray,
    candidate_type: np.ndarray,
    max_positives: int,
    max_hard_negatives: int,
    max_negatives: int,
    search_k: int,
    caliper_quantile: float,
    caliper_multiplier: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    n = int(len(vectors))
    positive = np.full((n, max_positives), -1, dtype=np.int64)
    positive_distance = np.full((n, max_positives), np.inf, dtype=np.float32)
    hard = np.full((n, max_hard_negatives), -1, dtype=np.int64)
    hard_distance = np.full((n, max_hard_negatives), np.inf, dtype=np.float32)
    negative = np.full((n, max_negatives), -1, dtype=np.int64)
    negative_distance = np.full((n, max_negatives), np.inf, dtype=np.float32)

    strata: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for idx in np.flatnonzero(eligible):
        key = (int(lead_steps[idx]), int(relation[idx]), int(focus_type[idx]), int(candidate_type[idx]))
        strata[key].append(int(idx))

    calipers: dict[str, float] = {}
    stratum_sizes: dict[str, int] = {}
    nearest_distances: list[float] = []
    for key, members_list in strata.items():
        members = np.asarray(members_list, dtype=np.int64)
        name = "/".join(map(str, key))
        stratum_sizes[name] = int(len(members))
        if len(members) < 2:
            continue
        tree = cKDTree(vectors[members])
        k = min(len(members), max(2, search_k + 1))
        distances, local_indices = tree.query(vectors[members], k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            local_indices = local_indices[:, None]

        base: list[float] = []
        for row, anchor in enumerate(members):
            global_neighbours = members[np.asarray(local_indices[row], dtype=np.int64)]
            valid = list(_different_scene_neighbours(global_neighbours, distances[row], int(anchor), scene_ids))
            if valid:
                base.append(valid[0][1])
                nearest_distances.append(valid[0][1])
        if not base:
            continue
        caliper = float(np.quantile(np.asarray(base), caliper_quantile) * caliper_multiplier)
        calipers[name] = caliper

        for row, anchor in enumerate(members):
            global_neighbours = members[np.asarray(local_indices[row], dtype=np.int64)]
            p_count = 0
            h_count = 0
            for other, distance in _different_scene_neighbours(
                global_neighbours, distances[row], int(anchor), scene_ids
            ):
                if distance > caliper:
                    break
                if response[other] == response[anchor] and p_count < max_positives:
                    positive[anchor, p_count] = other
                    positive_distance[anchor, p_count] = distance
                    p_count += 1
                elif response[other] != response[anchor] and h_count < max_hard_negatives:
                    hard[anchor, h_count] = other
                    hard_distance[anchor, h_count] = distance
                    h_count += 1
                if p_count >= max_positives and h_count >= max_hard_negatives:
                    break

    # Ordinary/easy negatives differ in relation type at the same lead time.
    # They are sampled separately and never used to define the hard-match caliper.
    rng = np.random.default_rng(seed)
    lead_pools = {int(lead): np.flatnonzero(eligible & (lead_steps == lead)) for lead in np.unique(lead_steps)}
    for anchor in np.flatnonzero(eligible):
        pool = lead_pools[int(lead_steps[anchor])]
        if len(pool) == 0:
            continue
        # Inspect a bounded random subset instead of permuting a potentially
        # large lead-time pool once per anchor.
        attempts = min(len(pool), max(64, max_negatives * 32))
        order = rng.choice(pool, size=attempts, replace=False)
        count = 0
        for other in order:
            other = int(other)
            if scene_ids[other] == scene_ids[anchor] or relation[other] == relation[anchor]:
                continue
            negative[anchor, count] = other
            negative_distance[anchor, count] = float(np.linalg.norm(vectors[anchor] - vectors[other]))
            count += 1
            if count >= max_negatives:
                break

    has_positive = (positive >= 0).any(axis=1)
    has_hard = (hard >= 0).any(axis=1)
    trainable = eligible & has_positive & has_hard
    nearest = np.asarray(nearest_distances, dtype=np.float32)
    stats: dict[str, object] = {
        "num_strata": len(strata),
        "stratum_sizes": stratum_sizes,
        "calipers": calipers,
        "anchors_with_positive": int((eligible & has_positive).sum()),
        "anchors_with_hard_negative": int((eligible & has_hard).sum()),
        "trainable_anchors": int(trainable.sum()),
        "positive_edges": int((positive >= 0).sum()),
        "hard_negative_edges": int((hard >= 0).sum()),
        "negative_edges": int((negative >= 0).sum()),
        "different_scene_nearest_distance_quantiles": (
            {}
            if nearest.size == 0
            else {
                str(q): float(np.quantile(nearest, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
            }
        ),
    }
    arrays = {
        "positive_indices": positive,
        "positive_distances": positive_distance,
        "hard_negative_indices": hard,
        "hard_negative_distances": hard_distance,
        "negative_indices": negative,
        "negative_distances": negative_distance,
        "trainable_anchor_mask": trainable,
    }
    return arrays, stats


def validate_matches(
    sample_arrays: dict[str, np.ndarray], match_arrays: dict[str, np.ndarray]
) -> dict[str, int]:
    """Fail fast if a stored edge violates its semantic definition."""
    checks = {"positive_edges_checked": 0, "hard_negative_edges_checked": 0, "negative_edges_checked": 0}
    scene = sample_arrays["scenario_id"]
    lead = sample_arrays["lead_steps"]
    relation = sample_arrays["relation_index"]
    response = sample_arrays["response_index"]
    focus_type = sample_arrays["focus_type"]
    candidate_type = sample_arrays["candidate_type"]
    for anchor in range(len(scene)):
        for other in match_arrays["positive_indices"][anchor]:
            if other < 0:
                continue
            other = int(other)
            assert scene[anchor] != scene[other]
            assert lead[anchor] == lead[other]
            assert relation[anchor] == relation[other]
            assert focus_type[anchor] == focus_type[other]
            assert candidate_type[anchor] == candidate_type[other]
            assert response[anchor] == response[other]
            checks["positive_edges_checked"] += 1
        for other in match_arrays["hard_negative_indices"][anchor]:
            if other < 0:
                continue
            other = int(other)
            assert scene[anchor] != scene[other]
            assert lead[anchor] == lead[other]
            assert relation[anchor] == relation[other]
            assert focus_type[anchor] == focus_type[other]
            assert candidate_type[anchor] == candidate_type[other]
            assert response[anchor] != response[other]
            checks["hard_negative_edges_checked"] += 1
        for other in match_arrays["negative_indices"][anchor]:
            if other < 0:
                continue
            other = int(other)
            assert scene[anchor] != scene[other]
            assert lead[anchor] == lead[other]
            assert relation[anchor] != relation[other]
            checks["negative_edges_checked"] += 1
    return checks


def _count_names(values: np.ndarray, names: tuple[str, ...], mask: np.ndarray | None = None) -> dict[str, int]:
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    return {name: int(((values == idx) & mask).sum()) for idx, name in enumerate(names)}


def _write_sample_csv(path: Path, arrays: dict[str, np.ndarray], matches: dict[str, np.ndarray]) -> None:
    fields = (
        "sample_index",
        "scenario_id",
        "source_path",
        "focus_agent_id",
        "candidate_agent_id",
        "candidate_index",
        "event_step",
        "query_step",
        "lead_steps",
        "relation",
        "response",
        "eligible",
        "delta_arrival_time_s",
        "pet_s",
        "spatial_min_dist_m",
        "num_positives",
        "num_hard_negatives",
        "num_negatives",
        "trainable_anchor",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx in range(len(arrays["query_step"])):
            relation_idx = int(arrays["relation_index"][idx])
            response_idx = int(arrays["response_index"][idx])
            writer.writerow(
                {
                    "sample_index": idx,
                    "scenario_id": arrays["scenario_id"][idx],
                    "source_path": arrays["source_path"][idx],
                    "focus_agent_id": int(arrays["focus_agent_id"][idx]),
                    "candidate_agent_id": int(arrays["candidate_agent_id"][idx]),
                    "candidate_index": int(arrays["candidate_index"][idx]),
                    "event_step": int(arrays["event_step"][idx]),
                    "query_step": int(arrays["query_step"][idx]),
                    "lead_steps": int(arrays["lead_steps"][idx]),
                    "relation": RELATION_NAMES[relation_idx],
                    "response": RESPONSE_NAMES[response_idx] if response_idx >= 0 else "ambiguous",
                    "eligible": int(arrays["eligible"][idx]),
                    "delta_arrival_time_s": float(arrays["delta_arrival_time_s"][idx]),
                    "pet_s": float(arrays["pet_s"][idx]),
                    "spatial_min_dist_m": float(arrays["spatial_min_dist_m"][idx]),
                    "num_positives": int((matches["positive_indices"][idx] >= 0).sum()),
                    "num_hard_negatives": int((matches["hard_negative_indices"][idx] >= 0).sum()),
                    "num_negatives": int((matches["negative_indices"][idx] >= 0).sum()),
                    "trainable_anchor": int(matches["trainable_anchor_mask"][idx]),
                }
            )


def build(args: argparse.Namespace) -> None:
    paths, selection_metadata = select_input_paths(args)
    if not paths:
        raise RuntimeError("No input NPZ files selected")
    cfg = SampleConfig(
        dt=args.dt,
        event_search_start=args.event_search_start,
        history_steps=args.history_steps,
        lead_steps=args.lead_steps,
        path_overlap_dist_m=args.path_overlap_dist_m,
        pet_relevant_s=args.pet_relevant_s,
        crossing_heading_deg=args.crossing_heading_deg,
        same_direction_deg=args.same_direction_deg,
        same_corridor_lateral_m=args.same_corridor_lateral_m,
        following_headway_m=args.following_headway_m,
        speed_drop_mps=args.speed_drop_mps,
        decel_mps2=args.decel_mps2,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    histories: list[np.ndarray] = []
    rows: dict[str, list[object]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    processed = 0
    started = time.time()
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                agents = np.asarray(data["agents"], dtype=np.float32)
                agent_mask = np.asarray(data["agent_mask"], dtype=bool)
                agent_ids = np.asarray(data["agent_ids"], dtype=np.int64)
                scenario_id = str(data["scenario_id"]) if "scenario_id" in data else Path(path).stem
                samples = build_scene_samples(agents, agent_mask, cfg)
                for sample in samples:
                    histories.append(sample.history)
                    rows["scenario_id"].append(scenario_id)
                    rows["source_path"].append(path)
                    rows["focus_agent_id"].append(int(agent_ids[0]))
                    rows["candidate_agent_id"].append(int(agent_ids[sample.candidate_index]))
                    rows["candidate_index"].append(sample.candidate_index)
                    rows["event_step"].append(sample.event_step)
                    rows["query_step"].append(sample.query_step)
                    rows["lead_steps"].append(sample.lead_steps)
                    rows["relation_index"].append(sample.relation_index)
                    rows["response_index"].append(sample.response_index)
                    rows["eligible"].append(sample.eligible)
                    rows["focus_type"].append(sample.focus_type)
                    rows["candidate_type"].append(sample.candidate_type)
                    rows["delta_arrival_time_s"].append(sample.delta_arrival_time_s)
                    rows["pet_s"].append(sample.pet_s)
                    rows["spatial_min_dist_m"].append(sample.spatial_min_dist_m)
        except Exception as exc:  # Preserve the rest of a large offline scan.
            failures.append({"path": path, "error": repr(exc)})
            if len(failures) <= 5:
                print(f"warning: failed {path}: {exc!r}", flush=True)
        processed += 1
        if processed % args.log_every == 0:
            print(
                f"processed={processed}/{len(paths)} samples={len(histories)} failures={len(failures)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if not histories:
        raise RuntimeError("No event-aligned pair samples were found")

    arrays: dict[str, np.ndarray] = {
        "history": np.stack(histories).astype(np.float32),
        "scenario_id": np.asarray(rows["scenario_id"]),
        "source_path": np.asarray(rows["source_path"]),
        "focus_agent_id": np.asarray(rows["focus_agent_id"], dtype=np.int64),
        "candidate_agent_id": np.asarray(rows["candidate_agent_id"], dtype=np.int64),
        "candidate_index": np.asarray(rows["candidate_index"], dtype=np.int16),
        "event_step": np.asarray(rows["event_step"], dtype=np.int16),
        "query_step": np.asarray(rows["query_step"], dtype=np.int16),
        "lead_steps": np.asarray(rows["lead_steps"], dtype=np.int16),
        "relation_index": np.asarray(rows["relation_index"], dtype=np.int8),
        "response_index": np.asarray(rows["response_index"], dtype=np.int8),
        "eligible": np.asarray(rows["eligible"], dtype=bool),
        "focus_type": np.asarray(rows["focus_type"], dtype=np.int8),
        "candidate_type": np.asarray(rows["candidate_type"], dtype=np.int8),
        "delta_arrival_time_s": np.asarray(rows["delta_arrival_time_s"], dtype=np.float32),
        "pet_s": np.asarray(rows["pet_s"], dtype=np.float32),
        "spatial_min_dist_m": np.asarray(rows["spatial_min_dist_m"], dtype=np.float32),
        "history_feature_names": np.asarray(HISTORY_FEATURE_NAMES),
        "relation_names": np.asarray(RELATION_NAMES),
        "response_names": np.asarray(RESPONSE_NAMES),
    }
    vectors, median, scale = _normalise_histories(arrays["history"], arrays["eligible"])
    arrays["matching_vector"] = vectors.astype(np.float16)
    arrays["normalization_median"] = median
    arrays["normalization_iqr"] = scale

    matches, match_stats = build_matches(
        vectors,
        eligible=arrays["eligible"],
        scene_ids=arrays["scenario_id"],
        lead_steps=arrays["lead_steps"],
        relation=arrays["relation_index"],
        response=arrays["response_index"],
        focus_type=arrays["focus_type"],
        candidate_type=arrays["candidate_type"],
        max_positives=args.max_positives,
        max_hard_negatives=args.max_hard_negatives,
        max_negatives=args.max_negatives,
        search_k=args.search_k,
        caliper_quantile=args.caliper_quantile,
        caliper_multiplier=args.caliper_multiplier,
        seed=args.seed,
    )
    validation_stats = validate_matches(arrays, matches)

    sample_path = output_dir / f"{args.split}_samples.npz"
    match_path = output_dir / f"{args.split}_matches.npz"
    np.savez_compressed(sample_path, **arrays)
    np.savez_compressed(match_path, **matches)
    _write_sample_csv(output_dir / f"{args.split}_samples.csv", arrays, matches)

    eligible = arrays["eligible"]
    summary = {
        "split": args.split,
        "data_root": args.data_root,
        "selected_focus_samples": len(paths),
        "processed_focus_samples": processed,
        "failed_focus_samples": len(failures),
        "num_pair_samples": int(len(histories)),
        "eligible_pair_samples": int(eligible.sum()),
        "unique_scenarios_with_samples": int(len(set(arrays["scenario_id"].tolist()))),
        "selection": selection_metadata,
        "sample_config": asdict(cfg),
        "matching_config": {
            "metric": "robust-standardized time-aligned weighted euclidean",
            "dtw": False,
            "stratum": ["lead_steps", "relation", "focus_type", "candidate_type"],
            "search_k": args.search_k,
            "caliper_quantile": args.caliper_quantile,
            "caliper_multiplier": args.caliper_multiplier,
            "max_positives": args.max_positives,
            "max_hard_negatives": args.max_hard_negatives,
            "max_negatives": args.max_negatives,
        },
        "relation_counts_all": _count_names(arrays["relation_index"], RELATION_NAMES),
        "relation_counts_eligible": _count_names(arrays["relation_index"], RELATION_NAMES, eligible),
        "response_counts_eligible": _count_names(arrays["response_index"], RESPONSE_NAMES, eligible),
        "lead_step_counts_all": dict(Counter(map(str, arrays["lead_steps"].tolist()))),
        "lead_step_counts_eligible": dict(Counter(map(str, arrays["lead_steps"][eligible].tolist()))),
        "matches": match_stats,
        "match_validation": validation_stats,
        "outputs": {
            "samples_npz": str(sample_path),
            "matches_npz": str(match_path),
            "samples_csv": str(output_dir / f"{args.split}_samples.csv"),
        },
        "elapsed_seconds": time.time() - started,
        "failures_preview": failures[:20],
    }
    summary_path = output_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"saved {sample_path}", flush=True)
    print(f"saved {match_path}", flush=True)
    print(f"saved {summary_path}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default="/p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset_ooi_centered_training_all",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--output_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_contrastive_learning_5k",
    )
    parser.add_argument("--max_focus_samples", type=int, default=5000)
    parser.add_argument("--selection", choices=("random", "first"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)

    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--event_search_start", type=int, default=10)
    parser.add_argument("--history_steps", type=int, default=20)
    parser.add_argument("--lead_steps", type=_parse_int_tuple, default=(10, 20, 30))
    parser.add_argument("--path_overlap_dist_m", type=float, default=4.0)
    parser.add_argument("--pet_relevant_s", type=float, default=3.0)
    parser.add_argument("--crossing_heading_deg", type=float, default=60.0)
    parser.add_argument("--same_direction_deg", type=float, default=45.0)
    parser.add_argument("--same_corridor_lateral_m", type=float, default=4.5)
    parser.add_argument("--following_headway_m", type=float, default=20.0)
    parser.add_argument("--speed_drop_mps", type=float, default=1.5)
    parser.add_argument("--decel_mps2", type=float, default=1.0)

    parser.add_argument("--search_k", type=int, default=128)
    parser.add_argument("--caliper_quantile", type=float, default=0.9)
    parser.add_argument("--caliper_multiplier", type=float, default=1.5)
    parser.add_argument("--max_positives", type=int, default=2)
    parser.add_argument("--max_hard_negatives", type=int, default=4)
    parser.add_argument("--max_negatives", type=int, default=4)
    return parser


if __name__ == "__main__":
    build(build_argparser().parse_args())
