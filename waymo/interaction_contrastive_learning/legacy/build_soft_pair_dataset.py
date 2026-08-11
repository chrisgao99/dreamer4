"""Build 60-step soft-contrastive interaction pairs and neighbour caches."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.spatial import cKDTree

try:
    from .soft_pair_samples import PAIR_FEATURE_NAMES, SoftPairConfig, build_focus_samples
except ImportError:
    from soft_pair_samples import PAIR_FEATURE_NAMES, SoftPairConfig, build_focus_samples  # type: ignore


def read_manifest(data_root: Path, splits: tuple[str, ...], limit: int) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {split: [] for split in splits}
    manifest = data_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", "")
            if split not in paths:
                continue
            if limit > 0 and sum(map(len, paths.values())) >= limit:
                break
            path = row.get("npz_path", "")
            if path:
                paths[split].append(path)
    return paths


def _dedupe_key(
    scenario_id: str,
    first_agent_id: int,
    second_agent_id: int,
    first_arrival: int,
    second_arrival: int,
) -> tuple[object, ...]:
    first = (int(first_agent_id), int(first_arrival))
    second = (int(second_agent_id), int(second_arrival))
    low, high = sorted((first, second))
    return (scenario_id, low[0], high[0], low[1], high[1])


def extract_split(
    paths: list[str],
    *,
    split: str,
    cfg: SoftPairConfig,
    log_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    sequences: list[np.ndarray] = []
    rows: dict[str, list[object]] = defaultdict(list)
    seen: set[tuple[object, ...]] = set()
    stats = Counter()
    failures: list[dict[str, str]] = []
    started = time.time()

    for file_index, path in enumerate(paths, start=1):
        try:
            with np.load(path, allow_pickle=False) as data:
                agents = np.asarray(data["agents"], dtype=np.float32)
                agent_mask = np.asarray(data["agent_mask"], dtype=bool)
                agent_ids = np.asarray(data["agent_ids"], dtype=np.int64)
                scenario_id = str(data["scenario_id"]) if "scenario_id" in data else Path(path).stem
                stats["candidate_focus_pairs"] += max(0, int(agent_mask.sum()) - 1)
                samples = build_focus_samples(agents, agent_mask, cfg)
                stats["retained_before_dedup"] += len(samples)
                for sample in samples:
                    first_agent_id = int(agent_ids[sample.first_index])
                    second_agent_id = int(agent_ids[sample.second_index])
                    key = _dedupe_key(
                        scenario_id,
                        first_agent_id,
                        second_agent_id,
                        sample.first_arrival_step,
                        sample.second_arrival_step,
                    )
                    if key in seen:
                        stats["duplicate_pairs"] += 1
                        continue
                    seen.add(key)
                    sequences.append(sample.sequence)
                    rows["scenario_id"].append(scenario_id)
                    rows["source_path"].append(path)
                    rows["first_agent_id"].append(first_agent_id)
                    rows["second_agent_id"].append(second_agent_id)
                    rows["first_agent_index"].append(sample.first_index)
                    rows["second_agent_index"].append(sample.second_index)
                    rows["first_agent_type"].append(int(round(float(agents[sample.first_index, sample.first_step, 7]))))
                    rows["second_agent_type"].append(int(round(float(agents[sample.second_index, sample.first_step, 7]))))
                    rows["first_arrival_step"].append(sample.first_arrival_step)
                    rows["second_arrival_step"].append(sample.second_arrival_step)
                    rows["pet_steps"].append(sample.pet_steps)
                    rows["pet_s"].append(sample.pet_steps * cfg.dt)
                    rows["spatial_min_dist_m"].append(sample.spatial_min_dist_m)
                    rows["relevance_score"].append(sample.relevance_score)
                    rows["conflict_x"].append(float(sample.conflict_xy[0]))
                    rows["conflict_y"].append(float(sample.conflict_xy[1]))
        except Exception as exc:
            failures.append({"path": path, "error": repr(exc)})
            if len(failures) <= 10:
                print(f"warning [{split}]: failed {path}: {exc!r}", flush=True)
        if log_every > 0 and file_index % log_every == 0:
            print(
                f"[{split}] files={file_index}/{len(paths)} pairs={len(sequences)} "
                f"duplicates={stats['duplicate_pairs']} failures={len(failures)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if not sequences:
        raise RuntimeError(f"No soft pair samples extracted for split={split}")
    arrays: dict[str, np.ndarray] = {
        "sequence": np.stack(sequences).astype(np.float32),
        "scenario_id": np.asarray(rows["scenario_id"]),
        "source_path": np.asarray(rows["source_path"]),
        "first_agent_id": np.asarray(rows["first_agent_id"], dtype=np.int64),
        "second_agent_id": np.asarray(rows["second_agent_id"], dtype=np.int64),
        "first_agent_index": np.asarray(rows["first_agent_index"], dtype=np.int16),
        "second_agent_index": np.asarray(rows["second_agent_index"], dtype=np.int16),
        "first_agent_type": np.asarray(rows["first_agent_type"], dtype=np.int8),
        "second_agent_type": np.asarray(rows["second_agent_type"], dtype=np.int8),
        "first_arrival_step": np.asarray(rows["first_arrival_step"], dtype=np.int16),
        "second_arrival_step": np.asarray(rows["second_arrival_step"], dtype=np.int16),
        "pet_steps": np.asarray(rows["pet_steps"], dtype=np.int16),
        "pet_s": np.asarray(rows["pet_s"], dtype=np.float32),
        "spatial_min_dist_m": np.asarray(rows["spatial_min_dist_m"], dtype=np.float32),
        "relevance_score": np.asarray(rows["relevance_score"], dtype=np.float32),
        "conflict_x": np.asarray(rows["conflict_x"], dtype=np.float32),
        "conflict_y": np.asarray(rows["conflict_y"], dtype=np.float32),
        "feature_names": np.asarray(PAIR_FEATURE_NAMES),
        "time_offsets": np.arange(-cfg.history_steps + 1, cfg.post_first_steps + 1, dtype=np.int16),
    }
    summary: dict[str, object] = {
        "split": split,
        "focus_files": len(paths),
        "failed_focus_files": len(failures),
        "candidate_focus_pairs": int(stats["candidate_focus_pairs"]),
        "retained_before_dedup": int(stats["retained_before_dedup"]),
        "duplicate_pairs": int(stats["duplicate_pairs"]),
        "retained_pair_samples": len(sequences),
        "unique_scenarios": len(set(rows["scenario_id"])),
        "failures_preview": failures[:20],
        "elapsed_seconds": time.time() - started,
    }
    return arrays, summary


def fit_robust_scaler(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(sequence, dtype=np.float32).reshape(-1, sequence.shape[-1])
    median = np.median(flat, axis=0).astype(np.float32)
    q25, q75 = np.percentile(flat, [25.0, 75.0], axis=0)
    iqr = (q75 - q25).astype(np.float32)
    # Straight-driving data can have an almost-zero lateral-velocity or
    # cos(yaw) IQR.  Do not turn sensor noise in those channels into the main
    # trajectory distance.  These are global physical floors, never per-sample
    # normalization; heading sin/cos remain in their native [-1, 1] geometry.
    position_channels = np.asarray([0, 1, 6, 7])
    velocity_channels = np.asarray([2, 3, 8, 9])
    heading_channels = np.asarray([4, 5, 10, 11])
    iqr[position_channels] = np.maximum(iqr[position_channels], 1.0)
    iqr[velocity_channels] = np.maximum(iqr[velocity_channels], 1.0)
    median[heading_channels] = 0.0
    iqr[heading_channels] = 1.0
    return median, iqr


def normalise_sequence(sequence: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    return ((np.asarray(sequence, dtype=np.float32) - median[None, None]) / iqr[None, None]).astype(np.float32)


def retrieval_descriptor(normalised: np.ndarray, dct_coefficients: int) -> np.ndarray:
    coefficients = dct(normalised, axis=1, norm="ortho")[:, :dct_coefficients]
    return coefficients.reshape(len(normalised), -1).astype(np.float32)


def build_soft_neighbours(
    normalised: np.ndarray,
    scenario_ids: np.ndarray,
    *,
    num_neighbours: int,
    retrieval_candidates: int,
    dct_coefficients: int,
    local_scale_k: int,
    query_chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    n = int(len(normalised))
    descriptor = retrieval_descriptor(normalised, dct_coefficients)
    tree = cKDTree(descriptor)
    stored = min(num_neighbours, max(0, n - 1))
    indices = np.full((n, num_neighbours), -1, dtype=np.int64)
    distances = np.full((n, num_neighbours), np.inf, dtype=np.float32)
    query_k = min(n, max(stored + 1, retrieval_candidates + 1))
    started = time.time()

    for start in range(0, n, query_chunk_size):
        end = min(n, start + query_chunk_size)
        _, candidate_local = tree.query(descriptor[start:end], k=query_k, workers=-1)
        candidate_local = np.atleast_2d(candidate_local)
        for row, anchor in enumerate(range(start, end)):
            candidates = np.asarray(candidate_local[row], dtype=np.int64)
            candidates = candidates[(candidates != anchor) & (scenario_ids[candidates] != scenario_ids[anchor])]
            if not len(candidates):
                continue
            diff = normalised[candidates] - normalised[anchor][None]
            exact = np.sqrt(np.mean(diff * diff, axis=(1, 2)))
            order = np.argsort(exact)[:stored]
            selected = candidates[order]
            count = len(selected)
            indices[anchor, :count] = selected
            distances[anchor, :count] = exact[order]
        print(f"[neighbours] {end}/{n} elapsed={time.time() - started:.1f}s", flush=True)

    scale_column = max(0, min(local_scale_k, num_neighbours) - 1)
    local_scale = distances[:, scale_column].copy()
    valid_scale = np.isfinite(local_scale) & (local_scale > 1e-6)
    fallback = float(np.median(local_scale[valid_scale])) if bool(valid_scale.any()) else 1.0
    local_scale[~valid_scale] = fallback
    similarities = np.zeros_like(distances)
    for anchor in range(n):
        valid = indices[anchor] >= 0
        if not bool(valid.any()):
            continue
        neighbours = indices[anchor, valid]
        denominator = np.maximum(local_scale[anchor] * local_scale[neighbours], 1e-6)
        similarities[anchor, valid] = np.exp(-(distances[anchor, valid] ** 2) / denominator)

    arrays = {
        "neighbor_indices": indices,
        "sequence_distances": distances,
        "soft_similarity": similarities,
        "local_scale": local_scale.astype(np.float32),
    }
    stats: dict[str, object] = {
        "num_samples": n,
        "num_neighbours": num_neighbours,
        "retrieval_candidates": retrieval_candidates,
        "dct_coefficients_per_channel": dct_coefficients,
        "retrieval_dimension": int(descriptor.shape[1]),
        "local_scale_k": local_scale_k,
        "local_scale_fallback": fallback,
        "samples_with_neighbours": int((indices >= 0).any(axis=1).sum()),
        "stored_edges": int((indices >= 0).sum()),
        "elapsed_seconds": time.time() - started,
    }
    return arrays, stats


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {str(q): float(np.quantile(values, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}


def write_metadata_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    fields = (
        "sample_index", "scenario_id", "source_path", "first_agent_id", "second_agent_id",
        "first_agent_type", "second_agent_type", "first_arrival_step", "second_arrival_step",
        "pet_steps", "pet_s", "spatial_min_dist_m", "relevance_score", "conflict_x", "conflict_y",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(arrays["scenario_id"])):
            writer.writerow({
                field: index if field == "sample_index" else arrays[field][index].item()
                for field in fields
            })


def build(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    cfg = SoftPairConfig(
        dt=args.dt,
        event_search_start=args.event_search_start,
        history_steps=args.history_steps,
        post_first_steps=args.post_first_steps,
        max_pet_steps=args.max_pet_steps,
        max_spatial_distance_m=args.max_spatial_distance_m,
        relevance_distance_scale_m=args.relevance_distance_scale_m,
        relevance_pet_scale_s=args.relevance_pet_scale_s,
    )
    if cfg.sequence_steps != 60:
        raise ValueError(f"Version one requires exactly 60 steps, got {cfg.sequence_steps}")
    paths_by_split = read_manifest(Path(args.data_root), splits, args.max_focus_samples)

    extracted: dict[str, dict[str, np.ndarray]] = {}
    split_summaries: dict[str, dict[str, object]] = {}
    for split in splits:
        arrays, summary = extract_split(
            paths_by_split[split], split=split, cfg=cfg, log_every=args.log_every
        )
        extracted[split] = arrays
        split_summaries[split] = summary

    scaler_split = "train" if "train" in extracted else splits[0]
    median, iqr = fit_robust_scaler(extracted[scaler_split]["sequence"])
    for split in splits:
        arrays = extracted[split]
        normalised = normalise_sequence(arrays["sequence"], median, iqr)
        arrays["normalized_sequence"] = normalised.astype(np.float16)
        arrays["normalization_median"] = median
        arrays["normalization_iqr"] = iqr
        neighbours, neighbour_stats = build_soft_neighbours(
            normalised,
            arrays["scenario_id"],
            num_neighbours=args.num_neighbours,
            retrieval_candidates=args.retrieval_candidates,
            dct_coefficients=args.dct_coefficients,
            local_scale_k=args.local_scale_k,
            query_chunk_size=args.query_chunk_size,
        )
        sample_path = output_dir / f"{split}_samples.npz"
        match_path = output_dir / f"{split}_soft_neighbors.npz"
        np.savez_compressed(sample_path, **arrays)
        np.savez_compressed(match_path, **neighbours)
        write_metadata_csv(output_dir / f"{split}_samples.csv", arrays)
        split_summaries[split]["neighbours"] = neighbour_stats
        split_summaries[split]["pet_steps_quantiles"] = _quantiles(arrays["pet_steps"])
        split_summaries[split]["spatial_distance_quantiles_m"] = _quantiles(arrays["spatial_min_dist_m"])
        split_summaries[split]["relevance_score_quantiles"] = _quantiles(arrays["relevance_score"])
        split_summaries[split]["neighbor_distance_quantiles"] = _quantiles(neighbours["sequence_distances"])
        split_summaries[split]["soft_similarity_quantiles"] = _quantiles(neighbours["soft_similarity"][neighbours["neighbor_indices"] >= 0])
        split_summaries[split]["outputs"] = {
            "samples_npz": str(sample_path),
            "soft_neighbors_npz": str(match_path),
            "samples_csv": str(output_dir / f"{split}_samples.csv"),
        }
        del normalised

    summary = {
        "version": "shared_time_axis_soft_pairs_v1",
        "data_root": args.data_root,
        "splits": list(splits),
        "config": asdict(cfg),
        "feature_names": list(PAIR_FEATURE_NAMES),
        "normalization": {
            "fit_split": scaler_split,
            "method": "global per-channel median/IQR over all samples and timesteps; position/velocity scales have 1-unit floors and heading sin/cos remain native",
            "per_sample": False,
            "clipping": False,
            "median": median.tolist(),
            "iqr": iqr.tolist(),
        },
        "similarity": {
            "distance": "RMS Euclidean over the complete normalized 60x12 sequence",
            "dtw": False,
            "candidate_retrieval": "low-frequency DCT descriptor followed by exact full-sequence reranking",
            "different_scenario_only": True,
            "soft_kernel": "exp(-distance^2 / (local_scale_i * local_scale_j))",
        },
        "split_summaries": split_summaries,
        "total_focus_files": sum(len(paths) for paths in paths_by_split.values()),
        "total_retained_pair_samples": sum(int(value["retained_pair_samples"]) for value in split_summaries.values()),
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"saved {summary_path}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default="/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_ooi_centered_50k",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_soft_pairs_50k_v1",
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--max_focus_samples", type=int, default=0, help="0 uses all requested splits")
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--event_search_start", type=int, default=10)
    parser.add_argument("--history_steps", type=int, default=20)
    parser.add_argument("--post_first_steps", type=int, default=40)
    parser.add_argument("--max_pet_steps", type=int, default=30)
    parser.add_argument("--max_spatial_distance_m", type=float, default=6.0)
    parser.add_argument("--relevance_distance_scale_m", type=float, default=3.0)
    parser.add_argument("--relevance_pet_scale_s", type=float, default=1.5)
    parser.add_argument("--num_neighbours", type=int, default=32)
    parser.add_argument("--retrieval_candidates", type=int, default=256)
    parser.add_argument("--dct_coefficients", type=int, default=3)
    parser.add_argument("--local_scale_k", type=int, default=20)
    parser.add_argument("--query_chunk_size", type=int, default=2048)
    return parser


if __name__ == "__main__":
    build(build_argparser().parse_args())
