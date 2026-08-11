"""Build event-aligned, mask-aware RMS neighbours for full-pair v2 shards.

This is the first similarity baseline for the 91-step physical-contact pair
dataset.  It performs four operations:

1. align every sample to the first-arrival event time;
2. fit train-only robust feature scaling and apply it to every split;
3. retrieve coarse candidates with low-frequency DCT descriptors and cKDTree;
4. rerank candidates with exact mask-aware RMS over the complete aligned state.

DCT distances are never written as trajectory similarity.  Stored neighbour
distances are always exact RMS values over the normalized 60x2x6 sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.spatial import cKDTree


FEATURE_NAMES = ("x_m", "y_m", "vx_mps", "vy_mps", "heading_sin", "heading_cos")


@dataclass(frozen=True)
class RmsConfig:
    history_steps: int = 19
    future_steps: int = 40
    min_valid_fraction: float = 0.80
    min_pair_overlap: float = 0.70
    dct_coefficients: int = 3
    mask_descriptor_weight: float = 0.25
    # 1,024 gave 99%+ mean exact-top-32 recall in the 10k/split audit.
    retrieval_candidates: int = 1024
    retrieval_buffer: int = 64
    num_neighbours: int = 32
    query_chunk_size: int = 2048

    @property
    def sequence_steps(self) -> int:
        return self.history_steps + self.future_steps + 1

    @property
    def offsets(self) -> np.ndarray:
        return np.arange(-self.history_steps, self.future_steps + 1, dtype=np.float32)


def _split_shards(dataset_dir: Path, split: str) -> list[Path]:
    shards = sorted(dataset_dir.glob(f"{split}_samples_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No {split} shards found under {dataset_dir}")
    return shards


def align_trajectory_batch(
    trajectory: np.ndarray,
    valid_mask: np.ndarray,
    primary_step_first: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample a batch on a first-primary-step-relative time axis."""
    trajectory = np.asarray(trajectory, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    primary = np.asarray(primary_step_first, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    if trajectory.ndim != 4 or trajectory.shape[1:] != (2, 91, 6):
        raise ValueError(f"Expected trajectory (N,2,91,6), got {trajectory.shape}")
    if valid_mask.shape != trajectory.shape[:3]:
        raise ValueError(f"Expected valid mask {trajectory.shape[:3]}, got {valid_mask.shape}")
    if primary.shape != (len(trajectory),):
        raise ValueError(f"Expected primary steps ({len(trajectory)},), got {primary.shape}")

    n = len(trajectory)
    aligned = np.zeros((n, len(offsets), 2, 6), dtype=np.float32)
    aligned_mask = np.zeros((n, len(offsets), 2), dtype=bool)
    batch_index = np.arange(n)[:, None]
    agent_index = np.arange(2)[None, :]

    for target_step, offset in enumerate(offsets):
        raw = primary + float(offset)
        in_range = (raw >= 0.0) & (raw <= trajectory.shape[2] - 1.0)
        low = np.floor(np.clip(raw, 0.0, trajectory.shape[2] - 1.0)).astype(np.int64)
        high = np.minimum(low + 1, trajectory.shape[2] - 1)
        fraction = (raw - low).astype(np.float32)

        low_value = trajectory[batch_index, agent_index, low[:, None], :]
        high_value = trajectory[batch_index, agent_index, high[:, None], :]
        value = (1.0 - fraction[:, None, None]) * low_value + fraction[:, None, None] * high_value

        low_valid = valid_mask[batch_index, agent_index, low[:, None]]
        high_valid = valid_mask[batch_index, agent_index, high[:, None]]
        needs_high = fraction > 1e-6
        usable = in_range[:, None] & low_valid & (high_valid | ~needs_high[:, None])

        # Linear interpolation of sin/cos is followed by unit normalization.
        heading = value[:, :, 4:6]
        norm = np.linalg.norm(heading, axis=-1, keepdims=True)
        value[:, :, 4:6] = heading / np.maximum(norm, 1e-6)
        value[~usable] = 0.0
        aligned[:, target_step] = value
        aligned_mask[:, target_step] = usable
    return aligned, aligned_mask


def _take_prefix(array: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(array[:count])


def load_aligned_split(
    dataset_dir: Path,
    split: str,
    cfg: RmsConfig,
    *,
    max_samples: int = 0,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    remaining = int(max_samples)
    total = 0
    started = time.time()
    for shard_number, path in enumerate(_split_shards(dataset_dir, split), start=1):
        with np.load(path, allow_pickle=False) as data:
            count = len(data["trajectory"])
            if max_samples > 0:
                count = min(count, remaining)
            if count <= 0:
                break
            trajectory = _take_prefix(data["trajectory"], count).astype(np.float32, copy=False)
            valid_mask = _take_prefix(data["valid_mask"], count).astype(bool, copy=False)
            primary_first = _take_prefix(data["primary_step_first"], count).astype(np.float32, copy=False)
            aligned, mask = align_trajectory_batch(trajectory, valid_mask, primary_first, cfg.offsets)
            chunks["aligned_trajectory"].append(aligned)
            chunks["aligned_mask"].append(mask)
            for key in (
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
                "zone_pet_steps",
                "center_pet_steps",
                "min_clearance_m",
            ):
                chunks[key].append(_take_prefix(data[key], count))
        total += count
        if max_samples > 0:
            remaining -= count
        print(
            f"[{split}:align] shard={shard_number} samples={total} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        if max_samples > 0 and remaining <= 0:
            break

    if not chunks:
        raise RuntimeError(f"No samples loaded for split={split}")
    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    arrays["sample_index"] = np.arange(len(arrays["aligned_trajectory"]), dtype=np.int64)
    arrays["zone_pet_s"] = arrays["zone_pet_steps"].astype(np.float32) * 0.1
    arrays["center_pet_s"] = arrays["center_pet_steps"].astype(np.float32) * 0.1
    return arrays


def fit_robust_scaler(aligned: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a role-specific 2x6 train scaler without using invalid padding."""
    aligned = np.asarray(aligned, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    median = np.zeros((2, 6), dtype=np.float32)
    iqr = np.ones((2, 6), dtype=np.float32)
    for agent in range(2):
        usable = mask[:, :, agent]
        if not bool(usable.any()):
            raise ValueError(f"No valid values for agent role {agent}")
        for channel in range(4):
            values = aligned[:, :, agent, channel][usable]
            median[agent, channel] = np.median(values).astype(np.float32)
            q25, q75 = np.percentile(values, [25.0, 75.0])
            iqr[agent, channel] = max(float(q75 - q25), 1.0)
        # Heading sin/cos stay in their native unit-circle geometry.
        median[agent, 4:6] = 0.0
        iqr[agent, 4:6] = 1.0
    return median, iqr


def normalize_aligned(
    aligned: np.ndarray,
    mask: np.ndarray,
    median: np.ndarray,
    iqr: np.ndarray,
) -> np.ndarray:
    normalized = (np.asarray(aligned, dtype=np.float32) - median[None, None]) / iqr[None, None]
    normalized[~np.asarray(mask, dtype=bool)] = 0.0
    if not bool(np.isfinite(normalized).all()):
        raise ValueError("Normalized aligned trajectories contain non-finite values")
    return normalized.astype(np.float32, copy=False)


def retrieval_descriptor(
    normalized: np.ndarray,
    mask: np.ndarray,
    *,
    coefficients: int,
    mask_weight: float,
) -> np.ndarray:
    n, steps, _, _ = normalized.shape
    if coefficients <= 0 or coefficients > steps:
        raise ValueError(f"Invalid DCT coefficient count {coefficients} for T={steps}")
    sequence = normalized.reshape(n, steps, 12)
    trajectory_coefficients = dct(sequence, axis=1, norm="ortho")[:, :coefficients]
    mask_coefficients = dct(mask.astype(np.float32), axis=1, norm="ortho")[:, :coefficients]
    return np.concatenate(
        (
            trajectory_coefficients.reshape(n, -1),
            float(mask_weight) * mask_coefficients.reshape(n, -1),
        ),
        axis=1,
    ).astype(np.float32)


def stratum_keys(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Group ordered agent types and keep fallback events separate."""
    first = arrays["first_agent_type"].astype(np.int16)
    second = arrays["second_agent_type"].astype(np.int16)
    fallback = (arrays["event_mode"].astype(str) == "ooi_closest_fallback").astype(np.int16)
    return first * 100 + second * 10 + fallback


def sample_valid_fraction(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool).all(axis=2).mean(axis=1).astype(np.float32)


def exact_masked_rms(
    normalized: np.ndarray,
    mask: np.ndarray,
    anchor: int,
    candidates: np.ndarray,
    *,
    min_pair_overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.int64)
    if not len(candidates):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    common = mask[candidates] & mask[int(anchor)][None]
    joint_overlap = common.all(axis=2).mean(axis=1).astype(np.float32)
    difference = normalized[candidates] - normalized[int(anchor)][None]
    squared = np.sum(difference * difference * common[..., None], axis=(1, 2, 3), dtype=np.float64)
    denominator = np.maximum(common.sum(axis=(1, 2), dtype=np.int64) * normalized.shape[-1], 1)
    distance = np.sqrt(squared / denominator).astype(np.float32)
    distance[joint_overlap < float(min_pair_overlap)] = np.inf
    return distance, joint_overlap


def _coarse_candidates(
    tree: cKDTree,
    members: np.ndarray,
    descriptor: np.ndarray,
    anchor: int,
    scenario_id: np.ndarray,
    cfg: RmsConfig,
) -> np.ndarray:
    query_k = min(len(members), cfg.retrieval_candidates + cfg.retrieval_buffer + 1)
    if query_k <= 1:
        return np.empty((0,), dtype=np.int64)
    _, local = tree.query(descriptor[int(anchor)], k=query_k, workers=1)
    candidates = members[np.atleast_1d(local).astype(np.int64)]
    candidates = candidates[
        (candidates != int(anchor)) & (scenario_id[candidates] != scenario_id[int(anchor)])
    ]
    return candidates[: cfg.retrieval_candidates]


def build_exact_neighbours(
    normalized: np.ndarray,
    mask: np.ndarray,
    descriptor: np.ndarray,
    scenario_id: np.ndarray,
    strata: np.ndarray,
    eligible: np.ndarray,
    cfg: RmsConfig,
) -> tuple[dict[str, np.ndarray], dict[int, tuple[np.ndarray, cKDTree]], dict[str, object]]:
    n = len(normalized)
    indices = np.full((n, cfg.num_neighbours), -1, dtype=np.int32)
    distances = np.full((n, cfg.num_neighbours), np.inf, dtype=np.float32)
    overlaps = np.zeros((n, cfg.num_neighbours), dtype=np.float32)
    trees: dict[int, tuple[np.ndarray, cKDTree]] = {}
    started = time.time()
    processed = 0

    for key in np.unique(strata[eligible]):
        members = np.flatnonzero(eligible & (strata == key)).astype(np.int64)
        if len(members) < 2:
            continue
        tree = cKDTree(descriptor[members])
        trees[int(key)] = (members, tree)
        query_k = min(len(members), cfg.retrieval_candidates + cfg.retrieval_buffer + 1)
        for start in range(0, len(members), cfg.query_chunk_size):
            anchors = members[start : start + cfg.query_chunk_size]
            _, local_batch = tree.query(descriptor[anchors], k=query_k, workers=-1)
            local_batch = np.asarray(local_batch)
            if local_batch.ndim == 1:
                local_batch = local_batch[:, None]
            for row, anchor in enumerate(anchors):
                candidates = members[local_batch[row].astype(np.int64)]
                candidates = candidates[
                    (candidates != anchor) & (scenario_id[candidates] != scenario_id[anchor])
                ][: cfg.retrieval_candidates]
                exact, overlap = exact_masked_rms(
                    normalized,
                    mask,
                    int(anchor),
                    candidates,
                    min_pair_overlap=cfg.min_pair_overlap,
                )
                finite = np.flatnonzero(np.isfinite(exact))
                if len(finite):
                    order = finite[np.argsort(exact[finite], kind="stable")[: cfg.num_neighbours]]
                    count = len(order)
                    indices[anchor, :count] = candidates[order].astype(np.int32)
                    distances[anchor, :count] = exact[order]
                    overlaps[anchor, :count] = overlap[order]
            processed += len(anchors)
            print(
                f"[rms] stratum={int(key)} processed={processed}/{int(eligible.sum())} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    finite_distances = distances[np.isfinite(distances)]
    stats: dict[str, object] = {
        "eligible_samples": int(eligible.sum()),
        "samples_with_neighbours": int((indices >= 0).any(axis=1).sum()),
        "stored_edges": int((indices >= 0).sum()),
        "stratum_sizes": {
            str(int(key)): int((eligible & (strata == key)).sum()) for key in np.unique(strata[eligible])
        },
        "distance_quantiles": _quantiles(finite_distances),
        "elapsed_seconds": time.time() - started,
    }
    return {
        "neighbor_indices": indices,
        "rms_distances": distances,
        "common_valid_fraction": overlaps,
    }, trees, stats


def validate_coarse_recall(
    normalized: np.ndarray,
    mask: np.ndarray,
    descriptor: np.ndarray,
    scenario_id: np.ndarray,
    strata: np.ndarray,
    eligible: np.ndarray,
    trees: dict[int, tuple[np.ndarray, cKDTree]],
    cfg: RmsConfig,
    *,
    num_anchors: int,
    seed: int,
) -> dict[str, object]:
    candidates = np.flatnonzero(eligible & np.isin(strata, np.asarray(list(trees), dtype=strata.dtype)))
    if num_anchors <= 0 or not len(candidates):
        return {"anchors_checked": 0}
    rng = np.random.default_rng(seed)
    anchors = rng.choice(candidates, size=min(num_anchors, len(candidates)), replace=False)
    recalls = []
    true_counts = []
    started = time.time()
    for number, anchor in enumerate(anchors, start=1):
        members, tree = trees[int(strata[anchor])]
        coarse = _coarse_candidates(tree, members, descriptor, int(anchor), scenario_id, cfg)
        exact_parts = []
        member_parts = []
        for start in range(0, len(members), cfg.query_chunk_size):
            block = members[start : start + cfg.query_chunk_size]
            keep = (block != anchor) & (scenario_id[block] != scenario_id[anchor])
            block = block[keep]
            exact, _ = exact_masked_rms(
                normalized,
                mask,
                int(anchor),
                block,
                min_pair_overlap=cfg.min_pair_overlap,
            )
            exact_parts.append(exact)
            member_parts.append(block)
        exact_all = np.concatenate(exact_parts) if exact_parts else np.empty((0,), dtype=np.float32)
        members_all = np.concatenate(member_parts) if member_parts else np.empty((0,), dtype=np.int64)
        finite = np.flatnonzero(np.isfinite(exact_all))
        if not len(finite):
            continue
        true_order = finite[np.argsort(exact_all[finite], kind="stable")[: cfg.num_neighbours]]
        truth = members_all[true_order]
        recalls.append(float(np.isin(truth, coarse).sum() / len(truth)))
        true_counts.append(int(len(truth)))
        print(
            f"[recall] {number}/{len(anchors)} recall={recalls[-1]:.3f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    values = np.asarray(recalls, dtype=np.float32)
    return {
        "anchors_requested": int(num_anchors),
        "anchors_checked": len(recalls),
        "exact_top_k": cfg.num_neighbours,
        "coarse_candidates": cfg.retrieval_candidates,
        "mean_recall": float(values.mean()) if len(values) else None,
        "min_recall": float(values.min()) if len(values) else None,
        "fraction_at_least_0.95": float((values >= 0.95).mean()) if len(values) else None,
        "true_neighbour_count_min": min(true_counts) if true_counts else 0,
        "elapsed_seconds": time.time() - started,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {str(q): float(np.quantile(values, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)}


def write_top_neighbour_csv(
    path: Path,
    arrays: dict[str, np.ndarray],
    neighbours: dict[str, np.ndarray],
    *,
    top_k: int,
) -> None:
    fields = (
        "anchor_index",
        "anchor_scenario_id",
        "anchor_event_mode",
        "anchor_zone_pet_s",
        "rank",
        "neighbor_index",
        "neighbor_scenario_id",
        "neighbor_event_mode",
        "neighbor_zone_pet_s",
        "rms_distance",
        "common_valid_fraction",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for anchor in range(len(arrays["scenario_id"])):
            for rank in range(min(top_k, neighbours["neighbor_indices"].shape[1])):
                other = int(neighbours["neighbor_indices"][anchor, rank])
                if other < 0:
                    continue
                writer.writerow(
                    {
                        "anchor_index": anchor,
                        "anchor_scenario_id": arrays["scenario_id"][anchor],
                        "anchor_event_mode": arrays["event_mode"][anchor],
                        "anchor_zone_pet_s": float(arrays["zone_pet_s"][anchor]),
                        "rank": rank + 1,
                        "neighbor_index": other,
                        "neighbor_scenario_id": arrays["scenario_id"][other],
                        "neighbor_event_mode": arrays["event_mode"][other],
                        "neighbor_zone_pet_s": float(arrays["zone_pet_s"][other]),
                        "rms_distance": float(neighbours["rms_distances"][anchor, rank]),
                        "common_valid_fraction": float(neighbours["common_valid_fraction"][anchor, rank]),
                    }
                )


def save_split(
    output_dir: Path,
    split: str,
    arrays: dict[str, np.ndarray],
    normalized: np.ndarray,
    eligible: np.ndarray,
    valid_fraction: np.ndarray,
    strata: np.ndarray,
    median: np.ndarray,
    iqr: np.ndarray,
    cfg: RmsConfig,
    neighbours: dict[str, np.ndarray],
    csv_top_k: int,
) -> dict[str, str]:
    feature_path = output_dir / f"{split}_rms_features.npz"
    neighbour_path = output_dir / f"{split}_rms_neighbors.npz"
    csv_path = output_dir / f"{split}_rms_top{csv_top_k}.csv"
    np.savez_compressed(
        feature_path,
        normalized_sequence=normalized.astype(np.float16),
        aligned_mask=arrays["aligned_mask"],
        eligible_mask=eligible,
        aligned_valid_fraction=valid_fraction,
        stratum_key=strata,
        sample_index=arrays["sample_index"],
        scenario_id=arrays["scenario_id"],
        source_path=arrays["source_path"],
        first_agent_id=arrays["first_agent_id"],
        second_agent_id=arrays["second_agent_id"],
        first_agent_type=arrays["first_agent_type"],
        second_agent_type=arrays["second_agent_type"],
        is_original_ooi_pair=arrays["is_original_ooi_pair"],
        event_mode=arrays["event_mode"],
        primary_step_first=arrays["primary_step_first"],
        primary_step_second=arrays["primary_step_second"],
        zone_pet_s=arrays["zone_pet_s"],
        center_pet_s=arrays["center_pet_s"],
        min_clearance_m=arrays["min_clearance_m"],
        normalization_median=median,
        normalization_iqr=iqr,
        feature_names=np.asarray(FEATURE_NAMES),
        time_offsets=cfg.offsets,
    )
    np.savez_compressed(neighbour_path, **neighbours)
    write_top_neighbour_csv(csv_path, arrays, neighbours, top_k=csv_top_k)
    return {
        "features_npz": str(feature_path),
        "neighbors_npz": str(neighbour_path),
        "top_neighbors_csv": str(csv_path),
    }


def build(args: argparse.Namespace) -> None:
    started = time.time()
    dataset_dir = Path(args.dataset_dir).resolve()
    summary_path = dataset_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset is incomplete or missing summary: {summary_path}")
    source_summary = json.loads(summary_path.read_text())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = RmsConfig(
        history_steps=args.history_steps,
        future_steps=args.future_steps,
        min_valid_fraction=args.min_valid_fraction,
        min_pair_overlap=args.min_pair_overlap,
        dct_coefficients=args.dct_coefficients,
        mask_descriptor_weight=args.mask_descriptor_weight,
        retrieval_candidates=args.retrieval_candidates,
        retrieval_buffer=args.retrieval_buffer,
        num_neighbours=args.num_neighbours,
        query_chunk_size=args.query_chunk_size,
    )
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if not splits:
        raise ValueError("At least one split is required")
    if "train" not in splits:
        raise ValueError("Train split is required to fit normalization")

    loaded = {
        split: load_aligned_split(dataset_dir, split, cfg, max_samples=args.max_samples_per_split)
        for split in splits
    }
    median, iqr = fit_robust_scaler(loaded["train"]["aligned_trajectory"], loaded["train"]["aligned_mask"])

    split_summaries = {}
    for split in splits:
        arrays = loaded[split]
        normalized = normalize_aligned(
            arrays["aligned_trajectory"], arrays["aligned_mask"], median, iqr
        )
        valid_fraction = sample_valid_fraction(arrays["aligned_mask"])
        eligible = valid_fraction >= cfg.min_valid_fraction
        strata = stratum_keys(arrays)
        descriptor = retrieval_descriptor(
            normalized,
            arrays["aligned_mask"],
            coefficients=cfg.dct_coefficients,
            mask_weight=cfg.mask_descriptor_weight,
        )
        neighbours, trees, neighbour_stats = build_exact_neighbours(
            normalized,
            arrays["aligned_mask"],
            descriptor,
            arrays["scenario_id"],
            strata,
            eligible,
            cfg,
        )
        recall_stats = validate_coarse_recall(
            normalized,
            arrays["aligned_mask"],
            descriptor,
            arrays["scenario_id"],
            strata,
            eligible,
            trees,
            cfg,
            num_anchors=args.validation_anchors,
            seed=args.seed + (0 if split == "train" else 1),
        )
        outputs = save_split(
            output_dir,
            split,
            arrays,
            normalized,
            eligible,
            valid_fraction,
            strata,
            median,
            iqr,
            cfg,
            neighbours,
            args.csv_top_k,
        )
        split_summaries[split] = {
            "samples": len(normalized),
            "eligible_samples": int(eligible.sum()),
            "valid_fraction_quantiles": _quantiles(valid_fraction),
            "descriptor_dimension": int(descriptor.shape[1]),
            "neighbors": neighbour_stats,
            "coarse_retrieval_validation": recall_stats,
            "outputs": outputs,
        }
        # The unnormalized float32 representation is not part of the output.
        del normalized, descriptor, trees

    summary = {
        "version": "full_pair_event_aligned_masked_rms_v0",
        "dataset_dir": str(dataset_dir),
        "source_dataset_version": source_summary.get("version"),
        "source_dataset_config": source_summary.get("config"),
        "config": asdict(cfg),
        "alignment": {
            "anchor": "primary_step_first (first-arrival agent)",
            "anchor_output_index": cfg.history_steps,
            "time_offsets": cfg.offsets.tolist(),
            "fractional_steps": "linear interpolation; heading sin/cos renormalized",
        },
        "normalization": {
            "fit_split": "train",
            "method": "per-role/channel median and IQR over valid aligned states only",
            "position_and_velocity_iqr_floor": 1.0,
            "heading_median": 0.0,
            "heading_scale": 1.0,
            "median": median.tolist(),
            "iqr": iqr.tolist(),
        },
        "distance": {
            "metric": "RMS Euclidean over common valid normalized T x 2 x 6 states",
            "coarse_retrieval": "low-frequency DCT descriptor plus mask DCT",
            "exact_reranking": True,
            "different_scenario_only": True,
            "strata": ["ordered_first_agent_type", "ordered_second_agent_type", "fallback_vs_contact"],
        },
        "splits": split_summaries,
        "elapsed_seconds": time.time() - started,
    }
    output_summary = output_dir / "rms_summary.json"
    output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"saved {output_summary}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_full_pairs_50k_v2_no_topk",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_full_pairs_50k_v2_no_topk_rms_v0",
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--max_samples_per_split", type=int, default=0, help="0 uses every sample")
    parser.add_argument("--history_steps", type=int, default=19)
    parser.add_argument("--future_steps", type=int, default=40)
    parser.add_argument("--min_valid_fraction", type=float, default=0.80)
    parser.add_argument("--min_pair_overlap", type=float, default=0.70)
    parser.add_argument("--dct_coefficients", type=int, default=3)
    parser.add_argument("--mask_descriptor_weight", type=float, default=0.25)
    parser.add_argument("--retrieval_candidates", type=int, default=1024)
    parser.add_argument("--retrieval_buffer", type=int, default=64)
    parser.add_argument("--num_neighbours", type=int, default=32)
    parser.add_argument("--query_chunk_size", type=int, default=2048)
    parser.add_argument("--validation_anchors", type=int, default=32)
    parser.add_argument("--csv_top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    build(build_argparser().parse_args())


if __name__ == "__main__":
    main()
