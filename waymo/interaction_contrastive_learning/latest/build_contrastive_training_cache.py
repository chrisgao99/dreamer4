"""Build duplicate-safe positives and relation-outcome negatives for training.

The positive pool is the stored exact-RMS top-K after removing near-identical
trajectory slices.  Negatives are mined from all DCT-retrieved candidates and
must exhibit a different future pair-relation outcome.  Both splits are mined
independently and only causal query times with a full history window are kept.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from .build_full_pair_rms_neighbors import exact_masked_rms, retrieval_descriptor
    from .visualize_full_pair_relation_negatives import relation_outcomes
except ImportError:
    from build_full_pair_rms_neighbors import exact_masked_rms, retrieval_descriptor
    from visualize_full_pair_relation_negatives import relation_outcomes


REASON_SWAP_DIFF = 1
REASON_FUTURE_ORDER_DIFF = 2
REASON_FUTURE_ORDER_OPPOSITE = 4
REASON_GAP_TREND_OPPOSITE = 8
REASON_DISTANCE_TREND_OPPOSITE = 16


@dataclass(frozen=True)
class ContrastiveCacheConfig:
    history_steps: int = 32
    outcome_steps: int = 20
    relation_margin_m: float = 2.0
    min_overlap: float = 0.80
    duplicate_rms_threshold: float = 0.02
    min_negative_rms_ratio: float = 1.25
    min_negative_rms_delta: float = 0.05
    num_negatives: int = 16
    query_chunk_size: int = 256


def select_duplicate_safe_positives(
    neighbour_indices: np.ndarray,
    neighbour_distances: np.ndarray,
    candidate_causal_eligible: np.ndarray,
    source_paths: np.ndarray,
    *,
    duplicate_rms_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Filter near-identical/invalid neighbour edges while preserving RMS rank."""
    indices = np.asarray(neighbour_indices)
    distances = np.asarray(neighbour_distances)
    n, k = indices.shape
    positive_indices = np.full((n, k), -1, dtype=np.int32)
    positive_distances = np.full((n, k), np.inf, dtype=np.float32)
    positive_original_ranks = np.zeros((n, k), dtype=np.int16)
    duplicate_counts = np.zeros((n,), dtype=np.int16)
    source_paths = np.asarray(source_paths).astype(str)
    for anchor in range(n):
        write = 0
        for rank in range(k):
            candidate = int(indices[anchor, rank])
            distance = float(distances[anchor, rank])
            if candidate < 0 or not np.isfinite(distance):
                continue
            # Same-path is a direct duplicate. Near-zero event-aligned RMS is
            # the robust signal for the same continuous drive stored as two
            # different Waymo scenario slices.
            duplicate = (
                source_paths[candidate] == source_paths[anchor]
                or distance <= float(duplicate_rms_threshold)
            )
            if duplicate:
                duplicate_counts[anchor] += 1
                continue
            if not bool(candidate_causal_eligible[candidate]):
                continue
            positive_indices[anchor, write] = candidate
            positive_distances[anchor, write] = distance
            positive_original_ranks[anchor, write] = rank + 1
            write += 1
    return positive_indices, positive_distances, positive_original_ranks, duplicate_counts


def _relation_masks(
    anchor_outcome: dict[str, np.ndarray],
    candidate_outcomes: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_swap = bool(anchor_outcome["order_swap"][0])
    anchor_order = int(anchor_outcome["order_outcome"][0])
    anchor_gap_trend = int(anchor_outcome["gap_trend"][0])
    anchor_distance_trend = int(anchor_outcome["distance_trend"][0])

    swap_diff = candidate_outcomes["order_swap"] != anchor_swap
    future_order_diff = candidate_outcomes["order_outcome"] != anchor_order
    future_order_opposite = candidate_outcomes["order_outcome"] * anchor_order == -1
    gap_opposite = candidate_outcomes["gap_trend"] * anchor_gap_trend == -1
    distance_opposite = candidate_outcomes["distance_trend"] * anchor_distance_trend == -1
    tier1 = swap_diff | future_order_opposite
    tier2 = future_order_diff | gap_opposite | distance_opposite
    reasons = (
        swap_diff.astype(np.uint8) * REASON_SWAP_DIFF
        + future_order_diff.astype(np.uint8) * REASON_FUTURE_ORDER_DIFF
        + future_order_opposite.astype(np.uint8) * REASON_FUTURE_ORDER_OPPOSITE
        + gap_opposite.astype(np.uint8) * REASON_GAP_TREND_OPPOSITE
        + distance_opposite.astype(np.uint8) * REASON_DISTANCE_TREND_OPPOSITE
    )
    return tier1, tier2, reasons


def mine_relation_negatives(
    *,
    normalized: np.ndarray,
    masks: np.ndarray,
    positions: np.ndarray,
    descriptor: np.ndarray,
    scenario_ids: np.ndarray,
    source_paths: np.ndarray,
    strata: np.ndarray,
    rms_eligible: np.ndarray,
    causal_eligible: np.ndarray,
    stored_indices: np.ndarray,
    stored_distances: np.ndarray,
    rms_config: dict[str, object],
    cfg: ContrastiveCacheConfig,
) -> dict[str, np.ndarray]:
    n = len(normalized)
    negative_indices = np.full((n, cfg.num_negatives), -1, dtype=np.int32)
    negative_distances = np.full((n, cfg.num_negatives), np.inf, dtype=np.float32)
    negative_overlaps = np.zeros((n, cfg.num_negatives), dtype=np.float32)
    negative_dct_ranks = np.zeros((n, cfg.num_negatives), dtype=np.int16)
    negative_exact_ranks = np.zeros((n, cfg.num_negatives), dtype=np.int16)
    negative_tiers = np.zeros((n, cfg.num_negatives), dtype=np.uint8)
    negative_reasons = np.zeros((n, cfg.num_negatives), dtype=np.uint8)

    retrieval_candidates = int(rms_config["retrieval_candidates"])
    retrieval_buffer = int(rms_config["retrieval_buffer"])
    event_index = int(rms_config["history_steps"])
    outcome_index = event_index + int(cfg.outcome_steps)
    if outcome_index >= normalized.shape[1]:
        raise ValueError(
            f"outcome index {outcome_index} outside aligned sequence length {normalized.shape[1]}"
        )
    source_paths = np.asarray(source_paths).astype(str)
    started = time.time()
    processed = 0

    for key in np.unique(strata[causal_eligible]):
        members = np.flatnonzero(rms_eligible & (strata == key)).astype(np.int64)
        anchors = np.flatnonzero(causal_eligible & (strata == key)).astype(np.int64)
        if len(members) < 2 or not len(anchors):
            continue
        tree = cKDTree(descriptor[members])
        query_k = min(len(members), retrieval_candidates + retrieval_buffer + 1)
        for start in range(0, len(anchors), cfg.query_chunk_size):
            anchor_batch = anchors[start : start + cfg.query_chunk_size]
            _, local_batch = tree.query(descriptor[anchor_batch], k=query_k, workers=-1)
            local_batch = np.asarray(local_batch)
            if local_batch.ndim == 1:
                local_batch = local_batch[:, None]
            for row, anchor_value in enumerate(anchor_batch):
                anchor = int(anchor_value)
                candidates = members[local_batch[row].astype(np.int64)]
                keep = (
                    (candidates != anchor)
                    & (scenario_ids[candidates] != scenario_ids[anchor])
                    & (source_paths[candidates] != source_paths[anchor])
                )
                candidates = candidates[keep][:retrieval_candidates]
                exact, overlaps = exact_masked_rms(
                    normalized,
                    masks,
                    anchor,
                    candidates,
                    min_pair_overlap=float(rms_config["min_pair_overlap"]),
                )
                in_anchor_top_k = np.isin(candidates, stored_indices[anchor])
                anchor_in_candidate_top_k = (
                    stored_indices[candidates] == anchor
                ).any(axis=1)
                candidate_horizon_valid = (
                    causal_eligible[candidates]
                    & masks[candidates, event_index].all(axis=1)
                    & masks[candidates, outcome_index].all(axis=1)
                )
                finite_top = stored_distances[anchor][np.isfinite(stored_distances[anchor])]
                if not len(finite_top):
                    continue
                rank_edge = float(finite_top[-1])
                minimum_negative_rms = max(
                    rank_edge * float(cfg.min_negative_rms_ratio),
                    rank_edge + float(cfg.min_negative_rms_delta),
                )
                usable = (
                    np.isfinite(exact)
                    & (exact >= minimum_negative_rms)
                    & (overlaps >= float(cfg.min_overlap))
                    & candidate_horizon_valid
                    & ~in_anchor_top_k
                    & ~anchor_in_candidate_top_k
                )
                if not bool(usable.any()):
                    continue

                anchor_outcome = relation_outcomes(
                    positions,
                    np.asarray([anchor]),
                    event_index=event_index,
                    outcome_index=outcome_index,
                    margin_m=cfg.relation_margin_m,
                )
                candidate_outcomes = relation_outcomes(
                    positions,
                    candidates,
                    event_index=event_index,
                    outcome_index=outcome_index,
                    margin_m=cfg.relation_margin_m,
                )
                tier1, tier2, reasons = _relation_masks(anchor_outcome, candidate_outcomes)
                selected_parts = []
                for tier_mask in (usable & tier1, usable & ~tier1 & tier2):
                    selected_parts.extend(np.flatnonzero(tier_mask).tolist())
                    if len(selected_parts) >= cfg.num_negatives:
                        break
                selected = np.asarray(selected_parts[: cfg.num_negatives], dtype=np.int64)
                if not len(selected):
                    continue

                finite_offsets = np.flatnonzero(np.isfinite(exact))
                exact_order = finite_offsets[
                    np.argsort(exact[finite_offsets], kind="stable")
                ]
                exact_ranks = np.empty((len(candidates),), dtype=np.int32)
                exact_ranks.fill(0)
                exact_ranks[exact_order] = np.arange(1, len(exact_order) + 1)
                count = len(selected)
                negative_indices[anchor, :count] = candidates[selected].astype(np.int32)
                negative_distances[anchor, :count] = exact[selected]
                negative_overlaps[anchor, :count] = overlaps[selected]
                negative_dct_ranks[anchor, :count] = (selected + 1).astype(np.int16)
                negative_exact_ranks[anchor, :count] = exact_ranks[selected].astype(np.int16)
                negative_tiers[anchor, :count] = np.where(tier1[selected], 1, 2).astype(np.uint8)
                negative_reasons[anchor, :count] = reasons[selected]
            processed += len(anchor_batch)
            print(
                f"[negative-cache] stratum={int(key)} processed={processed}/"
                f"{int(causal_eligible.sum())} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    return {
        "negative_indices": negative_indices,
        "negative_rms_distances": negative_distances,
        "negative_common_valid_fraction": negative_overlaps,
        "negative_dct_ranks": negative_dct_ranks,
        "negative_exact_ranks": negative_exact_ranks,
        "negative_tiers": negative_tiers,
        "negative_reason_bits": negative_reasons,
    }


def build_split(rms_dir: Path, output_dir: Path, split: str, cfg: ContrastiveCacheConfig) -> dict[str, object]:
    summary = json.loads((rms_dir / "rms_summary.json").read_text())
    rms_config = summary["config"]
    with np.load(rms_dir / f"{split}_rms_features.npz", allow_pickle=False) as features:
        arrays = {key: np.asarray(features[key]) for key in features.files}
    with np.load(rms_dir / f"{split}_rms_neighbors.npz", allow_pickle=False) as neighbours:
        stored_indices = np.asarray(neighbours["neighbor_indices"])
        stored_distances = np.asarray(neighbours["rms_distances"])

    normalized = arrays["normalized_sequence"].astype(np.float32)
    masks = arrays["aligned_mask"].astype(bool)
    rms_eligible = arrays["eligible_mask"].astype(bool)
    query_steps = np.floor(arrays["primary_step_first"].astype(np.float32)).astype(np.int16)
    event_index = int(rms_config["history_steps"])
    outcome_index = event_index + int(cfg.outcome_steps)
    causal_eligible = (
        rms_eligible
        & (query_steps >= cfg.history_steps - 1)
        & masks[:, event_index].all(axis=1)
        & masks[:, outcome_index].all(axis=1)
    )
    positive_indices, positive_distances, positive_original_ranks, duplicate_counts = select_duplicate_safe_positives(
        stored_indices,
        stored_distances,
        causal_eligible,
        arrays["source_path"],
        duplicate_rms_threshold=cfg.duplicate_rms_threshold,
    )
    has_positive = (positive_indices >= 0).any(axis=1)
    causal_eligible &= has_positive

    descriptor = retrieval_descriptor(
        normalized,
        masks,
        coefficients=int(rms_config["dct_coefficients"]),
        mask_weight=float(rms_config["mask_descriptor_weight"]),
    )
    median = arrays["normalization_median"].astype(np.float32)
    iqr = arrays["normalization_iqr"].astype(np.float32)
    positions = normalized[..., 0:2] * iqr[None, None, :, 0:2] + median[None, None, :, 0:2]
    negatives = mine_relation_negatives(
        normalized=normalized,
        masks=masks,
        positions=positions,
        descriptor=descriptor,
        scenario_ids=arrays["scenario_id"],
        source_paths=arrays["source_path"],
        strata=arrays["stratum_key"],
        rms_eligible=rms_eligible,
        causal_eligible=causal_eligible,
        stored_indices=stored_indices,
        stored_distances=stored_distances,
        rms_config=rms_config,
        cfg=cfg,
    )
    negative_counts = (negatives["negative_indices"] >= 0).sum(axis=1).astype(np.int16)
    training_eligible = causal_eligible & (negative_counts > 0)
    output_path = output_dir / f"{split}_contrastive_training.npz"
    np.savez_compressed(
        output_path,
        sample_index=arrays["sample_index"],
        scenario_id=arrays["scenario_id"],
        source_path=arrays["source_path"],
        first_agent_id=arrays["first_agent_id"],
        second_agent_id=arrays["second_agent_id"],
        query_step=query_steps,
        stratum_key=arrays["stratum_key"],
        causal_eligible_mask=causal_eligible,
        training_eligible_mask=training_eligible,
        positive_indices=positive_indices,
        positive_rms_distances=positive_distances,
        positive_original_ranks=positive_original_ranks,
        duplicate_positive_counts=duplicate_counts,
        **negatives,
    )
    selected_positive_rank = positive_original_ranks[:, 0]
    return {
        "output": str(output_path),
        "samples": len(normalized),
        "rms_eligible": int(rms_eligible.sum()),
        "causal_eligible": int(causal_eligible.sum()),
        "training_eligible": int(training_eligible.sum()),
        "duplicate_edges_removed": int(duplicate_counts.sum()),
        "anchors_with_duplicates_removed": int((duplicate_counts > 0).sum()),
        "positive_selected_original_rank_quantiles": {
            str(q): float(np.quantile(selected_positive_rank[selected_positive_rank > 0], q))
            for q in (0.0, 0.5, 0.9, 0.99, 1.0)
        },
        "negative_count_quantiles": {
            str(q): float(np.quantile(negative_counts[causal_eligible], q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rms_dir",
        type=Path,
        default=Path("waymo/cache/interaction_full_pairs_50k_v2_no_topk_rms_v0"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("waymo/cache/interaction_full_pairs_50k_v2_contrastive_v1"),
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--history_steps", type=int, default=32)
    parser.add_argument("--outcome_steps", type=int, default=20)
    parser.add_argument("--relation_margin_m", type=float, default=2.0)
    parser.add_argument("--min_overlap", type=float, default=0.80)
    parser.add_argument("--duplicate_rms_threshold", type=float, default=0.02)
    parser.add_argument("--min_negative_rms_ratio", type=float, default=1.25)
    parser.add_argument("--min_negative_rms_delta", type=float, default=0.05)
    parser.add_argument("--num_negatives", type=int, default=16)
    parser.add_argument("--query_chunk_size", type=int, default=256)
    args = parser.parse_args()
    cfg = ContrastiveCacheConfig(
        history_steps=args.history_steps,
        outcome_steps=args.outcome_steps,
        relation_margin_m=args.relation_margin_m,
        min_overlap=args.min_overlap,
        duplicate_rms_threshold=args.duplicate_rms_threshold,
        min_negative_rms_ratio=args.min_negative_rms_ratio,
        min_negative_rms_delta=args.min_negative_rms_delta,
        num_negatives=args.num_negatives,
        query_chunk_size=args.query_chunk_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_summaries = {}
    for split in (part.strip() for part in args.splits.split(",") if part.strip()):
        split_summaries[split] = build_split(args.rms_dir, args.output_dir, split, cfg)
    summary = {
        "version": "interaction_contrastive_training_v1",
        "rms_dir": str(args.rms_dir.resolve()),
        "config": asdict(cfg),
        "reason_bits": {
            "order_swap_diff": REASON_SWAP_DIFF,
            "future_order_diff": REASON_FUTURE_ORDER_DIFF,
            "future_order_opposite": REASON_FUTURE_ORDER_OPPOSITE,
            "gap_trend_opposite": REASON_GAP_TREND_OPPOSITE,
            "distance_trend_opposite": REASON_DISTANCE_TREND_OPPOSITE,
        },
        "splits": split_summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
