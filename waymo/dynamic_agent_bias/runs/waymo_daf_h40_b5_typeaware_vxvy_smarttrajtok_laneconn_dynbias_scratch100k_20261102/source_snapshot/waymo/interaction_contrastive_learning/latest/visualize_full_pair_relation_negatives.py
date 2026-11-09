"""Visualize negatives whose future pair relation differs from fixed anchors."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from .build_full_pair_rms_neighbors import exact_masked_rms, retrieval_descriptor
    from .visualize_full_pair_rms_neighbors import (
        TYPE_NAMES,
        _bounds,
        _load_anchor_manifest,
        _panel_svg,
    )
except ImportError:
    from build_full_pair_rms_neighbors import exact_masked_rms, retrieval_descriptor
    from visualize_full_pair_rms_neighbors import TYPE_NAMES, _bounds, _load_anchor_manifest, _panel_svg


ORDER_NAMES = {-1: "second-ahead", 0: "overlap", 1: "first-ahead"}
TREND_NAMES = {-1: "decreasing", 0: "stable", 1: "increasing"}


def _three_way(values: np.ndarray, margin: float) -> np.ndarray:
    return np.where(values > margin, 1, np.where(values < -margin, -1, 0)).astype(np.int8)


def relation_outcomes(
    positions: np.ndarray,
    indices: np.ndarray,
    *,
    event_index: int,
    outcome_index: int,
    margin_m: float,
) -> dict[str, np.ndarray]:
    pair_positions = positions[np.asarray(indices, dtype=np.int64)]
    relative_event = pair_positions[:, event_index, 0] - pair_positions[:, event_index, 1]
    relative_outcome = pair_positions[:, outcome_index, 0] - pair_positions[:, outcome_index, 1]
    gap_event = relative_event[:, 0]
    gap_outcome = relative_outcome[:, 0]
    gap_change = gap_outcome - gap_event
    distance_event = np.linalg.norm(relative_event, axis=1)
    distance_outcome = np.linalg.norm(relative_outcome, axis=1)
    distance_change = distance_outcome - distance_event
    order_event = _three_way(gap_event, margin_m)
    order_outcome = _three_way(gap_outcome, margin_m)
    return {
        "gap_event": gap_event,
        "gap_outcome": gap_outcome,
        "gap_change": gap_change,
        "distance_event": distance_event,
        "distance_outcome": distance_outcome,
        "distance_change": distance_change,
        "order_event": order_event,
        "order_outcome": order_outcome,
        "order_swap": (order_event * order_outcome) == -1,
        "gap_trend": _three_way(gap_change, margin_m),
        "distance_trend": _three_way(distance_change, margin_m),
    }


def _select_relation_negative(
    *,
    anchor: int,
    candidates: np.ndarray,
    exact_distances: np.ndarray,
    overlaps: np.ndarray,
    stored_neighbours: np.ndarray,
    candidate_horizon_valid: np.ndarray,
    anchor_outcome: dict[str, np.ndarray],
    candidate_outcomes: dict[str, np.ndarray],
    min_overlap: float,
    rank32_distance: float,
    min_rms_ratio: float,
    min_rms_delta: float,
) -> tuple[int, int, int, list[str]] | None:
    in_anchor_top_k = np.isin(candidates, stored_neighbours[anchor])
    anchor_in_candidate_top_k = np.asarray(
        [anchor in stored_neighbours[int(candidate)] for candidate in candidates],
        dtype=bool,
    )
    usable = (
        np.isfinite(exact_distances)
        & (overlaps >= float(min_overlap))
        & (
            exact_distances
            >= max(
                float(rank32_distance) * float(min_rms_ratio),
                float(rank32_distance) + float(min_rms_delta),
            )
        )
        & candidate_horizon_valid
        & ~in_anchor_top_k
        & ~anchor_in_candidate_top_k
    )

    anchor_swap = bool(anchor_outcome["order_swap"][0])
    anchor_order = int(anchor_outcome["order_outcome"][0])
    anchor_gap_trend = int(anchor_outcome["gap_trend"][0])
    anchor_distance_trend = int(anchor_outcome["distance_trend"][0])
    swap_diff = candidate_outcomes["order_swap"] != anchor_swap
    end_order_diff = candidate_outcomes["order_outcome"] != anchor_order
    end_order_opposite = candidate_outcomes["order_outcome"] * anchor_order == -1
    gap_opposite = candidate_outcomes["gap_trend"] * anchor_gap_trend == -1
    distance_opposite = candidate_outcomes["distance_trend"] * anchor_distance_trend == -1
    # Prefer categorical pass/order outcomes. If none exist, fall back to an
    # opposite relative-gap/distance trend. Candidates are already DCT-sorted,
    # so the first usable offset is the DCT-nearest outcome-negative.
    tier_masks = (
        usable & (swap_diff | end_order_opposite),
        usable & (end_order_diff | gap_opposite | distance_opposite),
    )
    selected_tier = next(
        (tier for tier, mask in enumerate(tier_masks, start=1) if bool(mask.any())),
        None,
    )
    if selected_tier is None:
        return None
    selected_offset = int(np.flatnonzero(tier_masks[selected_tier - 1])[0])

    reasons = []
    if bool(swap_diff[selected_offset]):
        reasons.append("order-swap differs")
    if bool(end_order_diff[selected_offset]):
        reasons.append("future order differs")
    if bool(end_order_opposite[selected_offset]):
        reasons.append("future order opposite")
    if bool(gap_opposite[selected_offset]):
        reasons.append("longitudinal-gap trend opposite")
    if bool(distance_opposite[selected_offset]):
        reasons.append("distance trend opposite")

    finite_offsets = np.flatnonzero(np.isfinite(exact_distances))
    exact_order = finite_offsets[
        np.argsort(exact_distances[finite_offsets], kind="stable")
    ]
    exact_rank = int(np.flatnonzero(exact_order == selected_offset)[0]) + 1
    return selected_offset, exact_rank, selected_tier, reasons


def _outcome_text(outcome: dict[str, np.ndarray], index: int) -> str:
    swap = "swap" if bool(outcome["order_swap"][index]) else "no-swap"
    return (
        f"{swap}; future={ORDER_NAMES[int(outcome['order_outcome'][index])]}; "
        f"gap-change={float(outcome['gap_change'][index]):+.1f}m "
        f"({TREND_NAMES[int(outcome['gap_trend'][index])]}); "
        f"distance-change={float(outcome['distance_change'][index]):+.1f}m "
        f"({TREND_NAMES[int(outcome['distance_trend'][index])]})"
    )


def build_audit(
    *,
    rms_dir: Path,
    output_dir: Path,
    split: str,
    anchor_manifest: Path,
    outcome_steps: int,
    relation_margin_m: float,
    min_overlap: float,
    min_rms_ratio: float,
    min_rms_delta: float,
) -> None:
    summary = json.loads((rms_dir / "rms_summary.json").read_text())
    config = summary["config"]
    retrieval_candidates = int(config["retrieval_candidates"])
    retrieval_buffer = int(config["retrieval_buffer"])

    with np.load(rms_dir / f"{split}_rms_features.npz", allow_pickle=False) as features:
        arrays = {key: np.asarray(features[key]) for key in features.files}
    with np.load(rms_dir / f"{split}_rms_neighbors.npz", allow_pickle=False) as neighbours:
        stored_indices = np.asarray(neighbours["neighbor_indices"])
        stored_distances = np.asarray(neighbours["rms_distances"])

    anchors = _load_anchor_manifest(anchor_manifest)
    normalized = arrays["normalized_sequence"].astype(np.float32)
    masks = arrays["aligned_mask"].astype(bool)
    eligible = arrays["eligible_mask"].astype(bool)
    strata = arrays["stratum_key"]
    scenario_ids = arrays["scenario_id"]
    median = arrays["normalization_median"].astype(np.float32)
    iqr = arrays["normalization_iqr"].astype(np.float32)
    positions = (
        normalized[..., 0:2] * iqr[None, None, :, 0:2]
        + median[None, None, :, 0:2]
    )
    offsets = arrays["time_offsets"].astype(np.int64)
    event_index = int(np.argmin(np.abs(offsets)))
    matches = np.flatnonzero(offsets == int(outcome_steps))
    if len(matches) != 1:
        raise ValueError(f"outcome_steps={outcome_steps} is absent or duplicated in time_offsets")
    outcome_index = int(matches[0])
    anchor_horizon_valid = masks[anchors, event_index].all(axis=1) & masks[
        anchors, outcome_index
    ].all(axis=1)
    if not bool(anchor_horizon_valid.all()):
        invalid = anchors[~anchor_horizon_valid]
        raise ValueError(f"Anchors invalid at event/outcome horizon: {invalid.tolist()}")

    descriptor = retrieval_descriptor(
        normalized,
        masks,
        coefficients=int(config["dct_coefficients"]),
        mask_weight=float(config["mask_descriptor_weight"]),
    )
    trees: dict[int, tuple[np.ndarray, cKDTree]] = {}
    for key in np.unique(strata[anchors]):
        members = np.flatnonzero(eligible & (strata == key)).astype(np.int64)
        trees[int(key)] = (members, cKDTree(descriptor[members]))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fields = (
        "audit_order",
        "anchor_index",
        "negative_index",
        "difference_reasons",
        "selection_tier",
        "dct_rank",
        "dct_distance",
        "exact_rms_rank_within_candidates",
        "exact_rms_distance",
        "rank32_rms_distance",
        "common_valid_fraction",
        "anchor_order_swap",
        "negative_order_swap",
        "anchor_future_order",
        "negative_future_order",
        "anchor_gap_change_m",
        "negative_gap_change_m",
        "anchor_distance_change_m",
        "negative_distance_change_m",
        "anchor_scenario_id",
        "negative_scenario_id",
        "stratum_key",
    )
    cards: list[str] = []
    rows: list[dict[str, object]] = []
    skipped: list[int] = []

    for audit_order, anchor_value in enumerate(anchors):
        anchor = int(anchor_value)
        members, tree = trees[int(strata[anchor])]
        query_k = min(len(members), retrieval_candidates + retrieval_buffer + 1)
        dct_distances, local_indices = tree.query(
            descriptor[anchor], k=query_k, workers=1
        )
        candidates = members[np.atleast_1d(local_indices).astype(np.int64)]
        dct_distances = np.atleast_1d(dct_distances).astype(np.float32)
        keep = (candidates != anchor) & (scenario_ids[candidates] != scenario_ids[anchor])
        candidates = candidates[keep][:retrieval_candidates]
        dct_distances = dct_distances[keep][:retrieval_candidates]
        exact_distances, overlaps = exact_masked_rms(
            normalized,
            masks,
            anchor,
            candidates,
            min_pair_overlap=float(config["min_pair_overlap"]),
        )
        candidate_horizon_valid = masks[candidates, event_index].all(axis=1) & masks[
            candidates, outcome_index
        ].all(axis=1)
        anchor_outcome = relation_outcomes(
            positions,
            np.asarray([anchor]),
            event_index=event_index,
            outcome_index=outcome_index,
            margin_m=relation_margin_m,
        )
        candidate_outcomes = relation_outcomes(
            positions,
            candidates,
            event_index=event_index,
            outcome_index=outcome_index,
            margin_m=relation_margin_m,
        )
        selected = _select_relation_negative(
            anchor=anchor,
            candidates=candidates,
            exact_distances=exact_distances,
            overlaps=overlaps,
            stored_neighbours=stored_indices,
            candidate_horizon_valid=candidate_horizon_valid,
            anchor_outcome=anchor_outcome,
            candidate_outcomes=candidate_outcomes,
            min_overlap=min_overlap,
            rank32_distance=float(stored_distances[anchor, -1]),
            min_rms_ratio=min_rms_ratio,
            min_rms_delta=min_rms_delta,
        )
        if selected is None:
            skipped.append(anchor)
            continue
        selected_offset, exact_rank, selected_tier, reasons = selected
        negative = int(candidates[selected_offset])
        exact_distance = float(exact_distances[selected_offset])

        anchor_position = positions[anchor]
        negative_position = positions[negative]
        bounds = _bounds(
            anchor_position,
            masks[anchor],
            negative_position,
            masks[negative],
        )

        def pair_label(index: int) -> str:
            types = "/".join(
                TYPE_NAMES.get(int(value), str(int(value)))
                for value in (
                    arrays["first_agent_type"][index],
                    arrays["second_agent_type"][index],
                )
            )
            return f"{arrays['event_mode'][index]} | {types}"

        svg = (
            '<svg viewBox="0 0 712 350" role="img" aria-label="relation-outcome negative">'
            + _panel_svg(
                anchor_position,
                masks[anchor],
                bounds=bounds,
                x_offset=0,
                title=f"anchor #{anchor}",
                subtitle=pair_label(anchor),
                event_index=event_index,
            )
            + _panel_svg(
                negative_position,
                masks[negative],
                bounds=bounds,
                x_offset=362,
                title=f"relation negative #{negative}",
                subtitle=pair_label(negative),
                event_index=event_index,
            )
            + "</svg>"
        )
        anchor_text = _outcome_text(anchor_outcome, 0)
        negative_text = _outcome_text(candidate_outcomes, selected_offset)
        reason_text = "; ".join(reasons)
        cards.append(
            '<article><div class="metric">'
            f"tier {selected_tier}: {html.escape(reason_text)} &middot; "
            f"DCT rank {selected_offset + 1}/{retrieval_candidates} "
            f"&middot; RMS {exact_distance:.3f} (exact rank {exact_rank}/{len(candidates)})"
            f'</div>{svg}<div class="outcome"><b>anchor:</b> {html.escape(anchor_text)}<br>'
            f"<b>negative:</b> {html.escape(negative_text)}</div>"
            f'<div class="scene">overlap {float(overlaps[selected_offset]):.1%} &middot; '
            f"rank-32 RMS {float(stored_distances[anchor, -1]):.3f}<br>"
            f"anchor scene {html.escape(str(scenario_ids[anchor]))}<br>"
            f"negative scene {html.escape(str(scenario_ids[negative]))}</div></article>"
        )
        rows.append(
            {
                "audit_order": audit_order,
                "anchor_index": anchor,
                "negative_index": negative,
                "difference_reasons": ";".join(reasons),
                "selection_tier": selected_tier,
                "dct_rank": selected_offset + 1,
                "dct_distance": float(dct_distances[selected_offset]),
                "exact_rms_rank_within_candidates": exact_rank,
                "exact_rms_distance": exact_distance,
                "rank32_rms_distance": float(stored_distances[anchor, -1]),
                "common_valid_fraction": float(overlaps[selected_offset]),
                "anchor_order_swap": bool(anchor_outcome["order_swap"][0]),
                "negative_order_swap": bool(candidate_outcomes["order_swap"][selected_offset]),
                "anchor_future_order": ORDER_NAMES[int(anchor_outcome["order_outcome"][0])],
                "negative_future_order": ORDER_NAMES[
                    int(candidate_outcomes["order_outcome"][selected_offset])
                ],
                "anchor_gap_change_m": float(anchor_outcome["gap_change"][0]),
                "negative_gap_change_m": float(candidate_outcomes["gap_change"][selected_offset]),
                "anchor_distance_change_m": float(anchor_outcome["distance_change"][0]),
                "negative_distance_change_m": float(
                    candidate_outcomes["distance_change"][selected_offset]
                ),
                "anchor_scenario_id": scenario_ids[anchor],
                "negative_scenario_id": scenario_ids[negative],
                "stratum_key": int(strata[anchor]),
            }
        )

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    skipped_text = ", ".join(f"#{value}" for value in skipped) or "none"
    (output_dir / "index.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>Relation-outcome negatives</title>"
        "<style>body{background:#0f1217;color:#eee;font-family:system-ui;margin:24px}"
        "h1{font-size:24px}p{color:#aeb7c4;max-width:1150px}.grid{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(720px,1fr));gap:18px}"
        "article{background:#222832;border:1px solid #394250;border-radius:12px;padding:12px}"
        "svg{width:100%;height:auto}.metric{font-size:15px;font-weight:650;margin-bottom:8px}"
        ".outcome{font:12px ui-monospace;line-height:1.6;color:#e3e8ef;margin-top:7px}"
        ".scene{font:11px ui-monospace;color:#aeb7c4;margin-top:7px}"
        "</style></head><body><h1>Relation-outcome negatives</h1>"
        f"<p>Search all {retrieval_candidates} same-stratum DCT candidates. At event + "
        f"{outcome_steps * 0.1:.1f}s, require a different order-swap/future-order outcome or "
        "an opposite longitudinal-gap/distance trend. Speed is not filtered. Stored top-32 "
        f"edges are excluded in both directions; overlap must be at least {min_overlap:.0%}; "
        f"RMS must be at least max({min_rms_ratio:.2f}&times;rank-32, rank-32+"
        f"{min_rms_delta:.2f}). Tier 1 prioritizes order-swap/opposite-order outcomes; tier 2 "
        "uses other future-order or opposite-trend outcomes. Within the first available tier, "
        "the DCT-nearest candidate is selected. "
        f"Relation margin={relation_margin_m:.1f}m. Skipped anchors with no strong candidate: "
        f"{html.escape(skipped_text)}.</p><div class='grid'>{''.join(cards)}</div></body></html>",
        encoding="utf-8",
    )
    print(
        f"saved {output_dir / 'index.html'} ({len(rows)} relation negatives; "
        f"skipped={skipped})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rms_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--anchor_manifest", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--outcome_steps", type=int, default=20)
    parser.add_argument("--relation_margin_m", type=float, default=2.0)
    parser.add_argument("--min_overlap", type=float, default=0.80)
    parser.add_argument("--min_rms_ratio", type=float, default=1.25)
    parser.add_argument("--min_rms_delta", type=float, default=0.05)
    args = parser.parse_args()
    build_audit(
        rms_dir=args.rms_dir,
        output_dir=args.output_dir,
        split=args.split,
        anchor_manifest=args.anchor_manifest,
        outcome_steps=args.outcome_steps,
        relation_margin_m=args.relation_margin_m,
        min_overlap=args.min_overlap,
        min_rms_ratio=args.min_rms_ratio,
        min_rms_delta=args.min_rms_delta,
    )


if __name__ == "__main__":
    main()
