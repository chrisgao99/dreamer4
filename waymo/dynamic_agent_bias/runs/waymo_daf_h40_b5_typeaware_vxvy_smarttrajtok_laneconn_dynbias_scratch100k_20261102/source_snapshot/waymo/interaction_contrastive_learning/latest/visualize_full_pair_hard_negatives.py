"""Visualize DCT-near but exact-RMS-far hard negatives for fixed anchors.

For every anchor, this audit rebuilds the same-stratum DCT candidate list used
by the RMS neighbour pipeline.  It searches only the first ``dct_pool`` coarse
candidates, excludes the stored top-neighbour region in both directions, and
selects the remaining candidate with the largest full-trajectory exact RMS.
"""

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


def _select_hard_negative(
    *,
    anchor: int,
    candidates: np.ndarray,
    exact_distances: np.ndarray,
    overlaps: np.ndarray,
    stored_neighbours: np.ndarray,
    dct_pool: int,
    min_overlap: float,
) -> tuple[int, int]:
    """Return the candidate-list offset and its exact-RMS rank."""
    candidate_offsets = np.arange(len(candidates), dtype=np.int64)
    in_anchor_top_k = np.isin(candidates, stored_neighbours[anchor])
    anchor_in_candidate_top_k = np.asarray(
        [anchor in stored_neighbours[int(candidate)] for candidate in candidates],
        dtype=bool,
    )
    usable = (
        (candidate_offsets < int(dct_pool))
        & np.isfinite(exact_distances)
        & (overlaps >= float(min_overlap))
        & ~in_anchor_top_k
        & ~anchor_in_candidate_top_k
    )
    usable_offsets = np.flatnonzero(usable)
    if not len(usable_offsets):
        raise ValueError(
            f"No hard-negative candidate for anchor {anchor} within the first "
            f"{dct_pool} DCT candidates"
        )
    selected_offset = int(usable_offsets[np.argmax(exact_distances[usable_offsets])])

    finite_offsets = np.flatnonzero(np.isfinite(exact_distances))
    exact_order = finite_offsets[
        np.argsort(exact_distances[finite_offsets], kind="stable")
    ]
    exact_rank = int(np.flatnonzero(exact_order == selected_offset)[0]) + 1
    return selected_offset, exact_rank


def build_audit(
    *,
    rms_dir: Path,
    output_dir: Path,
    split: str,
    anchor_manifest: Path,
    dct_pool: int,
    retrieval_candidates: int | None,
    min_overlap: float,
) -> None:
    summary = json.loads((rms_dir / "rms_summary.json").read_text())
    config = summary["config"]
    if retrieval_candidates is None:
        retrieval_candidates = int(config["retrieval_candidates"])
    retrieval_buffer = int(config["retrieval_buffer"])

    with np.load(rms_dir / f"{split}_rms_features.npz", allow_pickle=False) as features:
        arrays = {key: np.asarray(features[key]) for key in features.files}
    with np.load(rms_dir / f"{split}_rms_neighbors.npz", allow_pickle=False) as neighbours:
        stored_indices = np.asarray(neighbours["neighbor_indices"])
        stored_distances = np.asarray(neighbours["rms_distances"])

    anchors = _load_anchor_manifest(anchor_manifest)
    if dct_pool <= 0 or dct_pool > retrieval_candidates:
        raise ValueError(
            f"dct_pool must be in [1, {retrieval_candidates}], got {dct_pool}"
        )
    out_of_range = anchors[(anchors < 0) | (anchors >= len(arrays["sample_index"]))]
    if len(out_of_range):
        raise ValueError(f"Anchor manifest contains out-of-range indices: {out_of_range.tolist()}")

    normalized = arrays["normalized_sequence"].astype(np.float32)
    masks = arrays["aligned_mask"].astype(bool)
    eligible = arrays["eligible_mask"].astype(bool)
    strata = arrays["stratum_key"]
    scenario_ids = arrays["scenario_id"]
    descriptor = retrieval_descriptor(
        normalized,
        masks,
        coefficients=int(config["dct_coefficients"]),
        mask_weight=float(config["mask_descriptor_weight"]),
    )

    trees: dict[int, tuple[np.ndarray, cKDTree]] = {}
    for key in np.unique(strata[anchors]):
        members = np.flatnonzero(eligible & (strata == key)).astype(np.int64)
        if len(members) < 2:
            raise ValueError(f"Stratum {int(key)} has fewer than two eligible samples")
        trees[int(key)] = (members, cKDTree(descriptor[members]))

    median = arrays["normalization_median"].astype(np.float32)
    iqr = arrays["normalization_iqr"].astype(np.float32)
    positions = normalized[..., 0:2] * iqr[None, None, :, 0:2] + median[None, None, :, 0:2]
    event_index = int(np.argmin(np.abs(arrays["time_offsets"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fields = (
        "audit_order",
        "anchor_index",
        "negative_index",
        "dct_rank",
        "dct_distance",
        "exact_rms_rank_within_candidates",
        "exact_rms_distance",
        "rank1_rms_distance",
        "rank32_rms_distance",
        "negative_to_rank32_ratio",
        "common_valid_fraction",
        "anchor_scenario_id",
        "negative_scenario_id",
        "stratum_key",
        "anchor_event_mode",
        "negative_event_mode",
        "anchor_zone_pet_s",
        "negative_zone_pet_s",
    )
    cards: list[str] = []
    rows: list[dict[str, object]] = []

    for audit_order, anchor_value in enumerate(anchors):
        anchor = int(anchor_value)
        members, tree = trees[int(strata[anchor])]
        query_k = min(
            len(members),
            int(retrieval_candidates) + retrieval_buffer + 1,
        )
        dct_distances, local_indices = tree.query(
            descriptor[anchor],
            k=query_k,
            workers=1,
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
        selected_offset, exact_rank = _select_hard_negative(
            anchor=anchor,
            candidates=candidates,
            exact_distances=exact_distances,
            overlaps=overlaps,
            stored_neighbours=stored_indices,
            dct_pool=dct_pool,
            min_overlap=min_overlap,
        )
        negative = int(candidates[selected_offset])
        exact_distance = float(exact_distances[selected_offset])
        rank1_distance = float(stored_distances[anchor, 0])
        rank32_distance = float(stored_distances[anchor, stored_distances.shape[1] - 1])
        ratio = exact_distance / max(rank32_distance, 1e-8)

        anchor_position = positions[anchor]
        negative_position = positions[negative]
        anchor_mask = masks[anchor]
        negative_mask = masks[negative]
        bounds = _bounds(anchor_position, anchor_mask, negative_position, negative_mask)

        def label(index: int) -> str:
            types = "/".join(
                TYPE_NAMES.get(int(value), str(int(value)))
                for value in (
                    arrays["first_agent_type"][index],
                    arrays["second_agent_type"][index],
                )
            )
            return (
                f"{arrays['event_mode'][index]} | {types} | "
                f"PET={float(arrays['zone_pet_s'][index]):.1f}s"
            )

        svg = (
            '<svg viewBox="0 0 712 350" role="img" aria-label="DCT-near RMS-far hard negative">'
            + _panel_svg(
                anchor_position,
                anchor_mask,
                bounds=bounds,
                x_offset=0,
                title=f"anchor #{anchor}",
                subtitle=label(anchor),
                event_index=event_index,
            )
            + _panel_svg(
                negative_position,
                negative_mask,
                bounds=bounds,
                x_offset=362,
                title=f"hard negative #{negative}",
                subtitle=label(negative),
                event_index=event_index,
            )
            + "</svg>"
        )
        cards.append(
            '<article><div class="metric">'
            f"exact RMS {exact_distance:.4f} ({ratio:.2f}&times; rank-32 edge) &middot; "
            f"DCT rank {selected_offset + 1}/{retrieval_candidates} &middot; "
            f"exact rank {exact_rank}/{len(candidates)} &middot; overlap {float(overlaps[selected_offset]):.1%}"
            f'</div>{svg}<div class="scene">'
            f"rank-1 RMS {rank1_distance:.4f} &middot; rank-32 RMS {rank32_distance:.4f}<br>"
            f"anchor scene {html.escape(str(scenario_ids[anchor]))}<br>"
            f"negative scene {html.escape(str(scenario_ids[negative]))}</div></article>"
        )
        rows.append(
            {
                "audit_order": audit_order,
                "anchor_index": anchor,
                "negative_index": negative,
                "dct_rank": selected_offset + 1,
                "dct_distance": float(dct_distances[selected_offset]),
                "exact_rms_rank_within_candidates": exact_rank,
                "exact_rms_distance": exact_distance,
                "rank1_rms_distance": rank1_distance,
                "rank32_rms_distance": rank32_distance,
                "negative_to_rank32_ratio": ratio,
                "common_valid_fraction": float(overlaps[selected_offset]),
                "anchor_scenario_id": scenario_ids[anchor],
                "negative_scenario_id": scenario_ids[negative],
                "stratum_key": int(strata[anchor]),
                "anchor_event_mode": arrays["event_mode"][anchor],
                "negative_event_mode": arrays["event_mode"][negative],
                "anchor_zone_pet_s": float(arrays["zone_pet_s"][anchor]),
                "negative_zone_pet_s": float(arrays["zone_pet_s"][negative]),
            }
        )

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "index.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>DCT-near RMS-far hard-negative audit</title>"
        "<style>body{background:#0f1217;color:#eee;font-family:system-ui;margin:24px}"
        "h1{font-size:24px}p{color:#aeb7c4;max-width:1100px}.grid{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(720px,1fr));gap:18px}"
        "article{background:#222832;border:1px solid #394250;border-radius:12px;padding:12px}"
        "svg{width:100%;height:auto}.metric{font-size:16px;font-weight:650;margin-bottom:8px}"
        ".scene{font:11px ui-monospace;color:#aeb7c4;margin-top:7px}"
        "</style></head><body><h1>DCT-near, exact-RMS-far hard negatives</h1>"
        f"<p>{html.escape(split)} split &middot; anchors loaded in order from "
        f"{html.escape(str(anchor_manifest))}. For each anchor, search the first {dct_pool} "
        "of 1,024 same-stratum DCT candidates, require at least "
        f"{min_overlap:.0%} common-valid overlap, exclude stored top-neighbour edges in both "
        "directions, and show the remaining candidate with the largest exact RMS. "
        "Circle=start, square=end, star=aligned event, white dots=1 s ticks.</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>",
        encoding="utf-8",
    )
    print(f"saved {output_dir / 'index.html'} ({len(rows)} hard negatives)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rms_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--anchor_manifest", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dct_pool", type=int, default=256)
    parser.add_argument(
        "--retrieval_candidates",
        type=int,
        default=None,
        help="Defaults to the value recorded in rms_summary.json.",
    )
    parser.add_argument("--min_overlap", type=float, default=0.80)
    args = parser.parse_args()
    build_audit(
        rms_dir=args.rms_dir,
        output_dir=args.output_dir,
        split=args.split,
        anchor_manifest=args.anchor_manifest,
        dct_pool=args.dct_pool,
        retrieval_candidates=args.retrieval_candidates,
        min_overlap=args.min_overlap,
    )


if __name__ == "__main__":
    main()
