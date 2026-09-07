"""Retrieve interaction scenes from the full causally encodable validation corpus.

This report complements ``visualize_contrastive_retrieval_stages.py``.  The
older report ranks a small, fixed RMS-positive/relation-negative pool.  This
one scores every eligible validation pair, removes the anchor scenario, and
collapses pair rows by scenario before taking the global top-K.

The four representation stages are deliberately ordered by training depth:

1. ``raw_z``: flattened base-tokenizer z;
2. ``reader_z``: the hard Stage-A pair reader applied to that same base z;
3. ``hard``: the hard Stage-B encoder and pair reader;
4. ``hybrid``: the hybrid Stage-B encoder and pair reader.

Raw-z and reader+z share one base-encoder pass.  Only the Stage-A reader
weights are loaded for reader+z; the encoder comes from ``base_checkpoint``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .build_full_pair_rms_neighbors import exact_masked_rms
    from .train_interaction_contrastive import (
        _tensor_scene_from_npz,
        load_baseline_tokenizer,
    )
    from .visualize_contrastive_retrieval_stages import (
        _build_head,
        _page_radius,
        _stack_scenes,
        decode_reason_bits,
        physical_trajectory,
        select_anchor_rows,
        stratum_label,
        trajectory_svg,
    )
    from .visualize_full_pair_relation_negatives import relation_outcomes
except ImportError:
    from build_full_pair_rms_neighbors import exact_masked_rms  # type: ignore
    from train_interaction_contrastive import (  # type: ignore
        _tensor_scene_from_npz,
        load_baseline_tokenizer,
    )
    from visualize_contrastive_retrieval_stages import (  # type: ignore
        _build_head,
        _page_radius,
        _stack_scenes,
        decode_reason_bits,
        physical_trajectory,
        select_anchor_rows,
        stratum_label,
        trajectory_svg,
    )
    from visualize_full_pair_relation_negatives import relation_outcomes  # type: ignore


STAGE_ORDER = ("raw_z", "reader_z", "hard", "hybrid")
STAGE_LABELS = {
    "raw_z": "Raw z: base tokenizer",
    "reader_z": "Reader + z: hard Stage-A reader on base z",
    "hard": "Hard: Stage-B encoder + reader",
    "hybrid": "Hybrid: Stage-B encoder + reader",
}
STAGE_DESCRIPTIONS = {
    "raw_z": "Flattened and L2-normalized query-time z from the untouched base tokenizer.",
    "reader_z": "Pair representation produced by the hard Stage-A reader from the same untouched base z.",
    "hard": "Pair representation from the hard Stage-B checkpoint, after encoder fine-tuning.",
    "hybrid": "Pair representation from the hybrid-soft Stage-B checkpoint.",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def parse_anchor_indices(value: str) -> np.ndarray:
    rows = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not rows:
        raise ValueError("--anchor_indices did not contain any rows")
    if len(set(rows)) != len(rows):
        raise ValueError("--anchor_indices contains duplicate rows")
    return np.asarray(rows, dtype=np.int64)


def select_candidate_rows(
    cache: dict[str, np.ndarray],
    *,
    history_steps: int,
    scope: str,
) -> np.ndarray:
    """Select rows usable as retrieval candidates under an explicit policy."""
    query_ok = np.asarray(cache["query_step"], dtype=np.int64) >= int(history_steps) - 1
    if scope == "history":
        mask = query_ok
    elif scope == "causal":
        mask = query_ok & np.asarray(cache["causal_eligible_mask"], dtype=bool)
    elif scope == "training":
        mask = query_ok & np.asarray(cache["training_eligible_mask"], dtype=bool)
    else:
        raise ValueError(f"Unknown candidate scope: {scope}")
    return np.flatnonzero(mask).astype(np.int64)


def rank_unique_scenarios(
    *,
    anchor: int,
    candidate_rows: np.ndarray,
    similarities: np.ndarray,
    scenario_ids: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank all rows, exclude the query scene, and keep one pair per scene."""
    candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
    similarities = np.asarray(similarities, dtype=np.float32)
    scenario_ids = np.asarray(scenario_ids).astype(str)
    if similarities.shape != (len(candidate_rows),):
        raise ValueError(
            f"Expected {len(candidate_rows)} similarities, got {similarities.shape}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    anchor_scenario = scenario_ids[int(anchor)]
    order = np.argsort(-similarities, kind="stable")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for offset in order.tolist():
        row = int(candidate_rows[int(offset)])
        scenario = scenario_ids[row]
        score = float(similarities[int(offset)])
        if not np.isfinite(score) or scenario == anchor_scenario or scenario in seen:
            continue
        seen.add(scenario)
        result.append(
            {
                "rank": len(result) + 1,
                "row": row,
                "scenario_id": scenario,
                "cosine": score,
            }
        )
        if len(result) >= int(top_k):
            break
    return result


def _scene_batch(
    rows: np.ndarray,
    cache: dict[str, np.ndarray],
    history_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    scenes = [
        _tensor_scene_from_npz(
            str(cache["source_path"][row]),
            query_step=int(cache["query_step"][row]),
            history_steps=int(history_steps),
            first_agent_id=int(cache["first_agent_id"][row]),
            second_agent_id=int(cache["second_agent_id"][row]),
        )
        for row in np.asarray(rows, dtype=np.int64).tolist()
    ]
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in _stack_scenes(scenes).items()
    }


@torch.inference_mode()
def _encode_batch(
    *,
    tokenizer: torch.nn.Module,
    heads: dict[str, torch.nn.Module | None],
    rows: np.ndarray,
    cache: dict[str, np.ndarray],
    history_steps: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = _scene_batch(rows, cache, history_steps, device)
    output = tokenizer.encoder(
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    z_current = output.z[:, -1].float()
    embeddings: dict[str, torch.Tensor] = {}
    for stage, head in heads.items():
        if head is None:
            embeddings[stage] = F.normalize(z_current.flatten(1), dim=-1, eps=1e-6)
        else:
            embeddings[stage], _ = head(
                z_current, batch["first_slot"], batch["second_slot"]
            )
    return embeddings


def score_encoder_group(
    *,
    encoder_checkpoint_path: Path,
    reader_checkpoint_paths: dict[str, Path | None],
    anchor_rows: np.ndarray,
    candidate_rows: np.ndarray,
    cache: dict[str, np.ndarray],
    history_steps: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Score one encoder and one or more views/readers without storing a bank."""
    print(f"loading encoder {encoder_checkpoint_path}", flush=True)
    tokenizer, model_args, encoder_checkpoint = load_baseline_tokenizer(
        encoder_checkpoint_path, device
    )
    tokenizer.eval()
    heads: dict[str, torch.nn.Module | None] = {}
    reader_checkpoints: dict[str, dict[str, Any] | None] = {}
    for stage, reader_path in reader_checkpoint_paths.items():
        if reader_path is None:
            heads[stage] = None
            reader_checkpoints[stage] = None
            continue
        if reader_path.resolve() == encoder_checkpoint_path.resolve():
            checkpoint = encoder_checkpoint
        else:
            checkpoint = torch.load(reader_path, map_location="cpu")
        heads[stage] = _build_head(tokenizer, checkpoint, model_args, device)
        reader_checkpoints[stage] = checkpoint

    anchor_embeddings = _encode_batch(
        tokenizer=tokenizer,
        heads=heads,
        rows=anchor_rows,
        cache=cache,
        history_steps=history_steps,
        device=device,
    )
    score_matrices = {
        stage: np.empty((len(anchor_rows), len(candidate_rows)), dtype=np.float32)
        for stage in heads
    }
    for start in range(0, len(candidate_rows), int(batch_size)):
        batch_rows = candidate_rows[start : start + int(batch_size)]
        candidate_embeddings = _encode_batch(
            tokenizer=tokenizer,
            heads=heads,
            rows=batch_rows,
            cache=cache,
            history_steps=history_steps,
            device=device,
        )
        stop = start + len(batch_rows)
        for stage in heads:
            scores = anchor_embeddings[stage] @ candidate_embeddings[stage].T
            score_matrices[stage][:, start:stop] = scores.float().cpu().numpy()
        print(
            f"[{'+'.join(heads)}] scored {stop}/{len(candidate_rows)} candidate pairs",
            flush=True,
        )

    metadata: dict[str, dict[str, Any]] = {}
    for stage, reader_path in reader_checkpoint_paths.items():
        reader_checkpoint = reader_checkpoints[stage]
        metadata[stage] = {
            "label": STAGE_LABELS[stage],
            "definition": STAGE_DESCRIPTIONS[stage],
            "encoder_checkpoint": str(encoder_checkpoint_path.resolve()),
            "encoder_checkpoint_step": int(encoder_checkpoint.get("step", -1)),
            "reader_checkpoint": (
                None if reader_path is None else str(reader_path.resolve())
            ),
            "reader_checkpoint_step": (
                None
                if reader_checkpoint is None
                else int(reader_checkpoint.get("step", -1))
            ),
            "embedding_dimension": int(anchor_embeddings[stage].shape[1]),
        }

    del heads, reader_checkpoints, anchor_embeddings, tokenizer, encoder_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score_matrices, metadata


def _valid_pool(cache: dict[str, np.ndarray], key: str, anchor: int) -> np.ndarray:
    values = np.asarray(cache[key][int(anchor)], dtype=np.int64)
    return values[values >= 0]


def _relation_bits(
    anchor_outcome: dict[str, np.ndarray],
    candidate_outcomes: dict[str, np.ndarray],
) -> np.ndarray:
    swap_diff = candidate_outcomes["order_swap"] != bool(anchor_outcome["order_swap"][0])
    future_order_diff = (
        candidate_outcomes["order_outcome"] != int(anchor_outcome["order_outcome"][0])
    )
    future_order_opposite = (
        candidate_outcomes["order_outcome"] * int(anchor_outcome["order_outcome"][0])
        == -1
    )
    gap_opposite = (
        candidate_outcomes["gap_trend"] * int(anchor_outcome["gap_trend"][0]) == -1
    )
    distance_opposite = (
        candidate_outcomes["distance_trend"]
        * int(anchor_outcome["distance_trend"][0])
        == -1
    )
    return (
        swap_diff.astype(np.uint8)
        + future_order_diff.astype(np.uint8) * 2
        + future_order_opposite.astype(np.uint8) * 4
        + gap_opposite.astype(np.uint8) * 8
        + distance_opposite.astype(np.uint8) * 16
    )


def annotate_rows(
    *,
    anchor: int,
    rows: np.ndarray,
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    min_pair_overlap: float,
    relation_outcome_steps: int,
    relation_margin_m: float,
) -> dict[int, dict[str, Any]]:
    rows = np.asarray(sorted(set(np.asarray(rows, dtype=np.int64).tolist())), dtype=np.int64)
    distances, overlaps = exact_masked_rms(
        features["normalized_sequence"],
        features["aligned_mask"],
        int(anchor),
        rows,
        min_pair_overlap=float(min_pair_overlap),
    )
    positives = set(_valid_pool(cache, "positive_indices", anchor).tolist())
    negatives = set(_valid_pool(cache, "negative_indices", anchor).tolist())
    reasons_by_row: dict[int, str] = {}
    event_index = int(np.argmin(np.abs(np.asarray(features["time_offsets"], dtype=np.float32))))
    outcome_index = event_index + int(relation_outcome_steps)
    masks = np.asarray(features["aligned_mask"], dtype=bool)
    if outcome_index < masks.shape[1]:
        endpoint_valid = (
            masks[rows, event_index].all(axis=1)
            & masks[rows, outcome_index].all(axis=1)
            & bool(masks[int(anchor), event_index].all())
            & bool(masks[int(anchor), outcome_index].all())
        )
        relation_rows = rows[endpoint_valid]
        if len(relation_rows):
            normalized = np.asarray(features["normalized_sequence"], dtype=np.float32)
            median = np.asarray(features["normalization_median"], dtype=np.float32)
            iqr = np.asarray(features["normalization_iqr"], dtype=np.float32)
            positions = normalized[..., :2] * iqr[None, None, :, :2] + median[
                None, None, :, :2
            ]
            anchor_outcome = relation_outcomes(
                positions,
                np.asarray([anchor], dtype=np.int64),
                event_index=event_index,
                outcome_index=outcome_index,
                margin_m=float(relation_margin_m),
            )
            candidate_outcomes = relation_outcomes(
                positions,
                relation_rows,
                event_index=event_index,
                outcome_index=outcome_index,
                margin_m=float(relation_margin_m),
            )
            bits = _relation_bits(anchor_outcome, candidate_outcomes)
            reasons_by_row = {
                int(row): decode_reason_bits(int(reason)) if int(reason) else "same outcome"
                for row, reason in zip(relation_rows.tolist(), bits.tolist())
            }

    annotations: dict[int, dict[str, Any]] = {}
    for row, distance, overlap in zip(rows.tolist(), distances.tolist(), overlaps.tolist()):
        stored_label = "positive" if row in positives else "negative" if row in negatives else "unlabelled"
        annotations[int(row)] = {
            "stored_label": stored_label,
            "same_stratum": bool(cache["stratum_key"][row] == cache["stratum_key"][anchor]),
            "exact_rms": None if not np.isfinite(distance) else float(distance),
            "common_valid_fraction": float(overlap),
            "relation_outcome": reasons_by_row.get(int(row)),
        }
    return annotations


def _safe_median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def retrieval_metrics(
    *,
    anchor: int,
    results: list[dict[str, Any]],
    annotations: dict[int, dict[str, Any]],
    cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    scenario_ids = np.asarray(cache["scenario_id"]).astype(str)
    positive_rows = _valid_pool(cache, "positive_indices", anchor)
    positive_scenarios = {
        scenario_ids[int(row)]
        for row in positive_rows.tolist()
        if scenario_ids[int(row)] != scenario_ids[int(anchor)]
    }
    retrieved_scenarios = {str(result["scenario_id"]) for result in results}
    rms_values = [
        float(annotations[int(result["row"])]["exact_rms"])
        for result in results
        if annotations[int(result["row"])]["exact_rms"] is not None
    ]
    same_stratum = [
        bool(annotations[int(result["row"])]["same_stratum"]) for result in results
    ]
    relation_known = [
        annotations[int(result["row"])]["relation_outcome"]
        for result in results
        if annotations[int(result["row"])]["relation_outcome"] is not None
    ]
    return {
        "rms_positive_scene_recall_at_k": (
            None
            if not positive_scenarios
            else float(len(positive_scenarios & retrieved_scenarios) / len(positive_scenarios))
        ),
        "same_stratum_fraction_at_k": (
            None if not same_stratum else float(np.mean(same_stratum))
        ),
        "median_exact_rms_at_k": _safe_median(rms_values),
        "finite_rms_count_at_k": len(rms_values),
        "relation_difference_fraction_at_k": (
            None
            if not relation_known
            else float(np.mean([value != "same outcome" for value in relation_known]))
        ),
        "relation_known_count_at_k": len(relation_known),
        "stored_positive_count_at_k": sum(
            annotations[int(result["row"])]["stored_label"] == "positive"
            for result in results
        ),
        "stored_negative_count_at_k": sum(
            annotations[int(result["row"])]["stored_label"] == "negative"
            for result in results
        ),
    }


def _format_optional(value: Any, format_spec: str) -> str:
    return "n/a" if value is None else format(value, format_spec)


def _page_css() -> str:
    return """
body{font:14px system-ui;margin:22px;background:#eef2f7;color:#172033}
a{color:#1659b7}.note,.metrics,.stage{background:white;border:1px solid #d7dee8;border-radius:10px;padding:14px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:10px;overflow-x:auto}
.candidate{min-width:220px;border:2px solid #d9e0e9;border-radius:10px;padding:7px;background:#f8fafc}
.candidate svg{width:100%;height:auto;display:block}.candidate p{font-size:11px;line-height:1.4;margin:7px 3px 2px}
.candidate.positive{border-color:#65a879}.candidate.negative{border-color:#d87979}.candidate.anchor{border-color:#7085bd}
.badge{font-weight:750;border-radius:10px;padding:2px 6px;margin-right:5px;background:#e2e8f0}
table{border-collapse:collapse}th,td{border:1px solid #d6dde7;padding:6px 9px;text-align:right}th:first-child{text-align:left}
.small,.legend{font-size:12px;color:#526174}
"""


def _anchor_card(
    *, anchor: int, cache: dict[str, np.ndarray], features: dict[str, np.ndarray], radius: float
) -> str:
    positions, mask = physical_trajectory(features, anchor)
    svg = trajectory_svg(
        positions,
        mask,
        radius=radius,
        title="ANCHOR",
        subtitle=f"row {anchor} · {str(cache['scenario_id'][anchor])}",
    )
    return f'<article class="candidate anchor">{svg}<p>fixed query pair</p></article>'


def _result_card(
    *,
    result: dict[str, Any],
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    radius: float,
) -> str:
    row = int(result["row"])
    positions, mask = physical_trajectory(features, row)
    svg = trajectory_svg(
        positions,
        mask,
        radius=radius,
        title=f"rank {int(result['rank'])} · row {row}",
        subtitle=f"{str(result['scenario_id'])} · {stratum_label(int(cache['stratum_key'][row]))}",
    )
    label = str(result["stored_label"])
    card_class = label if label in ("positive", "negative") else "unlabelled"
    rms = "n/a" if result["exact_rms"] is None else f"{float(result['exact_rms']):.3f}"
    relation = result["relation_outcome"] or "future unavailable"
    same_stratum = "same stratum" if result["same_stratum"] else "cross stratum"
    return (
        f'<article class="candidate {card_class}">{svg}<p>'
        f'<span class="badge">{html.escape(label)}</span>'
        f'cosine {float(result["cosine"]):.4f} · RMS {rms} · overlap {float(result["common_valid_fraction"]):.1%}<br>'
        f'{html.escape(same_stratum)} · {html.escape(str(relation))}</p></article>'
    )


def write_anchor_page(
    *,
    path: Path,
    anchor: int,
    stage_results: dict[str, list[dict[str, Any]]],
    stage_metrics: dict[str, dict[str, Any]],
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    corpus_rows: int,
    corpus_scenarios: int,
) -> None:
    displayed_rows = [anchor]
    for results in stage_results.values():
        displayed_rows.extend(int(result["row"]) for result in results)
    radius = _page_radius(displayed_rows, features)
    metric_rows = []
    sections = []
    for stage in STAGE_ORDER:
        metrics = stage_metrics[stage]
        metric_rows.append(
            f"<tr><th>{html.escape(STAGE_LABELS[stage])}</th>"
            f"<td>{_format_optional(metrics['rms_positive_scene_recall_at_k'], '.1%')}</td>"
            f"<td>{_format_optional(metrics['same_stratum_fraction_at_k'], '.1%')}</td>"
            f"<td>{_format_optional(metrics['median_exact_rms_at_k'], '.3f')}</td>"
            f"<td>{_format_optional(metrics['relation_difference_fraction_at_k'], '.1%')}</td></tr>"
        )
        cards = [_anchor_card(anchor=anchor, cache=cache, features=features, radius=radius)]
        cards.extend(
            _result_card(result=result, cache=cache, features=features, radius=radius)
            for result in stage_results[stage]
        )
        sections.append(
            f'<section class="stage"><h2>{html.escape(STAGE_LABELS[stage])}</h2>'
            f'<p class="small">{html.escape(STAGE_DESCRIPTIONS[stage])}</p>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )
    path.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<title>Global retrieval anchor {anchor}</title><style>{_page_css()}</style></head><body>'
        '<p><a href="index.html">← all anchors</a></p>'
        f'<h1>Anchor {anchor}: raw z → reader + z → hard → hybrid</h1>'
        f'<p class="legend">{html.escape(stratum_label(int(cache["stratum_key"][anchor])))} · '
        f'scenario {html.escape(str(cache["scenario_id"][anchor]))} · query step {int(cache["query_step"][anchor])}</p>'
        f'<div class="note"><strong>Global retrieval.</strong> Every representation searches {corpus_rows:,} causally encodable pair rows '
        f'across {corpus_scenarios:,} physical scenarios. The anchor scenario is excluded and only the best pair from each candidate scenario is kept. '
        'Green/red borders mean membership in the old stored RMS-positive/relation-negative pools; grey results were not labelled by that small cache.</div>'
        '<div class="metrics"><table><thead><tr><th>representation</th><th>RMS-positive scene recall@K</th>'
        '<th>same stratum@K</th><th>median RMS@K</th><th>relation differs@K</th></tr></thead>'
        f'<tbody>{"".join(metric_rows)}</tbody></table></div>{"".join(sections)}</body></html>'
    )


def write_index(
    *,
    path: Path,
    reports: list[dict[str, Any]],
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    corpus_rows: int,
    corpus_scenarios: int,
) -> None:
    cards = []
    for position, report in enumerate(reports, start=1):
        anchor = int(report["anchor_row"])
        points, mask = physical_trajectory(features, anchor)
        radius = _page_radius([anchor], features)
        svg = trajectory_svg(
            points,
            mask,
            radius=radius,
            title=f"{position:02d} · anchor {anchor}",
            subtitle=str(report["scenario_id"]),
        )
        cards.append(
            f'<article><a href="{html.escape(str(report["page"]))}">{svg}</a>'
            f'<h3><a href="{html.escape(str(report["page"]))}">{html.escape(str(report["stratum"]))}</a></h3>'
            f'<p>query step {int(report["query_step"])}</p></article>'
        )
    path.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Global contrastive retrieval</title>'
        f'<style>{_page_css()} .anchors{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}}'
        '.anchors article{background:white;border:1px solid #d7dee8;border-radius:10px;padding:10px}.anchors svg{width:100%}</style>'
        '</head><body><h1>Global interaction retrieval: raw z → reader + z → hard → hybrid</h1>'
        f'<div class="note"><p><strong>Ten fixed anchors, full valid corpus.</strong> Each stage searches {corpus_rows:,} pair rows '
        f'covering {corpus_scenarios:,} physical scenarios, then returns unique scenarios.</p>'
        '<p>Reader + z uses the hard Stage-A reader on untouched base-tokenizer z. Hard and hybrid use their respective Stage-B encoders and readers.</p></div>'
        f'<div class="anchors">{"".join(cards)}</div></body></html>'
    )


def _write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fields = [
        "anchor_row",
        "scenario_id",
        "stratum_key",
        "stage",
        "rms_positive_scene_recall_at_k",
        "same_stratum_fraction_at_k",
        "median_exact_rms_at_k",
        "relation_difference_fraction_at_k",
        "stored_positive_count_at_k",
        "stored_negative_count_at_k",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for stage in STAGE_ORDER:
                writer.writerow(
                    {
                        "anchor_row": report["anchor_row"],
                        "scenario_id": report["scenario_id"],
                        "stratum_key": report["stratum_key"],
                        "stage": stage,
                        **{
                            key: report["metrics"][stage][key]
                            for key in fields[4:]
                        },
                    }
                )


def build(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    cache = _load_npz(args.cache)
    features = _load_npz(args.rms_features)
    if len(cache["scenario_id"]) != len(features["scenario_id"]):
        raise ValueError("Contrastive cache and RMS feature rows do not align")

    if args.anchor_indices:
        anchor_rows = parse_anchor_indices(args.anchor_indices)
    else:
        manifest = _load_npz(args.validation_manifest)
        anchor_rows = select_anchor_rows(
            manifest["cache_row"], cache["stratum_key"], args.num_anchors
        )
    if np.any(anchor_rows < 0) or np.any(anchor_rows >= len(cache["scenario_id"])):
        raise IndexError("An anchor row lies outside the validation cache")
    if np.any(cache["query_step"][anchor_rows] < args.history_steps - 1):
        raise ValueError("At least one anchor lacks the requested causal history")

    candidate_rows = select_candidate_rows(
        cache, history_steps=args.history_steps, scope=args.candidate_scope
    )
    if args.candidate_limit > 0:
        candidate_rows = candidate_rows[: int(args.candidate_limit)]
        candidate_rows = np.asarray(
            sorted(set(candidate_rows.tolist()) | set(anchor_rows.tolist())), dtype=np.int64
        )
    if not len(candidate_rows):
        raise RuntimeError("Candidate selection produced no rows")
    corpus_scenarios = len(
        np.unique(np.asarray(cache["scenario_id"])[candidate_rows].astype(str))
    )
    print(
        f"anchors={anchor_rows.tolist()} candidates={len(candidate_rows)} "
        f"scenarios={corpus_scenarios} scope={args.candidate_scope} device={device}",
        flush=True,
    )

    all_scores: dict[str, np.ndarray] = {}
    stage_metadata: dict[str, dict[str, Any]] = {}
    base_scores, base_metadata = score_encoder_group(
        encoder_checkpoint_path=args.base_checkpoint,
        reader_checkpoint_paths={
            "raw_z": None,
            "reader_z": args.reader_checkpoint,
        },
        anchor_rows=anchor_rows,
        candidate_rows=candidate_rows,
        cache=cache,
        history_steps=args.history_steps,
        batch_size=args.batch_size,
        device=device,
    )
    all_scores.update(base_scores)
    stage_metadata.update(base_metadata)
    for stage, checkpoint_path in (
        ("hard", args.hard_checkpoint),
        ("hybrid", args.hybrid_checkpoint),
    ):
        scores, metadata = score_encoder_group(
            encoder_checkpoint_path=checkpoint_path,
            reader_checkpoint_paths={stage: checkpoint_path},
            anchor_rows=anchor_rows,
            candidate_rows=candidate_rows,
            cache=cache,
            history_steps=args.history_steps,
            batch_size=args.batch_size,
            device=device,
        )
        all_scores.update(scores)
        stage_metadata.update(metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "retrieval_scores.npz",
        anchor_rows=anchor_rows,
        candidate_rows=candidate_rows,
        **{stage: all_scores[stage] for stage in STAGE_ORDER},
    )

    reports: list[dict[str, Any]] = []
    scenario_ids = np.asarray(cache["scenario_id"]).astype(str)
    for anchor_position, anchor_value in enumerate(anchor_rows.tolist(), start=1):
        anchor = int(anchor_value)
        stage_results: dict[str, list[dict[str, Any]]] = {}
        selected_rows: set[int] = set()
        for stage in STAGE_ORDER:
            results = rank_unique_scenarios(
                anchor=anchor,
                candidate_rows=candidate_rows,
                similarities=all_scores[stage][anchor_position - 1],
                scenario_ids=scenario_ids,
                top_k=args.top_k,
            )
            if len(results) < args.top_k:
                raise RuntimeError(
                    f"Only {len(results)} unique scenarios available for anchor {anchor} stage {stage}"
                )
            stage_results[stage] = results
            selected_rows.update(int(result["row"]) for result in results)
        annotations = annotate_rows(
            anchor=anchor,
            rows=np.asarray(sorted(selected_rows), dtype=np.int64),
            cache=cache,
            features=features,
            min_pair_overlap=args.min_pair_overlap,
            relation_outcome_steps=args.relation_outcome_steps,
            relation_margin_m=args.relation_margin_m,
        )
        stage_metrics: dict[str, dict[str, Any]] = {}
        for stage in STAGE_ORDER:
            for result in stage_results[stage]:
                result.update(annotations[int(result["row"])])
            stage_metrics[stage] = retrieval_metrics(
                anchor=anchor,
                results=stage_results[stage],
                annotations=annotations,
                cache=cache,
            )
        page = args.output_dir / f"anchor_{anchor_position:02d}_idx{anchor:05d}.html"
        write_anchor_page(
            path=page,
            anchor=anchor,
            stage_results=stage_results,
            stage_metrics=stage_metrics,
            cache=cache,
            features=features,
            corpus_rows=len(candidate_rows),
            corpus_scenarios=corpus_scenarios,
        )
        reports.append(
            {
                "anchor_row": anchor,
                "sample_index": int(cache["sample_index"][anchor]),
                "scenario_id": str(cache["scenario_id"][anchor]),
                "stratum_key": int(cache["stratum_key"][anchor]),
                "stratum": stratum_label(int(cache["stratum_key"][anchor])),
                "query_step": int(cache["query_step"][anchor]),
                "page": page.name,
                "metrics": stage_metrics,
                "retrievals": stage_results,
            }
        )
        print(f"rendered {page.name}", flush=True)

    _write_summary_csv(args.output_dir / "summary.csv", reports)
    manifest = {
        "definition": "Full-corpus exact cosine retrieval followed by scenario-level max-pair deduplication.",
        "stage_order": list(STAGE_ORDER),
        "device": str(device),
        "history_steps": int(args.history_steps),
        "top_k": int(args.top_k),
        "candidate_scope": args.candidate_scope,
        "candidate_pair_rows": int(len(candidate_rows)),
        "candidate_unique_scenarios": int(corpus_scenarios),
        "same_scenario_excluded": True,
        "scenario_aggregation": "maximum pair cosine per scenario",
        "validation_manifest": str(args.validation_manifest.resolve()),
        "contrastive_cache": str(args.cache.resolve()),
        "rms_features": str(args.rms_features.resolve()),
        "stages": stage_metadata,
        "anchors": reports,
    }
    (args.output_dir / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_index(
        path=args.output_dir / "index.html",
        reports=reports,
        cache=cache,
        features=features,
        corpus_rows=len(candidate_rows),
        corpus_scenarios=corpus_scenarios,
    )
    print(f"saved {args.output_dir / 'index.html'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    waymo_root = repo_root / "waymo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        default=waymo_root
        / "checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt",
    )
    parser.add_argument(
        "--reader_checkpoint",
        type=Path,
        default=waymo_root
        / "checkpoints/interaction_contrastive_hard_relneg_dupfiltered_v1/best_stage_a.pt",
        help="Hard Stage-A reader. Its encoder weights are intentionally ignored.",
    )
    parser.add_argument(
        "--hard_checkpoint",
        type=Path,
        default=waymo_root
        / "checkpoints/interaction_contrastive_hard_relneg_dupfiltered_v1/best.pt",
    )
    parser.add_argument(
        "--hybrid_checkpoint",
        type=Path,
        default=waymo_root
        / "checkpoints/interaction_contrastive_hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3/best.pt",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=waymo_root
        / "cache/interaction_full_pairs_50k_v2_contrastive_v1/val_contrastive_training.npz",
    )
    parser.add_argument(
        "--rms_features",
        type=Path,
        default=waymo_root
        / "cache/interaction_full_pairs_50k_v2_no_topk_rms_v0/val_rms_features.npz",
    )
    parser.add_argument(
        "--validation_manifest",
        type=Path,
        default=waymo_root
        / "checkpoints/interaction_contrastive_hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3/validation_manifest.npz",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=waymo_root / "eval_results/interaction_contrastive_global_retrieval_10",
    )
    parser.add_argument("--num_anchors", type=int, default=10)
    parser.add_argument(
        "--anchor_indices",
        default="",
        help="Optional comma-separated cache rows; overrides manifest selection.",
    )
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--history_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--candidate_scope",
        choices=("history", "causal", "training"),
        default="history",
        help="history is the widest scientifically valid corpus: every row with 32 causal steps.",
    )
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=0,
        help="Testing only; 0 scores the complete selected candidate scope.",
    )
    parser.add_argument("--min_pair_overlap", type=float, default=0.70)
    parser.add_argument("--relation_outcome_steps", type=int, default=20)
    parser.add_argument("--relation_margin_m", type=float, default=2.0)
    return parser


def main() -> None:
    build(build_argparser().parse_args())


if __name__ == "__main__":
    main()
