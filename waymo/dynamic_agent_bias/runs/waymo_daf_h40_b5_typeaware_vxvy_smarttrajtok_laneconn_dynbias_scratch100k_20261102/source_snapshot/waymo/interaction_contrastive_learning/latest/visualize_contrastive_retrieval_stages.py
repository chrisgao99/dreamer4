"""Build a trajectory gallery for raw-z, hard, and hybrid retrieval stages.

The positive and negative labels in this report are fixed by the shared
event-aligned RMS/relation cache.  A model does not relabel candidates.  For
each stage, the report instead shows which ground-truth positive it regards as
most/least similar and which ground-truth negative is hardest/easiest.

The raw-z baseline is the flattened query-time tokenizer latent, L2-normalized
as one vector.  Hard and hybrid use their learned pair-specific contrastive
heads.  All three stages rank exactly the same candidate pools.
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
    from .train_interaction_contrastive import (
        ArbitraryPairContrastiveHead,
        _tensor_scene_from_npz,
        load_baseline_tokenizer,
        select_stratified_validation_anchors,
    )
except ImportError:
    from train_interaction_contrastive import (  # type: ignore
        ArbitraryPairContrastiveHead,
        _tensor_scene_from_npz,
        load_baseline_tokenizer,
        select_stratified_validation_anchors,
    )


TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
STAGE_LABELS = {
    "raw_z": "Before contrastive learning: raw tokenizer z",
    "hard": "After hard contrastive learning",
    "hybrid": "After hybrid-soft contrastive learning",
}
REASON_BITS = (
    (1, "order swap"),
    (2, "future order differs"),
    (4, "future order opposite"),
    (8, "gap trend opposite"),
    (16, "distance trend opposite"),
)
AGENT_COLORS = ("#42a5f5", "#ff9f43")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, argparse.Namespace):
        return vars(value)
    return dict(value) if isinstance(value, dict) else {}


def decode_reason_bits(value: int) -> str:
    labels = [label for bit, label in REASON_BITS if int(value) & bit]
    return ", ".join(labels) if labels else "unspecified relation change"


def pairwise_ranking_accuracy(similarities: np.ndarray, distances: np.ndarray) -> float:
    """Fraction of unequal-distance positive pairs in the correct order."""
    similarities = np.asarray(similarities, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    correct = 0
    count = 0
    for first in range(len(similarities)):
        for second in range(first + 1, len(similarities)):
            if not np.isfinite(distances[first]) or not np.isfinite(distances[second]):
                continue
            if abs(float(distances[first] - distances[second])) <= 1e-9:
                continue
            nearer, farther = (
                (first, second)
                if distances[first] < distances[second]
                else (second, first)
            )
            correct += int(similarities[nearer] > similarities[farther])
            count += 1
    return float(correct / count) if count else float("nan")


def choose_stage_examples(
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    similarities: dict[int, float],
) -> list[tuple[str, int]]:
    """Return positive-best/worst and negative-hardest/easiest examples."""
    positive_rows = np.asarray(positive_rows, dtype=np.int64)
    negative_rows = np.asarray(negative_rows, dtype=np.int64)
    positive_scores = np.asarray([similarities[int(row)] for row in positive_rows])
    negative_scores = np.asarray([similarities[int(row)] for row in negative_rows])
    return [
        ("most similar GT positive", int(positive_rows[int(np.argmax(positive_scores))])),
        ("least similar GT positive", int(positive_rows[int(np.argmin(positive_scores))])),
        ("hardest GT negative", int(negative_rows[int(np.argmax(negative_scores))])),
        ("easiest GT negative", int(negative_rows[int(np.argmin(negative_scores))])),
    ]


def choose_rms_examples(
    positive_rows: np.ndarray,
    positive_distances: np.ndarray,
    negative_rows: np.ndarray,
    negative_distances: np.ndarray,
) -> list[tuple[str, int]]:
    positive_order = np.argsort(positive_distances, kind="stable")
    negative_order = np.argsort(negative_distances, kind="stable")
    return [
        ("nearest RMS positive", int(positive_rows[int(positive_order[0])])),
        ("furthest stored RMS positive", int(positive_rows[int(positive_order[-1])])),
        ("closest relation negative", int(negative_rows[int(negative_order[0])])),
        ("furthest stored relation negative", int(negative_rows[int(negative_order[-1])])),
    ]


def _valid_pool(cache: dict[str, np.ndarray], key: str, row: int) -> np.ndarray:
    values = np.asarray(cache[key][int(row)], dtype=np.int64)
    return values[values >= 0]


def _candidate_distances(
    cache: dict[str, np.ndarray], key: str, row: int, count: int
) -> np.ndarray:
    values = np.asarray(cache[key][int(row)], dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) != count:
        raise RuntimeError(
            f"Candidate/distance mismatch for row={row}: {count} candidates, "
            f"{len(values)} finite {key} values"
        )
    return values


def select_anchor_rows(
    manifest_rows: np.ndarray,
    stratum_key: np.ndarray,
    count: int,
) -> np.ndarray:
    """Use the training-time deterministic stratifier on the fixed manifest."""
    return select_stratified_validation_anchors(
        np.asarray(manifest_rows, dtype=np.int64),
        np.asarray(stratum_key),
        int(count),
    )


def _stack_scenes(scenes: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = (
        "agents",
        "agent_mask",
        "map_polylines",
        "map_mask",
        "lights",
        "light_mask",
        "first_slot",
        "second_slot",
    )
    return {key: torch.stack([scene[key] for scene in scenes]) for key in keys}


def _build_head(
    tokenizer: torch.nn.Module,
    checkpoint: dict[str, Any],
    model_args: argparse.Namespace,
    device: torch.device,
) -> ArbitraryPairContrastiveHead:
    contrastive_args = _as_dict(checkpoint.get("contrastive_args", {}))
    state = checkpoint.get("contrastive_head")
    if not state:
        raise ValueError("Learned retrieval stage requires a contrastive_head")
    n_agents = int(state["slot_queries"].shape[0])
    head = ArbitraryPairContrastiveHead(
        d_bottleneck=int(model_args.d_bottleneck),
        d_model=int(contrastive_args.get("head_dim", 256)),
        n_heads=int(contrastive_args.get("head_heads", 4)),
        n_latents=int(model_args.n_latents),
        n_agents=n_agents,
        embedding_dim=int(contrastive_args.get("embedding_dim", 128)),
        dropout=float(contrastive_args.get("head_dropout", 0.0)),
        scale_pos_embeds=bool(model_args.scale_pos_embeds),
    )
    head.load_state_dict(state, strict=True)
    head.to(device).eval()
    return head


@torch.inference_mode()
def encode_stage(
    *,
    stage: str,
    checkpoint_path: Path,
    sample_rows: np.ndarray,
    cache: dict[str, np.ndarray],
    history_steps: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    print(f"[{stage}] loading {checkpoint_path}", flush=True)
    tokenizer, model_args, checkpoint = load_baseline_tokenizer(checkpoint_path, device)
    tokenizer.eval()
    head = None if stage == "raw_z" else _build_head(tokenizer, checkpoint, model_args, device)
    embeddings: dict[int, np.ndarray] = {}
    rows = np.asarray(sample_rows, dtype=np.int64)
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start : start + int(batch_size)]
        scenes = [
            _tensor_scene_from_npz(
                str(cache["source_path"][row]),
                query_step=int(cache["query_step"][row]),
                history_steps=int(history_steps),
                first_agent_id=int(cache["first_agent_id"][row]),
                second_agent_id=int(cache["second_agent_id"][row]),
            )
            for row in batch_rows
        ]
        batch = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in _stack_scenes(scenes).items()
        }
        output = tokenizer.encoder(
            agents=batch["agents"],
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_mask=batch["map_mask"],
            lights=batch["lights"],
            light_mask=batch["light_mask"],
        )
        z_current = output.z[:, -1].float()
        if stage == "raw_z":
            # Latent slots have fixed learned identities, so flattening retains
            # slot correspondence rather than introducing an arbitrary pooling.
            embedding = F.normalize(z_current.flatten(1), dim=-1, eps=1e-6)
        else:
            assert head is not None
            embedding, _ = head(
                z_current, batch["first_slot"], batch["second_slot"]
            )
        for row, vector in zip(batch_rows.tolist(), embedding.cpu().numpy()):
            embeddings[int(row)] = np.asarray(vector, dtype=np.float32)
        print(
            f"[{stage}] encoded {min(start + len(batch_rows), len(rows))}/{len(rows)}",
            flush=True,
        )
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "embedding_dimension": int(next(iter(embeddings.values())).shape[0]),
        "definition": (
            "flattened, L2-normalized query-time tokenizer z"
            if stage == "raw_z"
            else "L2-normalized pair-specific contrastive-head embedding"
        ),
    }
    del head, tokenizer, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings, metadata


def cosine_lookup(
    anchor: int,
    candidates: np.ndarray,
    embeddings: dict[int, np.ndarray],
) -> dict[int, float]:
    anchor_embedding = embeddings[int(anchor)]
    return {
        int(row): float(np.dot(anchor_embedding, embeddings[int(row)]))
        for row in np.asarray(candidates, dtype=np.int64)
    }


def _segments(valid: np.ndarray, start: int, stop: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    segment_start = None
    for index in range(start, stop):
        if bool(valid[index]) and segment_start is None:
            segment_start = index
        at_end = index == stop - 1
        if segment_start is not None and (not bool(valid[index]) or at_end):
            end = index + 1 if bool(valid[index]) and at_end else index
            if end - segment_start >= 2:
                result.append((segment_start, end))
            segment_start = None
    return result


def trajectory_svg(
    positions: np.ndarray,
    mask: np.ndarray,
    *,
    radius: float,
    title: str,
    subtitle: str,
    event_index: int = 19,
) -> str:
    """Render a compact aligned trajectory; future is dashed."""
    width, height, pad = 330, 285, 28

    def xy(point: np.ndarray) -> tuple[float, float]:
        x = pad + (float(point[0]) + radius) / (2.0 * radius) * (width - 2 * pad)
        y = 42 + (radius - float(point[1])) / (2.0 * radius) * (height - 76)
        return x, y

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect width="{width}" height="{height}" rx="10" fill="#111827"/>',
        f'<text x="12" y="19" fill="#f8fafc" font-size="12" font-weight="650">{html.escape(title)}</text>',
        f'<text x="12" y="35" fill="#aeb9c9" font-size="9">{html.escape(subtitle)}</text>',
    ]
    origin = xy(np.zeros((2,), dtype=np.float32))
    parts.extend(
        [
            f'<line x1="{origin[0]:.1f}" y1="42" x2="{origin[0]:.1f}" y2="{height-34}" stroke="#334155" stroke-width="0.8"/>',
            f'<line x1="{pad}" y1="{origin[1]:.1f}" x2="{width-pad}" y2="{origin[1]:.1f}" stroke="#334155" stroke-width="0.8"/>',
        ]
    )
    for agent in range(2):
        for phase_start, phase_stop, dash, opacity in (
            (0, event_index + 1, "", 0.95),
            (event_index, len(mask), ' stroke-dasharray="5 4"', 0.62),
        ):
            for segment_start, segment_stop in _segments(
                mask[:, agent], phase_start, phase_stop
            ):
                points = " ".join(
                    f"{px:.1f},{py:.1f}"
                    for px, py in (
                        xy(point)
                        for point in positions[segment_start:segment_stop, agent]
                    )
                )
                parts.append(
                    f'<polyline points="{points}" fill="none" stroke="{AGENT_COLORS[agent]}" '
                    f'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash}/>'
                )
        if 0 <= event_index < len(mask) and bool(mask[event_index, agent]):
            event_x, event_y = xy(positions[event_index, agent])
            parts.append(
                f'<circle cx="{event_x:.1f}" cy="{event_y:.1f}" r="5" fill="{AGENT_COLORS[agent]}" stroke="white" stroke-width="1.3"/>'
            )
    parts.extend(
        [
            f'<text x="12" y="{height-12}" fill="#94a3b8" font-size="9">solid: history · dashed: future · radius ±{radius:.0f}m</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def physical_trajectory(
    features: dict[str, np.ndarray], row: int
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.asarray(features["normalized_sequence"][int(row)], dtype=np.float32)
    median = np.asarray(features["normalization_median"], dtype=np.float32)
    iqr = np.asarray(features["normalization_iqr"], dtype=np.float32)
    physical = normalized * iqr[None, :, :] + median[None, :, :]
    return physical[..., 0:2], np.asarray(features["aligned_mask"][int(row)], dtype=bool)


def stratum_label(key: int) -> str:
    first = int(key) // 100
    second = (int(key) // 10) % 10
    fallback = bool(int(key) % 10)
    suffix = "fallback" if fallback else "contact"
    return f"{TYPE_NAMES.get(first, first)} → {TYPE_NAMES.get(second, second)} · {suffix}"


def _sample_meta(cache: dict[str, np.ndarray], row: int) -> str:
    return (
        f"idx {row} · {str(cache['scenario_id'][row])} · "
        f"agents {int(cache['first_agent_id'][row])}/{int(cache['second_agent_id'][row])}"
    )


def _candidate_card(
    *,
    role: str,
    row: int,
    gt_label: str,
    anchor: int,
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    radius: float,
    positive_distance_by_row: dict[int, float],
    negative_distance_by_row: dict[int, float],
    similarities: dict[int, float] | None,
) -> str:
    rms = (
        positive_distance_by_row[int(row)]
        if gt_label == "P"
        else negative_distance_by_row[int(row)]
    )
    similarity = None if similarities is None else similarities[int(row)]
    reason = ""
    badge_class = "positive" if gt_label == "P" else "negative"
    if gt_label == "N":
        negative_pool = _valid_pool(cache, "negative_indices", anchor)
        offset = int(np.flatnonzero(negative_pool == int(row))[0])
        reason = decode_reason_bits(int(cache["negative_reason_bits"][anchor, offset]))
    positions, mask = physical_trajectory(features, row)
    metric = f"GT {gt_label} · RMS {rms:.3f}"
    if similarity is not None:
        metric += f" · cosine {similarity:.3f}"
    if reason:
        metric += f" · {reason}"
    svg = trajectory_svg(
        positions,
        mask,
        radius=radius,
        title=role,
        subtitle=_sample_meta(cache, row),
    )
    return (
        f'<article class="candidate {badge_class}">{svg}'
        f'<p><span class="badge">{gt_label}</span>{html.escape(metric)}</p></article>'
    )


def _anchor_card(
    *,
    anchor: int,
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    radius: float,
) -> str:
    positions, mask = physical_trajectory(features, anchor)
    svg = trajectory_svg(
        positions,
        mask,
        radius=radius,
        title="ANCHOR",
        subtitle=_sample_meta(cache, anchor),
    )
    return f'<article class="candidate anchor">{svg}<p>fixed query for every row</p></article>'


def _page_css() -> str:
    return """
body{font:14px system-ui;margin:22px;background:#eef2f7;color:#172033}
a{color:#1659b7}.note,.metrics,.stage{background:white;border:1px solid #d7dee8;border-radius:10px;padding:14px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:10px;overflow-x:auto}
.candidate{min-width:220px;border:2px solid #d9e0e9;border-radius:10px;padding:7px;background:#f8fafc}
.candidate svg{width:100%;height:auto;display:block}.candidate p{font-size:11px;line-height:1.35;margin:7px 3px 2px}
.candidate.positive{border-color:#65a879}.candidate.negative{border-color:#d87979}.candidate.anchor{border-color:#7085bd}
.badge{font-weight:750;border-radius:10px;padding:2px 6px;margin-right:5px;background:#e2e8f0}
table{border-collapse:collapse}th,td{border:1px solid #d6dde7;padding:6px 9px;text-align:right}th:first-child{text-align:left}
.legend{color:#526174}.small{font-size:12px;color:#526174}
"""


def _page_radius(rows: list[int], features: dict[str, np.ndarray]) -> float:
    absolute_values = []
    for row in sorted(set(rows)):
        positions, mask = physical_trajectory(features, row)
        if bool(mask.any()):
            absolute_values.extend(np.abs(positions[mask]).reshape(-1).tolist())
    quantile = float(np.quantile(absolute_values, 0.98)) if absolute_values else 20.0
    return float(np.clip(np.ceil(quantile / 5.0) * 5.0, 20.0, 80.0))


def build_anchor_report(
    *,
    anchor: int,
    cache: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    stage_similarities: dict[str, dict[int, float]],
    output_path: Path,
) -> dict[str, Any]:
    positive_rows = _valid_pool(cache, "positive_indices", anchor)
    negative_rows = _valid_pool(cache, "negative_indices", anchor)
    positive_distances = _candidate_distances(
        cache, "positive_rms_distances", anchor, len(positive_rows)
    )
    negative_distances = _candidate_distances(
        cache, "negative_rms_distances", anchor, len(negative_rows)
    )
    positive_distance_by_row = {
        int(row): float(distance)
        for row, distance in zip(positive_rows.tolist(), positive_distances.tolist())
    }
    negative_distance_by_row = {
        int(row): float(distance)
        for row, distance in zip(negative_rows.tolist(), negative_distances.tolist())
    }
    rms_examples = choose_rms_examples(
        positive_rows, positive_distances, negative_rows, negative_distances
    )
    examples_by_stage = {
        stage: choose_stage_examples(positive_rows, negative_rows, similarities)
        for stage, similarities in stage_similarities.items()
    }
    displayed_rows = [anchor]
    displayed_rows.extend(row for _, row in rms_examples)
    for examples in examples_by_stage.values():
        displayed_rows.extend(row for _, row in examples)
    radius = _page_radius(displayed_rows, features)

    metric_rows = []
    metric_payload: dict[str, dict[str, float]] = {}
    nearest_positive = int(positive_rows[int(np.argmin(positive_distances))])
    for stage, similarities in stage_similarities.items():
        positive_sims = np.asarray(
            [similarities[int(row)] for row in positive_rows], dtype=np.float32
        )
        negative_sims = np.asarray(
            [similarities[int(row)] for row in negative_rows], dtype=np.float32
        )
        ranking = pairwise_ranking_accuracy(positive_sims, positive_distances)
        margin = float(similarities[nearest_positive] - negative_sims.max())
        payload = {
            "nearest_rms_positive_cosine": float(similarities[nearest_positive]),
            "hardest_negative_cosine": float(negative_sims.max()),
            "separation_margin": margin,
            "positive_pairwise_ranking_accuracy": ranking,
        }
        metric_payload[stage] = payload
        metric_rows.append(
            f"<tr><th>{html.escape(STAGE_LABELS[stage])}</th>"
            f"<td>{payload['nearest_rms_positive_cosine']:.3f}</td>"
            f"<td>{payload['hardest_negative_cosine']:.3f}</td>"
            f"<td>{payload['separation_margin']:+.3f}</td>"
            f"<td>{payload['positive_pairwise_ranking_accuracy']:.1%}</td></tr>"
        )

    sections = []
    rms_cards = [_anchor_card(anchor=anchor, cache=cache, features=features, radius=radius)]
    for role, row in rms_examples:
        gt_label = "P" if row in set(positive_rows.tolist()) else "N"
        rms_cards.append(
            _candidate_card(
                role=role,
                row=row,
                gt_label=gt_label,
                anchor=anchor,
                cache=cache,
                features=features,
                radius=radius,
                positive_distance_by_row=positive_distance_by_row,
                negative_distance_by_row=negative_distance_by_row,
                similarities=None,
            )
        )
    sections.append(
        '<section class="stage"><h2>RMS / relation ground-truth reference</h2>'
        '<p class="small">Positive order is exact event-aligned masked RMS. Negatives are mined because their future pair-relation outcome differs.</p>'
        f'<div class="grid">{"".join(rms_cards)}</div></section>'
    )
    for stage in ("raw_z", "hard", "hybrid"):
        cards = [_anchor_card(anchor=anchor, cache=cache, features=features, radius=radius)]
        for role, row in examples_by_stage[stage]:
            gt_label = "P" if row in set(positive_rows.tolist()) else "N"
            cards.append(
                _candidate_card(
                    role=role,
                    row=row,
                    gt_label=gt_label,
                    anchor=anchor,
                    cache=cache,
                    features=features,
                    radius=radius,
                    positive_distance_by_row=positive_distance_by_row,
                    negative_distance_by_row=negative_distance_by_row,
                    similarities=stage_similarities[stage],
                )
            )
        sections.append(
            f'<section class="stage"><h2>{html.escape(STAGE_LABELS[stage])}</h2>'
            '<p class="small">Extremes are selected inside the same fixed GT pools; “hardest negative” means the relation negative with highest cosine.</p>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    key = int(cache["stratum_key"][anchor])
    output_path.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<title>Anchor {anchor} retrieval stages</title><style>{_page_css()}</style></head><body>'
        '<p><a href="index.html">← all anchors</a></p>'
        f'<h1>Anchor {anchor}: raw z → hard → hybrid</h1>'
        f'<p class="legend">{html.escape(stratum_label(key))} · scenario '
        f'{html.escape(str(cache["scenario_id"][anchor]))} · query step {int(cache["query_step"][anchor])}</p>'
        '<div class="note"><strong>Reading guide.</strong> Green cards are GT RMS positives; red cards are GT relation negatives. '
        'The pools never change across rows. Solid trajectory is the 1.9 s pre-event history available in the RMS representation; dashed trajectory is the 4.0 s future used only for physical/relation auditing.</div>'
        '<div class="metrics"><table><thead><tr><th>representation</th><th>nearest RMS P cosine</th>'
        '<th>hardest N cosine</th><th>margin</th><th>positive ranking</th></tr></thead>'
        f'<tbody>{"".join(metric_rows)}</tbody></table></div>{"".join(sections)}</body></html>'
    )
    return {
        "anchor_row": int(anchor),
        "sample_index": int(cache["sample_index"][anchor]),
        "scenario_id": str(cache["scenario_id"][anchor]),
        "stratum_key": key,
        "stratum": stratum_label(key),
        "query_step": int(cache["query_step"][anchor]),
        "positive_pool_size": int(len(positive_rows)),
        "negative_pool_size": int(len(negative_rows)),
        "page": output_path.name,
        "radius_m": radius,
        "metrics": metric_payload,
        "rms_examples": [
            {"role": role, "row": int(row)} for role, row in rms_examples
        ],
        "stage_examples": {
            stage: [{"role": role, "row": int(row)} for role, row in examples]
            for stage, examples in examples_by_stage.items()
        },
    }


def _write_summary_csv(path: Path, reports: list[dict[str, Any]]) -> None:
    fields = [
        "anchor_row",
        "scenario_id",
        "stratum_key",
        "stratum",
        "stage",
        "nearest_rms_positive_cosine",
        "hardest_negative_cosine",
        "separation_margin",
        "positive_pairwise_ranking_accuracy",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for stage, metrics in report["metrics"].items():
                writer.writerow(
                    {
                        "anchor_row": report["anchor_row"],
                        "scenario_id": report["scenario_id"],
                        "stratum_key": report["stratum_key"],
                        "stratum": report["stratum"],
                        "stage": stage,
                        **metrics,
                    }
                )


def _write_index(
    path: Path,
    reports: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
) -> None:
    cards = []
    for position, report in enumerate(reports, start=1):
        anchor = int(report["anchor_row"])
        points, mask = physical_trajectory(features, anchor)
        svg = trajectory_svg(
            points,
            mask,
            radius=float(report["radius_m"]),
            title=f"{position:02d} · anchor {anchor}",
            subtitle=str(report["scenario_id"]),
        )
        raw_margin = float(report["metrics"]["raw_z"]["separation_margin"])
        hard_margin = float(report["metrics"]["hard"]["separation_margin"])
        hybrid_margin = float(report["metrics"]["hybrid"]["separation_margin"])
        cards.append(
            f'<article><a href="{html.escape(str(report["page"]))}">{svg}</a>'
            f'<h3><a href="{html.escape(str(report["page"]))}">{html.escape(str(report["stratum"]))}</a></h3>'
            f'<p>margin raw {raw_margin:+.3f} → hard {hard_margin:+.3f} → hybrid {hybrid_margin:+.3f}</p></article>'
        )
    path.write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>Contrastive retrieval stages</title>'
        f'<style>{_page_css()} .anchors{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:14px}}'
        '.anchors article{background:white;border:1px solid #d7dee8;border-radius:10px;padding:10px}.anchors svg{width:100%}</style>'
        '</head><body><h1>Interaction retrieval: raw z → hard → hybrid</h1>'
        '<div class="note"><p><strong>Ten deterministic, stratum-aware anchors from the fixed 512-anchor validation manifest.</strong></p>'
        '<p>Open an anchor to compare the same GT positive/negative pools under RMS, raw tokenizer z, hard contrastive embedding, and hybrid-soft embedding.</p>'
        '<p>Raw z is the flattened query-time latent with L2 normalization. It is scene-level; the learned heads are pair-specific.</p></div>'
        f'<div class="anchors">{"".join(cards)}</div></body></html>'
    )


def build(args: argparse.Namespace) -> None:
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache, allow_pickle=False) as loaded:
        cache = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.rms_features, allow_pickle=False) as loaded:
        features = {key: np.asarray(loaded[key]) for key in loaded.files}
    with np.load(args.validation_manifest, allow_pickle=False) as loaded:
        manifest_rows = np.asarray(loaded["cache_row"], dtype=np.int64)

    anchor_rows = select_anchor_rows(
        manifest_rows, cache["stratum_key"], args.num_anchors
    )
    sample_rows = set(anchor_rows.tolist())
    for anchor in anchor_rows.tolist():
        sample_rows.update(_valid_pool(cache, "positive_indices", anchor).tolist())
        sample_rows.update(_valid_pool(cache, "negative_indices", anchor).tolist())
    sample_rows_array = np.asarray(sorted(sample_rows), dtype=np.int64)
    print(
        f"selected anchors={anchor_rows.tolist()} unique scenes={len(sample_rows_array)} "
        f"device={device}",
        flush=True,
    )

    stage_paths = {
        "raw_z": args.base_checkpoint,
        "hard": args.hard_checkpoint,
        "hybrid": args.hybrid_checkpoint,
    }
    stage_embeddings: dict[str, dict[int, np.ndarray]] = {}
    stage_metadata: dict[str, dict[str, Any]] = {}
    for stage, checkpoint_path in stage_paths.items():
        embeddings, metadata = encode_stage(
            stage=stage,
            checkpoint_path=checkpoint_path,
            sample_rows=sample_rows_array,
            cache=cache,
            history_steps=args.history_steps,
            device=device,
            batch_size=args.batch_size,
        )
        stage_embeddings[stage] = embeddings
        stage_metadata[stage] = metadata

    reports = []
    for position, anchor in enumerate(anchor_rows.tolist(), start=1):
        candidates = np.concatenate(
            (
                _valid_pool(cache, "positive_indices", anchor),
                _valid_pool(cache, "negative_indices", anchor),
            )
        )
        stage_similarities = {
            stage: cosine_lookup(anchor, candidates, embeddings)
            for stage, embeddings in stage_embeddings.items()
        }
        page = args.output_dir / f"anchor_{position:02d}_idx{anchor:05d}.html"
        report = build_anchor_report(
            anchor=anchor,
            cache=cache,
            features=features,
            stage_similarities=stage_similarities,
            output_path=page,
        )
        reports.append(report)
        print(f"rendered {page.name}", flush=True)

    _write_summary_csv(args.output_dir / "summary.csv", reports)
    manifest = {
        "definition": (
            "Fixed RMS/relation GT pools; stages rank within those pools. "
            "Raw z is flattened and L2-normalized; learned embeddings are pair-specific."
        ),
        "device": str(device),
        "history_steps": int(args.history_steps),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "contrastive_cache": str(args.cache.resolve()),
        "rms_features": str(args.rms_features.resolve()),
        "stages": stage_metadata,
        "anchors": reports,
    }
    (args.output_dir / "gallery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    _write_index(args.output_dir / "index.html", reports, features, cache)
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
        default=waymo_root / "eval_results/interaction_contrastive_retrieval_stages_10",
    )
    parser.add_argument("--num_anchors", type=int, default=10)
    parser.add_argument("--history_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    build(build_argparser().parse_args())


if __name__ == "__main__":
    main()
