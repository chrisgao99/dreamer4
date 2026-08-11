"""Create a dependency-light HTML audit for event-aligned RMS neighbours."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np


COLORS = ("#2389da", "#f28e2b")
TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}


def _segments(points: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    start = None
    for index, usable in enumerate(valid):
        if usable and start is None:
            start = index
        if start is not None and (not usable or index == len(valid) - 1):
            end = index + 1 if usable else index
            if end - start >= 2:
                result.append(points[start:end])
            start = None
    return result


def _bounds(first: np.ndarray, first_mask: np.ndarray, second: np.ndarray, second_mask: np.ndarray):
    points = np.concatenate((first[first_mask], second[second_mask]), axis=0)
    if not len(points):
        return -10.0, 10.0, -10.0, 10.0
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    span = max(float(np.max(high - low)), 10.0)
    half = 0.56 * span
    return center[0] - half, center[0] + half, center[1] - half, center[1] + half


def _panel_svg(
    position: np.ndarray,
    mask: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    x_offset: int,
    title: str,
    subtitle: str,
    event_index: int,
) -> str:
    size = 350
    pad = 30
    xmin, xmax, ymin, ymax = bounds

    def xy(point: np.ndarray) -> tuple[float, float]:
        x = x_offset + pad + (float(point[0]) - xmin) / max(xmax - xmin, 1e-6) * (size - 2 * pad)
        y = pad + (ymax - float(point[1])) / max(ymax - ymin, 1e-6) * (size - 2 * pad)
        return x, y

    parts = [
        f'<rect x="{x_offset}" y="0" width="{size}" height="{size}" rx="8" fill="#181c22"/>',
        f'<text x="{x_offset + 12}" y="20" fill="#f5f5f5" font-size="13">{html.escape(title)}</text>',
        f'<text x="{x_offset + 12}" y="338" fill="#aeb7c4" font-size="10">{html.escape(subtitle)}</text>',
    ]
    for agent in range(2):
        for segment in _segments(position[:, agent], mask[:, agent]):
            coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, segment))
            parts.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{COLORS[agent]}" '
                'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        valid_indices = np.flatnonzero(mask[:, agent])
        if not len(valid_indices):
            continue
        start = xy(position[int(valid_indices[0]), agent])
        end = xy(position[int(valid_indices[-1]), agent])
        parts.append(
            f'<circle cx="{start[0]:.1f}" cy="{start[1]:.1f}" r="5" fill="none" '
            f'stroke="{COLORS[agent]}" stroke-width="2"><title>agent {agent + 1} start</title></circle>'
        )
        parts.append(
            f'<rect x="{end[0] - 4:.1f}" y="{end[1] - 4:.1f}" width="8" height="8" '
            f'fill="{COLORS[agent]}"><title>agent {agent + 1} end</title></rect>'
        )
        if 0 <= event_index < len(mask) and mask[event_index, agent]:
            event = xy(position[event_index, agent])
            parts.append(
                f'<path d="M {event[0]:.1f} {event[1] - 7:.1f} L {event[0] + 2:.1f} {event[1] - 2:.1f} '
                f'L {event[0] + 7:.1f} {event[1]:.1f} L {event[0] + 2:.1f} {event[1] + 2:.1f} '
                f'L {event[0]:.1f} {event[1] + 7:.1f} L {event[0] - 2:.1f} {event[1] + 2:.1f} '
                f'L {event[0] - 7:.1f} {event[1]:.1f} L {event[0] - 2:.1f} {event[1] - 2:.1f} Z" '
                f'fill="white" stroke="{COLORS[agent]}" stroke-width="1.5"><title>aligned event</title></path>'
            )
        for step in range(10, len(mask), 10):
            if mask[step, agent]:
                tick = xy(position[step, agent])
                parts.append(
                    f'<circle cx="{tick[0]:.1f}" cy="{tick[1]:.1f}" r="2" fill="#f7f7f7" '
                    f'opacity="0.75"><title>t={step}</title></circle>'
                )
    return "".join(parts)


def _sample_anchors(distances: np.ndarray, count: int) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(distances))
    if not len(finite):
        return np.empty((0,), dtype=np.int64)
    order = finite[np.argsort(distances[finite], kind="stable")]
    if len(order) <= count:
        return order
    positions = np.linspace(0.02, 0.98, count)
    return order[np.round(positions * (len(order) - 1)).astype(np.int64)]


def build_audit(rms_dir: Path, output_dir: Path, split: str, count: int, rank: int) -> None:
    with np.load(rms_dir / f"{split}_rms_features.npz", allow_pickle=False) as features:
        arrays = {key: np.asarray(features[key]) for key in features.files}
    with np.load(rms_dir / f"{split}_rms_neighbors.npz", allow_pickle=False) as neighbours:
        indices = np.asarray(neighbours["neighbor_indices"])
        distances = np.asarray(neighbours["rms_distances"])
        overlap = np.asarray(neighbours["common_valid_fraction"])

    rank_index = rank - 1
    if rank_index < 0 or rank_index >= indices.shape[1]:
        raise ValueError(f"rank must be in [1, {indices.shape[1]}]")
    usable_distance = distances[:, rank_index].copy()
    usable_distance[indices[:, rank_index] < 0] = np.inf
    anchors = _sample_anchors(usable_distance, count)

    median = arrays["normalization_median"].astype(np.float32)
    iqr = arrays["normalization_iqr"].astype(np.float32)
    normalized = arrays["normalized_sequence"].astype(np.float32)
    position = normalized[..., 0:2] * iqr[None, None, :, 0:2] + median[None, None, :, 0:2]
    masks = arrays["aligned_mask"].astype(bool)
    offsets = arrays["time_offsets"]
    event_index = int(np.argmin(np.abs(offsets)))

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fields = (
        "audit_order", "anchor_index", "neighbor_index", "rank", "rms_distance",
        "common_valid_fraction", "anchor_scenario_id", "neighbor_scenario_id",
        "anchor_event_mode", "neighbor_event_mode", "anchor_zone_pet_s", "neighbor_zone_pet_s",
    )
    cards = []
    rows = []
    for order, anchor in enumerate(anchors):
        other = int(indices[anchor, rank_index])
        first_position = position[anchor]
        second_position = position[other]
        first_mask = masks[anchor]
        second_mask = masks[other]
        bounds = _bounds(first_position, first_mask, second_position, second_mask)

        def label(index: int) -> str:
            types = "/".join(
                TYPE_NAMES.get(int(value), str(int(value)))
                for value in (arrays["first_agent_type"][index], arrays["second_agent_type"][index])
            )
            return f"{arrays['event_mode'][index]} | {types} | PET={float(arrays['zone_pet_s'][index]):.1f}s"

        svg = (
            '<svg viewBox="0 0 712 350" role="img" aria-label="RMS trajectory match">'
            + _panel_svg(
                first_position, first_mask, bounds=bounds, x_offset=0, title=f"anchor #{int(anchor)}",
                subtitle=label(int(anchor)), event_index=event_index,
            )
            + _panel_svg(
                second_position, second_mask, bounds=bounds, x_offset=362, title=f"top-{rank} #{other}",
                subtitle=label(other), event_index=event_index,
            )
            + "</svg>"
        )
        distance = float(distances[anchor, rank_index])
        cards.append(
            '<article><div class="metric">'
            f'RMS {distance:.4f} · common-valid {float(overlap[anchor, rank_index]):.1%}'
            f'</div>{svg}<div class="scene">anchor scene {html.escape(str(arrays["scenario_id"][anchor]))}<br>'
            f'neighbor scene {html.escape(str(arrays["scenario_id"][other]))}</div></article>'
        )
        rows.append(
            {
                "audit_order": order,
                "anchor_index": int(anchor),
                "neighbor_index": other,
                "rank": rank,
                "rms_distance": distance,
                "common_valid_fraction": float(overlap[anchor, rank_index]),
                "anchor_scenario_id": arrays["scenario_id"][anchor],
                "neighbor_scenario_id": arrays["scenario_id"][other],
                "anchor_event_mode": arrays["event_mode"][anchor],
                "neighbor_event_mode": arrays["event_mode"][other],
                "anchor_zone_pet_s": float(arrays["zone_pet_s"][anchor]),
                "neighbor_zone_pet_s": float(arrays["zone_pet_s"][other]),
            }
        )

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "index.html").write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>RMS match audit</title>"
        "<style>body{background:#0f1217;color:#eee;font-family:system-ui;margin:24px}"
        "h1{font-size:24px}p{color:#aeb7c4}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(720px,1fr));gap:18px}"
        "article{background:#222832;border:1px solid #394250;border-radius:12px;padding:12px}svg{width:100%;height:auto}"
        ".metric{font-size:16px;font-weight:650;margin-bottom:8px}.scene{font:11px ui-monospace;color:#aeb7c4;margin-top:7px}"
        "</style></head><body><h1>Event-aligned exact RMS matches</h1>"
        f"<p>{html.escape(split)} split · top-{rank} · sampled across the RMS distribution. "
        "Circle=start, square=end, star=aligned event, white dots=1 s ticks. Both panels share the same scale.</p>"
        f"<div class='grid'>{''.join(cards)}</div></body></html>",
        encoding="utf-8",
    )
    print(f"saved {output_dir / 'index.html'} ({len(rows)} matches)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rms_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=24)
    parser.add_argument("--rank", type=int, default=1)
    args = parser.parse_args()
    build_audit(Path(args.rms_dir), Path(args.output_dir), args.split, args.num_samples, args.rank)


if __name__ == "__main__":
    main()
