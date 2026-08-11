"""Build a visual audit gallery for matched interaction samples.

Each selected anchor is shown together with every stored positive, hard
negative, and easy negative.  Trajectories from different scenes are aligned
to the focus pose at their own query timestep so that the offline matching
decision can be inspected directly.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from .pair_samples import ensure_agent_time_layout
except ImportError:
    from pair_samples import ensure_agent_time_layout  # type: ignore


ROLE_COLORS = {
    "anchor": "#222222",
    "positive": "#2ca02c",
    "hard_negative": "#d62728",
    "easy_negative": "#9467bd",
}
FOCUS_COLOR = "#1f77b4"
CANDIDATE_COLOR = "#ff7f0e"
AGENT_TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}

SAMPLE_FIELDS = (
    "scenario_id",
    "source_path",
    "focus_agent_id",
    "candidate_agent_id",
    "candidate_index",
    "event_step",
    "query_step",
    "lead_steps",
    "relation_index",
    "response_index",
    "eligible",
    "focus_type",
    "candidate_type",
    "delta_arrival_time_s",
    "pet_s",
    "spatial_min_dist_m",
    "relation_names",
    "response_names",
)

MATCH_FIELDS = (
    "positive_indices",
    "positive_distances",
    "hard_negative_indices",
    "hard_negative_distances",
    "negative_indices",
    "negative_distances",
    "trainable_anchor_mask",
)


def _load_selected_arrays(path: Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in fields if field not in data]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        return {field: data[field] for field in fields}


@lru_cache(maxsize=32)
def _load_scene(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("agents", "agent_mask", "agent_ids", "map_polylines", "map_mask")
        missing = [field for field in required if field not in data]
        if missing:
            raise KeyError(f"{path} is missing required scene arrays: {missing}")
        agent_mask = np.asarray(data["agent_mask"], dtype=bool)
        return {
            "agents": ensure_agent_time_layout(np.asarray(data["agents"], dtype=np.float32), agent_mask),
            "agent_mask": agent_mask,
            "agent_ids": np.asarray(data["agent_ids"], dtype=np.int64),
            "map_polylines": np.asarray(data["map_polylines"], dtype=np.float32),
            "map_mask": np.asarray(data["map_mask"], dtype=bool),
        }


def _parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_name_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def select_stratified_anchors(
    candidates: np.ndarray,
    *,
    relation_index: np.ndarray,
    response_index: np.ndarray,
    lead_steps: np.ndarray,
    count: int,
    seed: int,
) -> list[int]:
    """Round-robin candidates across (relation, response, lead) strata."""
    if count <= 0 or len(candidates) == 0:
        return []
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in candidates:
        idx = int(idx)
        groups[(int(relation_index[idx]), int(response_index[idx]), int(lead_steps[idx]))].append(idx)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    keys = [keys[int(i)] for i in rng.permutation(len(keys))]

    selected: list[int] = []
    offset = 0
    while len(selected) < count:
        added = False
        for key in keys:
            values = groups[key]
            if offset < len(values):
                selected.append(int(values[offset]))
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        offset += 1
    return selected


def select_anchor_indices(
    samples: dict[str, np.ndarray],
    matches: dict[str, np.ndarray],
    *,
    count: int,
    seed: int,
    selection: str,
    relation_filter: set[str],
    response_filter: set[str],
    lead_filter: set[int],
) -> list[int]:
    relation_names = samples["relation_names"].tolist()
    response_names = samples["response_names"].tolist()
    unknown_relations = relation_filter - set(relation_names)
    unknown_responses = response_filter - set(response_names)
    if unknown_relations:
        raise ValueError(f"Unknown relation names: {sorted(unknown_relations)}")
    if unknown_responses:
        raise ValueError(f"Unknown response names: {sorted(unknown_responses)}")

    mask = np.asarray(matches["trainable_anchor_mask"], dtype=bool).copy()
    if relation_filter:
        allowed = {relation_names.index(name) for name in relation_filter}
        mask &= np.isin(samples["relation_index"], list(allowed))
    if response_filter:
        allowed = {response_names.index(name) for name in response_filter}
        mask &= np.isin(samples["response_index"], list(allowed))
    if lead_filter:
        mask &= np.isin(samples["lead_steps"], list(lead_filter))
    candidates = np.flatnonzero(mask)
    if len(candidates) == 0:
        raise ValueError("No trainable anchors satisfy the requested filters")

    count = min(int(count), len(candidates))
    if selection == "random":
        rng = np.random.default_rng(seed)
        return [int(idx) for idx in rng.choice(candidates, size=count, replace=False)]
    return select_stratified_anchors(
        candidates,
        relation_index=samples["relation_index"],
        response_index=samples["response_index"],
        lead_steps=samples["lead_steps"],
        count=count,
        seed=seed,
    )


def related_samples(anchor: int, matches: dict[str, np.ndarray]) -> list[tuple[str, int, int, float | None]]:
    """Return (role, rank, sample index, matching distance) for one anchor."""
    result: list[tuple[str, int, int, float | None]] = [("anchor", 0, int(anchor), None)]
    specs = (
        ("positive", "positive_indices", "positive_distances"),
        ("hard_negative", "hard_negative_indices", "hard_negative_distances"),
        ("easy_negative", "negative_indices", "negative_distances"),
    )
    for role, index_key, distance_key in specs:
        rank = 0
        for sample_index, distance in zip(matches[index_key][anchor], matches[distance_key][anchor]):
            sample_index = int(sample_index)
            if sample_index < 0:
                continue
            rank += 1
            result.append((role, rank, sample_index, float(distance)))
    return result


def _query_frame(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(xy, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    return delta @ rotation.T


def _masked_xy(agent: np.ndarray, origin: np.ndarray, yaw: float, start: int, end: int) -> np.ndarray:
    points = _query_frame(agent[start : end + 1, 0:2], origin, yaw)
    valid = agent[start : end + 1, 5] > 0.5
    points[~valid] = np.nan
    return points


def _arrival_steps(event_step: int, delta_arrival_time_s: float, dt: float) -> tuple[int, int]:
    delta_steps = int(round(float(delta_arrival_time_s) / float(dt)))
    if delta_steps >= 0:
        return int(event_step + delta_steps), int(event_step)
    return int(event_step), int(event_step - delta_steps)


def _safe_point(agent: np.ndarray, step: int, origin: np.ndarray, yaw: float) -> np.ndarray | None:
    if step < 0 or step >= len(agent) or agent[step, 5] <= 0.5:
        return None
    return _query_frame(agent[step, 0:2][None], origin, yaw)[0]


def _draw_map(
    ax: plt.Axes,
    scene: dict[str, np.ndarray],
    *,
    origin: np.ndarray,
    yaw: float,
    radius_m: float,
) -> None:
    for polyline, mask in zip(scene["map_polylines"], scene["map_mask"]):
        points = polyline[mask, 0:2]
        if len(points) < 2:
            continue
        points = _query_frame(points, origin, yaw)
        if not bool((np.abs(points) <= radius_m * 1.25).all(axis=1).any()):
            continue
        ax.plot(points[:, 0], points[:, 1], color="#c7c7c7", linewidth=0.45, alpha=0.65, zorder=1)


def _draw_context_agents(
    ax: plt.Axes,
    scene: dict[str, np.ndarray],
    *,
    query_step: int,
    candidate_index: int,
    origin: np.ndarray,
    yaw: float,
) -> None:
    agents = scene["agents"]
    visible = []
    for idx in np.flatnonzero(scene["agent_mask"]):
        idx = int(idx)
        if idx in (0, candidate_index) or agents[idx, query_step, 5] <= 0.5:
            continue
        visible.append(agents[idx, query_step, 0:2])
    if visible:
        xy = _query_frame(np.asarray(visible), origin, yaw)
        ax.scatter(xy[:, 0], xy[:, 1], s=9, color="#9e9e9e", alpha=0.55, zorder=2)


def _draw_spatial_panel(
    ax: plt.Axes,
    scene: dict[str, np.ndarray],
    samples: dict[str, np.ndarray],
    sample_index: int,
    *,
    role: str,
    rank: int,
    distance: float | None,
    radius_m: float,
    post_event_steps: int,
    history_steps: int,
    dt: float,
) -> tuple[int, int, int, int, np.ndarray, np.ndarray]:
    agents = scene["agents"]
    candidate_index = int(samples["candidate_index"][sample_index])
    if candidate_index <= 0 or candidate_index >= len(agents):
        raise IndexError(f"sample {sample_index} has invalid candidate_index={candidate_index}")
    focus = agents[0]
    candidate = agents[candidate_index]
    query_step = int(samples["query_step"][sample_index])
    event_step = int(samples["event_step"][sample_index])
    start = max(0, query_step - history_steps + 1)
    end = min(len(focus) - 1, max(event_step + post_event_steps, query_step + 1))
    origin = focus[query_step, 0:2]
    yaw = float(focus[query_step, 6])

    _draw_map(ax, scene, origin=origin, yaw=yaw, radius_m=radius_m)
    _draw_context_agents(
        ax,
        scene,
        query_step=query_step,
        candidate_index=candidate_index,
        origin=origin,
        yaw=yaw,
    )

    for agent, color, label in (
        (focus, FOCUS_COLOR, "focus"),
        (candidate, CANDIDATE_COLOR, "candidate"),
    ):
        history_xy = _masked_xy(agent, origin, yaw, start, query_step)
        future_xy = _masked_xy(agent, origin, yaw, query_step, end)
        ax.plot(history_xy[:, 0], history_xy[:, 1], color=color, linewidth=2.2, label=label, zorder=4)
        ax.plot(future_xy[:, 0], future_xy[:, 1], color=color, linewidth=1.45, linestyle="--", alpha=0.8, zorder=3)
        query_xy = _safe_point(agent, query_step, origin, yaw)
        if query_xy is not None:
            heading = float(agent[query_step, 6] - yaw)
            ax.scatter(query_xy[0], query_xy[1], s=42, color=color, edgecolors="white", linewidths=0.8, zorder=6)
            ax.arrow(
                query_xy[0],
                query_xy[1],
                3.0 * np.cos(heading),
                3.0 * np.sin(heading),
                color=color,
                width=0.08,
                head_width=0.8,
                head_length=0.9,
                length_includes_head=True,
                zorder=6,
            )

    focus_arrival, candidate_arrival = _arrival_steps(
        event_step, float(samples["delta_arrival_time_s"][sample_index]), dt
    )
    for agent, step, color in (
        (focus, focus_arrival, FOCUS_COLOR),
        (candidate, candidate_arrival, CANDIDATE_COLOR),
    ):
        point = _safe_point(agent, step, origin, yaw)
        if point is not None:
            ax.scatter(point[0], point[1], marker="*", s=95, color=color, edgecolors="black", linewidths=0.6, zorder=7)

    relation_names = samples["relation_names"].tolist()
    response_names = samples["response_names"].tolist()
    relation = relation_names[int(samples["relation_index"][sample_index])]
    response_index = int(samples["response_index"][sample_index])
    response = response_names[response_index] if response_index >= 0 else "ambiguous"
    role_label = role.upper().replace("_", " ")
    if rank:
        role_label += f" {rank}"
    distance_text = "" if distance is None else f"  d={distance:.2f}"
    title = (
        f"{role_label}  idx={sample_index}{distance_text}\n"
        f"{relation} | {response} | lead={int(samples['lead_steps'][sample_index]) * dt:.1f}s "
        f"| PET={float(samples['pet_s'][sample_index]):.1f}s"
    )
    ax.set_title(title, fontsize=8.5, color=ROLE_COLORS[role], fontweight="bold", loc="left")
    ax.set_xlim(-radius_m, radius_m)
    ax.set_ylim(-radius_m, radius_m)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18, linewidth=0.45)
    ax.axhline(0.0, color="#888888", linewidth=0.4, alpha=0.45)
    ax.axvline(0.0, color="#888888", linewidth=0.4, alpha=0.45)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("query-frame longitudinal (m)", fontsize=6.5)
    ax.set_ylabel("lateral (m)", fontsize=6.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(ROLE_COLORS[role])
        spine.set_linewidth(1.5 if role == "anchor" else 1.0)
    return start, end, query_step, event_step, focus, candidate


def _draw_speed_panel(
    ax: plt.Axes,
    focus: np.ndarray,
    candidate: np.ndarray,
    *,
    start: int,
    end: int,
    query_step: int,
    event_step: int,
    dt: float,
) -> None:
    steps = np.arange(start, end + 1)
    time_s = (steps - event_step) * dt
    for agent, color, label in (
        (focus, FOCUS_COLOR, "focus"),
        (candidate, CANDIDATE_COLOR, "candidate"),
    ):
        speed = agent[start : end + 1, 2].astype(np.float32).copy()
        speed[agent[start : end + 1, 5] <= 0.5] = np.nan
        ax.plot(time_s, speed, color=color, linewidth=1.25, label=label)
    ax.axvline((query_step - event_step) * dt, color="#2ca02c", linestyle=":", linewidth=1.0)
    ax.axvline(0.0, color="#d62728", linestyle=":", linewidth=1.0)
    ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax.set_ylabel("m/s", fontsize=6)
    ax.set_xlabel("time from event (s)", fontsize=6)
    ax.grid(alpha=0.18, linewidth=0.4)
    ax.tick_params(labelsize=5.5)


def _sample_row(
    samples: dict[str, np.ndarray],
    *,
    anchor_index: int,
    role: str,
    rank: int,
    sample_index: int,
    distance: float | None,
    figure: str,
    dt: float,
) -> dict[str, object]:
    relation_names = samples["relation_names"].tolist()
    response_names = samples["response_names"].tolist()
    response_index = int(samples["response_index"][sample_index])
    focus_type = int(samples["focus_type"][sample_index])
    candidate_type = int(samples["candidate_type"][sample_index])
    return {
        "anchor_index": int(anchor_index),
        "role": role,
        "rank": int(rank),
        "sample_index": int(sample_index),
        "distance": None if distance is None else float(distance),
        "scenario_id": str(samples["scenario_id"][sample_index]),
        "source_path": str(samples["source_path"][sample_index]),
        "focus_agent_id": int(samples["focus_agent_id"][sample_index]),
        "candidate_agent_id": int(samples["candidate_agent_id"][sample_index]),
        "candidate_index": int(samples["candidate_index"][sample_index]),
        "event_step": int(samples["event_step"][sample_index]),
        "query_step": int(samples["query_step"][sample_index]),
        "lead_steps": int(samples["lead_steps"][sample_index]),
        "lead_time_s": float(samples["lead_steps"][sample_index]) * float(dt),
        "relation": relation_names[int(samples["relation_index"][sample_index])],
        "response": response_names[response_index] if response_index >= 0 else "ambiguous",
        "focus_type": AGENT_TYPE_NAMES.get(focus_type, str(focus_type)),
        "candidate_type": AGENT_TYPE_NAMES.get(candidate_type, str(candidate_type)),
        "delta_arrival_time_s": float(samples["delta_arrival_time_s"][sample_index]),
        "pet_s": float(samples["pet_s"][sample_index]),
        "spatial_min_dist_m": float(samples["spatial_min_dist_m"][sample_index]),
        "figure": figure,
    }


def render_sample_figure(
    sample_index: int,
    samples: dict[str, np.ndarray],
    *,
    role: str,
    rank: int,
    distance: float | None,
    output_path: Path,
    radius_m: float,
    post_event_steps: int,
    history_steps: int,
    dt: float,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(6.2, 7.4), dpi=dpi, constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(4.2, 1.0), hspace=0.04)
    spatial_ax = fig.add_subplot(grid[0])
    speed_ax = fig.add_subplot(grid[1])
    scene = _load_scene(str(samples["source_path"][sample_index]))
    start, end, query_step, event_step, focus, candidate = _draw_spatial_panel(
        spatial_ax,
        scene,
        samples,
        sample_index,
        role=role,
        rank=rank,
        distance=distance,
        radius_m=radius_m,
        post_event_steps=post_event_steps,
        history_steps=history_steps,
        dt=dt,
    )
    _draw_speed_panel(
        speed_ax,
        focus,
        candidate,
        start=start,
        end=end,
        query_step=query_step,
        event_step=event_step,
        dt=dt,
    )
    scenario_id = str(samples["scenario_id"][sample_index])
    source_name = Path(str(samples["source_path"][sample_index])).name
    speed_ax.text(
        0.99,
        0.96,
        scenario_id,
        transform=speed_ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#555555",
    )
    fig.suptitle(
        "blue=focus, orange=candidate, gray=context agents; solid=20-step history, dashed=future, star=arrival\n"
        f"source={source_name}",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_html(path: Path, rows: list[dict[str, object]]) -> None:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["anchor_index"])].append(row)
    cards: list[str] = []
    for anchor_index, group in grouped.items():
        anchor = group[0]
        role_sections: list[str] = []
        for role in ("anchor", "positive", "hard_negative", "easy_negative"):
            role_rows = [row for row in group if row["role"] == role]
            tiles: list[str] = []
            for row in role_rows:
                figure = str(row["figure"])
                source_name = Path(str(row["source_path"])).name
                rank = "" if not row["rank"] else f" {row['rank']}"
                tiles.append(
                    "<article class='tile'>"
                    f"<a href='{html.escape(figure)}'><img src='{html.escape(figure)}' "
                    f"alt='{html.escape(role)} {row['sample_index']}'></a>"
                    f"<h4 class='{html.escape(role)}'>{html.escape(role)}{rank} — idx={row['sample_index']}</h4>"
                    f"<p>d={_fmt(row['distance'])}; {html.escape(str(row['relation']))} / "
                    f"{html.escape(str(row['response']))}; lead={float(row['lead_time_s']):.1f}s; "
                    f"PET={_fmt(row['pet_s'], 1)}s; Δarrival={_fmt(row['delta_arrival_time_s'], 1)}s</p>"
                    f"<p class='source' title='{html.escape(str(row['source_path']))}'>{html.escape(source_name)}</p>"
                    "</article>"
                )
            role_sections.append(
                f"<h3 class='{html.escape(role)}'>{html.escape(role.replace('_', ' ').title())} ({len(role_rows)})</h3>"
                f"<div class='grid'>{''.join(tiles)}</div>"
            )
        cards.append(
            "<section>"
            f"<h2>Anchor {anchor_index}: {html.escape(str(anchor['relation']))} / "
            f"{html.escape(str(anchor['response']))} / lead={float(anchor['lead_time_s']):.1f}s</h2>"
            f"{''.join(role_sections)}"
            "</section>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Matched interaction audit</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #222; background: #f5f5f5; }}
h1 {{ margin-bottom: 6px; }} .note {{ color: #555; margin-bottom: 24px; }}
section {{ background: white; padding: 18px; margin: 0 0 28px; border-radius: 8px; box-shadow: 0 1px 5px #ccc; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; margin-bottom: 20px; }}
.tile {{ border: 1px solid #ddd; border-radius: 6px; padding: 8px; background: #fff; }}
.tile img {{ display: block; width: 100%; height: auto; }} .tile h4 {{ margin: 8px 0 4px; }}
.tile p {{ margin: 3px 0; font-size: 12px; }} .source {{ color: #777; overflow-wrap: anywhere; }}
.anchor {{ color: #222; font-weight: 700; }} .positive {{ color: #2ca02c; font-weight: 700; }}
.hard_negative {{ color: #d62728; font-weight: 700; }} .easy_negative {{ color: #9467bd; font-weight: 700; }}
</style></head><body>
<h1>Matched interaction visual audit</h1>
<p class="note">Each sample is a separate image stored under its anchor/role directory. This page is only a browsing index.</p>
{''.join(cards)}
</body></html>"""
    path.write_text(document)


def build_gallery(args: argparse.Namespace) -> None:
    samples_path = Path(args.samples_npz)
    matches_path = Path(args.matches_npz)
    samples = _load_selected_arrays(samples_path, SAMPLE_FIELDS)
    matches = _load_selected_arrays(matches_path, MATCH_FIELDS)
    n = len(samples["scenario_id"])
    if len(matches["trainable_anchor_mask"]) != n:
        raise ValueError("Sample and match caches have different lengths")

    explicit = _parse_int_list(args.anchor_indices)
    if explicit:
        invalid = [idx for idx in explicit if idx < 0 or idx >= n]
        if invalid:
            raise IndexError(f"Anchor indices outside [0, {n}): {invalid}")
        anchor_indices = list(dict.fromkeys(explicit))
    else:
        anchor_indices = select_anchor_indices(
            samples,
            matches,
            count=args.num_anchors,
            seed=args.seed,
            selection=args.selection,
            relation_filter=_parse_name_set(args.relations),
            response_filter=_parse_name_set(args.responses),
            lead_filter=set(_parse_int_list(args.lead_steps)),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    relation_names = samples["relation_names"].tolist()
    response_names = samples["response_names"].tolist()
    all_rows: list[dict[str, object]] = []
    for position, anchor_index in enumerate(anchor_indices, start=1):
        relation = relation_names[int(samples["relation_index"][anchor_index])]
        response_idx = int(samples["response_index"][anchor_index])
        response = response_names[response_idx] if response_idx >= 0 else "ambiguous"
        lead = int(samples["lead_steps"][anchor_index])
        anchor_dir = output_dir / f"anchor_{anchor_index:05d}_{relation}_{response}_lead{lead}"
        print(f"[{position}/{len(anchor_indices)}] rendering anchor {anchor_index} -> {anchor_dir.name}/", flush=True)
        anchor_rows: list[dict[str, object]] = []
        for role, rank, sample_index, distance in related_samples(anchor_index, matches):
            role_dir = anchor_dir / role
            distance_suffix = "" if distance is None else f"_d{distance:.2f}"
            filename = f"{role}_{rank:02d}_idx{sample_index:05d}{distance_suffix}.png"
            output_path = role_dir / filename
            render_sample_figure(
                sample_index,
                samples,
                role=role,
                rank=rank,
                distance=distance,
                output_path=output_path,
                radius_m=args.spatial_radius_m,
                post_event_steps=args.post_event_steps,
                history_steps=args.history_steps,
                dt=args.dt,
                dpi=args.dpi,
            )
            anchor_rows.append(
                _sample_row(
                    samples,
                    anchor_index=anchor_index,
                    role=role,
                    rank=rank,
                    sample_index=sample_index,
                    distance=distance,
                    figure=output_path.relative_to(output_dir).as_posix(),
                    dt=args.dt,
                )
            )
        _write_csv(anchor_dir / "manifest.csv", anchor_rows)
        (anchor_dir / "manifest.json").write_text(json.dumps(anchor_rows, indent=2))
        all_rows.extend(anchor_rows)

    _write_csv(output_dir / "gallery_manifest.csv", all_rows)
    (output_dir / "gallery_manifest.json").write_text(json.dumps(all_rows, indent=2))
    _write_html(output_dir / "index.html", all_rows)
    config = {
        "samples_npz": str(samples_path),
        "matches_npz": str(matches_path),
        "anchor_indices": anchor_indices,
        "selection": args.selection if not explicit else "explicit",
        "seed": args.seed,
        "num_anchors": len(anchor_indices),
        "relations": sorted(_parse_name_set(args.relations)),
        "responses": sorted(_parse_name_set(args.responses)),
        "lead_steps": sorted(set(_parse_int_list(args.lead_steps))),
        "spatial_radius_m": args.spatial_radius_m,
        "history_steps": args.history_steps,
        "post_event_steps": args.post_event_steps,
        "dt": args.dt,
    }
    (output_dir / "gallery_config.json").write_text(json.dumps(config, indent=2))
    print(f"saved {output_dir / 'index.html'}", flush=True)
    print(f"saved {output_dir / 'gallery_manifest.csv'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples_npz",
        default="cache/interaction_contrastive_learning_5k/train_samples.npz",
    )
    parser.add_argument(
        "--matches_npz",
        default="cache/interaction_contrastive_learning_5k/train_matches.npz",
    )
    parser.add_argument(
        "--output_dir",
        default="cache/interaction_contrastive_learning_5k/visual_audit_folders",
    )
    parser.add_argument("--num_anchors", type=int, default=24)
    parser.add_argument("--selection", choices=("stratified", "random"), default="stratified")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor_indices", default="", help="Comma-separated explicit sample indices")
    parser.add_argument("--relations", default="", help="Comma-separated relation names")
    parser.add_argument("--responses", default="", help="Comma-separated response names")
    parser.add_argument("--lead_steps", default="", help="Comma-separated lead steps, e.g. 10,20,30")
    parser.add_argument("--spatial_radius_m", type=float, default=60.0)
    parser.add_argument("--history_steps", type=int, default=20)
    parser.add_argument("--post_event_steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=120)
    return parser


def main() -> None:
    build_gallery(build_argparser().parse_args())


if __name__ == "__main__":
    main()
