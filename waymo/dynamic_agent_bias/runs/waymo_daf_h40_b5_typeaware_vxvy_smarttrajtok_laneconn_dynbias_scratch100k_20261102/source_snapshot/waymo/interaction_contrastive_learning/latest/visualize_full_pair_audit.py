"""Build a stratified visual audit of the v2 full 91-step pair dataset."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


COLORS = ("#1f77b4", "#ff7f0e")
CATEGORY_SPECS = (
    ("following", "Same-lane following", 6),
    ("merge_cut_in", "Merge / cut-in", 6),
    ("crossing", "Crossing paths", 8),
    ("oncoming", "Oncoming", 4),
    ("vru", "Vehicle-pedestrian / cyclist", 6),
    ("ooi_fallback", "OOI closest fallback", 6),
)
TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
CURATED_MERGE_SAMPLES = (
    ("train", 49615, "ooi_turn_in"),
    ("train", 83196, "mined_shallow_merge"),
    ("train", 57692, "high_pet_delayed_merge"),
    ("val", 11261, "long_gradual_merge"),
    ("train", 16840, "ooi_two_branch_merge"),
    ("train", 65945, "mined_lane_convergence"),
)


def _wrap_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _heading_at(trajectory: np.ndarray, agent: int, step: float) -> float:
    index = int(np.clip(round(float(step)), 0, trajectory.shape[1] - 1))
    return float(np.arctan2(trajectory[agent, index, 4], trajectory[agent, index, 5]))


def _speed(trajectory: np.ndarray) -> np.ndarray:
    return np.linalg.norm(trajectory[:, :, 2:4], axis=-1)


def _window_values(values: np.ndarray, mask: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, int(start))
    end = min(len(values), int(end))
    if end <= start:
        return np.empty((0,), dtype=np.float32)
    return values[start:end][mask[start:end]]


def _path_endpoint_change(trajectory: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    """Return coarse path convergence/divergence, independent of synchronized time."""
    paths = []
    for agent in range(2):
        points = trajectory[agent, valid_mask[agent], 0:2]
        stride = max(1, int(np.ceil(len(points) / 24)))
        paths.append(points[::stride])
    if min(map(len, paths)) < 4:
        return 0.0, 0.0
    changes = []
    for agent in range(2):
        own, other = paths[agent], paths[1 - agent]
        width = max(2, len(own) // 4)
        endpoint_distance = []
        for endpoint in (own[:width], own[-width:]):
            pairwise = np.linalg.norm(endpoint[:, None, :] - other[None, :, :], axis=-1)
            endpoint_distance.append(float(np.median(pairwise.min(axis=1))))
        changes.append(endpoint_distance[0] - endpoint_distance[1])
    return float(max(changes)), float(max(-value for value in changes))


def _trajectory_descriptors(
    trajectory: np.ndarray,
    valid_mask: np.ndarray,
    primary_steps: tuple[float, float],
) -> dict[str, float | bool]:
    yaw_first = _heading_at(trajectory, 0, primary_steps[0])
    yaw_second = _heading_at(trajectory, 1, primary_steps[1])
    heading_diff = abs(float(_wrap_angle(yaw_first - yaw_second))) * 180.0 / np.pi
    speed = _speed(trajectory)
    event_speed = []
    speed_drop = []
    yaw_change = []
    for agent, step in enumerate(primary_steps):
        centre = int(round(step))
        before = _window_values(speed[agent], valid_mask[agent], centre - 20, centre + 1)
        after = _window_values(speed[agent], valid_mask[agent], centre, centre + 21)
        at = _window_values(speed[agent], valid_mask[agent], centre, centre + 1)
        event_speed.append(float(at[0]) if len(at) else 0.0)
        speed_drop.append(max(0.0, float(before.max() - after.min())) if len(before) and len(after) else 0.0)
        start, end = max(0, centre - 20), min(trajectory.shape[1], centre + 21)
        usable = valid_mask[agent, start:end]
        yaw = np.arctan2(trajectory[agent, start:end, 4], trajectory[agent, start:end, 5])
        yaw = np.unwrap(yaw[usable])
        yaw_change.append(float(np.ptp(yaw)) * 180.0 / np.pi if len(yaw) else 0.0)

    relative_lateral = trajectory[1, :, 1] - trajectory[0, :, 1]
    joint = valid_mask[0] & valid_mask[1]
    reference = int(round(min(primary_steps)))
    before_lat = _window_values(relative_lateral, joint, reference - 25, reference - 5)
    after_lat = _window_values(relative_lateral, joint, reference + 5, reference + 26)
    lateral_change = abs(float(np.median(after_lat) - np.median(before_lat))) if len(before_lat) and len(after_lat) else 0.0
    valid_speed = [speed[agent, valid_mask[agent]] for agent in range(2)]
    median_speed = [float(np.median(values)) if len(values) else 0.0 for values in valid_speed]
    max_valid_speed = max((float(values.max()) for values in valid_speed if len(values)), default=0.0)
    joint_lateral = np.abs(relative_lateral[joint])
    path_convergence, path_divergence = _path_endpoint_change(trajectory, valid_mask)
    return {
        "heading_diff_deg": heading_diff,
        "mean_event_speed_mps": float(np.mean(event_speed)),
        "min_median_speed_mps": float(min(median_speed)),
        "max_valid_speed_mps": max_valid_speed,
        "max_speed_drop_mps": float(max(speed_drop)),
        "max_yaw_change_deg": float(max(yaw_change)),
        "relative_lateral_change_m": lateral_change,
        "median_abs_relative_lateral_m": float(np.median(joint_lateral)) if len(joint_lateral) else float("inf"),
        "path_convergence_m": path_convergence,
        "path_divergence_m": path_divergence,
        "partial_valid": bool(valid_mask.sum() < valid_mask.size),
        "valid_fraction": float(valid_mask.mean()),
    }


def load_descriptor_index(dataset_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "val"):
        csv_path = dataset_dir / f"{split}_samples.csv"
        with csv_path.open(newline="") as handle:
            split_rows = [dict(row) for row in csv.DictReader(handle)]
        by_shard: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in split_rows:
            by_shard[row["shard"]].append(row)
        for shard_name, shard_rows in by_shard.items():
            with np.load(dataset_dir / shard_name, allow_pickle=False) as data:
                trajectory = data["trajectory"]
                valid_mask = data["valid_mask"]
                primary_first = data["primary_step_first"]
                primary_second = data["primary_step_second"]
                intervals = data["interaction_interval"]
                for row in shard_rows:
                    shard_row = int(row["shard_row"])
                    descriptor = _trajectory_descriptors(
                        trajectory[shard_row],
                        valid_mask[shard_row],
                        (float(primary_first[shard_row]), float(primary_second[shard_row])),
                    )
                    types = tuple(sorted((int(row["first_agent_type"]), int(row["second_agent_type"]))))
                    interval = intervals[shard_row]
                    parsed: dict[str, object] = {
                        **row,
                        **descriptor,
                        "sample_index": int(row["sample_index"]),
                        "shard_row": shard_row,
                        "first_agent_id": int(row["first_agent_id"]),
                        "second_agent_id": int(row["second_agent_id"]),
                        "first_agent_type": int(row["first_agent_type"]),
                        "second_agent_type": int(row["second_agent_type"]),
                        "is_original_ooi_pair": row["is_original_ooi_pair"] == "True",
                        "primary_step_first": float(row["primary_step_first"]),
                        "primary_step_second": float(row["primary_step_second"]),
                        "zone_pet_s": float(row["zone_pet_s"]),
                        "center_pet_s": float(row["center_pet_s"]),
                        "min_clearance_m": float(row["min_clearance_m"]),
                        "interval_length_steps": float(max(interval[1] - interval[0], interval[3] - interval[2])),
                        "types": types,
                    }
                    rows.append(parsed)
    return rows


def category_pools(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    pools = {name: [] for name, _, _ in CATEGORY_SPECS}
    for row in rows:
        mode = str(row["event_mode"])
        types = row["types"]
        heading = float(row["heading_diff_deg"])
        interval = float(row["interval_length_steps"])
        lateral_change = float(row["relative_lateral_change_m"])
        lateral_offset = float(row["median_abs_relative_lateral_m"])
        yaw_change = float(row["max_yaw_change_deg"])
        path_convergence = float(row["path_convergence_m"])
        path_divergence = float(row["path_divergence_m"])
        moving = float(row["min_median_speed_mps"]) >= 1.0
        plausible = float(row["max_valid_speed_mps"]) <= 40.0
        mostly_valid = float(row["valid_fraction"]) >= 0.80
        if mode == "ooi_closest_fallback":
            pools["ooi_fallback"].append(row)
        if 2 in types or 3 in types:
            if 1 in types and plausible and float(row["valid_fraction"]) >= 0.70:
                pools["vru"].append(row)
            continue
        if types != (1, 1) or mode == "ooi_closest_fallback":
            continue
        if mode == "path_intersection" and 45.0 <= heading < 135.0 and plausible and mostly_valid:
            pools["crossing"].append(row)
        if heading >= 135.0 and moving and plausible and mostly_valid:
            pools["oncoming"].append(row)
        merge_signal = path_convergence >= 1.5 and (
            lateral_change >= 1.5
            or (yaw_change >= 12.0 and lateral_change >= 0.5)
            or (15.0 <= heading < 45.0 and lateral_change >= 0.5)
        )
        if heading < 45.0 and merge_signal and moving and plausible and mostly_valid:
            pools["merge_cut_in"].append(row)
        stable_following = (
            heading < 15.0
            and interval >= 40.0
            and lateral_change < 1.5
            and lateral_offset < 2.5
            and yaw_change < 12.0
            and moving
            and plausible
            and mostly_valid
        )
        if stable_following:
            pools["following"].append(row)
    return pools


def _available(pool: list[dict[str, object]], used_scenes: set[str], predicate=None) -> list[dict[str, object]]:
    result = [row for row in pool if str(row["scenario_id"]) not in used_scenes]
    if predicate is not None:
        result = [row for row in result if predicate(row)]
    return result


def _pick_metric(
    pool: list[dict[str, object]],
    used_scenes: set[str],
    metric: str,
    quantile: float,
    predicate=None,
) -> dict[str, object] | None:
    available = _available(pool, used_scenes, predicate)
    if not available:
        return None
    values = np.asarray([float(row[metric]) for row in available])
    target = float(np.quantile(values, quantile))
    return min(available, key=lambda row: (abs(float(row[metric]) - target), str(row["scenario_id"])))


def _pick_medoid(pool: list[dict[str, object]], used_scenes: set[str], predicate=None) -> dict[str, object] | None:
    available = _available(pool, used_scenes, predicate)
    if not available:
        return None
    keys = ("zone_pet_s", "interval_length_steps", "heading_diff_deg", "mean_event_speed_mps", "max_speed_drop_mps")
    values = np.asarray([[float(row[key]) for key in keys] for row in available], dtype=np.float32)
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-3)
    distance = np.sqrt(np.mean(((values - median) / scale) ** 2, axis=1))
    return available[int(np.argmin(distance))]


def _pick_partial_response(pool: list[dict[str, object]], used_scenes: set[str]) -> dict[str, object] | None:
    available = _available(
        pool,
        used_scenes,
        lambda row: bool(row["partial_valid"])
        and float(row["valid_fraction"]) >= 0.75
        and float(row["max_valid_speed_mps"]) <= 40.0,
    )
    if not available:
        available = _available(pool, used_scenes)
    if not available:
        return None
    values = np.asarray([float(row["max_speed_drop_mps"]) for row in available])
    target = float(np.quantile(values, 0.85))
    return min(
        available,
        key=lambda row: (
            abs(float(row["max_speed_drop_mps"]) - target),
            -float(row["valid_fraction"]),
            str(row["scenario_id"]),
        ),
    )


def _commit(
    selected: list[dict[str, object]],
    row: dict[str, object] | None,
    category: str,
    role: str,
    used_scenes: set[str],
) -> None:
    if row is None:
        return
    item = dict(row)
    item["audit_category"] = category
    item["selection_role"] = role
    selected.append(item)
    used_scenes.add(str(row["scenario_id"]))


def select_gallery(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pools = category_pools(rows)
    selected: list[dict[str, object]] = []
    used_scenes: set[str] = set()
    for category, _, count in CATEGORY_SPECS:
        pool = pools[category]
        before = len(selected)
        if category == "merge_cut_in":
            for split, sample_index, role in CURATED_MERGE_SAMPLES:
                matches = [
                    row for row in rows
                    if str(row["split"]) == split and int(row["sample_index"]) == sample_index
                ]
                if len(matches) != 1:
                    raise RuntimeError(f"Cannot resolve curated merge sample {split}:{sample_index}")
                _commit(selected, matches[0], category, role, used_scenes)
        elif category == "vru":
            for types, label in (((1, 2), "vehicle_pedestrian"), ((1, 3), "vehicle_cyclist")):
                subgroup = [row for row in pool if row["types"] == types]
                _commit(selected, _pick_medoid(subgroup, used_scenes, lambda row: bool(row["is_original_ooi_pair"])), category, f"{label}_ooi_typical", used_scenes)
                _commit(selected, _pick_medoid(subgroup, used_scenes, lambda row: not bool(row["is_original_ooi_pair"])), category, f"{label}_mined_typical", used_scenes)
                _commit(selected, _pick_metric(subgroup, used_scenes, "zone_pet_s", 0.9), category, f"{label}_high_pet", used_scenes)
        elif category == "ooi_fallback":
            specs = (
                ("zone_pet_s", 0.1, "low_pet"),
                ("zone_pet_s", 0.5, "median_pet"),
                ("zone_pet_s", 0.9, "high_pet"),
                ("min_clearance_m", 0.1, "low_clearance"),
                ("min_clearance_m", 0.9, "high_clearance"),
            )
            for metric, quantile, role in specs:
                _commit(selected, _pick_metric(pool, used_scenes, metric, quantile), category, role, used_scenes)
            _commit(selected, _pick_partial_response(pool, used_scenes), category, "partial_or_strong_response", used_scenes)
        else:
            _commit(selected, _pick_medoid(pool, used_scenes, lambda row: bool(row["is_original_ooi_pair"])), category, "ooi_typical", used_scenes)
            _commit(selected, _pick_medoid(pool, used_scenes, lambda row: not bool(row["is_original_ooi_pair"])), category, "mined_typical", used_scenes)
            low_pet_predicate = None
            if category == "oncoming":
                low_pet_predicate = lambda row: (
                    str(row["event_mode"]) == "path_intersection"
                    and 0.0 < float(row["zone_pet_s"])
                    and float(row["interval_length_steps"]) < 60.0
                )
            _commit(
                selected,
                _pick_metric(pool, used_scenes, "zone_pet_s", 0.1, low_pet_predicate),
                category,
                "low_pet",
                used_scenes,
            )
            _commit(selected, _pick_metric(pool, used_scenes, "zone_pet_s", 0.9), category, "high_pet", used_scenes)
            if count >= 6:
                _commit(selected, _pick_metric(pool, used_scenes, "interval_length_steps", 0.9), category, "long_interval", used_scenes)
                _commit(selected, _pick_partial_response(pool, used_scenes), category, "partial_or_strong_response", used_scenes)
            if count >= 8:
                _commit(selected, _pick_metric(pool, used_scenes, "heading_diff_deg", 0.5), category, "median_heading", used_scenes)
                _commit(selected, _pick_metric(pool, used_scenes, "max_speed_drop_mps", 0.9), category, "strong_response", used_scenes)
        while len(selected) - before < count:
            fallback = _pick_medoid(pool, used_scenes)
            if fallback is None:
                raise RuntimeError(f"Not enough unique scenes for category {category}: pool={len(pool)}")
            _commit(selected, fallback, category, "diversity_fill", used_scenes)
    return selected


@lru_cache(maxsize=32)
def _load_source(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "agents": np.asarray(data["agents"], dtype=np.float32),
            "agent_mask": np.asarray(data["agent_mask"], dtype=bool),
            "agent_ids": np.asarray(data["agent_ids"], dtype=np.int64),
            "map_polylines": np.asarray(data["map_polylines"], dtype=np.float32),
            "map_mask": np.asarray(data["map_mask"], dtype=bool),
        }


def _interpolate_yaw(agent: np.ndarray, step: float) -> float:
    step = float(np.clip(step, 0.0, len(agent) - 1.0))
    low = int(np.floor(step))
    high = min(len(agent) - 1, low + 1)
    fraction = step - low
    return float(_wrap_angle(float(agent[low, 6]) + fraction * float(_wrap_angle(agent[high, 6] - agent[low, 6]))))


def _transform(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(xy, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return delta @ np.asarray([[c, s], [-s, c]], dtype=np.float32).T


def _draw_map(ax: plt.Axes, source: dict[str, np.ndarray], origin: np.ndarray, yaw: float, radius: float) -> None:
    for polyline, mask in zip(source["map_polylines"], source["map_mask"]):
        points = polyline[mask, 0:2]
        if len(points) < 2:
            continue
        points = _transform(points, origin, yaw)
        if bool((np.abs(points) <= radius * 1.3).all(axis=1).any()):
            ax.plot(points[:, 0], points[:, 1], color="#c9c9c9", linewidth=0.45, alpha=0.65, zorder=1)


def _obb_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    local = np.asarray([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2],
    ], dtype=np.float32)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return local @ np.asarray([[c, s], [-s, c]], dtype=np.float32) + np.asarray([x, y])


def _plot_agent(
    ax: plt.Axes,
    trajectory: np.ndarray,
    valid: np.ndarray,
    interval: tuple[float, float],
    primary: float,
    color: str,
    label: str,
) -> None:
    xy = trajectory[:, 0:2].copy()
    xy[~valid] = np.nan
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.7, alpha=0.65, label=label, zorder=4)
    start = int(np.clip(np.floor(interval[0]), 0, len(xy) - 1))
    end = int(np.clip(np.ceil(interval[1]), 0, len(xy) - 1))
    active = xy[start : end + 1]
    ax.plot(active[:, 0], active[:, 1], color=color, linewidth=3.2, alpha=0.95, zorder=5)
    primary_index = int(np.clip(round(primary), 0, len(xy) - 1))
    ax.scatter(xy[primary_index, 0], xy[primary_index, 1], marker="*", s=120, color=color,
               edgecolors="black", linewidths=0.7, zorder=8)
    for index, marker in ((start, "s"), (end, "D")):
        if valid[index]:
            ax.scatter(xy[index, 0], xy[index, 1], marker=marker, s=38, color=color,
                       edgecolors="white", linewidths=0.7, zorder=7)


def render_sample(row: dict[str, object], dataset_dir: Path, output_path: Path, dpi: int) -> None:
    shard_path = dataset_dir / str(row["shard"])
    shard_row = int(row["shard_row"])
    with np.load(shard_path, allow_pickle=False) as data:
        trajectory = np.asarray(data["trajectory"][shard_row], dtype=np.float32)
        valid_mask = np.asarray(data["valid_mask"][shard_row], dtype=bool)
        size = np.asarray(data["agent_size_m"][shard_row], dtype=np.float32)
        interval = np.asarray(data["interaction_interval"][shard_row], dtype=np.float32)
        conflict_xy = np.asarray(data["conflict_xy"][shard_row], dtype=np.float32)

    source = _load_source(str(row["source_path"]))
    first_id = int(row["first_agent_id"])
    found = np.flatnonzero(source["agent_mask"] & (source["agent_ids"] == first_id))
    if not len(found):
        raise RuntimeError(f"Cannot find first agent id={first_id} in {row['source_path']}")
    frame_yaw = _interpolate_yaw(source["agents"][int(found[0])], float(row["primary_step_first"]))
    primary = (float(row["primary_step_first"]), float(row["primary_step_second"]))
    intervals = ((float(interval[0]), float(interval[1])), (float(interval[2]), float(interval[3])))

    fig = plt.figure(figsize=(12.2, 8.4), dpi=dpi, constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.45, 1.0), height_ratios=(1.35, 1.0, 1.0))
    full_ax = fig.add_subplot(grid[:, 0])
    zoom_ax = fig.add_subplot(grid[0, 1])
    speed_ax = fig.add_subplot(grid[1, 1])
    relative_ax = fig.add_subplot(grid[2, 1])

    valid_xy = np.concatenate([trajectory[a, valid_mask[a], 0:2] for a in range(2)], axis=0)
    radius = max(25.0, min(100.0, float(np.quantile(np.abs(valid_xy), 0.98)) * 1.2)) if len(valid_xy) else 40.0
    _draw_map(full_ax, source, conflict_xy, frame_yaw, radius)
    for agent in range(2):
        _plot_agent(
            full_ax, trajectory[agent], valid_mask[agent], intervals[agent], primary[agent], COLORS[agent],
            f"agent {agent + 1}: id={row['first_agent_id'] if agent == 0 else row['second_agent_id']}",
        )
    full_ax.set_xlim(-radius, radius)
    full_ax.set_ylim(-radius, radius)
    full_ax.set_aspect("equal", adjustable="box")
    full_ax.grid(alpha=0.18, linewidth=0.45)
    full_ax.axhline(0, color="#777", linewidth=0.4)
    full_ax.axvline(0, color="#777", linewidth=0.4)
    full_ax.set_xlabel("event-frame longitudinal position (m)")
    full_ax.set_ylabel("event-frame lateral position (m)")
    full_ax.set_title("Full 91-step trajectories; thick = interaction interval", fontsize=10)
    full_ax.legend(loc="best", fontsize=8)

    _draw_map(zoom_ax, source, conflict_xy, frame_yaw, 22.0)
    for agent in range(2):
        _plot_agent(zoom_ax, trajectory[agent], valid_mask[agent], intervals[agent], primary[agent], COLORS[agent], f"agent {agent + 1}")
        index = int(np.clip(round(primary[agent]), 0, 90))
        yaw = float(np.arctan2(trajectory[agent, index, 4], trajectory[agent, index, 5]))
        corners = _obb_corners(
            float(trajectory[agent, index, 0]), float(trajectory[agent, index, 1]), yaw,
            float(size[agent, 0]), float(size[agent, 1]),
        )
        zoom_ax.add_patch(Polygon(corners, closed=True, facecolor=COLORS[agent], edgecolor="black", alpha=0.3, zorder=7))
    zoom_ax.set_xlim(-22, 22)
    zoom_ax.set_ylim(-22, 22)
    zoom_ax.set_aspect("equal", adjustable="box")
    zoom_ax.grid(alpha=0.18, linewidth=0.4)
    zoom_ax.set_title("Contact-region zoom and OBBs", fontsize=9)
    zoom_ax.tick_params(labelsize=7)

    times = np.arange(91) * 0.1
    speeds = _speed(trajectory)
    for agent in range(2):
        values = speeds[agent].copy()
        values[~valid_mask[agent]] = np.nan
        speed_ax.plot(times, values, color=COLORS[agent], linewidth=1.5, label=f"agent {agent + 1}")
        speed_ax.axvline(primary[agent] * 0.1, color=COLORS[agent], linestyle="--", linewidth=1.0)
        speed_ax.axvspan(intervals[agent][0] * 0.1, intervals[agent][1] * 0.1, color=COLORS[agent], alpha=0.08)
    speed_ax.set_xlim(0, 9)
    speed_ax.set_ylabel("speed (m/s)")
    speed_ax.set_title("Speed; dashed = primary arrival, shading = interval", fontsize=9)
    speed_ax.grid(alpha=0.2)
    speed_ax.legend(fontsize=7)

    joint = valid_mask[0] & valid_mask[1]
    rel_x = trajectory[1, :, 0] - trajectory[0, :, 0]
    rel_y = trajectory[1, :, 1] - trajectory[0, :, 1]
    rel_x[~joint] = np.nan
    rel_y[~joint] = np.nan
    relative_ax.plot(times, rel_x, color="#2ca02c", linewidth=1.4, label="relative longitudinal")
    relative_ax.plot(times, rel_y, color="#9467bd", linewidth=1.4, label="relative lateral")
    invalid = ~joint
    relative_ax.fill_between(times, 0, 1, where=invalid, transform=relative_ax.get_xaxis_transform(),
                             color="#d62728", alpha=0.1, label="not jointly valid")
    relative_ax.axhline(0, color="#777", linewidth=0.5)
    relative_ax.set_xlim(0, 9)
    relative_ax.set_xlabel("scenario time (s)")
    relative_ax.set_ylabel("relative position (m)")
    relative_ax.set_title(
        f"Relative state and validity: {int(valid_mask[0].sum())}/91, {int(valid_mask[1].sum())}/91 valid",
        fontsize=9,
    )
    relative_ax.grid(alpha=0.2)
    relative_ax.legend(fontsize=7, ncol=2)

    source_label = "OOI" if bool(row["is_original_ooi_pair"]) else "mined"
    type_label = f"{TYPE_NAMES.get(int(row['first_agent_type']), row['first_agent_type'])}–{TYPE_NAMES.get(int(row['second_agent_type']), row['second_agent_type'])}"
    fig.suptitle(
        f"{row['audit_category']} | {row['selection_role']} | {source_label} | {type_label}\n"
        f"{row['split']} scenario={row['scenario_id']} sample={row['sample_index']} | mode={row['event_mode']} | "
        f"heading={float(row['heading_diff_deg']):.1f}° | zone PET={float(row['zone_pet_s']):.1f}s | "
        f"center PET={float(row['center_pet_s']):.1f}s | clearance={float(row['min_clearance_m']):.2f}m | "
        f"interval={float(row['interval_length_steps']):.1f} steps",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_html(path: Path, rows: list[dict[str, object]]) -> None:
    sections = []
    for category, title, _ in CATEGORY_SPECS:
        category_rows = [row for row in rows if row["audit_category"] == category]
        tiles = []
        for row in category_rows:
            figure = html.escape(str(row["figure"]))
            source = "OOI" if row["is_original_ooi_pair"] else "mined"
            tiles.append(
                f"<article><a href='{figure}'><img src='{figure}'></a>"
                f"<h3>{html.escape(str(row['selection_role']))}</h3>"
                f"<p>{source}; {html.escape(str(row['event_mode']))}; scenario={html.escape(str(row['scenario_id']))}</p>"
                f"<p>heading={float(row['heading_diff_deg']):.1f}°; PET={float(row['zone_pet_s']):.1f}s; "
                f"interval={float(row['interval_length_steps']):.1f} steps; valid={float(row['valid_fraction']):.0%}</p></article>"
            )
        sections.append(f"<section><h2>{html.escape(title)} ({len(category_rows)})</h2><div>{''.join(tiles)}</div></section>")
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Full-pair v2 visual audit</title>"
        "<style>body{font-family:system-ui;margin:24px;background:#f4f4f4;color:#222}"
        "section{background:white;padding:18px;margin:20px 0;border-radius:8px}"
        "section>div{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}"
        "article{border:1px solid #ddd;padding:8px;border-radius:6px}img{width:100%}"
        "h3,p{margin:6px 0}p{font-size:13px;color:#555}</style></head><body>"
        "<h1>Full 91-step interaction-pair visual audit</h1>"
        "<p>Temporary geometry groups are used only for audit sampling, not as training labels.</p>"
        + "".join(sections) + "</body></html>"
    )


def build(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("loading descriptors", flush=True)
    rows = load_descriptor_index(dataset_dir)
    pools = category_pools(rows)
    print("candidate pools: " + ", ".join(f"{name}={len(pool)}" for name, pool in pools.items()), flush=True)
    selected = select_gallery(rows)
    for position, row in enumerate(selected, start=1):
        category = str(row["audit_category"])
        filename = f"{position:02d}_{row['selection_role']}_{row['split']}_idx{int(row['sample_index']):06d}_{row['scenario_id']}.png"
        relative = Path(category) / filename
        print(f"[{position}/{len(selected)}] rendering {relative}", flush=True)
        render_sample(row, dataset_dir, output_dir / relative, args.dpi)
        row["figure"] = relative.as_posix()
    _write_csv(output_dir / "selected_samples.csv", selected)
    (output_dir / "selected_samples.json").write_text(json.dumps(selected, indent=2))
    _write_html(output_dir / "index.html", selected)
    (output_dir / "gallery_config.json").write_text(json.dumps({
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "num_samples": len(selected),
        "category_counts": {category: sum(row["audit_category"] == category for row in selected)
                            for category, _, _ in CATEGORY_SPECS},
        "candidate_pool_counts": {name: len(pool) for name, pool in pools.items()},
    }, indent=2))
    print(f"saved {output_dir / 'index.html'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_full_pairs_50k_v2",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_full_pairs_50k_v2/visual_audit_36_curated",
    )
    parser.add_argument("--dpi", type=int, default=130)
    return parser


def main() -> None:
    build(build_argparser().parse_args())


if __name__ == "__main__":
    main()
