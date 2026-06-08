"""Audit filtered Waymo vector-tokenizer NPZ datasets.

This script computes two kinds of metadata:

1. Low-level data quality metrics: shapes, finite checks, valid masks, map
   coverage, traffic-light availability, and ego-frame sanity checks.
2. Heuristic scene labels: ego motion type, intersection-like signals, close
   interactions, following/leading, crossing, and cut-in/merge-like cues.

The labels are intentionally heuristic. They are meant to guide visual
inspection and dataset understanding, not to replace human scenario labels.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


CURRENT_IDX = 10
FPS_HZ = 10.0
EXPECTED_SHAPES = {
    "agents": (16, 91, 8),
    "agent_mask": (16,),
    "map_polylines": (256, 20, 6),
    "map_mask": (256, 20),
    "lights": (91, 16, 4),
    "light_mask": (91, 16),
}


NUMERIC_FIELDS = [
    "num_valid_agent_slots",
    "mean_valid_agents_per_timestep",
    "agent_valid_ratio",
    "valid_agents_current",
    "ego_valid_steps",
    "ego_displacement_m",
    "ego_observed_displacement_m",
    "ego_future_displacement_m",
    "ego_path_length_m",
    "ego_heading_change_deg",
    "ego_lateral_displacement_m",
    "ego_longitudinal_displacement_m",
    "ego_mean_speed_mps",
    "ego_max_speed_mps",
    "num_valid_map_polylines",
    "num_valid_map_points",
    "map_valid_point_ratio",
    "light_valid_ratio",
    "mean_valid_lights_per_timestep",
    "valid_lights_current",
    "min_ego_agent_distance_m",
    "min_ego_agent_distance_observed_m",
    "min_ego_agent_distance_current_m",
    "num_close_agents_10m",
    "num_close_agents_20m",
    "num_crossing_close_agents",
    "num_cut_in_merge_like_agents",
]


CSV_FIELDS = [
    "scenario_id",
    "path",
    "ego_maneuver_label",
    "interaction_labels",
    "quality_flags",
    "intersection_like",
    "has_close_agent_10m",
    "has_close_agent_20m",
    "dense_scene",
    "sparse_scene",
    "map_saturated",
    "has_traffic_lights",
    *NUMERIC_FIELDS,
]


def _as_str(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _first_last_valid(values: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray | None, np.ndarray | None]:
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return None, None
    return values[idx[0]], values[idx[-1]]


def _heading_change(yaw: np.ndarray, valid: np.ndarray) -> float:
    first, last = _first_last_valid(yaw, valid)
    if first is None or last is None:
        return 0.0
    return float(_wrap_angle(float(last) - float(first)))


def _collect_npz_paths(roots: Sequence[str], max_files: int | None) -> List[str]:
    paths: List[str] = []
    for root in roots:
        if root.endswith(".npz"):
            paths.append(root)
        else:
            paths.extend(glob.glob(str(Path(root) / "*.npz")))
    paths = sorted(set(paths))
    if max_files is not None and max_files > 0:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No .npz files found in: {roots}")
    return paths


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _shape_flags(item: Dict[str, np.ndarray]) -> List[str]:
    flags: List[str] = []
    for key, shape in EXPECTED_SHAPES.items():
        if key not in item:
            flags.append(f"missing_{key}")
        elif tuple(item[key].shape) != shape:
            flags.append(f"bad_shape_{key}_{tuple(item[key].shape)}")
    return flags


def _finite_flags(item: Dict[str, np.ndarray]) -> List[str]:
    flags: List[str] = []
    for key in ("agents", "map_polylines", "lights", "ego_origin_xy", "ego_heading"):
        if key in item and not np.isfinite(item[key]).all():
            flags.append(f"nonfinite_{key}")
    return flags


def _ego_metrics(agents: np.ndarray) -> Dict[str, float]:
    ego = agents[0]
    valid = ego[:, 5] > 0.5
    xy = ego[:, 0:2]
    yaw = ego[:, 6]
    speed = ego[:, 2]

    first_xy, last_xy = _first_last_valid(xy, valid)
    displacement = float(np.linalg.norm(last_xy - first_xy)) if first_xy is not None and last_xy is not None else 0.0
    heading_signed = _heading_change(yaw, valid)

    observed_valid = valid[: CURRENT_IDX + 1]
    future_valid = valid[CURRENT_IDX:]
    obs_first, obs_last = _first_last_valid(xy[: CURRENT_IDX + 1], observed_valid)
    fut_first, fut_last = _first_last_valid(xy[CURRENT_IDX:], future_valid)

    valid_speed = speed[valid]
    return {
        "ego_valid_steps": int(valid.sum()),
        "ego_displacement_m": displacement,
        "ego_observed_displacement_m": float(np.linalg.norm(obs_last - obs_first))
        if obs_first is not None and obs_last is not None
        else 0.0,
        "ego_future_displacement_m": float(np.linalg.norm(fut_last - fut_first))
        if fut_first is not None and fut_last is not None
        else 0.0,
        "ego_path_length_m": _path_length(xy[valid]),
        "ego_heading_change_signed_deg": math.degrees(heading_signed),
        "ego_heading_change_deg": abs(math.degrees(heading_signed)),
        "ego_lateral_displacement_m": float(last_xy[1] - first_xy[1])
        if first_xy is not None and last_xy is not None
        else 0.0,
        "ego_longitudinal_displacement_m": float(last_xy[0] - first_xy[0])
        if first_xy is not None and last_xy is not None
        else 0.0,
        "ego_mean_speed_mps": float(valid_speed.mean()) if len(valid_speed) else 0.0,
        "ego_max_speed_mps": float(valid_speed.max()) if len(valid_speed) else 0.0,
        "ego_current_x_m": float(ego[CURRENT_IDX, 0]),
        "ego_current_y_m": float(ego[CURRENT_IDX, 1]),
        "ego_current_yaw_rad": float(ego[CURRENT_IDX, 6]),
        "ego_current_valid": bool(ego[CURRENT_IDX, 5] > 0.5),
    }


def _ego_maneuver_label(metrics: Dict[str, float], thresholds: Dict[str, float]) -> str:
    disp = metrics["ego_displacement_m"]
    path_len = metrics["ego_path_length_m"]
    heading = metrics["ego_heading_change_deg"]
    heading_signed = metrics["ego_heading_change_signed_deg"]
    lateral_abs = abs(metrics["ego_lateral_displacement_m"])
    longitudinal = metrics["ego_longitudinal_displacement_m"]

    if disp < thresholds["static_displacement_m"] or path_len < thresholds["static_path_length_m"]:
        return "stopped_or_slow"
    if heading >= thresholds["large_turn_deg"]:
        return "large_turn_or_uturn"
    if heading >= thresholds["turn_deg"]:
        return "left_turn" if heading_signed > 0 else "right_turn"
    if (
        heading <= thresholds["lane_change_max_heading_deg"]
        and lateral_abs >= thresholds["lane_change_lateral_m"]
        and longitudinal >= thresholds["lane_change_longitudinal_m"]
    ):
        return "lane_change_like"
    if heading <= thresholds["straight_max_heading_deg"] and lateral_abs <= thresholds["straight_max_lateral_m"]:
        return "straight"
    return "curved_or_other_moving"


def _agent_quality_and_counts(agents: np.ndarray, agent_mask: np.ndarray) -> Dict[str, Any]:
    agent_valid = agents[:, :, 5] > 0.5
    active = agent_mask.astype(bool)
    active_valid = agent_valid & active[:, None]
    den = max(1, int(active.sum()) * agents.shape[1])
    invalid_values = agents[:, :, [0, 1, 2, 3, 4, 6]]
    invalid_mask = ~agent_valid
    invalid_nonzero = bool(np.abs(invalid_values[invalid_mask]).max() > 1e-5) if invalid_mask.any() else False

    return {
        "num_valid_agent_slots": int(active.sum()),
        "mean_valid_agents_per_timestep": float(active_valid.sum(axis=0).mean()),
        "agent_valid_ratio": _safe_ratio(float(active_valid.sum()), float(den)),
        "valid_agents_current": int(active_valid[:, CURRENT_IDX].sum()),
        "agent_invalid_values_nonzero": invalid_nonzero,
    }


def _map_metrics(map_polylines: np.ndarray, map_mask: np.ndarray) -> Dict[str, Any]:
    valid_poly = map_mask.astype(bool).sum(axis=1) > 0
    valid_points = int(map_mask.astype(bool).sum())
    total_points = int(np.prod(map_mask.shape))
    invalid_values = map_polylines[:, :, 0:5]
    invalid_mask = ~map_mask.astype(bool)
    invalid_nonzero = bool(np.abs(invalid_values[invalid_mask]).max() > 1e-5) if invalid_mask.any() else False
    return {
        "num_valid_map_polylines": int(valid_poly.sum()),
        "num_valid_map_points": valid_points,
        "map_valid_point_ratio": _safe_ratio(valid_points, total_points),
        "map_saturated": bool(valid_poly.sum() >= map_mask.shape[0]),
        "map_invalid_values_nonzero": invalid_nonzero,
    }


def _light_metrics(lights: np.ndarray, light_mask: np.ndarray) -> Dict[str, Any]:
    valid = light_mask.astype(bool)
    valid_states = lights[:, :, 2][valid].astype(int)
    state_counts = Counter(valid_states.tolist())
    stop_count = sum(state_counts.get(s, 0) for s in (1, 4, 7))
    caution_count = sum(state_counts.get(s, 0) for s in (2, 5, 8))
    go_count = sum(state_counts.get(s, 0) for s in (3, 6))
    # The filter zeros invalid light xy coordinates. It may still preserve a
    # raw state value for masked lights, so state is not treated as a quality
    # failure here.
    invalid_values = lights[:, :, 0:2]
    invalid_mask = ~valid
    invalid_nonzero = bool(np.abs(invalid_values[invalid_mask]).max() > 1e-5) if invalid_mask.any() else False
    return {
        "light_valid_ratio": _safe_ratio(int(valid.sum()), int(valid.size)),
        "mean_valid_lights_per_timestep": float(valid.sum(axis=1).mean()),
        "valid_lights_current": int(valid[CURRENT_IDX].sum()),
        "has_traffic_lights": bool(valid.any()),
        "traffic_light_stop_count": int(stop_count),
        "traffic_light_caution_count": int(caution_count),
        "traffic_light_go_count": int(go_count),
        "light_invalid_values_nonzero": invalid_nonzero,
    }


def _pair_heading_delta(a: float, b: float) -> float:
    return abs(math.degrees(float(_wrap_angle(a - b))))


def _interaction_metrics(agents: np.ndarray, agent_mask: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, Any]:
    ego = agents[0]
    ego_valid = ego[:, 5] > 0.5
    min_dist = float("inf")
    min_dist_obs = float("inf")
    min_dist_current = float("inf")
    close_10 = 0
    close_20 = 0
    crossing_close = 0
    cut_in_merge_like = 0
    current_leading = 0
    current_following = 0

    for k in range(1, agents.shape[0]):
        if not bool(agent_mask[k]):
            continue
        other = agents[k]
        pair_valid = ego_valid & (other[:, 5] > 0.5)
        if not pair_valid.any():
            continue

        rel = other[:, 0:2] - ego[:, 0:2]
        dist = np.linalg.norm(rel[pair_valid], axis=1)
        other_min = float(dist.min())
        min_dist = min(min_dist, other_min)
        if other_min < thresholds["close_agent_10m"]:
            close_10 += 1
        if other_min < thresholds["close_agent_20m"]:
            close_20 += 1

        obs_pair_valid = pair_valid[: CURRENT_IDX + 1]
        if obs_pair_valid.any():
            obs_dist = np.linalg.norm(rel[: CURRENT_IDX + 1][obs_pair_valid], axis=1)
            min_dist_obs = min(min_dist_obs, float(obs_dist.min()))

        if pair_valid[CURRENT_IDX]:
            current_rel = rel[CURRENT_IDX]
            current_dist = float(np.linalg.norm(current_rel))
            min_dist_current = min(min_dist_current, current_dist)
            heading_delta = _pair_heading_delta(ego[CURRENT_IDX, 6], other[CURRENT_IDX, 6])
            if heading_delta <= thresholds["same_direction_deg"] and abs(float(current_rel[1])) <= thresholds["same_lane_lateral_m"]:
                if 0.0 < float(current_rel[0]) <= thresholds["following_leading_x_m"]:
                    current_leading += 1
                if -thresholds["following_leading_x_m"] <= float(current_rel[0]) < 0.0:
                    current_following += 1

        min_idx_all = np.flatnonzero(pair_valid)
        if len(min_idx_all):
            pair_dists_all = np.linalg.norm(rel[min_idx_all], axis=1)
            t_min = int(min_idx_all[int(pair_dists_all.argmin())])
            heading_delta = _pair_heading_delta(ego[t_min, 6], other[t_min, 6])
            if other_min < thresholds["crossing_close_m"] and heading_delta >= thresholds["crossing_heading_deg"]:
                crossing_close += 1

        other_valid_xy = other[pair_valid, 0:2]
        if len(other_valid_xy) >= 2:
            other_lateral = float(other_valid_xy[-1, 1] - other_valid_xy[0, 1])
            other_longitudinal = float(other_valid_xy[-1, 0] - other_valid_xy[0, 0])
            if (
                abs(other_lateral) >= thresholds["cut_in_lateral_m"]
                and other_min <= thresholds["cut_in_close_m"]
                and other_longitudinal > thresholds["cut_in_min_longitudinal_m"]
            ):
                cut_in_merge_like += 1

    return {
        "min_ego_agent_distance_m": min_dist if math.isfinite(min_dist) else -1.0,
        "min_ego_agent_distance_observed_m": min_dist_obs if math.isfinite(min_dist_obs) else -1.0,
        "min_ego_agent_distance_current_m": min_dist_current if math.isfinite(min_dist_current) else -1.0,
        "num_close_agents_10m": int(close_10),
        "num_close_agents_20m": int(close_20),
        "num_crossing_close_agents": int(crossing_close),
        "num_cut_in_merge_like_agents": int(cut_in_merge_like),
        "num_current_leading_agents": int(current_leading),
        "num_current_following_agents": int(current_following),
    }


def _quality_flags(row: Dict[str, Any], item: Dict[str, np.ndarray]) -> List[str]:
    flags: List[str] = []
    flags.extend(_shape_flags(item))
    flags.extend(_finite_flags(item))
    if not row["ego_current_valid"]:
        flags.append("ego_current_invalid")
    if row["ego_valid_steps"] < 80:
        flags.append("ego_low_valid_steps")
    if math.hypot(row["ego_current_x_m"], row["ego_current_y_m"]) > 1e-3:
        flags.append("ego_current_not_at_origin")
    if abs(row["ego_current_yaw_rad"]) > 1e-3:
        flags.append("ego_current_yaw_not_zero")
    if row["num_valid_agent_slots"] <= 1:
        flags.append("no_other_valid_agents")
    if row["agent_invalid_values_nonzero"]:
        flags.append("agent_invalid_values_nonzero")
    if row["num_valid_map_polylines"] == 0:
        flags.append("map_empty")
    if row["map_saturated"]:
        flags.append("map_saturated")
    if row["map_invalid_values_nonzero"]:
        flags.append("map_invalid_values_nonzero")
    if not row["has_traffic_lights"]:
        flags.append("no_valid_traffic_lights")
    if row["light_invalid_values_nonzero"]:
        flags.append("light_invalid_values_nonzero")
    return flags


def _interaction_labels(row: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    if row["num_close_agents_10m"] > 0:
        labels.append("close_agent_10m")
    if row["num_current_leading_agents"] > 0:
        labels.append("leading_agent_current")
    if row["num_current_following_agents"] > 0:
        labels.append("following_agent_current")
    if row["num_crossing_close_agents"] > 0:
        labels.append("crossing_conflict_like")
    if row["num_cut_in_merge_like_agents"] > 0:
        labels.append("cut_in_or_merge_like")
    if row["dense_scene"]:
        labels.append("dense_scene")
    if row["sparse_scene"]:
        labels.append("sparse_scene")
    if row["intersection_like"]:
        labels.append("intersection_like")
    return labels


def analyze_file(path: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
    item = _load_npz(path)
    agents = item["agents"]
    agent_mask = item["agent_mask"].astype(bool)
    map_polylines = item["map_polylines"]
    map_mask = item["map_mask"].astype(bool)
    lights = item["lights"]
    light_mask = item["light_mask"].astype(bool)

    row: Dict[str, Any] = {
        "scenario_id": _as_str(item.get("scenario_id", Path(path).stem)),
        "path": path,
    }
    row.update(_agent_quality_and_counts(agents, agent_mask))
    row.update(_ego_metrics(agents))
    row.update(_map_metrics(map_polylines, map_mask))
    row.update(_light_metrics(lights, light_mask))
    row.update(_interaction_metrics(agents, agent_mask, thresholds))

    row["ego_maneuver_label"] = _ego_maneuver_label(row, thresholds)
    row["has_close_agent_10m"] = row["num_close_agents_10m"] > 0
    row["has_close_agent_20m"] = row["num_close_agents_20m"] > 0
    row["dense_scene"] = row["valid_agents_current"] >= thresholds["dense_scene_current_agents"]
    row["sparse_scene"] = row["valid_agents_current"] <= thresholds["sparse_scene_current_agents"]
    row["intersection_like"] = bool(
        row["has_traffic_lights"]
        and row["light_valid_ratio"] >= thresholds["intersection_light_valid_ratio"]
        or row["num_crossing_close_agents"] > 0
        or row["ego_maneuver_label"] in ("left_turn", "right_turn", "large_turn_or_uturn")
    )

    flags = _quality_flags(row, item)
    labels = _interaction_labels(row)
    row["quality_flags"] = ";".join(flags)
    row["interaction_labels"] = ";".join(labels)
    return row


def _numeric_summary(rows: List[Dict[str, Any]], field: str) -> Dict[str, float]:
    values = np.asarray([_safe_float(row.get(field)) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {}
    return {
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def build_summary(rows: List[Dict[str, Any]], thresholds: Dict[str, float], paths: Dict[str, str]) -> Dict[str, Any]:
    maneuver_counts = Counter(row["ego_maneuver_label"] for row in rows)
    quality_counts: Counter[str] = Counter()
    interaction_counts: Counter[str] = Counter()
    for row in rows:
        quality_counts.update(flag for flag in str(row["quality_flags"]).split(";") if flag)
        interaction_counts.update(label for label in str(row["interaction_labels"]).split(";") if label)

    bool_counts = {}
    for field in (
        "intersection_like",
        "has_close_agent_10m",
        "has_close_agent_20m",
        "dense_scene",
        "sparse_scene",
        "map_saturated",
        "has_traffic_lights",
    ):
        bool_counts[field] = int(sum(bool(row[field]) for row in rows))

    return {
        "dataset_count": len(rows),
        "outputs": paths,
        "thresholds": thresholds,
        "ego_maneuver_counts": dict(sorted(maneuver_counts.items())),
        "boolean_counts": bool_counts,
        "interaction_label_counts": dict(sorted(interaction_counts.items())),
        "quality_flag_counts": dict(sorted(quality_counts.items())),
        "numeric_summary": {field: _numeric_summary(rows, field) for field in NUMERIC_FIELDS},
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outliers(rows: List[Dict[str, Any]], path: Path, top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged = [row for row in rows if row["quality_flags"]]
    closest = sorted(
        [row for row in rows if row["min_ego_agent_distance_m"] >= 0.0],
        key=lambda row: row["min_ego_agent_distance_m"],
    )[:top_k]
    largest_heading = sorted(rows, key=lambda row: row["ego_heading_change_deg"], reverse=True)[:top_k]
    largest_lateral = sorted(rows, key=lambda row: abs(row["ego_lateral_displacement_m"]), reverse=True)[:top_k]

    with path.open("w") as f:
        f.write("# Waymo Vector Dataset Outliers\n\n")
        f.write("## Quality-Flagged Scenarios\n")
        for row in flagged[:top_k]:
            f.write(f"- {row['scenario_id']} flags={row['quality_flags']} path={row['path']}\n")

        f.write("\n## Closest Ego-Agent Interactions\n")
        for row in closest:
            f.write(
                f"- {row['scenario_id']} min_dist={row['min_ego_agent_distance_m']:.2f}m "
                f"labels={row['interaction_labels']} path={row['path']}\n"
            )

        f.write("\n## Largest Ego Heading Changes\n")
        for row in largest_heading:
            f.write(
                f"- {row['scenario_id']} heading={row['ego_heading_change_deg']:.1f}deg "
                f"maneuver={row['ego_maneuver_label']} path={row['path']}\n"
            )

        f.write("\n## Largest Ego Lateral Displacements\n")
        for row in largest_lateral:
            f.write(
                f"- {row['scenario_id']} lateral={row['ego_lateral_displacement_m']:.1f}m "
                f"maneuver={row['ego_maneuver_label']} path={row['path']}\n"
            )


def default_thresholds() -> Dict[str, float]:
    return {
        "static_displacement_m": 2.0,
        "static_path_length_m": 3.0,
        "straight_max_heading_deg": 15.0,
        "straight_max_lateral_m": 2.0,
        "lane_change_max_heading_deg": 30.0,
        "lane_change_lateral_m": 2.5,
        "lane_change_longitudinal_m": 10.0,
        "turn_deg": 45.0,
        "large_turn_deg": 120.0,
        "close_agent_10m": 10.0,
        "close_agent_20m": 20.0,
        "crossing_close_m": 15.0,
        "crossing_heading_deg": 45.0,
        "same_direction_deg": 30.0,
        "same_lane_lateral_m": 4.0,
        "following_leading_x_m": 30.0,
        "cut_in_lateral_m": 2.5,
        "cut_in_close_m": 20.0,
        "cut_in_min_longitudinal_m": 5.0,
        "dense_scene_current_agents": 10.0,
        "sparse_scene_current_agents": 3.0,
        "intersection_light_valid_ratio": 0.05,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit filtered Waymo vector NPZ data.")
    p.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        default=["/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_5k"],
        help="One or more NPZ directories or individual NPZ files.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/p/yufeng/tri30/dreamer4/waymo/reports/vector_dataset_5k_audit",
    )
    p.add_argument("--max_files", type=int, default=0, help="Optional limit for quick debug runs; 0 means all files.")
    p.add_argument("--outlier_top_k", type=int, default=50)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    thresholds = default_thresholds()
    paths = _collect_npz_paths(args.data_dir, max_files=args.max_files if args.max_files > 0 else None)

    rows: List[Dict[str, Any]] = []
    for i, path in enumerate(paths, start=1):
        rows.append(analyze_file(path, thresholds))
        if i == 1 or i % 500 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] analyzed {path}")

    out_dir = Path(args.output_dir)
    csv_path = out_dir / "vector_dataset_scene_labels.csv"
    summary_path = out_dir / "vector_dataset_quality_summary.json"
    outlier_path = out_dir / "vector_dataset_outliers.md"

    write_csv(rows, csv_path)
    write_outliers(rows, outlier_path, top_k=args.outlier_top_k)
    summary = build_summary(
        rows,
        thresholds=thresholds,
        paths={
            "csv": str(csv_path),
            "summary_json": str(summary_path),
            "outliers": str(outlier_path),
        },
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary["ego_maneuver_counts"], indent=2, sort_keys=True))
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote outliers: {outlier_path}")


if __name__ == "__main__":
    main()
