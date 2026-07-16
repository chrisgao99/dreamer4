"""Analyze Waymo Motion objects-of-interest directly from raw TFRecords.

This script scans tf.Example TFRecords without writing tokenizer NPZ files. It
answers:

- how many scenarios contain `state/objects_of_interest`
- how often the SDC/ego track is one of those OOI tracks
- what focus-agent rule would be usable for OOI-centered filtering
- rough heuristic interaction labels among the OOI tracks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
CORE_ROOT = WAYMO_ROOT / "core"
for path in (REPO_ROOT, CORE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from waymo_vector_filter import (
        CURRENT_IDX,
        N_AGENTS_WAYMO,
        N_STEPS,
        _build_agent_arrays,
        _maybe_reshape,
        _temporal_agent_feature,
        iter_tfrecord_examples,
    )
except ModuleNotFoundError:
    from waymo.core.waymo_vector_filter import (
        CURRENT_IDX,
        N_AGENTS_WAYMO,
        N_STEPS,
        _build_agent_arrays,
        _maybe_reshape,
        _temporal_agent_feature,
        iter_tfrecord_examples,
    )


TYPE_NAMES = {
    0: "unknown",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
}

CSV_FIELDS = [
    "scenario_id",
    "tfrecord_path",
    "record_index",
    "has_ooi",
    "num_ooi",
    "num_ooi_current_valid",
    "sdc_in_ooi",
    "sdc_current_valid",
    "num_ooi_vehicle",
    "num_ooi_pedestrian",
    "num_ooi_cyclist",
    "num_ooi_unknown",
    "num_tracks_to_predict",
    "num_ooi_tracks_to_predict",
    "ooi_src_indices",
    "ooi_track_ids",
    "ooi_types",
    "focus_available",
    "focus_src_index",
    "focus_track_id",
    "focus_type",
    "focus_is_sdc",
    "focus_in_tracks_to_predict",
    "focus_rule",
    "num_ooi_pairs",
    "ooi_pair_min_distance_m",
    "ooi_pair_min_distance_observed_m",
    "ooi_pair_heading_delta_at_closest_deg",
    "ooi_group_spread_current_m",
    "ooi_group_spread_min_m",
    "num_close_ooi_pairs_5m",
    "num_close_ooi_pairs_10m",
    "num_crossing_ooi_pairs",
    "num_following_leading_ooi_pairs",
    "num_cut_in_merge_like_ooi_pairs",
    "num_turning_ooi_agents",
    "num_lane_change_like_ooi_agents",
    "num_stopped_or_yield_like_ooi_agents",
    "num_current_agents_near_focus_30m",
    "interaction_labels",
]


def _as_str(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _heading_delta_deg(a: float, b: float) -> float:
    return abs(math.degrees(float(_wrap_angle(a - b))))


def _rotate_offsets(offsets: np.ndarray, heading: float) -> np.ndarray:
    c = math.cos(heading)
    s = math.sin(heading)
    out = np.empty_like(offsets, dtype=np.float32)
    out[..., 0] = c * offsets[..., 0] + s * offsets[..., 1]
    out[..., 1] = -s * offsets[..., 0] + c * offsets[..., 1]
    return out


def _collect_tfrecords(raw_dir: str, max_files: int | None) -> List[str]:
    paths = sorted(str(p) for p in Path(raw_dir).glob("*") if p.is_file())
    if max_files is not None and max_files > 0:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No TFRecord files found under {raw_dir}")
    return paths


def _valid_ooi_indices(agent: Dict[str, np.ndarray]) -> np.ndarray:
    valid_any = agent["valid"].any(axis=1)
    return np.flatnonzero(agent["objects_of_interest"] & valid_any)


def _sdc_index(agent: Dict[str, np.ndarray]) -> int:
    sdc = np.flatnonzero(agent["is_sdc"])
    if len(sdc):
        return int(sdc[0])
    return int(np.argmax(agent["valid"].sum(axis=1)))


def _choose_focus(agent: Dict[str, np.ndarray], ooi_idx: np.ndarray, sdc_idx: int) -> Dict[str, Any]:
    valid_current = agent["valid"][:, CURRENT_IDX]
    ooi_set = set(int(i) for i in ooi_idx.tolist())
    if sdc_idx in ooi_set and bool(valid_current[sdc_idx]):
        focus = sdc_idx
        rule = "sdc_ooi_current_valid"
    else:
        candidates = [int(i) for i in ooi_idx.tolist() if bool(valid_current[i])]
        if not candidates:
            return {
                "focus_available": False,
                "focus_src_index": -1,
                "focus_track_id": -1,
                "focus_type": "none",
                "focus_is_sdc": False,
                "focus_in_tracks_to_predict": False,
                "focus_rule": "no_current_valid_ooi",
            }

        def priority(i: int) -> Tuple[int, int, int]:
            type_rank = {1: 0, 3: 1, 2: 2}.get(int(agent["type"][i]), 3)
            predict_rank = 0 if bool(agent["tracks_to_predict"][i]) else 1
            return (type_rank, predict_rank, i)

        focus = sorted(candidates, key=priority)[0]
        rule = "ooi_vehicle_priority_current_valid"

    return {
        "focus_available": True,
        "focus_src_index": int(focus),
        "focus_track_id": int(agent["id"][focus]),
        "focus_type": TYPE_NAMES.get(int(agent["type"][focus]), f"type_{int(agent['type'][focus])}"),
        "focus_is_sdc": bool(focus == sdc_idx),
        "focus_in_tracks_to_predict": bool(agent["tracks_to_predict"][focus]),
        "focus_rule": rule,
    }


def _agent_displacement_in_initial_frame(agent: Dict[str, np.ndarray], idx: int) -> Tuple[float, float, float]:
    valid = agent["valid"][idx]
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) < 2:
        return 0.0, 0.0, 0.0
    first = int(valid_idx[0])
    last = int(valid_idx[-1])
    xy = np.stack([agent["x"][idx], agent["y"][idx]], axis=-1)
    rel = xy[last] - xy[first]
    heading0 = float(agent["yaw"][idx, first])
    local = _rotate_offsets(rel.reshape(1, 2), heading0)[0]
    heading_delta = _heading_delta_deg(float(agent["yaw"][idx, first]), float(agent["yaw"][idx, last]))
    return float(local[0]), float(local[1]), heading_delta


def _group_spread(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    return float(np.linalg.norm(diff, axis=-1).max())


def _interaction_metrics(agent: Dict[str, np.ndarray], ooi_idx: np.ndarray, focus_idx: int, args) -> Dict[str, Any]:
    min_dist = float("inf")
    min_dist_obs = float("inf")
    heading_at_min = float("nan")
    close_5 = 0
    close_10 = 0
    crossing = 0
    following = 0
    cut_in = 0

    pairs = [(int(ooi_idx[i]), int(ooi_idx[j])) for i in range(len(ooi_idx)) for j in range(i + 1, len(ooi_idx))]
    xy = np.stack([agent["x"], agent["y"]], axis=-1)

    for a, b in pairs:
        pair_valid = agent["valid"][a] & agent["valid"][b]
        if not pair_valid.any():
            continue
        rel = xy[b] - xy[a]
        valid_steps = np.flatnonzero(pair_valid)
        dists = np.linalg.norm(rel[valid_steps], axis=1)
        pair_min_idx = int(valid_steps[int(dists.argmin())])
        pair_min = float(dists.min())
        min_dist = min(min_dist, pair_min)
        heading_delta = _heading_delta_deg(float(agent["yaw"][a, pair_min_idx]), float(agent["yaw"][b, pair_min_idx]))
        if pair_min <= min_dist:
            heading_at_min = heading_delta
        if pair_min <= args.close_distance_5m:
            close_5 += 1
        if pair_min <= args.close_distance_10m:
            close_10 += 1
        if pair_min <= args.crossing_distance_m and heading_delta >= args.crossing_heading_deg:
            crossing += 1

        obs_valid = pair_valid[: CURRENT_IDX + 1]
        if obs_valid.any():
            obs_steps = np.flatnonzero(obs_valid)
            obs_dists = np.linalg.norm(rel[obs_steps], axis=1)
            min_dist_obs = min(min_dist_obs, float(obs_dists.min()))

        if pair_valid[CURRENT_IDX]:
            delta_now = _heading_delta_deg(float(agent["yaw"][a, CURRENT_IDX]), float(agent["yaw"][b, CURRENT_IDX]))
            rel_now_a = _rotate_offsets(rel[CURRENT_IDX].reshape(1, 2), float(agent["yaw"][a, CURRENT_IDX]))[0]
            if (
                delta_now <= args.same_direction_deg
                and abs(float(rel_now_a[1])) <= args.same_lane_lateral_m
                and 0.0 < abs(float(rel_now_a[0])) <= args.following_leading_x_m
            ):
                following += 1

        for src, ref in ((a, b), (b, a)):
            valid = agent["valid"][src]
            if valid.sum() < 2:
                continue
            long_disp, lat_disp, heading_change = _agent_displacement_in_initial_frame(agent, src)
            ref_close = pair_min <= args.cut_in_close_m
            if (
                ref_close
                and heading_change <= args.cut_in_max_heading_change_deg
                and abs(lat_disp) >= args.cut_in_lateral_m
                and long_disp >= args.cut_in_min_longitudinal_m
            ):
                cut_in += 1
                break

    turning = 0
    lane_change = 0
    stopped = 0
    for idx in [int(i) for i in ooi_idx.tolist()]:
        valid = agent["valid"][idx]
        if not valid.any():
            continue
        long_disp, lat_disp, heading_change = _agent_displacement_in_initial_frame(agent, idx)
        speed = agent["speed"][idx][valid]
        if heading_change >= args.turn_heading_deg:
            turning += 1
        if (
            heading_change <= args.lane_change_max_heading_deg
            and abs(lat_disp) >= args.lane_change_lateral_m
            and long_disp >= args.lane_change_longitudinal_m
        ):
            lane_change += 1
        if len(speed) and float(speed.min()) <= args.stopped_speed_mps and float(speed.max()) >= args.moving_speed_mps:
            stopped += 1

    current_points = []
    for idx in [int(i) for i in ooi_idx.tolist()]:
        if bool(agent["valid"][idx, CURRENT_IDX]):
            current_points.append(xy[idx, CURRENT_IDX])
    spread_current = _group_spread(np.asarray(current_points, dtype=np.float32)) if current_points else 0.0

    spread_min = float("inf")
    for t in range(N_STEPS):
        pts = [xy[int(i), t] for i in ooi_idx.tolist() if bool(agent["valid"][int(i), t])]
        if len(pts) >= 2:
            spread_min = min(spread_min, _group_spread(np.asarray(pts, dtype=np.float32)))

    near_focus = 0
    if focus_idx >= 0 and bool(agent["valid"][focus_idx, CURRENT_IDX]):
        focus_xy = xy[focus_idx, CURRENT_IDX]
        current_valid = agent["valid"][:, CURRENT_IDX]
        d = np.linalg.norm(xy[:, CURRENT_IDX] - focus_xy[None, :], axis=1)
        near_focus = int(((d <= args.dense_radius_m) & current_valid).sum())

    labels: List[str] = []
    if close_5 > 0:
        labels.append("close_interaction_5m")
    if close_10 > 0:
        labels.append("close_interaction_10m")
    if following > 0:
        labels.append("following_or_leading")
    if crossing > 0:
        labels.append("crossing_conflict_like")
    if cut_in > 0:
        labels.append("cut_in_or_merge_like")
    if turning > 0:
        labels.append("intersection_turn_or_large_turn")
    if lane_change > 0:
        labels.append("lane_change_like")
    if stopped > 0:
        labels.append("stopped_or_yield_like")
    if near_focus >= args.dense_current_agents:
        labels.append("dense_interaction")

    return {
        "num_ooi_pairs": len(pairs),
        "ooi_pair_min_distance_m": min_dist if math.isfinite(min_dist) else -1.0,
        "ooi_pair_min_distance_observed_m": min_dist_obs if math.isfinite(min_dist_obs) else -1.0,
        "ooi_pair_heading_delta_at_closest_deg": heading_at_min if math.isfinite(heading_at_min) else -1.0,
        "ooi_group_spread_current_m": spread_current,
        "ooi_group_spread_min_m": spread_min if math.isfinite(spread_min) else -1.0,
        "num_close_ooi_pairs_5m": close_5,
        "num_close_ooi_pairs_10m": close_10,
        "num_crossing_ooi_pairs": crossing,
        "num_following_leading_ooi_pairs": following,
        "num_cut_in_merge_like_ooi_pairs": cut_in,
        "num_turning_ooi_agents": turning,
        "num_lane_change_like_ooi_agents": lane_change,
        "num_stopped_or_yield_like_ooi_agents": stopped,
        "num_current_agents_near_focus_30m": near_focus,
        "interaction_labels": ";".join(labels),
    }


def analyze_scenario(data: Dict[str, np.ndarray], tfrecord_path: str, record_index: int, args) -> Dict[str, Any]:
    agent = _build_agent_arrays(data)
    ooi_idx = _valid_ooi_indices(agent)
    sdc_idx = _sdc_index(agent)
    sdc_in_ooi = bool(agent["objects_of_interest"][sdc_idx] and agent["valid"][sdc_idx].any())
    focus = _choose_focus(agent, ooi_idx, sdc_idx) if len(ooi_idx) else {
        "focus_available": False,
        "focus_src_index": -1,
        "focus_track_id": -1,
        "focus_type": "none",
        "focus_is_sdc": False,
        "focus_in_tracks_to_predict": False,
        "focus_rule": "no_ooi",
    }

    ooi_types = [int(agent["type"][int(i)]) for i in ooi_idx.tolist()]
    type_counts = Counter(ooi_types)
    tracks_to_predict = agent["tracks_to_predict"] & agent["valid"].any(axis=1)
    ooi_ttp = int((agent["tracks_to_predict"][ooi_idx]).sum()) if len(ooi_idx) else 0
    interaction = _interaction_metrics(agent, ooi_idx, int(focus["focus_src_index"]), args) if len(ooi_idx) else {
        "num_ooi_pairs": 0,
        "ooi_pair_min_distance_m": -1.0,
        "ooi_pair_min_distance_observed_m": -1.0,
        "ooi_pair_heading_delta_at_closest_deg": -1.0,
        "ooi_group_spread_current_m": 0.0,
        "ooi_group_spread_min_m": -1.0,
        "num_close_ooi_pairs_5m": 0,
        "num_close_ooi_pairs_10m": 0,
        "num_crossing_ooi_pairs": 0,
        "num_following_leading_ooi_pairs": 0,
        "num_cut_in_merge_like_ooi_pairs": 0,
        "num_turning_ooi_agents": 0,
        "num_lane_change_like_ooi_agents": 0,
        "num_stopped_or_yield_like_ooi_agents": 0,
        "num_current_agents_near_focus_30m": 0,
        "interaction_labels": "",
    }

    row: Dict[str, Any] = {
        "scenario_id": _as_str(data.get("scenario/id", "")),
        "tfrecord_path": tfrecord_path,
        "record_index": record_index,
        "has_ooi": bool(len(ooi_idx) > 0),
        "num_ooi": int(len(ooi_idx)),
        "num_ooi_current_valid": int(agent["valid"][ooi_idx, CURRENT_IDX].sum()) if len(ooi_idx) else 0,
        "sdc_in_ooi": sdc_in_ooi,
        "sdc_current_valid": bool(agent["valid"][sdc_idx, CURRENT_IDX]),
        "num_ooi_vehicle": int(type_counts.get(1, 0)),
        "num_ooi_pedestrian": int(type_counts.get(2, 0)),
        "num_ooi_cyclist": int(type_counts.get(3, 0)),
        "num_ooi_unknown": int(sum(v for k, v in type_counts.items() if k not in (1, 2, 3))),
        "num_tracks_to_predict": int(tracks_to_predict.sum()),
        "num_ooi_tracks_to_predict": ooi_ttp,
        "ooi_src_indices": ";".join(str(int(i)) for i in ooi_idx.tolist()),
        "ooi_track_ids": ";".join(str(int(agent["id"][int(i)])) for i in ooi_idx.tolist()),
        "ooi_types": ";".join(TYPE_NAMES.get(t, f"type_{t}") for t in ooi_types),
    }
    row.update(focus)
    row.update(interaction)
    return row


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def build_summary(rows: Sequence[Dict[str, Any]], args: argparse.Namespace, paths: Dict[str, str]) -> Dict[str, Any]:
    total = len(rows)
    has_ooi = [r for r in rows if bool(r["has_ooi"])]
    has_ooi_pair = [r for r in rows if int(r["num_ooi_pairs"]) > 0]
    sdc_in_ooi = [r for r in has_ooi if bool(r["sdc_in_ooi"])]
    focus_available = [r for r in has_ooi if bool(r["focus_available"])]
    label_counts: Counter[str] = Counter()
    for row in has_ooi:
        label_counts.update(label for label in str(row["interaction_labels"]).split(";") if label)

    ooi_count_hist = Counter(str(int(r["num_ooi"])) for r in rows)
    focus_type_counts = Counter(str(r["focus_type"]) for r in focus_available)
    focus_rule_counts = Counter(str(r["focus_rule"]) for r in has_ooi)

    return {
        "total_scenarios": total,
        "num_has_ooi": len(has_ooi),
        "ratio_has_ooi": _ratio(len(has_ooi), total),
        "num_has_ooi_pair": len(has_ooi_pair),
        "ratio_has_ooi_pair": _ratio(len(has_ooi_pair), total),
        "ratio_has_ooi_pair_among_ooi": _ratio(len(has_ooi_pair), len(has_ooi)),
        "total_ooi_pairs": int(sum(int(r["num_ooi_pairs"]) for r in rows)),
        "num_sdc_in_ooi": len(sdc_in_ooi),
        "ratio_sdc_in_ooi_among_ooi": _ratio(len(sdc_in_ooi), len(has_ooi)),
        "num_ooi_without_sdc": len(has_ooi) - len(sdc_in_ooi),
        "ratio_ooi_without_sdc_among_ooi": _ratio(len(has_ooi) - len(sdc_in_ooi), len(has_ooi)),
        "num_focus_available": len(focus_available),
        "ratio_focus_available_among_ooi": _ratio(len(focus_available), len(has_ooi)),
        "num_focus_is_sdc": int(sum(bool(r["focus_is_sdc"]) for r in focus_available)),
        "ratio_focus_is_sdc_among_focus_available": _ratio(
            int(sum(bool(r["focus_is_sdc"]) for r in focus_available)), len(focus_available)
        ),
        "ooi_count_histogram": dict(sorted(ooi_count_hist.items(), key=lambda kv: int(kv[0]))),
        "ooi_type_counts": {
            "vehicle": int(sum(int(r["num_ooi_vehicle"]) for r in rows)),
            "pedestrian": int(sum(int(r["num_ooi_pedestrian"]) for r in rows)),
            "cyclist": int(sum(int(r["num_ooi_cyclist"]) for r in rows)),
            "unknown": int(sum(int(r["num_ooi_unknown"]) for r in rows)),
        },
        "focus_type_counts": dict(sorted(focus_type_counts.items())),
        "focus_rule_counts": dict(sorted(focus_rule_counts.items())),
        "interaction_label_counts": dict(sorted(label_counts.items())),
        "thresholds": {
            key: value
            for key, value in vars(args).items()
            if key.endswith("_m")
            or key.endswith("_deg")
            or key.endswith("_mps")
            or key in ("dense_current_agents",)
        },
        "outputs": paths,
    }


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfrecords = _collect_tfrecords(args.raw_dir, args.max_files)
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for file_idx, tfrecord in enumerate(tfrecords, start=1):
        print(f"[{file_idx}/{len(tfrecords)}] {tfrecord}", flush=True)
        for record_idx, data in enumerate(iter_tfrecord_examples(tfrecord, max_records=args.max_records_per_file)):
            rows.append(analyze_scenario(data, tfrecord, record_idx, args))
            if args.log_every > 0 and len(rows) % args.log_every == 0:
                elapsed = max(1e-6, time.time() - t0)
                print(f"  processed={len(rows)} scenarios ({len(rows) / elapsed:.2f}/s)", flush=True)

    csv_path = out_dir / "waymo_ooi_scenario_stats.csv"
    summary_path = out_dir / "waymo_ooi_summary.json"
    write_csv(rows, csv_path)
    summary = build_summary(rows, args, {"csv": str(csv_path), "summary": str(summary_path)})
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze Waymo raw objects-of-interest statistics.")
    p.add_argument("--raw_dir", type=str, default="/p/liverobotics/waymo_open_dataset_motion/tf_example/training")
    p.add_argument("--output_dir", type=str, default="/p/yufeng/tri30/dreamer4/waymo/evaluation/reports/ooi_raw_stats")
    p.add_argument("--max_files", type=int, default=0, help="0 means all TFRecord files.")
    p.add_argument("--max_records_per_file", type=int, default=None)
    p.add_argument("--log_every", type=int, default=1000)

    p.add_argument("--close_distance_5m", type=float, default=5.0)
    p.add_argument("--close_distance_10m", type=float, default=10.0)
    p.add_argument("--crossing_distance_m", type=float, default=10.0)
    p.add_argument("--crossing_heading_deg", type=float, default=45.0)
    p.add_argument("--same_direction_deg", type=float, default=30.0)
    p.add_argument("--same_lane_lateral_m", type=float, default=4.0)
    p.add_argument("--following_leading_x_m", type=float, default=30.0)
    p.add_argument("--cut_in_close_m", type=float, default=15.0)
    p.add_argument("--cut_in_lateral_m", type=float, default=2.5)
    p.add_argument("--cut_in_min_longitudinal_m", type=float, default=5.0)
    p.add_argument("--cut_in_max_heading_change_deg", type=float, default=35.0)
    p.add_argument("--turn_heading_deg", type=float, default=35.0)
    p.add_argument("--lane_change_max_heading_deg", type=float, default=25.0)
    p.add_argument("--lane_change_lateral_m", type=float, default=2.0)
    p.add_argument("--lane_change_longitudinal_m", type=float, default=8.0)
    p.add_argument("--stopped_speed_mps", type=float, default=1.0)
    p.add_argument("--moving_speed_mps", type=float, default=3.0)
    p.add_argument("--dense_radius_m", type=float, default=30.0)
    p.add_argument("--dense_current_agents", type=int, default=8)
    return p


if __name__ == "__main__":
    main()
