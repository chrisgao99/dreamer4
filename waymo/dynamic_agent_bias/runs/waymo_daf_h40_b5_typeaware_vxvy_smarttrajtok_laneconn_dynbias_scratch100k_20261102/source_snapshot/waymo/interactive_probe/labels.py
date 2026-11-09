"""Future-interaction labels for pair probes.

The probe inputs are current-state pair features plus optional z[query_step].
Labels may use future ground-truth trajectories so the probe asks whether the
representation contains interaction foresight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


FEATURE_NAMES: List[str] = [
    "focus_x",
    "focus_y",
    "focus_speed",
    "focus_vx",
    "focus_vy",
    "focus_sin_yaw",
    "focus_cos_yaw",
    "focus_type",
    "cand_x",
    "cand_y",
    "cand_speed",
    "cand_vx",
    "cand_vy",
    "cand_sin_yaw",
    "cand_cos_yaw",
    "cand_type",
    "rel_dx",
    "rel_dy",
    "rel_dist",
    "rel_vx",
    "rel_vy",
    "rel_speed",
    "bearing_sin",
    "bearing_cos",
    "heading_diff_sin",
    "heading_diff_cos",
    "longitudinal_offset",
    "lateral_offset",
    "abs_lateral_offset",
    "closing_speed",
    "same_direction_proxy",
    "crossing_angle_proxy",
    "current_close_5m",
    "current_close_10m",
]

TYPE_NAMES: List[str] = [
    "other_leads_focus",
    "other_follows_focus",
    "crossing_or_oncoming_conflict",
    "converging_conflict",
]

RESPONSE_BINARY_NAMES: List[str] = [
    "focus_goes_first",
    "focus_yields_to_other",
    "focus_decelerates_for_interaction",
]

RESPONSE_REGRESSION_NAMES: List[str] = [
    "delta_arrival_time_s",
]

DIAGNOSTIC_NAMES: List[str] = [
    "future_min_time_aligned_dist_m",
    "future_min_spatial_dist_m",
    "pet_s",
    "future_min_abs_lateral_m",
    "future_min_abs_longitudinal_m",
    "focus_speed_drop_mps",
    "focus_max_decel_mps2",
]


@dataclass(frozen=True)
class InteractiveLabelConfig:
    dt: float = 0.1
    future_steps: int = 50
    relevance_dist_m: float = 8.0
    path_overlap_dist_m: float = 4.0
    pet_relevant_s: float = 3.0
    same_direction_deg: float = 45.0
    crossing_heading_deg: float = 60.0
    oncoming_heading_deg: float = 135.0
    same_corridor_lateral_m: float = 4.5
    following_headway_m: float = 30.0
    following_relevant_headway_m: float = 20.0
    converging_current_lateral_m: float = 2.0
    converging_future_lateral_m: float = 3.5
    priority_pet_s: float = 4.0
    priority_time_margin_s: float = 0.2
    yield_time_margin_s: float = 0.5
    speed_drop_mps: float = 1.5
    decel_mps2: float = 1.0


@dataclass(frozen=True)
class InteractivePairLabels:
    candidate_index: np.ndarray
    pair_raw: np.ndarray
    relevance_targets: np.ndarray
    relevance_masks: np.ndarray
    type_targets: np.ndarray
    type_masks: np.ndarray
    response_bin_targets: np.ndarray
    response_bin_masks: np.ndarray
    response_reg_targets: np.ndarray
    response_reg_masks: np.ndarray
    diagnostics: np.ndarray


def _wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _safe_norm2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(x * x + y * y, 0.0))


def _agent_time_layout(agents: np.ndarray, agent_mask: np.ndarray) -> np.ndarray:
    """Return agents as (K,T,F)."""
    if agents.ndim != 3:
        raise ValueError(f"Expected agents with shape (K,T,F) or (T,K,F), got {agents.shape}")
    k = int(agent_mask.shape[0])
    if agents.shape[0] == k:
        return agents
    if agents.shape[1] == k:
        return np.transpose(agents, (1, 0, 2))
    raise ValueError(f"Could not infer agent/time layout from agents={agents.shape}, mask={agent_mask.shape}")


def _heading_delta_deg(a: float, b: float) -> float:
    return abs(float(_wrap_angle(a - b))) * 180.0 / float(np.pi)


def _empty() -> InteractivePairLabels:
    return InteractivePairLabels(
        candidate_index=np.zeros((0,), dtype=np.int64),
        pair_raw=np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
        relevance_targets=np.zeros((0, 1), dtype=np.float32),
        relevance_masks=np.zeros((0, 1), dtype=np.float32),
        type_targets=np.zeros((0,), dtype=np.int64),
        type_masks=np.zeros((0,), dtype=np.float32),
        response_bin_targets=np.zeros((0, len(RESPONSE_BINARY_NAMES)), dtype=np.float32),
        response_bin_masks=np.zeros((0, len(RESPONSE_BINARY_NAMES)), dtype=np.float32),
        response_reg_targets=np.zeros((0, len(RESPONSE_REGRESSION_NAMES)), dtype=np.float32),
        response_reg_masks=np.zeros((0, len(RESPONSE_REGRESSION_NAMES)), dtype=np.float32),
        diagnostics=np.zeros((0, len(DIAGNOSTIC_NAMES)), dtype=np.float32),
    )


def _closest_spatial_pair(f_xy: np.ndarray, o_xy: np.ndarray) -> tuple[float, int, int]:
    diff = f_xy[:, None, :] - o_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    flat_idx = int(np.argmin(dist))
    i, j = np.unravel_index(flat_idx, dist.shape)
    return float(dist[i, j]), int(i), int(j)


def _pair_raw_from_current(focus: np.ndarray, cand: np.ndarray) -> list[float]:
    fx, fy, fspeed, fvx, fvy, _, fyaw, ftype = focus[:8]
    fcos = float(np.cos(fyaw))
    fsin = float(np.sin(fyaw))
    dx = float(cand[0] - fx)
    dy = float(cand[1] - fy)
    dist = float(np.sqrt(max(dx * dx + dy * dy, 0.0)))
    rel_vx = float(cand[3] - fvx)
    rel_vy = float(cand[4] - fvy)
    rel_speed = float(np.sqrt(max(rel_vx * rel_vx + rel_vy * rel_vy, 0.0)))
    bearing = float(np.arctan2(dy, dx))
    heading_diff = float(_wrap_angle(cand[6] - fyaw))
    longitudinal = float(dx * fcos + dy * fsin)
    lateral = float(-dx * fsin + dy * fcos)
    closing_speed = 0.0
    if dist > 1e-4:
        closing_speed = -float((dx * rel_vx + dy * rel_vy) / dist)
    same_direction = float(np.cos(heading_diff) > np.cos(np.deg2rad(45.0)))
    crossing_angle = float(abs(np.sin(heading_diff)) > np.sin(np.deg2rad(45.0)))
    return [
        float(fx),
        float(fy),
        float(fspeed),
        float(fvx),
        float(fvy),
        float(np.sin(fyaw)),
        float(np.cos(fyaw)),
        float(ftype),
        float(cand[0]),
        float(cand[1]),
        float(cand[2]),
        float(cand[3]),
        float(cand[4]),
        float(np.sin(cand[6])),
        float(np.cos(cand[6])),
        float(cand[7]),
        dx,
        dy,
        dist,
        rel_vx,
        rel_vy,
        rel_speed,
        float(np.sin(bearing)),
        float(np.cos(bearing)),
        float(np.sin(heading_diff)),
        float(np.cos(heading_diff)),
        longitudinal,
        lateral,
        abs(lateral),
        closing_speed,
        same_direction,
        crossing_angle,
        float(dist <= 5.0),
        float(dist <= 10.0),
    ]


def _long_lat(dx: np.ndarray, dy: np.ndarray, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    longitudinal = dx * cos_yaw + dy * sin_yaw
    lateral = -dx * sin_yaw + dy * cos_yaw
    return longitudinal, lateral


def _focus_response(focus: np.ndarray, start: int, end: int, cfg: InteractiveLabelConfig) -> tuple[float, float, float]:
    valid = focus[start:end, 5] > 0.5
    if not valid.any():
        return 0.0, 0.0, 0.0
    speeds = focus[start:end, 2].astype(np.float32)[valid]
    current_speed = float(focus[start, 2])
    speed_drop = max(0.0, current_speed - float(np.min(speeds)))
    max_decel = 0.0
    if len(speeds) >= 2:
        accel = np.diff(speeds) / float(cfg.dt)
        max_decel = max(0.0, -float(np.min(accel)))
    decel_flag = float(speed_drop >= cfg.speed_drop_mps and max_decel >= cfg.decel_mps2)
    return decel_flag, speed_drop, max_decel


def build_scene_interactive_labels(
    agents: np.ndarray,
    agent_mask: np.ndarray,
    *,
    query_step: int = 31,
    focus_index: int = 0,
    cfg: InteractiveLabelConfig | None = None,
) -> InteractivePairLabels:
    """Build future-interaction labels for valid current focus-candidate pairs."""
    cfg = cfg or InteractiveLabelConfig()
    agents_ktf = _agent_time_layout(agents, agent_mask).astype(np.float32)
    k, t, f = agents_ktf.shape
    if f < 8:
        raise ValueError(f"Expected at least 8 agent features, got {f}")
    if query_step < 0 or query_step >= t:
        raise ValueError(f"query_step={query_step} outside scene with T={t}")
    if focus_index < 0 or focus_index >= k:
        raise ValueError(f"focus_index={focus_index} outside K={k}")

    current_valid = agent_mask.astype(bool) & (agents_ktf[:, query_step, 5] > 0.5)
    if not current_valid[focus_index]:
        return _empty()

    candidate_mask = current_valid.copy()
    candidate_mask[focus_index] = False
    candidate_indices = np.nonzero(candidate_mask)[0]
    if candidate_indices.size == 0:
        return _empty()

    start = int(query_step)
    end = min(t, int(query_step + cfg.future_steps + 1))
    focus = agents_ktf[focus_index]
    focus_cur = focus[query_step]
    focus_decel_flag, focus_speed_drop, focus_max_decel = _focus_response(focus, start, end, cfg)

    rows_raw: list[list[float]] = []
    rows_rel: list[list[float]] = []
    rows_rel_mask: list[list[float]] = []
    rows_type: list[int] = []
    rows_type_mask: list[float] = []
    rows_resp_bin: list[list[float]] = []
    rows_resp_bin_mask: list[list[float]] = []
    rows_resp_reg: list[list[float]] = []
    rows_resp_reg_mask: list[list[float]] = []
    rows_diag: list[list[float]] = []

    type_index = {name: idx for idx, name in enumerate(TYPE_NAMES)}

    for j in candidate_indices:
        other = agents_ktf[j]
        other_cur = other[query_step]
        rows_raw.append(_pair_raw_from_current(focus_cur, other_cur))

        pair_valid = (focus[start:end, 5] > 0.5) & (other[start:end, 5] > 0.5)
        if not pair_valid.any():
            rows_rel.append([0.0])
            rows_rel_mask.append([1.0])
            rows_type.append(0)
            rows_type_mask.append(0.0)
            rows_resp_bin.append([0.0] * len(RESPONSE_BINARY_NAMES))
            rows_resp_bin_mask.append([0.0] * len(RESPONSE_BINARY_NAMES))
            rows_resp_reg.append([0.0] * len(RESPONSE_REGRESSION_NAMES))
            rows_resp_reg_mask.append([0.0] * len(RESPONSE_REGRESSION_NAMES))
            rows_diag.append([80.0, 80.0, 99.0, 80.0, 80.0, focus_speed_drop, focus_max_decel])
            continue

        valid_offsets = np.flatnonzero(pair_valid)
        abs_steps = start + valid_offsets
        f_seq = focus[abs_steps]
        o_seq = other[abs_steps]
        rel_xy = o_seq[:, 0:2] - f_seq[:, 0:2]
        time_dists = np.linalg.norm(rel_xy, axis=1)
        min_time_idx = int(np.argmin(time_dists))
        min_time_dist = float(time_dists[min_time_idx])

        longitudinal, lateral = _long_lat(rel_xy[:, 0], rel_xy[:, 1], f_seq[:, 6])
        heading_delta = np.asarray([_heading_delta_deg(float(a), float(b)) for a, b in zip(f_seq[:, 6], o_seq[:, 6])])
        same_dir = heading_delta <= cfg.same_direction_deg
        same_corridor = same_dir & (np.abs(lateral) <= cfg.same_corridor_lateral_m)
        future_lead = same_corridor & (longitudinal > 0.0) & (longitudinal <= cfg.following_headway_m)
        future_rear = same_corridor & (longitudinal < 0.0) & (-longitudinal <= cfg.following_headway_m)
        lead_relevant = bool(np.any(future_lead & (longitudinal <= cfg.following_relevant_headway_m)))
        rear_relevant = bool(np.any(future_rear & (-longitudinal <= cfg.following_relevant_headway_m)))

        spatial_min_dist, focus_spatial_idx, other_spatial_idx = _closest_spatial_pair(f_seq[:, 0:2], o_seq[:, 0:2])
        pet = abs(float(focus_spatial_idx - other_spatial_idx)) * float(cfg.dt)
        conflict_heading = _heading_delta_deg(float(f_seq[focus_spatial_idx, 6]), float(o_seq[other_spatial_idx, 6]))
        conflict_long = float(longitudinal[min(min_time_idx, len(longitudinal) - 1)])
        conflict_lat = float(lateral[min(min_time_idx, len(lateral) - 1)])

        path_overlap = spatial_min_dist <= cfg.path_overlap_dist_m and pet <= cfg.pet_relevant_s
        crossing = path_overlap and conflict_heading >= cfg.crossing_heading_deg
        current_dx = float(other_cur[0] - focus_cur[0])
        current_dy = float(other_cur[1] - focus_cur[1])
        current_long, current_lat = _long_lat(
            np.asarray([current_dx], dtype=np.float32),
            np.asarray([current_dy], dtype=np.float32),
            np.asarray([focus_cur[6]], dtype=np.float32),
        )
        current_heading = _heading_delta_deg(float(focus_cur[6]), float(other_cur[6]))
        current_same_corridor = current_heading <= cfg.same_direction_deg and abs(float(current_lat[0])) <= cfg.same_corridor_lateral_m
        future_same_close = bool(np.any(same_corridor & (np.abs(lateral) <= cfg.converging_future_lateral_m) & (time_dists <= cfg.relevance_dist_m)))
        converging = (
            future_same_close
            and (path_overlap or min_time_dist <= cfg.relevance_dist_m)
            and conflict_heading < cfg.crossing_heading_deg
            and (
                not current_same_corridor
                or abs(float(current_lat[0])) >= cfg.converging_current_lateral_m
            )
        )

        relevant = bool(min_time_dist <= cfg.relevance_dist_m or path_overlap or lead_relevant or rear_relevant)
        rows_rel.append([float(relevant)])
        rows_rel_mask.append([1.0])

        type_target = 0
        type_mask = 0.0
        if relevant:
            if crossing:
                type_target = type_index["crossing_or_oncoming_conflict"]
                type_mask = 1.0
            elif converging:
                type_target = type_index["converging_conflict"]
                type_mask = 1.0
            elif lead_relevant:
                type_target = type_index["other_leads_focus"]
                type_mask = 1.0
            elif rear_relevant:
                type_target = type_index["other_follows_focus"]
                type_mask = 1.0
        rows_type.append(type_target)
        rows_type_mask.append(type_mask)

        delta_arrival = (float(focus_spatial_idx) - float(other_spatial_idx)) * float(cfg.dt)
        priority_mask = float(
            type_mask > 0.5
            and TYPE_NAMES[type_target] in {"crossing_or_oncoming_conflict", "converging_conflict"}
            and spatial_min_dist <= cfg.path_overlap_dist_m
            and pet <= cfg.priority_pet_s
        )
        decel_mask = float(
            type_mask > 0.5
            and TYPE_NAMES[type_target] in {"other_leads_focus", "crossing_or_oncoming_conflict", "converging_conflict"}
        )

        focus_goes_first = float(delta_arrival <= -cfg.priority_time_margin_s)
        focus_yields = float(delta_arrival >= cfg.yield_time_margin_s and focus_decel_flag > 0.5)
        rows_resp_bin.append([focus_goes_first, focus_yields, focus_decel_flag])
        rows_resp_bin_mask.append([priority_mask, priority_mask, decel_mask])
        rows_resp_reg.append([delta_arrival])
        rows_resp_reg_mask.append([priority_mask])
        rows_diag.append(
            [
                min_time_dist,
                spatial_min_dist,
                pet,
                float(np.min(np.abs(lateral))),
                float(np.min(np.abs(longitudinal))),
                focus_speed_drop,
                focus_max_decel,
            ]
        )

    return InteractivePairLabels(
        candidate_index=candidate_indices.astype(np.int64),
        pair_raw=np.asarray(rows_raw, dtype=np.float32),
        relevance_targets=np.asarray(rows_rel, dtype=np.float32),
        relevance_masks=np.asarray(rows_rel_mask, dtype=np.float32),
        type_targets=np.asarray(rows_type, dtype=np.int64),
        type_masks=np.asarray(rows_type_mask, dtype=np.float32),
        response_bin_targets=np.asarray(rows_resp_bin, dtype=np.float32),
        response_bin_masks=np.asarray(rows_resp_bin_mask, dtype=np.float32),
        response_reg_targets=np.asarray(rows_resp_reg, dtype=np.float32),
        response_reg_masks=np.asarray(rows_resp_reg_mask, dtype=np.float32),
        diagnostics=np.asarray(rows_diag, dtype=np.float32),
    )


def label_metadata() -> Dict[str, List[str]]:
    return {
        "feature_names": FEATURE_NAMES,
        "type_names": TYPE_NAMES,
        "response_binary_names": RESPONSE_BINARY_NAMES,
        "response_regression_names": RESPONSE_REGRESSION_NAMES,
        "diagnostic_names": DIAGNOSTIC_NAMES,
    }
