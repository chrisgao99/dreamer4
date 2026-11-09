"""Current-state labels and raw pair features for z[31] relation probes.

The first probe version intentionally uses only labels available at the current
frame.  Future relation labels can be added later without changing the cache
format: add names, targets, and masks to the same arrays.
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


REGRESSION_NAMES: List[str] = [
    "geom_rel_dx",
    "geom_rel_dy",
    "geom_dist",
    "geom_rel_vx",
    "geom_rel_vy",
    "geom_rel_speed",
    "geom_longitudinal_offset",
    "geom_lateral_offset",
    "geom_heading_diff_sin",
    "geom_heading_diff_cos",
    "context_dist_rank_norm",
    "context_front_gap_m",
    "context_rear_gap_m",
    "context_local_density_20m",
]


BINARY_NAMES: List[str] = [
    "geom_is_front",
    "geom_is_behind",
    "geom_is_left",
    "geom_is_right",
    "geom_same_direction",
    "geom_crossing_angle",
    "geom_current_close_5m",
    "geom_current_close_10m",
    "context_nearest_any",
    "context_top3_nearest_any",
    "context_nearest_front",
    "context_nearest_rear",
    "context_left_adjacent",
    "context_right_adjacent",
    "context_same_corridor",
    "context_close_front_gap",
    "context_close_rear_gap",
    "context_in_focus_neighborhood_20m",
]


@dataclass(frozen=True)
class ScenePairLabels:
    candidate_index: np.ndarray
    pair_raw: np.ndarray
    reg_targets: np.ndarray
    reg_masks: np.ndarray
    bin_targets: np.ndarray
    bin_masks: np.ndarray


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


def _valid_current(agents_ktf: np.ndarray, agent_mask: np.ndarray, query_step: int) -> np.ndarray:
    return agent_mask.astype(bool) & (agents_ktf[:, query_step, 5] > 0.5)


def _rank_norm(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    denom = max(1, len(values) - 1)
    return ranks / float(denom)


def build_scene_pair_labels(
    agents: np.ndarray,
    agent_mask: np.ndarray,
    *,
    query_step: int = 31,
    focus_index: int = 0,
) -> ScenePairLabels:
    """Build valid focus-candidate current-state pair samples for one scene."""
    agents_ktf = _agent_time_layout(agents, agent_mask)
    k, t, f = agents_ktf.shape
    if f < 8:
        raise ValueError(f"Expected at least 8 agent features, got {f}")
    if query_step < 0 or query_step >= t:
        raise ValueError(f"query_step={query_step} outside scene with T={t}")
    if focus_index < 0 or focus_index >= k:
        raise ValueError(f"focus_index={focus_index} outside K={k}")

    cur = agents_ktf[:, query_step].astype(np.float32)
    valid = _valid_current(agents_ktf, agent_mask, query_step)
    if not valid[focus_index]:
        empty = np.zeros((0,), dtype=np.int64)
        return ScenePairLabels(
            candidate_index=empty,
            pair_raw=np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            reg_targets=np.zeros((0, len(REGRESSION_NAMES)), dtype=np.float32),
            reg_masks=np.zeros((0, len(REGRESSION_NAMES)), dtype=np.float32),
            bin_targets=np.zeros((0, len(BINARY_NAMES)), dtype=np.float32),
            bin_masks=np.zeros((0, len(BINARY_NAMES)), dtype=np.float32),
        )

    focus = cur[focus_index]
    fx, fy, fspeed, fvx, fvy, _, fyaw, ftype = focus[:8]
    fcos = float(np.cos(fyaw))
    fsin = float(np.sin(fyaw))

    all_dx = cur[:, 0] - fx
    all_dy = cur[:, 1] - fy
    all_dist = _safe_norm2(all_dx, all_dy)
    all_heading_diff = _wrap_angle(cur[:, 6] - fyaw)
    all_same_dir = np.cos(all_heading_diff) > np.cos(np.deg2rad(45.0))
    all_long = all_dx * fcos + all_dy * fsin
    all_lat = -all_dx * fsin + all_dy * fcos

    candidate_mask = valid.copy()
    candidate_mask[focus_index] = False
    candidate_indices = np.nonzero(candidate_mask)[0]
    if candidate_indices.size == 0:
        return ScenePairLabels(
            candidate_index=np.zeros((0,), dtype=np.int64),
            pair_raw=np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32),
            reg_targets=np.zeros((0, len(REGRESSION_NAMES)), dtype=np.float32),
            reg_masks=np.zeros((0, len(REGRESSION_NAMES)), dtype=np.float32),
            bin_targets=np.zeros((0, len(BINARY_NAMES)), dtype=np.float32),
            bin_masks=np.zeros((0, len(BINARY_NAMES)), dtype=np.float32),
        )

    cand_dist = all_dist[candidate_indices]
    dist_rank = _rank_norm(cand_dist)
    top3_order = candidate_indices[np.argsort(cand_dist, kind="stable")[: min(3, len(candidate_indices))]]
    nearest_any = int(candidate_indices[np.argmin(cand_dist)])

    same_corridor_all = valid & (np.arange(k) != focus_index) & all_same_dir & (np.abs(all_lat) < 4.5)
    front_pool = same_corridor_all & (all_long > 0.0)
    rear_pool = same_corridor_all & (all_long < 0.0)
    left_pool = valid & (np.arange(k) != focus_index) & all_same_dir & (all_lat > 2.0) & (all_lat < 7.5) & (np.abs(all_long) < 20.0)
    right_pool = valid & (np.arange(k) != focus_index) & all_same_dir & (all_lat < -2.0) & (all_lat > -7.5) & (np.abs(all_long) < 20.0)

    nearest_front = -1
    nearest_rear = -1
    nearest_left = -1
    nearest_right = -1
    front_gap = 80.0
    rear_gap = 80.0
    if front_pool.any():
        front_candidates = np.nonzero(front_pool)[0]
        nearest_front = int(front_candidates[np.argmin(all_long[front_candidates])])
        front_gap = float(all_long[nearest_front])
    if rear_pool.any():
        rear_candidates = np.nonzero(rear_pool)[0]
        nearest_rear = int(rear_candidates[np.argmax(all_long[rear_candidates])])
        rear_gap = float(-all_long[nearest_rear])
    if left_pool.any():
        left_candidates = np.nonzero(left_pool)[0]
        nearest_left = int(left_candidates[np.argmin(np.abs(all_lat[left_candidates]))])
    if right_pool.any():
        right_candidates = np.nonzero(right_pool)[0]
        nearest_right = int(right_candidates[np.argmin(np.abs(all_lat[right_candidates]))])

    local_density_20m = float(((candidate_mask) & (all_dist <= 20.0)).sum())

    rows_raw: list[list[float]] = []
    rows_reg: list[list[float]] = []
    rows_bin: list[list[float]] = []

    for row_idx, j in enumerate(candidate_indices):
        cand = cur[j]
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

        rows_raw.append(
            [
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
        )
        rows_reg.append(
            [
                dx,
                dy,
                dist,
                rel_vx,
                rel_vy,
                rel_speed,
                longitudinal,
                lateral,
                float(np.sin(heading_diff)),
                float(np.cos(heading_diff)),
                float(dist_rank[row_idx]),
                front_gap,
                rear_gap,
                local_density_20m,
            ]
        )
        rows_bin.append(
            [
                float(longitudinal > 0.0),
                float(longitudinal < 0.0),
                float(lateral > 0.0),
                float(lateral < 0.0),
                same_direction,
                crossing_angle,
                float(dist <= 5.0),
                float(dist <= 10.0),
                float(j == nearest_any),
                float(j in set(top3_order.tolist())),
                float(j == nearest_front),
                float(j == nearest_rear),
                float(j == nearest_left),
                float(j == nearest_right),
                float(same_direction and abs(lateral) < 4.5),
                float(j == nearest_front and front_gap <= 15.0),
                float(j == nearest_rear and rear_gap <= 15.0),
                float(dist <= 20.0),
            ]
        )

    pair_raw = np.asarray(rows_raw, dtype=np.float32)
    reg_targets = np.asarray(rows_reg, dtype=np.float32)
    bin_targets = np.asarray(rows_bin, dtype=np.float32)
    return ScenePairLabels(
        candidate_index=candidate_indices.astype(np.int64),
        pair_raw=pair_raw,
        reg_targets=reg_targets,
        reg_masks=np.ones_like(reg_targets, dtype=np.float32),
        bin_targets=bin_targets,
        bin_masks=np.ones_like(bin_targets, dtype=np.float32),
    )


def label_metadata() -> Dict[str, List[str]]:
    return {
        "feature_names": FEATURE_NAMES,
        "regression_names": REGRESSION_NAMES,
        "binary_names": BINARY_NAMES,
    }

