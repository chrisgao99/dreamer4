"""Detect pair interaction events and extract causal query-time-frame histories.

This module deliberately separates future-derived supervision from matching
features.  Event and response labels may inspect the future offline, while
``history`` contains only states at or before ``query_step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


RELATION_NAMES = (
    "other_leads_focus",
    "other_follows_focus",
    "crossing_or_oncoming_conflict",
    "converging_conflict",
)
RESPONSE_NAMES = (
    "goes_first",
    "yields",
    "decelerates",
    "maintains",
)
HISTORY_FEATURE_NAMES = (
    "rel_longitudinal_m",
    "rel_lateral_m",
    "rel_v_longitudinal_mps",
    "rel_v_lateral_mps",
    "heading_diff_sin",
    "heading_diff_cos",
    "focus_speed_mps",
    "focus_accel_mps2",
    "focus_yaw_rate_rps",
    "other_speed_mps",
    "other_accel_mps2",
)


@dataclass(frozen=True)
class SampleConfig:
    dt: float = 0.1
    event_search_start: int = 10
    history_steps: int = 20
    lead_steps: tuple[int, ...] = (10, 20, 30)
    path_overlap_dist_m: float = 4.0
    pet_relevant_s: float = 3.0
    crossing_heading_deg: float = 60.0
    same_direction_deg: float = 45.0
    same_corridor_lateral_m: float = 4.5
    following_headway_m: float = 20.0
    speed_drop_mps: float = 1.5
    decel_mps2: float = 1.0
    decel_lookahead_steps: int = 20
    attribution_before_steps: int = 5
    attribution_after_steps: int = 20
    attribution_ambiguity_steps: int = 3
    priority_time_margin_s: float = 0.2
    yield_time_margin_s: float = 0.5
    maintain_max_speed_drop_mps: float = 0.5


@dataclass(frozen=True)
class PairEvent:
    candidate_index: int
    event_step: int
    relation_index: int
    focus_arrival_step: int
    other_arrival_step: int
    spatial_min_dist_m: float
    pet_s: float
    attributed_decel: bool = False
    attribution_ambiguous: bool = False


@dataclass(frozen=True)
class PairSample:
    candidate_index: int
    event_step: int
    query_step: int
    lead_steps: int
    relation_index: int
    response_index: int
    eligible: bool
    focus_type: int
    candidate_type: int
    delta_arrival_time_s: float
    pet_s: float
    spatial_min_dist_m: float
    history: np.ndarray


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def ensure_agent_time_layout(agents: np.ndarray, agent_mask: np.ndarray) -> np.ndarray:
    """Return agents with shape (K,T,F)."""
    if agents.ndim != 3:
        raise ValueError(f"Expected rank-3 agents, got {agents.shape}")
    k = int(agent_mask.shape[0])
    if agents.shape[0] == k:
        return agents
    if agents.shape[1] == k:
        return np.transpose(agents, (1, 0, 2))
    raise ValueError(f"Cannot infer agent/time layout from {agents.shape} and mask {agent_mask.shape}")


def _heading_delta_deg(a: float, b: float) -> float:
    return abs(float(wrap_angle(a - b))) * 180.0 / float(np.pi)


def _rotate_xy(xy: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate world/pre-normalized vectors into the frame with heading ``yaw``."""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    return np.asarray(xy, dtype=np.float32) @ rotation.T


def _aligned_long_lat(focus: np.ndarray, other: np.ndarray, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = other[steps, 0:2] - focus[steps, 0:2]
    yaw = focus[steps, 6]
    c = np.cos(yaw)
    s = np.sin(yaw)
    longitudinal = rel[:, 0] * c + rel[:, 1] * s
    lateral = -rel[:, 0] * s + rel[:, 1] * c
    return longitudinal, lateral


def detect_pair_event(
    focus: np.ndarray,
    other: np.ndarray,
    *,
    candidate_index: int,
    cfg: SampleConfig,
) -> PairEvent | None:
    """Detect one geometry-defined event for a focus/other pair.

    Conflict events use closest points on the two future paths.  Following
    events use the first aligned time at which the pair enters a same-corridor
    headway threshold.  No response label is used to choose the event.
    """
    t = int(focus.shape[0])
    jointly_valid = (focus[:, 5] > 0.5) & (other[:, 5] > 0.5)
    valid_steps = np.flatnonzero(jointly_valid & (np.arange(t) >= int(cfg.event_search_start)))
    if valid_steps.size < 2:
        return None

    f_xy = focus[valid_steps, 0:2]
    o_xy = other[valid_steps, 0:2]
    distances = np.linalg.norm(f_xy[:, None, :] - o_xy[None, :, :], axis=-1)
    flat = int(np.argmin(distances))
    f_local, o_local = np.unravel_index(flat, distances.shape)
    f_step = int(valid_steps[f_local])
    o_step = int(valid_steps[o_local])
    spatial_min = float(distances[f_local, o_local])
    pet_s = abs(f_step - o_step) * float(cfg.dt)
    conflict_heading = _heading_delta_deg(float(focus[f_step, 6]), float(other[o_step, 6]))

    if spatial_min <= cfg.path_overlap_dist_m and pet_s <= cfg.pet_relevant_s:
        event_step = min(f_step, o_step)
        if conflict_heading >= cfg.crossing_heading_deg:
            relation = RELATION_NAMES.index("crossing_or_oncoming_conflict")
        else:
            # Same-heading paths may be following or merging.  Classify them
            # using the time-aligned geometry immediately before the event.
            aligned_step = int(np.clip(event_step, valid_steps[0], valid_steps[-1]))
            rel_xy = other[aligned_step, 0:2] - focus[aligned_step, 0:2]
            long_lat = _rotate_xy(rel_xy[None], float(focus[aligned_step, 6]))[0]
            aligned_heading = _heading_delta_deg(float(focus[aligned_step, 6]), float(other[aligned_step, 6]))
            same_corridor = (
                aligned_heading <= cfg.same_direction_deg
                and abs(float(long_lat[1])) <= cfg.same_corridor_lateral_m
            )
            if same_corridor and abs(float(long_lat[0])) <= cfg.following_headway_m:
                relation = RELATION_NAMES.index(
                    "other_leads_focus" if float(long_lat[0]) > 0.0 else "other_follows_focus"
                )
            else:
                relation = RELATION_NAMES.index("converging_conflict")
        return PairEvent(
            candidate_index=int(candidate_index),
            event_step=event_step,
            relation_index=relation,
            focus_arrival_step=f_step,
            other_arrival_step=o_step,
            spatial_min_dist_m=spatial_min,
            pet_s=pet_s,
        )

    # No path conflict: retain a close same-corridor following event.
    longitudinal, lateral = _aligned_long_lat(focus, other, valid_steps)
    heading = np.asarray(
        [_heading_delta_deg(float(focus[s, 6]), float(other[s, 6])) for s in valid_steps], dtype=np.float32
    )
    following = (
        (heading <= cfg.same_direction_deg)
        & (np.abs(lateral) <= cfg.same_corridor_lateral_m)
        & (np.abs(longitudinal) <= cfg.following_headway_m)
    )
    if not bool(following.any()):
        return None
    first = int(np.flatnonzero(following)[0])
    event_step = int(valid_steps[first])
    relation = RELATION_NAMES.index(
        "other_leads_focus" if float(longitudinal[first]) > 0.0 else "other_follows_focus"
    )
    return PairEvent(
        candidate_index=int(candidate_index),
        event_step=event_step,
        relation_index=relation,
        focus_arrival_step=event_step,
        other_arrival_step=event_step,
        spatial_min_dist_m=spatial_min,
        pet_s=pet_s,
    )


def detect_deceleration_onsets(focus: np.ndarray, cfg: SampleConfig) -> list[int]:
    """Return high-confidence focus deceleration episode onsets."""
    valid = focus[:, 5] > 0.5
    speed = focus[:, 2].astype(np.float32)
    onsets: list[int] = []
    last_kept = -10_000
    for step in range(max(1, cfg.event_search_start), len(speed) - 1):
        if not (valid[step - 1] and valid[step]):
            continue
        decel = float((speed[step - 1] - speed[step]) / cfg.dt)
        if decel < cfg.decel_mps2:
            continue
        end = min(len(speed), step + cfg.decel_lookahead_steps + 1)
        future_valid = valid[step:end]
        if not bool(future_valid.any()):
            continue
        future_min = float(np.min(speed[step:end][future_valid]))
        if float(speed[step - 1] - future_min) < cfg.speed_drop_mps:
            continue
        if step - last_kept > cfg.decel_lookahead_steps // 2:
            onsets.append(step)
            last_kept = step
    return onsets


def attribute_decelerations(events: Iterable[PairEvent], onsets: Iterable[int], cfg: SampleConfig) -> list[PairEvent]:
    """Assign each deceleration episode to at most one pair event."""
    events = list(events)
    attributed: set[int] = set()
    ambiguous: set[int] = set()
    for onset in onsets:
        candidates: list[tuple[int, int]] = []
        for idx, event in enumerate(events):
            delta = int(event.event_step - onset)
            if -cfg.attribution_before_steps <= delta <= cfg.attribution_after_steps:
                candidates.append((abs(delta), idx))
        candidates.sort()
        if not candidates:
            continue
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= cfg.attribution_ambiguity_steps:
            best_distance = candidates[0][0]
            for distance, idx in candidates:
                if distance - best_distance <= cfg.attribution_ambiguity_steps:
                    ambiguous.add(idx)
            continue
        attributed.add(candidates[0][1])

    return [
        PairEvent(
            **{
                **event.__dict__,
                "attributed_decel": idx in attributed and idx not in ambiguous,
                "attribution_ambiguous": idx in ambiguous,
            }
        )
        for idx, event in enumerate(events)
    ]


def extract_query_history(
    focus: np.ndarray,
    other: np.ndarray,
    *,
    query_step: int,
    history_steps: int,
    dt: float,
) -> np.ndarray | None:
    """Extract a causal history in one fixed query-time focus frame."""
    start = int(query_step - history_steps + 1)
    if start < 0 or query_step >= focus.shape[0]:
        return None
    steps = np.arange(start, query_step + 1)
    if not bool(((focus[steps, 5] > 0.5) & (other[steps, 5] > 0.5)).all()):
        return None

    query_yaw = float(focus[query_step, 6])
    # Relative position is translation-invariant; using one query-time rotation
    # keeps every history timestep in a single non-warped coordinate frame.
    rel_pos = _rotate_xy(other[steps, 0:2] - focus[steps, 0:2], query_yaw)
    rel_vel = _rotate_xy(other[steps, 3:5] - focus[steps, 3:5], query_yaw)
    heading_diff = wrap_angle(other[steps, 6] - focus[steps, 6])
    focus_speed = focus[steps, 2]
    other_speed = other[steps, 2]
    focus_accel = np.zeros_like(focus_speed)
    other_accel = np.zeros_like(other_speed)
    focus_yaw_rate = np.zeros_like(focus_speed)
    if history_steps > 1:
        focus_accel[1:] = np.diff(focus_speed) / dt
        other_accel[1:] = np.diff(other_speed) / dt
        focus_yaw_rate[1:] = wrap_angle(np.diff(focus[steps, 6])) / dt
        focus_accel[0] = focus_accel[1]
        other_accel[0] = other_accel[1]
        focus_yaw_rate[0] = focus_yaw_rate[1]

    history = np.stack(
        [
            rel_pos[:, 0],
            rel_pos[:, 1],
            rel_vel[:, 0],
            rel_vel[:, 1],
            np.sin(heading_diff),
            np.cos(heading_diff),
            focus_speed,
            focus_accel,
            focus_yaw_rate,
            other_speed,
            other_accel,
        ],
        axis=-1,
    )
    return history.astype(np.float32)


def _future_speed_drop(focus: np.ndarray, query_step: int, event_step: int, cfg: SampleConfig) -> float:
    end = min(len(focus), max(query_step + 1, event_step + cfg.decel_lookahead_steps + 1))
    valid = focus[query_step:end, 5] > 0.5
    if not bool(valid.any()):
        return 0.0
    return max(0.0, float(focus[query_step, 2] - np.min(focus[query_step:end, 2][valid])))


def _response_for_event(event: PairEvent, focus: np.ndarray, query_step: int, cfg: SampleConfig) -> tuple[int, bool]:
    if event.attribution_ambiguous:
        return -1, False
    relation = RELATION_NAMES[event.relation_index]
    delta_arrival = (event.focus_arrival_step - event.other_arrival_step) * cfg.dt
    if relation in {"crossing_or_oncoming_conflict", "converging_conflict"}:
        if delta_arrival <= -cfg.priority_time_margin_s:
            return RESPONSE_NAMES.index("goes_first"), True
        if delta_arrival >= cfg.yield_time_margin_s and event.attributed_decel:
            return RESPONSE_NAMES.index("yields"), True
        return -1, False

    if event.attributed_decel:
        return RESPONSE_NAMES.index("decelerates"), True
    speed_drop = _future_speed_drop(focus, query_step, event.event_step, cfg)
    if speed_drop <= cfg.maintain_max_speed_drop_mps:
        return RESPONSE_NAMES.index("maintains"), True
    return -1, False


def build_scene_samples(agents: np.ndarray, agent_mask: np.ndarray, cfg: SampleConfig) -> list[PairSample]:
    agents = ensure_agent_time_layout(np.asarray(agents, dtype=np.float32), np.asarray(agent_mask))
    if agents.shape[-1] < 8:
        raise ValueError(f"Expected at least 8 agent features, got {agents.shape}")
    if not bool(agent_mask[0]):
        return []
    focus = agents[0]
    events: list[PairEvent] = []
    for candidate_index in np.flatnonzero(np.asarray(agent_mask, dtype=bool)):
        candidate_index = int(candidate_index)
        if candidate_index == 0:
            continue
        event = detect_pair_event(focus, agents[candidate_index], candidate_index=candidate_index, cfg=cfg)
        if event is not None:
            events.append(event)
    events = attribute_decelerations(events, detect_deceleration_onsets(focus, cfg), cfg)

    samples: list[PairSample] = []
    for event in events:
        other = agents[event.candidate_index]
        for lead_steps in cfg.lead_steps:
            query_step = int(event.event_step - lead_steps)
            history = extract_query_history(
                focus,
                other,
                query_step=query_step,
                history_steps=cfg.history_steps,
                dt=cfg.dt,
            )
            if history is None:
                continue
            response_index, eligible = _response_for_event(event, focus, query_step, cfg)
            samples.append(
                PairSample(
                    candidate_index=event.candidate_index,
                    event_step=event.event_step,
                    query_step=query_step,
                    lead_steps=int(lead_steps),
                    relation_index=event.relation_index,
                    response_index=response_index,
                    eligible=eligible,
                    focus_type=int(round(float(focus[query_step, 7]))),
                    candidate_type=int(round(float(other[query_step, 7]))),
                    delta_arrival_time_s=(event.focus_arrival_step - event.other_arrival_step) * cfg.dt,
                    pet_s=event.pet_s,
                    spatial_min_dist_m=event.spatial_min_dist_m,
                    history=history,
                )
            )
    return samples
