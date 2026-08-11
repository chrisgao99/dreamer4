"""Extract fixed-length, shared-time-axis two-agent interaction sequences.

Version one intentionally has no discrete relation or response labels.  A pair
is retained when its future paths contain an asynchronously close point pair
within a bounded arrival-time difference.  The first arrival is placed at a
fixed sequence index so that the full pair trajectories can be compared
directly without DTW.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .pair_samples import ensure_agent_time_layout, wrap_angle
except ImportError:
    from pair_samples import ensure_agent_time_layout, wrap_angle  # type: ignore


PAIR_FEATURE_NAMES = (
    "first_x_m",
    "first_y_m",
    "first_vx_mps",
    "first_vy_mps",
    "first_heading_sin",
    "first_heading_cos",
    "second_x_m",
    "second_y_m",
    "second_vx_mps",
    "second_vy_mps",
    "second_heading_sin",
    "second_heading_cos",
)


@dataclass(frozen=True)
class SoftPairConfig:
    dt: float = 0.1
    event_search_start: int = 10
    history_steps: int = 20
    post_first_steps: int = 40
    max_pet_steps: int = 30
    max_spatial_distance_m: float = 6.0
    relevance_distance_scale_m: float = 3.0
    relevance_pet_scale_s: float = 1.5

    @property
    def sequence_steps(self) -> int:
        return int(self.history_steps + self.post_first_steps)


@dataclass(frozen=True)
class SoftPairSample:
    first_index: int
    second_index: int
    first_arrival_step: int
    second_arrival_step: int
    first_step: int
    pet_steps: int
    spatial_min_dist_m: float
    relevance_score: float
    conflict_xy: np.ndarray
    sequence: np.ndarray


def _rotate_xy(xy: np.ndarray, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    return np.asarray(xy, dtype=np.float32) @ rotation.T


def find_constrained_closest_points(
    agent_a: np.ndarray,
    agent_b: np.ndarray,
    cfg: SoftPairConfig,
) -> tuple[int, int, float] | None:
    """Find the closest asynchronous path points that support a full window.

    The PET constraint is applied before minimizing spatial distance.  This
    avoids discarding a useful 1 m / 2 s point pair merely because an unrelated
    0 m / 5 s point pair is the unconstrained spatial minimum.
    """
    if agent_a.shape != agent_b.shape or agent_a.ndim != 2:
        raise ValueError(f"Expected equal (T,F) arrays, got {agent_a.shape} and {agent_b.shape}")
    t = int(agent_a.shape[0])
    if t < cfg.sequence_steps:
        return None

    valid_a = agent_a[:, 5] > 0.5
    valid_b = agent_b[:, 5] > 0.5
    first_ok = np.zeros((t,), dtype=bool)
    first_min = cfg.history_steps - 1
    first_max = t - cfg.post_first_steps - 1
    for first_step in range(first_min, first_max + 1):
        start = first_step - cfg.history_steps + 1
        end = first_step + cfg.post_first_steps + 1
        first_ok[first_step] = bool(valid_a[start:end].all() and valid_b[start:end].all())
    if not bool(first_ok.any()):
        return None

    steps = np.arange(t, dtype=np.int16)
    step_a = steps[:, None]
    step_b = steps[None, :]
    first = np.minimum(step_a, step_b)
    admissible = (
        valid_a[:, None]
        & valid_b[None, :]
        & (step_a >= cfg.event_search_start)
        & (step_b >= cfg.event_search_start)
        & (np.abs(step_a - step_b) <= cfg.max_pet_steps)
        & first_ok[first]
    )
    if not bool(admissible.any()):
        return None

    delta = agent_a[:, None, 0:2] - agent_b[None, :, 0:2]
    distance_sq = np.sum(delta * delta, axis=-1)
    distance_sq[~admissible] = np.inf
    flat = int(np.argmin(distance_sq))
    step_a_value, step_b_value = np.unravel_index(flat, distance_sq.shape)
    return int(step_a_value), int(step_b_value), float(np.sqrt(distance_sq[step_a_value, step_b_value]))


def _pair_sequence(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_step: int,
    first_arrival_step: int,
    second_arrival_step: int,
    cfg: SoftPairConfig,
) -> tuple[np.ndarray, np.ndarray]:
    start = int(first_step - cfg.history_steps + 1)
    end = int(first_step + cfg.post_first_steps + 1)
    if end - start != cfg.sequence_steps:
        raise AssertionError("Unexpected shared-window length")

    conflict_xy = 0.5 * (
        first[first_arrival_step, 0:2] + second[second_arrival_step, 0:2]
    )
    frame_yaw = float(first[first_arrival_step, 6])

    features: list[np.ndarray] = []
    for agent in (first, second):
        xy = _rotate_xy(agent[start:end, 0:2] - conflict_xy[None], frame_yaw)
        velocity = _rotate_xy(agent[start:end, 3:5], frame_yaw)
        yaw = wrap_angle(agent[start:end, 6] - frame_yaw)
        features.extend((xy, velocity, np.sin(yaw)[:, None], np.cos(yaw)[:, None]))
    sequence = np.concatenate(features, axis=1).astype(np.float32)
    return sequence, np.asarray(conflict_xy, dtype=np.float32)


def extract_soft_pair_sample(
    agent_a: np.ndarray,
    agent_b: np.ndarray,
    *,
    index_a: int,
    index_b: int,
    cfg: SoftPairConfig,
) -> SoftPairSample | None:
    closest = find_constrained_closest_points(agent_a, agent_b, cfg)
    if closest is None:
        return None
    step_a, step_b, spatial_distance = closest
    if spatial_distance > cfg.max_spatial_distance_m:
        return None

    if (step_a, int(index_a)) <= (step_b, int(index_b)):
        first, second = agent_a, agent_b
        first_index, second_index = int(index_a), int(index_b)
        first_arrival, second_arrival = step_a, step_b
    else:
        first, second = agent_b, agent_a
        first_index, second_index = int(index_b), int(index_a)
        first_arrival, second_arrival = step_b, step_a

    pet_steps = int(second_arrival - first_arrival)
    pet_s = pet_steps * float(cfg.dt)
    relevance = float(
        np.exp(
            -0.5
            * (
                (spatial_distance / cfg.relevance_distance_scale_m) ** 2
                + (pet_s / cfg.relevance_pet_scale_s) ** 2
            )
        )
    )
    sequence, conflict_xy = _pair_sequence(
        first,
        second,
        first_step=first_arrival,
        first_arrival_step=first_arrival,
        second_arrival_step=second_arrival,
        cfg=cfg,
    )
    return SoftPairSample(
        first_index=first_index,
        second_index=second_index,
        first_arrival_step=first_arrival,
        second_arrival_step=second_arrival,
        first_step=first_arrival,
        pet_steps=pet_steps,
        spatial_min_dist_m=spatial_distance,
        relevance_score=relevance,
        conflict_xy=conflict_xy,
        sequence=sequence,
    )


def build_focus_samples(
    agents: np.ndarray,
    agent_mask: np.ndarray,
    cfg: SoftPairConfig,
) -> list[SoftPairSample]:
    """Extract every retained focus-versus-selected-agent pair in one NPZ."""
    agents = ensure_agent_time_layout(np.asarray(agents, dtype=np.float32), np.asarray(agent_mask))
    if agents.shape[-1] < 8:
        raise ValueError(f"Expected at least 8 agent features, got {agents.shape}")
    if not bool(agent_mask[0]):
        return []
    result: list[SoftPairSample] = []
    for candidate_index in np.flatnonzero(np.asarray(agent_mask, dtype=bool)):
        candidate_index = int(candidate_index)
        if candidate_index == 0:
            continue
        sample = extract_soft_pair_sample(
            agents[0],
            agents[candidate_index],
            index_a=0,
            index_b=candidate_index,
            cfg=cfg,
        )
        if sample is not None:
            result.append(sample)
    return result
