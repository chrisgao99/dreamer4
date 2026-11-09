"""Extract lossless 91-step pair samples with physical contact intervals.

The inclusion decision and event representation are deliberately separate:
original two-OOI pairs are always retained, while mined pairs require at least
one physical path-contact component.  Track validity never removes an OOI
pair; it is preserved as an explicit mask on the full 91-step sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

try:
    from .common import wrap_angle
except ImportError:
    from common import wrap_angle  # type: ignore


FULL_PAIR_FEATURE_NAMES = (
    "x_m",
    "y_m",
    "vx_mps",
    "vy_mps",
    "heading_sin",
    "heading_cos",
)


@dataclass(frozen=True)
class FullPairConfig:
    dt: float = 0.1
    contact_buffer_m: float = 1.0
    pet_soft_scale_s: float = 3.0
    # A non-positive value retains every mined non-OOI physical-contact pair.
    non_ooi_top_k_per_focus: int = 0


@dataclass(frozen=True)
class PathIntersection:
    step_a: float
    step_b: float
    xy: np.ndarray


@dataclass(frozen=True)
class ContactComponent:
    label: int
    start_a: float
    end_a: float
    start_b: float
    end_b: float
    primary_step_a: float
    primary_step_b: float
    primary_xy: np.ndarray
    zone_pet_steps: float
    center_pet_steps: float
    min_clearance_m: float
    num_cells: int
    has_path_intersection: bool


@dataclass(frozen=True)
class FullPairSample:
    first_index: int
    second_index: int
    first_agent_id: int
    second_agent_id: int
    first_agent_type: int
    second_agent_type: int
    is_original_ooi_pair: bool
    event_mode: str
    primary_step_first: float
    primary_step_second: float
    interval_start_first: float
    interval_end_first: float
    interval_start_second: float
    interval_end_second: float
    zone_pet_steps: float
    center_pet_steps: float
    min_clearance_m: float
    num_contact_components: int
    primary_component_cells: int
    relevance_score: float
    conflict_xy: np.ndarray
    trajectory: np.ndarray
    valid_mask: np.ndarray
    agent_size_m: np.ndarray


def _cross_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _valid_segments(agent: np.ndarray) -> np.ndarray:
    valid = np.asarray(agent[:, 5] > 0.5, dtype=bool)
    return valid[:-1] & valid[1:]


def continuous_path_intersections(agent_a: np.ndarray, agent_b: np.ndarray) -> list[PathIntersection]:
    """Return proper intersections between linearly interpolated path segments."""
    p = agent_a[:-1, 0:2][:, None, :]
    r = (agent_a[1:, 0:2] - agent_a[:-1, 0:2])[:, None, :]
    q = agent_b[:-1, 0:2][None, :, :]
    s = (agent_b[1:, 0:2] - agent_b[:-1, 0:2])[None, :, :]
    denominator = _cross_2d(r, s)
    non_parallel = np.abs(denominator) > 1e-7
    qp = q - p
    safe_denominator = np.where(non_parallel, denominator, 1.0)
    fraction_a = _cross_2d(qp, s) / safe_denominator
    fraction_b = _cross_2d(qp, r) / safe_denominator
    admissible = (
        non_parallel
        & _valid_segments(agent_a)[:, None]
        & _valid_segments(agent_b)[None, :]
        & (fraction_a >= -1e-6)
        & (fraction_a <= 1.0 + 1e-6)
        & (fraction_b >= -1e-6)
        & (fraction_b <= 1.0 + 1e-6)
    )
    result: list[PathIntersection] = []
    for index_a, index_b in np.argwhere(admissible):
        fa = float(np.clip(fraction_a[index_a, index_b], 0.0, 1.0))
        fb = float(np.clip(fraction_b[index_a, index_b], 0.0, 1.0))
        xy = agent_a[index_a, 0:2] + fa * (agent_a[index_a + 1, 0:2] - agent_a[index_a, 0:2])
        result.append(PathIntersection(float(index_a) + fa, float(index_b) + fb, xy.astype(np.float32)))
    return result


def obb_separation_matrix(
    agent_a: np.ndarray,
    agent_b: np.ndarray,
    size_a: np.ndarray,
    size_b: np.ndarray,
) -> np.ndarray:
    """Return a vectorized separating-axis clearance proxy for oriented boxes.

    Positive values are a separating gap on at least one rectangle axis;
    non-positive values mean the boxes overlap.  Thresholding this value at a
    physical buffer is equivalent to an axis-aligned Minkowski inflation and
    is substantially more faithful than a fixed centre-distance threshold.
    """
    yaw_a = agent_a[:, 6]
    yaw_b = agent_b[:, 6]
    cos_a, sin_a = np.cos(yaw_a), np.sin(yaw_a)
    cos_b, sin_b = np.cos(yaw_b), np.sin(yaw_b)
    forward_a = np.stack((cos_a, sin_a), axis=-1)
    lateral_a = np.stack((-sin_a, cos_a), axis=-1)
    forward_b = np.stack((cos_b, sin_b), axis=-1)
    lateral_b = np.stack((-sin_b, cos_b), axis=-1)
    delta = agent_b[None, :, 0:2] - agent_a[:, None, 0:2]

    half_length_a = 0.5 * size_a[:, 0]
    half_width_a = 0.5 * size_a[:, 1]
    half_length_b = 0.5 * size_b[:, 0]
    half_width_b = 0.5 * size_b[:, 1]
    cosine = np.abs(forward_a @ forward_b.T)
    sine = np.abs(lateral_a @ forward_b.T)

    projection_a_forward = np.abs(np.einsum("ijc,ic->ij", delta, forward_a))
    radius_b_on_a_forward = half_length_b[None, :] * cosine + half_width_b[None, :] * sine
    gap_a_forward = projection_a_forward - half_length_a[:, None] - radius_b_on_a_forward

    projection_a_lateral = np.abs(np.einsum("ijc,ic->ij", delta, lateral_a))
    radius_b_on_a_lateral = half_length_b[None, :] * sine + half_width_b[None, :] * cosine
    gap_a_lateral = projection_a_lateral - half_width_a[:, None] - radius_b_on_a_lateral

    projection_b_forward = np.abs(np.einsum("ijc,jc->ij", delta, forward_b))
    radius_a_on_b_forward = half_length_a[:, None] * cosine + half_width_a[:, None] * sine
    gap_b_forward = projection_b_forward - radius_a_on_b_forward - half_length_b[None, :]

    projection_b_lateral = np.abs(np.einsum("ijc,jc->ij", delta, lateral_b))
    radius_a_on_b_lateral = half_length_a[:, None] * sine + half_width_a[:, None] * cosine
    gap_b_lateral = projection_b_lateral - radius_a_on_b_lateral - half_width_b[None, :]

    separation = np.maximum.reduce((gap_a_forward, gap_a_lateral, gap_b_forward, gap_b_lateral))
    valid = (
        (agent_a[:, 5] > 0.5)[:, None]
        & (agent_b[:, 5] > 0.5)[None, :]
        & (size_a[:, 0] > 0.0)[:, None]
        & (size_a[:, 1] > 0.0)[:, None]
        & (size_b[:, 0] > 0.0)[None, :]
        & (size_b[:, 1] > 0.0)[None, :]
    )
    separation[~valid] = np.inf
    return separation.astype(np.float32)


def _interpolate_xy(agent: np.ndarray, step: float) -> np.ndarray:
    step = float(np.clip(step, 0.0, len(agent) - 1.0))
    low = int(np.floor(step))
    high = min(len(agent) - 1, low + 1)
    fraction = step - low
    return ((1.0 - fraction) * agent[low, 0:2] + fraction * agent[high, 0:2]).astype(np.float32)


def _interpolate_yaw(agent: np.ndarray, step: float) -> float:
    step = float(np.clip(step, 0.0, len(agent) - 1.0))
    low = int(np.floor(step))
    high = min(len(agent) - 1, low + 1)
    fraction = step - low
    delta = float(wrap_angle(float(agent[high, 6]) - float(agent[low, 6])))
    return float(wrap_angle(float(agent[low, 6]) + fraction * delta))


def continuous_closest_points(agent_a: np.ndarray, agent_b: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Closest points between valid linearly interpolated polylines."""
    intersections = continuous_path_intersections(agent_a, agent_b)
    if intersections:
        chosen = min(intersections, key=lambda item: (abs(item.step_a - item.step_b), item.step_a, item.step_b))
        return chosen.step_a, chosen.step_b, 0.0, chosen.xy

    candidates: list[tuple[float, float, float, np.ndarray]] = []
    for points, segments, points_are_a in ((agent_a, agent_b, True), (agent_b, agent_a, False)):
        point_valid = points[:, 5] > 0.5
        segment_valid = _valid_segments(segments)
        p = points[:, None, 0:2]
        q = segments[:-1, 0:2][None, :, :]
        direction = (segments[1:, 0:2] - segments[:-1, 0:2])[None, :, :]
        denominator = np.sum(direction * direction, axis=-1)
        fraction = np.sum((p - q) * direction, axis=-1) / np.maximum(denominator, 1e-9)
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = q + fraction[..., None] * direction
        distance_sq = np.sum((p - projected) ** 2, axis=-1)
        admissible = point_valid[:, None] & segment_valid[None, :] & (denominator > 1e-9)
        distance_sq[~admissible] = np.inf
        if not bool(np.isfinite(distance_sq).any()):
            continue
        flat = int(np.argmin(distance_sq))
        point_step, segment_step = np.unravel_index(flat, distance_sq.shape)
        segment_fraction = float(fraction[point_step, segment_step])
        xy_point = points[point_step, 0:2]
        xy_segment = projected[point_step, segment_step]
        midpoint = (0.5 * (xy_point + xy_segment)).astype(np.float32)
        if points_are_a:
            step_a, step_b = float(point_step), float(segment_step) + segment_fraction
        else:
            step_a, step_b = float(segment_step) + segment_fraction, float(point_step)
        candidates.append((step_a, step_b, float(np.sqrt(distance_sq[point_step, segment_step])), midpoint))
    if not candidates:
        valid_a = np.flatnonzero(agent_a[:, 5] > 0.5)
        valid_b = np.flatnonzero(agent_b[:, 5] > 0.5)
        if not len(valid_a) or not len(valid_b):
            raise ValueError("Cannot find closest points for trajectories without valid states")
        delta = agent_a[valid_a, None, 0:2] - agent_b[None, valid_b, 0:2]
        distance_sq = np.sum(delta * delta, axis=-1)
        local_a, local_b = np.unravel_index(int(np.argmin(distance_sq)), distance_sq.shape)
        step_a, step_b = float(valid_a[local_a]), float(valid_b[local_b])
        midpoint = (0.5 * (agent_a[int(step_a), 0:2] + agent_b[int(step_b), 0:2])).astype(np.float32)
        return step_a, step_b, float(np.sqrt(distance_sq[local_a, local_b])), midpoint
    return min(candidates, key=lambda item: (item[2], abs(item[0] - item[1])))


def contact_components(
    agent_a: np.ndarray,
    agent_b: np.ndarray,
    size_a: np.ndarray,
    size_b: np.ndarray,
    cfg: FullPairConfig,
) -> list[ContactComponent]:
    separation = obb_separation_matrix(agent_a, agent_b, size_a, size_b)
    contact = separation <= cfg.contact_buffer_m
    intersections = continuous_path_intersections(agent_a, agent_b)
    for item in intersections:
        index_a = int(np.clip(round(item.step_a), 0, len(agent_a) - 1))
        index_b = int(np.clip(round(item.step_b), 0, len(agent_b) - 1))
        contact[index_a, index_b] = True
        separation[index_a, index_b] = min(0.0, float(separation[index_a, index_b]))
    if not bool(contact.any()):
        return []

    labels, count = ndimage.label(contact, structure=np.ones((3, 3), dtype=np.int8))
    result: list[ContactComponent] = []
    for label in range(1, int(count) + 1):
        cells = np.argwhere(labels == label)
        if not len(cells):
            continue
        steps_a = cells[:, 0]
        steps_b = cells[:, 1]
        start_a = max(0.0, float(steps_a.min()) - 0.5)
        end_a = min(len(agent_a) - 1.0, float(steps_a.max()) + 0.5)
        start_b = max(0.0, float(steps_b.min()) - 0.5)
        end_b = min(len(agent_b) - 1.0, float(steps_b.max()) + 0.5)
        # The projections alone can overlap in absolute timestep even when a
        # following component is a slanted band far from the i=j diagonal.
        # PET is therefore the closest temporal separation of an actual
        # contact cell.  Each sampled state represents a half-step temporal
        # extent on either side, hence the one-step interval correction.
        cell_pet = np.abs(steps_a - steps_b).astype(np.float32)
        zone_pet = max(0.0, float(cell_pet.min()) - 1.0)

        component_intersections = []
        for item in intersections:
            index_a = int(np.clip(round(item.step_a), 0, len(agent_a) - 1))
            index_b = int(np.clip(round(item.step_b), 0, len(agent_b) - 1))
            if labels[index_a, index_b] == label:
                component_intersections.append(item)
        if component_intersections:
            primary_intersection = min(
                component_intersections,
                key=lambda item: (abs(item.step_a - item.step_b), item.step_a, item.step_b),
            )
            primary_a = primary_intersection.step_a
            primary_b = primary_intersection.step_b
            primary_xy = primary_intersection.xy
        else:
            pet = np.abs(steps_a - steps_b)
            min_pet = int(pet.min())
            candidates = cells[pet == min_pet]
            centre = np.asarray([(start_a + end_a) * 0.5, (start_b + end_b) * 0.5])
            chosen = candidates[int(np.argmin(np.sum((candidates - centre[None]) ** 2, axis=1)))]
            primary_a, primary_b = float(chosen[0]), float(chosen[1])
            primary_xy = 0.5 * (_interpolate_xy(agent_a, primary_a) + _interpolate_xy(agent_b, primary_b))
        component_clearance = np.maximum(separation[labels == label], 0.0)
        result.append(
            ContactComponent(
                label=label,
                start_a=start_a,
                end_a=end_a,
                start_b=start_b,
                end_b=end_b,
                primary_step_a=primary_a,
                primary_step_b=primary_b,
                primary_xy=np.asarray(primary_xy, dtype=np.float32),
                zone_pet_steps=float(zone_pet),
                center_pet_steps=abs(float(primary_a - primary_b)),
                min_clearance_m=float(component_clearance.min()) if len(component_clearance) else 0.0,
                num_cells=int(len(cells)),
                has_path_intersection=bool(component_intersections),
            )
        )
    return result


def select_primary_component(components: list[ContactComponent]) -> ContactComponent:
    if not components:
        raise ValueError("Cannot select a primary contact component from an empty list")
    return min(
        components,
        key=lambda item: (
            item.zone_pet_steps,
            0 if item.has_path_intersection else 1,
            item.center_pet_steps,
            -item.num_cells,
        ),
    )


def _rotate_xy(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.asarray(xy, dtype=np.float32) @ np.asarray([[c, s], [-s, c]], dtype=np.float32).T


def _trajectory_features(
    agent: np.ndarray,
    *,
    origin: np.ndarray,
    frame_yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(agent[:, 5] > 0.5, dtype=bool)
    xy = _rotate_xy(agent[:, 0:2] - origin[None], frame_yaw)
    velocity = _rotate_xy(agent[:, 3:5], frame_yaw)
    yaw = wrap_angle(agent[:, 6] - frame_yaw)
    features = np.concatenate((xy, velocity, np.sin(yaw)[:, None], np.cos(yaw)[:, None]), axis=1).astype(np.float32)
    features[~valid] = 0.0
    return features, valid


def _median_size(size: np.ndarray, valid: np.ndarray) -> np.ndarray:
    usable = valid & (size[:, 0] > 0.0) & (size[:, 1] > 0.0)
    if not bool(usable.any()):
        return np.asarray([0.0, 0.0], dtype=np.float32)
    return np.median(size[usable], axis=0).astype(np.float32)


def extract_full_pair_sample(
    agent_a: np.ndarray,
    agent_b: np.ndarray,
    size_a: np.ndarray,
    size_b: np.ndarray,
    *,
    index_a: int,
    index_b: int,
    agent_id_a: int,
    agent_id_b: int,
    is_original_ooi_pair: bool,
    cfg: FullPairConfig,
) -> FullPairSample | None:
    """Extract a full pair; only a non-OOI pair without contact is rejected."""
    components = contact_components(agent_a, agent_b, size_a, size_b, cfg)
    if components:
        primary = select_primary_component(components)
        step_a, step_b = primary.primary_step_a, primary.primary_step_b
        interval_a = (primary.start_a, primary.end_a)
        interval_b = (primary.start_b, primary.end_b)
        zone_pet = primary.zone_pet_steps
        center_pet = primary.center_pet_steps
        min_clearance = primary.min_clearance_m
        conflict_xy = primary.primary_xy
        event_mode = "path_intersection" if primary.has_path_intersection else "obb_contact_interval"
        primary_cells = primary.num_cells
    else:
        if not is_original_ooi_pair:
            return None
        step_a, step_b, centre_distance, conflict_xy = continuous_closest_points(agent_a, agent_b)
        interval_a = (step_a, step_a)
        interval_b = (step_b, step_b)
        zone_pet = abs(step_a - step_b)
        center_pet = zone_pet
        min_clearance = max(0.0, centre_distance - float(_median_size(size_a, agent_a[:, 5] > 0.5)[0]) * 0.5
                            - float(_median_size(size_b, agent_b[:, 5] > 0.5)[0]) * 0.5)
        event_mode = "ooi_closest_fallback"
        primary_cells = 0

    if (step_a, int(agent_id_a)) <= (step_b, int(agent_id_b)):
        first_agent, second_agent = agent_a, agent_b
        first_size, second_size = size_a, size_b
        first_index, second_index = int(index_a), int(index_b)
        first_id, second_id = int(agent_id_a), int(agent_id_b)
        first_step, second_step = float(step_a), float(step_b)
        first_interval, second_interval = interval_a, interval_b
    else:
        first_agent, second_agent = agent_b, agent_a
        first_size, second_size = size_b, size_a
        first_index, second_index = int(index_b), int(index_a)
        first_id, second_id = int(agent_id_b), int(agent_id_a)
        first_step, second_step = float(step_b), float(step_a)
        first_interval, second_interval = interval_b, interval_a

    frame_yaw = _interpolate_yaw(first_agent, first_step)
    first_features, first_valid = _trajectory_features(first_agent, origin=conflict_xy, frame_yaw=frame_yaw)
    second_features, second_valid = _trajectory_features(second_agent, origin=conflict_xy, frame_yaw=frame_yaw)
    trajectory = np.stack((first_features, second_features), axis=0)
    valid_mask = np.stack((first_valid, second_valid), axis=0)
    agent_size = np.stack(
        (_median_size(first_size, first_valid), _median_size(second_size, second_valid)), axis=0
    )
    pet_s = zone_pet * cfg.dt
    relevance = float(np.exp(-0.5 * (pet_s / cfg.pet_soft_scale_s) ** 2))
    first_type = int(round(float(first_agent[np.flatnonzero(first_valid)[0], 7]))) if bool(first_valid.any()) else 0
    second_type = int(round(float(second_agent[np.flatnonzero(second_valid)[0], 7]))) if bool(second_valid.any()) else 0
    return FullPairSample(
        first_index=first_index,
        second_index=second_index,
        first_agent_id=first_id,
        second_agent_id=second_id,
        first_agent_type=first_type,
        second_agent_type=second_type,
        is_original_ooi_pair=bool(is_original_ooi_pair),
        event_mode=event_mode,
        primary_step_first=first_step,
        primary_step_second=second_step,
        interval_start_first=float(first_interval[0]),
        interval_end_first=float(first_interval[1]),
        interval_start_second=float(second_interval[0]),
        interval_end_second=float(second_interval[1]),
        zone_pet_steps=float(zone_pet),
        center_pet_steps=float(center_pet),
        min_clearance_m=float(min_clearance),
        num_contact_components=len(components),
        primary_component_cells=int(primary_cells),
        relevance_score=relevance,
        conflict_xy=np.asarray(conflict_xy, dtype=np.float32),
        trajectory=trajectory,
        valid_mask=valid_mask,
        agent_size_m=agent_size,
    )
