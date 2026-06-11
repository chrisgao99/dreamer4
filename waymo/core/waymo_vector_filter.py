"""Waymo tf.Example vector filtering for the Dreamer-style tokenizer.

This module converts one Waymo motion tf.Example scenario into fixed-size
agent, map-polyline, and traffic-light tensors. It intentionally keeps the
output simple and numpy-native so it can be inspected, saved as NPZ, or wrapped
by a PyTorch Dataset later.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


N_AGENTS_WAYMO = 128
N_ROADGRAPH_SAMPLES = 30000
N_LIGHTS_WAYMO = 16
N_PAST = 10
N_CURRENT = 1
N_FUTURE = 80
N_STEPS = N_PAST + N_CURRENT + N_FUTURE
CURRENT_IDX = N_PAST


@dataclass(frozen=True)
class WaymoVectorConfig:
    num_agents: int = 32
    agent_distance_threshold: float = 80.0
    map_distance_threshold: float = 100.0
    max_map_polylines: int = 256
    max_points_per_polyline: int = 20
    use_all_timesteps_for_selection: bool = True
    normalize_to_ego: bool = True
    require_objects_of_interest: bool = False
    min_objects_of_interest: int = 1
    require_sdc_object_of_interest: bool = False
    prioritize_objects_of_interest: bool = True


_EXAMPLE_CLASS = None


def _get_tfexample_class():
    """Build a minimal tf.train.Example protobuf class without TensorFlow."""
    global _EXAMPLE_CLASS
    if _EXAMPLE_CLASS is not None:
        return _EXAMPLE_CLASS

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "minimal_tensorflow_example.proto"
    file_proto.package = "tensorflow"
    file_proto.syntax = "proto3"

    bytes_list = file_proto.message_type.add()
    bytes_list.name = "BytesList"
    value = bytes_list.field.add()
    value.name = "value"
    value.number = 1
    value.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    value.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    float_list = file_proto.message_type.add()
    float_list.name = "FloatList"
    value = float_list.field.add()
    value.name = "value"
    value.number = 1
    value.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    value.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    value.options.packed = True

    int64_list = file_proto.message_type.add()
    int64_list.name = "Int64List"
    value = int64_list.field.add()
    value.name = "value"
    value.number = 1
    value.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    value.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    value.options.packed = True

    feature = file_proto.message_type.add()
    feature.name = "Feature"
    feature.oneof_decl.add().name = "kind"
    for field_name, number, type_name in [
        ("bytes_list", 1, ".tensorflow.BytesList"),
        ("float_list", 2, ".tensorflow.FloatList"),
        ("int64_list", 3, ".tensorflow.Int64List"),
    ]:
        field = feature.field.add()
        field.name = field_name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
        field.type_name = type_name
        field.oneof_index = 0

    features = file_proto.message_type.add()
    features.name = "Features"
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    key = entry.field.add()
    key.name = "key"
    key.number = 1
    key.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    key.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    val = entry.field.add()
    val.name = "value"
    val.number = 2
    val.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    val.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    val.type_name = ".tensorflow.Feature"
    field = features.field.add()
    field.name = "feature"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features.FeatureEntry"

    example = file_proto.message_type.add()
    example.name = "Example"
    field = example.field.add()
    field.name = "features"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".tensorflow.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    descriptor = pool.FindMessageTypeByName("tensorflow.Example")
    if hasattr(message_factory, "GetMessageClass"):
        _EXAMPLE_CLASS = message_factory.GetMessageClass(descriptor)
    else:
        _EXAMPLE_CLASS = message_factory.MessageFactory(pool).GetPrototype(descriptor)
    return _EXAMPLE_CLASS


def _iter_tfrecord_records(tfrecord_path: str) -> Iterator[bytes]:
    with open(tfrecord_path, "rb") as f:
        while True:
            header = f.read(12)
            if not header:
                break
            if len(header) != 12:
                raise IOError(f"Truncated TFRecord header in {tfrecord_path}")
            length = struct.unpack("<Q", header[:8])[0]
            data = f.read(length)
            if len(data) != length:
                raise IOError(f"Truncated TFRecord payload in {tfrecord_path}")
            crc = f.read(4)
            if len(crc) != 4:
                raise IOError(f"Truncated TFRecord footer in {tfrecord_path}")
            yield data


def _feature_to_numpy(feature):
    kind = feature.WhichOneof("kind")
    if kind == "float_list":
        return np.asarray(feature.float_list.value, dtype=np.float32)
    if kind == "int64_list":
        return np.asarray(feature.int64_list.value, dtype=np.int64)
    if kind == "bytes_list":
        values = feature.bytes_list.value
        if len(values) == 1:
            return values[0].decode("utf-8")
        return [v.decode("utf-8") for v in values]
    return None


def parse_tfexample(raw_record) -> Dict[str, np.ndarray]:
    example_cls = _get_tfexample_class()
    example = example_cls()
    if hasattr(raw_record, "numpy"):
        raw_record = raw_record.numpy()
    example.ParseFromString(raw_record)
    return {
        key: _feature_to_numpy(feature)
        for key, feature in example.features.feature.items()
    }


def iter_tfrecord_examples(tfrecord_path: str, max_records: Optional[int] = None) -> Iterator[Dict[str, np.ndarray]]:
    for idx, raw_record in enumerate(_iter_tfrecord_records(tfrecord_path)):
        if max_records is not None and idx >= max_records:
            break
        yield parse_tfexample(raw_record)


def _require(data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required Waymo feature: {key}")
    return data[key]


def _reshape(data: Dict[str, np.ndarray], key: str, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
    value = _require(data, key)
    return np.asarray(value, dtype=dtype).reshape(shape)


def _maybe_reshape(
    data: Dict[str, np.ndarray],
    key: str,
    shape: Tuple[int, ...],
    dtype=np.float32,
    default: float = 0.0,
) -> np.ndarray:
    if key not in data:
        return np.full(shape, default, dtype=dtype)
    return np.asarray(data[key], dtype=dtype).reshape(shape)


def _temporal_agent_feature(
    data: Dict[str, np.ndarray],
    name: str,
    dtype=np.float32,
    default: float = 0.0,
) -> np.ndarray:
    past = _maybe_reshape(data, f"state/past/{name}", (N_AGENTS_WAYMO, N_PAST), dtype=dtype, default=default)
    current = _maybe_reshape(data, f"state/current/{name}", (N_AGENTS_WAYMO, N_CURRENT), dtype=dtype, default=default)
    future = _maybe_reshape(data, f"state/future/{name}", (N_AGENTS_WAYMO, N_FUTURE), dtype=dtype, default=default)
    return np.concatenate([past, current, future], axis=1)


def _temporal_light_feature(
    data: Dict[str, np.ndarray],
    name: str,
    dtype=np.float32,
    default: float = 0.0,
) -> np.ndarray:
    past = _maybe_reshape(data, f"traffic_light_state/past/{name}", (N_PAST, N_LIGHTS_WAYMO), dtype=dtype, default=default)
    current = _maybe_reshape(data, f"traffic_light_state/current/{name}", (N_LIGHTS_WAYMO,), dtype=dtype, default=default)
    future = _maybe_reshape(data, f"traffic_light_state/future/{name}", (N_FUTURE, N_LIGHTS_WAYMO), dtype=dtype, default=default)
    return np.concatenate([past, current[None, :], future], axis=0)


def _rotate_xy(xy: np.ndarray, heading: float) -> np.ndarray:
    """World xy offsets -> ego frame using ego heading as +x."""
    c = np.cos(heading)
    s = np.sin(heading)
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = c * xy[..., 0] + s * xy[..., 1]
    out[..., 1] = -s * xy[..., 0] + c * xy[..., 1]
    return out


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _safe_speed(speed: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    if np.any(speed):
        return speed
    return np.sqrt(vx * vx + vy * vy).astype(np.float32)


def _build_agent_arrays(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    x = _temporal_agent_feature(data, "x")
    y = _temporal_agent_feature(data, "y")
    vx = _temporal_agent_feature(data, "velocity_x")
    vy = _temporal_agent_feature(data, "velocity_y")
    speed = _safe_speed(_temporal_agent_feature(data, "speed"), vx, vy)
    yaw = _temporal_agent_feature(data, "bbox_yaw")
    valid = _temporal_agent_feature(data, "valid", dtype=np.int64).astype(bool)

    agent_type = _maybe_reshape(data, "state/type", (N_AGENTS_WAYMO,), dtype=np.int64, default=0)
    agent_id = _maybe_reshape(data, "state/id", (N_AGENTS_WAYMO,), dtype=np.int64, default=-1)
    is_sdc = _maybe_reshape(data, "state/is_sdc", (N_AGENTS_WAYMO,), dtype=np.int64, default=0).astype(bool)
    objects_of_interest = _maybe_reshape(
        data, "state/objects_of_interest", (N_AGENTS_WAYMO,), dtype=np.int64, default=0
    ).astype(bool)
    tracks_to_predict = _maybe_reshape(
        data, "state/tracks_to_predict", (N_AGENTS_WAYMO,), dtype=np.int64, default=0
    ).astype(bool)

    return {
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "yaw": yaw,
        "valid": valid,
        "type": agent_type,
        "id": agent_id,
        "is_sdc": is_sdc,
        "objects_of_interest": objects_of_interest,
        "tracks_to_predict": tracks_to_predict,
    }


def _select_agents(agent: Dict[str, np.ndarray], cfg: WaymoVectorConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = agent["id"]
    types = agent["type"]
    valid = agent["valid"]
    is_sdc = agent["is_sdc"]

    sdc_candidates = np.flatnonzero(is_sdc)
    if len(sdc_candidates) > 0:
        ego_idx = int(sdc_candidates[0])
    else:
        valid_counts = valid.sum(axis=1)
        ego_idx = int(np.argmax(valid_counts))

    if cfg.use_all_timesteps_for_selection:
        time_mask = np.ones((N_STEPS,), dtype=bool)
    else:
        time_mask = np.zeros((N_STEPS,), dtype=bool)
        time_mask[: CURRENT_IDX + 1] = True

    ego_xy = np.stack([agent["x"][ego_idx], agent["y"][ego_idx]], axis=-1)
    ego_valid = valid[ego_idx] & time_mask
    xy = np.stack([agent["x"], agent["y"]], axis=-1)
    pair_valid = valid & ego_valid[None, :]
    dist = np.linalg.norm(xy - ego_xy[None, :, :], axis=-1)
    dist = np.where(pair_valid, dist, np.inf)
    min_dist = dist.min(axis=1)

    objects_of_interest = agent.get("objects_of_interest", np.zeros_like(ids, dtype=bool))
    tracks_to_predict = agent.get("tracks_to_predict", np.zeros_like(ids, dtype=bool))
    usable = (ids >= 0) & (types >= 0) & valid.any(axis=1)
    near = usable & (
        (min_dist <= cfg.agent_distance_threshold)
        | objects_of_interest
        | tracks_to_predict
    )
    near[ego_idx] = True

    candidates = np.flatnonzero(near)

    def priority(i: int) -> Tuple[int, int, int, float, int]:
        is_ego = i == ego_idx
        is_ooi = bool(objects_of_interest[i])
        is_predict = bool(tracks_to_predict[i])
        if cfg.prioritize_objects_of_interest:
            return (0 if is_ego else 1, 0 if is_ooi else 1, 0 if is_predict else 1, float(min_dist[i]), int(i))
        return (0 if is_ego else 1, 1, 1, float(min_dist[i]), int(i))

    candidates = sorted(candidates.tolist(), key=priority)
    selected = candidates[: cfg.num_agents]

    if ego_idx not in selected:
        selected = [ego_idx] + selected[: cfg.num_agents - 1]

    selected_idx = np.full((cfg.num_agents,), -1, dtype=np.int64)
    selected_idx[: len(selected)] = np.asarray(selected, dtype=np.int64)
    agent_mask = selected_idx >= 0
    selected_ids = np.full((cfg.num_agents,), -1, dtype=np.int64)
    selected_ids[agent_mask] = ids[selected_idx[agent_mask]]

    return selected_idx, selected_ids, agent_mask


def _selection_time_mask(cfg: WaymoVectorConfig) -> np.ndarray:
    if cfg.use_all_timesteps_for_selection:
        return np.ones((N_STEPS,), dtype=bool)
    time_mask = np.zeros((N_STEPS,), dtype=bool)
    time_mask[: CURRENT_IDX + 1] = True
    return time_mask


def _select_agents_around_focus(
    agent: Dict[str, np.ndarray],
    cfg: WaymoVectorConfig,
    focus_idx: int,
    priority_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = agent["id"]
    types = agent["type"]
    valid = agent["valid"]
    objects_of_interest = agent.get("objects_of_interest", np.zeros_like(ids, dtype=bool))
    tracks_to_predict = agent.get("tracks_to_predict", np.zeros_like(ids, dtype=bool))

    focus_idx = int(focus_idx)
    priority_set = set(int(i) for i in (priority_indices or []) if 0 <= int(i) < len(ids))
    priority_set.add(focus_idx)

    time_mask = _selection_time_mask(cfg)
    focus_xy = np.stack([agent["x"][focus_idx], agent["y"][focus_idx]], axis=-1)
    focus_valid = valid[focus_idx] & time_mask
    xy = np.stack([agent["x"], agent["y"]], axis=-1)
    pair_valid = valid & focus_valid[None, :]
    dist = np.linalg.norm(xy - focus_xy[None, :, :], axis=-1)
    dist = np.where(pair_valid, dist, np.inf)
    min_dist = dist.min(axis=1)

    usable = (ids >= 0) & (types >= 0) & valid.any(axis=1)
    keep = usable & (
        (min_dist <= cfg.agent_distance_threshold)
        | objects_of_interest
        | tracks_to_predict
        | np.asarray([i in priority_set for i in range(len(ids))], dtype=bool)
    )
    keep[focus_idx] = True

    candidates = np.flatnonzero(keep)

    def priority(i: int) -> Tuple[int, int, int, int, float, int]:
        return (
            0 if i == focus_idx else 1,
            0 if i in priority_set and i != focus_idx else 1,
            0 if bool(objects_of_interest[i]) else 1,
            0 if bool(tracks_to_predict[i]) else 1,
            float(min_dist[i]),
            int(i),
        )

    selected = sorted(candidates.tolist(), key=priority)[: cfg.num_agents]
    if focus_idx not in selected:
        selected = [focus_idx] + selected[: cfg.num_agents - 1]

    selected_idx = np.full((cfg.num_agents,), -1, dtype=np.int64)
    selected_idx[: len(selected)] = np.asarray(selected, dtype=np.int64)
    agent_mask = selected_idx >= 0
    selected_ids = np.full((cfg.num_agents,), -1, dtype=np.int64)
    selected_ids[agent_mask] = ids[selected_idx[agent_mask]]
    return selected_idx, selected_ids, agent_mask


def scenario_has_required_objects_of_interest(
    data: Dict[str, np.ndarray],
    cfg: WaymoVectorConfig = WaymoVectorConfig(),
) -> bool:
    if not cfg.require_objects_of_interest and not cfg.require_sdc_object_of_interest:
        return True
    objects_of_interest = _maybe_reshape(
        data, "state/objects_of_interest", (N_AGENTS_WAYMO,), dtype=np.int64, default=0
    ).astype(bool)
    valid_any = (
        _temporal_agent_feature(data, "valid", dtype=np.int64).astype(bool).any(axis=1)
    )
    if cfg.require_sdc_object_of_interest:
        is_sdc = _maybe_reshape(data, "state/is_sdc", (N_AGENTS_WAYMO,), dtype=np.int64, default=0).astype(bool)
        sdc_ooi = bool((objects_of_interest & is_sdc & valid_any).any())
        if not sdc_ooi:
            return False
    usable_ooi = objects_of_interest & valid_any
    return int(usable_ooi.sum()) >= int(cfg.min_objects_of_interest)


def _build_selected_agent_features(
    agent: Dict[str, np.ndarray],
    selected_idx: np.ndarray,
    agent_mask: np.ndarray,
    origin_xy: np.ndarray,
    ego_heading: float,
    normalize: bool,
) -> np.ndarray:
    # Feature order: x, y, speed, vx, vy, valid, yaw, type
    out = np.zeros((len(selected_idx), N_STEPS, 8), dtype=np.float32)
    valid_slots = np.flatnonzero(agent_mask)
    src = selected_idx[valid_slots]

    xy = np.stack([agent["x"][src], agent["y"][src]], axis=-1).astype(np.float32)
    vel = np.stack([agent["vx"][src], agent["vy"][src]], axis=-1).astype(np.float32)
    yaw = agent["yaw"][src].astype(np.float32)
    if normalize:
        xy = _rotate_xy(xy - origin_xy.reshape(1, 1, 2), ego_heading)
        vel = _rotate_xy(vel, ego_heading)
        yaw = _wrap_angle(yaw - ego_heading)

    out[valid_slots, :, 0:2] = xy
    out[valid_slots, :, 2] = agent["speed"][src]
    out[valid_slots, :, 3:5] = vel
    out[valid_slots, :, 5] = agent["valid"][src].astype(np.float32)
    out[valid_slots, :, 6] = yaw
    out[valid_slots, :, 7] = agent["type"][src, None].astype(np.float32)

    # Zero invalid timesteps so losses can mask cleanly.
    step_valid = out[:, :, 5:6]
    out[:, :, 0:5] *= step_valid
    out[:, :, 6:7] *= step_valid
    return out


def _build_map_polylines(
    data: Dict[str, np.ndarray],
    ego_xy_world: np.ndarray,
    ego_valid: np.ndarray,
    origin_xy: np.ndarray,
    ego_heading: float,
    cfg: WaymoVectorConfig,
    crop_xy_world: Optional[np.ndarray] = None,
    crop_valid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = _maybe_reshape(data, "roadgraph_samples/xyz", (N_ROADGRAPH_SAMPLES, 3), dtype=np.float32, default=0.0)
    direction = _maybe_reshape(data, "roadgraph_samples/dir", (N_ROADGRAPH_SAMPLES, 3), dtype=np.float32, default=0.0)
    rg_type = _maybe_reshape(data, "roadgraph_samples/type", (N_ROADGRAPH_SAMPLES,), dtype=np.int64, default=0)
    rg_valid = _maybe_reshape(data, "roadgraph_samples/valid", (N_ROADGRAPH_SAMPLES,), dtype=np.int64, default=1).astype(bool)
    if "roadgraph_samples/id" in data:
        rg_id = np.asarray(data["roadgraph_samples/id"], dtype=np.int64).reshape((N_ROADGRAPH_SAMPLES,))
    else:
        rg_id = np.arange(N_ROADGRAPH_SAMPLES, dtype=np.int64)

    point_valid = rg_valid & (rg_id >= 0)
    valid_idx = np.flatnonzero(point_valid)
    if len(valid_idx) == 0:
        features = np.zeros((cfg.max_map_polylines, cfg.max_points_per_polyline, 6), dtype=np.float32)
        mask = np.zeros((cfg.max_map_polylines, cfg.max_points_per_polyline), dtype=bool)
        ids = np.full((cfg.max_map_polylines,), -1, dtype=np.int64)
        return features, mask, ids

    points_xy = xyz[valid_idx, :2]
    if crop_xy_world is not None and crop_valid is not None:
        crop_xy = np.asarray(crop_xy_world, dtype=np.float32).reshape(-1, 2)
        crop_mask = np.asarray(crop_valid, dtype=bool).reshape(-1)
        ego_xy_valid = crop_xy[crop_mask]
    else:
        ego_xy_valid = ego_xy_world[ego_valid]
    if len(ego_xy_valid) == 0:
        ego_xy_valid = ego_xy_world[[CURRENT_IDX]]

    # 30000 x 91 is small enough and keeps the cropping rule easy to inspect.
    dists = np.linalg.norm(points_xy[:, None, :] - ego_xy_valid[None, :, :], axis=-1)
    min_point_dist = dists.min(axis=1)
    close_point = min_point_dist <= cfg.map_distance_threshold
    close_ids = set(rg_id[valid_idx[close_point]].tolist())

    chunks: List[Tuple[float, int, np.ndarray]] = []
    for map_id in close_ids:
        idx = np.flatnonzero(point_valid & (rg_id == map_id))
        if len(idx) == 0:
            continue
        chunk_dist = float(np.min(np.linalg.norm(xyz[idx, None, :2] - ego_xy_valid[None, :, :], axis=-1)))
        for start in range(0, len(idx), cfg.max_points_per_polyline):
            chunk = idx[start : start + cfg.max_points_per_polyline]
            if len(chunk) > 0:
                chunks.append((chunk_dist, int(map_id), chunk))

    chunks.sort(key=lambda item: (item[0], item[1]))
    chunks = chunks[: cfg.max_map_polylines]

    features = np.zeros((cfg.max_map_polylines, cfg.max_points_per_polyline, 6), dtype=np.float32)
    mask = np.zeros((cfg.max_map_polylines, cfg.max_points_per_polyline), dtype=bool)
    ids = np.full((cfg.max_map_polylines,), -1, dtype=np.int64)

    for out_idx, (_, map_id, chunk) in enumerate(chunks):
        n = len(chunk)
        xy = xyz[chunk, :2].astype(np.float32)
        dirs = direction[chunk, :2].astype(np.float32)
        if cfg.normalize_to_ego:
            xy = _rotate_xy(xy - origin_xy.reshape(1, 2), ego_heading)
            dirs = _rotate_xy(dirs, ego_heading)

        features[out_idx, :n, 0:2] = xy
        features[out_idx, :n, 2:4] = dirs
        features[out_idx, :n, 4] = rg_type[chunk].astype(np.float32)
        features[out_idx, :n, 5] = 1.0
        mask[out_idx, :n] = True
        ids[out_idx] = map_id

    return features, mask, ids


def _build_lights(
    data: Dict[str, np.ndarray],
    origin_xy: np.ndarray,
    ego_heading: float,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _temporal_light_feature(data, "x")
    y = _temporal_light_feature(data, "y")
    state = _temporal_light_feature(data, "state", dtype=np.int64)
    light_id = _temporal_light_feature(data, "id", dtype=np.int64, default=-1)
    valid = _temporal_light_feature(data, "valid", dtype=np.int64, default=-1)

    if (valid >= 0).any():
        light_mask = valid.astype(bool)
    else:
        light_mask = (state > 0) & np.isfinite(x) & np.isfinite(y)

    xy = np.stack([x, y], axis=-1).astype(np.float32)
    if normalize:
        xy = _rotate_xy(xy - origin_xy.reshape(1, 1, 2), ego_heading)

    lights = np.zeros((N_STEPS, N_LIGHTS_WAYMO, 4), dtype=np.float32)
    lights[:, :, 0:2] = xy
    lights[:, :, 2] = state.astype(np.float32)
    lights[:, :, 3] = light_mask.astype(np.float32)
    lights[:, :, 0:2] *= lights[:, :, 3:4]
    return lights, light_mask, light_id.astype(np.int64)


def filter_scenario(data: Dict[str, np.ndarray], cfg: WaymoVectorConfig = WaymoVectorConfig()) -> Dict[str, np.ndarray]:
    agent = _build_agent_arrays(data)
    selected_idx, selected_ids, agent_mask = _select_agents(agent, cfg)

    ego_src_idx = int(selected_idx[0]) if selected_idx[0] >= 0 else int(np.argmax(agent["valid"].sum(axis=1)))
    ego_xy_world = np.stack([agent["x"][ego_src_idx], agent["y"][ego_src_idx]], axis=-1).astype(np.float32)
    ego_valid = agent["valid"][ego_src_idx]
    origin_xy = ego_xy_world[CURRENT_IDX].astype(np.float32)
    ego_heading = float(agent["yaw"][ego_src_idx, CURRENT_IDX])

    agents = _build_selected_agent_features(
        agent=agent,
        selected_idx=selected_idx,
        agent_mask=agent_mask,
        origin_xy=origin_xy,
        ego_heading=ego_heading,
        normalize=cfg.normalize_to_ego,
    )

    map_polylines, map_mask, map_ids = _build_map_polylines(
        data=data,
        ego_xy_world=ego_xy_world,
        ego_valid=ego_valid,
        origin_xy=origin_xy,
        ego_heading=ego_heading,
        cfg=cfg,
    )

    lights, light_mask, light_ids = _build_lights(
        data=data,
        origin_xy=origin_xy,
        ego_heading=ego_heading,
        normalize=cfg.normalize_to_ego,
    )

    scenario_id = data.get("scenario/id", "")
    if not isinstance(scenario_id, str):
        scenario_id = str(scenario_id)

    return {
        "scenario_id": np.asarray(scenario_id),
        "agents": agents,
        "agent_mask": agent_mask,
        "agent_ids": selected_ids,
        "agent_src_indices": selected_idx,
        "agent_objects_of_interest": agent["objects_of_interest"][np.maximum(selected_idx, 0)] & agent_mask,
        "agent_tracks_to_predict": agent["tracks_to_predict"][np.maximum(selected_idx, 0)] & agent_mask,
        "num_raw_objects_of_interest": np.asarray(int((agent["objects_of_interest"] & agent["valid"].any(axis=1)).sum())),
        "map_polylines": map_polylines,
        "map_mask": map_mask,
        "map_ids": map_ids,
        "lights": lights,
        "light_mask": light_mask,
        "light_ids": light_ids,
        "ego_origin_xy": origin_xy,
        "ego_heading": np.asarray(ego_heading, dtype=np.float32),
        "config_json": np.asarray(json.dumps(asdict(cfg), sort_keys=True)),
    }


def filter_scenario_around_focus(
    data: Dict[str, np.ndarray],
    focus_src_index: int,
    cfg: WaymoVectorConfig = WaymoVectorConfig(),
    priority_src_indices: Optional[List[int]] = None,
    map_crop_src_indices: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Filter a scenario using an arbitrary focus track as slot 0 and origin.

    This is intended for OOI-centered tokenizer pretraining. It keeps the raw
    SDC metadata, but the output tensor's slot 0 and coordinate frame are the
    requested focus track.
    """
    agent = _build_agent_arrays(data)
    focus_src_index = int(focus_src_index)
    if not (0 <= focus_src_index < N_AGENTS_WAYMO):
        raise IndexError(f"focus_src_index out of range: {focus_src_index}")
    if not bool(agent["valid"][focus_src_index, CURRENT_IDX]):
        raise ValueError(f"focus_src_index {focus_src_index} is not valid at current timestep")

    selected_idx, selected_ids, agent_mask = _select_agents_around_focus(
        agent=agent,
        cfg=cfg,
        focus_idx=focus_src_index,
        priority_indices=priority_src_indices,
    )

    focus_xy_world = np.stack([agent["x"][focus_src_index], agent["y"][focus_src_index]], axis=-1).astype(np.float32)
    focus_valid = agent["valid"][focus_src_index]
    origin_xy = focus_xy_world[CURRENT_IDX].astype(np.float32)
    focus_heading = float(agent["yaw"][focus_src_index, CURRENT_IDX])

    agents = _build_selected_agent_features(
        agent=agent,
        selected_idx=selected_idx,
        agent_mask=agent_mask,
        origin_xy=origin_xy,
        ego_heading=focus_heading,
        normalize=cfg.normalize_to_ego,
    )

    crop_indices = [focus_src_index]
    for src_idx in map_crop_src_indices or priority_src_indices or []:
        src_idx = int(src_idx)
        if 0 <= src_idx < N_AGENTS_WAYMO and src_idx not in crop_indices:
            crop_indices.append(src_idx)
    crop_xy_world = np.stack(
        [np.stack([agent["x"][idx], agent["y"][idx]], axis=-1) for idx in crop_indices],
        axis=0,
    ).astype(np.float32)
    crop_valid = np.stack([agent["valid"][idx] for idx in crop_indices], axis=0)

    map_polylines, map_mask, map_ids = _build_map_polylines(
        data=data,
        ego_xy_world=focus_xy_world,
        ego_valid=focus_valid,
        origin_xy=origin_xy,
        ego_heading=focus_heading,
        cfg=cfg,
        crop_xy_world=crop_xy_world,
        crop_valid=crop_valid,
    )

    lights, light_mask, light_ids = _build_lights(
        data=data,
        origin_xy=origin_xy,
        ego_heading=focus_heading,
        normalize=cfg.normalize_to_ego,
    )

    scenario_id = data.get("scenario/id", "")
    if not isinstance(scenario_id, str):
        scenario_id = str(scenario_id)

    sdc_candidates = np.flatnonzero(agent["is_sdc"])
    original_sdc_src_index = int(sdc_candidates[0]) if len(sdc_candidates) else int(np.argmax(agent["valid"].sum(axis=1)))
    ooi_src_indices = np.flatnonzero(agent["objects_of_interest"] & agent["valid"].any(axis=1)).astype(np.int64)
    ttp_src_indices = np.flatnonzero(agent["tracks_to_predict"] & agent["valid"].any(axis=1)).astype(np.int64)

    return {
        "scenario_id": np.asarray(scenario_id),
        "agents": agents,
        "agent_mask": agent_mask,
        "agent_ids": selected_ids,
        "agent_src_indices": selected_idx,
        "agent_objects_of_interest": agent["objects_of_interest"][np.maximum(selected_idx, 0)] & agent_mask,
        "agent_tracks_to_predict": agent["tracks_to_predict"][np.maximum(selected_idx, 0)] & agent_mask,
        "focus_src_index": np.asarray(focus_src_index, dtype=np.int64),
        "focus_track_id": np.asarray(int(agent["id"][focus_src_index]), dtype=np.int64),
        "focus_type": np.asarray(int(agent["type"][focus_src_index]), dtype=np.int64),
        "original_sdc_src_index": np.asarray(original_sdc_src_index, dtype=np.int64),
        "original_sdc_track_id": np.asarray(int(agent["id"][original_sdc_src_index]), dtype=np.int64),
        "ooi_src_indices": ooi_src_indices,
        "ooi_track_ids": agent["id"][ooi_src_indices].astype(np.int64),
        "tracks_to_predict_src_indices": ttp_src_indices,
        "tracks_to_predict_ids": agent["id"][ttp_src_indices].astype(np.int64),
        "num_raw_objects_of_interest": np.asarray(int(len(ooi_src_indices)), dtype=np.int64),
        "map_crop_src_indices": np.asarray(crop_indices, dtype=np.int64),
        "map_polylines": map_polylines,
        "map_mask": map_mask,
        "map_ids": map_ids,
        "lights": lights,
        "light_mask": light_mask,
        "light_ids": light_ids,
        "ego_origin_xy": origin_xy,
        "ego_heading": np.asarray(focus_heading, dtype=np.float32),
        "center_is_focus": np.asarray(True),
        "config_json": np.asarray(json.dumps(asdict(cfg), sort_keys=True)),
    }


def iter_filtered_scenarios(
    tfrecord_path: str,
    cfg: WaymoVectorConfig = WaymoVectorConfig(),
    max_records: Optional[int] = None,
) -> Iterator[Dict[str, np.ndarray]]:
    for data in iter_tfrecord_examples(tfrecord_path, max_records=max_records):
        if not scenario_has_required_objects_of_interest(data, cfg=cfg):
            continue
        yield filter_scenario(data, cfg)


def save_filtered_tfrecord(
    tfrecord_path: str,
    output_dir: str,
    cfg: WaymoVectorConfig = WaymoVectorConfig(),
    max_records: Optional[int] = None,
) -> List[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    stem = Path(tfrecord_path).name.replace(".", "_")

    saved_idx = 0
    for item in iter_filtered_scenarios(tfrecord_path, cfg=cfg, max_records=max_records):
        scenario_id = str(item["scenario_id"])
        safe_id = scenario_id if scenario_id else f"{stem}_{saved_idx:06d}"
        out_path = out_dir / f"{safe_id}.npz"
        np.savez_compressed(out_path, **item)
        written.append(out_path)
        saved_idx += 1
    return written


def summarize_filtered(item: Dict[str, np.ndarray]) -> Dict[str, object]:
    agents = item["agents"]
    map_mask = item["map_mask"]
    light_mask = item["light_mask"]
    agent_mask = item["agent_mask"]
    valid_agent_steps = agents[:, :, 5] > 0.5

    return {
        "scenario_id": str(item["scenario_id"]),
        "agents_shape": list(agents.shape),
        "agent_mask_shape": list(agent_mask.shape),
        "num_agent_slots_valid": int(agent_mask.sum()),
        "agent_ids": item["agent_ids"][agent_mask].astype(int).tolist(),
        "map_polylines_shape": list(item["map_polylines"].shape),
        "map_mask_shape": list(map_mask.shape),
        "num_map_polylines_valid": int((map_mask.sum(axis=1) > 0).sum()),
        "num_map_points_valid": int(map_mask.sum()),
        "lights_shape": list(item["lights"].shape),
        "light_mask_shape": list(light_mask.shape),
        "num_light_steps_valid": int(light_mask.sum()),
        "ego_origin_xy": item["ego_origin_xy"].astype(float).tolist(),
        "ego_heading": float(item["ego_heading"]),
        "agent_xy_min": agents[:, :, 0:2][valid_agent_steps].min(axis=0).astype(float).tolist()
        if valid_agent_steps.any()
        else None,
        "agent_xy_max": agents[:, :, 0:2][valid_agent_steps].max(axis=0).astype(float).tolist()
        if valid_agent_steps.any()
        else None,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter Waymo tf.Example scenarios into vector-tokenizer tensors.")
    p.add_argument("tfrecord", type=str)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_records", type=int, default=1)
    p.add_argument("--num_agents", type=int, default=32)
    p.add_argument("--agent_distance_threshold", type=float, default=80.0)
    p.add_argument("--map_distance_threshold", type=float, default=100.0)
    p.add_argument("--max_map_polylines", type=int, default=256)
    p.add_argument("--max_points_per_polyline", type=int, default=20)
    p.add_argument("--history_only_selection", action="store_true")
    p.add_argument("--no_ego_normalize", action="store_true")
    p.add_argument("--require_objects_of_interest", action="store_true")
    p.add_argument("--min_objects_of_interest", type=int, default=1)
    p.add_argument("--require_sdc_object_of_interest", action="store_true")
    p.add_argument("--no_prioritize_objects_of_interest", action="store_true")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    cfg = WaymoVectorConfig(
        num_agents=args.num_agents,
        agent_distance_threshold=args.agent_distance_threshold,
        map_distance_threshold=args.map_distance_threshold,
        max_map_polylines=args.max_map_polylines,
        max_points_per_polyline=args.max_points_per_polyline,
        use_all_timesteps_for_selection=not args.history_only_selection,
        normalize_to_ego=not args.no_ego_normalize,
        require_objects_of_interest=args.require_objects_of_interest,
        min_objects_of_interest=args.min_objects_of_interest,
        require_sdc_object_of_interest=args.require_sdc_object_of_interest,
        prioritize_objects_of_interest=not args.no_prioritize_objects_of_interest,
    )

    if args.output_dir is not None:
        written = save_filtered_tfrecord(args.tfrecord, args.output_dir, cfg=cfg, max_records=args.max_records)
        print(json.dumps({"written": [str(p) for p in written], "count": len(written)}, indent=2))
        return

    for item in iter_filtered_scenarios(args.tfrecord, cfg=cfg, max_records=args.max_records):
        print(json.dumps(summarize_filtered(item), indent=2))


if __name__ == "__main__":
    main()
