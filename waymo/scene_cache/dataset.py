"""Load the whole-scene WOMD cache into the direct-action-flow batch format.

The cache (``/p/yufeng/waymo_scene_prep``) stores every track, the whole map and
the whole lane graph per scenario, independent of any ego choice. This module
picks a center agent, rotates everything into its frame and emits exactly the
fields that ``ConnectivityDataset`` + ``collate_connectivity`` produce, so the
existing models, ``prepare_batch`` and ``rollout_receding_horizon`` run unchanged.

Agent slots: slot 0 is the center agent (SDC by default), then the selected
agents sorted by distance to it. Agent features follow
``waymo_vector_filter._build_selected_agent_features``:
x, y, speed, vx, vy, valid, yaw, type (invalid steps zeroed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CACHE_FORMAT_VERSION = 1
MAP_FIELDS = ("map_polylines", "map_mask", "map_ids", "map_is_lane", "map_stop_sign",
              "map_stop_point", "lights", "light_mask", "light_ids")


def _rotate(xy, heading):
    c, s = np.cos(heading), np.sin(heading)
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = c * xy[..., 0] + s * xy[..., 1]
    out[..., 1] = -s * xy[..., 0] + c * xy[..., 1]
    return out


def _wrap(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# Last resort only; no track in WOMD training/validation needs it (see
# /p/yufeng/waymo_scene_prep/data/invalid_lengths_summary.json).
DEFAULT_LENGTH_M = {1: 4.5, 3: 1.8}


def _agent_lengths(sel, valid, types, current):
    """Current-frame box length per selected agent, as the type-aware executor needs.

    WOMD occasionally has a non-positive current length on a valid track (2 of
    ~16M training tracks: short noisy vehicle tracks with length < 0); those
    use the median positive length over the track's valid frames instead.
    """
    lengths = sel[:, current, 6].astype(np.float32)
    for j in np.flatnonzero(~(np.isfinite(lengths) & (lengths > 0))):
        good = valid[j] & np.isfinite(sel[j, :, 6]) & (sel[j, :, 6] > 0)
        if good.any():
            lengths[j] = np.median(sel[j, good, 6])
        elif int(types[j]) in DEFAULT_LENGTH_M:
            lengths[j] = DEFAULT_LENGTH_M[int(types[j])]
    return lengths


def read_manifest(path, min_current_valid=0, max_current_valid=0):
    rows = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            n = row["num_current_valid"]
            if n >= min_current_valid and (max_current_valid <= 0 or n <= max_current_valid):
                rows.append(row)
    return rows


class SceneCacheDataset(Dataset):
    """One item per scenario, centered on ``center``.

    Args:
        rows: manifest rows (``read_manifest``) or a list of NPZ paths.
        center: ``"sdc"``, ``"random_ttp"`` (random tracks_to_predict agent) or
            ``"mixed"`` (SDC with probability ``p_sdc``, else random_ttp).
        p_sdc: SDC probability for ``center="mixed"``.
        loss_agents: size of ``agent_loss_mask``. Center, SDC,
            tracks_to_predict and objects_of_interest are always included,
            the rest is filled with random agents; 0 = every agent.
        agent_radius_m: keep current-valid agents within this distance of the
            center; 0 keeps all of them (WOSAC evaluation).
        max_agents: cap after distance sorting; 0 = no cap.
        map_crop_m: keep map segments within this distance of any selected
            agent's current position; 0 keeps the whole Scenario map.
    """

    def __init__(self, rows, *, center="sdc", p_sdc=0.5, agent_radius_m=0.0, max_agents=0,
                 map_crop_m=0.0, loss_agents=0, seed=0):
        self.paths = [r["file"] if isinstance(r, dict) else str(r) for r in rows]
        if not self.paths:
            raise ValueError("Empty scene list")
        if center not in ("sdc", "random_ttp", "mixed"):
            raise ValueError(f"Unknown center {center!r}")
        self.center = center
        self.p_sdc = float(p_sdc)
        self.loss_agents = int(loss_agents)
        self.agent_radius_m = float(agent_radius_m)
        self.max_agents = int(max_agents)
        self.map_crop_m = float(map_crop_m)
        self.seed = int(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            d = {key: data[key] for key in data.files}
        if int(d["format_version"]) != CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported cache version in {self.paths[index]}")
        # torch.initial_seed() differs per DataLoader worker and per epoch (with
        # non-persistent workers), so each sample draws fresh randomness.
        self.rng = np.random.default_rng([self.seed, torch.initial_seed() % 2**63, index])
        item = self._build(d)
        item["path"] = self.paths[index]
        return item

    def _choose_center(self, d, current):
        sdc = int(d["sdc_track_index"])
        if self.center == "random_ttp" or (self.center == "mixed" and self.rng.random() >= self.p_sdc):
            ttp = [int(i) for i in d["tracks_to_predict"] if d["tracks"][int(i), current, 9] > 0.5]
            if ttp:
                return int(self.rng.choice(ttp))
        return sdc

    def _build(self, d):
        tracks = d["tracks"]  # (N, T, 10): x y z heading vx vy length width height valid
        current = int(d["current_time_index"])
        center = self._choose_center(d, current)
        origin = tracks[center, current, :2].astype(np.float64)
        heading = float(tracks[center, current, 3])

        alive = tracks[:, current, 9] > 0.5
        distance = np.linalg.norm(tracks[:, current, :2] - origin, axis=-1)
        keep = alive.copy()
        if self.agent_radius_m > 0:
            keep &= distance <= self.agent_radius_m
        keep[center] = True
        others = [i for i in np.argsort(distance, kind="stable") if keep[i] and i != center]
        order = np.asarray([center] + others, np.int64)
        if self.max_agents > 0:
            order = order[: self.max_agents]

        sel = tracks[order]
        valid = sel[..., 9] > 0.5
        xy = _rotate(sel[..., :2] - origin, heading)
        vel = _rotate(sel[..., 4:6], heading)
        agents = np.zeros(sel.shape[:2] + (8,), np.float32)
        agents[..., 0:2] = xy
        agents[..., 2] = np.linalg.norm(sel[..., 4:6], axis=-1)
        agents[..., 3:5] = vel
        agents[..., 5] = valid
        agents[..., 6] = _wrap(sel[..., 3] - heading)
        agents[..., 7] = d["track_types"][order, None]
        agents[..., 0:5] *= agents[..., 5:6]
        agents[..., 6] *= agents[..., 5]

        segments = d["map_segments"]
        keep_map = np.ones(len(segments), bool)
        if self.map_crop_m > 0:
            centers = segments[..., :2].mean(1)
            agent_xy = sel[:, current, :2]
            keep_map = np.zeros(len(segments), bool)
            for start in range(0, len(segments), 1024):
                block = centers[start:start + 1024, None] - agent_xy[None]
                keep_map[start:start + 1024] = (np.linalg.norm(block, axis=-1) <= self.map_crop_m).any(1)
            if not keep_map.any():
                keep_map[np.argmin(np.linalg.norm(centers - origin, axis=-1))] = True
        polylines = segments[keep_map].copy()
        polylines[..., 0:2] = _rotate(polylines[..., 0:2] - origin, heading)
        polylines[..., 2:4] = _rotate(polylines[..., 2:4], heading)

        # Remap lane-graph indices onto the kept segments; dropped targets -> -1.
        new_index = np.full(len(segments), -1, np.int64)
        new_index[keep_map] = np.arange(int(keep_map.sum()))
        edges = d["lane_edges"][keep_map].astype(np.int64)
        edge_types = d["lane_edge_types"][keep_map].astype(np.int64)
        edges = np.where(edges >= 0, new_index[np.maximum(edges, 0)], -1)
        edge_types = np.where(edges >= 0, edge_types, 0)

        stop_point = d["map_stop_point"][keep_map]
        stop_flag = d["map_stop_sign"][keep_map]
        stop_point = np.where(stop_flag[:, None], _rotate(stop_point - origin, heading), 0.0)

        lights = d["lights"].copy()
        light_mask = lights[..., 3] > 0.5
        lights[..., 0:2] = np.where(light_mask[..., None], _rotate(lights[..., 0:2] - origin, heading), 0.0)

        # WOSAC evaluation agents: tracks_to_predict plus the SDC.
        eval_tracks = set(int(i) for i in d["tracks_to_predict"]) | {int(d["sdc_track_index"])}
        eval_mask = np.asarray([int(i) in eval_tracks for i in order], bool)
        is_sdc = order == int(d["sdc_track_index"])

        loss_mask = np.ones(len(order), bool)
        if self.loss_agents > 0 and len(order) > self.loss_agents:
            interest = set(int(x) for x in d["objects_of_interest"])
            must = eval_mask | np.isin(d["track_ids"][order], list(interest))
            must[0] = True  # center agent
            loss_mask = must.copy()
            rest = np.flatnonzero(~must)
            fill = max(0, self.loss_agents - int(must.sum()))
            if fill and len(rest):
                loss_mask[self.rng.choice(rest, size=min(fill, len(rest)), replace=False)] = True

        tensor = torch.from_numpy
        return {
            "agent_lengths": tensor(_agent_lengths(sel, valid, d["track_types"][order], current)),
            "agent_eval_mask": tensor(eval_mask),
            "agent_is_sdc": tensor(is_sdc),
            "agent_loss_mask": tensor(loss_mask),
            "agents": tensor(agents),
            "agent_mask": torch.ones(len(order), dtype=torch.bool),
            "agent_ids": tensor(d["track_ids"][order].astype(np.int64)),
            "agent_src_indices": tensor(order),
            "map_polylines": tensor(polylines.astype(np.float32)),
            "map_mask": tensor(polylines[..., 5] > 0.5),
            "map_ids": tensor(d["map_ids"][keep_map].astype(np.int64)),
            "map_is_lane": tensor(d["map_is_lane"][keep_map]),
            "map_stop_sign": tensor(stop_flag),
            "map_stop_point": tensor(stop_point.astype(np.float32)),
            "lights": tensor(lights.astype(np.float32)),
            "light_mask": tensor(light_mask),
            "light_ids": tensor(d["light_ids"].astype(np.int64)),
            "lane_edges": tensor(edges),
            "lane_edge_types": tensor(edge_types),
            # Global WOMD coordinates of the center agent (same meaning as legacy NPZs).
            "ego_origin_xy": tensor(origin + d["scene_origin_xy"]),
            "ego_heading": torch.tensor(heading, dtype=torch.float32),
            "scene_origin_xy": tensor(d["scene_origin_xy"]),
            "num_current_valid": torch.tensor(int(alive.sum())),
            "map_adaptation": torch.tensor(True),
            "scenario_id": str(d["scenario_id"]),
        }


def _pad_stack(values, axis, fill):
    size = max(v.shape[axis] for v in values)
    out = []
    for v in values:
        shape = list(v.shape)
        shape[axis] = size
        target = v.new_full(shape, fill)
        index = [slice(None)] * v.ndim
        index[axis] = slice(0, v.shape[axis])
        target[tuple(index)] = v
        out.append(target)
    return torch.stack(out)


# (axis to pad, fill). Padded agent slots are masked by ``agent_mask``.
PAD = {
    "agents": (0, 0), "agent_mask": (0, False), "agent_eval_mask": (0, False),
    "agent_is_sdc": (0, False), "agent_loss_mask": (0, False),
    "agent_ids": (0, -1), "agent_lengths": (0, 0),
    "agent_src_indices": (0, -1),
    "map_polylines": (0, 0), "map_mask": (0, False), "map_ids": (0, -1), "map_is_lane": (0, False),
    "map_stop_sign": (0, False), "map_stop_point": (0, 0),
    "lights": (1, 0), "light_mask": (1, False), "light_ids": (1, -1),
}


def collate_scenes(items):
    """Pad agents, map segments, lights and lane edges within the minibatch."""
    batch = {}
    for key, value in items[0].items():
        if not torch.is_tensor(value) or key in ("lane_edges", "lane_edge_types"):
            continue
        values = [item[key] for item in items]
        if key in PAD:
            axis, fill = PAD[key]
            batch[key] = _pad_stack(values, axis, fill)
        else:
            batch[key] = torch.stack(values)
    m = batch["map_ids"].shape[1]
    e = max(item["lane_edges"].shape[1] for item in items)
    for key, fill in (("lane_edges", -1), ("lane_edge_types", 0)):
        out = items[0][key].new_full((len(items), m, e), fill)
        for i, item in enumerate(items):
            v = item[key]
            out[i, : v.shape[0], : v.shape[1]] = v
        batch[key] = out
    batch["scenario_id"] = [item["scenario_id"] for item in items]
    batch["path"] = [item["path"] for item in items]
    return batch
