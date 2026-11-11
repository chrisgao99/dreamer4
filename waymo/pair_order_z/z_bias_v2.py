"""z v2: denser pair passing-order labels + the same z -> agent_attn bias mechanism.

Pair record (P, 8): i, j, enter_i, exit_i, enter_j, exit_j, angle_deg, kind
  kind 0 = crossing / opposing (heading difference >= 30 deg at the closest approach), as in v1
  kind 1 = same direction (merge, lane change, following), new in v2
  v2 also keeps pre-existing pairs (a car already in the overlap before frame 5 goes first).

Timing rule (unchanged from v1, confirmed with the user 2026-11-11): z is on from the first frame,
and turns off (bias 0) once the first car has left the overlap; no lead-time limit. Codes 1-8 are the
v1 codes (z_bias.pair_codes); crossing pairs use them unchanged. Same-direction pairs differ in two
places (plan 8.1 point 3: an unreleased "leader first" stalls the follower):
  * both cars in the shared segment (v1 code 4, unknown): the car in front goes first (code 10;
    "in front" = positive projection on the other car's heading, current pose). Following pairs
    spend most of their time in this state, and the first car has not left yet.
  * gate: on only while close, centre distance <= 20 m and distance / max(v_i, v_j, 5 m/s) <= 2 s
    (so <= 10 m in slow traffic, <= 20 m above 10 m/s); otherwise code 9.
  Both use the current state only (log state at the training anchor, rollout state in the loop).
  Pairs with a car that moves < 2 m over the whole log (parked) are not labelled (make_z_labels_v2).
Directed z (B,N,N): 1 = query goes first, 2 = key goes first, 0 = none (z_bias.z_matrix).
g v2 = g v1 plus one input feature: whether the edge belongs to a same-direction pair.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from waymo.pair_order_z.z_bias import BiasStats, atomic_save_npz, pair_codes, z_matrix  # noqa: F401
from waymo.pair_order_z.z_bias import CODE_NAMES as _V1_CODES

REC = 8
KIND_CROSS, KIND_SAME = 0, 1
SAME_DIR_MAX_DIST_M = 20.0
SAME_DIR_MAX_GAP_S = 2.0
GAP_MIN_SPEED_MPS = 5.0
CODE_NAMES = {**_V1_CODES, 9: "same_dir_far", 10: "same_dir_both_in_front_first"}
ACTIVE_CODES = (1, 2, 3, 10)


class PairLabelsV2:
    def __init__(self, path: str):
        with np.load(path, allow_pickle=False) as d:
            names, offsets, pairs = d["names"], d["offsets"], d["pairs"]
        if pairs.ndim != 2 or pairs.shape[1] != REC:
            raise ValueError(f"{path}: expected (P, {REC}) v2 records, got {pairs.shape}")
        self.path = str(path)
        self.index = {str(n): (int(offsets[k]), int(offsets[k + 1])) for k, n in enumerate(names)}
        self.pairs = pairs.astype(np.int64)
        widest = int(np.diff(offsets).max()) if len(names) else 0
        self.p_max = max(8, int(np.ceil(widest / 8) * 8))

    def has_pairs(self, name: str) -> bool:
        lo, hi = self.index[name]
        return hi > lo

    def padded(self, name: str) -> torch.Tensor:
        lo, hi = self.index[name]
        out = np.full((self.p_max, REC), -1, dtype=np.int64)
        out[: hi - lo] = self.pairs[lo:hi]
        return torch.from_numpy(out)


def with_z_pairs_v2(base_cls, labels: PairLabelsV2):
    class ZPairDatasetV2(base_cls):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item["z_pairs"] = labels.padded(Path(self.paths[index]).name)
            return item

    ZPairDatasetV2.__name__ = f"ZPairV2{base_cls.__name__}"
    return ZPairDatasetV2


def same_dir_close(z_pairs: torch.Tensor, pose: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    """(B,P) bool: the two cars of each pair are within the same-direction gate (current state)."""
    zp = z_pairs.clamp_min(0)
    i, j = zp[..., 0], zp[..., 1]
    xy = torch.nan_to_num(pose[..., 0:2].float(), nan=1e6)
    v = torch.nan_to_num(speed.float(), nan=0.0).abs()
    gi = xy.gather(1, i[..., None].expand(-1, -1, 2))
    gj = xy.gather(1, j[..., None].expand(-1, -1, 2))
    dist = torch.linalg.vector_norm(gi - gj, dim=-1)
    vmax = torch.maximum(v.gather(1, i), v.gather(1, j)).clamp_min(GAP_MIN_SPEED_MPS)
    return (dist <= SAME_DIR_MAX_DIST_M) & (dist / vmax <= SAME_DIR_MAX_GAP_S)


def pair_codes_v2(z_pairs, progress, alive, pose, speed, deviated=None, reversed_order=False):
    codes, priority, yielder = pair_codes(z_pairs[..., :7], progress, alive, deviated, reversed_order)
    same = (z_pairs[..., 7] == KIND_SAME) & (z_pairs[..., 0] >= 0)
    both_in = same & (codes == 4)
    if both_in.any():
        zp = z_pairs.clamp_min(0)
        i, j = zp[..., 0], zp[..., 1]
        pz = torch.nan_to_num(pose.float())
        pi = pz.gather(1, i[..., None].expand(-1, -1, 3))
        pj = pz.gather(1, j[..., None].expand(-1, -1, 3))
        d = pj[..., :2] - pi[..., :2]
        j_ahead = (d[..., 0] * torch.cos(pi[..., 2]) + d[..., 1] * torch.sin(pi[..., 2])) > 0
        front, back = torch.where(j_ahead, j, i), torch.where(j_ahead, i, j)
        if reversed_order:
            front, back = back, front
        codes = torch.where(both_in, torch.full_like(codes, 10), codes)
        priority = torch.where(both_in, front, priority)
        yielder = torch.where(both_in, back, yielder)
    active = (codes == 1) | (codes == 2) | (codes == 3) | (codes == 10)
    far = same & active & ~same_dir_close(z_pairs, pose, speed)
    codes = torch.where(far, torch.full_like(codes, 9), codes)
    priority = torch.where(far, torch.full_like(priority, -1), priority)
    yielder = torch.where(far, torch.full_like(yielder, -1), yielder)
    return codes, priority, yielder


def kind_matrix(z_pairs: torch.Tensor, num_agents: int) -> torch.Tensor:
    """(B,N,N) float: 1 on both directed edges of same-direction pairs."""
    bsz = int(z_pairs.shape[0])
    out = torch.zeros((bsz, num_agents, num_agents), device=z_pairs.device)
    sel = (z_pairs[..., 0] >= 0) & (z_pairs[..., 7] == KIND_SAME)
    if sel.any():
        b = torch.arange(bsz, device=z_pairs.device)[:, None].expand_as(sel)[sel]
        i, j = z_pairs[..., 0][sel], z_pairs[..., 1][sel]
        out[b, i, j] = 1.0
        out[b, j, i] = 1.0
    return out


def training_z_v2(z_pairs, anchors, alive, pose, speed):
    """Log state at the anchor for every car (teacher forcing); pose/speed = history last frame."""
    progress = anchors[:, None].expand(-1, alive.shape[1]).long()
    codes, priority, yielder = pair_codes_v2(z_pairs, progress, alive, pose, speed)
    return z_matrix(priority, yielder, alive.shape[1]), codes, priority, yielder


class OracleZV2:
    """Closed loop: log order, rollout state (projection onto the logged path), same-dir gate on the
    rollout's current poses and speeds."""

    def __init__(self, agents, agent_mask, z_pairs, *, reversed_order=False, max_deviation_m=2.0):
        from waymo.training.world_model.direct_action_flow import agents_to_bntf
        a = agents_to_bntf(agents, agent_mask).float()
        self.log_xy = a[..., 0:2]
        self.log_valid = (a[..., 5] > 0.5) & torch.isfinite(a[..., 0:2]).all(-1) & agent_mask[:, :, None].bool()
        self.z_pairs = z_pairs.long()
        self.reversed_order = bool(reversed_order)
        self.max_deviation_m = float(max_deviation_m)
        self.kind = kind_matrix(self.z_pairs, a.shape[1])

    def step(self, pose, alive, speed):
        dist = torch.linalg.vector_norm(self.log_xy - pose[:, :, None, 0:2].float(), dim=-1)
        dist = torch.nan_to_num(dist, nan=float("inf")).masked_fill(~self.log_valid, float("inf"))
        nearest, progress = dist.min(dim=-1)
        codes, priority, yielder = pair_codes_v2(self.z_pairs, progress, alive.bool(), pose, speed,
                                                 nearest > self.max_deviation_m, self.reversed_order)
        return z_matrix(priority, yielder, pose.shape[1]), codes


class ZBiasV2(nn.Module):
    FEATURES = 8  # dx, dy (k in q's frame), sin/cos dyaw, distance, speed_q, speed_k, same-direction flag

    def __init__(self, n_heads: int = 8, hidden: int = 128, limit: float = 3.0,
                 position_scale_m: float = 20.0, speed_scale_mps: float = 10.0):
        super().__init__()
        self.config = dict(n_heads=n_heads, hidden=hidden, limit=limit,
                           position_scale_m=position_scale_m, speed_scale_mps=speed_scale_mps)
        self.limit = float(limit)
        self.position_scale_m = float(position_scale_m)
        self.speed_scale_mps = float(speed_scale_mps)
        self.mlp = nn.Sequential(nn.Linear(2 + self.FEATURES, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_heads))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def pair_features(self, pose, speed, kind):
        pose = torch.nan_to_num(pose.float())
        delta = pose[:, None, :, 0:2] - pose[:, :, None, 0:2]          # [b, q, k] = k - q
        c, s = torch.cos(pose[:, :, None, 2]), torch.sin(pose[:, :, None, 2])
        dx = (c * delta[..., 0] + s * delta[..., 1]) / self.position_scale_m
        dy = (-s * delta[..., 0] + c * delta[..., 1]) / self.position_scale_m
        dyaw = pose[:, None, :, 2] - pose[:, :, None, 2]
        dist = torch.linalg.vector_norm(delta, dim=-1) / self.position_scale_m
        v = torch.nan_to_num(speed.float()) / self.speed_scale_mps
        return torch.stack((dx, dy, torch.sin(dyaw), torch.cos(dyaw), dist, v[:, :, None].expand_as(dx),
                            v[:, None, :].expand_as(dx), kind.float()), dim=-1)

    def forward(self, z, pose, speed, alive, kind) -> Optional[torch.Tensor]:
        """Returns chunk_bias (B,1,heads,N,N) or None when no pair is active."""
        active = (z > 0) & alive[:, :, None].bool() & alive[:, None, :].bool()
        if not bool(active.any()):
            return None
        with torch.autocast(device_type=pose.device.type, enabled=False):
            onehot = F.one_hot(z.clamp(0, 2), 3)[..., 1:].float()
            x = torch.cat((onehot, self.pair_features(pose, speed, kind)), dim=-1)
            b = self.limit * torch.tanh(self.mlp(torch.nan_to_num(x)))
            b = torch.where(active[..., None], b, torch.zeros_like(b))
        return b.permute(0, 3, 1, 2)[:, None].contiguous()


def load_zbias_v2(path: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    g = ZBiasV2(**payload["config"])
    g.load_state_dict(payload["state_dict"], strict=True)
    return g.to(device).eval(), {"path": str(path), "step": payload.get("step"), "val": payload.get("val")}
