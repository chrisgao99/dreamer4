"""g: oracle pair passing-order z -> per-head agent_attn bias (plan §8.2).

One labelling rule is shared by training (log state at anchor t) and the
closed-loop oracle (rollout state projected onto each car's logged path):

  pair record (i, j, enter_i, exit_i, enter_j, exit_j, angle); f = earlier entry, s = later
  progress p_x: frame index along x's own logged path (training: p = t for every car)
  in_x = enter_x <= p_x <= exit_x ; passed_x = p_x > exit_x
    either passed           -> released, no bias        (code 5)
    both in overlap         -> unknown, no bias         (code 4)
    only s in overlap       -> s first                  (code 3; rollout only: s entered first)
    only f in overlap       -> f first                  (code 2)
    neither entered         -> log order: f first       (code 1; ``reversed``: s first)
    tie (same entry frame)  -> unknown                  (code 7)
    a car > max_dev from its logged path -> unknown     (code 6; rollout only)
    a car not alive         -> no bias                  (code 8)

Directed z for query q / key k: 1 = q goes first, 2 = k goes first, 0 = none.
b[q, k, h] = 3 * tanh(MLP(onehot(z), current pair features)); zero-initialized last layer.
Pair features use only the current state (no logged future), so g is reusable with
a predicted z (experiment 2).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P_MAX = 32  # padded pairs per sample (labels v1: max per sample is far below this; checked on load)
CODE_NAMES = {0: "pad", 1: "pre_entry_log_order", 2: "first_in_overlap", 3: "second_in_overlap",
              4: "both_in_unknown", 5: "released", 6: "deviated_unknown", 7: "tie_unknown",
              8: "not_alive"}
ACTIVE_CODES = (1, 2, 3)


def atomic_save_npz(path: Path, **arrays) -> None:
    path = Path(path)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class PairLabels:
    def __init__(self, path: str):
        with np.load(path, allow_pickle=False) as d:
            names, offsets, pairs = d["names"], d["offsets"], d["pairs"]
        self.path = str(path)
        self.index = {str(n): (int(offsets[k]), int(offsets[k + 1])) for k, n in enumerate(names)}
        self.pairs = pairs.astype(np.int64)
        widest = int(np.diff(offsets).max()) if len(names) else 0
        if widest > P_MAX:
            raise ValueError(f"{path}: a sample has {widest} pairs > P_MAX={P_MAX}")

    def has_pairs(self, name: str) -> bool:
        lo, hi = self.index[name]
        return hi > lo

    def padded(self, name: str) -> torch.Tensor:
        lo, hi = self.index[name]
        out = np.full((P_MAX, 7), -1, dtype=np.int64)
        out[: hi - lo] = self.pairs[lo:hi]
        return torch.from_numpy(out)


def with_z_pairs(base_cls, labels: PairLabels):
    """Dataset subclass adding fixed-size ``z_pairs`` (P_MAX, 7), -1 padded (stacks in any collate)."""

    class ZPairDataset(base_cls):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            item["z_pairs"] = labels.padded(Path(self.paths[index]).name)
            return item

    ZPairDataset.__name__ = f"ZPair{base_cls.__name__}"
    return ZPairDataset


def pair_codes(z_pairs: torch.Tensor, progress: torch.Tensor, alive: torch.Tensor,
               deviated: Optional[torch.Tensor] = None, reversed_order: bool = False):
    """z_pairs (B,P,7) long; progress (B,N) long; alive (B,N) bool.

    Returns codes (B,P) long and priority / yielder slot indices (B,P) (-1 when inactive).
    """
    valid = z_pairs[..., 0] >= 0
    zp = z_pairs.clamp_min(0)
    i, j, ei, xi, ej, xj = (zp[..., k] for k in range(6))
    swap = ej < ei
    f, s = torch.where(swap, j, i), torch.where(swap, i, j)
    ef, xf = torch.where(swap, ej, ei), torch.where(swap, xj, xi)
    es, xs = torch.where(swap, ei, ej), torch.where(swap, xi, xj)
    pf, ps = progress.gather(1, f), progress.gather(1, s)
    in_f, in_s = (pf >= ef) & (pf <= xf), (ps >= es) & (ps <= xs)
    passed = (pf > xf) | (ps > xs)
    codes = torch.ones_like(f)
    codes = torch.where(in_f & ~in_s, torch.full_like(codes, 2), codes)
    codes = torch.where(in_s & ~in_f, torch.full_like(codes, 3), codes)
    codes = torch.where(in_f & in_s, torch.full_like(codes, 4), codes)
    codes = torch.where(passed, torch.full_like(codes, 5), codes)
    codes = torch.where(ei == ej, torch.full_like(codes, 7), codes)
    if deviated is not None:
        dev = deviated.gather(1, f) | deviated.gather(1, s)
        codes = torch.where(dev, torch.full_like(codes, 6), codes)
    both_alive = alive.gather(1, f) & alive.gather(1, s)
    codes = torch.where(both_alive, codes, torch.full_like(codes, 8))
    codes = torch.where(valid, codes, torch.zeros_like(codes))
    pre_first, pre_second = (s, f) if reversed_order else (f, s)
    priority = torch.where(codes == 3, s, torch.where(codes == 1, pre_first, f))
    yielder = torch.where(codes == 3, f, torch.where(codes == 1, pre_second, s))
    active = (codes == 1) | (codes == 2) | (codes == 3)
    priority = torch.where(active, priority, torch.full_like(priority, -1))
    yielder = torch.where(active, yielder, torch.full_like(yielder, -1))
    return codes, priority, yielder


def z_matrix(priority: torch.Tensor, yielder: torch.Tensor, num_agents: int) -> torch.Tensor:
    """(B,N,N) long: z[b,q,k] = 1 if q goes first over k, 2 if k goes first, else 0."""
    bsz = int(priority.shape[0])
    z = torch.zeros((bsz, num_agents, num_agents), dtype=torch.long, device=priority.device)
    active = priority >= 0
    if active.any():
        b = torch.arange(bsz, device=priority.device)[:, None].expand_as(priority)[active]
        p, y = priority[active], yielder[active]
        z[b, y, p] = 2
        z[b, p, y] = 1
    return z


def training_z(z_pairs: torch.Tensor, anchors: torch.Tensor, alive: torch.Tensor):
    """Log-state labels at anchor t for every car (teacher forcing)."""
    progress = anchors[:, None].expand(-1, alive.shape[1]).long()
    codes, priority, yielder = pair_codes(z_pairs, progress, alive)
    return z_matrix(priority, yielder, alive.shape[1]), codes, priority, yielder


class OracleZ:
    """Closed-loop oracle: log order, rollout state (projection onto each car's logged path)."""

    def __init__(self, agents: torch.Tensor, agent_mask: torch.Tensor, z_pairs: torch.Tensor, *,
                 reversed_order: bool = False, max_deviation_m: float = 2.0):
        from waymo.training.world_model.direct_action_flow import agents_to_bntf
        a = agents_to_bntf(agents, agent_mask).float()
        self.log_xy = a[..., 0:2]
        self.log_valid = (a[..., 5] > 0.5) & torch.isfinite(a[..., 0:2]).all(-1) & agent_mask[:, :, None].bool()
        self.z_pairs = z_pairs.long()
        self.reversed_order = bool(reversed_order)
        self.max_deviation_m = float(max_deviation_m)

    def step(self, pose: torch.Tensor, alive: torch.Tensor):
        dist = torch.linalg.vector_norm(self.log_xy - pose[:, :, None, 0:2].float(), dim=-1)
        dist = torch.nan_to_num(dist, nan=float("inf")).masked_fill(~self.log_valid, float("inf"))
        nearest, progress = dist.min(dim=-1)
        deviated = nearest > self.max_deviation_m
        codes, priority, yielder = pair_codes(self.z_pairs, progress, alive.bool(), deviated,
                                              self.reversed_order)
        return z_matrix(priority, yielder, pose.shape[1]), codes


class ZBias(nn.Module):
    FEATURES = 7  # dx, dy (j in i's frame), sin/cos dyaw, distance, speed_i, speed_j

    def __init__(self, n_heads: int = 8, hidden: int = 128, limit: float = 3.0,
                 position_scale_m: float = 20.0, speed_scale_mps: float = 10.0):
        super().__init__()
        self.config = dict(n_heads=n_heads, hidden=hidden, limit=limit,
                           position_scale_m=position_scale_m, speed_scale_mps=speed_scale_mps)
        self.limit = float(limit)
        self.position_scale_m = float(position_scale_m)
        self.speed_scale_mps = float(speed_scale_mps)
        self.mlp = nn.Sequential(
            nn.Linear(2 + self.FEATURES, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_heads),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def pair_features(self, pose: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        pose = torch.nan_to_num(pose.float())
        delta = pose[:, None, :, 0:2] - pose[:, :, None, 0:2]          # [b, q, k] = k - q
        c, s = torch.cos(pose[:, :, None, 2]), torch.sin(pose[:, :, None, 2])
        dx = (c * delta[..., 0] + s * delta[..., 1]) / self.position_scale_m
        dy = (-s * delta[..., 0] + c * delta[..., 1]) / self.position_scale_m
        dyaw = pose[:, None, :, 2] - pose[:, :, None, 2]
        dist = torch.linalg.vector_norm(delta, dim=-1) / self.position_scale_m
        v = torch.nan_to_num(speed.float()) / self.speed_scale_mps
        vq = v[:, :, None].expand_as(dx)
        vk = v[:, None, :].expand_as(dx)
        return torch.stack((dx, dy, torch.sin(dyaw), torch.cos(dyaw), dist, vq, vk), dim=-1)

    def forward(self, z: torch.Tensor, pose: torch.Tensor, speed: torch.Tensor,
                alive: torch.Tensor) -> Optional[torch.Tensor]:
        """Returns chunk_bias (B,1,heads,N,N) or None when no pair is active."""
        active = (z > 0) & alive[:, :, None].bool() & alive[:, None, :].bool()
        if not bool(active.any()):
            return None
        with torch.autocast(device_type=pose.device.type, enabled=False):
            onehot = F.one_hot(z.clamp(0, 2), 3)[..., 1:].float()
            x = torch.cat((onehot, self.pair_features(pose, speed)), dim=-1)
            b = self.limit * torch.tanh(self.mlp(torch.nan_to_num(x)))
            b = torch.where(active[..., None], b, torch.zeros_like(b))
        return b.permute(0, 3, 1, 2)[:, None].contiguous()


def load_zbias(path: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    g = ZBias(**payload["config"])
    g.load_state_dict(payload["state_dict"], strict=True)
    return g.to(device).eval(), {"path": str(path), "step": payload.get("step"), "val": payload.get("val")}


class BiasStats:
    """Running per-head stats of b on active directed edges (yield: query yields to key)."""

    def __init__(self):
        self.sums = {}

    @torch.no_grad()
    def update(self, bias: torch.Tensor, z: torch.Tensor) -> None:
        b = bias[:, 0].permute(0, 2, 3, 1).float()     # (B,N,N,H)
        for name, cls in (("yield", 2), ("priority", 1)):
            sel = b[z == cls]                            # (E,H)
            if sel.numel() == 0:
                continue
            acc = self.sums.setdefault(name, {"n": 0, "sum": torch.zeros(b.shape[-1]),
                                              "abs_sum": 0.0, "gt25": 0, "max_head_sum": 0.0})
            acc["n"] += int(sel.shape[0])
            acc["sum"] += sel.sum(0).cpu()
            acc["abs_sum"] += float(sel.abs().mean(1).sum())
            acc["gt25"] += int((sel.abs() > 2.5).any(1).sum())
            acc["max_head_sum"] += float(sel.max(1).values.sum())

    def summary(self) -> dict:
        out = {}
        for name, acc in self.sums.items():
            n = max(1, acc["n"])
            out[name] = {"edges": acc["n"], "per_head_mean": [round(float(v) / n, 4) for v in acc["sum"]],
                         "mean_abs": acc["abs_sum"] / n, "max_head_mean": acc["max_head_sum"] / n,
                         "frac_any_head_abs_gt_2.5": acc["gt25"] / n}
        return out
