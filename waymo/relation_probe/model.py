"""Probe models for current-state pair relation experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int, dropout: float):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        dim = in_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationProbe(nn.Module):
    """Multi-task current relation probe.

    Modes:
      - raw_only: pair_raw -> heads
      - raw_z/raw_shuffled_z: pair_raw -> query, query attends to z_current -> heads
    """

    def __init__(
        self,
        *,
        mode: str,
        pair_dim: int,
        z_dim: int,
        n_reg: int,
        n_bin: int,
        d_model: int = 128,
        n_heads: int = 4,
        depth: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        if mode not in {"raw_only", "raw_z", "raw_shuffled_z"}:
            raise ValueError(f"Unknown probe mode: {mode}")
        self.mode = mode
        self.pair_dim = int(pair_dim)
        self.z_dim = int(z_dim)
        self.n_reg = int(n_reg)
        self.n_bin = int(n_bin)
        self.d_model = int(d_model)

        self.pair_encoder = MLP(pair_dim, d_model, d_model, depth=depth, dropout=dropout)
        if mode == "raw_only":
            head_in = d_model
            self.z_proj = None
            self.attn = None
            self.fuse = None
        else:
            self.z_proj = nn.Sequential(nn.Linear(z_dim, d_model), nn.LayerNorm(d_model))
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.fuse = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            head_in = d_model

        self.reg_head = MLP(head_in, d_model, n_reg, depth=2, dropout=dropout)
        self.bin_head = MLP(head_in, d_model, n_bin, depth=2, dropout=dropout)

    def forward(self, pair_raw: torch.Tensor, z_current: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.pair_encoder(pair_raw)
        if self.mode == "raw_only":
            h = q
        else:
            if z_current is None:
                raise ValueError(f"mode={self.mode} requires z_current")
            memory = self.z_proj(z_current)
            attn_out, _ = self.attn(q[:, None, :], memory, memory, need_weights=False)
            h = self.fuse(torch.cat([q, attn_out[:, 0]], dim=-1))
        return self.reg_head(h), self.bin_head(h), h


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    denom = mask.sum(dim=0).clamp_min(1.0)
    return ((loss * mask).sum(dim=0) / denom).mean()


def masked_bce_with_logits(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none", pos_weight=pos_weight)
    denom = mask.sum(dim=0).clamp_min(1.0)
    return ((loss * mask).sum(dim=0) / denom).mean()

