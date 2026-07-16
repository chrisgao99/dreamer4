"""Focus-only Waymo tokenizers for agent-token and 1x16 latent ablations.

Both variants consume the raw focus-agent features saved by the Waymo vector
filter.  There is deliberately no hand-written feature scaling or dataset
normalization in this module.

Experiment A (``agent_token``):
    raw focus agent -> agent token -> static-map cross attention
    -> causal temporal attention -> reconstruction heads

Experiment B/C (``latent_z16`` or ``latent_z64``):
    raw focus agent -> agent token
    learned z -> cross attention to agent -> cross attention to static map
    -> causal temporal attention -> 1x16 bottleneck -> decoder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from vector_tokenizer_encoder import (
        AgentFeatureEncoder,
        MapFeatureEncoder,
        MapSelfAttentionLayer,
        MultiheadCrossAttention,
        TimeSelfAttention,
    )
except ModuleNotFoundError:
    from waymo.core.vector_tokenizer_encoder import (
        AgentFeatureEncoder,
        MapFeatureEncoder,
        MapSelfAttentionLayer,
        MultiheadCrossAttention,
        TimeSelfAttention,
    )

from dreamer4.model import MLP, RMSNorm, add_sinusoidal_positions


@dataclass(frozen=True)
class FocusTokenizerOutput:
    representation: torch.Tensor  # (B,T,1,D): agent token or 1x16 z
    agent_continuous: torch.Tensor  # (B,T,7): x,y,speed,vx,vy,sin(yaw),cos(yaw)
    agent_valid_logits: torch.Tensor  # (B,T)
    map_tokens: torch.Tensor  # (B,M,D_model)
    map_mask: torch.Tensor  # (B,M)


def focus_agent_btf(agents: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    """Return slot-0 focus agent as (B,T,F) without changing its values."""
    if agents.dim() != 4:
        raise ValueError(f"Expected agents with four dimensions, got {tuple(agents.shape)}")
    k = int(agent_mask.shape[-1])
    if agents.shape[1] == k:  # (B,K,T,F)
        return agents[:, 0]
    if agents.shape[2] == k:  # (B,T,K,F)
        return agents[:, :, 0]
    raise ValueError(f"Cannot identify agent dimension: agents={tuple(agents.shape)} mask={tuple(agent_mask.shape)}")


def focus_agent_targets(focus_btf: torch.Tensor) -> torch.Tensor:
    """Raw reconstruction target with a continuous yaw representation."""
    yaw = focus_btf[..., 6]
    return torch.stack(
        (
            focus_btf[..., 0],
            focus_btf[..., 1],
            focus_btf[..., 2],
            focus_btf[..., 3],
            focus_btf[..., 4],
            torch.sin(yaw),
            torch.cos(yaw),
        ),
        dim=-1,
    )


class StaticMapMemory(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_dim: int,
        n_heads: int,
        depth: int,
        dropout: float,
        mlp_ratio: float,
        scale_pos_embeds: bool,
    ):
        super().__init__()
        self.scale_pos_embeds = bool(scale_pos_embeds)
        self.stem = MapFeatureEncoder(d_model=d_model, hidden_dim=hidden_dim)
        self.layers = nn.ModuleList(
            [
                MapSelfAttentionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, map_polylines: torch.Tensor, map_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_mask = map_mask.bool().any(dim=-1)
        tokens = self.stem(map_polylines, map_mask.bool())
        tokens = tokens * token_mask[..., None].to(tokens.dtype)
        tokens = add_sinusoidal_positions(tokens[:, None], self.scale_pos_embeds).squeeze(1)
        tokens = tokens * token_mask[..., None].to(tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens, map_mask=token_mask)
        return tokens, token_mask


class StaticMemoryCrossAttention(nn.Module):
    """Cross-attend (B,T,Q,D) queries to one static (B,M,D) memory."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiheadCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        b, t, q, d = query.shape
        m = int(memory.shape[1])
        query_flat = query.reshape(b * t, q, d)
        memory_flat = memory[:, None].expand(b, t, m, d).reshape(b * t, m, d)
        safe_mask = memory_mask.bool().clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        mask_flat = safe_mask[:, None].expand(b, t, m).reshape(b * t, m)
        return self.attn(query_flat, memory_flat, memory_mask=mask_flat).reshape(b, t, q, d)


class MapTemporalLayer(nn.Module):
    """Static-map cross attention followed by causal temporal attention."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.norm_map_q = RMSNorm(d_model)
        self.norm_map_memory = RMSNorm(d_model)
        self.map_cross = StaticMemoryCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.drop_map = nn.Dropout(dropout)
        self.norm_time = RMSNorm(d_model)
        self.time = TimeSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.drop_time = nn.Dropout(dropout)
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, map_tokens: torch.Tensor, map_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_map(
            self.map_cross(self.norm_map_q(x), self.norm_map_memory(map_tokens), map_mask)
        )
        token_mask = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        x = x + self.drop_time(self.time(self.norm_time(x), token_mask=token_mask))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LatentEncoderLayer(nn.Module):
    """Directed z<-agent attention, then z<-map, then causal z-time attention."""

    def __init__(self, *, d_model: int, n_heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.norm_z_agent = RMSNorm(d_model)
        self.norm_agent = RMSNorm(d_model)
        self.agent_cross = MultiheadCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.drop_agent = nn.Dropout(dropout)
        self.map_time = MapTemporalLayer(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        z: torch.Tensor,
        agent_token: torch.Tensor,
        map_tokens: torch.Tensor,
        map_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t, _, d = z.shape
        query = self.norm_z_agent(z).reshape(b * t, 1, d)
        memory = self.norm_agent(agent_token).reshape(b * t, 1, d)
        memory_mask = torch.ones((b * t, 1), dtype=torch.bool, device=z.device)
        z = z + self.drop_agent(self.agent_cross(query, memory, memory_mask).reshape(b, t, 1, d))
        return self.map_time(z, map_tokens, map_mask)


class FocusPredictionHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.continuous = nn.Linear(d_model, 7)
        self.valid = nn.Linear(d_model, 1)

    def forward(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token = token.squeeze(2)
        return self.continuous(token), self.valid(token).squeeze(-1)


class FocusAgentTokenizer(nn.Module):
    """Two focus-only tokenizer variants with a shared public interface."""

    def __init__(
        self,
        *,
        representation: str,
        d_model: int = 256,
        d_latent: int = 16,
        hidden_dim: int = 128,
        n_heads: int = 4,
        depth: int = 4,
        decoder_depth: int = 2,
        map_depth: int = 2,
        dropout: float = 0.05,
        mlp_ratio: float = 4.0,
        scale_pos_embeds: bool = True,
    ):
        super().__init__()
        if representation not in {"agent_token", "latent_z16", "latent_z64"}:
            raise ValueError(f"Unknown representation={representation!r}")
        expected_latent_dim = {"latent_z16": 16, "latent_z64": 64}.get(representation)
        if expected_latent_dim is not None and int(d_latent) != expected_latent_dim:
            raise ValueError(
                f"{representation} requires one {expected_latent_dim}-dimensional latent token; got d_latent={d_latent}"
            )
        self.representation_type = str(representation)
        self.d_model = int(d_model)
        self.d_latent = int(d_latent)
        self.scale_pos_embeds = bool(scale_pos_embeds)

        self.agent_encoder = AgentFeatureEncoder(d_model=d_model, hidden_dim=hidden_dim)
        self.map_encoder = StaticMapMemory(
            d_model=d_model,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            depth=map_depth,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            scale_pos_embeds=scale_pos_embeds,
        )

        if self.representation_type == "agent_token":
            self.encoder_layers = nn.ModuleList(
                [
                    MapTemporalLayer(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(int(depth))
                ]
            )
            self.head = FocusPredictionHead(d_model)
        else:
            self.latent_seed = nn.Parameter(torch.empty(1, d_model))
            nn.init.normal_(self.latent_seed, std=0.02)
            self.encoder_layers = nn.ModuleList(
                [
                    LatentEncoderLayer(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(int(depth))
                ]
            )
            self.bottleneck_proj = nn.Linear(d_model, d_latent)
            self.bottleneck_norm = nn.LayerNorm(d_latent)
            self.decoder_up = nn.Linear(d_latent, d_model)
            self.decoder_layers = nn.ModuleList(
                [
                    MapTemporalLayer(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(int(decoder_depth))
                ]
            )
            self.head = FocusPredictionHead(d_model)

    def encode_static_map(self, map_polylines: torch.Tensor, map_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.map_encoder(map_polylines, map_mask)

    def encode(
        self,
        *,
        agents: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        focus = focus_agent_btf(agents, agent_mask)
        agent_token = self.agent_encoder(focus[:, :, None, :])
        agent_token = add_sinusoidal_positions(agent_token, self.scale_pos_embeds)
        map_tokens, static_map_mask = self.encode_static_map(map_polylines, map_mask)

        if self.representation_type == "agent_token":
            state = agent_token
            for layer in self.encoder_layers:
                state = layer(state, map_tokens, static_map_mask)
            return state, map_tokens, static_map_mask

        b, t = focus.shape[:2]
        z = self.latent_seed.view(1, 1, 1, self.d_model).expand(b, t, 1, self.d_model)
        z = add_sinusoidal_positions(z, self.scale_pos_embeds)
        for layer in self.encoder_layers:
            z = layer(z, agent_token, map_tokens, static_map_mask)
        z16 = self.bottleneck_norm(self.bottleneck_proj(z))
        return z16, map_tokens, static_map_mask

    def decode(
        self,
        representation: torch.Tensor,
        *,
        map_tokens: torch.Tensor,
        map_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.representation_type == "agent_token":
            return self.head(representation)
        x = self.decoder_up(representation)
        x = add_sinusoidal_positions(x, self.scale_pos_embeds)
        for layer in self.decoder_layers:
            x = layer(x, map_tokens, map_mask)
        return self.head(x)

    def forward(
        self,
        *,
        agents: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
    ) -> FocusTokenizerOutput:
        representation, map_tokens, static_map_mask = self.encode(
            agents=agents,
            agent_mask=agent_mask,
            map_polylines=map_polylines,
            map_mask=map_mask,
        )
        continuous, valid_logits = self.decode(
            representation,
            map_tokens=map_tokens,
            map_mask=static_map_mask,
        )
        return FocusTokenizerOutput(
            representation=representation,
            agent_continuous=continuous,
            agent_valid_logits=valid_logits,
            map_tokens=map_tokens,
            map_mask=static_map_mask,
        )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _angle_error(pred_yaw: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(pred_yaw - target_yaw), torch.cos(pred_yaw - target_yaw)).abs()


def focus_tokenizer_loss(
    output: FocusTokenizerOutput,
    *,
    agents: torch.Tensor,
    agent_mask: torch.Tensor,
    xy_weight: float = 1.0,
    velocity_weight: float = 0.5,
    yaw_weight: float = 0.5,
    valid_weight: float = 0.2,
    delta_xy_weight: float = 0.0,
    kinematic_xy_weight: float = 0.0,
    speed_yaw_kinematic_weight: float = 0.0,
    kinematic_dt: float = 0.1,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    focus = focus_agent_btf(agents, agent_mask)
    target = focus_agent_targets(focus)
    valid = focus[..., 5] > 0.5
    valid_f = valid.to(dtype=target.dtype)

    xy_loss = _weighted_mean(F.smooth_l1_loss(output.agent_continuous[..., 0:2], target[..., 0:2], reduction="none"), valid_f)
    velocity_loss = _weighted_mean(F.smooth_l1_loss(output.agent_continuous[..., 2:5], target[..., 2:5], reduction="none"), valid_f)
    yaw_loss = _weighted_mean(F.smooth_l1_loss(output.agent_continuous[..., 5:7], target[..., 5:7], reduction="none"), valid_f)
    valid_loss = F.binary_cross_entropy_with_logits(output.agent_valid_logits, valid.to(output.agent_valid_logits.dtype))

    consecutive = valid[:, 1:] & valid[:, :-1]
    consecutive_f = consecutive.to(dtype=target.dtype)
    pred_delta = output.agent_continuous[:, 1:, 0:2] - output.agent_continuous[:, :-1, 0:2]
    target_delta = target[:, 1:, 0:2] - target[:, :-1, 0:2]
    delta_loss = _weighted_mean(F.smooth_l1_loss(pred_delta, target_delta, reduction="none"), consecutive_f)

    pred_v_delta = output.agent_continuous[:, :-1, 3:5] * float(kinematic_dt)
    kinematic_loss = _weighted_mean(F.smooth_l1_loss(pred_delta, pred_v_delta, reduction="none"), consecutive_f)
    pred_yaw = torch.atan2(output.agent_continuous[:, :-1, 5], output.agent_continuous[:, :-1, 6])
    speed_delta = torch.stack(
        (
            output.agent_continuous[:, :-1, 2] * torch.cos(pred_yaw) * float(kinematic_dt),
            output.agent_continuous[:, :-1, 2] * torch.sin(pred_yaw) * float(kinematic_dt),
        ),
        dim=-1,
    )
    speed_yaw_kinematic_loss = _weighted_mean(
        F.smooth_l1_loss(pred_delta, speed_delta, reduction="none"), consecutive_f
    )

    total = (
        float(xy_weight) * xy_loss
        + float(velocity_weight) * velocity_loss
        + float(yaw_weight) * yaw_loss
        + float(valid_weight) * valid_loss
        + float(delta_xy_weight) * delta_loss
        + float(kinematic_xy_weight) * kinematic_loss
        + float(speed_yaw_kinematic_weight) * speed_yaw_kinematic_loss
    )

    xy_error = (output.agent_continuous[..., 0:2] - target[..., 0:2]).norm(dim=-1)
    xy_mae = _weighted_mean(xy_error, valid_f)
    any_valid = valid.any(dim=1)
    time_idx = torch.arange(valid.shape[1], device=valid.device).view(1, -1)
    last_idx = torch.where(valid, time_idx, torch.zeros_like(time_idx)).amax(dim=1)
    batch_idx = torch.arange(valid.shape[0], device=valid.device)
    fde = (output.agent_continuous[batch_idx, last_idx, 0:2] - target[batch_idx, last_idx, 0:2]).norm(dim=-1)
    fde = _weighted_mean(fde, any_valid.to(target.dtype))
    speed_mae = _weighted_mean((output.agent_continuous[..., 2] - target[..., 2]).abs(), valid_f)
    vxvy_mae = _weighted_mean((output.agent_continuous[..., 3:5] - target[..., 3:5]).norm(dim=-1), valid_f)
    yaw_pred = torch.atan2(output.agent_continuous[..., 5], output.agent_continuous[..., 6])
    yaw_mae_deg = _weighted_mean(_angle_error(yaw_pred, focus[..., 6]) * (180.0 / torch.pi), valid_f)
    valid_acc = ((output.agent_valid_logits > 0.0) == valid).float().mean()

    metrics = {
        "loss_total": total.detach(),
        "loss_xy": xy_loss.detach(),
        "loss_velocity": velocity_loss.detach(),
        "loss_yaw": yaw_loss.detach(),
        "loss_valid": valid_loss.detach(),
        "loss_delta_xy": delta_loss.detach(),
        "loss_kinematic_xy": kinematic_loss.detach(),
        "loss_speed_yaw_kinematic": speed_yaw_kinematic_loss.detach(),
        "focus_xy_mae_m": xy_mae.detach(),
        "focus_fde_m": fde.detach(),
        "focus_speed_mae_mps": speed_mae.detach(),
        "focus_vxvy_mae_mps": vxvy_mae.detach(),
        "focus_yaw_mae_deg": yaw_mae_deg.detach(),
        "focus_valid_acc": valid_acc.detach(),
        "representation_rms": output.representation.float().pow(2).mean().sqrt().detach(),
    }
    return total, metrics
