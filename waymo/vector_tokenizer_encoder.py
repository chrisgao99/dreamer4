"""Dreamer-4-style encoder for filtered Waymo vector scenes.

The encoder follows the same broad pattern as Dreamer 4:

- build per-timestep spatial tokens
- run block-causal transformer layers
- each layer mixes tokens in space, then causally mixes each token slot in time
- output bottleneck latent tokens plus agent/map/light representations

The map polyline stem follows the MTR idea: point-wise MLP, max-pool over a
polyline, concatenate local/global features, point-wise MLP, max-pool again.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dreamer4.model import MLP, RMSNorm, add_sinusoidal_positions
from waymo_vector_dataset import WaymoVectorDataset


@dataclass(frozen=True)
class VectorEncoderOutput:
    z: torch.Tensor                 # (B,T,N_latents,D_bottleneck)
    agent_tokens: torch.Tensor      # (B,T,K,D)
    map_tokens: torch.Tensor        # (B,T,M,D)
    light_tokens: torch.Tensor      # (B,T,L,D)
    token_mask: torch.Tensor        # (B,T,S)
    map_token_mask: Optional[torch.Tensor] = None  # (B,M), True for valid static map tokens


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        n, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(n, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(n, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(n, length, self.n_heads, self.head_dim).transpose(1, 2)
        dropout = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(n, length, dim)
        return self.out(y)


class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout_p = float(dropout)

        self.q = nn.Linear(self.d_model, self.d_model)
        self.kv = nn.Linear(self.d_model, 2 * self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        n, q_len, dim = query.shape
        m_len = memory.shape[1]
        q = self.q(query)
        k, v = self.kv(memory).chunk(2, dim=-1)
        q = q.view(n, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(n, m_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(n, m_len, self.n_heads, self.head_dim).transpose(1, 2)
        dropout = self.dropout_p if self.training else 0.0
        attn_mask = memory_mask[:, None, None, :].expand(n, 1, q_len, m_len)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(n, q_len, dim)
        return self.out(y)


class PointNetPolylineEncoder(nn.Module):
    """Small MTR-style polyline encoder."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.post = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, polylines: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode polylines.

        Args:
            polylines: (B,M,P,F)
            mask: (B,M,P), True for valid points

        Returns:
            (B,M,D)
        """
        b, m, p, _ = polylines.shape
        h = self.pre(polylines)
        point_mask = mask[..., None]
        h = h.masked_fill(~point_mask, 0.0)
        pooled = h.masked_fill(~point_mask, -1e4).max(dim=2).values
        pooled = torch.where(mask.any(dim=2, keepdim=True), pooled, torch.zeros_like(pooled))

        h2 = torch.cat([h, pooled[:, :, None, :].expand(b, m, p, -1)], dim=-1)
        h2 = self.post(h2).masked_fill(~point_mask, 0.0)
        out = h2.masked_fill(~point_mask, -1e4).max(dim=2).values
        out = torch.where(mask.any(dim=2, keepdim=True), out, torch.zeros_like(out))
        return out


class AgentFeatureEncoder(nn.Module):
    """Encodes selected agent state at each timestep.

    Input feature order from the filter:
    x, y, speed, vx, vy, valid, yaw, type
    """

    def __init__(self, d_model: int, hidden_dim: int, max_agent_type: int = 16):
        super().__init__()
        self.type_emb = nn.Embedding(max_agent_type, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(8 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, agents: torch.Tensor) -> torch.Tensor:
        agent_type = agents[..., 7].long().clamp(min=0, max=self.type_emb.num_embeddings - 1)
        yaw = agents[..., 6]
        # Mild scale normalization for meter-like fields.
        cont = torch.stack(
            [
                agents[..., 0] / 100.0,
                agents[..., 1] / 100.0,
                agents[..., 2] / 30.0,
                agents[..., 3] / 30.0,
                agents[..., 4] / 30.0,
                agents[..., 5],
                torch.sin(yaw),
                torch.cos(yaw),
            ],
            dim=-1,
        )
        return self.mlp(torch.cat([cont, self.type_emb(agent_type)], dim=-1))


class TrafficLightEncoder(nn.Module):
    """Encodes dynamic traffic-light tokens at each timestep."""

    def __init__(self, d_model: int, hidden_dim: int, max_state: int = 16):
        super().__init__()
        self.state_emb = nn.Embedding(max_state, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, lights: torch.Tensor) -> torch.Tensor:
        state = lights[..., 2].long().clamp(min=0, max=self.state_emb.num_embeddings - 1)
        cont = torch.stack(
            [
                lights[..., 0] / 100.0,
                lights[..., 1] / 100.0,
                lights[..., 3],
                (lights[..., 2] > 0).to(lights.dtype),
            ],
            dim=-1,
        )
        return self.mlp(torch.cat([cont, self.state_emb(state)], dim=-1))


class MapFeatureEncoder(nn.Module):
    """Encodes filtered map polylines into map tokens."""

    def __init__(self, d_model: int, hidden_dim: int, max_lane_type: int = 64):
        super().__init__()
        self.type_emb = nn.Embedding(max_lane_type, hidden_dim)
        # x, y, dir_x, dir_y, valid + type embedding
        self.pointnet = PointNetPolylineEncoder(in_dim=5 + hidden_dim, hidden_dim=hidden_dim, out_dim=d_model)

    def forward(self, map_polylines: torch.Tensor, map_mask: torch.Tensor) -> torch.Tensor:
        lane_type = map_polylines[..., 4].long().clamp(min=0, max=self.type_emb.num_embeddings - 1)
        cont = torch.stack(
            [
                map_polylines[..., 0] / 100.0,
                map_polylines[..., 1] / 100.0,
                map_polylines[..., 2],
                map_polylines[..., 3],
                map_polylines[..., 5],
            ],
            dim=-1,
        )
        point_features = torch.cat([cont, self.type_emb(lane_type)], dim=-1)
        return self.pointnet(point_features, map_mask)


class SpaceSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        b, t, s, d = x.shape
        x_flat = x.reshape(b * t, s, d)
        mask_flat = token_mask.reshape(b * t, s)
        # True means allowed in PyTorch SDPA bool masks.
        attn_mask = mask_flat[:, None, None, :].expand(b * t, 1, s, s)
        out = self.attn(x_flat, attn_mask=attn_mask, is_causal=False).reshape(b, t, s, d)
        return out * token_mask[..., None].to(out.dtype)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiheadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        b, t, s, d = x.shape
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(b * s, t, d)
        mask_flat = token_mask.permute(0, 2, 1).contiguous().view(b * s, t)
        causal = torch.ones((t, t), dtype=torch.bool, device=x.device).tril()
        # Fully padded slots would otherwise have no valid keys and may produce
        # non-finite SDPA rows. Let them attend to their zero token at t=0; the
        # output is masked back to zero immediately after attention.
        safe_mask = mask_flat.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        key_mask = safe_mask[:, None, None, :]
        attn_mask = causal[None, None, :, :] & key_mask
        out = self.attn(x_flat, attn_mask=attn_mask, is_causal=False)
        out = out.view(b, s, t, d).permute(0, 2, 1, 3).contiguous()
        return out * token_mask[..., None].to(out.dtype)


class MapSelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.norm_attn = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, map_mask: torch.Tensor) -> torch.Tensor:
        b, m, _ = x.shape
        safe_mask = map_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        attn_mask = safe_mask[:, None, None, :].expand(b, 1, m, m)
        x = x + self.drop_attn(self.attn(self.norm_attn(x), attn_mask=attn_mask, is_causal=False))
        x = x + self.mlp(self.norm_mlp(x))
        return x * map_mask[..., None].to(x.dtype)


class StaticMapCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = MultiheadCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        map_tokens: torch.Tensor,
        map_mask: torch.Tensor,
        token_slice: slice,
    ) -> torch.Tensor:
        b, t, _, d = x.shape
        q = x[:, :, token_slice, :]
        q_mask = token_mask[:, :, token_slice]
        q_len = q.shape[2]
        q_flat = q.reshape(b * t, q_len, d)
        memory = map_tokens[:, None, :, :].expand(b, t, map_tokens.shape[1], d).reshape(b * t, map_tokens.shape[1], d)
        safe_map_mask = map_mask.clone()
        empty = ~safe_map_mask.any(dim=1)
        if empty.any():
            safe_map_mask[empty, 0] = True
        memory_mask = safe_map_mask[:, None, :].expand(b, t, map_tokens.shape[1]).reshape(b * t, map_tokens.shape[1])
        out = self.attn(q_flat, memory, memory_mask=memory_mask).reshape(b, t, q_len, d)
        out = out * q_mask[..., None].to(out.dtype)
        updated = x.clone()
        updated[:, :, token_slice, :] = updated[:, :, token_slice, :] + out
        return updated * token_mask[..., None].to(updated.dtype)


class VectorBlockCausalLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, mlp_ratio: float, layer_index: int, time_every: int):
        super().__init__()
        self.do_time = ((layer_index + 1) % int(time_every) == 0)
        self.norm_space = RMSNorm(d_model)
        self.space = SpaceSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.drop_space = nn.Dropout(dropout)

        if self.do_time:
            self.norm_time = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            self.drop_time = nn.Dropout(dropout)

        self.norm_mlp = RMSNorm(d_model)
        self.mlp = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_space(self.space(self.norm_space(x), token_mask=token_mask))
        if self.do_time:
            x = x + self.drop_time(self.time(self.norm_time(x), token_mask=token_mask))
        x = x + self.mlp(self.norm_mlp(x))
        return x * token_mask[..., None].to(x.dtype)


class VectorStaticMapQueryLayer(nn.Module):
    """Block-causal dynamic layer with static map cross-attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        mlp_ratio: float,
        layer_index: int,
        time_every: int,
        map_cross_every: int,
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % int(time_every) == 0)
        self.do_map_cross = int(map_cross_every) > 0 and ((layer_index + 1) % int(map_cross_every) == 0)

        self.norm_space = RMSNorm(d_model)
        self.space = SpaceSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.drop_space = nn.Dropout(dropout)

        if self.do_map_cross:
            self.norm_map_query = RMSNorm(d_model)
            self.norm_map_memory = RMSNorm(d_model)
            self.map_cross = StaticMapCrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            self.drop_map = nn.Dropout(dropout)

        if self.do_time:
            self.norm_time = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            self.drop_time = nn.Dropout(dropout)

        self.norm_mlp = RMSNorm(d_model)
        self.mlp = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        map_tokens: torch.Tensor,
        map_mask: torch.Tensor,
        map_query_slice: slice,
    ) -> torch.Tensor:
        x = x + self.drop_space(self.space(self.norm_space(x), token_mask=token_mask))
        if self.do_map_cross and map_query_slice.stop > map_query_slice.start:
            map_update = self.map_cross(
                self.norm_map_query(x),
                token_mask=token_mask,
                map_tokens=self.norm_map_memory(map_tokens),
                map_mask=map_mask,
                token_slice=map_query_slice,
            )
            x = x + self.drop_map(map_update - self.norm_map_query(x))
            x = x * token_mask[..., None].to(x.dtype)
        if self.do_time:
            x = x + self.drop_time(self.time(self.norm_time(x), token_mask=token_mask))
        x = x + self.mlp(self.norm_mlp(x))
        return x * token_mask[..., None].to(x.dtype)


class VectorBlockCausalEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 256,
        n_heads: int = 4,
        depth: int = 6,
        n_latents: int = 8,
        d_bottleneck: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        mlp_ratio: float = 4.0,
        time_every: int = 1,
        scale_pos_embeds: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_latents = int(n_latents)
        self.d_bottleneck = int(d_bottleneck)
        self.scale_pos_embeds = bool(scale_pos_embeds)

        self.agent_encoder = AgentFeatureEncoder(d_model=d_model, hidden_dim=hidden_dim)
        self.map_encoder = MapFeatureEncoder(d_model=d_model, hidden_dim=hidden_dim)
        self.light_encoder = TrafficLightEncoder(d_model=d_model, hidden_dim=hidden_dim)

        self.latent_tokens = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latent_tokens, std=0.02)

        self.layers = nn.ModuleList(
            [
                VectorBlockCausalLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    layer_index=i,
                    time_every=time_every,
                )
                for i in range(depth)
            ]
        )
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

    @staticmethod
    def _maybe_transpose_agents(agents: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        # Filter currently saves (K,T,F). Accept (B,K,T,F) and convert to (B,T,K,F).
        if agents.dim() != 4:
            raise ValueError(f"Expected agents with shape (B,K,T,F) or (B,T,K,F), got {tuple(agents.shape)}")
        k = agent_mask.shape[-1]
        if agents.shape[1] == k:
            return agents.transpose(1, 2).contiguous()
        return agents

    def forward(
        self,
        agents: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        lights: torch.Tensor,
        light_mask: torch.Tensor,
    ) -> VectorEncoderOutput:
        """Encode a vector scene.

        Args:
            agents: (B,K,T,8) or (B,T,K,8)
            agent_mask: (B,K)
            map_polylines: (B,M,P,6)
            map_mask: (B,M,P)
            lights: (B,T,L,4)
            light_mask: (B,T,L)
        """
        agents = self._maybe_transpose_agents(agents, agent_mask)
        b, t, k, _ = agents.shape
        m = map_polylines.shape[1]
        l = lights.shape[2]

        agent_valid = (agents[..., 5] > 0.5) & agent_mask[:, None, :]
        map_valid = map_mask.any(dim=-1)
        light_valid = light_mask.bool()

        agent_tokens = self.agent_encoder(agents) * agent_valid[..., None].to(agents.dtype)
        map_once = self.map_encoder(map_polylines, map_mask.bool()) * map_valid[..., None].to(map_polylines.dtype)
        map_tokens = map_once[:, None, :, :].expand(b, t, m, self.d_model)
        light_tokens = self.light_encoder(lights) * light_valid[..., None].to(lights.dtype)

        latents = self.latent_tokens.view(1, 1, self.n_latents, self.d_model).expand(b, t, self.n_latents, self.d_model)
        tokens = torch.cat([latents, agent_tokens, map_tokens, light_tokens], dim=2)

        latent_mask = torch.ones((b, t, self.n_latents), dtype=torch.bool, device=tokens.device)
        map_token_mask = map_valid[:, None, :].expand(b, t, m)
        token_mask = torch.cat([latent_mask, agent_valid, map_token_mask, light_valid], dim=2)

        tokens = add_sinusoidal_positions(tokens, self.scale_pos_embeds)
        tokens = tokens * token_mask[..., None].to(tokens.dtype)

        for layer in self.layers:
            tokens = layer(tokens, token_mask=token_mask)

        latent_slice = slice(0, self.n_latents)
        agent_slice = slice(self.n_latents, self.n_latents + k)
        map_slice = slice(agent_slice.stop, agent_slice.stop + m)
        light_slice = slice(map_slice.stop, map_slice.stop + l)

        z = torch.tanh(self.bottleneck_proj(tokens[:, :, latent_slice, :]))
        return VectorEncoderOutput(
            z=z,
            agent_tokens=tokens[:, :, agent_slice, :],
            map_tokens=tokens[:, :, map_slice, :],
            light_tokens=tokens[:, :, light_slice, :],
            token_mask=token_mask,
            map_token_mask=map_valid,
        )


class VectorStaticMapQueryEncoder(nn.Module):
    """Encoder B: dynamic tokens query static map memory instead of repeating map over time."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        n_heads: int = 4,
        depth: int = 6,
        n_latents: int = 8,
        d_bottleneck: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        mlp_ratio: float = 4.0,
        time_every: int = 1,
        scale_pos_embeds: bool = True,
        map_depth: int = 2,
        map_cross_every: int = 1,
        map_query_tokens: str = "latent_agent",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_latents = int(n_latents)
        self.d_bottleneck = int(d_bottleneck)
        self.scale_pos_embeds = bool(scale_pos_embeds)
        self.map_query_tokens = str(map_query_tokens)

        self.agent_encoder = AgentFeatureEncoder(d_model=d_model, hidden_dim=hidden_dim)
        self.map_encoder = MapFeatureEncoder(d_model=d_model, hidden_dim=hidden_dim)
        self.light_encoder = TrafficLightEncoder(d_model=d_model, hidden_dim=hidden_dim)

        self.latent_tokens = nn.Parameter(torch.empty(n_latents, d_model))
        nn.init.normal_(self.latent_tokens, std=0.02)

        self.map_layers = nn.ModuleList(
            [
                MapSelfAttentionLayer(d_model=d_model, n_heads=n_heads, dropout=dropout, mlp_ratio=mlp_ratio)
                for _ in range(int(map_depth))
            ]
        )
        self.layers = nn.ModuleList(
            [
                VectorStaticMapQueryLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    layer_index=i,
                    time_every=time_every,
                    map_cross_every=map_cross_every,
                )
                for i in range(depth)
            ]
        )
        self.bottleneck_proj = nn.Linear(d_model, d_bottleneck)

    @staticmethod
    def _maybe_transpose_agents(agents: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        return VectorBlockCausalEncoder._maybe_transpose_agents(agents, agent_mask)

    def _map_query_slice(self, k: int, l: int) -> slice:
        if self.map_query_tokens == "latent":
            return slice(0, self.n_latents)
        if self.map_query_tokens == "agent":
            return slice(self.n_latents, self.n_latents + k)
        if self.map_query_tokens == "latent_agent":
            return slice(0, self.n_latents + k)
        if self.map_query_tokens == "all":
            return slice(0, self.n_latents + k + l)
        raise ValueError(
            "map_query_tokens must be one of: latent, agent, latent_agent, all; "
            f"got {self.map_query_tokens!r}"
        )

    def forward(
        self,
        agents: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        lights: torch.Tensor,
        light_mask: torch.Tensor,
    ) -> VectorEncoderOutput:
        """Encode a vector scene with static map memory.

        Dynamic token layout per timestep:
            [latent_1..N, agent_1..K, light_1..L]

        Map tokens stay in a separate static memory:
            map_polylines -> (B,M,D)
        """
        agents = self._maybe_transpose_agents(agents, agent_mask)
        b, t, k, _ = agents.shape
        m = map_polylines.shape[1]
        l = lights.shape[2]

        agent_valid = (agents[..., 5] > 0.5) & agent_mask[:, None, :]
        map_valid = map_mask.bool().any(dim=-1)
        light_valid = light_mask.bool()

        agent_tokens = self.agent_encoder(agents) * agent_valid[..., None].to(agents.dtype)
        map_tokens = self.map_encoder(map_polylines, map_mask.bool()) * map_valid[..., None].to(map_polylines.dtype)
        map_tokens = add_sinusoidal_positions(map_tokens[:, None, :, :], self.scale_pos_embeds).squeeze(1)
        map_tokens = map_tokens * map_valid[..., None].to(map_tokens.dtype)
        for layer in self.map_layers:
            map_tokens = layer(map_tokens, map_mask=map_valid)

        light_tokens = self.light_encoder(lights) * light_valid[..., None].to(lights.dtype)
        latents = self.latent_tokens.view(1, 1, self.n_latents, self.d_model).expand(b, t, self.n_latents, self.d_model)
        tokens = torch.cat([latents, agent_tokens, light_tokens], dim=2)

        latent_mask = torch.ones((b, t, self.n_latents), dtype=torch.bool, device=tokens.device)
        token_mask = torch.cat([latent_mask, agent_valid, light_valid], dim=2)

        tokens = add_sinusoidal_positions(tokens, self.scale_pos_embeds)
        tokens = tokens * token_mask[..., None].to(tokens.dtype)

        map_query_slice = self._map_query_slice(k, l)
        for layer in self.layers:
            tokens = layer(
                tokens,
                token_mask=token_mask,
                map_tokens=map_tokens,
                map_mask=map_valid,
                map_query_slice=map_query_slice,
            )

        latent_slice = slice(0, self.n_latents)
        agent_slice = slice(self.n_latents, self.n_latents + k)
        light_slice = slice(agent_slice.stop, agent_slice.stop + l)

        z = torch.tanh(self.bottleneck_proj(tokens[:, :, latent_slice, :]))
        map_tokens_btmd = map_tokens[:, None, :, :].expand(b, t, m, self.d_model)
        return VectorEncoderOutput(
            z=z,
            agent_tokens=tokens[:, :, agent_slice, :],
            map_tokens=map_tokens_btmd,
            light_tokens=tokens[:, :, light_slice, :],
            token_mask=token_mask,
            map_token_mask=map_valid,
        )


def _collate(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    tensor_keys = [k for k, v in batch[0].items() if torch.is_tensor(v)]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def build_encoder_for_variant(
    *,
    encoder_variant: str,
    d_model: int,
    n_heads: int,
    depth: int,
    n_latents: int,
    d_bottleneck: int,
    hidden_dim: int,
    dropout: float,
    time_every: int,
    map_depth: int,
    map_cross_every: int,
    map_query_tokens: str,
) -> nn.Module:
    kwargs = dict(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        n_latents=n_latents,
        d_bottleneck=d_bottleneck,
        hidden_dim=hidden_dim,
        dropout=dropout,
        time_every=time_every,
    )
    if encoder_variant == "repeat_map":
        return VectorBlockCausalEncoder(**kwargs)
    if encoder_variant == "static_map_query":
        return VectorStaticMapQueryEncoder(
            **kwargs,
            map_depth=map_depth,
            map_cross_every=map_cross_every,
            map_query_tokens=map_query_tokens,
        )
    raise ValueError(f"Unknown encoder_variant: {encoder_variant}")


@torch.no_grad()
def smoke_test(
    data_dir: str,
    batch_size: int,
    time_window: int,
    encoder_variant: str,
    map_depth: int,
    map_cross_every: int,
    map_query_tokens: str,
) -> None:
    ds = WaymoVectorDataset(data_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    batch = next(iter(loader))
    if time_window > 0:
        # Dataset stores agents as (B,K,T,F), lights as (B,T,L,F).
        batch["agents"] = batch["agents"][:, :, :time_window]
        batch["lights"] = batch["lights"][:, :time_window]
        batch["light_mask"] = batch["light_mask"][:, :time_window]

    model = build_encoder_for_variant(
        encoder_variant=encoder_variant,
        d_model=128,
        n_heads=4,
        depth=3,
        n_latents=8,
        d_bottleneck=32,
        hidden_dim=64,
        dropout=0.0,
        time_every=1,
        map_depth=map_depth,
        map_cross_every=map_cross_every,
        map_query_tokens=map_query_tokens,
    ).eval()
    out = model(
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    print(f"batch agents: {tuple(batch['agents'].shape)}")
    print(f"batch map: {tuple(batch['map_polylines'].shape)}")
    print(f"batch lights: {tuple(batch['lights'].shape)}")
    print(f"encoder_variant: {encoder_variant}")
    print(f"z: {tuple(out.z.shape)}")
    print(f"agent_tokens: {tuple(out.agent_tokens.shape)}")
    print(f"map_tokens: {tuple(out.map_tokens.shape)}")
    print(f"light_tokens: {tuple(out.light_tokens.shape)}")
    print(f"token_mask valid: {int(out.token_mask.sum().item())}/{out.token_mask.numel()}")
    if not torch.isfinite(out.z).all():
        raise RuntimeError("Non-finite values in encoder z")


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test the Waymo vector tokenizer encoder.")
    p.add_argument("--data_dir", type=str, default="/p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--time_window", type=int, default=32)
    p.add_argument("--encoder_variant", choices=["repeat_map", "static_map_query"], default="repeat_map")
    p.add_argument("--map_depth", type=int, default=2)
    p.add_argument("--map_cross_every", type=int, default=1)
    p.add_argument("--map_query_tokens", choices=["latent", "agent", "latent_agent", "all"], default="latent_agent")
    args = p.parse_args()
    smoke_test(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        time_window=args.time_window,
        encoder_variant=args.encoder_variant,
        map_depth=args.map_depth,
        map_cross_every=args.map_cross_every,
        map_query_tokens=args.map_query_tokens,
    )


if __name__ == "__main__":
    main()
