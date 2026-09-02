"""Explicit-agent, metric-action flow model for Waymo simulation.

This module intentionally has no tokenizer or learned scene bottleneck.  Each
selected track owns one agent token, and the stochastic variable is the joint
H-step local action tensor ``(a_longitudinal, a_lateral, delta_yaw)``.

The first version uses a holonomic executor for every agent.  Agent type is a
static conditioning feature; validity is always a mask and is never generated
by the continuous flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_angle_rad(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def agents_to_bntf(agents: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    """Accept dataset ``(B,N,T,F)`` or legacy ``(B,T,N,F)`` layout."""
    if agents.dim() != 4:
        raise ValueError(f"Expected a rank-4 agent tensor, got {tuple(agents.shape)}")
    num_agents = int(agent_mask.shape[-1])
    if int(agents.shape[1]) == num_agents:
        return agents
    if int(agents.shape[2]) == num_agents:
        return agents.transpose(1, 2).contiguous()
    raise ValueError(
        f"Cannot identify agent axis for agents={tuple(agents.shape)}, "
        f"agent_mask={tuple(agent_mask.shape)}"
    )


def gather_agent_window(
    agents_bntf: torch.Tensor,
    anchors: torch.Tensor,
    *,
    history_length: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-example history ending at anchor and following future."""
    bsz, num_agents, total_steps, feat_dim = agents_bntf.shape
    if anchors.shape != (bsz,):
        raise ValueError(f"Expected anchors={(bsz,)}, got {tuple(anchors.shape)}")
    offsets = torch.arange(
        -int(history_length) + 1,
        int(horizon) + 1,
        device=agents_bntf.device,
    )
    indices = anchors[:, None] + offsets[None, :]
    if int(indices.min()) < 0 or int(indices.max()) >= total_steps:
        raise ValueError(
            f"Window indices [{int(indices.min())}, {int(indices.max())}] "
            f"outside T={total_steps}"
        )
    gather_index = indices[:, None, :, None].expand(-1, num_agents, -1, feat_dim)
    window = agents_bntf.gather(dim=2, index=gather_index)
    return window[:, :, :history_length], window[:, :, history_length:]


def select_window_anchors(
    agents_bntf: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    history_length: int,
    horizon: int,
    random_start: bool,
) -> torch.Tensor:
    """Choose anchors with a complete valid H-step focus plan when possible."""
    bsz, _, total_steps, _ = agents_bntf.shape
    first = int(history_length) - 1
    last = total_steps - int(horizon) - 1
    if last < first:
        raise ValueError(
            f"Need L+H={history_length + horizon} states, dataset only has {total_steps}"
        )

    focus_valid = (agents_bntf[:, 0, :, 5] > 0.5) & agent_mask[:, 0, None].bool()
    candidates = torch.arange(first, last + 1, device=agents_bntf.device)
    valid_candidates = []
    for anchor in range(first, last + 1):
        valid_candidates.append(focus_valid[:, anchor : anchor + horizon + 1].all(dim=1))
    valid_matrix = torch.stack(valid_candidates, dim=1)

    # Every prepared sample is centered on a current-valid focus at index 10,
    # but a track can disappear before all H future steps.  Prefer a complete
    # focus plan and fall back to the candidate with the most valid focus steps.
    future_scores = []
    for anchor in range(first, last + 1):
        future_scores.append(
            focus_valid[:, anchor : anchor + horizon + 1].float().sum(dim=1)
        )
    score_matrix = torch.stack(future_scores, dim=1)

    if random_start:
        random_score = torch.rand_like(score_matrix)
        random_score = random_score.masked_fill(~valid_matrix, -1.0)
        chosen = random_score.argmax(dim=1)
    else:
        # argmax returns the earliest complete candidate, normally index 10.
        chosen = valid_matrix.float().argmax(dim=1)
    no_complete = ~valid_matrix.any(dim=1)
    chosen = torch.where(no_complete, score_matrix.argmax(dim=1), chosen)
    return candidates[chosen]


@dataclass(frozen=True)
class ActionTargets:
    actions: torch.Tensor       # (B,N,H,3), metric local-frame action
    valid: torch.Tensor         # (B,N,H), contiguous validity from current
    future_pose: torch.Tensor   # (B,N,H,3), x/y/yaw
    current_pose: torch.Tensor  # (B,N,3), x/y/yaw
    agent_type: torch.Tensor    # (B,N)


def inverse_holonomic_actions(
    history: torch.Tensor,
    future: torch.Tensor,
    agent_mask: torch.Tensor,
) -> ActionTargets:
    """Invert logged poses recursively using the same holonomic executor.

    ``history`` and ``future`` use the raw feature order
    ``x,y,speed,vx,vy,valid,yaw,type``.  Only x, y, valid, yaw and type are
    accessed.  An agent is modeled only while it stays continuously valid from
    the current state; birth/reappearance is deliberately outside V1 scope.
    """
    if history.dim() != 4 or future.dim() != 4:
        raise ValueError("history and future must both have shape (B,N,T,F)")
    if history.shape[:2] != future.shape[:2] or history.shape[-1] < 8:
        raise ValueError(
            f"Incompatible history={tuple(history.shape)} future={tuple(future.shape)}"
        )
    current = history[:, :, -1]
    current_pose = torch.stack(
        (current[..., 0], current[..., 1], current[..., 6]), dim=-1
    )
    agent_type = current[..., 7].round().long().clamp(min=0)
    alive = agent_mask.bool() & (current[..., 5] > 0.5)
    pose = current_pose
    actions = []
    valids = []
    future_poses = []
    for step in range(int(future.shape[2])):
        nxt = future[:, :, step]
        nxt_pose = torch.stack((nxt[..., 0], nxt[..., 1], nxt[..., 6]), dim=-1)
        step_valid = alive & (nxt[..., 5] > 0.5)
        delta_world = nxt_pose[..., 0:2] - pose[..., 0:2]
        c = torch.cos(pose[..., 2])
        s = torch.sin(pose[..., 2])
        a_long = c * delta_world[..., 0] + s * delta_world[..., 1]
        a_lat = -s * delta_world[..., 0] + c * delta_world[..., 1]
        a_yaw = wrap_angle_rad(nxt_pose[..., 2] - pose[..., 2])
        action = torch.stack((a_long, a_lat, a_yaw), dim=-1)
        action = action * step_valid[..., None].to(action.dtype)
        actions.append(action)
        valids.append(step_valid)

        # Recursive execution.  For the all-holonomic inverse this is equal to
        # nxt_pose up to floating point error, but keeping the executor here
        # prevents target/inference semantics from silently diverging later.
        executed = execute_holonomic_step(pose, action)
        pose = torch.where(step_valid[..., None], executed, pose)
        future_poses.append(nxt_pose)
        alive = step_valid

    return ActionTargets(
        actions=torch.stack(actions, dim=2),
        valid=torch.stack(valids, dim=2),
        future_pose=torch.stack(future_poses, dim=2),
        current_pose=current_pose,
        agent_type=agent_type,
    )


def execute_holonomic_step(pose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Execute one local displacement using the pre-action heading."""
    c = torch.cos(pose[..., 2])
    s = torch.sin(pose[..., 2])
    dx = c * action[..., 0] - s * action[..., 1]
    dy = s * action[..., 0] + c * action[..., 1]
    return torch.stack(
        (
            pose[..., 0] + dx,
            pose[..., 1] + dy,
            wrap_angle_rad(pose[..., 2] + action[..., 2]),
        ),
        dim=-1,
    )


def execute_holonomic_actions(
    current_pose: torch.Tensor,
    actions: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute ``(B,N,H,3)`` actions and return ``(B,N,H,3)`` poses."""
    pose = current_pose
    output = []
    for step in range(int(actions.shape[2])):
        nxt = execute_holonomic_step(pose, actions[:, :, step])
        if valid is not None:
            nxt = torch.where(valid[:, :, step, None].bool(), nxt, pose)
        pose = nxt
        output.append(pose)
    return torch.stack(output, dim=2)


class ActionNormalizer(nn.Module):
    """Per-agent-type, per-action-channel affine normalization."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        if mean.ndim != 2 or mean.shape[-1] != 3 or std.shape != mean.shape:
            raise ValueError("Action statistics must have shape (num_types, 3)")
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-4))

    def _select(self, agent_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        index = agent_type.long().clamp(min=0, max=int(self.mean.shape[0]) - 1)
        return self.mean[index], self.std[index]

    def normalize(self, action: torch.Tensor, agent_type: torch.Tensor) -> torch.Tensor:
        mean, std = self._select(agent_type)
        return (action - mean.unsqueeze(-2)) / std.unsqueeze(-2)

    def denormalize(self, action: torch.Tensor, agent_type: torch.Tensor) -> torch.Tensor:
        mean, std = self._select(agent_type)
        return action * std.unsqueeze(-2) + mean.unsqueeze(-2)


class MultiheadAttention(nn.Module):
    """SDPA attention supporting per-head additive relation bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, *, cross: bool = False):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)
        self.cross = bool(cross)
        if self.cross:
            self.q = nn.Linear(d_model, d_model)
            self.kv = nn.Linear(d_model, 2 * d_model)
        else:
            self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        *,
        key_value: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, query_len, dim = query.shape
        memory = query if key_value is None else key_value
        key_len = int(memory.shape[1])
        if self.cross:
            q = self.q(query)
            k, v = self.kv(memory).chunk(2, dim=-1)
        else:
            if key_value is not None:
                raise ValueError("Self-attention does not accept separate key_value")
            q, k, v = self.qkv(query).chunk(3, dim=-1)
        q = q.view(bsz, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, key_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, key_len, self.n_heads, self.head_dim).transpose(1, 2)

        safe_key_mask = None
        if key_mask is not None:
            safe_key_mask = key_mask.bool().clone()
            empty = ~safe_key_mask.any(dim=1)
            if empty.any():
                safe_key_mask[empty, 0] = True
        attn_mask: Optional[torch.Tensor] = None
        if bias is not None:
            attn_mask = bias.to(dtype=q.dtype)
        if safe_key_mask is not None:
            allowed = safe_key_mask[:, None, None, :]
            if attn_mask is None:
                attn_mask = allowed
            else:
                attn_mask = attn_mask.masked_fill(~allowed, torch.finfo(q.dtype).min)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, query_len, dim)
        out = self.out(out)
        if query_mask is not None:
            out = out * query_mask[..., None].to(out.dtype)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ratio: float, dropout: float):
        super().__init__()
        hidden = int(round(d_model * ratio))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask_f = mask[..., None].to(x.dtype)
        x = x + self.attn(
            self.norm_attn(x), query_mask=mask, key_mask=mask, bias=bias
        )
        x = x + self.ffn(self.norm_ffn(x)) * mask_f
        return x * mask_f


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_memory = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads, dropout, cross=True)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_mask: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.attn(
            self.norm_q(query),
            key_value=self.norm_memory(memory),
            query_mask=query_mask,
            key_mask=memory_mask,
        )
        return (query + out) * query_mask[..., None].to(query.dtype)


class AgentHistoryEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        history_length: int,
        dropout: float,
        mlp_ratio: float,
        position_scale_m: float,
        max_agent_type: int,
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.position_scale_m = float(position_scale_m)
        self.state_mlp = nn.Sequential(
            nn.Linear(4, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.type_embed = nn.Embedding(max_agent_type, d_model)
        self.time_embed = nn.Parameter(torch.empty(history_length, d_model))
        nn.init.normal_(self.time_embed, std=0.02)
        self.layers = nn.ModuleList(
            MaskedTransformerBlock(d_model, n_heads, dropout, mlp_ratio)
            for _ in range(int(depth))
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, history: torch.Tensor, agent_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_agents, hist_steps, _ = history.shape
        if hist_steps != self.history_length:
            raise ValueError(f"Expected L={self.history_length}, got {hist_steps}")
        yaw = history[..., 6]
        state = torch.stack(
            (
                history[..., 0] / self.position_scale_m,
                history[..., 1] / self.position_scale_m,
                torch.sin(yaw),
                torch.cos(yaw),
            ),
            dim=-1,
        )
        agent_type = history[:, :, -1, 7].round().long().clamp(
            min=0, max=self.type_embed.num_embeddings - 1
        )
        valid = (history[..., 5] > 0.5) & agent_mask[:, :, None].bool()
        x = self.state_mlp(state)
        x = x + self.type_embed(agent_type)[:, :, None]
        x = x + self.time_embed[None, None]
        x = x * valid[..., None].to(x.dtype)
        x = x.reshape(bsz * num_agents, hist_steps, -1)
        flat_valid = valid.reshape(bsz * num_agents, hist_steps)
        for layer in self.layers:
            x = layer(x, flat_valid)
        x = self.final_norm(x).reshape(bsz, num_agents, hist_steps, -1)
        current_mask = valid[:, :, -1]
        token = x[:, :, -1] * current_mask[..., None].to(x.dtype)
        current_pose = torch.stack(
            (history[:, :, -1, 0], history[:, :, -1, 1], history[:, :, -1, 6]),
            dim=-1,
        )
        return token, current_mask, current_pose, agent_type


class PolylineEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        max_map_type: int,
        position_scale_m: float,
    ) -> None:
        super().__init__()
        self.position_scale_m = float(position_scale_m)
        self.type_embed = nn.Embedding(max_map_type, hidden_dim)
        self.pre = nn.Sequential(
            nn.Linear(5 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.post = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self, polylines: torch.Tensor, point_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lane_type = polylines[..., 4].round().long().clamp(
            min=0, max=self.type_embed.num_embeddings - 1
        )
        cont = torch.stack(
            (
                polylines[..., 0] / self.position_scale_m,
                polylines[..., 1] / self.position_scale_m,
                polylines[..., 2],
                polylines[..., 3],
                polylines[..., 5],
            ),
            dim=-1,
        )
        x = self.pre(torch.cat((cont, self.type_embed(lane_type)), dim=-1))
        valid = point_mask.bool()
        x = x.masked_fill(~valid[..., None], 0.0)
        pooled = x.masked_fill(~valid[..., None], -1e4).amax(dim=2)
        polyline_mask = valid.any(dim=2)
        pooled = torch.where(polyline_mask[..., None], pooled, torch.zeros_like(pooled))
        expanded = pooled[:, :, None].expand(-1, -1, int(x.shape[2]), -1)
        x = self.post(torch.cat((x, expanded), dim=-1))
        x = x.masked_fill(~valid[..., None], -1e4).amax(dim=2)
        x = torch.where(polyline_mask[..., None], x, torch.zeros_like(x))
        return x, polyline_mask


class CurrentLightEncoder(nn.Module):
    """Encode only the light states available at the current planning time."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        max_light_state: int,
        position_scale_m: float,
    ) -> None:
        super().__init__()
        self.position_scale_m = float(position_scale_m)
        self.state_embed = nn.Embedding(max_light_state, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, lights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        state = lights[..., 2].round().long().clamp(
            min=0, max=self.state_embed.num_embeddings - 1
        )
        cont = torch.stack(
            (
                lights[..., 0] / self.position_scale_m,
                lights[..., 1] / self.position_scale_m,
                lights[..., 3],
            ),
            dim=-1,
        )
        x = self.mlp(torch.cat((cont, self.state_embed(state)), dim=-1))
        return x * mask[..., None].to(x.dtype)


class RelativeAgentBias(nn.Module):
    def __init__(self, n_heads: int, hidden_dim: int, position_scale_m: float):
        super().__init__()
        self.position_scale_m = float(position_scale_m)
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_heads),
        )

    def forward(self, pose: torch.Tensor, agent_type: torch.Tensor) -> torch.Tensor:
        # Directed geometry j in the local frame of query agent i.
        delta = pose[:, None, :, 0:2] - pose[:, :, None, 0:2]
        yaw_i = pose[:, :, None, 2]
        c = torch.cos(yaw_i)
        s = torch.sin(yaw_i)
        dx = (c * delta[..., 0] + s * delta[..., 1]) / self.position_scale_m
        dy = (-s * delta[..., 0] + c * delta[..., 1]) / self.position_scale_m
        dyaw = pose[:, None, :, 2] - pose[:, :, None, 2]
        distance = torch.linalg.vector_norm(delta, dim=-1) / self.position_scale_m
        same_type = (agent_type[:, :, None] == agent_type[:, None, :]).to(pose.dtype)
        type_delta = (agent_type[:, None, :] - agent_type[:, :, None]).to(pose.dtype) / 4.0
        features = torch.stack(
            (dx, dy, torch.sin(dyaw), torch.cos(dyaw), distance, same_type, type_delta),
            dim=-1,
        )
        return self.mlp(features).permute(0, 3, 1, 2).contiguous()


class SceneInteractionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.map_light_cross = CrossAttentionBlock(d_model, n_heads, dropout)
        self.agent_self = MaskedTransformerBlock(
            d_model, n_heads, dropout, mlp_ratio
        )

    def forward(
        self,
        agents: torch.Tensor,
        agent_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        relative_bias: torch.Tensor,
    ) -> torch.Tensor:
        agents = self.map_light_cross(agents, memory, agent_mask, memory_mask)
        return self.agent_self(agents, agent_mask, bias=relative_bias)


@dataclass(frozen=True)
class SceneEncoding:
    agent_tokens: torch.Tensor
    agent_mask: torch.Tensor
    agent_pose: torch.Tensor
    agent_type: torch.Tensor
    map_tokens: torch.Tensor
    map_mask: torch.Tensor
    light_tokens: torch.Tensor
    light_mask: torch.Tensor
    memory: torch.Tensor
    memory_mask: torch.Tensor
    relative_bias: torch.Tensor


class DirectActionSceneEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        history_depth: int,
        map_depth: int,
        scene_depth: int,
        history_length: int,
        hidden_dim: int,
        dropout: float,
        mlp_ratio: float,
        position_scale_m: float,
        max_agent_type: int = 16,
        max_map_type: int = 64,
        max_light_state: int = 16,
    ) -> None:
        super().__init__()
        self.agent_encoder = AgentHistoryEncoder(
            d_model,
            n_heads,
            history_depth,
            history_length,
            dropout,
            mlp_ratio,
            position_scale_m,
            max_agent_type,
        )
        self.map_encoder = PolylineEncoder(
            d_model, hidden_dim, max_map_type, position_scale_m
        )
        self.map_layers = nn.ModuleList(
            MaskedTransformerBlock(d_model, n_heads, dropout, mlp_ratio)
            for _ in range(int(map_depth))
        )
        self.light_encoder = CurrentLightEncoder(
            d_model, hidden_dim, max_light_state, position_scale_m
        )
        self.relative_bias = RelativeAgentBias(
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            position_scale_m=position_scale_m,
        )
        self.scene_layers = nn.ModuleList(
            SceneInteractionBlock(d_model, n_heads, dropout, mlp_ratio)
            for _ in range(int(scene_depth))
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        *,
        history: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        current_lights: torch.Tensor,
        current_light_mask: torch.Tensor,
    ) -> SceneEncoding:
        agent_tokens, current_mask, pose, agent_type = self.agent_encoder(
            history, agent_mask
        )
        map_tokens, map_token_mask = self.map_encoder(map_polylines, map_mask)
        for layer in self.map_layers:
            map_tokens = layer(map_tokens, map_token_mask)
        light_mask = current_light_mask.bool()
        light_tokens = self.light_encoder(current_lights, light_mask)
        map_light_memory = torch.cat((map_tokens, light_tokens), dim=1)
        map_light_mask = torch.cat((map_token_mask, light_mask), dim=1)
        pair_bias = self.relative_bias(pose, agent_type)
        for layer in self.scene_layers:
            agent_tokens = layer(
                agent_tokens,
                current_mask,
                map_light_memory,
                map_light_mask,
                pair_bias,
            )
        agent_tokens = self.final_norm(agent_tokens)
        agent_tokens = agent_tokens * current_mask[..., None].to(agent_tokens.dtype)
        memory = torch.cat((agent_tokens, map_tokens, light_tokens), dim=1)
        memory_mask = torch.cat((current_mask, map_token_mask, light_mask), dim=1)
        return SceneEncoding(
            agent_tokens=agent_tokens,
            agent_mask=current_mask,
            agent_pose=pose,
            agent_type=agent_type,
            map_tokens=map_tokens,
            map_mask=map_token_mask,
            light_tokens=light_tokens,
            light_mask=light_mask,
            memory=memory,
            memory_mask=memory_mask,
            relative_bias=pair_bias,
        )


class FlowTimeEmbedding(nn.Module):
    def __init__(self, d_model: int, fourier_dim: int = 128):
        super().__init__()
        if fourier_dim % 2:
            raise ValueError("fourier_dim must be even")
        frequencies = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), fourier_dim // 2)
        )
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * flow_time.float()[:, None] * self.frequencies[None]
        return self.mlp(torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1))


def _modulate(x: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift, scale, gate = params.chunk(3, dim=-1)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
    return x * (1.0 + scale) + shift, torch.sigmoid(gate)


class JointActionDiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.norm_time = nn.LayerNorm(d_model, elementwise_affine=False)
        self.time_attn = MultiheadAttention(d_model, n_heads, dropout)
        self.mod_time = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))

        self.norm_agent = nn.LayerNorm(d_model, elementwise_affine=False)
        self.agent_attn = MultiheadAttention(d_model, n_heads, dropout)
        self.mod_agent = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))

        self.norm_cross = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_memory = nn.LayerNorm(d_model)
        self.scene_cross = MultiheadAttention(d_model, n_heads, dropout, cross=True)
        self.mod_cross = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))

        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = FeedForward(d_model, mlp_ratio, dropout)
        self.mod_ffn = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))

    def forward(
        self,
        x: torch.Tensor,
        chunk_mask: torch.Tensor,
        condition: torch.Tensor,
        scene: SceneEncoding,
    ) -> torch.Tensor:
        bsz, num_agents, num_chunks, dim = x.shape

        # Within-agent temporal attention over action chunks.
        flat = x.reshape(bsz * num_agents, num_chunks, dim)
        flat_mask = chunk_mask.reshape(bsz * num_agents, num_chunks)
        cond_time = condition[:, None].expand(-1, num_agents, -1).reshape(
            bsz * num_agents, dim
        )
        normed, gate = _modulate(self.norm_time(flat), self.mod_time(cond_time))
        flat = flat + gate * self.time_attn(
            normed, query_mask=flat_mask, key_mask=flat_mask
        )
        x = flat.reshape(bsz, num_agents, num_chunks, dim)

        # At each future chunk, jointly coordinate all agent plans.
        flat = x.permute(0, 2, 1, 3).reshape(bsz * num_chunks, num_agents, dim)
        flat_mask = chunk_mask.permute(0, 2, 1).reshape(bsz * num_chunks, num_agents)
        cond_agent = condition[:, None].expand(-1, num_chunks, -1).reshape(
            bsz * num_chunks, dim
        )
        relation_bias = scene.relative_bias[:, None].expand(
            -1, num_chunks, -1, -1, -1
        ).reshape(bsz * num_chunks, scene.relative_bias.shape[1], num_agents, num_agents)
        normed, gate = _modulate(self.norm_agent(flat), self.mod_agent(cond_agent))
        flat = flat + gate * self.agent_attn(
            normed,
            query_mask=flat_mask,
            key_mask=flat_mask,
            bias=relation_bias,
        )
        x = flat.reshape(bsz, num_chunks, num_agents, dim).permute(0, 2, 1, 3)

        # Every action chunk can query all contextual agent, map and light tokens.
        flat = x.reshape(bsz, num_agents * num_chunks, dim)
        flat_mask = chunk_mask.reshape(bsz, num_agents * num_chunks)
        normed, gate = _modulate(
            self.norm_cross(flat), self.mod_cross(condition)
        )
        cross = self.scene_cross(
            normed,
            key_value=self.norm_memory(scene.memory),
            query_mask=flat_mask,
            key_mask=scene.memory_mask,
        )
        flat = flat + gate * cross

        normed, gate = _modulate(self.norm_ffn(flat), self.mod_ffn(condition))
        flat = flat + gate * self.ffn(normed) * flat_mask[..., None].to(flat.dtype)
        return flat.reshape(bsz, num_agents, num_chunks, dim) * chunk_mask[
            ..., None
        ].to(flat.dtype)


class DirectActionFlowModel(nn.Module):
    """Scene-conditioned joint flow over explicit per-agent action sequences."""

    def __init__(
        self,
        *,
        d_model: int = 256,
        n_heads: int = 8,
        history_length: int = 11,
        horizon: int = 30,
        chunk_size: int = 5,
        history_depth: int = 2,
        map_depth: int = 2,
        scene_depth: int = 4,
        action_depth: int = 8,
        step_refiner_depth: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        mlp_ratio: float = 4.0,
        position_scale_m: float = 100.0,
    ) -> None:
        super().__init__()
        if horizon % chunk_size:
            raise ValueError(f"horizon={horizon} must be divisible by chunk_size={chunk_size}")
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.chunk_size = int(chunk_size)
        self.num_chunks = self.horizon // self.chunk_size
        self.scene_encoder = DirectActionSceneEncoder(
            d_model=d_model,
            n_heads=n_heads,
            history_depth=history_depth,
            map_depth=map_depth,
            scene_depth=scene_depth,
            history_length=history_length,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            position_scale_m=position_scale_m,
        )
        self.action_chunk_embed = nn.Sequential(
            nn.Linear(chunk_size * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.chunk_position = nn.Parameter(torch.empty(self.num_chunks, d_model))
        nn.init.normal_(self.chunk_position, std=0.02)
        self.flow_time = FlowTimeEmbedding(d_model)
        self.action_layers = nn.ModuleList(
            JointActionDiTBlock(d_model, n_heads, dropout, mlp_ratio)
            for _ in range(int(action_depth))
        )
        self.action_final_norm = nn.LayerNorm(d_model)
        self.chunk_to_steps = nn.Linear(d_model, chunk_size * d_model)
        self.noisy_step_embed = nn.Linear(3, d_model)
        self.step_position = nn.Parameter(torch.empty(chunk_size, d_model))
        nn.init.normal_(self.step_position, std=0.02)
        self.step_layers = nn.ModuleList(
            MaskedTransformerBlock(d_model, n_heads, dropout, mlp_ratio)
            for _ in range(int(step_refiner_depth))
        )
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3),
        )

    def encode_scene(self, **kwargs: torch.Tensor) -> SceneEncoding:
        return self.scene_encoder(**kwargs)

    def decode_velocity(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        scene: SceneEncoding,
        action_mask: torch.Tensor,
        *,
        focus_actions: Optional[torch.Tensor] = None,
        focus_index: int = 0,
    ) -> torch.Tensor:
        bsz, num_agents, horizon, channels = noisy_actions.shape
        if horizon != self.horizon or channels != 3:
            raise ValueError(
                f"Expected noisy actions (B,N,{self.horizon},3), got {tuple(noisy_actions.shape)}"
            )
        x_input = noisy_actions
        if focus_actions is not None:
            if focus_actions.shape != (bsz, horizon, 3):
                raise ValueError(
                    f"Expected focus_actions={(bsz, horizon, 3)}, got {tuple(focus_actions.shape)}"
                )
            x_input = x_input.clone()
            x_input[:, int(focus_index)] = focus_actions
        x_input = x_input * action_mask[..., None].to(x_input.dtype)
        chunks = x_input.reshape(
            bsz, num_agents, self.num_chunks, self.chunk_size * 3
        )
        chunk_mask = action_mask.reshape(
            bsz, num_agents, self.num_chunks, self.chunk_size
        ).any(dim=-1)
        x = self.action_chunk_embed(chunks)
        x = x + scene.agent_tokens[:, :, None]
        x = x + self.chunk_position[None, None]
        x = x * chunk_mask[..., None].to(x.dtype)
        condition = self.flow_time(flow_time).to(x.dtype)
        for layer in self.action_layers:
            x = layer(x, chunk_mask, condition, scene)
        x = self.action_final_norm(x)

        step_tokens = self.chunk_to_steps(x).reshape(
            bsz, num_agents, self.num_chunks, self.chunk_size, self.d_model
        )
        noisy_steps = x_input.reshape(
            bsz, num_agents, self.num_chunks, self.chunk_size, 3
        )
        step_tokens = step_tokens + self.noisy_step_embed(noisy_steps)
        step_tokens = step_tokens + self.step_position[None, None, None]
        step_mask = action_mask.reshape(
            bsz, num_agents, self.num_chunks, self.chunk_size
        )
        flat = step_tokens.reshape(-1, self.chunk_size, self.d_model)
        flat_mask = step_mask.reshape(-1, self.chunk_size)
        for layer in self.step_layers:
            flat = layer(flat, flat_mask)
        velocity = self.velocity_head(flat).reshape(
            bsz, num_agents, self.horizon, 3
        )
        return velocity * action_mask[..., None].to(velocity.dtype)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        flow_time: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        focus_actions: Optional[torch.Tensor] = None,
        focus_index: int = 0,
        **scene_kwargs: torch.Tensor,
    ) -> torch.Tensor:
        scene = self.encode_scene(**scene_kwargs)
        return self.decode_velocity(
            noisy_actions,
            flow_time,
            scene,
            action_mask,
            focus_actions=focus_actions,
            focus_index=focus_index,
        )

    @torch.no_grad()
    def sample_normalized_actions(
        self,
        scene: SceneEncoding,
        action_mask: torch.Tensor,
        focus_actions: torch.Tensor,
        *,
        solver_steps: int = 8,
        focus_index: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if solver_steps < 1:
            raise ValueError("solver_steps must be >= 1")
        x = torch.randn(
            (*action_mask.shape, 3),
            device=scene.agent_tokens.device,
            dtype=scene.agent_tokens.dtype,
            generator=generator,
        )
        generated_mask = action_mask.clone().bool()
        generated_mask[:, int(focus_index)] = False
        x = x * generated_mask[..., None].to(x.dtype)
        x[:, int(focus_index)] = focus_actions.to(x.dtype)
        dt = 1.0 / float(solver_steps)
        for step in range(int(solver_steps)):
            flow_time = torch.full(
                (int(x.shape[0]),),
                float(step) / float(solver_steps),
                device=x.device,
                dtype=torch.float32,
            )
            velocity = self.decode_velocity(
                x,
                flow_time,
                scene,
                action_mask,
                focus_actions=focus_actions,
                focus_index=focus_index,
            )
            x = x + dt * velocity * generated_mask[..., None].to(velocity.dtype)
            x[:, int(focus_index)] = focus_actions.to(x.dtype)
            x = x * action_mask[..., None].to(x.dtype)
        return x


def flow_matching_loss(
    model: DirectActionFlowModel,
    scene: SceneEncoding,
    normalized_actions: torch.Tensor,
    action_valid: torch.Tensor,
    *,
    focus_index: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Standard affine-path conditional flow-matching objective."""
    bsz = int(normalized_actions.shape[0])
    x0 = torch.randn_like(normalized_actions)
    flow_time = torch.rand((bsz,), device=normalized_actions.device)
    lam = flow_time[:, None, None, None].to(normalized_actions.dtype)
    x_lambda = (1.0 - lam) * x0 + lam * normalized_actions
    focus_actions = normalized_actions[:, int(focus_index)]
    x_lambda = x_lambda.clone()
    x_lambda[:, int(focus_index)] = focus_actions
    target_velocity = normalized_actions - x0
    # Future validity is a supervision mask, not a model input.  V1 keeps the
    # roster of current-valid agents fixed across H and therefore must not leak
    # logged disappearance times through the action attention mask.
    model_action_mask = scene.agent_mask[:, :, None].expand_as(action_valid)
    pred_velocity = model.decode_velocity(
        x_lambda,
        flow_time,
        scene,
        model_action_mask,
        focus_actions=focus_actions,
        focus_index=focus_index,
    )
    generated_valid = action_valid.clone().bool()
    generated_valid[:, int(focus_index)] = False
    weight = generated_valid[..., None].to(pred_velocity.dtype)
    denom = weight.sum().clamp_min(1.0) * pred_velocity.shape[-1]
    sq_error = (pred_velocity.float() - target_velocity.float()).pow(2)
    loss = (sq_error * weight.float()).sum() / denom.float()
    endpoint = x_lambda.float() + (1.0 - lam.float()) * pred_velocity.float()
    endpoint_mae = (
        (endpoint - normalized_actions.float()).abs() * weight.float()
    ).sum() / denom.float()
    return loss, {
        "flow_loss": loss.detach(),
        "normalized_endpoint_mae": endpoint_mae.detach(),
        "flow_time_mean": flow_time.mean().detach(),
        "valid_action_fraction": generated_valid.float().mean().detach(),
    }


@torch.no_grad()
def rollout_receding_horizon(
    model: DirectActionFlowModel,
    normalizer: ActionNormalizer,
    *,
    initial_history: torch.Tensor,
    agent_mask: torch.Tensor,
    map_polylines: torch.Tensor,
    map_mask: torch.Tensor,
    current_light_sequence: torch.Tensor,
    current_light_mask_sequence: torch.Tensor,
    focus_action_sequence: torch.Tensor,
    focus_action_valid: torch.Tensor,
    rollout_steps: int,
    commitment: int,
    solver_steps: int,
    focus_index: int = 0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Run closed-loop generate-H/execute-B simulation.

    Traffic lights are supplied only one current frame per replan through
    ``current_light_sequence``; the model is never given future light states
    inside an H-step plan.  Focus actions are exogenous and may be padded past
    the requested rollout using ``focus_action_valid=False``.

    Returns:
        Executed poses with shape ``(B,N,rollout_steps,3)``.
    """
    if commitment < 1 or commitment > model.horizon:
        raise ValueError(
            f"commitment must be in [1,{model.horizon}], got {commitment}"
        )
    bsz, num_agents, history_length, feature_dim = initial_history.shape
    if feature_dim < 8:
        raise ValueError("initial_history must use the raw >=8 feature layout")
    if int(current_light_sequence.shape[1]) < int(rollout_steps):
        raise ValueError("current_light_sequence is shorter than rollout_steps")
    if focus_action_sequence.shape[:2] != focus_action_valid.shape:
        raise ValueError("focus action values and validity have incompatible shapes")

    history = initial_history.clone()
    static_type = history[:, :, -1, 7].round().long().clamp(min=0)
    executed_parts = []
    elapsed = 0
    while elapsed < int(rollout_steps):
        available = int(focus_action_sequence.shape[1]) - elapsed
        take = min(model.horizon, max(0, available))
        focus_metric = history.new_zeros((bsz, model.horizon, 3))
        focus_valid = torch.zeros(
            (bsz, model.horizon), dtype=torch.bool, device=history.device
        )
        if take > 0:
            focus_metric[:, :take] = focus_action_sequence[:, elapsed : elapsed + take]
            focus_valid[:, :take] = focus_action_valid[:, elapsed : elapsed + take].bool()

        scene = model.encode_scene(
            history=history,
            agent_mask=agent_mask,
            map_polylines=map_polylines,
            map_mask=map_mask,
            current_lights=current_light_sequence[:, elapsed],
            current_light_mask=current_light_mask_sequence[:, elapsed],
        )
        action_mask = scene.agent_mask[:, :, None].expand(
            -1, -1, model.horizon
        ).clone()
        action_mask[:, int(focus_index)] = focus_valid
        focus_norm = normalizer.normalize(
            focus_metric[:, None], static_type[:, int(focus_index) : int(focus_index) + 1]
        )[:, 0]
        normalized = model.sample_normalized_actions(
            scene,
            action_mask,
            focus_norm,
            solver_steps=solver_steps,
            focus_index=focus_index,
            generator=generator,
        )
        metric_actions = normalizer.denormalize(normalized, static_type)
        execute_steps = min(int(commitment), int(rollout_steps) - elapsed)
        current_pose = torch.stack(
            (history[:, :, -1, 0], history[:, :, -1, 1], history[:, :, -1, 6]),
            dim=-1,
        )
        committed_valid = action_mask[:, :, :execute_steps]
        poses = execute_holonomic_actions(
            current_pose,
            metric_actions[:, :, :execute_steps],
            committed_valid,
        )
        executed_parts.append(poses)

        new_frames = history.new_zeros((bsz, num_agents, execute_steps, feature_dim))
        new_frames[..., 0:2] = poses[..., 0:2]
        new_frames[..., 5] = committed_valid.to(new_frames.dtype)
        new_frames[..., 6] = poses[..., 2]
        new_frames[..., 7] = static_type[:, :, None].to(new_frames.dtype)
        history = torch.cat((history, new_frames), dim=2)[:, :, -history_length:]
        elapsed += execute_steps
    return torch.cat(executed_parts, dim=2)
