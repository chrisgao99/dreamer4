"""Single-q motion/latent dynamics with a four-step latent shortcut solver.

One physical transition predicts and integrates q exactly once from a
deterministic learned target query.  The resulting q_next is then held fixed
while the latent endpoint is refined over four shortcut solver substeps.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from dreamer4.model import Dynamics
from waymo.training.world_model.motion_latent_v1 import integrate_motion


class MotionLatentSingleQ4(nn.Module):
    """Shared backbone with one physical-q pass and repeated latent passes."""

    def __init__(
        self,
        *,
        d_model: int,
        d_bottleneck: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int,
        n_agents: int,
        n_heads: int,
        depth: int,
        k_max: int,
        dropout: float,
        mlp_ratio: float,
        time_every: int,
        scale_pos_embeds: bool,
        action_clamp_inputs: bool,
        map_memory_dim: Optional[int],
        map_cross_every: int,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_spatial = int(d_spatial)
        self.n_spatial = int(n_spatial)
        self.n_agents = int(n_agents)
        self.k_max = int(k_max)

        self.q_continuous_proj = nn.Sequential(
            nn.Linear(8, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.q_type_embed = nn.Embedding(5, self.d_model)
        self.q_slot_embed = nn.Parameter(torch.empty(self.n_agents, self.d_model))
        nn.init.normal_(self.q_slot_embed, std=0.02)

        self.backbone = Dynamics(
            d_model=self.d_model,
            d_bottleneck=int(d_bottleneck),
            d_spatial=self.d_spatial,
            n_spatial=self.n_spatial,
            n_register=int(n_register),
            n_agent=self.n_agents,
            n_heads=int(n_heads),
            depth=int(depth),
            k_max=self.k_max,
            dropout=float(dropout),
            mlp_ratio=float(mlp_ratio),
            time_every=int(time_every),
            space_mode="wm_agent",
            scale_pos_embeds=bool(scale_pos_embeds),
            action_clamp_inputs=bool(action_clamp_inputs),
            map_memory_dim=map_memory_dim,
            map_cross_every=int(map_cross_every),
        )

        # This query replaces the unknown target latent only in the single
        # deterministic motion pass.  It prevents q_next from depending on
        # the random latent solver initialization.
        self.motion_query = nn.Parameter(torch.empty(self.n_spatial, self.d_spatial))
        nn.init.normal_(self.motion_query, std=0.02)
        self.motion_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 6),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

        self.latent_query_proj = nn.Linear(self.d_spatial, self.d_model)
        self.q_to_latent = nn.MultiheadAttention(
            self.d_model,
            int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.q_latent_norm = nn.LayerNorm(self.d_model)
        self.q_latent_out = nn.Linear(self.d_model, self.d_spatial)
        nn.init.zeros_(self.q_latent_out.weight)
        nn.init.zeros_(self.q_latent_out.bias)

    @property
    def shortcut_steps(self) -> int:
        return 4

    @property
    def shortcut_step_idx(self) -> int:
        return int(round(math.log2(self.shortcut_steps)))

    def encode_q(self, q: torch.Tensor, agent_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        yaw = q[..., 6]
        continuous = torch.stack(
            (
                q[..., 0] / 20.0,
                q[..., 1] / 20.0,
                q[..., 2] / 10.0,
                q[..., 3] / 10.0,
                q[..., 4] / 10.0,
                q[..., 5],
                torch.sin(yaw),
                torch.cos(yaw),
            ),
            dim=-1,
        )
        types = q[..., 7].round().long().clamp(0, 4)
        out = self.q_continuous_proj(continuous) + self.q_type_embed(types)
        out = out + self.q_slot_embed[: q.shape[-2]].view(1, 1, q.shape[-2], self.d_model)
        if agent_mask is not None:
            out = out * agent_mask[:, None, :, None].to(out.dtype)
        return out

    def condition_latent_on_q(
        self,
        latent_endpoint: torch.Tensor,
        q_next: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        if q_next.dim() == 3:
            q_next = q_next[:, None]
        bsz, time_steps, n_spatial, _ = latent_endpoint.shape
        q_tokens = self.encode_q(q_next, agent_mask).reshape(
            bsz * time_steps, self.n_agents, self.d_model
        )
        query = self.latent_query_proj(latent_endpoint).reshape(
            bsz * time_steps, n_spatial, self.d_model
        )
        key_padding_mask = (~agent_mask.to(torch.bool))[:, None, :].expand(
            bsz, time_steps, -1
        )
        key_padding_mask = key_padding_mask.reshape(
            bsz * time_steps, self.n_agents
        )
        update, _ = self.q_to_latent(
            query,
            q_tokens,
            q_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        update = self.q_latent_out(self.q_latent_norm(update)).reshape_as(latent_endpoint)
        return latent_endpoint + update

    def predict_single_q(
        self,
        *,
        past_packed: torch.Tensor,
        past_q: torch.Tensor,
        q_current: torch.Tensor,
        actions_sequence: torch.Tensor,
        act_mask_sequence: torch.Tensor,
        action_slots: torch.Tensor,
        agent_mask: torch.Tensor,
        map_tokens: Optional[torch.Tensor],
        map_mask: Optional[torch.Tensor],
        kinematic_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict one full physical transition without seeing solver noise."""
        bsz, past_steps = past_packed.shape[:2]
        query = self.motion_query.view(1, 1, self.n_spatial, self.d_spatial).expand(
            bsz, 1, -1, -1
        )
        packed = torch.cat([past_packed, query], dim=1)
        q_sequence = torch.cat([past_q, q_current[:, None]], dim=1)
        q_tokens = self.encode_q(q_sequence, agent_mask)
        emax = int(round(math.log2(self.k_max)))
        step_idxs = torch.full(
            (bsz, past_steps + 1), emax, device=packed.device, dtype=torch.long
        )
        signal_idxs = torch.full_like(step_idxs, self.k_max - 1)
        step_idxs[:, -1] = 0
        signal_idxs[:, -1] = 0
        _, agent_hidden = self.backbone(
            actions_sequence,
            step_idxs,
            signal_idxs,
            packed,
            act_mask=act_mask_sequence,
            agent_tokens=q_tokens,
            map_tokens=map_tokens,
            map_mask=map_mask,
        )
        if agent_hidden is None:
            raise RuntimeError("Single-q motion pass requires agent hidden states")
        motion_raw = self.motion_head(agent_hidden[:, -1])
        q_next = integrate_motion(
            q_current,
            motion_raw,
            actions_next=actions_sequence[:, -1],
            action_slots=action_slots,
            dt=kinematic_dt,
        )
        return motion_raw, q_next

    def predict_latent_endpoint(
        self,
        *,
        past_packed: torch.Tensor,
        past_q: torch.Tensor,
        z_tau: torch.Tensor,
        q_condition: torch.Tensor,
        actions_sequence: torch.Tensor,
        act_mask_sequence: torch.Tensor,
        agent_mask: torch.Tensor,
        map_tokens: Optional[torch.Tensor],
        map_mask: Optional[torch.Tensor],
        tau_index: int,
    ) -> torch.Tensor:
        """Predict z_{t+1} endpoint for one of the four latent solver steps."""
        bsz, past_steps = past_packed.shape[:2]
        packed = torch.cat([past_packed, z_tau[:, None]], dim=1)
        q_sequence = torch.cat([past_q, q_condition[:, None]], dim=1)
        q_tokens = self.encode_q(q_sequence, agent_mask)
        emax = int(round(math.log2(self.k_max)))
        step_idxs = torch.full(
            (bsz, past_steps + 1), emax, device=packed.device, dtype=torch.long
        )
        signal_idxs = torch.full_like(step_idxs, self.k_max - 1)
        step_idxs[:, -1] = self.shortcut_step_idx
        signal_idxs[:, -1] = int(tau_index)
        latent_base, _ = self.backbone(
            actions_sequence,
            step_idxs,
            signal_idxs,
            packed,
            act_mask=act_mask_sequence,
            agent_tokens=q_tokens,
            map_tokens=map_tokens,
            map_mask=map_mask,
        )
        return self.condition_latent_on_q(
            latent_base[:, -1:], q_condition, agent_mask
        )[:, 0]

