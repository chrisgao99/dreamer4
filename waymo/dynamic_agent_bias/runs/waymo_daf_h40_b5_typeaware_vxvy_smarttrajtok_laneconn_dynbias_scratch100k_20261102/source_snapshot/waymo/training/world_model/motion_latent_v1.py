"""Shared motion/latent world-model components for the first kinematic V1 run.

The full world model keeps agent state ``q`` explicit.  Agent q tokens and
packed tokenizer latents share the same block-causal Transformer.  A motion
head predicts small physical residuals, hard integration produces q_next, and
q_next is injected into the latent endpoint before it is returned.  A small,
separately pretrained semantic reader is used to keep z_next readable as q_next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamer4.model import Dynamics, add_sinusoidal_positions
from waymo.core.vector_tokenizer_encoder import VectorBlockCausalLayer


AGENT_CONTINUOUS_SCALE = (20.0, 20.0, 10.0, 10.0, 10.0, 1.0, 1.0)
MOTION_DV_SCALE = 5.0
MOTION_DYAW_SCALE = 0.5
MOTION_CORRECTION_SCALE = 2.0


@dataclass(frozen=True)
class SemanticReaderOutput:
    continuous: torch.Tensor       # (B,T,K,7): x,y,speed,vx,vy,sin(yaw),cos(yaw)
    valid_logits: torch.Tensor     # (B,T,K)
    agent_tokens: torch.Tensor     # (B,T,K,D)


class LightweightAgentSemanticReader(nn.Module):
    """Agent-only two-block decoder P(q|z), with no q input and no map input."""

    def __init__(
        self,
        *,
        d_bottleneck: int = 64,
        d_model: int = 256,
        n_heads: int = 4,
        n_latents: int = 64,
        n_agents: int = 32,
        depth: int = 2,
        dropout: float = 0.05,
        mlp_ratio: float = 4.0,
        scale_pos_embeds: bool = True,
    ) -> None:
        super().__init__()
        self.d_bottleneck = int(d_bottleneck)
        self.d_model = int(d_model)
        self.n_latents = int(n_latents)
        self.n_agents = int(n_agents)
        self.depth = int(depth)
        self.scale_pos_embeds = bool(scale_pos_embeds)

        self.up_proj = nn.Linear(self.d_bottleneck, self.d_model)
        self.agent_queries = nn.Parameter(torch.empty(self.n_agents, self.d_model))
        nn.init.normal_(self.agent_queries, std=0.02)
        self.layers = nn.ModuleList(
            VectorBlockCausalLayer(
                d_model=self.d_model,
                n_heads=int(n_heads),
                dropout=float(dropout),
                mlp_ratio=float(mlp_ratio),
                layer_index=i,
                time_every=1,
            )
            for i in range(self.depth)
        )
        self.continuous_head = nn.Linear(self.d_model, 7)
        self.valid_head = nn.Linear(self.d_model, 1)

    @torch.no_grad()
    def init_from_tokenizer_decoder(self, decoder: nn.Module) -> None:
        """Warm-start from the already trained full tokenizer decoder."""
        if tuple(decoder.up_proj.weight.shape) != tuple(self.up_proj.weight.shape):
            raise ValueError("Tokenizer decoder and semantic reader dimensions do not match")
        self.up_proj.load_state_dict(decoder.up_proj.state_dict())
        self.agent_queries.copy_(decoder.agent_queries[: self.n_agents])
        self.continuous_head.load_state_dict(decoder.agent_continuous_head.state_dict())
        self.valid_head.load_state_dict(decoder.agent_valid_head.state_dict())
        if len(decoder.layers) < self.depth:
            raise ValueError(f"Tokenizer decoder has {len(decoder.layers)} layers, reader needs {self.depth}")
        for index, layer in enumerate(self.layers):
            layer.load_state_dict(decoder.layers[index].state_dict(), strict=True)

    def forward(self, z: torch.Tensor, *, agent_mask: Optional[torch.Tensor] = None) -> SemanticReaderOutput:
        if z.dim() != 4:
            raise ValueError(f"Expected z=(B,T,L,D), got {tuple(z.shape)}")
        bsz, time_steps, n_latents, _ = z.shape
        if n_latents != self.n_latents:
            raise ValueError(f"Expected {self.n_latents} latent slots, got {n_latents}")
        n_agents = self.n_agents if agent_mask is None else int(agent_mask.shape[-1])

        latents = torch.tanh(self.up_proj(z))
        queries = self.agent_queries[:n_agents].view(1, 1, n_agents, self.d_model)
        queries = queries.expand(bsz, time_steps, -1, -1)
        latent_mask = torch.ones((bsz, time_steps, n_latents), dtype=torch.bool, device=z.device)
        if agent_mask is None:
            query_mask = torch.ones((bsz, time_steps, n_agents), dtype=torch.bool, device=z.device)
        else:
            query_mask = agent_mask.to(device=z.device, dtype=torch.bool)[:, None].expand(-1, time_steps, -1)
        token_mask = torch.cat([latent_mask, query_mask], dim=2)
        tokens = torch.cat([latents, queries], dim=2)
        tokens = add_sinusoidal_positions(tokens, self.scale_pos_embeds)
        tokens = tokens * token_mask[..., None].to(tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens, token_mask=token_mask)
        agent_tokens = tokens[:, :, n_latents:]
        return SemanticReaderOutput(
            continuous=self.continuous_head(agent_tokens),
            valid_logits=self.valid_head(agent_tokens).squeeze(-1),
            agent_tokens=agent_tokens,
        )


def agent_semantic_targets(agents_btkf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw q=(x,y,speed,vx,vy,valid,yaw,type) to reader targets."""
    yaw = agents_btkf[..., 6]
    continuous = torch.cat(
        [agents_btkf[..., 0:5], torch.sin(yaw).unsqueeze(-1), torch.cos(yaw).unsqueeze(-1)],
        dim=-1,
    )
    return continuous, agents_btkf[..., 5]


def semantic_reader_loss(
    pred: SemanticReaderOutput,
    target_continuous: torch.Tensor,
    target_valid: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    validity_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Featurewise q reconstruction loss; intentionally no ADE or FDE objective."""
    mask = agent_mask.to(torch.bool)[:, None, :].expand_as(target_valid)
    valid_state = mask & (target_valid > 0.5)
    scales = target_continuous.new_tensor(AGENT_CONTINUOUS_SCALE)
    delta = (pred.continuous.float() - target_continuous.float()) / scales
    per_feature = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=0.1, reduction="none")
    denom = valid_state.sum().clamp_min(1).to(per_feature.dtype)
    continuous_loss = (per_feature * valid_state[..., None]).sum() / (denom * per_feature.shape[-1])
    valid_loss = F.binary_cross_entropy_with_logits(
        pred.valid_logits.float()[mask], target_valid.float()[mask], reduction="mean"
    )
    loss = continuous_loss + float(validity_weight) * valid_loss
    abs_err = (pred.continuous.float() - target_continuous.float()).abs()
    feature_mae = (abs_err * valid_state[..., None]).sum(dim=(0, 1, 2)) / denom
    metrics = {
        "loss_total": loss.detach(),
        "loss_continuous": continuous_loss.detach(),
        "loss_valid": valid_loss.detach(),
        "mae_x": feature_mae[0].detach(),
        "mae_y": feature_mae[1].detach(),
        "mae_speed": feature_mae[2].detach(),
        "mae_vx": feature_mae[3].detach(),
        "mae_vy": feature_mae[4].detach(),
        "valid_acc": ((pred.valid_logits[mask] > 0) == (target_valid[mask] > 0.5)).float().mean().detach(),
    }
    return loss, metrics


class MotionLatentDynamicsV1(nn.Module):
    """One shared Transformer with motion and q-conditioned latent heads."""

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
        self.n_agents = int(n_agents)
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
            n_spatial=int(n_spatial),
            n_register=int(n_register),
            n_agent=self.n_agents,
            n_heads=int(n_heads),
            depth=int(depth),
            k_max=int(k_max),
            dropout=float(dropout),
            mlp_ratio=float(mlp_ratio),
            time_every=int(time_every),
            space_mode="wm_agent",
            scale_pos_embeds=bool(scale_pos_embeds),
            action_clamp_inputs=bool(action_clamp_inputs),
            map_memory_dim=map_memory_dim,
            map_cross_every=int(map_cross_every),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 6),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

        self.latent_query_proj = nn.Linear(self.d_spatial, self.d_model)
        self.q_to_latent = nn.MultiheadAttention(self.d_model, int(n_heads), dropout=float(dropout), batch_first=True)
        self.q_latent_norm = nn.LayerNorm(self.d_model)
        self.q_latent_out = nn.Linear(self.d_model, self.d_spatial)
        nn.init.zeros_(self.q_latent_out.weight)
        nn.init.zeros_(self.q_latent_out.bias)

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

    def forward(
        self,
        actions: torch.Tensor,
        step_idxs: torch.Tensor,
        signal_idxs: torch.Tensor,
        packed_tokens: torch.Tensor,
        q_sequence: torch.Tensor,
        *,
        act_mask: Optional[torch.Tensor],
        agent_mask: torch.Tensor,
        map_tokens: Optional[torch.Tensor],
        map_mask: Optional[torch.Tensor],
        q_current: Optional[torch.Tensor] = None,
        action_slots: Optional[torch.Tensor] = None,
        kinematic_dt: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        q_tokens = self.encode_q(q_sequence, agent_mask)
        latent_base, agent_hidden = self.backbone(
            actions,
            step_idxs,
            signal_idxs,
            packed_tokens,
            act_mask=act_mask,
            agent_tokens=q_tokens,
            map_tokens=map_tokens,
            map_mask=map_mask,
        )
        if agent_hidden is None:
            raise RuntimeError("MotionLatentDynamicsV1 requires backbone agent tokens")
        motion_raw = self.motion_head(agent_hidden)
        q_next = None
        if q_current is not None:
            if action_slots is None:
                raise ValueError("action_slots is required when q_current is provided")
            q_next = integrate_motion(
                q_current,
                motion_raw[:, -1],
                actions_next=actions[:, -1],
                action_slots=action_slots,
                dt=kinematic_dt,
            )
            conditioned_last = self.condition_latent_on_q(latent_base[:, -1:], q_next, agent_mask)
            latent_base = torch.cat([latent_base[:, :-1], conditioned_last], dim=1)
        return latent_base, motion_raw, q_next

    def condition_latent_on_q(
        self,
        latent_base: torch.Tensor,
        q_next: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inject the hard-integrated q_next into the latent endpoint."""
        if q_next.dim() == 3:
            q_next = q_next[:, None]
        bsz, time_steps, n_spatial, _ = latent_base.shape
        q_tokens = self.encode_q(q_next, agent_mask).reshape(bsz * time_steps, self.n_agents, self.d_model)
        query = self.latent_query_proj(latent_base).reshape(bsz * time_steps, n_spatial, self.d_model)
        key_padding_mask = (~agent_mask.to(torch.bool))[:, None, :].expand(bsz, time_steps, -1)
        key_padding_mask = key_padding_mask.reshape(bsz * time_steps, self.n_agents)
        update, _ = self.q_to_latent(query, q_tokens, q_tokens, key_padding_mask=key_padding_mask, need_weights=False)
        update = self.q_latent_out(self.q_latent_norm(update)).reshape_as(latent_base)
        return latent_base + update


def motion_targets(
    q_current: torch.Tensor,
    q_gt_next: torch.Tensor,
    *,
    dt: float = 0.1,
) -> torch.Tensor:
    """Residual targets relative to the current *predicted* physical state."""
    target_dv = (q_gt_next[..., 3:5] - q_current[..., 3:5]) / MOTION_DV_SCALE
    target_dyaw = torch.atan2(
        torch.sin(q_gt_next[..., 6] - q_current[..., 6]),
        torch.cos(q_gt_next[..., 6] - q_current[..., 6]),
    ).unsqueeze(-1) / MOTION_DYAW_SCALE
    v_next = q_gt_next[..., 3:5]
    kinematic_xy = q_current[..., 0:2] + 0.5 * (q_current[..., 3:5] + v_next) * float(dt)
    target_correction = (q_gt_next[..., 0:2] - kinematic_xy) / MOTION_CORRECTION_SCALE
    return torch.cat([target_dv, target_dyaw, target_correction, q_gt_next[..., 5:6]], dim=-1)


def integrate_motion(
    q_current: torch.Tensor,
    motion_raw: torch.Tensor,
    *,
    actions_next: torch.Tensor,
    action_slots: torch.Tensor,
    dt: float = 0.1,
) -> torch.Tensor:
    """Hard constant-acceleration integration plus bounded learned correction."""
    dv = motion_raw[..., 0:2].clamp(-3.0, 3.0) * MOTION_DV_SCALE
    dyaw = motion_raw[..., 2].clamp(-3.0, 3.0) * MOTION_DYAW_SCALE
    correction = motion_raw[..., 3:5].clamp(-3.0, 3.0) * MOTION_CORRECTION_SCALE
    valid_prob = torch.sigmoid(motion_raw[..., 5])
    v_next = q_current[..., 3:5] + dv
    xy_next = q_current[..., 0:2] + 0.5 * (q_current[..., 3:5] + v_next) * float(dt) + correction
    yaw_next = torch.atan2(torch.sin(q_current[..., 6] + dyaw), torch.cos(q_current[..., 6] + dyaw))
    q_next = q_current.clone()
    q_next[..., 0:2] = xy_next
    q_next[..., 2] = torch.linalg.vector_norm(v_next, dim=-1)
    q_next[..., 3:5] = v_next
    q_next[..., 5] = valid_prob
    q_next[..., 6] = yaw_next

    # The controlled focus slot follows the supplied action exactly.  With a
    # GT action sequence this makes explicit what is conditioned rather than a
    # hidden trajectory loss.
    rows = torch.arange(q_next.shape[0], device=q_next.device)
    slots = action_slots.to(torch.long)
    prev_focus = q_current[rows, slots]
    focus = q_next[rows, slots]
    focus[..., 0:2] = prev_focus[..., 0:2] + actions_next[..., 0:2]
    focus[..., 2] = actions_next[..., 3]
    focus[..., 3:5] = actions_next[..., 4:6]
    focus[..., 5] = actions_next[..., 6]
    focus[..., 6] = torch.atan2(
        torch.sin(prev_focus[..., 6] + actions_next[..., 2]),
        torch.cos(prev_focus[..., 6] + actions_next[..., 2]),
    )
    q_next[rows, slots] = focus
    return q_next


def motion_residual_loss(
    motion_raw: torch.Tensor,
    target: torch.Tensor,
    q_current: torch.Tensor,
    q_gt_next: torch.Tensor,
    agent_mask: torch.Tensor,
    action_slots: torch.Tensor,
    *,
    validity_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz, n_agents = motion_raw.shape[:2]
    controlled = torch.zeros((bsz, n_agents), dtype=torch.bool, device=motion_raw.device)
    controlled[torch.arange(bsz, device=motion_raw.device), action_slots.long()] = True
    static = agent_mask.to(torch.bool) & ~controlled
    # Continuous residual targets only exist for GT-valid next states.  GT
    # invalid slots remain in the validity-classification loss below.
    state_mask = static & (q_gt_next[..., 5] > 0.5)
    target_residual = target[..., :5].clamp(-3.0, 3.0)
    per = F.smooth_l1_loss(motion_raw[..., :5].float(), target_residual.float(), beta=0.1, reduction="none")
    denom = state_mask.sum().clamp_min(1).to(per.dtype)
    residual_loss = (per * state_mask[..., None]).sum() / (denom * 5.0)
    if static.any():
        valid_loss = F.binary_cross_entropy_with_logits(
            motion_raw[..., 5].float()[static], target[..., 5].float()[static], reduction="mean"
        )
    else:
        valid_loss = motion_raw[..., 5].float().sum() * 0.0
    loss = residual_loss + float(validity_weight) * valid_loss
    target_abs = target_residual[state_mask].abs().mean() if state_mask.any() else target_residual.sum() * 0.0
    return loss, {
        "motion_total": loss.detach(),
        "motion_residual": residual_loss.detach(),
        "motion_valid": valid_loss.detach(),
        "motion_target_abs": target_abs.detach(),
    }


def latent_q_consistency_loss(
    pred: SemanticReaderOutput,
    q_struct: torch.Tensor,
    agent_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Align P(z_next) to stopgrad(q_struct); gradients flow only into z/model."""
    target_cont, target_valid = agent_semantic_targets(q_struct.detach()[:, None])
    return semantic_reader_loss(pred, target_cont, target_valid, agent_mask, validity_weight=0.2)
