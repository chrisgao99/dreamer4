"""Decoder for the Waymo vector tokenizer.

By default this is Decoder 1 from the tokenizer plan: reconstructed agent and
traffic light states are decoded only from bottleneck latents ``z`` plus learned
query tokens. Optional ablation flags can expose all or only the focus encoder
agent token to the decoder or interpret XY head outputs as per-step deltas.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
CORE_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, CORE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from vector_tokenizer_encoder import (
        VectorBlockCausalEncoder,
        VectorBlockCausalLayer,
        VectorEncoderOutput,
        VectorStaticMapQueryLayer,
        _collate,
    )
    from waymo_vector_dataset import WaymoVectorDataset
except ModuleNotFoundError:
    from waymo.core.vector_tokenizer_encoder import (
        VectorBlockCausalEncoder,
        VectorBlockCausalLayer,
        VectorEncoderOutput,
        VectorStaticMapQueryLayer,
        _collate,
    )
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset

from dreamer4.model import add_sinusoidal_positions


@dataclass(frozen=True)
class VectorDecoderOutput:
    agent_continuous: torch.Tensor      # (B,T,K,7): x,y,speed,vx,vy,sin(yaw),cos(yaw)
    agent_valid_logits: torch.Tensor    # (B,T,K)
    light_state_logits: torch.Tensor    # (B,T,L,num_light_states)
    light_valid_logits: torch.Tensor    # (B,T,L)
    agent_tokens: torch.Tensor          # (B,T,K,D)
    light_tokens: torch.Tensor          # (B,T,L,D)
    token_mask: torch.Tensor            # (B,T,S)
    agent_xy_gmm: Optional[torch.Tensor] = None  # (B,T,K,5): mux,muy,log_sx,log_sy,rho_raw


@dataclass(frozen=True)
class VectorTokenizerOutput:
    encoder: VectorEncoderOutput
    decoder: VectorDecoderOutput
    interaction: Optional["InteractionAuxOutput"] = None


@dataclass(frozen=True)
class InteractionAuxOutput:
    relevance_logits: torch.Tensor      # (B,T,K)
    type_logits: torch.Tensor           # (B,T,K,4)
    response_bin_logits: torch.Tensor   # (B,T,K,3)
    response_regression: torch.Tensor   # (B,T,K,1)
    pair_tokens: torch.Tensor           # (B,T,K,D)


def decoder_agent_xy(
    pred: VectorDecoderOutput,
    agent_xy_loss: str = "smooth_l1",
    agent_xy_parameterization: str = "absolute",
    *,
    anchor_xy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the reconstructed XY tensor used by the configured XY objective.

    XY metrics, FDE metrics, delta-XY losses, and visualizations should all use
    this helper so GMM checkpoints are evaluated from the GMM mean while
    smooth-L1 checkpoints are evaluated from the continuous head.
    """
    if agent_xy_loss == "smooth_l1":
        xy_head = pred.agent_continuous[..., 0:2]
    elif agent_xy_loss == "gmm":
        if pred.agent_xy_gmm is None:
            raise ValueError("agent_xy_loss='gmm' requires pred.agent_xy_gmm")
        xy_head = pred.agent_xy_gmm[..., 0:2]
    else:
        raise ValueError(f"Unknown agent_xy_loss: {agent_xy_loss}")

    if agent_xy_parameterization == "absolute":
        return xy_head
    if agent_xy_parameterization == "delta":
        if agent_xy_loss != "smooth_l1":
            raise ValueError("agent_xy_parameterization='delta' currently supports only agent_xy_loss='smooth_l1'")
        if anchor_xy is None:
            raise ValueError("agent_xy_parameterization='delta' requires anchor_xy with shape (B,K,2)")
        if anchor_xy.dim() != 3 or anchor_xy.shape[-1] != 2:
            raise ValueError(f"Expected anchor_xy with shape (B,K,2), got {tuple(anchor_xy.shape)}")
        zero_delta = torch.zeros_like(xy_head[:, :1])
        deltas = torch.cat([zero_delta, xy_head[:, 1:]], dim=1)
        return anchor_xy[:, None, :, :].to(device=xy_head.device, dtype=xy_head.dtype) + deltas.cumsum(dim=1)
    raise ValueError(f"Unknown agent_xy_parameterization: {agent_xy_parameterization}")


def _maybe_transpose_agents(agents: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    if agents.dim() != 4:
        raise ValueError(f"Expected agents with shape (B,K,T,F) or (B,T,K,F), got {tuple(agents.shape)}")
    k = agent_mask.shape[-1]
    if agents.shape[1] == k:
        return agents.transpose(1, 2).contiguous()
    return agents


class VectorBlockCausalTokenizerDecoder(nn.Module):
    """Dreamer-style block-causal decoder from latent tokens to vector states."""

    def __init__(
        self,
        *,
        d_bottleneck: int = 32,
        d_model: int = 256,
        n_heads: int = 4,
        depth: int = 4,
        n_latents: int = 8,
        n_agents: int = 32,
        n_lights: int = 16,
        num_light_states: int = 16,
        dropout: float = 0.05,
        mlp_ratio: float = 4.0,
        time_every: int = 1,
        scale_pos_embeds: bool = True,
        use_agent_tokens: bool = False,
        agent_token_mode: str = "none",
        attend_map: bool = False,
        map_cross_every: int = 1,
        map_query_tokens: str = "all",
        predict_agent_xy_gmm: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_latents = int(n_latents)
        self.n_agents = int(n_agents)
        self.n_lights = int(n_lights)
        self.num_light_states = int(num_light_states)
        self.scale_pos_embeds = bool(scale_pos_embeds)
        if bool(use_agent_tokens) and agent_token_mode == "none":
            agent_token_mode = "all"
        if agent_token_mode not in {"none", "all", "focus"}:
            raise ValueError(f"agent_token_mode must be one of none, all, focus; got {agent_token_mode!r}")
        self.agent_token_mode = agent_token_mode
        self.use_agent_tokens = self.agent_token_mode != "none"
        self.attend_map = bool(attend_map)
        self.map_cross_every = int(map_cross_every)
        if map_query_tokens not in {"latent", "agent", "light", "latent_agent", "agent_light", "all"}:
            raise ValueError(
                "map_query_tokens must be one of: latent, agent, light, latent_agent, agent_light, all; "
                f"got {map_query_tokens!r}"
            )
        self.map_query_tokens = str(map_query_tokens)
        self.predict_agent_xy_gmm = bool(predict_agent_xy_gmm)

        self.up_proj = nn.Linear(d_bottleneck, d_model)
        self.agent_skip_proj = nn.Linear(d_model, d_model) if self.use_agent_tokens else None
        self.agent_queries = nn.Parameter(torch.empty(n_agents, d_model))
        self.light_queries = nn.Parameter(torch.empty(n_lights, d_model))
        nn.init.normal_(self.agent_queries, std=0.02)
        nn.init.normal_(self.light_queries, std=0.02)

        self.layers = nn.ModuleList(
            [
                (
                    VectorStaticMapQueryLayer(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                        layer_index=i,
                        time_every=time_every,
                        map_cross_every=map_cross_every,
                    )
                    if self.attend_map
                    else VectorBlockCausalLayer(
                        d_model=d_model,
                        n_heads=n_heads,
                        dropout=dropout,
                        mlp_ratio=mlp_ratio,
                        layer_index=i,
                        time_every=time_every,
                    )
                )
                for i in range(depth)
            ]
        )

        self.agent_continuous_head = nn.Linear(d_model, 7)
        self.agent_xy_gmm_head = nn.Linear(d_model, 5) if self.predict_agent_xy_gmm else None
        self.agent_valid_head = nn.Linear(d_model, 1)
        self.light_state_head = nn.Linear(d_model, num_light_states)
        self.light_valid_head = nn.Linear(d_model, 1)

    def _map_query_slice(self, agent_skip_count: int, k: int, l: int) -> slice:
        latent_start = 0
        latent_end = self.n_latents
        agent_start = latent_end + agent_skip_count
        agent_end = agent_start + k
        light_start = agent_end
        light_end = light_start + l
        if self.map_query_tokens == "latent":
            return slice(latent_start, latent_end)
        if self.map_query_tokens == "agent":
            return slice(agent_start, agent_end)
        if self.map_query_tokens == "light":
            return slice(light_start, light_end)
        if self.map_query_tokens == "latent_agent":
            return slice(latent_start, agent_end)
        if self.map_query_tokens == "agent_light":
            return slice(agent_start, light_end)
        if self.map_query_tokens == "all":
            return slice(latent_start, light_end)
        raise ValueError(f"Unknown decoder map_query_tokens={self.map_query_tokens!r}")

    @staticmethod
    def _static_map_memory(
        encoder_map_tokens: torch.Tensor,
        encoder_map_mask: torch.Tensor,
        *,
        time_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_map_tokens.dim() == 4:
            if encoder_map_tokens.shape[1] != time_steps:
                raise ValueError(
                    "encoder_map_tokens with shape (B,T,M,D) must match decoder T; "
                    f"got T={encoder_map_tokens.shape[1]} expected {time_steps}"
                )
            encoder_map_tokens = encoder_map_tokens[:, 0]
        elif encoder_map_tokens.dim() != 3:
            raise ValueError(
                "encoder_map_tokens must have shape (B,M,D) or (B,T,M,D), "
                f"got {tuple(encoder_map_tokens.shape)}"
            )
        if encoder_map_mask.dim() != 2:
            raise ValueError(f"encoder_map_mask must have shape (B,M), got {tuple(encoder_map_mask.shape)}")
        return encoder_map_tokens, encoder_map_mask.to(device=encoder_map_tokens.device, dtype=torch.bool)

    def forward(
        self,
        z: torch.Tensor,
        *,
        agent_mask: Optional[torch.Tensor] = None,
        light_mask: Optional[torch.Tensor] = None,
        encoder_agent_tokens: Optional[torch.Tensor] = None,
        encoder_agent_mask: Optional[torch.Tensor] = None,
        encoder_map_tokens: Optional[torch.Tensor] = None,
        encoder_map_mask: Optional[torch.Tensor] = None,
    ) -> VectorDecoderOutput:
        """Decode bottleneck latents.

        Args:
            z: (B,T,N_latents,D_bottleneck)
            agent_mask: optional (B,K), True for selected agent slots
            light_mask: optional (B,T,L), True for valid light observations
        """
        if z.dim() != 4:
            raise ValueError(f"Expected z with shape (B,T,N_latents,D), got {tuple(z.shape)}")
        b, t, n_latents, _ = z.shape
        if n_latents != self.n_latents:
            raise ValueError(f"Expected {self.n_latents} latent slots, got {n_latents}")

        k = self.n_agents if agent_mask is None else int(agent_mask.shape[-1])
        l = self.n_lights if light_mask is None else int(light_mask.shape[-1])
        if k > self.n_agents:
            raise ValueError(f"Decoder was built for at most {self.n_agents} agent slots, got {k}")
        if l > self.n_lights:
            raise ValueError(f"Decoder was built for at most {self.n_lights} light slots, got {l}")

        latents = torch.tanh(self.up_proj(z))
        agent_queries = self.agent_queries[:k].view(1, 1, k, self.d_model).expand(b, t, k, self.d_model)
        light_queries = self.light_queries[:l].view(1, 1, l, self.d_model).expand(b, t, l, self.d_model)
        latent_mask = torch.ones((b, t, self.n_latents), dtype=torch.bool, device=z.device)
        if agent_mask is None:
            agent_query_mask = torch.ones((b, t, k), dtype=torch.bool, device=z.device)
        else:
            agent_query_mask = agent_mask.to(device=z.device, dtype=torch.bool)[:, None, :].expand(b, t, k)
        if light_mask is None:
            light_query_mask = torch.ones((b, t, l), dtype=torch.bool, device=z.device)
        else:
            light_ever_valid = light_mask.to(device=z.device, dtype=torch.bool).any(dim=1)
            light_query_mask = light_ever_valid[:, None, :].expand(b, t, l)
        token_parts = [latents]
        mask_parts = [latent_mask]
        agent_skip_count = 0
        if self.use_agent_tokens:
            if encoder_agent_tokens is None:
                raise ValueError("Decoder was built with use_agent_tokens=True but encoder_agent_tokens is None")
            if encoder_agent_tokens.shape[:3] != (b, t, k):
                raise ValueError(
                    "Expected encoder_agent_tokens with shape "
                    f"{(b, t, k, self.d_model)}, got {tuple(encoder_agent_tokens.shape)}"
                )
            if self.agent_token_mode == "focus":
                agent_skip_count = 1 if k > 0 else 0
            else:
                agent_skip_count = k
            skip_tokens = self.agent_skip_proj(encoder_agent_tokens[:, :, :agent_skip_count, :])
            if encoder_agent_mask is None:
                skip_mask = agent_query_mask[:, :, :agent_skip_count]
            else:
                skip_mask = encoder_agent_mask.to(device=z.device, dtype=torch.bool)[:, :, :agent_skip_count]
            token_parts.append(skip_tokens)
            mask_parts.append(skip_mask)
        token_parts.extend([agent_queries, light_queries])
        mask_parts.extend([agent_query_mask, light_query_mask])
        tokens = torch.cat(token_parts, dim=2)
        token_mask = torch.cat(mask_parts, dim=2)

        tokens = add_sinusoidal_positions(tokens, self.scale_pos_embeds)
        tokens = tokens * token_mask[..., None].to(tokens.dtype)
        map_query_slice = self._map_query_slice(agent_skip_count=agent_skip_count, k=k, l=l)
        if self.attend_map:
            if encoder_map_tokens is None or encoder_map_mask is None:
                raise ValueError("Decoder was built with attend_map=True but encoder_map_tokens or encoder_map_mask is None")
            encoder_map_tokens, encoder_map_mask = self._static_map_memory(
                encoder_map_tokens,
                encoder_map_mask,
                time_steps=t,
            )
        for layer in self.layers:
            if self.attend_map:
                tokens = layer(
                    tokens,
                    token_mask=token_mask,
                    map_tokens=encoder_map_tokens,
                    map_mask=encoder_map_mask,
                    map_query_slice=map_query_slice,
                )
            else:
                tokens = layer(tokens, token_mask=token_mask)

        agent_start = self.n_latents + agent_skip_count
        light_start = agent_start + k
        agent_tokens = tokens[:, :, agent_start:light_start, :]
        light_tokens = tokens[:, :, light_start : light_start + l, :]

        return VectorDecoderOutput(
            agent_continuous=self.agent_continuous_head(agent_tokens),
            agent_valid_logits=self.agent_valid_head(agent_tokens).squeeze(-1),
            light_state_logits=self.light_state_head(light_tokens),
            light_valid_logits=self.light_valid_head(light_tokens).squeeze(-1),
            agent_tokens=agent_tokens,
            light_tokens=light_tokens,
            token_mask=token_mask,
            agent_xy_gmm=None if self.agent_xy_gmm_head is None else self.agent_xy_gmm_head(agent_tokens),
        )


class TokenizerInteractionAuxHead(nn.Module):
    """Read interaction labels from z using decoder-style agent slot queries.

    The head intentionally receives no raw pair geometry. It forms focus-candidate
    slot queries from learned agent query embeddings, attends those queries to the
    same z latents that feed the decoder, and predicts pair labels for (focus=0,j).
    """

    def __init__(
        self,
        *,
        d_bottleneck: int,
        d_model: int,
        n_heads: int,
        n_latents: int,
        n_agents: int,
        dropout: float = 0.05,
        scale_pos_embeds: bool = True,
        n_types: int = 4,
        n_response_binary: int = 3,
        n_response_regression: int = 1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_latents = int(n_latents)
        self.n_agents = int(n_agents)
        self.scale_pos_embeds = bool(scale_pos_embeds)

        self.z_proj = nn.Linear(d_bottleneck, d_model)
        self.slot_queries = nn.Parameter(torch.empty(n_agents, d_model))
        nn.init.normal_(self.slot_queries, std=0.02)

        self.pair_query_mlp = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.relevance_head = nn.Linear(d_model, 1)
        self.type_head = nn.Linear(d_model, n_types)
        self.response_bin_head = nn.Linear(d_model, n_response_binary)
        self.response_reg_head = nn.Linear(d_model, n_response_regression)

    @torch.no_grad()
    def init_from_agent_queries(self, agent_queries: torch.Tensor) -> None:
        n = min(self.slot_queries.shape[0], agent_queries.shape[0])
        self.slot_queries[:n].copy_(agent_queries[:n].detach().to(device=self.slot_queries.device, dtype=self.slot_queries.dtype))

    def forward(self, z: torch.Tensor, *, agent_mask: Optional[torch.Tensor] = None) -> InteractionAuxOutput:
        if z.dim() != 4:
            raise ValueError(f"Expected z with shape (B,T,N_latents,D), got {tuple(z.shape)}")
        b, t, n_latents, _ = z.shape
        if n_latents != self.n_latents:
            raise ValueError(f"Expected {self.n_latents} latent slots, got {n_latents}")
        k = self.n_agents if agent_mask is None else int(agent_mask.shape[-1])
        if k > self.n_agents:
            raise ValueError(f"Interaction aux head was built for at most {self.n_agents} agent slots, got {k}")

        z_tokens = torch.tanh(self.z_proj(z))
        z_tokens = add_sinusoidal_positions(z_tokens, self.scale_pos_embeds)
        memory = z_tokens.reshape(b * t, n_latents, self.d_model)

        slots = self.slot_queries[:k]
        focus = slots[:1].expand(k, self.d_model)
        pair_query = torch.cat([focus, slots, focus * slots, (focus - slots).abs()], dim=-1)
        pair_query = self.pair_query_mlp(pair_query).view(1, 1, k, self.d_model).expand(b, t, k, self.d_model)
        query = pair_query.reshape(b * t, k, self.d_model)

        attended, _ = self.attn(query=query, key=memory, value=memory, need_weights=False)
        pair_tokens = self.norm(query + attended)
        pair_tokens = self.norm(pair_tokens + self.ff(pair_tokens))
        pair_tokens = pair_tokens.reshape(b, t, k, self.d_model)

        if agent_mask is not None:
            mask = agent_mask.to(device=z.device, dtype=torch.bool)[:, None, :, None]
            pair_tokens = pair_tokens * mask.to(dtype=pair_tokens.dtype)

        return InteractionAuxOutput(
            relevance_logits=self.relevance_head(pair_tokens).squeeze(-1),
            type_logits=self.type_head(pair_tokens),
            response_bin_logits=self.response_bin_head(pair_tokens),
            response_regression=self.response_reg_head(pair_tokens),
            pair_tokens=pair_tokens,
        )


class VectorTokenizer(nn.Module):
    """Encoder plus latent-only decoder wrapper."""

    def __init__(
        self,
        encoder: VectorBlockCausalEncoder,
        decoder: VectorBlockCausalTokenizerDecoder,
        interaction_aux: Optional[TokenizerInteractionAuxHead] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.interaction_aux = interaction_aux

    @torch.no_grad()
    def init_interaction_aux_from_decoder_queries(self) -> None:
        if self.interaction_aux is not None:
            self.interaction_aux.init_from_agent_queries(self.decoder.agent_queries)

    def forward(
        self,
        *,
        agents: torch.Tensor,
        agent_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_mask: torch.Tensor,
        lights: torch.Tensor,
        light_mask: torch.Tensor,
    ) -> VectorTokenizerOutput:
        enc = self.encoder(
            agents=agents,
            agent_mask=agent_mask,
            map_polylines=map_polylines,
            map_mask=map_mask,
            lights=lights,
            light_mask=light_mask,
        )
        decoder_kwargs = {}
        if getattr(self.decoder, "use_agent_tokens", False):
            agents_btkf = _maybe_transpose_agents(agents, agent_mask)
            decoder_kwargs["encoder_agent_tokens"] = enc.agent_tokens
            decoder_kwargs["encoder_agent_mask"] = (agents_btkf[..., 5] > 0.5) & agent_mask[:, None, :].to(
                device=agents_btkf.device, dtype=torch.bool
            )
        if getattr(self.decoder, "attend_map", False):
            decoder_kwargs["encoder_map_tokens"] = enc.map_tokens
            decoder_kwargs["encoder_map_mask"] = enc.map_token_mask
        dec = self.decoder(enc.z, agent_mask=agent_mask, light_mask=light_mask, **decoder_kwargs)
        interaction = None
        if self.interaction_aux is not None:
            interaction = self.interaction_aux(enc.z, agent_mask=agent_mask)
        return VectorTokenizerOutput(encoder=enc, decoder=dec, interaction=interaction)


def agent_reconstruction_targets(
    agents: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    already_btkf: bool = False,
) -> torch.Tensor:
    if not already_btkf:
        agents = _maybe_transpose_agents(agents, agent_mask)
    yaw = agents[..., 6]
    return torch.stack(
        [
            agents[..., 0],
            agents[..., 1],
            agents[..., 2],
            agents[..., 3],
            agents[..., 4],
            torch.sin(yaw),
            torch.cos(yaw),
        ],
        dim=-1,
    )


def normalized_agent_targets(
    agents: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    already_btkf: bool = False,
) -> torch.Tensor:
    """Backward-compatible alias for raw reconstruction targets."""
    return agent_reconstruction_targets(agents, agent_mask, already_btkf=already_btkf)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _weighted_masked_mean(values, mask.to(dtype=values.dtype))


def _weighted_masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(device=values.device, dtype=values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def _wrapped_angle_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs()


def _bivariate_gaussian_nll(
    pred_xy_gmm: torch.Tensor,
    target_xy: torch.Tensor,
    *,
    log_std_range: tuple[float, float] = (-1.609, 5.0),
    rho_limit: float = 0.5,
) -> torch.Tensor:
    mux, muy, log_sx, log_sy, rho_raw = pred_xy_gmm.unbind(dim=-1)
    log_sx = torch.clamp(log_sx, min=log_std_range[0], max=log_std_range[1])
    log_sy = torch.clamp(log_sy, min=log_std_range[0], max=log_std_range[1])
    sx = torch.exp(log_sx)
    sy = torch.exp(log_sy)
    rho = torch.clamp(torch.tanh(rho_raw), min=-rho_limit, max=rho_limit)

    dx = target_xy[..., 0] - mux
    dy = target_xy[..., 1] - muy
    one_minus_rho2 = torch.clamp(1.0 - rho * rho, min=1e-6)
    z = (dx / sx) ** 2 + (dy / sy) ** 2 - 2.0 * rho * dx * dy / (sx * sy)
    return log_sx + log_sy + 0.5 * torch.log(one_minus_rho2) + 0.5 * z / one_minus_rho2


def vector_tokenizer_reconstruction_loss(
    pred: VectorDecoderOutput,
    *,
    agents: torch.Tensor,
    agent_mask: torch.Tensor,
    lights: torch.Tensor,
    light_mask: torch.Tensor,
    agent_xy_weight: float = 1.0,
    agent_vel_weight: float = 0.5,
    agent_yaw_weight: float = 0.5,
    agent_valid_weight: float = 0.2,
    light_state_weight: float = 0.5,
    light_valid_weight: float = 0.1,
    agent_delta_xy_weight: float = 0.0,
    agent_fde_xy_weight: float = 0.0,
    agent_kinematic_xy_weight: float = 0.0,
    agent_speed_yaw_kinematic_weight: float = 0.0,
    kinematic_dt: float = 0.1,
    focus_agent_weight: float = 1.0,
    agent_xy_loss: str = "smooth_l1",
    agent_xy_parameterization: str = "absolute",
    agent_loss_weight_multiplier: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute first-pass reconstruction losses for tokenizer training."""
    agents_btkf = _maybe_transpose_agents(agents, agent_mask)
    agent_slot_mask = agent_mask[:, None, :].to(device=agents_btkf.device, dtype=torch.bool)
    agent_valid_target = (agents_btkf[..., 5] > 0.5) & agent_slot_mask
    agent_target = agent_reconstruction_targets(agents_btkf, agent_mask, already_btkf=True)
    agent_loss_weight = agent_valid_target.to(dtype=agents_btkf.dtype)
    if agent_loss_weight_multiplier is not None:
        mult = agent_loss_weight_multiplier.to(device=agents_btkf.device, dtype=agents_btkf.dtype)
        if mult.dim() == 2:
            mult = mult[:, None, :]
        if mult.shape != agent_loss_weight.shape:
            raise ValueError(
                "agent_loss_weight_multiplier must have shape "
                f"{tuple(agent_loss_weight.shape)} or {(agent_loss_weight.shape[0], agent_loss_weight.shape[2])}, "
                f"got {tuple(mult.shape)}"
            )
        agent_loss_weight = agent_loss_weight * mult
    if focus_agent_weight != 1.0 and agent_loss_weight.shape[-1] > 0:
        agent_loss_weight = agent_loss_weight.clone()
        agent_loss_weight[..., 0] = agent_loss_weight[..., 0] * float(focus_agent_weight)

    target_xy = agent_target[..., 0:2]
    anchor_xy = target_xy[:, 0] if agent_xy_parameterization == "delta" else None
    pred_xy = decoder_agent_xy(
        pred,
        agent_xy_loss,
        agent_xy_parameterization,
        anchor_xy=anchor_xy,
    )
    if agent_xy_loss == "smooth_l1":
        agent_xy_raw = F.smooth_l1_loss(pred_xy, target_xy, reduction="none")
        agent_xy_loss_value = _weighted_masked_mean(agent_xy_raw, agent_loss_weight)
    elif agent_xy_loss == "gmm":
        if agent_xy_parameterization != "absolute":
            raise ValueError("GMM XY loss supports only agent_xy_parameterization='absolute'")
        agent_xy_raw = _bivariate_gaussian_nll(pred.agent_xy_gmm, agent_target[..., 0:2])
        agent_xy_loss_value = _weighted_masked_mean(agent_xy_raw, agent_loss_weight)
    else:
        raise ValueError(f"Unknown agent_xy_loss: {agent_xy_loss}")
    agent_vel_raw = F.smooth_l1_loss(pred.agent_continuous[..., 2:5], agent_target[..., 2:5], reduction="none")
    agent_vel_loss = _weighted_masked_mean(agent_vel_raw, agent_loss_weight)
    agent_yaw_raw = F.smooth_l1_loss(pred.agent_continuous[..., 5:7], agent_target[..., 5:7], reduction="none")
    agent_yaw_loss = _weighted_masked_mean(agent_yaw_raw, agent_loss_weight)

    consecutive_valid = agent_valid_target[:, 1:, :] & agent_valid_target[:, :-1, :]
    consecutive_weight = consecutive_valid.to(dtype=agents_btkf.dtype)
    if agent_loss_weight_multiplier is not None:
        mult = agent_loss_weight_multiplier.to(device=agents_btkf.device, dtype=agents_btkf.dtype)
        if mult.dim() == 2:
            mult = mult[:, None, :]
        consecutive_weight = consecutive_weight * mult[:, 1:, :]
    if focus_agent_weight != 1.0 and consecutive_weight.shape[-1] > 0:
        consecutive_weight = consecutive_weight.clone()
        consecutive_weight[..., 0] = consecutive_weight[..., 0] * float(focus_agent_weight)
    pred_delta_xy = pred_xy[:, 1:, :, :] - pred_xy[:, :-1, :, :]
    target_delta_xy = target_xy[:, 1:, :, :] - target_xy[:, :-1, :, :]
    agent_delta_xy_raw = F.smooth_l1_loss(pred_delta_xy, target_delta_xy, reduction="none")
    agent_delta_xy_loss = _weighted_masked_mean(agent_delta_xy_raw, consecutive_weight)

    # Kinematic consistency ties reconstructed positions in meters to
    # reconstructed motion state in m/s.
    dt_scale = float(kinematic_dt)
    pred_vxvy_delta_xy = pred.agent_continuous[:, :-1, :, 3:5] * dt_scale
    agent_kinematic_xy_raw = F.smooth_l1_loss(pred_delta_xy, pred_vxvy_delta_xy, reduction="none")
    agent_kinematic_xy_loss = _weighted_masked_mean(agent_kinematic_xy_raw, consecutive_weight)

    pred_yaw_for_delta = torch.atan2(pred.agent_continuous[:, :-1, :, 5], pred.agent_continuous[:, :-1, :, 6])
    pred_speed_delta_xy = torch.stack(
        (
            pred.agent_continuous[:, :-1, :, 2] * torch.cos(pred_yaw_for_delta) * dt_scale,
            pred.agent_continuous[:, :-1, :, 2] * torch.sin(pred_yaw_for_delta) * dt_scale,
        ),
        dim=-1,
    )
    agent_speed_yaw_kinematic_raw = F.smooth_l1_loss(pred_delta_xy, pred_speed_delta_xy, reduction="none")
    agent_speed_yaw_kinematic_loss = _weighted_masked_mean(agent_speed_yaw_kinematic_raw, consecutive_weight)

    any_valid = agent_valid_target.any(dim=1)
    time_idx = torch.arange(agent_valid_target.shape[1], device=agent_valid_target.device).view(1, -1, 1)
    last_idx = torch.where(agent_valid_target, time_idx, torch.zeros_like(time_idx)).amax(dim=1)
    gather_idx = last_idx[:, None, :, None].expand(-1, 1, -1, 2)
    pred_final_xy = pred_xy.gather(dim=1, index=gather_idx).squeeze(1)
    target_final_xy = target_xy.gather(dim=1, index=gather_idx).squeeze(1)
    final_weight = any_valid.to(dtype=agents_btkf.dtype)
    if agent_loss_weight_multiplier is not None:
        mult = agent_loss_weight_multiplier.to(device=agents_btkf.device, dtype=agents_btkf.dtype)
        if mult.dim() == 3:
            mult = mult.gather(dim=1, index=last_idx[:, None, :]).squeeze(1)
        final_weight = final_weight * mult
    if focus_agent_weight != 1.0 and final_weight.shape[-1] > 0:
        final_weight = final_weight.clone()
        final_weight[..., 0] = final_weight[..., 0] * float(focus_agent_weight)
    agent_fde_xy_raw = F.smooth_l1_loss(pred_final_xy, target_final_xy, reduction="none")
    agent_fde_xy_loss = _weighted_masked_mean(agent_fde_xy_raw, final_weight)

    agent_valid_raw = F.binary_cross_entropy_with_logits(
        pred.agent_valid_logits,
        agent_valid_target.to(dtype=pred.agent_valid_logits.dtype),
        reduction="none",
    )
    agent_valid_loss = _masked_mean(agent_valid_raw, agent_slot_mask.expand_as(agent_valid_target))

    light_valid_target = light_mask.to(device=pred.light_valid_logits.device, dtype=torch.bool)
    light_state_target = lights[..., 2].long().clamp(min=0, max=pred.light_state_logits.shape[-1] - 1)
    if light_valid_target.any():
        light_state_loss = F.cross_entropy(pred.light_state_logits[light_valid_target], light_state_target[light_valid_target])
    else:
        light_state_loss = pred.light_state_logits.sum() * 0.0
    light_valid_loss = F.binary_cross_entropy_with_logits(
        pred.light_valid_logits,
        light_valid_target.to(dtype=pred.light_valid_logits.dtype),
    )

    pred_xy_m = pred_xy
    target_xy_m = agents_btkf[..., 0:2]
    agent_xy_mae_m = _masked_mean((pred_xy_m - target_xy_m).norm(dim=-1), agent_valid_target)
    agent_delta_xy_mae_m = _masked_mean((pred_delta_xy - target_delta_xy).norm(dim=-1), consecutive_valid)
    agent_kinematic_xy_mae_m = _masked_mean((pred_delta_xy - pred_vxvy_delta_xy).norm(dim=-1), consecutive_valid)
    agent_speed_yaw_kinematic_mae_m = _masked_mean(
        (pred_delta_xy - pred_speed_delta_xy).norm(dim=-1), consecutive_valid
    )
    agent_fde_mae_m = _masked_mean((pred_final_xy - target_final_xy).norm(dim=-1), any_valid)
    if agent_valid_target.shape[-1] > 0:
        focus_valid = agent_valid_target[..., 0]
        focus_agent_xy_mae_m = _masked_mean((pred_xy_m[..., 0, :] - target_xy_m[..., 0, :]).norm(dim=-1), focus_valid)
        focus_any_valid = any_valid[..., 0]
        focus_agent_fde_m = _masked_mean((pred_final_xy[..., 0, :] - target_final_xy[..., 0, :]).norm(dim=-1), focus_any_valid)
    else:
        focus_agent_xy_mae_m = pred_xy_m.sum() * 0.0
        focus_agent_fde_m = pred_xy_m.sum() * 0.0

    pred_speed_mps = pred.agent_continuous[..., 2]
    agent_speed_mae_mps = _masked_mean((pred_speed_mps - agents_btkf[..., 2]).abs(), agent_valid_target)

    pred_vel_mps = pred.agent_continuous[..., 3:5]
    target_vel_mps = agents_btkf[..., 3:5]
    agent_vxvy_mae_mps = _masked_mean((pred_vel_mps - target_vel_mps).norm(dim=-1), agent_valid_target)

    pred_yaw = torch.atan2(pred.agent_continuous[..., 5], pred.agent_continuous[..., 6])
    agent_yaw_mae_deg = _masked_mean(_wrapped_angle_error(pred_yaw, agents_btkf[..., 6]) * (180.0 / torch.pi), agent_valid_target)

    agent_valid_pred = pred.agent_valid_logits > 0.0
    agent_valid_acc = _masked_mean(
        (agent_valid_pred == agent_valid_target).to(dtype=pred.agent_valid_logits.dtype),
        agent_slot_mask.expand_as(agent_valid_target),
    )

    light_valid_pred = pred.light_valid_logits > 0.0
    light_valid_acc = (light_valid_pred == light_valid_target).to(dtype=pred.light_valid_logits.dtype).mean()
    if light_valid_target.any():
        light_state_pred = pred.light_state_logits.argmax(dim=-1)
        light_state_acc = (light_state_pred[light_valid_target] == light_state_target[light_valid_target]).float().mean()
    else:
        light_state_acc = pred.light_state_logits.sum() * 0.0

    total = (
        agent_xy_weight * agent_xy_loss_value
        + agent_vel_weight * agent_vel_loss
        + agent_yaw_weight * agent_yaw_loss
        + agent_valid_weight * agent_valid_loss
        + light_state_weight * light_state_loss
        + light_valid_weight * light_valid_loss
        + agent_delta_xy_weight * agent_delta_xy_loss
        + agent_fde_xy_weight * agent_fde_xy_loss
        + agent_kinematic_xy_weight * agent_kinematic_xy_loss
        + agent_speed_yaw_kinematic_weight * agent_speed_yaw_kinematic_loss
    )
    metrics = {
        "loss_total": total.detach(),
        "loss_agent_xy": agent_xy_loss_value.detach(),
        "loss_agent_vel": agent_vel_loss.detach(),
        "loss_agent_yaw": agent_yaw_loss.detach(),
        "loss_agent_valid": agent_valid_loss.detach(),
        "loss_agent_delta_xy": agent_delta_xy_loss.detach(),
        "loss_agent_fde_xy": agent_fde_xy_loss.detach(),
        "loss_agent_kinematic_xy": agent_kinematic_xy_loss.detach(),
        "loss_agent_speed_yaw_kinematic": agent_speed_yaw_kinematic_loss.detach(),
        "loss_light_state": light_state_loss.detach(),
        "loss_light_valid": light_valid_loss.detach(),
        "agent_xy_mae_m": agent_xy_mae_m.detach(),
        "agent_delta_xy_mae_m": agent_delta_xy_mae_m.detach(),
        "agent_kinematic_xy_mae_m": agent_kinematic_xy_mae_m.detach(),
        "agent_speed_yaw_kinematic_mae_m": agent_speed_yaw_kinematic_mae_m.detach(),
        "agent_fde_mae_m": agent_fde_mae_m.detach(),
        "focus_agent_xy_mae_m": focus_agent_xy_mae_m.detach(),
        "focus_agent_fde_m": focus_agent_fde_m.detach(),
        "agent_speed_mae_mps": agent_speed_mae_mps.detach(),
        "agent_vxvy_mae_mps": agent_vxvy_mae_mps.detach(),
        "agent_yaw_mae_deg": agent_yaw_mae_deg.detach(),
        "agent_valid_acc": agent_valid_acc.detach(),
        "light_state_acc": light_state_acc.detach(),
        "light_valid_acc": light_valid_acc.detach(),
    }
    return total, metrics


def _slice_time_window(batch: Dict[str, torch.Tensor], time_window: int) -> Dict[str, torch.Tensor]:
    if time_window <= 0:
        return batch
    batch = dict(batch)
    k = batch["agent_mask"].shape[-1]
    if batch["agents"].shape[1] == k:
        batch["agents"] = batch["agents"][:, :, :time_window]
    else:
        batch["agents"] = batch["agents"][:, :time_window]
    batch["lights"] = batch["lights"][:, :time_window]
    batch["light_mask"] = batch["light_mask"][:, :time_window]
    return batch


def smoke_test(data_dir: str, batch_size: int, time_window: int, backward: bool) -> None:
    ds = WaymoVectorDataset(data_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    batch = _slice_time_window(next(iter(loader)), time_window)

    _, _, k, _ = VectorBlockCausalEncoder._maybe_transpose_agents(batch["agents"], batch["agent_mask"]).shape
    l = batch["lights"].shape[2]
    encoder = VectorBlockCausalEncoder(
        d_model=128,
        n_heads=4,
        depth=3,
        n_latents=8,
        d_bottleneck=32,
        hidden_dim=64,
        dropout=0.0,
        time_every=1,
    )
    decoder = VectorBlockCausalTokenizerDecoder(
        d_bottleneck=32,
        d_model=128,
        n_heads=4,
        depth=3,
        n_latents=8,
        n_agents=k,
        n_lights=l,
        dropout=0.0,
        time_every=1,
    )
    model = VectorTokenizer(encoder=encoder, decoder=decoder)
    model.train(backward)

    out = model(
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    loss, metrics = vector_tokenizer_reconstruction_loss(
        out.decoder,
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    if backward:
        loss.backward()

    print(f"batch agents: {tuple(batch['agents'].shape)}")
    print(f"batch lights: {tuple(batch['lights'].shape)}")
    print(f"encoder z: {tuple(out.encoder.z.shape)}")
    print(f"decoder agent_continuous: {tuple(out.decoder.agent_continuous.shape)}")
    print(f"decoder agent_valid_logits: {tuple(out.decoder.agent_valid_logits.shape)}")
    print(f"decoder light_state_logits: {tuple(out.decoder.light_state_logits.shape)}")
    print(f"decoder light_valid_logits: {tuple(out.decoder.light_valid_logits.shape)}")
    print(f"decoder token_mask valid: {int(out.decoder.token_mask.sum().item())}/{out.decoder.token_mask.numel()}")
    print("losses: " + ", ".join(f"{k}={float(v.item()):.4f}" for k, v in metrics.items()))

    tensors = [
        out.encoder.z,
        out.decoder.agent_continuous,
        out.decoder.agent_valid_logits,
        out.decoder.light_state_logits,
        out.decoder.light_valid_logits,
        loss,
    ]
    if not all(torch.isfinite(x).all().item() for x in tensors):
        raise RuntimeError("Non-finite values in vector tokenizer smoke test")


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke-test the Waymo vector tokenizer encoder+decoder.")
    p.add_argument("--data_dir", type=str, default="/p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--time_window", type=int, default=16)
    p.add_argument("--no_backward", action="store_true", help="Skip backward pass in the smoke test.")
    args = p.parse_args()
    smoke_test(data_dir=args.data_dir, batch_size=args.batch_size, time_window=args.time_window, backward=not args.no_backward)


if __name__ == "__main__":
    main()
