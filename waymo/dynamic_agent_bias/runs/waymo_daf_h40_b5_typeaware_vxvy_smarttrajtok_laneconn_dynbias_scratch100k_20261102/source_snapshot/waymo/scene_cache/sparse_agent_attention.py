"""Memory-bounded agent-agent relation bias, with optional kNN sparsity.

All agent-agent attention in the direct-action-flow models (scene encoder
self-attention, decoder per-agent memory, ``JointActionDiTBlock.agent_attn``)
consumes the single pairwise bias returned by ``RelativeAgentBias``. Patching
that one module therefore changes every agent-agent interaction consistently.

``RelativeAgentBias`` runs a hidden-128 MLP on every (i, j) pair, so the dense
version needs O(N^2 * hidden) memory: ~13M pairs at 324 agents. This patch:

* ``knn=0`` (dense): the same function, evaluated in query-row chunks. Exact,
  so weights trained dense give identical outputs; only peak memory changes.
* ``knn=K``: each agent attends only to its K nearest valid agents within
  ``radius_m`` (itself included), following SMART/TrajTok's radius graph. The
  MLP runs only on those N*K pairs, and other pairs get ``MASK_VALUE``, which
  is equivalent to a gather-based sparse attention. Neighbors are rebuilt from
  the current pose at every encode, i.e. every replanning step.

The scene encoder calls ``agent_encoder`` (which yields the valid-agent mask)
before ``relative_bias``, so the mask is handed over through an attribute.
"""
from __future__ import annotations

import dataclasses
import types

import torch

MASK_VALUE = -1.0e4  # finite: exp() underflows to 0 without inf/NaN arithmetic
ROW_CHUNK = 64


def _pair_features(module, pose_q, pose_k, type_q, type_k):
    """RelativeAgentBias features for queries (B,Q,3) x keys (B,Q,K,3) or (B,K,3)."""
    if pose_k.dim() == 3:
        pose_k = pose_k[:, None].expand(-1, pose_q.shape[1], -1, -1)
        type_k = type_k[:, None].expand(-1, pose_q.shape[1], -1)
    delta = pose_k[..., 0:2] - pose_q[:, :, None, 0:2]
    yaw_i = pose_q[:, :, None, 2]
    c, s = torch.cos(yaw_i), torch.sin(yaw_i)
    scale = module.position_scale_m
    dx = (c * delta[..., 0] + s * delta[..., 1]) / scale
    dy = (-s * delta[..., 0] + c * delta[..., 1]) / scale
    dyaw = pose_k[..., 2] - pose_q[:, :, None, 2]
    distance = torch.linalg.vector_norm(delta, dim=-1) / scale
    same_type = (type_q[:, :, None] == type_k).to(pose_q.dtype)
    type_delta = (type_k - type_q[:, :, None]).to(pose_q.dtype) / 4.0
    return torch.stack((dx, dy, torch.sin(dyaw), torch.cos(dyaw), distance, same_type, type_delta), -1)


def _forward(self, pose, agent_type):
    bsz, n, _ = pose.shape
    alive = getattr(self, "_alive", None)
    if alive is None or alive.shape != (bsz, n):
        alive = torch.ones(bsz, n, dtype=torch.bool, device=pose.device)
    heads = self.mlp[-1].out_features
    if self.knn <= 0:
        rows = []
        for start in range(0, n, ROW_CHUNK):
            sl = slice(start, start + ROW_CHUNK)
            feats = _pair_features(self, pose[:, sl], pose, agent_type[:, sl], agent_type)
            rows.append(self.mlp(feats))
        return torch.cat(rows, 1).permute(0, 3, 1, 2).contiguous()

    k = min(self.knn, n)
    with torch.no_grad():
        distance = torch.cdist(pose[..., :2].float(), pose[..., :2].float())
        distance = distance.masked_fill(~alive[:, None, :], float("inf"))
        eye = torch.eye(n, dtype=torch.bool, device=pose.device)[None]
        distance = distance.masked_fill(eye, -1.0)  # self always first
        dist, idx = distance.topk(k, dim=-1, largest=False)
        keep = torch.isfinite(dist) & (dist <= self.radius_m)
    batch = torch.arange(bsz, device=pose.device)[:, None, None]
    feats = _pair_features(self, pose, pose[batch, idx], agent_type, agent_type[batch, idx])
    values = self.mlp(feats)  # (B, N, K, heads)
    out = values.new_full((bsz, n, n, heads), MASK_VALUE)
    values = torch.where(keep[..., None], values, torch.full_like(values, MASK_VALUE))
    out.scatter_(2, idx[..., None].expand(-1, -1, -1, heads), values)
    return out.permute(0, 3, 1, 2).contiguous()


def _wrap_agent_encoder(encoder, bias_module):
    original = encoder.forward

    def forward(*args, **kwargs):
        result = original(*args, **kwargs)
        bias_module._alive = result[1].bool()
        return result

    encoder.forward = forward


def patch_agent_attention(model, *, knn=0, radius_m=60.0):
    """Replace the model's pairwise agent bias in place (see module docstring)."""
    scene = model.scene_encoder
    bias = scene.relative_bias
    bias.knn = int(knn)
    bias.radius_m = float(radius_m)
    bias.forward = types.MethodType(_forward, bias)
    _wrap_agent_encoder(scene.agent_encoder, bias)
    return model


def patch_sdc_factorization(model):
    """WOSAC factorization: denoise the SDC independently of all other agents.

    Only ``JointActionDiTBlock.agent_attn`` mixes noisy action tokens across
    agents, and it reads ``scene.relative_bias``. The decoder gets a copy where
    every SDC<->other pair is masked; the scene encoder (history only) and the
    decoder's state-context memory stay shared, which the rules allow. Set the
    slot mask with ``set_sdc_mask(model, batch["agent_is_sdc"])``.
    """
    original = model.decode_velocity

    def decode_velocity(noisy_actions, flow_time, scene, action_mask, **kwargs):
        sdc = getattr(model, "_sdc_mask", None)
        if sdc is not None and sdc.shape == scene.agent_mask.shape and bool(sdc.any()):
            crossing = sdc[:, :, None] ^ sdc[:, None, :]
            bias = scene.relative_bias.masked_fill(crossing[:, None], MASK_VALUE)
            scene = dataclasses.replace(scene, relative_bias=bias)
        return original(noisy_actions, flow_time, scene, action_mask, **kwargs)

    model.decode_velocity = decode_velocity
    model.sdc_factorized = True
    return model


def set_sdc_mask(model, sdc_mask):
    if getattr(model, "sdc_factorized", False):
        model._sdc_mask = None if sdc_mask is None else sdc_mask.bool()


def patch_model(model, *, knn=0, radius_m=60.0, sdc_factorized=False):
    """Apply both patches; call on each model copy (e.g. after building the EMA)."""
    if getattr(model, "_scene_cache_patched", False):
        raise RuntimeError("Model already patched")
    patch_agent_attention(model, knn=knn, radius_m=radius_m)
    if sdc_factorized:
        patch_sdc_factorization(model)
    model._scene_cache_patched = True
    return model


def patch_from_args(model, args):
    """Patch a model from checkpoint/training args (``agent_knn`` etc.)."""
    return patch_model(model, knn=int(getattr(args, "agent_knn", 0) or 0),
                       radius_m=float(getattr(args, "agent_radius_m", 60.0) or 60.0),
                       sdc_factorized=bool(getattr(args, "sdc_factorized", False)))
