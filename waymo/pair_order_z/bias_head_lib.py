"""Shared pieces of the learned bias controller (plan BIAS_CONTROLLER_PLAN_2026_11_12 §2-§3).

- PairDeltaSteer: adds a per-row matrix delta (B, N query, N key) to the RelativeAgentBias entry on all
  8 heads, in all three places (scene-encoder agent self-attention, denoiser agent_attn, scene_cross
  agent part). Same knob as wm_interaction_probe/bias_steer.py BiasSteer, but per batch row.
- rollout(): the eval protocol of rollout_receding_horizon (generate-all, commitment 5, 8 solver steps)
  with an explicit noise per replan. At every replan: draft = sample with this replan's noise and no
  delta; conflict features of the draft; delta = delta_fn(features, gate); if any delta != 0, re-encode
  with delta and sample again with the SAME noise; execute the first 5 steps.
- pair_features(): ~23 features per directed vehicle pair i -> j from the current state and the draft.
  Gate: both are valid vehicles, their draft paths come within GATE_PATH_M, and at least one moves.
- BiasHead: per-directed-pair MLP, 3 classes = delta in {0, 2, 4}.
"""
from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn as nn

DELTAS = (0.0, 2.0, 4.0)
COMBOS = [(a, b) for a in (0, 2, 4) for b in (0, 2, 4)]          # index 0 = (0, 0) = base
ANCHOR, STEPS, COMMIT, SOLVER = 10, 80, 5, 8
GATE_PATH_M = 3.0      # draft paths within 3 m (any times)
MOVE_M = 1.0           # at least one car plans to move > 1 m in the 4 s draft
DT = 0.1
FEATURES = ("dx", "dy", "sin_dyaw", "cos_dyaw", "dist", "v_i", "v_j",
            "path_dmin", "t_i", "t_j", "dt_ij", "s_i", "s_j", "same_t_dmin", "same_t_tmin",
            "draft_hit", "t_hit", "min_gap", "sin_dh_conf", "cos_dh_conf", "plen_i", "plen_j", "overlap_now")
NFEAT = len(FEATURES)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


class PairDeltaSteer:
    """steer.delta = None or a (B, N, N) tensor [query, key]; added to every head of RelativeAgentBias."""

    def __init__(self, model):
        self.delta, self._base = None, None
        rb = model.scene_encoder.relative_bias
        rb_fwd, enc_fwd = rb.forward, model.encode_scene

        def bias_forward(pose, agent_type):
            base = rb_fwd(pose, agent_type)
            self._base = base
            return base if self.delta is None else base + self.delta[:, None].to(base.dtype)

        def encode_scene(**kw):
            scene = enc_fwd(**kw)
            if self.delta is None:
                return scene
            base, n = self._base, self._base.shape[-1]
            pair = base + self.delta[:, None].to(base.dtype)                   # (B, H, N, N)
            mem = scene.agent_memory_bias.clone()                              # (B, N q, H, 1, N + map + path)
            mem[..., :n] = pair.permute(0, 2, 1, 3).unsqueeze(-2).to(mem.dtype)
            return dataclasses.replace(scene, relative_bias=pair, agent_memory_bias=mem)

        rb.forward, model.encode_scene = bias_forward, encode_scene


def take_rows(rep, rows):
    """Index every batched tensor of a collated batch dict by rows (LongTensor)."""
    B = int(rep["agent_mask"].shape[0])
    return {k: (v[rows] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v) for k, v in rep.items()}


def make_noise(seeds, N, device, horizon=40):
    """One independent noise sequence per row: (B, R, N, H, 3); row noise depends only on its seed."""
    R = (STEPS + COMMIT - 1) // COMMIT
    out = torch.empty((len(seeds), R, N, horizon, 3), device=device)
    for b, s in enumerate(seeds):
        g = torch.Generator(device=device).manual_seed(int(s))
        out[b] = torch.randn((R, N, horizon, 3), device=device, generator=g)
    return out


def current_speed(history):
    xy, ok = history[..., 0:2].float(), history[..., 5] > 0.5
    v = torch.linalg.vector_norm(xy[:, :, -1] - xy[:, :, -2], dim=-1) / DT
    return torch.where(ok[:, :, -1] & ok[:, :, -2], v, torch.zeros_like(v))


@torch.no_grad()
def pair_features(plan, cur, speed, valid, types, chunk=16):
    """plan (B,N,H,3) draft poses, cur (B,N,3), speed (B,N), valid (B,N) bool, types (B,N).
    Returns feats (B,N,N,F) for directed pairs [i query, j key] and gate (B,N,N) bool."""
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    B, N, H, _ = plan.shape
    P = torch.cat((cur[:, :, None], plan), 2).float()                          # (B,N,H+1,3)
    xy, yaw = P[..., :2], P[..., 2]
    seg = torch.linalg.vector_norm(xy[:, :, 1:] - xy[:, :, :-1], dim=-1)
    cum = torch.cat((torch.zeros_like(seg[..., :1]), seg.cumsum(-1)), -1)     # (B,N,H+1) arc length
    plen = cum[..., -1]
    veh = valid & (types == 1)
    eye = torch.eye(N, dtype=torch.bool, device=P.device)
    feats = torch.zeros((B, N, N, NFEAT), device=P.device)
    gate = torch.zeros((B, N, N), dtype=torch.bool, device=P.device)
    T1 = H + 1
    tt = torch.arange(T1, device=P.device, dtype=torch.float32) * DT
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        X = xy[s:e]
        D = torch.linalg.vector_norm(X[:, :, None, :, None] - X[:, None, :, None, :], dim=-1)   # (b,N,N,T1,T1)
        dmin, idx = D.flatten(-2).min(-1)
        a, bb = idx // T1, idx % T1                                            # conflict-point step of i, of j
        ci = cum[s:e, :, None, :].expand(-1, -1, N, -1)
        cj = cum[s:e, None, :, :].expand(-1, N, -1, -1)
        s_i, s_j = ci.gather(-1, a[..., None])[..., 0], cj.gather(-1, bb[..., None])[..., 0]
        yi = yaw[s:e, :, None, :].expand(-1, -1, N, -1).gather(-1, a[..., None])[..., 0]
        yj = yaw[s:e, None, :, :].expand(-1, N, -1, -1).gather(-1, bb[..., None])[..., 0]
        dh = wrap(yi - yj)
        dsame = torch.linalg.vector_norm(X[:, :, None] - X[:, None], dim=-1)  # (b,N,N,T1)
        dsm, tsm = dsame.min(-1)
        g = collision_gap(P[s:e], types[s:e]).permute(0, 2, 3, 1)             # (b,N,N,T1)
        hit_t = g[..., 1:] < 0
        hit = hit_t.any(-1)
        t_hit = torch.where(hit, (hit_t.float().argmax(-1) + 1).float() * DT, torch.full_like(dsm, 5.0))
        min_gap = g[..., 1:].min(-1).values.clamp(-3.0, 10.0)
        c = cur[s:e].float()
        rel = c[:, None, :, :2] - c[:, :, None, :2]                           # j - i
        cy, sy = torch.cos(c[:, :, None, 2]), torch.sin(c[:, :, None, 2])
        dx, dy = cy * rel[..., 0] + sy * rel[..., 1], -sy * rel[..., 0] + cy * rel[..., 1]
        dyaw = wrap(c[:, None, :, 2] - c[:, :, None, 2])
        dist = torch.linalg.vector_norm(rel, dim=-1)
        vi, vj = speed[s:e, :, None].expand(-1, -1, N), speed[s:e, None, :].expand(-1, N, -1)
        pli, plj = plen[s:e, :, None].expand(-1, -1, N), plen[s:e, None, :].expand(-1, N, -1)
        t_i, t_j = a.float() * DT, bb.float() * DT
        f = torch.stack((dx / 20, dy / 20, torch.sin(dyaw), torch.cos(dyaw), dist / 20, vi / 10, vj / 10,
                         dmin / GATE_PATH_M, t_i / 4, t_j / 4, (t_i - t_j) / 4, s_i / 20, s_j / 20,
                         dsm / 10, tsm.float() * DT / 4, hit.float(), t_hit / 4, min_gap / 5,
                         torch.sin(dh), torch.cos(dh), pli / 40, plj / 40, (g[..., 0] < 0).float()), -1)
        feats[s:e] = f
        vv = veh[s:e, :, None] & veh[s:e, None, :] & ~eye
        gate[s:e] = vv & (dmin < GATE_PATH_M) & ((pli > MOVE_M) | (plj > MOVE_M))
    del tt
    return feats, gate


def scene_kwargs(model, rep, history, elapsed):
    lights = rep["lights"][:, ANCHOR:ANCHOR + STEPS]
    light_mask = rep["light_mask"][:, ANCHOR:ANCHOR + STEPS]
    kw = dict(history=history, agent_mask=rep["agent_mask"], map_polylines=rep["map_polylines"],
              map_mask=rep["map_mask"], current_lights=lights[:, elapsed], current_light_mask=light_mask[:, elapsed])
    if getattr(model, "map_adaptation", False):
        kw.update(map_ids=rep["map_ids"], map_is_lane=rep["map_is_lane"], map_stop_sign=rep["map_stop_sign"],
                  map_stop_point=rep["map_stop_point"],
                  current_light_ids=rep["light_ids"][:, ANCHOR:ANCHOR + STEPS][:, elapsed])
    if getattr(model, "lane_connectivity", False):
        kw.update(lane_edges=rep["lane_edges"], lane_edge_types=rep["lane_edge_types"])
    return kw


def _sub_kwargs(kw, rows, B):
    return {k: (v[rows] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v) for k, v in kw.items()}


@torch.inference_mode()
def rollout(model, normalizer, rep, history, noise, steer, delta_fn=None, record=False, force_steer=False,
            r_start=0, prefix=None, snapshots=False):
    """Closed loop, generate-all. noise (B,R,N,H,3). delta_fn(r, feats, gate) -> (B,N,N) or None.
    The steered second pass runs only on the rows with some delta != 0 (all rows with force_steer).
    r_start / prefix: resume at replan r_start from `history` (the history at that replan) with the
    already executed poses `prefix` (B,N,5*r_start,3) -- for forking a search off the base rollout.
    Returns poses (B,N,STEPS,3), records (if record: per-replan dicts with the gated directed pairs:
    idx (n,3) [row, i, j], feat (n,F) fp16, delta (n,) applied; CPU) and, if snapshots, the history at
    the start of every replan (list of (B,N,Th,F))."""
    from waymo.training.world_model.action_kinematics import execute_model_actions
    history = history.clone()
    B, N, Th, Fd = history.shape
    static_type = history[:, :, -1, 7].round().long().clamp(min=0)
    agent_lengths = rep.get("agent_lengths")
    parts, recs, snaps = ([prefix] if prefix is not None and prefix.shape[2] > 0 else []), [], []
    elapsed, r = COMMIT * r_start, r_start
    while elapsed < STEPS:
        if snapshots:
            snaps.append(history.clone())
        kw = scene_kwargs(model, rep, history, elapsed)
        steer.delta = None
        scene = model.encode_scene(**kw)
        action_mask = scene.agent_mask[:, :, None].expand(-1, -1, model.horizon).clone()
        norm = model.sample_normalized_actions(scene, action_mask, None, solver_steps=SOLVER, initial_noise=noise[:, r])
        current_pose = torch.stack((history[:, :, -1, 0], history[:, :, -1, 1], history[:, :, -1, 6]), dim=-1)
        if delta_fn is not None or record:
            metric_full = normalizer.denormalize(norm, static_type)
            plan = execute_model_actions(model, current_pose, metric_full, action_mask,
                                         agent_type=static_type, agent_lengths=agent_lengths)
            feats, gate = pair_features(plan, current_pose, current_speed(history), scene.agent_mask.bool(), static_type)
            delta = delta_fn(r, feats, gate) if delta_fn is not None else None
            if delta is not None:
                on = (delta != 0).flatten(1).any(1)
                if force_steer:
                    on = torch.ones_like(on)
                if bool(on.any()):
                    sub = on.nonzero()[:, 0]
                    steer.delta = delta[sub]
                    scene_d = model.encode_scene(**_sub_kwargs(kw, sub, B))
                    norm_d = model.sample_normalized_actions(scene_d, action_mask[sub], None, solver_steps=SOLVER,
                                                             initial_noise=noise[sub, r])
                    steer.delta = None
                    norm = norm.clone()
                    norm[sub] = norm_d
            if record:
                idx = gate.nonzero()
                d = delta[gate] if delta is not None else torch.zeros(len(idx), device=gate.device)
                recs.append(dict(r=r, idx=idx.int().cpu(), feat=feats[gate].half().cpu(), delta=d.float().cpu()))
        metric_actions = normalizer.denormalize(norm, static_type)
        execute_steps = min(COMMIT, STEPS - elapsed)
        committed_valid = action_mask[:, :, :execute_steps]
        poses = execute_model_actions(model, current_pose, metric_actions[:, :, :execute_steps], committed_valid,
                                      agent_type=static_type, agent_lengths=agent_lengths)
        parts.append(poses)
        new = history.new_zeros((B, N, execute_steps, Fd))
        new[..., 0:2] = poses[..., 0:2]
        if getattr(model, "include_agent_velocity", False):
            prev = torch.cat((current_pose[:, :, None, :2], poses[:, :, :-1, :2]), dim=2)
            vel = (poses[..., :2] - prev) / 0.1
            vel = torch.where(committed_valid[..., None], vel, torch.zeros_like(vel))
            new[..., 3:5] = vel
            new[..., 2] = torch.linalg.vector_norm(vel, dim=-1)
        new[..., 5] = committed_valid.to(new.dtype)
        new[..., 6] = poses[..., 2]
        new[..., 7] = static_type[:, :, None].to(new.dtype)
        history = torch.cat((history, new), dim=2)[:, :, -Th:]
        elapsed += execute_steps
        r += 1
    out = (torch.cat(parts, dim=2), (recs if record else None))
    return out + (snaps,) if snapshots else out


def gt_prep(agents_s, agent_mask_s, history_s, future_s):
    """Per-scene ground truth for the metrics (same as bias_ceiling.run_scenes). Inputs are row 0 of a scene."""
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    valid = (future_s[..., 5] > 0.5) & agent_mask_s.bool()[:, None]              # (N,T)
    types = agents_s[:, ANCHOR, 7].round().long()
    init = history_s[:, -1, [0, 1, 6]].float()
    fut = torch.where(valid[..., None], future_s[..., [0, 1, 6]].float(), torch.zeros_like(future_s[..., [0, 1, 6]].float()))
    v0 = torch.linalg.vector_norm(future_s[:, 0, 0:2].float() - init[:, :2], dim=-1) / 0.1
    N = types.shape[0]
    gt_gap = collision_gap(fut[None], types[None])[0]
    gt_both = (valid[:, None, :] & valid[None, :, :]).permute(2, 0, 1)
    log_overlap = ((gt_gap < 0) & gt_both & ~torch.eye(N, dtype=torch.bool, device=types.device)).any(0)
    return dict(valid=valid, types=types, init=init, fut=fut, v0=v0, log_overlap=log_overlap)


def real_collisions(P, g):
    """P (K,N,T,3); g = gt_prep(...). Returns real (K,T,N,N) bool (same rule as bias_ceiling)."""
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    K, N, T, _ = P.shape
    gap = collision_gap(P, g["types"][None].expand(K, -1))
    both = (g["valid"][:, None, :] & g["valid"][None, :, :]).permute(2, 0, 1)
    eye = torch.eye(N, dtype=torch.bool, device=P.device)
    raw = (gap < 0) & both[None] & ~eye
    return raw & ~(gap[:, :1] < 0) & ~g["log_overlap"][None, None]


class BiasHead(nn.Module):
    def __init__(self, nfeat=NFEAT, hidden=128, ncls=len(DELTAS)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nfeat, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, ncls))
        self.register_buffer("mean", torch.zeros(nfeat))
        self.register_buffer("std", torch.ones(nfeat))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def controller_fn(head, tau=0.5):
    """delta_fn for rollout(): delta != 0 when P(delta != 0) > tau; the value is the likelier of 2 and 4."""
    vals = torch.tensor(DELTAS)

    def fn(r, feats, gate):
        d = torch.zeros(gate.shape, device=gate.device)
        if gate.any():
            p = torch.softmax(head(feats[gate].float()), -1)
            on = (1 - p[:, 0]) > tau
            v = torch.where(p[:, 2] > p[:, 1], vals[2], vals[1]).to(d.device)
            d[gate] = torch.where(on, v, torch.zeros_like(v))
        return d
    return fn


def search_fn(q, k, dq, dk):
    """delta_fn for the label search: per row, (q->k, k->q) = (dq, dk) on the replans where the pair is gated."""
    def fn(r, feats, gate):
        B, N, _ = gate.shape
        d = torch.zeros((B, N, N), device=gate.device)
        ar = torch.arange(B, device=gate.device)
        d[ar, q, k] = dq * gate[ar, q, k].float()
        d[ar, k, q] = dk * gate[ar, k, q].float()
        return d
    return fn
