#!/usr/bin/env python3
"""Ceiling of the attention-bias knob on the 512 eval scenes (plan BIAS_CONTROLLER_PLAN_2026_11_12 step 1).

The frozen 85k stage-1 WM, eval protocol (EMA weights, generate-all, 80 steps, commitment 5, 8 solver
steps, fp32). Per scene, K rollouts share one generator seed, so rollout k gets the same noise in every
condition (paired). Steering = wm_interaction_probe/bias_steer.py BiasSteer: +delta on all 8 heads of the
RelativeAgentBias entry, in all three places (encoder agent self-attention, denoiser agent_attn,
scene_cross agent part), every replan -- the same knob as the manual 6062/2310 experiments.

Per scene:
  1. base rollouts (no delta). "Real" collisions = proxy-box overlap of two GT-valid agents, excluding
     pairs that already overlap in the log (proxy false alarms, parked cars, groups of pedestrians) and
     pairs already overlapping at the first step. Target pair = the real colliding pair (not ped-ped)
     that collides in the most base rollouts; scenes without one keep the base result.
  2. for the target pair (q, k): every (delta q->k, delta k->q) in {0, 2, 4}^2 \ {(0, 0)}, same seeds.
Cost per rollout = target-pair collision + other real collisions + lambda * (q or k stalls)
+ mu * (increase of ADE_q + ADE_k over the base rollout, metres); a stall is a car that moves > 5 m in
the log but < 30% of that in the rollout. Default mu = 0.05/m (one collision ~ 20 m of extra ADE).
``--summarize_only`` recomputes the ceilings from the saved per-scene files with other lambda / mu.
Ceilings: per scene (one constant delta pair per scene, chosen by mean cost over its K rollouts) and per
rollout (best delta pair for each noise; what a controller reading the noise-specific draft could reach).
Per-scene results go to <out>/scenes/<dataset_index>.npz (resumable); summary to <out>/summary.{json,md}.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from waymo.pair_order_z import compat  # noqa: E402

sys.path.insert(0, str(compat.TRI30 / "wm_interaction_probe"))
sys.path.insert(0, str(compat.TRI30 / "waymo_interaction_screening/code"))

COMBOS = [(a, b) for a in (0, 2, 4) for b in (0, 2, 4)]          # index 0 = (0, 0) = base
HEADS = list(range(8))
ANCHOR, STEPS = 10, 80
METRICS = ("pair_hit", "other_hit", "real_hit", "proxy_hit", "stall_q", "stall_k", "never_q", "never_k",
           "ade_q", "ade_k", "ade_scene")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "runs/bias_ceiling_85k_20261112"))
    p.add_argument("--checkpoint", default=str(compat.STAGE1_CKPT))
    p.add_argument("--scene_source", default=str(Path(__file__).resolve().parent /
                                                    "runs/exp1_oracle_z_frozen_20261110/eval/unknown/rollout_batches"))
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--seed", type=int, default=12346)
    p.add_argument("--lambda_stall", type=float, default=1.0)
    p.add_argument("--mu_ade", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--summarize_only", action="store_true")
    return p.parse_args()


def scene_metrics(poses, init, future, valid, types, log_overlap, q, k, v0):
    """poses (K,N,T,3) on GPU. Returns dict of (K,) arrays and the (K,N,N) real-hit matrix."""
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    K, N, T, _ = poses.shape
    gap = collision_gap(poses, types[None].expand(K, -1))                         # (K,T,N,N)
    both = (valid[:, None, :] & valid[None, :, :]).permute(2, 0, 1)                # (T,N,N)
    eye = torch.eye(N, dtype=torch.bool, device=poses.device)
    raw = (gap < 0) & both[None] & ~eye
    real = raw & ~(gap[:, :1] < 0) & ~log_overlap[None, None]
    real_pair = real.any(1)                                                        # (K,N,N)
    out = {"real_hit": real_pair.flatten(1).any(1), "proxy_hit": raw.flatten(1).any(1)}
    xy = poses[..., :2]
    gt = future[..., :2]
    err = torch.linalg.vector_norm(xy - gt[None], dim=-1)                          # (K,N,T)
    vf = valid.float()
    out["ade_scene"] = (err * vf).sum((1, 2)) / vf.sum().clamp_min(1)
    seg_log = torch.linalg.vector_norm(torch.diff(torch.cat((init[:, None, :2], gt), 1), dim=1), dim=-1) * vf
    log_len = seg_log.sum(-1)                                                      # (N,)
    roll = torch.cat((init[None, :, None, :2].expand(K, -1, -1, -1), xy), 2)
    roll_len = torch.linalg.vector_norm(torch.diff(roll, dim=2), dim=-1).sum(-1)   # (K,N)
    if q is None:
        for m in METRICS:
            out.setdefault(m, torch.zeros(K, device=poses.device))
        return {m: v.float().cpu().numpy() for m, v in out.items()}, real_pair
    out["pair_hit"] = real_pair[:, q, k] | real_pair[:, k, q]
    others = real_pair.clone()
    others[:, q, k] = others[:, k, q] = False
    out["other_hit"] = others.flatten(1).any(1)
    for tag, a in (("q", q), ("k", k)):
        moved = log_len[a] > 5.0
        out[f"stall_{tag}"] = moved & (roll_len[:, a] < 0.3 * log_len[a])
        out[f"never_{tag}"] = moved & (v0[a] < 0.5) & (roll_len[:, a] < 1.0)
        out[f"ade_{tag}"] = (err[:, a] * vf[a]).sum(-1) / vf[a].sum().clamp_min(1)
    return {m: v.float().cpu().numpy() for m, v in out.items()}, real_pair


def run_scenes(a):
    compat.patch_map_cache_identity()
    from waymo.evaluation import eval_waymo_direct_action_flow_multisample as ev
    from waymo.training.world_model import train_waymo_direct_action_flow as twm
    from waymo.training.world_model.direct_action_flow import agents_to_bntf, gather_agent_window, rollout_receding_horizon
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    from waymo.lane_connectivity.dataset import ConnectivityDataset
    from bias_steer import BiasSteer

    out = Path(a.out_dir)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model, normalizer, train_args, step = ev.load_model_and_normalizer(Path(a.checkpoint), device, "ema")
    steer = BiasSteer(model)
    order = np.concatenate([np.load(f)["dataset_index"] for f in sorted(glob.glob(a.scene_source + "/batch_*.npz"))])
    if a.limit:
        order = order[: a.limit]
    todo = [int(i) for i in order if not (out / "scenes" / f"{int(i)}.npz").is_file()]
    print(f"checkpoint step {step}; scenes {len(order)}, to do {len(todo)}, K={a.K}", flush=True)
    ds = ConnectivityDataset(str(compat.VAL_DIR), str(compat.MAP_CACHE), str(compat.LANE_TOPOLOGY))
    loader = twm.make_loader(Subset(ds, todo), batch_size=1, shuffle=False, num_workers=2, device=device)
    K = a.K

    @torch.inference_mode()
    def rollout(rep, history, seed, cfg):
        steer.cfg = cfg
        gen = torch.Generator(device=device).manual_seed(seed)
        try:
            return rollout_receding_horizon(
                model, normalizer, initial_history=history, agent_lengths=rep.get("agent_lengths"),
                agent_mask=rep["agent_mask"], map_polylines=rep["map_polylines"], map_mask=rep["map_mask"],
                current_light_sequence=rep["lights"][:, ANCHOR:ANCHOR + STEPS],
                current_light_mask_sequence=rep["light_mask"][:, ANCHOR:ANCHOR + STEPS],
                map_ids=rep["map_ids"], map_is_lane=rep["map_is_lane"], map_stop_sign=rep["map_stop_sign"],
                map_stop_point=rep["map_stop_point"], light_id_sequence=rep["light_ids"][:, ANCHOR:ANCHOR + STEPS],
                lane_edges=rep["lane_edges"], lane_edge_types=rep["lane_edge_types"],
                focus_action_sequence=None, focus_action_valid=None, rollout_steps=STEPS,
                commitment=5, solver_steps=8, generator=gen).float()
        finally:
            steer.cfg = None

    started = time.time()
    for n, (ds_idx, raw) in enumerate(zip(todo, loader)):
        t0 = time.time()
        batch = twm.move_batch(raw, device)
        rep = {kk: (v.repeat(K, *[1] * (v.dim() - 1)) if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1 else v)
               for kk, v in batch.items()}
        agents = agents_to_bntf(rep["agents"], rep["agent_mask"])
        anchors = torch.full((K,), ANCHOR, dtype=torch.long, device=device)
        history, future = gather_agent_window(agents, anchors, history_length=int(train_args.history_length), horizon=STEPS)
        valid = (future[0, ..., 5] > 0.5) & batch["agent_mask"][0].bool()[:, None]          # (N,T)
        types = agents[0, :, ANCHOR, 7].round().long()
        init = history[0, :, -1, [0, 1, 6]].float()
        fut = torch.where(valid[..., None], future[0, ..., [0, 1, 6]].float(), torch.zeros_like(future[0, ..., [0, 1, 6]].float()))
        v0 = torch.linalg.vector_norm(future[0, :, 0, 0:2].float() - init[:, :2], dim=-1) / 0.1
        N = types.shape[0]
        gt_gap = collision_gap(fut[None], types[None])[0]
        gt_both = (valid[:, None, :] & valid[None, :, :]).permute(2, 0, 1)
        log_overlap = ((gt_gap < 0) & gt_both & ~torch.eye(N, dtype=torch.bool, device=device)).any(0)
        seed = a.seed + ds_idx
        base = rollout(rep, history, seed, None)
        m0, real_pair = scene_metrics(base, init, fut, valid, types, log_overlap, None, None, v0)
        counts = real_pair.sum(0).cpu().numpy()
        counts = np.triu(counts, 1)                     # real_pair is symmetric: count each pair once
        tn = types.cpu().numpy()
        for i, j in zip(*np.nonzero(counts)):
            if tn[i] == 2 and tn[j] == 2:
                counts[i, j] = 0
        record = {"dataset_index": ds_idx, "seed": seed, "K": K, "types": tn}
        if counts.max() == 0:
            rows = {m: np.stack([m0[m]] + [m0[m]] * (len(COMBOS) - 1)) for m in METRICS}
            record.update(target=np.array([-1, -1]), kind="none", base_hits=0)
            np.savez_compressed(out / "scenes" / f"{ds_idx}.npz", **record, **rows, combos=np.array(COMBOS))
            print(f"[{n + 1}/{len(todo)}] scene {ds_idx}: no real colliding pair ({time.time() - t0:.0f}s)", flush=True)
            continue
        q, k = (int(x) for x in np.unravel_index(np.argmax(counts), counts.shape))
        first = int(torch.nonzero((torch.as_tensor(real_pair[:, q, k] | real_pair[:, k, q]))).flatten()[0])
        res = []
        poses_all = []
        for c, (dq, dk) in enumerate(COMBOS):
            if c == 0:
                P = base
            else:
                pairs = [p for p in ((q, k, float(dq)), (k, q, float(dk))) if p[2] != 0]
                P = rollout(rep, history, seed, dict(pairs=pairs, heads=HEADS, delta=None, where=("enc", "agent", "cross")))
            m, _ = scene_metrics(P, init, fut, valid, types, log_overlap, q, k, v0)
            res.append(m)
            poses_all.append(P[:, [q, k]].half().cpu().numpy())
        # heading difference of the target pair at its first base contact
        g = collision_gap(base, types[None].expand(K, -1))[first, :, q, k]
        t_hit = int(torch.nonzero(g < 0).flatten()[0]) if (g < 0).any() else 0
        dyaw = float(torch.rad2deg(torch.atan2(torch.sin(base[first, q, t_hit, 2] - base[first, k, t_hit, 2]),
                                               torch.cos(base[first, q, t_hit, 2] - base[first, k, t_hit, 2]))).abs())
        kind = "vru" if (tn[q] != 1 or tn[k] != 1) else ("crossing" if dyaw >= 30 else "same_dir")
        rows = {m: np.stack([r[m] for r in res]) for m in METRICS}                           # (C, K)
        record.update(target=np.array([q, k]), kind=kind, dyaw=dyaw, base_hits=int(counts.max()),
                      pair_poses=np.stack(poses_all))                                        # (C,K,2,T,3) fp16
        np.savez_compressed(out / "scenes" / f"{ds_idx}.npz", **record, **rows, combos=np.array(COMBOS))
        best = rows["pair_hit"].mean(1)
        print(f"[{n + 1}/{len(todo)}] scene {ds_idx}: pair ({q},{k}) {kind} hits {counts.max()}/{K}; pair-hit by combo "
              + " ".join(f"{x:.2f}" for x in best) + f" ({time.time() - t0:.0f}s, elapsed {(time.time() - started) / 60:.0f} min)",
              flush=True)


def summarize(a):
    out = Path(a.out_dir)
    files = sorted((out / "scenes").glob("*.npz"))
    lam = a.lambda_stall
    agg = {}
    def add(name, rows_k):  # rows_k: dict metric -> (K,) array for one scene
        acc = agg.setdefault(name, {m: [] for m in METRICS})
        for m in METRICS:
            acc[m].append(rows_k[m])
    target_scenes, kinds, choices = 0, {}, []
    for f in files:
        d = np.load(f)
        rows = {m: d[m] for m in METRICS}                                                   # (C,K)
        has_target = int(d["target"][0]) >= 0
        dade = np.maximum(0.0, rows["ade_q"] - rows["ade_q"][0]) + np.maximum(0.0, rows["ade_k"] - rows["ade_k"][0])
        cost = rows["pair_hit"] + rows["other_hit"] + lam * np.maximum(rows["stall_q"], rows["stall_k"]) + a.mu_ade * dade
        C, K = cost.shape
        cost_tb = cost + 1e-6 * np.array([dq + dk for dq, dk in COMBOS])[:, None]            # ties -> smaller delta
        per_scene = int(np.argmin(cost_tb.mean(1)))
        per_roll = np.argmin(cost_tb, axis=0)                                               # (K,)
        groups = ["all"] + ([f"target:{str(d['kind'])}", "target:any"] if has_target else [])
        for gname in groups:
            add(f"{gname}|base", {m: rows[m][0] for m in METRICS})
            add(f"{gname}|ceiling per scene", {m: rows[m][per_scene] for m in METRICS})
            add(f"{gname}|ceiling per rollout", {m: rows[m][per_roll, np.arange(K)] for m in METRICS})
            for c, (dq, dk) in enumerate(COMBOS[1:], 1):
                add(f"{gname}|fixed ({dq},{dk})", {m: rows[m][c] for m in METRICS})
        if has_target:
            target_scenes += 1
            kinds[str(d["kind"])] = kinds.get(str(d["kind"]), 0) + 1
            choices.append(COMBOS[per_scene])
    table = {name: {m: float(np.mean(np.concatenate(v))) for m, v in acc.items()} for name, acc in agg.items()}
    from collections import Counter
    summary = {"scenes": len(files), "target_scenes": target_scenes, "target_kinds": kinds, "lambda_stall": lam, "mu_ade": a.mu_ade,
               "per_scene_choice_counts": {str(k): v for k, v in Counter(choices).most_common()}, "table": table}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    L = ["# bias 旋钮上限测试（85k stage-1，冻结）", "",
         f"场景 {len(files)} 个，其中有真实碰撞车对（不含日志里本来就重叠的、开始就重叠的、行人-行人）的 {target_scenes} 个："
         + json.dumps(kinds, ensure_ascii=False) + f"。每场景 K 个配对 rollout，代价 = 目标车对碰撞 + 其他真实碰撞 + {lam}×(任一车卡住) + {a.mu_ade}×(两车 ADE 比 base 增加的米数)。", "",
         "δ 组合 (q→k, k→q) ∈ {0,2,4}²，8 个 head 全加，编码器 / agent_attn / scene_cross 三处都加，每次重规划都加。", ""]
    cols = ("real_hit", "proxy_hit", "pair_hit", "other_hit", "stall_q", "stall_k", "never_q", "never_k", "ade_q", "ade_k", "ade_scene")
    for g in ["all", "target:any"] + [f"target:{k}" for k in sorted(kinds)]:
        L += [f"## {g}", "", "| 条件 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        names = [n for n in table if n.startswith(g + "|")]
        for n in names:
            if g == "all" and n.split("|")[1].startswith("fixed"):
                continue
            L.append(f"| {n.split('|')[1]} | " + " | ".join(f"{table[n][c]:.3f}" for c in cols) + " |")
        L.append("")
    L += ["每个场景选中的 δ 组合（per-scene ceiling）：" + json.dumps(summary["per_scene_choice_counts"]), ""]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    a = parse_args()
    if not a.summarize_only:
        run_scenes(a)
    summarize(a)


if __name__ == "__main__":
    main()
