#!/usr/bin/env python3
"""Closed-loop eval of the learned bias controller (plan BIAS_CONTROLLER_PLAN_2026_11_12 step 4).

The 512 eval scenes of the ceiling test, frozen 85k WM, eval protocol (EMA, generate-all, 80 steps,
commitment 5, 8 solver steps, fp32). Per scene K rollouts; rollout k has its own noise (row seed =
seed + 16 * dataset_index + k), shared by all conditions, and every condition runs with the same batch
composition, so conditions are paired. Conditions: base (no delta) and ctrl_tau<t>: at every replan the
draft (no delta) gives the conflict features of gated vehicle pairs; the head predicts delta in {0,2,4}
per directed pair (non-zero when P(delta != 0) > tau); the final plan reuses the draft's noise.
Metrics per rollout: real collisions (any, vehicle-vehicle, involving a VRU), proxy collisions, stalls
of vehicles that move > 5 m in the log, never-start, scene ADE, and the ceiling-test target pair's
collision (target scenes). Summary with scene-paired bootstrap CIs of ctrl - base.
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
from waymo.pair_order_z import bias_head_lib as L  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS = ("real_hit", "vv_hit", "vru_hit", "proxy_hit", "pair_hit", "stall_frac", "stall_any", "never_any", "ade_scene")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--head", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoint", default=str(compat.STAGE1_CKPT))
    p.add_argument("--scene_source", default=str(HERE / "runs/exp1_oracle_z_frozen_20261110/eval/unknown/rollout_batches"))
    p.add_argument("--ceiling_dir", default=str(HERE / "runs/bias_ceiling_85k_20261112"))
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--scenes_per_batch", type=int, default=2)
    p.add_argument("--taus", default="0.5")
    p.add_argument("--seed", type=int, default=12346)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--summarize_only", action="store_true")
    return p.parse_args()


def metrics(P, g, target):
    """P (K,N,T,3) GPU, g = L.gt_prep(...). Returns dict of (K,) numpy arrays."""
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    K, N, T, _ = P.shape
    types, valid = g["types"], g["valid"]
    real = L.real_collisions(P, g).any(1)                                       # (K,N,N)
    gap = collision_gap(P, types[None].expand(K, -1))
    both = (valid[:, None, :] & valid[None, :, :]).permute(2, 0, 1)
    raw = ((gap < 0) & both[None] & ~torch.eye(N, dtype=torch.bool, device=P.device)).any(1)
    veh = types == 1
    vru = (types == 2) | (types == 3)
    ped = types == 2
    out = dict(real_hit=real.flatten(1).any(1), vv_hit=(real & veh[:, None] & veh[None, :]).flatten(1).any(1),
               vru_hit=(real & (vru[:, None] | vru[None, :]) & ~(ped[:, None] & ped[None, :])).flatten(1).any(1),
               proxy_hit=raw.flatten(1).any(1))
    q, k = target
    out["pair_hit"] = (real[:, q, k] | real[:, k, q]) if q >= 0 else torch.zeros(K, dtype=torch.bool, device=P.device)
    xy, gt = P[..., :2], g["fut"][..., :2]
    vf = valid.float()
    seg_log = torch.linalg.vector_norm(torch.diff(torch.cat((g["init"][:, None, :2], gt), 1), dim=1), dim=-1) * vf
    log_len = seg_log.sum(-1)
    roll = torch.cat((g["init"][None, :, None, :2].expand(K, -1, -1, -1), xy), 2)
    roll_len = torch.linalg.vector_norm(torch.diff(roll, dim=2), dim=-1).sum(-1)    # (K,N)
    moved = veh & (log_len > 5.0)
    stall = moved[None] & (roll_len < 0.3 * log_len[None])
    never = moved[None] & (g["v0"] < 0.5)[None] & (roll_len < 1.0)
    out["stall_frac"] = stall.float().sum(1) / moved.sum().clamp_min(1)
    out["stall_any"] = stall.any(1)
    out["never_any"] = never.any(1)
    err = torch.linalg.vector_norm(xy - gt[None], dim=-1)
    out["ade_scene"] = (err * vf).sum((1, 2)) / vf.sum().clamp_min(1)
    return {m: v.float().cpu().numpy() for m, v in out.items()}


def run(a, taus):
    compat.patch_map_cache_identity()
    from waymo.evaluation import eval_waymo_direct_action_flow_multisample as ev
    from waymo.training.world_model import train_waymo_direct_action_flow as twm
    from waymo.training.world_model.direct_action_flow import agents_to_bntf, gather_agent_window
    from waymo.lane_connectivity.dataset import ConnectivityDataset

    out = Path(a.out_dir)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model, normalizer, train_args, step = ev.load_model_and_normalizer(Path(a.checkpoint), device, "ema")
    steer = L.PairDeltaSteer(model)
    ck = torch.load(a.head, map_location=device, weights_only=False)
    head = L.BiasHead(hidden=ck["hidden"]).to(device)
    head.load_state_dict(ck["state_dict"]); head.eval()
    order = np.concatenate([np.load(f)["dataset_index"] for f in sorted(glob.glob(a.scene_source + "/batch_*.npz"))])
    if a.limit:
        order = order[: a.limit]
    order = [int(i) for i in order]
    S, K = a.scenes_per_batch, a.K
    groups = [order[i:i + S] for i in range(0, len(order), S)]
    todo = [g for g in groups if not all((out / "scenes" / f"{i}.npz").is_file() for i in g)]
    flat = [i for g in todo for i in g]
    print(f"checkpoint step {step}; head epoch {ck['epoch']} ({ck['n_files']} label files); scenes {len(order)}, "
          f"to do {len(flat)}; K={K}; taus {taus}", flush=True)
    ds = ConnectivityDataset(str(compat.VAL_DIR), str(compat.MAP_CACHE), str(compat.LANE_TOPOLOGY))
    loader = twm.make_loader(Subset(ds, flat), batch_size=S, shuffle=False, num_workers=2, device=device)
    started = time.time()
    for n, (ids, raw) in enumerate(zip(todo, loader)):
        t0 = time.time()
        batch = twm.move_batch(raw, device)
        Sb = len(ids)
        rep = L.take_rows(batch, torch.arange(Sb, device=device).repeat_interleave(K))
        B = Sb * K
        agents = agents_to_bntf(rep["agents"], rep["agent_mask"])
        history, future = gather_agent_window(agents, torch.full((B,), L.ANCHOR, dtype=torch.long, device=device),
                                              history_length=int(train_args.history_length), horizon=L.STEPS)
        N = history.shape[1]
        noise = L.make_noise([a.seed + 16 * ids[s] + k for s in range(Sb) for k in range(K)], N, device)
        conds = {"base": L.rollout(model, normalizer, rep, history, noise, steer)[0]}
        trig = {}
        for t in taus:
            P, recs = L.rollout(model, normalizer, rep, history, noise, steer, delta_fn=L.controller_fn(head, t), record=True)
            conds[f"ctrl_tau{t}"] = P
            trig[t] = recs
        for s in range(Sb):
            r0 = s * K
            g = L.gt_prep(agents[r0], rep["agent_mask"][r0], history[r0], future[r0])
            cf = Path(a.ceiling_dir) / "scenes" / f"{ids[s]}.npz"
            target, kind = (-1, -1), "none"
            if cf.is_file():
                c = np.load(cf, allow_pickle=True)
                if int(c["target"][0]) >= 0:
                    target, kind = (int(c["target"][0]), int(c["target"][1])), str(c["kind"])
            rec = dict(dataset_index=ids[s], target=np.array(target), kind=kind, K=K)
            for name, P in conds.items():
                for m, v in metrics(P[r0:r0 + K], g, target).items():
                    rec[f"{name}|{m}"] = v
            for t, recs in trig.items():
                gated = sum(int(((x["idx"][:, 0] >= r0) & (x["idx"][:, 0] < r0 + K)).sum()) for x in recs)
                on = sum(int((((x["idx"][:, 0] >= r0) & (x["idx"][:, 0] < r0 + K)) & (x["delta"] != 0)).sum()) for x in recs)
                rep_on = np.zeros(K)
                for x in recs:
                    for kk in range(K):
                        m_ = (x["idx"][:, 0] == r0 + kk) & (x["delta"] != 0)
                        rep_on[kk] += bool(m_.any())
                rec[f"ctrl_tau{t}|gated"] = gated
                rec[f"ctrl_tau{t}|on"] = on
                rec[f"ctrl_tau{t}|replans_on"] = rep_on
            np.savez_compressed(out / "scenes" / f"{ids[s]}.npz", **rec)
        msg = " ".join(f"{name} coll {float(np.mean([np.load(out / 'scenes' / f'{i}.npz')[f'{name}|real_hit'].mean() for i in ids])):.2f}"
                       for name in conds)
        print(f"[{n + 1}/{len(todo)}] scenes {ids}: {msg} ({time.time() - t0:.0f}s, elapsed {(time.time() - started) / 60:.0f} min)", flush=True)


def boot(diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(a, taus):
    out = Path(a.out_dir)
    files = sorted((out / "scenes").glob("*.npz"))
    recs = [dict(np.load(f, allow_pickle=True)) for f in files]
    conds = ["base"] + [f"ctrl_tau{t}" for t in taus]
    groups = {"all": recs, "target:any": [r for r in recs if str(r["kind"]) != "none"]}
    for kd in ("same_dir", "crossing", "vru"):
        groups[f"target:{kd}"] = [r for r in recs if str(r["kind"]) == kd]
    L_ = ["# bias head 闭环评估（v1）", "",
          f"512 个评估场景中已完成 {len(recs)} 个，每个场景 K 个 rollout；各条件噪声相同、batch 构成相同（配对）。"
          "真实碰撞的口径同上限测试。pair_hit 是上限测试的目标车对。stall 只统计日志里走了 5 m 以上的车。", ""]
    summary = {}
    for gname, rs in groups.items():
        if not rs:
            continue
        L_ += [f"## {gname}（{len(rs)} 个场景）", "", "| 条件 | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for c in conds:
            vals = {m: float(np.mean(np.concatenate([r[f"{c}|{m}"] for r in rs]))) for m in METRICS}
            summary[f"{gname}|{c}"] = vals
            L_.append(f"| {c} | " + " | ".join(f"{vals[m]:.3f}" for m in METRICS) + " |")
        L_.append("")
        for c in conds[1:]:
            parts = []
            for m in ("real_hit", "vv_hit", "pair_hit", "stall_frac", "never_any", "ade_scene"):
                d = np.array([r[f"{c}|{m}"].mean() - r[f"base|{m}"].mean() for r in rs])
                mu, lo, hi = boot(d)
                summary[f"{gname}|{c}|diff|{m}"] = (mu, lo, hi)
                parts.append(f"{m} {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            L_.append(f"- {c} − base（场景配对 bootstrap 95% CI）：" + "；".join(parts))
        L_.append("")
    for t in taus:
        g = sum(int(r[f"ctrl_tau{t}|gated"]) for r in recs)
        o = sum(int(r[f"ctrl_tau{t}|on"]) for r in recs)
        ro = np.concatenate([r[f"ctrl_tau{t}|replans_on"] for r in recs])
        L_.append(f"- 触发统计 τ={t}：门控内的有向车对-轮次 {g} 个，其中 δ≠0 的 {o} 个（{o / max(1, g):.1%}）；"
                  f"每条 rollout 有 δ≠0 的重规划轮数平均 {ro.mean():.2f}（16 轮中），至少一轮触发的 rollout 占 {(ro > 0).mean():.1%}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "summary.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def main():
    a = parse_args()
    taus = [float(t) for t in a.taus.split(",")]
    if not a.summarize_only:
        run(a, taus)
    summarize(a, taus)


if __name__ == "__main__":
    main()
