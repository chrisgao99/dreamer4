#!/usr/bin/env python3
"""Training labels for the learned bias controller (plan BIAS_CONTROLLER_PLAN_2026_11_12 §3.1-§3.2).

Training scenes only (one view per scenario, random order, fixed seed); the 512 eval scenes are from val.
Frozen 85k stage-1 WM, eval protocol (EMA, generate-all, 80 steps, commitment 5, 8 solver steps, fp32).
Vehicle-vehicle pairs only (ceiling-test decision: same-direction and crossing pairs work, VRU does not).

Per scene, K rollouts, each with its own noise sequence (row seed = seed + 16 * dataset_index + k):
  1. base rollout. At every replan the draft (= the plan itself, since delta = 0) gives the conflict
     features of every gated directed vehicle pair (bias_head_lib.pair_features).
  2. real collisions (proxy-box overlap, not overlapping in the log, not overlapping at step 1). Target
     = the vehicle-vehicle pair with the earliest real contact that the gate saw before the contact.
  3. search: for the target, every (delta q->k, delta k->q) in {0,2,4}^2, same noise, delta applied only
     on the replans where the draft gates the pair (as the controller would), forked off the base rollout
     at the first gated replan. (0,0) is re-run in the same batch as the other 8: GPU numerics depend on
     the batch shape (a fork with delta = 0 differs from the base by up to ~4 cm), so all 9 are compared
     under identical numerics.
  4. cost = target collision + other real collisions + lambda * (q or k stalls) + mu * (ADE_q + ADE_k
     increase over base); pick the lowest, ties -> smaller delta. Labels: the chosen rollout's gated
     replans; target pair -> chosen (dq, dk), every other pair -> 0; pairs that collide in that rollout
     (other than the target) are dropped. Rollouts without a target: all gated pairs -> 0, colliding
     pairs dropped.
Output: <out>/batches/<b>.npz (resumable), <out>/scenes.json.
"""
from __future__ import annotations

import argparse
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

KIND = {"none": 0, "same_dir": 1, "crossing": 2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoint", default=str(compat.STAGE1_CKPT))
    p.add_argument("--n_scenes", type=int, default=2000)
    p.add_argument("--scenes_per_batch", type=int, default=4)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--lambda_stall", type=float, default=1.0)
    p.add_argument("--mu_ade", type=float, default=0.05)
    p.add_argument("--search_chunk", type=int, default=18, help="rows per search batch (multiple of 9)")
    return p.parse_args()


def pick_scenes(ds, n, seed, out):
    f = out / "scenes.json"
    if f.is_file():
        return json.loads(f.read_text())
    first = {}
    for i, path in enumerate(map(str, ds.paths)):
        first.setdefault(Path(path).name.split("_focus")[0], i)
    idx = np.array(sorted(first.values()))
    np.random.default_rng(seed).shuffle(idx)
    sel = [int(i) for i in idx[:n]]
    f.write_text(json.dumps(sel))
    return sel


def main():
    a = parse_args()
    out = Path(a.out_dir)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    compat.patch_map_cache_identity()
    from waymo.evaluation import eval_waymo_direct_action_flow_multisample as ev
    from waymo.training.world_model import train_waymo_direct_action_flow as twm
    from waymo.training.world_model.direct_action_flow import agents_to_bntf, gather_agent_window
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    from waymo.lane_connectivity.dataset import ConnectivityDataset
    from waymo.pair_order_z.bias_ceiling import scene_metrics

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model, normalizer, train_args, step = ev.load_model_and_normalizer(Path(a.checkpoint), device, "ema")
    steer = L.PairDeltaSteer(model)
    ds = ConnectivityDataset(str(compat.TRAIN_DIR), str(compat.MAP_CACHE), str(compat.LANE_TOPOLOGY))
    sel = pick_scenes(ds, a.n_scenes, a.seed, out)
    S, K = a.scenes_per_batch, a.K
    groups = [sel[i:i + S] for i in range(0, len(sel), S)]
    todo = [b for b in range(len(groups)) if not (out / "batches" / f"{b}.npz").is_file()
            and not (out / "batches" / f"{b}.skipped").is_file()]
    flat = [i for b in todo for i in groups[b]]
    print(f"checkpoint step {step}; scenes {len(sel)} in {len(groups)} batches, to do {len(todo)}; K={K}", flush=True)
    loader = twm.make_loader(Subset(ds, flat), batch_size=S, shuffle=False, num_workers=3, device=device)
    lam, mu = a.lambda_stall, a.mu_ade
    started = time.time()
    tot = dict(rollouts=0, targets=0, nonzero=0)

    for n, (b, raw) in enumerate(zip(todo, loader)):
        t0 = time.time()
        ids = groups[b]
        assert len(ids) == int(raw["agent_mask"].shape[0])
        batch = twm.move_batch(raw, device)
        Sb = len(ids)
        rows = torch.arange(Sb, device=device).repeat_interleave(K)
        rep = L.take_rows(batch, rows)
        B = Sb * K
        agents = agents_to_bntf(rep["agents"], rep["agent_mask"])
        history, future = gather_agent_window(agents, torch.full((B,), L.ANCHOR, dtype=torch.long, device=device),
                                              history_length=int(train_args.history_length), horizon=L.STEPS)
        N = history.shape[1]
        seeds = [a.seed + 16 * ids[s] + k for s in range(Sb) for k in range(K)]
        noise = L.make_noise(seeds, N, device)
        try:
            base, recs, snaps = L.rollout(model, normalizer, rep, history, noise, steer, record=True, snapshots=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            (out / "batches" / f"{b}.skipped").write_text("OOM in the base rollout\n")
            print(f"[{n + 1}/{len(todo)}] batch {b}: OOM in the base rollout, skipped", flush=True)
            continue
        R = len(recs)
        gated = torch.zeros((B, R, N, N), dtype=torch.bool)
        for rec in recs:
            ii = rec["idx"].long()
            gated[ii[:, 0], rec["r"], ii[:, 1], ii[:, 2]] = True

        # per scene ground truth, base real collisions, targets
        gts, real_base = [], []
        targets = []          # (row, q, k, kind, t_contact, r0)
        for s in range(Sb):
            r0w = s * K
            g = L.gt_prep(agents[r0w], rep["agent_mask"][r0w], history[r0w], future[r0w])
            gts.append(g)
            real = L.real_collisions(base[r0w:r0w + K], g)                      # (K,T,N,N)
            real_base.append(real.any(1).cpu())
            veh = (g["types"] == 1).cpu()
            for k in range(K):
                rk = real[k].cpu()
                hit_pairs = torch.nonzero(rk.any(0) & veh[:, None] & veh[None, :])
                best = None
                for q, kk in hit_pairs.tolist():
                    if q >= kk:
                        continue
                    t_c = int(torch.nonzero(rk[:, q, kk] | rk[:, kk, q])[0])
                    r_gate = [r for r in range(R) if gated[r0w + k, r, q, kk] and r * L.COMMIT <= t_c]
                    if r_gate and (best is None or t_c < best[3]):
                        best = (q, kk, t_c, r_gate[0])
                if best is None:
                    continue
                q, kk, t_c, rg = best
                dyaw = float(torch.rad2deg(L.wrap(base[r0w + k, q, t_c, 2] - base[r0w + k, kk, t_c, 2])).abs())
                targets.append((r0w + k, q, kk, "crossing" if dyaw >= 30 else "same_dir", t_c, rg))

        # search, forked at the first gated replan. All 9 combos of a target (including (0,0)) run in the same
        # chunk: GPU numerics depend on the batch shape, so the (0,0) reference must be computed alongside.
        order = sorted(range(len(targets)), key=lambda t: targets[t][5])
        per = max(1, a.search_chunk // len(L.COMBOS))
        search_P, search_recs = {}, {}
        skipped = set()

        def run_search(tlist):
            chunk = [(t, c) for t in tlist for c in range(len(L.COMBOS))]
            rr = torch.tensor([targets[t][0] for t, _ in chunk], device=device)
            r_start = min(targets[t][5] for t, _ in chunk)
            q = torch.tensor([targets[t][1] for t, _ in chunk], device=device)
            kk = torch.tensor([targets[t][2] for t, _ in chunk], device=device)
            dq = torch.tensor([float(L.COMBOS[c][0]) for _, c in chunk], device=device)
            dk = torch.tensor([float(L.COMBOS[c][1]) for _, c in chunk], device=device)
            P, rc = L.rollout(model, normalizer, L.take_rows(rep, rr), snaps[r_start][rr], noise[rr], steer,
                              delta_fn=L.search_fn(q, kk, dq, dk), record=True, r_start=r_start,
                              prefix=base[rr][:, :, :L.COMMIT * r_start])
            for j, job in enumerate(chunk):
                search_P[job] = P[j]
                search_recs[job] = [dict(r=x["r"], idx=x["idx"][x["idx"][:, 0] == j], feat=x["feat"][x["idx"][:, 0] == j],
                                         delta=x["delta"][x["idx"][:, 0] == j]) for x in rc]
            # replans before the fork are the base ones
            for j, job in enumerate(chunk):
                row = targets[job[0]][0]
                pre = []
                for x in recs[:r_start]:
                    m = x["idx"][:, 0] == row
                    pre.append(dict(r=x["r"], idx=x["idx"][m].clone(), feat=x["feat"][m], delta=x["delta"][m]))
                for p_ in pre:
                    p_["idx"][:, 0] = j
                search_recs[job] = pre + search_recs[job]

        # big maps can exhaust memory: retry one target at a time, then skip the target
        for c0 in range(0, len(order), per):
            tl = order[c0:c0 + per]
            try:
                run_search(tl)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                for t in tl:
                    try:
                        run_search([t])
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        skipped.add(t)
                        print(f"  OOM: skipped target {targets[t][:3]} in batch {b}", flush=True)

        # choose combos, build labels
        F_, Y_, M_ = [], [], []
        roll_rows = []
        tgt_of_row = {t[0]: ti for ti, t in enumerate(targets) if ti not in skipped}
        for s in range(Sb):
            g = gts[s]
            for k in range(K):
                row = s * K + k
                info = dict(ds=ids[s], k=k, has_target=0, q=-1, kk=-1, kind=0, t_contact=-1, r0=-1, chosen=0,
                            cost=np.zeros(len(L.COMBOS)), pair_hit=np.zeros(len(L.COMBOS)), other_hit=np.zeros(len(L.COMBOS)),
                            stall=np.zeros(len(L.COMBOS)), dade=np.zeros(len(L.COMBOS)), real_any_base=float(real_base[s][k].any()))
                if row in tgt_of_row:
                    ti = tgt_of_row[row]
                    _, q, kk, kind, t_c, rg = targets[ti]
                    P9 = torch.stack([search_P[(ti, c)] for c in range(len(L.COMBOS))])
                    m, real_pair = scene_metrics(P9, g["init"], g["fut"], g["valid"], g["types"], g["log_overlap"], q, kk, g["v0"])
                    dade = np.maximum(0.0, m["ade_q"] - m["ade_q"][0]) + np.maximum(0.0, m["ade_k"] - m["ade_k"][0])
                    st = np.maximum(m["stall_q"], m["stall_k"])
                    cost = m["pair_hit"] + m["other_hit"] + lam * st + mu * dade
                    c_star = int(np.argmin(cost + 1e-6 * np.array([x + y for x, y in L.COMBOS])))
                    info.update(has_target=1, q=q, kk=kk, kind=KIND[kind], t_contact=t_c, r0=rg, chosen=c_star, cost=cost,
                                pair_hit=m["pair_hit"], other_hit=m["other_hit"], stall=st, dade=dade)
                    rp = real_pair[c_star].cpu()
                    rec_list = search_recs[(ti, c_star)]
                    dq_s, dk_s = L.COMBOS[c_star]
                else:
                    q = kk = -1
                    rp = real_base[s][k]
                    rec_list = [dict(r=x["r"], idx=x["idx"][x["idx"][:, 0] == row], feat=x["feat"][x["idx"][:, 0] == row])
                                for x in recs]
                    dq_s = dk_s = 0
                coll = rp | rp.T
                for x in rec_list:
                    if len(x["idx"]) == 0:
                        continue
                    i, j = x["idx"][:, 1].long(), x["idx"][:, 2].long()
                    is_t = ((i == q) & (j == kk)) | ((i == kk) & (j == q))
                    keep = is_t | ~coll[i, j]
                    lab = torch.zeros(len(i), dtype=torch.int8)
                    lab[(i == q) & (j == kk)] = L.DELTAS.index(float(dq_s))
                    lab[(i == kk) & (j == q)] = L.DELTAS.index(float(dk_s))
                    F_.append(x["feat"][keep])
                    Y_.append(lab[keep])
                    M_.append(torch.stack((torch.full_like(i, ids[s]), torch.full_like(i, k), torch.full_like(i, x["r"]),
                                           i, j, is_t.long()), 1)[keep].int())
                roll_rows.append(info)
        feats = torch.cat(F_).numpy() if F_ else np.zeros((0, L.NFEAT), np.float16)
        labels = torch.cat(Y_).numpy() if Y_ else np.zeros(0, np.int8)
        meta = torch.cat(M_).numpy() if M_ else np.zeros((0, 6), np.int32)
        np.savez_compressed(out / "batches" / f"{b}.npz", feats=feats, labels=labels, meta=meta,
                            scenario_ids=np.array([Path(str(ds.paths[i])).name.split("_focus")[0] for i in ids]),
                            dataset_index=np.array(ids),
                            **{f"roll_{key}": np.array([r_[key] for r_ in roll_rows]) for key in roll_rows[0]})
        nz = int((labels > 0).sum())
        tot["rollouts"] += B; tot["targets"] += len(targets); tot["nonzero"] += nz
        ch = [r_["chosen"] for r_ in roll_rows if r_["has_target"]]
        print(f"[{n + 1}/{len(todo)}] batch {b}: {B} rollouts, {len(targets)} targets, chosen {ch}, instances {len(labels)} "
              f"(nonzero {nz}) ({time.time() - t0:.0f}s, elapsed {(time.time() - started) / 60:.0f} min; "
              f"total targets {tot['targets']}/{tot['rollouts']}, nonzero {tot['nonzero']})", flush=True)


if __name__ == "__main__":
    main()
