#!/usr/bin/env python3
"""Exp 1a on one candidate-bank view: 85k stage-1 WM, z = unknown / oracle / reversed with the trained g.

Same protocol as wm_interaction_probe/bias_steer.py (scene cache, ego-centered on the view's ego, every
current-valid agent simulated jointly, EMA weights, bf16, generate-40 / execute-5, 8 solver steps,
K paired-seed 8 s rollouts with one generator seeded 12346, so rollout k gets the same noise in every mode).
Oracle z follows eval_zbias.py: pair records from this scene's logged tracks (make_z_labels rule), state from
the rollout at every replan (projection onto the logged path, > 2 m off-path -> unknown).

Writes <out>/{unknown,oracle,reversed}.npz, <out>/summary.json and <out>/view_render.npz (map + history for
render_scene_test.py, which runs in ipenv because the dreamer4 env has no matplotlib).
  cd dreamer4 && ~/.conda/envs/dreamer4/bin/python waymo/pair_order_z/scene_test_1a.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from waymo.pair_order_z import compat  # noqa: E402
from waymo.pair_order_z import z_bias as zb  # noqa: E402
from waymo.pair_order_z import z_bias_v2 as zb2  # noqa: E402
from waymo.pair_order_z.make_z_labels import pairs_for_sample  # noqa: E402

TRI30 = compat.TRI30
sys.path.insert(0, str(TRI30 / "wm_interaction_probe"))
sys.path.insert(0, str(TRI30 / "waymo_interaction_screening/code"))
import wm_common as ip  # noqa: E402
from bias_steer import rollout_metrics  # noqa: E402
from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap, footprint  # noqa: E402

RUN = Path(__file__).resolve().parent / "runs/exp1_oracle_z_frozen_20261110"


def load_batch(scenario, ego, device):
    files = ip.find_files(compat.map_path(ip.MANIFEST), {scenario})
    ds = ip.EgoCenteredScenes([{"file": compat.map_path(files[scenario])}], [ego])
    return ip.base.move_batch(ip.collate_scenes([ds[0]]), device)


def scene_pairs(batch):
    """make_z_labels rule on this view's logged tracks -> (P, 7) slot records."""
    agents = batch["agents"][0].float().cpu().numpy()
    mask = batch["agent_mask"][0].bool().cpu().numpy()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "view.npz"
        np.savez(path, agents=agents, agent_mask=mask)
        return pairs_for_sample(str(path))


class ZHook:
    """Wraps model.encode_scene for one rollout call (same as eval_zbias(_v2).install_z_hooks)."""

    def __init__(self, model, g, batch, z_pairs, reversed_order, v2=False):
        self.model, self.g, self.codes, self.first_bias, self.v2 = model, g, [], None, v2
        cls = zb2.OracleZV2 if v2 else zb.OracleZ
        self.oracle = cls(batch["agents"], batch["agent_mask"], z_pairs, reversed_order=reversed_order)
        self.encode = model.encode_scene

    def __enter__(self):
        def encode_scene(**kw):
            scene = self.encode(**kw)
            speed = kw["history"][:, :, -1, 2]
            if self.v2:
                z, codes = self.oracle.step(scene.agent_pose, scene.agent_mask, speed)
                bias = self.g(z, scene.agent_pose, speed, scene.agent_mask, self.oracle.kind)
            else:
                z, codes = self.oracle.step(scene.agent_pose, scene.agent_mask)
                bias = self.g(z, scene.agent_pose, speed, scene.agent_mask)
            self.codes.append(codes.cpu().numpy())
            if self.first_bias is None and bias is not None:
                self.first_bias = (bias[:, 0].float().cpu().numpy(), z.cpu().numpy())
            return scene if bias is None else replace(scene, chunk_bias=bias)
        self.model.encode_scene = encode_scene
        return self

    def __exit__(self, *exc):
        del self.model.encode_scene


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="95720e9471028705")
    p.add_argument("--ego", type=int, default=6062)
    p.add_argument("--other", type=int, default=2310)
    p.add_argument("--zbias_ckpt", default=str(RUN / "train/zbias_best.pt"))
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=12346)
    p.add_argument("--out", default=None)
    p.add_argument("--checkpoint", default=str(compat.STAGE1_CKPT), help="WM checkpoint (default: the frozen 85k of exp 1)")
    p.add_argument("--modes", default="unknown,oracle,reversed", help="oracle/reversed only make sense for the 85k + g")
    p.add_argument("--v2", action="store_true", help="labels v2 rule (same-direction pairs), g v2, OracleZV2")
    a = p.parse_args()
    out = Path(a.out or RUN / f"scene_{a.scenario}_{a.ego}")
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    model, normalizer, ta, _, meta = ip.load_model(a.checkpoint, device, "dense")
    g, ginfo = (zb2.load_zbias_v2 if a.v2 else zb.load_zbias)(a.zbias_ckpt, device)
    batch = load_batch(a.scenario, a.ego, device)
    ids = batch["agent_ids"][0].cpu().numpy()
    slot = {int(t): i for i, t in enumerate(ids)}
    ego, other = 0, slot[a.other]
    assert int(ids[0]) == a.ego
    types = batch["agents"][0, :, ta.history_length - 1, 7].round().long()
    alive = batch["agent_mask"][0].bool().cpu().numpy()
    print(f"checkpoint step {meta['step']}; view {a.scenario}:{a.ego}, N={len(ids)} (alive {alive.sum()}), "
          f"other {a.other} -> slot {other}", flush=True)

    if a.v2:
        from waymo.pair_order_z.make_z_labels_v2 import pairs_from_agents
        pairs = pairs_from_agents(batch["agents"][0].float().cpu().numpy(), alive)[0]
        stats = {"v2": True, "pairs": int(len(pairs)), "same_dir": int((pairs[:, 7] == 1).sum())}
    else:
        pairs, stats = scene_pairs(batch)
    pair_rows = [dict(i=int(r[0]), j=int(r[1]), id_i=int(ids[r[0]]), id_j=int(ids[r[1]]), enter_i=int(r[2]),
                      exit_i=int(r[3]), enter_j=int(r[4]), exit_j=int(r[5]), angle=int(r[6]),
                      kind=int(r[7]) if a.v2 else 0,
                      log_first=int(ids[r[0]] if r[2] < r[4] else ids[r[1]]) if r[2] != r[4] else None)
                 for r in pairs]
    print("labelled pairs:", json.dumps(pair_rows), "stats", stats, flush=True)
    width = max(8, len(pairs)) if a.v2 else zb.P_MAX
    if len(pairs) > width:
        raise ValueError(f"{len(pairs)} pairs > P_MAX")
    zp = np.full((width, pairs.shape[1] if len(pairs) else (8 if a.v2 else 7)), -1, np.int64)
    zp[:len(pairs)] = pairs
    z_pairs = torch.from_numpy(zp).to(device)[None].expand(a.K, -1, -1)
    rep = ip.rep(batch, a.K)
    try:
        labels_train = (zb2.PairLabelsV2 if a.v2 else zb.PairLabels)(
            str(compat.DATA / ("pair_order_z_labels_v2" if a.v2 else "pair_order_z_labels_v1") / "train.npz"))
        in_g_train = any(n.startswith(a.scenario) for n in labels_train.index)
    except FileNotFoundError:
        in_g_train = None

    # Identity check: zero-output g reproduces the unhooked rollout bit for bit.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p0 = ip.rollout(model, normalizer, ta, ip.rep(batch, 4), steps=10, seed=1)
        with ZHook(model, (zb2.ZBiasV2() if a.v2 else zb.ZBias()).to(device).eval(), ip.rep(batch, 4), z_pairs[:4], False, a.v2):
            p1 = ip.rollout(model, normalizer, ta, ip.rep(batch, 4), steps=10, seed=1)
    ident = float((p0 - p1).abs().max())
    print("identity check (zero g vs unhooked) max |dpose| =", ident, flush=True)
    assert ident == 0.0

    results = {}
    for mode in a.modes.split(","):
        hook = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "unknown":
                P = ip.rollout(model, normalizer, ta, rep, steps=a.steps, seed=a.seed)
            else:
                with ZHook(model, g, rep, z_pairs, mode == "reversed", a.v2) as hook:
                    P = ip.rollout(model, normalizer, ta, rep, steps=a.steps, seed=a.seed)
        m, per = rollout_metrics(P, types.to(P.device), ego, other)
        gap = collision_gap(P, types.to(P.device)[None].expand(a.K, -1)).cpu().numpy()      # (K,T,N,N)
        live = alive[:, None] & alive[None, :] & ~np.eye(len(ids), dtype=bool)
        new = (gap < 0) & live & (gap[:, :1] >= 0)                                           # ignore step-1 overlaps
        scene_hit = new.any((1, 2, 3))
        m["scene_any_collision"] = float(scene_hit.mean())
        extra = {}
        if hook is not None:
            codes = np.stack(hook.codes)                                                     # (replans, K, P)
            k = next(n for n, r in enumerate(pair_rows) if {r["id_i"], r["id_j"]} == {a.ego, a.other})
            names = dict(zb2.CODE_NAMES if a.v2 else zb.CODE_NAMES)
            m["ego_other_pair_code_per_replan_k0"] = [names[int(c)] for c in codes[:, 0, k]]
            m["ego_other_pair_code_fraction"] = {names[c]: float((codes[:, :, k] == c).mean())
                                                 for c in names if c > 0 and (codes[:, :, k] == c).any()}
            b0, z0 = hook.first_bias
            m["bias_t0_ego_to_other"] = [round(float(v), 3) for v in b0[0, :, ego, other]]
            m["bias_t0_other_to_ego"] = [round(float(v), 3) for v in b0[0, :, other, ego]]
            m["z_t0_ego_to_other"] = int(z0[0, ego, other])
            extra["codes"] = codes.astype(np.int8)
        results[mode] = m
        np.savez_compressed(out / f"{mode}.npz", poses=P.cpu().numpy().astype(np.float32), agent_ids=ids,
                            scene_hit=scene_hit, agent_hit=new.any(3), **per, **extra)
        print(f"{mode:8s} collision ego-{a.other} {m['collision_2310']:.3f}  ego any {m['ego_any_collision']:.3f}  "
              f"scene any {m['scene_any_collision']:.3f}  order {m['order']}  ego start median {m['ego_start_median']:.2f}s "
              f"never {m['ego_never_start']:.2f}", flush=True)

    bank = ip.CandidateBank if hasattr(ip, "CandidateBank") else None
    from candidate_bank import CandidateBank
    bank = CandidateBank(compat.map_path("/p/yufeng/tri30/waymo_interaction_screening/candidate_bank_v1_20261022"))
    view = bank.get_view(f"{a.scenario}:{a.ego}", future_steps=80, max_other_agents=None)
    bank.close()
    hist = batch["agents"][0, :, :11].float().cpu().numpy()
    bank_ids = [int(i) for i in view["inputs"]["agent_ids"]]
    for n, tid in enumerate(bank_ids):          # frame check: bank ego frame == scene-cache ego frame
        if tid in slot:
            j = slot[tid]
            mm = view["inputs"]["valid"][n, :11] & (hist[j, :, 5] > .5)
            assert np.abs(hist[j, mm, :2] - view["inputs"]["states"][n, mm, :2]).max(initial=0) < 0.05, tid
    L, W = (x.numpy() for x in footprint(types.cpu()))
    mp = view["inputs"]["map"]
    np.savez_compressed(out / "view_render.npz", hist=hist, ids=ids, types=types.cpu().numpy(), L=L, W=W,
                        alive=alive, ego=ego, other=other, map_points=mp["points"], map_offsets=mp["offsets"],
                        map_kind=mp["feature_kind"])
    summary = dict(view=f"{a.scenario}:{a.ego}", other=a.other, K=a.K, steps=a.steps, seed=a.seed,
                   checkpoint=a.checkpoint, checkpoint_step=meta["step"], g=ginfo["path"], g_step=ginfo["step"],
                   scenario_in_g_train_labels=in_g_train, labelled_pairs=pair_rows, label_stats=stats,
                   identity_check=ident, results=results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
