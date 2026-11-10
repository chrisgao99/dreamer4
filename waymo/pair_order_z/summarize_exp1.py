#!/usr/bin/env python3
"""Experiment 1 summary (plan §8.3): unknown vs oracle vs reversed on the same 512 scenes / seeds.

Reads <run>/eval/<mode>/{result.json, physical_metrics.npz, rollout_batches/, z_eval_info.json}
and <run>/train/metrics.jsonl; writes <run>/summary.json and <run>/summary.md.

Definitions (all on the same K rollouts, no best-of selection):
  scene collision      any proxy-box overlap in the scene-rollout (eval physical metric #2)
  labelled scenes      scenes with >= 1 crossing/opposing vehicle pair in labels v1
  pair collision       the two cars of a labelled pair overlap at any step (both GT-valid)
  pass order           per pair and rollout, path = log frames 0-10 + rollout frames 11-90; a car
                       "enters" at its first frame within 2.5 m of the other car's path; the earlier
                       entry passes first. Compared with the logged order (same definition on the log).
  never_start          vehicle slow at t0 (< 0.5 m/s), log moves > 5 m over 8 s, rollout path < 1 m
  stuck_late           vehicle whose log moves > 5 m in the last 4 s but rollout moves < 0.5 m there
  bootstrap            per-scene mean over rollouts, mode - unknown, 10k scene resamples, 95% CI
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from waymo.pair_order_z import compat  # noqa: E402
from waymo.pair_order_z.z_bias import CODE_NAMES  # noqa: E402

MODES = ("unknown", "oracle", "reversed")
THR = 2.5
ANCHOR = 10


def bootstrap(diff: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    if len(diff) == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(1)
    return {"n": int(len(diff)), "mean": float(diff.mean()), "ci95": [float(np.quantile(means, .025)),
            float(np.quantile(means, .975))], "p_ge_0": float((means >= 0).mean())}


def entry_frames(path_a: np.ndarray, path_b: np.ndarray):
    """First frame of a within THR of b's whole path, and vice versa (None if no overlap)."""
    d = np.linalg.norm(path_a[:, None] - path_b[None], axis=-1)
    close = d < THR
    if not close.any():
        return None, None
    return int(np.flatnonzero(close.any(1))[0]), int(np.flatnonzero(close.any(0))[0])


def load_mode(run: Path, mode: str):
    d = run / "eval" / mode
    if not (d / "result.json").is_file():
        return None
    out = {"result": json.loads((d / "result.json").read_text()), "dir": d}
    pm = np.load(d / "physical_metrics.npz")
    out["gen"], out["dataset_index"] = pm["generated_stats"], pm["dataset_index"]
    names = list(pm["metric_names"])
    out["metric_index"] = {n: k for k, n in enumerate(names)}
    info = d / "z_eval_info.json"
    out["z_info"] = json.loads(info.read_text()) if info.is_file() else {}
    return out


def pair_level(mode_data, scenes, labels, val_paths, collision_gap):
    """Per labelled pair x rollout: collision, rollout order vs log; per-vehicle freeze stats."""
    rec = {"pair_collision": {}, "order": {}, "never_start": {}, "stuck_late": {}}
    batch_dir = mode_data["dir"] / "rollout_batches"
    for f in sorted(batch_dir.glob("batch_*.npz")):
        z = np.load(f)
        poses, valid, types = z["poses"], z["valid"], z["agent_type"]       # (B,R,N,T,3), (B,N,T), (B,N)
        init, gt = z["initial_pose"], z["gt_pose"]
        for b, ds_idx in enumerate(z["dataset_index"]):
            ds_idx = int(ds_idx)
            name = Path(val_paths[ds_idx]).name
            lo, hi = labels["index"][name]
            pairs = labels["pairs"][lo:hi]
            # vehicle freeze metrics (all scenes)
            veh = types[b] == 1
            gt_xy = np.where(valid[b][..., None], gt[b][..., :2], np.nan)
            seg = np.linalg.norm(np.diff(np.concatenate((init[b][:, None, :2], gt_xy), 1), axis=1), axis=-1)
            log_len = np.nansum(seg, 1)
            log_late = np.nansum(seg[:, 40:], 1)
            v0 = seg[:, 0] / 0.1
            roll_xy = np.concatenate((np.repeat(init[b][None, :, None, :2], poses.shape[1], 0),
                                      poses[b][..., :2]), 2)                 # (R,N,T+1,2)
            rseg = np.linalg.norm(np.diff(roll_xy, axis=2), axis=-1)
            r_len, r_late = rseg.sum(2), rseg[:, :, 40:].sum(2)
            alive = np.isfinite(init[b][:, 0]) & valid[b].any(1)
            ns_elig = veh & alive & (np.nan_to_num(v0, nan=0) < 0.5) & (log_len > 5)
            sl_elig = veh & alive & (log_late > 5)
            rec["never_start"][ds_idx] = ((r_len[:, ns_elig] < 1.0).sum(), ns_elig.sum() * poses.shape[1])
            rec["stuck_late"][ds_idx] = ((r_late[:, sl_elig] < 0.5).sum(), sl_elig.sum() * poses.shape[1])
            if len(pairs) == 0:
                continue
            with np.load(Path(val_paths[ds_idx])) as s:
                log = s["agents"]
            gap = []
            for r in range(poses.shape[1]):
                g = collision_gap(torch.from_numpy(poses[b, r][None]).float(),
                                  torch.from_numpy(types[b][None]).long())[0].numpy()   # (T,N,N)
                gap.append(g)
            gap = np.stack(gap)                                                  # (R,T,N,N)
            for p in pairs:
                i, j, ei, xi, ej, xj, _ = (int(v) for v in p)
                both = valid[b, i] & valid[b, j]
                hit = ((gap[:, :, i, j] < 0) & both[None]).any(1)                # (R,)
                key = (ds_idx, i, j)
                rec["pair_collision"][key] = hit
                if ei == ej:
                    continue
                log_first = i if ei < ej else j
                outcomes = []
                for r in range(poses.shape[1]):
                    pa = np.concatenate((log[i, :ANCHOR + 1, :2], poses[b, r, i, :, :2]))
                    pb = np.concatenate((log[j, :ANCHOR + 1, :2], poses[b, r, j, :, :2]))
                    ea, eb = entry_frames(pa, pb)
                    if ea is None:
                        outcomes.append("no_conflict")
                    elif ea == eb:
                        outcomes.append("tie")
                    else:
                        outcomes.append("same_as_log" if (i if ea < eb else j) == log_first else "reversed")
                rec["order"][key] = outcomes
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--labels", default=str(compat.DATA / "pair_order_z_labels_v1" / "val.npz"))
    a = p.parse_args()
    run = Path(a.run)
    compat.patch_map_cache_identity()
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    val_paths = WaymoVectorDataset(str(compat.VAL_DIR)).paths
    with np.load(a.labels) as d:
        names, offsets, pairs = d["names"], d["offsets"], d["pairs"].astype(np.int64)
    labels = {"index": {str(n): (int(offsets[k]), int(offsets[k + 1])) for k, n in enumerate(names)},
              "pairs": pairs}

    data = {m: load_mode(run, m) for m in MODES}
    data = {m: v for m, v in data.items() if v is not None}
    if "unknown" not in data:
        raise SystemExit("baseline (unknown) result missing")
    base_idx = data["unknown"]["dataset_index"]
    labelled = np.array([labels["index"][Path(val_paths[int(k)]).name][1]
                         > labels["index"][Path(val_paths[int(k)]).name][0] for k in base_idx])
    summary = {"labelled_scenes": int(labelled.sum()), "scenes": int(len(base_idx)), "modes": {}}
    per_scene_col = {}
    pair_recs = {}
    for m, v in data.items():
        if not np.array_equal(v["dataset_index"], base_idx):
            raise ValueError(f"{m}: scene order differs from baseline")
        r, mi, gen = v["result"], v["metric_index"], v["gen"]
        ph = r.get("physical_metrics", {})
        hit = gen[:, :, mi["collision_scene_rollout_rate_proxy"], 0]
        has = gen[:, :, mi["collision_scene_rollout_rate_proxy"], 1]
        at = gen[:, :, mi["collision_agent_time_rate_proxy"]]
        per_scene_col[m] = np.where(has.sum(1) > 0, hit.sum(1) / np.maximum(1, has.sum(1)), np.nan)
        rec = pair_level(v, None, labels, val_paths, collision_gap)
        pair_recs[m] = rec
        pc = np.array([h.mean() for h in rec["pair_collision"].values()]) if rec["pair_collision"] else np.zeros(0)
        orders = [o for outs in rec["order"].values() for o in outs]
        ns = np.array(list(rec["never_start"].values())).sum(0)
        sl = np.array(list(rec["stuck_late"].values())).sum(0)
        ns_l = np.array([v_ for k_, v_ in rec["never_start"].items() if labelled[list(base_idx).index(k_)]]).sum(0)
        summary["modes"][m] = {
            "mean_ade_m": r["mean_ade_m"], "minade_m": r["minade_m"], "cpd": r["flow_erd_cpd"],
            "collision_scene_rollout": ph.get("collision_scene_rollout_rate_proxy"),
            "collision_agent_time": ph.get("collision_agent_time_rate_proxy"),
            "offroad_scene_rollout_reliable": ph.get("offroad_scene_rollout_rate_proxy_reliable"),
            "offroad_agent_time_reliable": ph.get("offroad_agent_time_rate_proxy_reliable"),
            "collision_scene_rollout_labelled_scenes": float(hit[labelled].sum() / max(1, has[labelled].sum())),
            "collision_agent_time_labelled_scenes": float(at[labelled, :, 0].sum() / max(1, at[labelled, :, 1].sum())),
            "collision_scene_rollout_unlabelled_scenes": float(hit[~labelled].sum() / max(1, has[~labelled].sum())),
            "labelled_pairs": int(len(pc)),
            "pair_collision_rate": float(pc.mean()) if len(pc) else None,
            "pass_order": {k: orders.count(k) / max(1, len(orders)) for k in ("same_as_log", "reversed", "no_conflict", "tie")},
            "never_start_rate": float(ns[0] / max(1, ns[1])), "never_start_eligible": int(ns[1]),
            "never_start_rate_labelled_scenes": float(ns_l[0] / max(1, ns_l[1])) if np.size(ns_l) == 2 else None,
            "stuck_late_rate": float(sl[0] / max(1, sl[1])), "stuck_late_eligible": int(sl[1]),
            "bias_stats": v["z_info"].get("bias_stats"),
        }
        zdir = v["dir"] / "z_states"
        if zdir.is_dir():
            counts = np.zeros(9, dtype=np.int64)
            for f in zdir.glob("batch_*.npz"):
                counts += np.bincount(np.load(f)["codes"].astype(np.int64).ravel(), minlength=9)[:9]
            tot = max(1, counts[1:].sum())
            summary["modes"][m]["oracle_pair_replan_codes"] = {CODE_NAMES[c]: float(counts[c] / tot) for c in range(1, 9)}

    for m in data:
        if m == "unknown":
            continue
        d = per_scene_col[m] - per_scene_col["unknown"]
        ok = np.isfinite(d)
        pk = [k for k in pair_recs[m]["pair_collision"] if k in pair_recs["unknown"]["pair_collision"]]
        by_scene = {}
        for k in pk:
            by_scene.setdefault(k[0], []).append(pair_recs[m]["pair_collision"][k].mean()
                                                 - pair_recs["unknown"]["pair_collision"][k].mean())
        summary["modes"][m]["bootstrap_vs_unknown"] = {
            "scene_collision_all": bootstrap(d[ok]),
            "scene_collision_labelled": bootstrap(d[ok & labelled]),
            "pair_collision_by_scene": bootstrap(np.array([np.mean(v_) for v_ in by_scene.values()])),
        }

    train_metrics = run / "train" / "metrics.jsonl"
    if train_metrics.is_file():
        vals = [json.loads(l) for l in train_metrics.read_text().splitlines() if '"val_all_loss_z"' in l]
        if vals:
            best = min(vals, key=lambda v_: v_["val_all_loss_z"])
            keys = ("val_all_gain_pct", "val_active_gain_pct", "val_yielder_gain_pct", "val_priority_gain_pct",
                    "val_active_in_window_gain_pct", "val_active_beyond_window_gain_pct")
            summary["train"] = {"best_step": best["step"], "last_step": vals[-1]["step"],
                                "best": {k: best.get(k) for k in keys}, "last": {k: vals[-1].get(k) for k in keys},
                                "best_bias": best.get("val_bias")}

    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    lines = ["# 实验 1 结果：oracle z，冻结 85k WM，只训练 g", "",
             f"512 场景 × 32 rollout，同一组种子。有标签车对的场景 {summary['labelled_scenes']} 个。", "",
             "| 指标 | " + " | ".join(data) + " |", "|---|" + "---|" * len(data)]
    rows = [("ADE", "mean_ade_m"), ("minADE", "minade_m"), ("CPD", "cpd"),
            ("碰撞 场景-rollout（全部）", "collision_scene_rollout"),
            ("碰撞 场景-rollout（有标签场景）", "collision_scene_rollout_labelled_scenes"),
            ("碰撞 场景-rollout（无标签场景）", "collision_scene_rollout_unlabelled_scenes"),
            ("碰撞 agent-time（全部）", "collision_agent_time"),
            ("碰撞 agent-time（有标签场景）", "collision_agent_time_labelled_scenes"),
            ("标签车对之间的碰撞", "pair_collision_rate"),
            ("offroad 场景-rollout", "offroad_scene_rollout_reliable"),
            ("一直不起步", "never_start_rate"), ("后 4 s 卡住", "stuck_late_rate")]
    for label, key in rows:
        vals_ = [summary["modes"][m].get(key) for m in data]
        lines.append(f"| {label} | " + " | ".join("—" if x is None else f"{x:.4f}" for x in vals_) + " |")
    for k in ("same_as_log", "reversed", "no_conflict", "tie"):
        lines.append(f"| 通行顺序：{k} | " + " | ".join(f"{summary['modes'][m]['pass_order'][k]:.3f}" for m in data) + " |")
    lines += ["", "## 场景配对 bootstrap（相对 unknown，负数 = 碰撞下降）", ""]
    for m in data:
        bs = summary["modes"][m].get("bootstrap_vs_unknown")
        if not bs:
            continue
        for k, v in bs.items():
            if v.get("n"):
                lines.append(f"- {m} {k}: {v['mean']:+.4f}，95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]，n={v['n']}，P(Δ≥0)≈{v['p_ge_0']:.3f}")
    if "train" in summary:
        lines += ["", "## 训练（验证集 flow loss：真值 z 相对 未知 的下降 %）", "",
                  f"best step {summary['train']['best_step']}：" + json.dumps(summary["train"]["best"], ensure_ascii=False)]
    for m in data:
        bsx = summary["modes"][m].get("bias_stats")
        if bsx:
            lines.append(f"- {m} bias：" + json.dumps(bsx, ensure_ascii=False))
        codes = summary["modes"][m].get("oracle_pair_replan_codes")
        if codes:
            lines.append(f"- {m} 重规划时车对状态比例：" + json.dumps({k: round(v, 3) for k, v in codes.items()}, ensure_ascii=False))
    (run / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
