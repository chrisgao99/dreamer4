#!/usr/bin/env python3
"""Summary of 1b-v2: fine-tuned WM with z v2, modes unknown / oracle / reversed (same seeds), plus the
frozen 85k baseline (exp 1 eval/unknown, 32 rollouts, different seeds: compared in aggregate only).

Pair-level metrics use labels v2. "z set" = pairs whose z is on at the first replan of the oracle run
(codes 1/2/3/10, identical across rollouts because the start state is shared) and whose order is not
already decided in the 1 s history (both cars enter the overlap after frame 10 in the log).
  pass order per pair and rollout (path = log frames 0-10 + rollout; entry = first frame within 2.5 m of
  the other's path): same_as_log / reversed / no_conflict / tie
  control = share of conflicts passing in the order z asked for (log order for oracle, reversed for
  reversed); for unknown models the log-order share is shown.
Writes <run>/summary.json and <run>/summary.md.
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
from waymo.pair_order_z.summarize_exp1 import ANCHOR, bootstrap, entry_frames, load_mode  # noqa: E402
from waymo.pair_order_z.z_bias_v2 import CODE_NAMES  # noqa: E402

ON = (1, 2, 3, 10)


def pair_level(mode_dir: Path, labels, val_paths, collision_gap, z_on_start):
    rec = {}
    for f in sorted((mode_dir / "rollout_batches").glob("batch_*.npz")):
        z = np.load(f)
        poses, valid, types = z["poses"], z["valid"], z["agent_type"]
        for b, ds in enumerate(z["dataset_index"]):
            ds = int(ds)
            lo, hi = labels["index"][Path(val_paths[ds]).name]
            if hi == lo:
                continue
            with np.load(Path(val_paths[ds])) as s:
                log = s["agents"]
            gap = collision_gap(torch.from_numpy(poses[b]).float(),
                                torch.from_numpy(types[b])[None].long().expand(poses.shape[1], -1)).numpy()  # (R,T,N,N)
            for pidx, p in enumerate(labels["pairs"][lo:hi]):
                i, j, ei, xi, ej, xj, _, kind = (int(v) for v in p)
                both = valid[b, i] & valid[b, j]
                hit = ((gap[:, :, i, j] < 0) & both[None]).any(1)
                outs = []
                if ei != ej:
                    first = i if ei < ej else j
                    for r in range(poses.shape[1]):
                        pa = np.concatenate((log[i, :ANCHOR + 1, :2], poses[b, r, i, :, :2]))
                        pb = np.concatenate((log[j, :ANCHOR + 1, :2], poses[b, r, j, :, :2]))
                        ea, eb = entry_frames(pa, pb)
                        outs.append("no_conflict" if ea is None else "tie" if ea == eb else
                                    "same_as_log" if (i if ea < eb else j) == first else "reversed")
                key = (ds, pidx)
                rec[key] = dict(kind=kind, decided=min(ei, ej) <= ANCHOR, hit=hit, order=outs,
                                z_on=bool(z_on_start.get(key, False)))
    return rec


def aggregate(rec, sel):
    keys = [k for k, v in rec.items() if sel(v)]
    if not keys:
        return {"pairs": 0}
    hits = np.concatenate([rec[k]["hit"] for k in keys])
    orders = [o for k in keys for o in rec[k]["order"]]
    conf = [o for o in orders if o in ("same_as_log", "reversed")]
    return {"pairs": len(keys), "pair_collision": float(hits.mean()),
            "log_order_share_of_conflicts": conf.count("same_as_log") / max(1, len(conf)),
            "reversed_share_of_conflicts": conf.count("reversed") / max(1, len(conf)),
            "no_conflict_share": orders.count("no_conflict") / max(1, len(orders)), "conflicts": len(conf)}


def z_on_at_start(oracle_dir: Path):
    out = {}
    for f in sorted((oracle_dir / "z_states").glob("batch_*.npz")):
        z = np.load(f)
        codes = z["codes"]                                   # (R, K, B, P)
        first = codes[:, 0]                                  # (R, B, P) first replan
        on = np.isin(first, ON).any(0)                       # (B, P)
        out[int(f.stem.split("_")[1])] = on
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--baseline_run", default=str(Path(__file__).resolve().parent / "runs/exp1_oracle_z_frozen_20261110"))
    p.add_argument("--labels", default=str(compat.DATA / "pair_order_z_labels_v2" / "val.npz"))
    a = p.parse_args()
    run = Path(a.run)
    compat.patch_map_cache_identity()
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    val_paths = WaymoVectorDataset(str(compat.VAL_DIR)).paths
    with np.load(a.labels) as d:
        names, offsets, pairs = d["names"], d["offsets"], d["pairs"].astype(np.int64)
    labels = {"index": {str(n): (int(offsets[k]), int(offsets[k + 1])) for k, n in enumerate(names)}, "pairs": pairs}

    dirs = {"85k frozen, unknown (32 rollouts)": Path(a.baseline_run) / "eval/unknown"}
    dirs.update({f"1b-v2 {m}": run / "eval" / m for m in ("unknown", "oracle", "reversed")})
    data = {k: load_mode(v.parent.parent, v.name) for k, v in dirs.items()}
    data = {k: v for k, v in data.items() if v is not None}
    oracle_key = "1b-v2 oracle"
    if oracle_key not in data:
        raise SystemExit("oracle eval missing")
    on_by_batch = z_on_at_start(dirs[oracle_key])
    base_idx = data[oracle_key]["dataset_index"]
    z_on = {}
    for f in sorted((dirs[oracle_key] / "rollout_batches").glob("batch_*.npz")):
        bi = int(f.stem.split("_")[1])
        for b, ds in enumerate(np.load(f)["dataset_index"]):
            lo, hi = labels["index"][Path(val_paths[int(ds)]).name]
            for pidx in range(hi - lo):
                z_on[(int(ds), pidx)] = bool(on_by_batch[bi][b, pidx])

    summary = {"modes": {}, "z_on_pairs_at_start": int(sum(z_on.values())), "labelled_pairs": len(z_on)}
    recs, scene_col = {}, {}
    sets = {
        "all pairs": lambda v: True,
        "z set, crossing": lambda v: v["z_on"] and not v["decided"] and v["kind"] == 0,
        "z set, same direction": lambda v: v["z_on"] and not v["decided"] and v["kind"] == 1,
        "z set, all": lambda v: v["z_on"] and not v["decided"],
    }
    for m, v in data.items():
        if not np.array_equal(v["dataset_index"], base_idx):
            raise ValueError(f"{m}: scene order differs")
        r, mi, gen = v["result"], v["metric_index"], v["gen"]
        ph = r.get("physical_metrics", {})
        hit = gen[:, :, mi["collision_scene_rollout_rate_proxy"], 0]
        has = gen[:, :, mi["collision_scene_rollout_rate_proxy"], 1]
        scene_col[m] = np.where(has.sum(1) > 0, hit.sum(1) / np.maximum(1, has.sum(1)), np.nan)
        rec = pair_level(dirs[m], labels, val_paths, collision_gap, z_on)
        recs[m] = rec
        summary["modes"][m] = {
            "mean_ade_m": r["mean_ade_m"], "minade_m": r["minade_m"], "cpd": r["flow_erd_cpd"],
            "collision_scene_rollout": ph.get("collision_scene_rollout_rate_proxy"),
            "collision_agent_time": ph.get("collision_agent_time_rate_proxy"),
            "offroad_scene_rollout_reliable": ph.get("offroad_scene_rollout_rate_proxy_reliable"),
            "pairs": {name: aggregate(rec, sel) for name, sel in sets.items()},
            "bias_stats": v["z_info"].get("bias_stats"),
        }
        zdir = dirs[m] / "z_states"
        if zdir.is_dir():
            counts = np.zeros(11, dtype=np.int64)
            for f in zdir.glob("batch_*.npz"):
                counts += np.bincount(np.load(f)["codes"].astype(np.int64).ravel(), minlength=11)[:11]
            tot = max(1, counts[1:].sum())
            summary["modes"][m]["replan_codes"] = {CODE_NAMES[c]: float(counts[c] / tot) for c in range(1, 11)}

    ref = "1b-v2 unknown"
    for m in ("1b-v2 oracle", "1b-v2 reversed"):
        if m not in data or ref not in data:
            continue
        d = scene_col[m] - scene_col[ref]
        out = {"scene_collision_all": bootstrap(d[np.isfinite(d)])}
        for name in ("z set, all", "z set, crossing", "z set, same direction"):
            sel = sets[name]
            by_scene, rev_scene = {}, {}
            for k, v in recs[m].items():
                if not sel(v) or k not in recs[ref]:
                    continue
                by_scene.setdefault(k[0], []).append(v["hit"].mean() - recs[ref][k]["hit"].mean())
                cm = [o for o in v["order"] if o in ("same_as_log", "reversed")]
                cu = [o for o in recs[ref][k]["order"] if o in ("same_as_log", "reversed")]
                if cm and cu:
                    rev_scene.setdefault(k[0], []).append(cm.count("reversed") / len(cm) - cu.count("reversed") / len(cu))
            out[f"pair_collision_by_scene [{name}]"] = bootstrap(np.array([np.mean(x) for x in by_scene.values()])) if by_scene else {}
            out[f"reversed_share_by_scene [{name}]"] = bootstrap(np.array([np.mean(x) for x in rev_scene.values()])) if rev_scene else {}
        summary["modes"][m]["bootstrap_vs_1b_unknown"] = out

    tm = run / "train" / "metrics.jsonl"
    if tm.is_file():
        vals = [json.loads(l) for l in tm.read_text().splitlines() if '"val_all_loss_z"' in l]
        keys = ("val_all_loss_unknown", "val_all_gain_pct", "val_z_gain_pct", "val_crossing_gain_pct",
                "val_same_dir_gain_pct", "val_yielder_gain_pct", "val_priority_gain_pct", "val_other_gain_pct")
        summary["train"] = [{"step": v["step"], **{k: v.get(k) for k in keys}} for v in vals]

    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    cols = list(data)
    L = ["# 实验 1b-v2 结果：z v2（加同向车对），解冻去噪器，loss 集中在有 z 的车上", "",
         f"512 场景；1b-v2 三种模式每场景 16 个 rollout、同一组种子；85k 基线是实验 1 的 32 个 rollout（种子不同，只比总体）。"
         f"标签 v2 车对 {summary['labelled_pairs']} 对，开始时 z 开着的 {summary['z_on_pairs_at_start']} 对。", "",
         "| 指标 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for label, key in (("ADE", "mean_ade_m"), ("minADE", "minade_m"), ("CPD", "cpd"),
                       ("碰撞 场景-rollout", "collision_scene_rollout"), ("碰撞 agent-time", "collision_agent_time"),
                       ("offroad 场景-rollout", "offroad_scene_rollout_reliable")):
        L.append(f"| {label} | " + " | ".join("—" if summary["modes"][m].get(key) is None else f"{summary['modes'][m][key]:.4f}" for m in cols) + " |")
    for name in sets:
        for label, key in (("车对碰撞", "pair_collision"), ("冲突中按日志顺序", "log_order_share_of_conflicts"),
                           ("冲突中反序", "reversed_share_of_conflicts"), ("没相遇", "no_conflict_share")):
            vals = [summary["modes"][m]["pairs"][name].get(key) for m in cols]
            L.append(f"| {name}：{label} | " + " | ".join("—" if x is None else f"{x:.3f}" for x in vals) + " |")
        L.append(f"| {name}：车对数 | " + " | ".join(str(summary["modes"][m]["pairs"][name].get("pairs", 0)) for m in cols) + " |")
    L += ["", "## 场景配对 bootstrap（相对 1b-v2 unknown，同一组种子）", ""]
    for m in cols:
        for k, v in (summary["modes"][m].get("bootstrap_vs_1b_unknown") or {}).items():
            if v.get("n"):
                L.append(f"- {m} {k}: {v['mean']:+.4f}，95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]，n={v['n']}，P(Δ≥0)≈{v['p_ge_0']:.3f}")
    if "train" in summary and summary["train"]:
        L += ["", "## 训练（EMA 权重，验证集 flow loss；gain = 真值 z 相对未知的下降 %）", "",
              "| step | " + " | ".join(k.replace("val_", "") for k in summary["train"][0] if k != "step") + " |",
              "|---|" + "---|" * (len(summary["train"][0]) - 1)]
        for row in summary["train"]:
            L.append(f"| {row['step']} | " + " | ".join("—" if v is None else f"{v:.4f}" for k, v in row.items() if k != "step") + " |")
    for m in cols:
        if summary["modes"][m].get("replan_codes"):
            L.append(f"- {m} 重规划时车对状态比例：" + json.dumps({k: round(v, 3) for k, v in summary["modes"][m]["replan_codes"].items()}, ensure_ascii=False))
        if summary["modes"][m].get("bias_stats"):
            L.append(f"- {m} bias：" + json.dumps(summary["modes"][m]["bias_stats"], ensure_ascii=False))
    (run / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
