"""Paired pair-level diagnostics for exp 1: does z move the cars at all, and in which direction?"""
import sys
from pathlib import Path

import numpy as np

D = Path("/scratch/baz7dy/tri30/dreamer4")
sys.path.insert(0, str(D))
from waymo.pair_order_z import compat  # noqa: E402
from waymo.pair_order_z.make_z_labels import label_code_at  # noqa: E402

RUN = D / "waymo/pair_order_z/runs/exp1_oracle_z_frozen_20261110"
THR, ANCHOR = 2.5, 10
compat.patch_map_cache_identity()
from waymo.core.waymo_vector_dataset import WaymoVectorDataset  # noqa: E402

val_paths = WaymoVectorDataset(str(compat.VAL_DIR)).paths
with np.load(compat.DATA / "pair_order_z_labels_v1" / "val.npz") as d:
    names, offsets, pairs_all = d["names"], d["offsets"], d["pairs"].astype(np.int64)
index = {str(n): (int(offsets[k]), int(offsets[k + 1])) for k, n in enumerate(names)}


def entry(pa, pb):
    dd = np.linalg.norm(pa[:, None] - pb[None], axis=-1) < THR
    if not dd.any():
        return None, None
    return int(np.flatnonzero(dd.any(1))[0]), int(np.flatnonzero(dd.any(0))[0])


def cumlen(init_xy, poses_xy):  # (R,T,2) -> (R,T) cumulative path length
    xy = np.concatenate((np.repeat(init_xy[None, None], poses_xy.shape[0], 0), poses_xy), 1)
    return np.cumsum(np.linalg.norm(np.diff(xy, axis=1), axis=-1), 1)


rows = []          # one per (pair, rollout)
ctrl = {m: [] for m in ("oracle", "reversed")}   # displacement diffs of non-pair vehicles
files = sorted((RUN / "eval/unknown/rollout_batches").glob("batch_*.npz"))
for f in files:
    Z = {m: np.load(RUN / "eval" / m / "rollout_batches" / f.name) for m in ("unknown", "oracle", "reversed")}
    assert all(np.array_equal(Z[m]["dataset_index"], Z["unknown"]["dataset_index"]) for m in Z)
    for b, ds in enumerate(Z["unknown"]["dataset_index"]):
        lo, hi = index[Path(val_paths[int(ds)]).name]
        if hi == lo:
            continue
        P = {m: Z[m]["poses"][b][..., :2] for m in Z}          # (R,N,T,2)
        init = Z["unknown"]["initial_pose"][b][:, :2]
        valid = Z["unknown"]["valid"][b]
        types = Z["unknown"]["agent_type"][b]
        with np.load(Path(val_paths[int(ds)])) as s:
            log = s["agents"]
        in_pair = set()
        for p in pairs_all[lo:hi]:
            in_pair |= {int(p[0]), int(p[1])}
        others = [n for n in range(len(types)) if types[n] == 1 and n not in in_pair
                  and np.isfinite(init[n]).all() and valid[n].any()]
        for m in ("oracle", "reversed"):
            for n in others:
                ctrl[m].append(np.linalg.norm(P[m][:, n] - P["unknown"][:, n], axis=-1).mean(0))  # (T,)
        for p in pairs_all[lo:hi]:
            i, j, ei, xi, ej, xj, ang = (int(v) for v in p)
            if ei == ej:
                continue
            pr, yl = (i, j) if ei < ej else (j, i)
            code10 = label_code_at(p, ANCHOR)
            L = {m: {a: cumlen(init[a], P[m][:, a]) for a in (pr, yl)} for m in P}
            for r in range(P["unknown"].shape[0]):
                row = {"ds": int(ds), "pair": (i, j), "r": r, "code10": code10, "angle": ang,
                       "y_entry_log": max(ei, ej)}
                for m in P:
                    pa = np.concatenate((log[pr, :ANCHOR + 1, :2], P[m][r, pr]))
                    pb = np.concatenate((log[yl, :ANCHOR + 1, :2], P[m][r, yl]))
                    ea, eb = entry(pa, pb)
                    row[m] = ("none" if ea is None else "tie" if ea == eb else "log" if ea < eb else "rev")
                    row[m + "_gap"] = None if ea is None else eb - ea      # yielder entry - priority entry
                    for a, tag in ((pr, "p"), (yl, "y")):
                        for t in (19, 39, 79):
                            row[f"{m}_{tag}len{t + 1}"] = L[m][a][r, t]
                for m in ("oracle", "reversed"):
                    for a, tag in ((pr, "p"), (yl, "y")):
                        row[f"{m}_{tag}disp"] = np.linalg.norm(P[m][r, a] - P["unknown"][r, a], axis=-1)  # (T,)
                rows.append(row)

print(f"labelled pair x rollout rows: {len(rows)}  (pairs {len(rows) // 32})")
codes = np.array([r["code10"] for r in rows])
print("code at anchor t=10 (fraction of pairs):",
      {c: round(float((codes == c).mean()), 3) for c in sorted(set(codes.tolist()))})


def disp_report(sel, label):
    print(f"\n== {label}: {sel.sum() // 32} pairs")
    for m in ("oracle", "reversed"):
        for tag in ("y", "p"):
            d = np.stack([r[f"{m}_{tag}disp"] for r, s in zip(rows, sel) if s])
            print(f"  {m:8s} {'yielder ' if tag == 'y' else 'priority'} |Δpos vs unknown| "
                  f"@2s {d[:, 19].mean():.3f}  @4s {d[:, 39].mean():.3f}  @8s {d[:, 79].mean():.3f} m")
        c = np.stack(ctrl[m])
        print(f"  {m:8s} non-pair veh (same scenes)        "
              f"@2s {c[:, 19].mean():.3f}  @4s {c[:, 39].mean():.3f}  @8s {c[:, 79].mean():.3f} m")
    for m in ("oracle", "reversed"):
        for tag in ("y", "p"):
            for t in (20, 40, 80):
                d = np.array([r[f"{m}_{tag}len{t}"] - r[f"unknown_{tag}len{t}"] for r, s in zip(rows, sel) if s])
                se = d.reshape(-1, 32).mean(1).std() / np.sqrt(len(d) / 32)
                print(f"  {m:8s} Δ distance travelled {'yielder ' if tag == 'y' else 'priority'} "
                      f"@{t / 10:.0f}s {d.mean():+.3f} m (±{se:.3f} pair-SE)", end="")
            print()
    trans = {}
    for r, s in zip(rows, sel):
        if s:
            for m in ("oracle", "reversed"):
                trans.setdefault(m, {}).setdefault((r["unknown"], r[m]), 0)
                trans[m][(r["unknown"], r[m])] += 1
    for m in ("oracle", "reversed"):
        tot = sum(trans[m].values())
        changed = sum(v for (a, b), v in trans[m].items() if a != b)
        print(f"  {m:8s} order outcome changed vs unknown: {changed}/{tot} = {changed / tot:.3f};",
              {f"{a}->{b}": v for (a, b), v in sorted(trans[m].items()) if a != b})
    for m in ("unknown", "oracle", "reversed"):
        o = [r[m] for r, s in zip(rows, sel) if s]
        n_log, n_rev = o.count("log"), o.count("rev")
        g = [r[m + "_gap"] for r, s in zip(rows, sel) if s and r[m + "_gap"] is not None]
        print(f"  {m:8s} log-order share among conflicts {n_log / max(1, n_log + n_rev):.3f} "
              f"(log {n_log}, rev {n_rev}, none {o.count('none')}); mean entry gap {np.mean(g):.2f} frames")


disp_report(np.ones(len(rows), bool), "all labelled pairs")
disp_report(np.isin(codes, [1, 2, 3, 4]), "pairs still unresolved at t=10 (z carries an order)")
disp_report(codes == 5, "pairs already released at t=10")
