#!/usr/bin/env python3
"""Where do rollout collisions come from?  Bounds what a crossing/opposing pair-order z can fix.

For one eval mode (default unknown), every scene-rollout, every colliding agent pair (proxy boxes overlap
at some step, both GT-valid) is put in one category:
  log_overlap      the same two agents also overlap in the log (proxy artefact / parked cars)
  labelled_pair    a labelled crossing/opposing vehicle pair (what z covers)
  veh_same_dir     vehicle-vehicle, heading difference < 30 deg at first contact (rear-end, side-swipe)
  veh_crossing_unl vehicle-vehicle, >= 30 deg, not labelled (log paths do not overlap, or pre-existing)
  vru              involves a pedestrian or cyclist
For labelled pairs it also reports collision rate by rollout pass order (log / reversed / tie / none).
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

THR, ANCHOR = 2.5, 10


def entry(pa, pb):
    dd = np.linalg.norm(pa[:, None] - pb[None], axis=-1) < THR
    if not dd.any():
        return None, None
    return int(np.flatnonzero(dd.any(1))[0]), int(np.flatnonzero(dd.any(0))[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--mode", default="unknown")
    a = p.parse_args()
    compat.patch_map_cache_identity()
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset
    from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap
    val_paths = WaymoVectorDataset(str(compat.VAL_DIR)).paths
    with np.load(compat.DATA / "pair_order_z_labels_v1" / "val.npz") as d:
        names, offsets, pairs_all = d["names"], d["offsets"], d["pairs"].astype(np.int64)
    index = {str(n): (int(offsets[k]), int(offsets[k + 1])) for k, n in enumerate(names)}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cats = ("log_overlap", "labelled_pair", "veh_same_dir", "veh_crossing_unl", "vru")
    pair_counts = dict.fromkeys(cats, 0)
    sr_any = dict.fromkeys(cats, 0)       # scene-rollouts with >= 1 colliding pair of this category
    sr_only = dict.fromkeys(cats, 0)      # ... whose only colliding category is this one
    sr_total = sr_coll = 0
    order_hits = {}                       # outcome -> [collisions, count]
    for f in sorted((Path(a.run) / "eval" / a.mode / "rollout_batches").glob("batch_*.npz")):
        z = np.load(f)
        poses, valid, types, gt, init = z["poses"], z["valid"], z["agent_type"], z["gt_pose"], z["initial_pose"]
        for b, ds in enumerate(z["dataset_index"]):
            lo, hi = index[Path(val_paths[int(ds)]).name]
            lab = {(int(r[0]), int(r[1])): r for r in pairs_all[lo:hi]}
            lab.update({(j, i): r for (i, j), r in list(lab.items())})
            R, N, T = poses.shape[1], poses.shape[2], poses.shape[3]
            ty = torch.from_numpy(types[b]).long().to(dev)
            gap = collision_gap(torch.from_numpy(poses[b]).float().to(dev), ty[None].expand(R, -1))  # (R,T,N,N)
            gt_b = np.where(valid[b][..., None], gt[b], np.nan)
            gt_gap = collision_gap(torch.from_numpy(np.nan_to_num(gt_b)).float()[None].to(dev), ty[None])[0]
            both = torch.from_numpy(valid[b]).to(dev)
            both = both[:, None, :] & both[None, :, :]                                   # (N,N,T)
            both = both.permute(2, 0, 1)                                                 # (T,N,N)
            eye = torch.eye(N, dtype=torch.bool, device=dev)
            hit = (gap < 0) & both[None] & ~eye                                          # (R,T,N,N)
            log_hit = ((gt_gap < 0) & both & ~eye).any(0).cpu().numpy()                  # (N,N)
            first_t = torch.where(hit.any(1), hit.float().argmax(1), -1).cpu().numpy()   # (R,N,N)
            with np.load(Path(val_paths[int(ds)])) as s:
                log = s["agents"]
            for r in range(R):
                sr_total += 1
                ii, jj = np.nonzero(np.triu(first_t[r] >= 0, 1))
                seen = set()
                for i, j in zip(ii, jj):
                    if log_hit[i, j]:
                        c = "log_overlap"
                    elif (i, j) in lab:
                        c = "labelled_pair"
                    elif types[b, i] != 1 or types[b, j] != 1:
                        c = "vru"
                    else:
                        t = first_t[r, i, j]
                        dyaw = np.angle(np.exp(1j * (poses[b, r, i, t, 2] - poses[b, r, j, t, 2])))
                        c = "veh_same_dir" if abs(np.degrees(dyaw)) < 30 else "veh_crossing_unl"
                    pair_counts[c] += 1
                    seen.add(c)
                if seen:
                    sr_coll += 1
                    for c in seen:
                        sr_any[c] += 1
                    if len(seen) == 1:
                        sr_only[next(iter(seen))] += 1
            for (i, j), rec in lab.items():
                if i > j:
                    continue
                ei, ej = int(rec[2]), int(rec[4])
                if ei == ej:
                    continue
                pr, yl = (i, j) if ei < ej else (j, i)
                for r in range(R):
                    pa = np.concatenate((log[pr, :ANCHOR + 1, :2], poses[b, r, pr, :, :2]))
                    pb = np.concatenate((log[yl, :ANCHOR + 1, :2], poses[b, r, yl, :, :2]))
                    ea, eb = entry(pa, pb)
                    o = "none" if ea is None else "tie" if ea == eb else "log" if ea < eb else "rev"
                    h = order_hits.setdefault(o, [0, 0])
                    h[0] += int(first_t[r, i, j] >= 0)
                    h[1] += 1
    tot_pairs = max(1, sum(pair_counts.values()))
    out = {
        "mode": a.mode, "scene_rollouts": sr_total, "colliding_scene_rollouts": sr_coll,
        "colliding_pairs_by_category": {c: [pair_counts[c], round(pair_counts[c] / tot_pairs, 3)] for c in cats},
        "share_of_colliding_scene_rollouts_with_category": {c: round(sr_any[c] / max(1, sr_coll), 3) for c in cats},
        "share_of_colliding_scene_rollouts_only_this_category": {c: round(sr_only[c] / max(1, sr_coll), 3) for c in cats},
        "labelled_pair_collision_by_pass_order": {o: {"rate": round(h[0] / max(1, h[1]), 4), "n": h[1], "hits": h[0]}
                                                  for o, h in sorted(order_hits.items())},
    }
    Path(a.run, f"collision_breakdown_{a.mode}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
