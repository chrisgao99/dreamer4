#!/usr/bin/env python3
"""Precompute crossing/opposing vehicle-pair records from the logged 9.1 s tracks (plan §8.2).

Pair: both vehicles (type 1) in agent_mask; logged paths overlap in space
(center distance < half-width sum + 0.5 m = 2.5 m, not necessarily at the same
time); heading difference at the closest approach >= 30 deg (same-direction pairs
are excluded). Per vehicle, its overlap segment is the set of its own logged
frames within 2.5 m of the other's logged path: enter = first such frame, exit =
last. As in the audit (plan §2), pairs where either vehicle is already in the
overlap before frame 5 (``pre_existing``) are dropped; same-frame entries (ties)
are kept and always labelled unknown.

Output (per split): ``<out>/<split>.npz`` with
  names (S,)            sample file names (dataset key)
  offsets (S+1,)        pairs[offsets[s]:offsets[s+1]] belong to names[s]
  pairs (P, 7) int16    i, j, enter_i, exit_i, enter_j, exit_j, angle_deg   (i < j, slot indices)
plus ``<split>_summary.json`` (counts, label-class fractions by anchor t).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from waymo.pair_order_z import compat  # noqa: E402

VEHICLE = 1
HALF_WIDTH = {1: 1.0, 2: 0.4, 3: 0.4}     # direct_action_flow_mon_losses.footprint widths / 2
MARGIN_M = 0.5
MIN_ANGLE_DEG = 30.0
PRE_EXISTING_FRAME = 5


def pairs_for_sample(path: str, drop_pre_existing: bool = True) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as d:
        agents = d["agents"].astype(np.float64)        # (N, T, 8)
        mask = d["agent_mask"].astype(bool)
    valid = (agents[..., 5] > 0.5) & np.isfinite(agents[..., :2]).all(-1) & np.isfinite(agents[..., 6])
    valid &= mask[:, None]
    has = valid.any(1)
    types = np.zeros(len(agents), dtype=np.int64)
    for n in np.flatnonzero(has):
        types[n] = int(round(float(np.median(agents[n, valid[n], 7]))))
    veh = np.flatnonzero(has & (types == VEHICLE) & (valid.sum(1) >= 2))
    stats = {"vehicles": int(len(veh)), "overlap": 0, "same_dir": 0, "pre_existing": 0, "tie": 0}
    out = []
    xy = agents[..., :2]
    yaw = agents[..., 6]
    thr = 2 * HALF_WIDTH[VEHICLE] + MARGIN_M
    for a_idx, i in enumerate(veh):
        for j in veh[a_idx + 1:]:
            dist = np.linalg.norm(xy[i][:, None, :] - xy[j][None, :, :], axis=-1)
            both = valid[i][:, None] & valid[j][None, :]
            close = (dist < thr) & both
            if not close.any():
                continue
            stats["overlap"] += 1
            masked = np.where(both, dist, np.inf)
            fi, fj = np.unravel_index(np.argmin(masked), masked.shape)
            angle = abs(np.degrees(np.arctan2(np.sin(yaw[i, fi] - yaw[j, fj]),
                                              np.cos(yaw[i, fi] - yaw[j, fj]))))
            if angle < MIN_ANGLE_DEG:
                stats["same_dir"] += 1
                continue
            oi = np.flatnonzero(close.any(1))
            oj = np.flatnonzero(close.any(0))
            ei, xi, ej, xj = int(oi[0]), int(oi[-1]), int(oj[0]), int(oj[-1])
            if drop_pre_existing and min(ei, ej) < PRE_EXISTING_FRAME:
                stats["pre_existing"] += 1
                continue
            stats["tie"] += int(ei == ej)
            out.append((int(i), int(j), ei, xi, ej, xj, int(round(angle))))
    return np.asarray(out, dtype=np.int16).reshape(-1, 7), stats


def label_code_at(rec: np.ndarray, t: int) -> int:
    """Same rule as z_bias.pair_codes with progress = t for both cars (log state)."""
    _, _, ei, xi, ej, xj, _ = (int(v) for v in rec)
    if ei == ej:
        return 7
    ef, xf, es, xs = (ei, xi, ej, xj) if ei < ej else (ej, xj, ei, xi)
    in_f, in_s = ef <= t <= xf, es <= t <= xs
    if t > xf or t > xs:
        return 5
    if in_f and in_s:
        return 4
    if in_s:
        return 3
    if in_f:
        return 2
    return 1


def _work(path: str):
    try:
        pairs, stats = pairs_for_sample(path)
        return Path(path).name, pairs, stats, None
    except Exception as exc:  # report, never silently drop
        return Path(path).name, np.zeros((0, 7), np.int16), {}, repr(exc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="+", default=["val", "train"])
    p.add_argument("--out", default=str(compat.DATA / "pair_order_z_labels_v1"))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        root = compat.TRAIN_DIR if split == "train" else compat.VAL_DIR
        files = sorted(str(f) for f in root.glob("*.npz"))
        if a.limit:
            files = files[: a.limit]
        t0 = time.time()
        with Pool(a.workers) as pool:
            results = pool.map(_work, files, chunksize=64)
        errors = [(n, e) for n, _, _, e in results if e]
        if errors:
            raise RuntimeError(f"{len(errors)} samples failed, first: {errors[0]}")
        names = np.asarray([r[0] for r in results])
        counts = np.asarray([len(r[1]) for r in results])
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        pairs = np.concatenate([r[1] for r in results]) if counts.sum() else np.zeros((0, 7), np.int16)
        tmp = out / f"{split}.tmp.npz"
        np.savez_compressed(tmp, names=names, offsets=offsets, pairs=pairs)
        os.replace(tmp, out / f"{split}.npz")

        totals = {}
        for _, _, s, _ in results:
            for k, v in s.items():
                totals[k] = totals.get(k, 0) + v
        code_names = {1: "pre_entry_log_order", 2: "first_in_overlap", 3: "second_in_overlap",
                      4: "both_in_unknown", 5: "released", 7: "tie_unknown"}
        by_t = {}
        for t in (10, 20, 30, 40, 50):
            c = np.bincount([label_code_at(r, t) for r in pairs], minlength=8)
            by_t[str(t)] = {code_names[k]: int(c[k]) for k in code_names}
            by_t[str(t)]["active_per_sample"] = float((c[1] + c[2] + c[3]) / max(1, len(files)))
        second_enter = np.maximum(pairs[:, 2], pairs[:, 4]) if len(pairs) else np.zeros(0)
        summary = {
            "split": split, "samples": len(files), "seconds": round(time.time() - t0, 1),
            "pairs_kept": int(len(pairs)), "pairs_per_sample_mean": float(counts.mean()),
            "pairs_per_sample_max": int(counts.max()), "samples_with_pairs": int((counts > 0).sum()),
            "pairs_per_sample_quantiles_50_90_99": [float(np.quantile(counts, q)) for q in (.5, .9, .99)],
            "filter_counts": totals, "label_classes_by_anchor_t": by_t,
            "angle_deg_quantiles_10_50_90": [float(np.quantile(pairs[:, 6], q)) for q in (.1, .5, .9)] if len(pairs) else [],
            "second_enter_frame_quantiles_10_50_90": [float(np.quantile(second_enter, q)) for q in (.1, .5, .9)] if len(pairs) else [],
            "definition": {"half_width_m": HALF_WIDTH[VEHICLE], "margin_m": MARGIN_M,
                           "min_angle_deg": MIN_ANGLE_DEG, "pre_existing_frame": PRE_EXISTING_FRAME,
                           "drop_pre_existing": True, "vehicle_only": True},
        }
        (out / f"{split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
