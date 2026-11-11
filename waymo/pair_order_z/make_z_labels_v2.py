#!/usr/bin/env python3
"""Labels v2 (denser z, see z_bias_v2): every vehicle-vehicle pair whose logged paths overlap in space.

Same overlap / entry / exit definition as make_z_labels.py (centre distance < 2.5 m to the other's
logged path, not necessarily at the same time). Differences from v1:
  * same-direction pairs (heading difference < 30 deg at the closest approach) are kept, kind = 1,
    unless a car of the pair moves < 2 m over its whole log (parked / static);
  * pre-existing pairs (a car already in the overlap before frame 5) are kept;
  * ties stay in the file and are labelled unknown, as in v1.
Output <out>/<split>.npz: names, offsets, pairs (P, 8) int16 = i, j, enter_i, exit_i, enter_j, exit_j,
angle_deg, kind; plus <split>_summary.json with counts and the fraction of pairs that are on (codes 1-3,
same-direction rules applied on the logged state) at anchors t = 10..50.
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
from waymo.pair_order_z.make_z_labels import HALF_WIDTH, MARGIN_M, MIN_ANGLE_DEG, PRE_EXISTING_FRAME, VEHICLE, label_code_at  # noqa: E402,E501
from waymo.pair_order_z.z_bias_v2 import GAP_MIN_SPEED_MPS, SAME_DIR_MAX_DIST_M, SAME_DIR_MAX_GAP_S  # noqa: E402

ANCHORS = (10, 20, 30, 40, 50)
STATIC_PATH_M = 2.0


def pairs_from_agents(agents: np.ndarray, mask: np.ndarray):
    agents = agents.astype(np.float64)
    valid = (agents[..., 5] > 0.5) & np.isfinite(agents[..., :2]).all(-1) & np.isfinite(agents[..., 6])
    valid &= mask[:, None]
    has = valid.any(1)
    types = np.zeros(len(agents), dtype=np.int64)
    for n in np.flatnonzero(has):
        types[n] = int(round(float(np.median(agents[n, valid[n], 7]))))
    veh = np.flatnonzero(has & (types == VEHICLE) & (valid.sum(1) >= 2))
    xy, yaw = agents[..., :2], agents[..., 6]
    seg = np.where(valid[:, 1:] & valid[:, :-1], np.linalg.norm(np.diff(np.nan_to_num(xy), axis=1), axis=-1), 0.0)
    path_len = seg.sum(1)
    thr = 2 * HALF_WIDTH[VEHICLE] + MARGIN_M
    out = []
    for a_idx, i in enumerate(veh):
        for j in veh[a_idx + 1:]:
            dist = np.linalg.norm(xy[i][:, None, :] - xy[j][None, :, :], axis=-1)
            both = valid[i][:, None] & valid[j][None, :]
            close = (dist < thr) & both
            if not close.any():
                continue
            masked = np.where(both, dist, np.inf)
            fi, fj = np.unravel_index(np.argmin(masked), masked.shape)
            angle = abs(np.degrees(np.arctan2(np.sin(yaw[i, fi] - yaw[j, fj]), np.cos(yaw[i, fi] - yaw[j, fj]))))
            oi, oj = np.flatnonzero(close.any(1)), np.flatnonzero(close.any(0))
            kind = 0 if angle >= MIN_ANGLE_DEG else 1
            if kind == 1 and min(path_len[i], path_len[j]) < STATIC_PATH_M:
                continue
            out.append((int(i), int(j), int(oi[0]), int(oi[-1]), int(oj[0]), int(oj[-1]), int(round(angle)), kind))
    return np.asarray(out, dtype=np.int16).reshape(-1, 8), agents, valid


def on_at(rec, t, agents, valid):
    """Code at anchor t on the logged state (v1 rule; same direction: both-in -> front first, gate)."""
    code = label_code_at(rec[:7], t)
    if rec[7] == 1 and code == 4:
        code = 10
    if code in (1, 2, 3, 10) and rec[7] == 1:
        i, j = int(rec[0]), int(rec[1])
        if not (valid[i, t] and valid[j, t]):
            return 9
        d = float(np.linalg.norm(agents[i, t, :2] - agents[j, t, :2]))
        v = max(abs(float(agents[i, t, 2])), abs(float(agents[j, t, 2])), GAP_MIN_SPEED_MPS)
        if d > SAME_DIR_MAX_DIST_M or d / v > SAME_DIR_MAX_GAP_S:
            return 9
    return code


def _work(path: str):
    try:
        with np.load(path, allow_pickle=False) as d:
            agents, mask = d["agents"], d["agent_mask"].astype(bool)
        pairs, ag, valid = pairs_from_agents(agents, mask)
        on = np.zeros((2, len(ANCHORS)), np.int64)          # kind x anchor: pairs on (codes 1-3)
        codes = np.zeros((2, len(ANCHORS), 11), np.int64)
        for rec in pairs:
            for k, t in enumerate(ANCHORS):
                c = on_at(rec, t, ag, valid)
                codes[rec[7], k, c] += 1
                on[rec[7], k] += c in (1, 2, 3, 10)
        agents_on = np.zeros(len(ANCHORS), np.int64)        # agents in >= 1 on pair at anchor t
        for k, t in enumerate(ANCHORS):
            members = set()
            for rec in pairs:
                if on_at(rec, t, ag, valid) in (1, 2, 3, 10):
                    members |= {int(rec[0]), int(rec[1])}
            agents_on[k] = len(members)
        return Path(path).name, pairs, dict(on=on, codes=codes, agents_on=agents_on,
                                            agents_alive=int(mask.sum()), vehicles=int(((ag[:, 10, 7] == 1) & mask).sum())), None
    except Exception as exc:  # report, never silently drop
        return Path(path).name, np.zeros((0, 8), np.int16), {}, repr(exc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="+", default=["val", "train"])
    p.add_argument("--out", default=str(compat.DATA / "pair_order_z_labels_v2"))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
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
        pairs = np.concatenate([r[1] for r in results]) if counts.sum() else np.zeros((0, 8), np.int16)
        tmp = out / f"{split}.tmp.npz"
        np.savez_compressed(tmp, names=names, offsets=offsets, pairs=pairs)
        os.replace(tmp, out / f"{split}.npz")

        on = sum(r[2]["on"] for r in results)
        codes = sum(r[2]["codes"] for r in results)
        agents_on = sum(r[2]["agents_on"] for r in results)
        alive = sum(r[2]["agents_alive"] for r in results)
        kind = pairs[:, 7] if len(pairs) else np.zeros(0)
        pre = np.minimum(pairs[:, 2], pairs[:, 4]) < PRE_EXISTING_FRAME if len(pairs) else np.zeros(0, bool)
        by_t = {}
        for k, t in enumerate(ANCHORS):
            by_t[str(t)] = {
                "crossing_on_per_sample": float(on[0, k] / len(files)),
                "same_dir_on_per_sample": float(on[1, k] / len(files)),
                "agents_in_on_pair_fraction": float(agents_on[k] / max(1, alive)),
                "crossing_codes": {name: int(codes[0, k, c]) for c, name in _names().items()},
                "same_dir_codes": {name: int(codes[1, k, c]) for c, name in _names().items()},
            }
        summary = {
            "split": split, "samples": len(files), "seconds": round(time.time() - t0, 1),
            "pairs": int(len(pairs)), "crossing_pairs": int((kind == 0).sum()), "same_dir_pairs": int((kind == 1).sum()),
            "pre_existing_pairs": int(pre.sum()), "tie_pairs": int((pairs[:, 2] == pairs[:, 4]).sum()) if len(pairs) else 0,
            "pairs_per_sample_mean": float(counts.mean()), "pairs_per_sample_max": int(counts.max()),
            "samples_with_pairs": int((counts > 0).sum()),
            "pairs_per_sample_quantiles_50_90_99": [float(np.quantile(counts, q)) for q in (.5, .9, .99)],
            "on_by_anchor_t": by_t,
            "definition": {"half_width_m": HALF_WIDTH[VEHICLE], "margin_m": MARGIN_M, "same_dir_below_deg": MIN_ANGLE_DEG,
                           "pre_existing_kept": True, "vehicle_only": True, "same_dir_static_path_below_m": STATIC_PATH_M,
                           "same_dir_gate": {"max_dist_m": SAME_DIR_MAX_DIST_M, "max_gap_s": SAME_DIR_MAX_GAP_S,
                                             "min_speed_mps": GAP_MIN_SPEED_MPS}},
        }
        (out / f"{split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "on_by_anchor_t"}, indent=1), flush=True)
        print(json.dumps({t: {k: v for k, v in d.items() if not k.endswith("codes")} for t, d in by_t.items()}), flush=True)


def _names():
    from waymo.pair_order_z.z_bias_v2 import CODE_NAMES
    return CODE_NAMES


if __name__ == "__main__":
    main()
