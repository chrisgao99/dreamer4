#!/usr/bin/env python3
"""Video of scene_test_1a.py rollouts (ipenv: matplotlib + /usr/bin/ffmpeg; no torch needed).

Panels are modes of the same paired rollout index k (same noise seed). Boxes are the proxy footprints of the
collision metric (vehicle 4.8 x 2.0 m); magenta edge = box overlapping another box at this step.
Ego red, the conflict car blue, others gray; history frames (first 1 s) are the log.
  ~/.conda/envs/ipenv/bin/python -I render_scene_test.py --dir <scene dir> --k 19 --modes oracle
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
try:  # /usr/bin/ffmpeg on this cluster is missing libvmaf; ipenv ships a static ffmpeg via imageio-ffmpeg
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

# Same style as waymo_interaction_screening/code/render_bank_videos.py (lane, road line, road edge, crosswalk...).
MAP_STYLE = {1: ("#c8c8c8", 0.6, "-"), 2: ("#8a8a8a", 0.8, "--"), 3: ("#333333", 1.2, "-"), 7: ("#d9c9a8", 0.8, "-")}
LABEL = {"unknown": "z = unknown (baseline)", "oracle": "oracle z (log order, g from 1a)", "reversed": "reversed z"}
CODE = {0: "-", 1: "before entry: yield per log order", 2: "first car in overlap: it goes first",
        3: "second car in overlap", 4: "both in overlap: unknown", 5: "released (first car passed)",
        6: "off log path > 2 m: unknown", 7: "tie: unknown", 8: "not alive"}


def box(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]])
    return np.stack([x + loc[:, 0] * c - loc[:, 1] * s, y + loc[:, 0] * s + loc[:, 1] * c], 1)


def draw_map(ax, pts, off, kinds):
    for kind, (color, lw, ls) in MAP_STYLE.items():
        segs = [pts[off[i]:off[i + 1], :2] for i in np.flatnonzero(kinds == kind) if off[i + 1] - off[i] > 1]
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, linestyles=ls, zorder=1))
    for i in np.flatnonzero(kinds == 5):
        ax.add_patch(Polygon(pts[off[i]:off[i + 1], :2], closed=True, facecolor="#e6e6e6", edgecolor="#b0b0b0",
                             lw=0.5, hatch="//", zorder=0.5))
    for i in np.flatnonzero(kinds == 4):
        ax.plot(pts[off[i], 0], pts[off[i], 1], marker="8", color="firebrick", ms=6, zorder=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--modes", default="unknown,oracle,reversed")
    ap.add_argument("--center", default="4,0")
    ap.add_argument("--half", type=float, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = Path(a.dir)
    v = np.load(d / "view_render.npz")
    S = json.load(open(d / "summary.json"))
    ids, hist, L, W = v["ids"], v["hist"], v["L"], v["W"]
    ego, oth = int(v["ego"]), int(v["other"])
    pair_idx = next(n for n, r in enumerate(S["labelled_pairs"]) if {r["id_i"], r["id_j"]} == {int(ids[ego]), int(ids[oth])})
    modes = a.modes.split(",")
    panels = []
    for m in modes:
        z = np.load(d / f"{m}.npz")
        fut = z["poses"][a.k]                                             # (N, 80, 3)
        full = np.concatenate([hist[:, :, [0, 1, 6]], fut], 1)            # (N, 91, 3)
        full[:, :11][hist[:, :, 5] < 0.5] = np.nan
        full[~v["alive"]] = np.nan
        R = S["results"][m]
        hit = bool(z["hit"][a.k])
        outcome = (f"collides with {S['other']} at {R and (np.argmax(z['min_gap'] < 0) if False else '')}" if False else
                   ("ego collides with %d" % S["other"]) if hit else
                   {"other_first": f"{S['other']} passes first", "ego_first": "ego passes first",
                    "no_cross": "no crossing within 8 s"}.get(str(z["order"][a.k]), str(z["order"][a.k])))
        scene_ok = "no collision anywhere in the scene" if not bool(z["scene_hit"][a.k]) else "some collision in the scene"
        title = (f"{LABEL[m]}\nrollout #{a.k}: {outcome}; {scene_ok}\n"
                 f"all {S['K']} rollouts: ego-{S['other']} collision {R['collision_2310']:.0%}, "
                 f"any collision in scene {R['scene_any_collision']:.0%}")
        codes = z["codes"][:, a.k, pair_idx] if "codes" in z.files else None
        panels.append(dict(full=full, hitmask=z["agent_hit"][a.k], title=title, codes=codes))

    cx, cy = (float(t) for t in a.center.split(","))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 8.0), dpi=90, squeeze=False)
    axes = axes[0]
    sep = "\n" if len(panels) == 1 else "  "
    fig.suptitle(f"{S['view']}  85k stage-1 WM, paired rollout #{a.k} (same noise seed in every panel).{sep}"
                 f"red = ego {ids[ego]}, blue = {ids[oth]}; magenta edge = proxy box overlap (4.8 x 2.0 m)", fontsize=10)
    for ax, p in zip(axes, panels):
        ax.set_xlim(cx - a.half, cx + a.half); ax.set_ylim(cy - a.half, cy + a.half); ax.set_aspect("equal")
        draw_map(ax, v["map_points"], v["map_offsets"], v["map_kind"]); ax.tick_params(labelsize=7)
        ax.set_title(p["title"], fontsize=9, loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tag = "_".join(modes)
    out = Path(a.out or d / "videos" / f"k{a.k}_{tag}.mp4"); out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=10, bitrate=4000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    dyn = []
    with writer.saving(fig, str(out), dpi=90):
        for t in range(91):
            for x in dyn:
                x.remove()
            dyn = []
            for ax, p in zip(axes, panels):
                f = p["full"]
                hitnow = p["hitmask"][t - 11] if t > 10 else np.zeros(len(ids), bool)
                for j in range(len(ids)):
                    col, zo = ("#d62728", 8) if j == ego else ("#1f77b4", 7) if j == oth else ("#9a9a9a", 5)
                    if j in (ego, oth):
                        dyn += ax.plot(f[j, :t + 1, 0], f[j, :t + 1, 1], color=col, lw=1.6, zorder=zo - 1)
                    if np.isnan(f[j, t, 0]):
                        continue
                    edge = "#ff00ff" if hitnow[j] else "k"
                    dyn.append(ax.add_patch(Polygon(box(f[j, t, 0], f[j, t, 1], f[j, t, 2], L[j], W[j]), closed=True,
                                                    facecolor=col, edgecolor=edge, lw=2.5 if hitnow[j] else 0.6,
                                                    alpha=0.9, zorder=zo)))
                lbl = f"t = {(t - 10) / 10:+.1f} s" + ("  (log history)" if t < 10 else "") + \
                      ("   COLLISION" if hitnow[[ego, oth]].any() else "")
                if p["codes"] is not None and t >= 10:
                    r = min((t - 10) // 5, len(p["codes"]) - 1)
                    lbl += f"\npair z: {CODE[int(p['codes'][r])]}"
                dyn.append(ax.text(0.02, 0.97, lbl, transform=ax.transAxes, fontsize=10, va="top",
                                   color="#ff00ff" if hitnow[[ego, oth]].any() else "k", weight="bold", zorder=20,
                                   bbox=dict(facecolor="white", alpha=0.8, lw=0)))
            writer.grab_frame()
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
