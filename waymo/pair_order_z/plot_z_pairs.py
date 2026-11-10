#!/usr/bin/env python3
"""Draw N random labelled pairs (plan §8.2 visual check). numpy + matplotlib only.

Usage: <python with matplotlib> plot_z_pairs.py --labels data/pair_order_z_labels_v1/val.npz \
         --data_dir data/waymo_vector_dataset_ooi_centered_50k_with_lengths/val --out <png>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    d = np.load(a.labels)
    names, offsets, pairs = d["names"], d["offsets"], d["pairs"]
    owner = np.repeat(np.arange(len(names)), np.diff(offsets))
    pick = np.random.default_rng(a.seed).choice(len(pairs), size=min(a.n, len(pairs)), replace=False)
    cols = 5
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    for ax, k in zip(np.ravel(axes), pick):
        i, j, ei, xi, ej, xj, ang = (int(v) for v in pairs[k])
        s = np.load(Path(a.data_dir) / str(names[owner[k]]))
        ag = s["agents"]
        for n in np.flatnonzero(s["agent_mask"]):
            v = ag[n, :, 5] > 0.5
            ax.plot(ag[n, v, 0], ag[n, v, 1], color="0.85", lw=0.6)
        for idx, (e, x, c) in ((i, (ei, xi, "tab:blue")), (j, (ej, xj, "tab:red"))):
            v = ag[idx, :, 5] > 0.5
            ax.plot(ag[idx, v, 0], ag[idx, v, 1], color=c, lw=1.2)
            ax.plot(ag[idx, e:x + 1, 0], ag[idx, e:x + 1, 1], color=c, lw=4, alpha=0.5)
            ax.scatter(*ag[idx, e, :2], color=c, marker=">", s=30, zorder=3)
            ax.scatter(*ag[idx, 10, :2], color=c, marker="o", s=25, zorder=3, edgecolor="k")
        first = "blue" if ei < ej else ("red" if ej < ei else "tie")
        ax.set_title(f"{str(names[owner[k]])[:16]} i{i}/j{j} {ang}deg\n"
                     f"blue enter {ei}-{xi}  red enter {ej}-{xj}  first={first}", fontsize=8)
        cx, cy = ag[i, ei, :2]
        ax.set_xlim(cx - 40, cx + 40)
        ax.set_ylim(cy - 40, cy + 40)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    for ax in np.ravel(axes)[len(pick):]:
        ax.axis("off")
    fig.suptitle("thick = overlap segment, > = enter frame, o = position at t=10 (eval start)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
