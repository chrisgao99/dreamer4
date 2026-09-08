#!/usr/bin/env python3
"""Plot logged H15 sample ADE for the generate-all, from-scratch experiment."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_LOG = REPO_ROOT / (
    "waymo/logs/wm/waymo_direct_action_flow_v1_h15_b5_generateall_scratch100k_"
    "phys5_yaw075_huber_bounded_tmax090_lr5e5.log"
)
OUTPUT_ROOT = REPO_ROOT / (
    "waymo/eval_results/world_model/direct_action_flow_h15_generateall_training_curve"
)


def main() -> None:
    points = []
    for line in SOURCE_LOG.read_text().splitlines():
        if not line.startswith("validation "):
            continue
        values = dict(re.findall(r"(\w+)=([0-9.eE+\-]+)", line))
        if "sample_mean_ade_m" not in values:
            continue
        points.append({
            "step": int(values["step"]),
            "sample_mean_ade_m": float(values["sample_mean_ade_m"]),
            "sample_minade_m": float(values["sample_minade_m"]),
            "num_rollouts": int(values["sample_num_rollouts"]),
            "metric_scope": "all_valid_agents_including_focus",
            "source_log": str(SOURCE_LOG),
        })
    assert [p["step"] for p in points] == list(range(10000, 100001, 10000))
    assert all(p["num_rollouts"] == 4 for p in points)
    assert all(0 <= p["sample_minade_m"] <= p["sample_mean_ade_m"] for p in points)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_ROOT / "h15_generateall_ade_training_curve"
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3),
                             gridspec_kw={"width_ratios": (1.06, 1.35)})
    steps = [p["step"] / 1000 for p in points]
    series = (
        ("sample_mean_ade_m", "H15 meanADE", "#d1495b"),
        ("sample_minade_m", "H15 minADE@4", "#2f6fbb"),
    )
    for axis in axes:
        axis.axvspan(10, 100, color="#54a24b", alpha=0.055)
        for key, label, color in series:
            axis.plot(steps, [p[key] for p in points], color=color, marker="o",
                      linewidth=2.2, markersize=5.8, label=label)
        axis.grid(True, alpha=0.24, linewidth=0.8)
        axis.set_xlabel("Training step (thousands)")
        axis.set_ylabel("ADE (m)")
    axes[0].set(xlim=(7, 103), ylim=(0, 2.5), title="From scratch: full training run")
    axes[1].set(xlim=(38, 102), ylim=(0.10, 0.25), title="Plateau after 40k (zoomed)")
    axes[0].text(60, 2.36, "All agents generated, including focus\n"
                 "H=15, B=5; Huber + physical filter\n"
                 "t_max=0.90, initial target lr=5e-5",
                 ha="center", va="top", fontsize=9, color="#444444")
    for point in points[3:]:
        for key, _, color in series:
            offset = 7 if key == "sample_mean_ade_m" else -13
            axes[1].annotate(f"{point[key]:.3f}", (point["step"] / 1000, point[key]),
                             xytext=(0, offset), textcoords="offset points",
                             ha="center", fontsize=7.5, color=color)
    fig.suptitle("DirectActionFlow (Generate All): 15-step Open-loop ADE During Training",
                 fontsize=15, fontweight="bold", y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=2, frameon=True)
    fig.text(0.5, -0.015,
             "EMA validation uses 4 stochastic joint rollouts and includes the focus agent. "
             "No focus future actions are provided; configuration is unchanged throughout.",
             ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.875))
    for extension in (".png", ".pdf"):
        path = stem.with_suffix(extension)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(path)
    plt.close(fig)
    print(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
