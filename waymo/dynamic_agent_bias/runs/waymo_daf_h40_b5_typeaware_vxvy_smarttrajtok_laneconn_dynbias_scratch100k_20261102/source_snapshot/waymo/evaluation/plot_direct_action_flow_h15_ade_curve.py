#!/usr/bin/env python3
"""Plot H15 open-loop sample ADE along the 90k checkpoint's training lineage."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "waymo/logs/wm"
OUTPUT_ROOT = (
    REPO_ROOT
    / "waymo/eval_results/world_model/direct_action_flow_h15_training_curve"
)

PHASES = (
    (
        "baseline",
        LOG_ROOT
        / "waymo_direct_action_flow_v1_explicitagent_h15_b5_d256_lr2e4_control_60k.log",
        0,
        45_000,
    ),
    (
        "tmax095",
        LOG_ROOT / "waymo_direct_action_flow_v1_h15_resume45k_tmax095.log",
        45_000,
        62_500,
    ),
    (
        "robust_v2",
        LOG_ROOT
        / "waymo_direct_action_flow_v1_h15_phys5_yaw075_huber_bounded_tmax090_lr5e5_from62500_v2.log",
        62_500,
        100_000,
    ),
)

VALIDATION = re.compile(
    r"validation step=(?P<step>\d+).*?"
    r"sample_mean_ade_m=(?P<mean>[0-9.eE+-]+).*?"
    r"sample_minade_m=(?P<min>[0-9.eE+-]+).*?"
    r"sample_num_rollouts=(?P<rollouts>[0-9.eE+-]+)"
)


def load_points() -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for phase, path, lower_exclusive, upper_inclusive in PHASES:
        if not path.is_file():
            raise FileNotFoundError(path)
        for match in VALIDATION.finditer(path.read_text(errors="replace")):
            step = int(match.group("step"))
            if not lower_exclusive < step <= upper_inclusive:
                continue
            points.append(
                {
                    "step": step,
                    "sample_mean_ade_m": float(match.group("mean")),
                    "sample_minade_m": float(match.group("min")),
                    "num_rollouts": int(float(match.group("rollouts"))),
                    "phase": phase,
                    "source_log": str(path),
                }
            )
    points.sort(key=lambda item: int(item["step"]))
    expected_steps = list(range(10_000, 100_001, 10_000))
    actual_steps = [int(item["step"]) for item in points]
    if actual_steps != expected_steps:
        raise RuntimeError(f"Expected steps {expected_steps}, found {actual_steps}")
    return points


def decorate_axis(axis: plt.Axes, *, zoom: bool) -> None:
    axis.axvspan(10, 45, color="#4c78a8", alpha=0.055)
    axis.axvspan(45, 62.5, color="#f58518", alpha=0.065)
    axis.axvspan(62.5, 100, color="#54a24b", alpha=0.055)
    axis.axvline(45, color="#6b6b6b", linestyle="--", linewidth=1.1)
    axis.axvline(62.5, color="#6b6b6b", linestyle="--", linewidth=1.1)
    axis.grid(True, alpha=0.24, linewidth=0.8)
    axis.set_xlabel("Training step (thousands)")
    axis.set_ylabel("ADE (m)")
    if zoom:
        axis.set_xlim(38, 102)
        axis.set_ylim(0.10, 0.34)
    else:
        axis.set_xlim(7, 103)
        axis.set_ylim(0.0, 2.08)


def main() -> None:
    points = load_points()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "h15_ade_training_curve.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(points[0].keys()))
        writer.writeheader()
        writer.writerows(points)

    steps = [int(item["step"]) / 1000.0 for item in points]
    mean_ade = [float(item["sample_mean_ade_m"]) for item in points]
    minade = [float(item["sample_minade_m"]) for item in points]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.3),
        gridspec_kw={"width_ratios": (1.06, 1.35)},
    )
    colors = {"mean": "#d1495b", "min": "#2f6fbb"}
    for axis in axes:
        axis.plot(
            steps,
            mean_ade,
            color=colors["mean"],
            marker="o",
            linewidth=2.2,
            markersize=5.8,
            label="H15 meanADE",
        )
        axis.plot(
            steps,
            minade,
            color=colors["min"],
            marker="o",
            linewidth=2.2,
            markersize=5.8,
            label="H15 minADE@4",
        )

    decorate_axis(axes[0], zoom=False)
    decorate_axis(axes[1], zoom=True)
    axes[0].set_title("Full training lineage")
    axes[1].set_title("Plateau after 40k (zoomed)")
    handles, labels = axes[0].get_legend_handles_labels()

    phase_y = 2.015
    axes[0].text(27.5, phase_y, "baseline", ha="center", va="top", fontsize=9)
    axes[0].text(53.75, phase_y, "t_max=0.95", ha="center", va="top", fontsize=9)
    axes[0].text(
        81.25,
        phase_y,
        "Huber + physical filter\n+t_max=0.90, lr=5e-5",
        ha="center",
        va="top",
        fontsize=8.5,
    )

    for step, mean_value, min_value in zip(steps[3:], mean_ade[3:], minade[3:]):
        axes[1].annotate(
            f"{mean_value:.3f}",
            (step, mean_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=colors["mean"],
        )
        axes[1].annotate(
            f"{min_value:.3f}",
            (step, min_value),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=colors["min"],
        )

    figure.suptitle(
        "DirectActionFlow: 15-step Open-loop ADE During Training",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=True,
    )
    figure.text(
        0.5,
        -0.015,
        "Validation uses 4 stochastic joint rollouts and excludes the conditioned focus agent. "
        "Dashed lines mark configuration changes.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.875))

    png_path = OUTPUT_ROOT / "h15_ade_training_curve.png"
    pdf_path = OUTPUT_ROOT / "h15_ade_training_curve.pdf"
    figure.savefig(png_path, dpi=180, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(png_path)
    print(pdf_path)
    print(csv_path)


if __name__ == "__main__":
    main()
