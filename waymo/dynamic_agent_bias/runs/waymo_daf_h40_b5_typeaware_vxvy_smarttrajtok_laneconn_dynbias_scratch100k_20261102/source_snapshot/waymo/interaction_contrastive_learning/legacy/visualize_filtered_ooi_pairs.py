"""Visualize original two-OOI pairs rejected by the soft-pair filter.

The audit reproduces the v1 screening rules, assigns every rejected original
OOI pair to one exclusive reason, and renders deterministic metric-quantile
examples for each reason.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from .pair_samples import ensure_agent_time_layout
    from .soft_pair_samples import SoftPairConfig
except ImportError:
    from pair_samples import ensure_agent_time_layout  # type: ignore
    from soft_pair_samples import SoftPairConfig  # type: ignore


COLORS = ("#1f77b4", "#ff7f0e")
REASON_LABELS = {
    "pet_over_30_only": "Close in space, but PET > 3 s",
    "distance_over_6_even_without_pet_limit": "Closest path distance > 6 m",
    "no_full_60_step_shared_window": "No complete shared 60-step window",
}


@dataclass
class SceneRow:
    split: str
    scenario_id: str
    source_path: str
    ooi_ids: tuple[int, int]


@dataclass
class RejectedPair:
    split: str
    scenario_id: str
    source_path: str
    ooi_a_id: int
    ooi_b_id: int
    reason: str
    severity: float
    diagnostic_step_a: int
    diagnostic_step_b: int
    diagnostic_distance_m: float
    diagnostic_pet_steps: int
    constrained_step_a: int
    constrained_step_b: int
    constrained_distance_m: float
    longest_joint_valid_steps: int


def _read_scenes(manifest_path: Path) -> dict[tuple[str, str], SceneRow]:
    scenes: dict[tuple[str, str], SceneRow] = {}
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ids = tuple(int(value) for value in row["ooi_track_ids"].split(";") if value)
            if len(ids) != 2:
                continue
            key = (row["split"], row["scenario_id"])
            if key not in scenes:
                scenes[key] = SceneRow(
                    split=row["split"],
                    scenario_id=row["scenario_id"],
                    source_path=row["npz_path"],
                    ooi_ids=(ids[0], ids[1]),
                )
    return scenes


def _retained_ooi_scenes(
    scenes: dict[tuple[str, str], SceneRow], samples_dir: Path
) -> set[tuple[str, str]]:
    retained: set[tuple[str, str]] = set()
    for split in ("train", "val"):
        path = samples_dir / f"{split}_samples.csv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (split, row["scenario_id"])
                scene = scenes.get(key)
                if scene is None:
                    continue
                pair_ids = {int(row["first_agent_id"]), int(row["second_agent_id"])}
                if pair_ids == set(scene.ooi_ids):
                    retained.add(key)
    return retained


def _first_window_mask(agent_a: np.ndarray, agent_b: np.ndarray, cfg: SoftPairConfig) -> np.ndarray:
    t = len(agent_a)
    valid_a = agent_a[:, 5] > 0.5
    valid_b = agent_b[:, 5] > 0.5
    result = np.zeros(t, dtype=bool)
    for first_step in range(cfg.history_steps - 1, t - cfg.post_first_steps):
        start = first_step - cfg.history_steps + 1
        end = first_step + cfg.post_first_steps + 1
        result[first_step] = bool(valid_a[start:end].all() and valid_b[start:end].all())
    return result


def _closest_points(
    agent_a: np.ndarray,
    agent_b: np.ndarray,
    cfg: SoftPairConfig,
    *,
    require_full_window: bool,
    pet_limit_steps: int | None,
) -> tuple[int, int, float] | None:
    t = len(agent_a)
    valid_a = agent_a[:, 5] > 0.5
    valid_b = agent_b[:, 5] > 0.5
    step_a = np.arange(t)[:, None]
    step_b = np.arange(t)[None, :]
    admissible = (
        valid_a[:, None]
        & valid_b[None, :]
        & (step_a >= cfg.event_search_start)
        & (step_b >= cfg.event_search_start)
    )
    if pet_limit_steps is not None:
        admissible &= np.abs(step_a - step_b) <= int(pet_limit_steps)
    if require_full_window:
        first_ok = _first_window_mask(agent_a, agent_b, cfg)
        admissible &= first_ok[np.minimum(step_a, step_b)]
    if not bool(admissible.any()):
        return None
    distance_sq = np.sum((agent_a[:, None, 0:2] - agent_b[None, :, 0:2]) ** 2, axis=-1)
    distance_sq[~admissible] = np.inf
    flat = int(np.argmin(distance_sq))
    a, b = np.unravel_index(flat, distance_sq.shape)
    return int(a), int(b), float(np.sqrt(distance_sq[a, b]))


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def _load_ooi_agents(scene: SceneRow) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with np.load(scene.source_path, allow_pickle=False) as data:
        mask = np.asarray(data["agent_mask"], dtype=bool)
        agents = ensure_agent_time_layout(np.asarray(data["agents"], dtype=np.float32), mask)
        ids = np.asarray(data["agent_ids"], dtype=np.int64)
        indices = []
        for ooi_id in scene.ooi_ids:
            found = np.flatnonzero(mask & (ids == ooi_id))
            if len(found) != 1:
                raise RuntimeError(f"{scene.scenario_id}: cannot locate OOI id {ooi_id}")
            indices.append(int(found[0]))
        loaded = {
            "agents": agents,
            "agent_mask": mask,
            "agent_ids": ids,
            "map_polylines": np.asarray(data["map_polylines"], dtype=np.float32),
            "map_mask": np.asarray(data["map_mask"], dtype=bool),
        }
    return loaded, agents[indices[0]], agents[indices[1]]


def classify_rejected(scene: SceneRow, cfg: SoftPairConfig) -> RejectedPair:
    _, agent_a, agent_b = _load_ooi_agents(scene)
    window_mask = _first_window_mask(agent_a, agent_b, cfg)
    joint_valid = (agent_a[:, 5] > 0.5) & (agent_b[:, 5] > 0.5)
    longest = _longest_true_run(joint_valid)
    constrained = _closest_points(
        agent_a, agent_b, cfg, require_full_window=True, pet_limit_steps=cfg.max_pet_steps
    )

    if not bool(window_mask.any()):
        reason = "no_full_60_step_shared_window"
        diagnostic = _closest_points(
            agent_a, agent_b, cfg, require_full_window=False, pet_limit_steps=cfg.max_pet_steps
        )
        if diagnostic is None:
            diagnostic = _closest_points(
                agent_a, agent_b, cfg, require_full_window=False, pet_limit_steps=None
            )
        severity = float(longest)
    else:
        if constrained is None:
            raise AssertionError(f"{scene.scenario_id}: full window exists but no constrained pair")
        if constrained[2] <= cfg.max_spatial_distance_m:
            raise AssertionError(f"{scene.scenario_id}: rejected pair would pass current filter")
        unconstrained = _closest_points(
            agent_a, agent_b, cfg, require_full_window=True, pet_limit_steps=None
        )
        if unconstrained is None:
            raise AssertionError(f"{scene.scenario_id}: no unconstrained pair")
        diagnostic = unconstrained
        if unconstrained[2] <= cfg.max_spatial_distance_m:
            reason = "pet_over_30_only"
            severity = float(abs(unconstrained[0] - unconstrained[1]))
        else:
            reason = "distance_over_6_even_without_pet_limit"
            severity = float(unconstrained[2])

    if diagnostic is None:
        diagnostic = (-1, -1, float("nan"))
    if constrained is None:
        constrained = (-1, -1, float("nan"))
    return RejectedPair(
        split=scene.split,
        scenario_id=scene.scenario_id,
        source_path=scene.source_path,
        ooi_a_id=scene.ooi_ids[0],
        ooi_b_id=scene.ooi_ids[1],
        reason=reason,
        severity=severity,
        diagnostic_step_a=diagnostic[0],
        diagnostic_step_b=diagnostic[1],
        diagnostic_distance_m=diagnostic[2],
        diagnostic_pet_steps=abs(diagnostic[0] - diagnostic[1]),
        constrained_step_a=constrained[0],
        constrained_step_b=constrained[1],
        constrained_distance_m=constrained[2],
        longest_joint_valid_steps=longest,
    )


def select_quantile_examples(records: list[RejectedPair], count: int) -> list[tuple[float, RejectedPair]]:
    ordered = sorted(records, key=lambda row: (row.severity, row.scenario_id))
    if count == 1:
        quantiles = [0.5]
    else:
        quantiles = np.linspace(0.1, 0.9, count).tolist()
    selected: list[tuple[float, RejectedPair]] = []
    used: set[int] = set()
    for quantile in quantiles:
        index = int(round(quantile * (len(ordered) - 1)))
        if index in used:
            for candidate in range(len(ordered)):
                if candidate not in used:
                    index = candidate
                    break
        used.add(index)
        selected.append((float(quantile), ordered[index]))
    return selected


def _transform(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(xy, dtype=np.float32) - origin
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return delta @ np.asarray([[c, s], [-s, c]], dtype=np.float32).T


def _plot_masked_trajectory(
    ax: plt.Axes, agent: np.ndarray, origin: np.ndarray, yaw: float, color: str, label: str
) -> np.ndarray:
    xy = _transform(agent[:, 0:2], origin, yaw)
    valid = agent[:, 5] > 0.5
    faded = xy.copy()
    faded[~valid] = np.nan
    ax.plot(faded[:, 0], faded[:, 1], color=color, linewidth=2.2, alpha=0.85, label=label, zorder=4)
    valid_steps = np.flatnonzero(valid)
    if len(valid_steps):
        ax.scatter(xy[valid_steps[0], 0], xy[valid_steps[0], 1], marker="o", s=30, color=color, zorder=5)
        ax.scatter(xy[valid_steps[-1], 0], xy[valid_steps[-1], 1], marker="X", s=38, color=color, zorder=5)
    return xy


def render_record(record: RejectedPair, cfg: SoftPairConfig, output_path: Path, quantile: float) -> None:
    scene = SceneRow(record.split, record.scenario_id, record.source_path, (record.ooi_a_id, record.ooi_b_id))
    loaded, agent_a, agent_b = _load_ooi_agents(scene)
    da, db = record.diagnostic_step_a, record.diagnostic_step_b
    if da >= 0 and db >= 0:
        conflict_midpoint = 0.5 * (agent_a[da, 0:2] + agent_b[db, 0:2])
        frame_yaw = float(agent_a[da, 6])
    else:
        conflict_midpoint = agent_a[10, 0:2]
        frame_yaw = float(agent_a[10, 6])

    fig = plt.figure(figsize=(9.5, 7.0), dpi=140, constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(5.0, 1.25))
    ax = fig.add_subplot(grid[0])
    timeline = fig.add_subplot(grid[1])

    for polyline, mask in zip(loaded["map_polylines"], loaded["map_mask"]):
        points = polyline[mask, 0:2]
        if len(points) >= 2:
            points = _transform(points, conflict_midpoint, frame_yaw)
            ax.plot(points[:, 0], points[:, 1], color="#c8c8c8", linewidth=0.55, alpha=0.65, zorder=1)

    xy_a = _plot_masked_trajectory(
        ax, agent_a, conflict_midpoint, frame_yaw, COLORS[0], f"OOI-A id={record.ooi_a_id}"
    )
    xy_b = _plot_masked_trajectory(
        ax, agent_b, conflict_midpoint, frame_yaw, COLORS[1], f"OOI-B id={record.ooi_b_id}"
    )

    if da >= 0 and db >= 0:
        for xy, step, color, name in ((xy_a, da, COLORS[0], "A"), (xy_b, db, COLORS[1], "B")):
            ax.scatter(xy[step, 0], xy[step, 1], marker="*", s=145, color=color,
                       edgecolors="black", linewidths=0.7, zorder=8)
            ax.annotate(f"{name}: t={step * cfg.dt:.1f}s", xy[step], xytext=(5, 7),
                        textcoords="offset points", fontsize=8, color=color)
        ax.plot(
            [xy_a[da, 0], xy_b[db, 0]], [xy_a[da, 1], xy_b[db, 1]],
            color="#d62728", linestyle="--", linewidth=1.4, zorder=7,
            label="diagnostic closest-point pair",
        )

    ca, cb = record.constrained_step_a, record.constrained_step_b
    if ca >= 0 and cb >= 0 and (ca, cb) != (da, db):
        ax.scatter(xy_a[ca, 0], xy_a[ca, 1], facecolors="none", edgecolors="#6a3d9a", s=70,
                   linewidths=1.5, zorder=7)
        ax.scatter(xy_b[cb, 0], xy_b[cb, 1], facecolors="none", edgecolors="#6a3d9a", s=70,
                   linewidths=1.5, zorder=7, label="best point pair under PET <= 3 s")

    all_valid_xy = np.concatenate((xy_a[agent_a[:, 5] > 0.5], xy_b[agent_b[:, 5] > 0.5]), axis=0)
    extent = float(np.quantile(np.abs(all_valid_xy), 0.98)) if len(all_valid_xy) else 30.0
    radius = min(90.0, max(20.0, extent * 1.15))
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.axhline(0, color="#777", linewidth=0.45, alpha=0.5)
    ax.axvline(0, color="#777", linewidth=0.45, alpha=0.5)
    ax.set_xlabel("longitudinal position in diagnostic frame (m)")
    ax.set_ylabel("lateral position in diagnostic frame (m)")
    ax.legend(loc="best", fontsize=8)

    valid_a = agent_a[:, 5] > 0.5
    valid_b = agent_b[:, 5] > 0.5
    times = np.arange(len(agent_a)) * cfg.dt
    timeline.fill_between(times, 1.65, 2.35, where=valid_a, step="mid", color=COLORS[0], alpha=0.85)
    timeline.fill_between(times, 0.65, 1.35, where=valid_b, step="mid", color=COLORS[1], alpha=0.85)
    timeline.fill_between(times, -0.35, 0.35, where=valid_a & valid_b, step="mid", color="#2ca02c", alpha=0.75)
    timeline.set_yticks((2, 1, 0), ("OOI-A valid", "OOI-B valid", "joint valid"))
    timeline.set_xlim(0, (len(agent_a) - 1) * cfg.dt)
    timeline.set_ylim(-0.55, 2.55)
    timeline.set_xlabel("scenario time (s)")
    timeline.grid(axis="x", alpha=0.25)
    if da >= 0:
        timeline.axvline(da * cfg.dt, color=COLORS[0], linestyle="--", linewidth=1.1)
    if db >= 0:
        timeline.axvline(db * cfg.dt, color=COLORS[1], linestyle="--", linewidth=1.1)

    if record.reason == "pet_over_30_only":
        details = (
            f"diagnostic distance={record.diagnostic_distance_m:.2f} m <= 6 m, "
            f"PET={record.diagnostic_pet_steps * cfg.dt:.1f} s > 3 s; "
            f"best distance with PET <= 3 s: {record.constrained_distance_m:.2f} m"
        )
    elif record.reason == "distance_over_6_even_without_pet_limit":
        details = (
            f"closest distance without PET limit={record.diagnostic_distance_m:.2f} m > 6 m; "
            f"PET at that pair={record.diagnostic_pet_steps * cfg.dt:.1f} s; "
            f"best distance with PET <= 3 s: {record.constrained_distance_m:.2f} m"
        )
    else:
        details = (
            f"longest jointly valid run={record.longest_joint_valid_steps} steps < 60; "
            f"relaxed closest distance={record.diagnostic_distance_m:.2f} m, "
            f"PET={record.diagnostic_pet_steps * cfg.dt:.1f} s"
        )
    fig.suptitle(
        f"Filtered original OOI pair: {REASON_LABELS[record.reason]}\n"
        f"{record.split} | scenario={record.scenario_id} | selection quantile={quantile:.0%}\n{details}",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_html(path: Path, selected: list[dict[str, object]]) -> None:
    sections = []
    for reason in REASON_LABELS:
        rows = [row for row in selected if row["reason"] == reason]
        tiles = []
        for row in rows:
            figure = html.escape(str(row["figure"]))
            tiles.append(
                "<article><a href='" + figure + "'><img src='" + figure + "'></a>"
                f"<h3>{html.escape(str(row['scenario_id']))}</h3>"
                f"<p>{html.escape(str(row['split']))}; selection quantile={float(row['selection_quantile']):.0%}</p>"
                f"<p>d={float(row['diagnostic_distance_m']):.2f} m; "
                f"PET={float(row['diagnostic_pet_steps']) * 0.1:.1f} s; "
                f"longest joint validity={int(row['longest_joint_valid_steps'])} steps</p></article>"
            )
        sections.append(f"<section><h2>{html.escape(REASON_LABELS[reason])}</h2><div>{''.join(tiles)}</div></section>")
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Filtered OOI pair audit</title>"
        "<style>body{font-family:system-ui;margin:24px;background:#f5f5f5;color:#222}"
        "section{background:white;padding:18px;margin:20px 0;border-radius:8px}"
        "section>div{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:14px}"
        "article{border:1px solid #ddd;padding:8px;border-radius:6px}img{width:100%;height:auto}"
        "h3,p{margin:6px 0}p{font-size:13px;color:#555}</style></head><body>"
        "<h1>Original OOI pairs rejected by the v1 soft-pair filter</h1>"
        "<p>Three deterministic metric-quantile examples per exclusive rejection reason.</p>"
        + "".join(sections)
        + "</body></html>"
    )


def build(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = SoftPairConfig()
    scenes = _read_scenes(data_root / "manifest.csv")
    retained = _retained_ooi_scenes(scenes, samples_dir)
    rejected_scenes = [scene for key, scene in scenes.items() if key not in retained]
    print(f"two-OOI scenes={len(scenes)} retained={len(retained)} rejected={len(rejected_scenes)}", flush=True)

    rejected: list[RejectedPair] = []
    for index, scene in enumerate(rejected_scenes, start=1):
        rejected.append(classify_rejected(scene, cfg))
        if index % 500 == 0:
            print(f"classified {index}/{len(rejected_scenes)}", flush=True)

    selected_rows: list[dict[str, object]] = []
    for reason in REASON_LABELS:
        group = [row for row in rejected if row.reason == reason]
        for rank, (quantile, record) in enumerate(select_quantile_examples(group, args.examples_per_reason), start=1):
            filename = f"{reason}_{rank}_q{int(round(quantile * 100)):02d}_{record.split}_{record.scenario_id}.png"
            relative = Path(reason) / filename
            print(f"rendering {relative}", flush=True)
            render_record(record, cfg, output_dir / relative, quantile)
            row = asdict(record)
            row["selection_quantile"] = quantile
            row["figure"] = relative.as_posix()
            selected_rows.append(row)

    with (output_dir / "selected_examples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    (output_dir / "selected_examples.json").write_text(json.dumps(selected_rows, indent=2))
    reason_counts = {reason: sum(row.reason == reason for row in rejected) for reason in REASON_LABELS}
    (output_dir / "summary.json").write_text(json.dumps({
        "two_ooi_scenes": len(scenes),
        "retained_ooi_pairs": len(retained),
        "rejected_ooi_pairs": len(rejected),
        "rejection_reason_counts": reason_counts,
        "selection": "10th, 50th, and 90th reason-specific metric quantiles",
        "config": asdict(cfg),
    }, indent=2))
    _write_html(output_dir / "index.html", selected_rows)
    print(f"saved {output_dir / 'index.html'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default="/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_ooi_centered_50k",
    )
    parser.add_argument(
        "--samples_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_soft_pairs_50k_v1",
    )
    parser.add_argument(
        "--output_dir",
        default="/p/yufeng/tri30/dreamer4/waymo/cache/interaction_soft_pairs_50k_v1/filtered_ooi_visual_audit",
    )
    parser.add_argument("--examples_per_reason", type=int, default=3)
    return parser


def main() -> None:
    build(build_argparser().parse_args())


if __name__ == "__main__":
    main()
