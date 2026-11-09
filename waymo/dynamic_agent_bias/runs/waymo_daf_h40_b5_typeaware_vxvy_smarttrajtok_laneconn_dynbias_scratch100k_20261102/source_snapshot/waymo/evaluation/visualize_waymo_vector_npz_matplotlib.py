"""Matplotlib visualization for filtered Waymo vector NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


CURRENT_IDX = 10


def _as_str(value) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        return str(value.item())
    return str(value)


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _valid_xy_bounds(item: Dict[str, np.ndarray], margin_m: float) -> Tuple[float, float, float, float]:
    points = []
    agents = item["agents"]
    agent_valid = agents[:, :, 5] > 0.5
    if agent_valid.any():
        points.append(agents[:, :, 0:2][agent_valid])

    map_mask = item["map_mask"].astype(bool)
    if map_mask.any():
        points.append(item["map_polylines"][:, :, 0:2][map_mask])

    light_mask = item["light_mask"].astype(bool)
    if light_mask.any():
        points.append(item["lights"][:, :, 0:2][light_mask])

    if not points:
        return -50.0, 50.0, -50.0, 50.0
    xy = np.concatenate(points, axis=0)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return -50.0, 50.0, -50.0, 50.0
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    return float(xmin - margin_m), float(xmax + margin_m), float(ymin - margin_m), float(ymax + margin_m)


def _agent_color(agent_type: int, is_focus: bool) -> str:
    if is_focus:
        return "#2ca02c"
    return {
        1: "#1f77b4",
        2: "#d627b0",
        3: "#ffbf00",
    }.get(int(agent_type), "#aaaaaa")


def _light_color(state: int) -> str:
    if int(state) in (1, 4, 7):
        return "#d62728"
    if int(state) in (2, 5, 8):
        return "#ffbf00"
    if int(state) in (3, 6):
        return "#2ca02c"
    return "#999999"


def _draw_map(ax, item: Dict[str, np.ndarray]) -> None:
    for poly, mask in zip(item["map_polylines"], item["map_mask"].astype(bool)):
        pts = poly[mask, 0:2]
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color="#6a6a6a", linewidth=0.45, alpha=0.75, zorder=1)


def _draw_lights(ax, item: Dict[str, np.ndarray], t: int) -> None:
    lights = item["lights"][t]
    mask = item["light_mask"][t].astype(bool)
    for light in lights[mask]:
        ax.scatter(
            light[0],
            light[1],
            s=28,
            color=_light_color(int(light[2])),
            edgecolors="black",
            linewidths=0.5,
            zorder=4,
        )


def _draw_agents(ax, item: Dict[str, np.ndarray], t: int, trail: int) -> None:
    agents = item["agents"]
    agent_mask = item["agent_mask"].astype(bool)
    agent_ids = item.get("agent_ids", np.arange(len(agent_mask)))
    ooi_mask = item.get("agent_objects_of_interest", np.zeros_like(agent_mask)).astype(bool)
    ttp_mask = item.get("agent_tracks_to_predict", np.zeros_like(agent_mask)).astype(bool)

    for k, agent in enumerate(agents):
        if not agent_mask[k] or agent[t, 5] <= 0.5:
            continue
        color = _agent_color(int(round(float(agent[t, 7]))), is_focus=(k == 0))

        if trail > 0:
            hist = agent[max(0, t - trail) : t + 1]
            hist = hist[hist[:, 5] > 0.5, 0:2]
            if len(hist) >= 2:
                ax.plot(hist[:, 0], hist[:, 1], color=color, linewidth=1.2, alpha=0.65, zorder=3)

        x, y = float(agent[t, 0]), float(agent[t, 1])
        yaw = float(agent[t, 6])
        ax.scatter(x, y, s=52, color=color, edgecolors="black", linewidths=0.7, zorder=5)
        ax.arrow(
            x,
            y,
            3.5 * np.cos(yaw),
            3.5 * np.sin(yaw),
            color=color,
            width=0.12,
            head_width=1.0,
            head_length=1.2,
            length_includes_head=True,
            zorder=6,
        )

        if ooi_mask[k]:
            ax.scatter(x, y, s=150, facecolors="none", edgecolors="#ff8c00", linewidths=2.0, zorder=7)
        if ttp_mask[k]:
            ax.scatter(x, y, s=205, facecolors="none", edgecolors="white", linewidths=1.5, zorder=8)

        label = "focus" if k == 0 else str(int(agent_ids[k]))
        if ooi_mask[k]:
            label += ":OOI"
        if ttp_mask[k]:
            label += ":TTP"
        ax.text(x + 1.2, y + 1.2, label, fontsize=7, color="white", zorder=9)


def _draw_frame(ax, item: Dict[str, np.ndarray], t: int, bounds, trail: int) -> None:
    ax.clear()
    ax.set_facecolor("#202020")
    _draw_map(ax, item)
    _draw_lights(ax, item, t)
    _draw_agents(ax, item, t, trail=trail)
    ax.scatter(0.0, 0.0, marker="+", s=100, color="white", linewidths=1.4, zorder=10)
    ax.arrow(0.0, 0.0, 8.0, 0.0, color="white", width=0.08, head_width=0.8, head_length=1.0, zorder=10)
    ax.arrow(0.0, 0.0, 0.0, 8.0, color="#dddddd", width=0.08, head_width=0.8, head_length=1.0, zorder=10)
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#444444", alpha=0.35, linewidth=0.5)
    scenario_id = _as_str(item.get("scenario_id", "unknown"))
    sample_label = _as_str(item.get("sample_label", ""))
    stats_labels = _as_str(item.get("stats_interaction_labels", ""))
    title = f"{scenario_id}  t={t:02d} ({(t - CURRENT_IDX) * 0.1:+.1f}s)"
    if sample_label:
        title += f"  sample={sample_label}"
    ax.set_title(title, color="white", fontsize=10)
    ax.text(
        0.01,
        0.01,
        f"orange=OOI, white=TTP, green=focus\n{stats_labels}",
        transform=ax.transAxes,
        fontsize=8,
        color="#eeeeee",
        va="bottom",
        bbox={"facecolor": "#111111", "alpha": 0.7, "edgecolor": "none", "pad": 4},
    )
    ax.tick_params(colors="#dddddd", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#888888")


def render_video(
    npz_path: str,
    output_path: str,
    fps: int = 10,
    trail: int = 15,
    margin_m: float = 15.0,
    start: int = 0,
    end: int | None = None,
    dpi: int = 130,
    preview_png: str | None = None,
    preview_frames: int = 16,
    preview_cols: int = 4,
) -> None:
    item = _load_npz(npz_path)
    num_steps = int(item["agents"].shape[1])
    start = max(0, int(start))
    end = num_steps if end is None else min(num_steps, int(end))
    if start >= end:
        raise ValueError(f"Invalid frame range start={start}, end={end}")
    bounds = _valid_xy_bounds(item, margin_m=margin_m)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    fig.patch.set_facecolor("#111111")

    def update(t: int):
        _draw_frame(ax, item, t, bounds, trail)
        return []

    ani = animation.FuncAnimation(fig, update, frames=list(range(start, end)), interval=1000 / max(1, fps), blit=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ani.save(output_path, writer="ffmpeg", fps=fps, dpi=dpi)

    if preview_png is not None:
        ts = np.linspace(start, end - 1, num=min(preview_frames, end - start), dtype=np.int64).tolist()
        cols = max(1, int(preview_cols))
        rows = int(np.ceil(len(ts) / cols))
        sheet_fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=dpi)
        sheet_fig.patch.set_facecolor("#111111")
        axes_arr = np.asarray(axes).reshape(-1)
        for axis in axes_arr:
            axis.set_visible(False)
        for axis, t in zip(axes_arr, ts):
            axis.set_visible(True)
            _draw_frame(axis, item, int(t), bounds, trail)
            axis.set_title(f"t={int(t):02d}", color="white", fontsize=9)
        Path(preview_png).parent.mkdir(parents=True, exist_ok=True)
        sheet_fig.tight_layout()
        sheet_fig.savefig(preview_png, facecolor=sheet_fig.get_facecolor(), bbox_inches="tight")
        plt.close(sheet_fig)

    plt.close(fig)


def _default_output_path(npz_path: str) -> str:
    path = Path(npz_path)
    return str(path.with_name(f"{path.stem}_mpl.mp4"))


def _default_preview_path(output_path: str) -> str:
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_preview.png"))


def main() -> None:
    p = argparse.ArgumentParser(description="Render a Waymo vector NPZ with matplotlib.")
    p.add_argument("npz", type=str)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--preview_png", type=str, default=None)
    p.add_argument("--no_preview_png", action="store_true")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--trail", type=int, default=15)
    p.add_argument("--margin_m", type=float, default=15.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--preview_frames", type=int, default=16)
    p.add_argument("--preview_cols", type=int, default=4)
    args = p.parse_args()

    output = args.output or _default_output_path(args.npz)
    preview_png = None if args.no_preview_png else (args.preview_png or _default_preview_path(output))
    render_video(
        npz_path=args.npz,
        output_path=output,
        fps=args.fps,
        trail=args.trail,
        margin_m=args.margin_m,
        start=args.start,
        end=args.end,
        dpi=args.dpi,
        preview_png=preview_png,
        preview_frames=args.preview_frames,
        preview_cols=args.preview_cols,
    )
    print(output)
    if preview_png is not None:
        print(preview_png)


if __name__ == "__main__":
    main()
