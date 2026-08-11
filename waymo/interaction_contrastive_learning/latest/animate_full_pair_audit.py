"""Animate selected full-pair audit trajectories on a shared time axis.

The animation is intentionally simpler than the static diagnostic figure: the
two colored paths grow from the first frame to the last frame, while a marker
shows each agent's current position.  For ``ooi_closest_fallback`` samples,
stars and a dashed connector show the two (possibly asynchronous) path points
used to define the fallback event.
"""

from __future__ import annotations

import argparse
import csv
import html
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


COLORS = ("#1f77b4", "#ff7f0e")
TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _interpolate_yaw(agent: np.ndarray, step: float) -> float:
    step = float(np.clip(step, 0.0, len(agent) - 1.0))
    low = int(np.floor(step))
    high = min(len(agent) - 1, low + 1)
    fraction = step - low
    delta = _wrap_angle(float(agent[high, 6]) - float(agent[low, 6]))
    return _wrap_angle(float(agent[low, 6]) + fraction * delta)


def _interpolate_xy(trajectory: np.ndarray, step: float) -> np.ndarray:
    step = float(np.clip(step, 0.0, len(trajectory) - 1.0))
    low = int(np.floor(step))
    high = min(len(trajectory) - 1, low + 1)
    fraction = step - low
    return ((1.0 - fraction) * trajectory[low, 0:2] + fraction * trajectory[high, 0:2]).astype(np.float32)


def _transform(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    delta = np.asarray(xy, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return delta @ np.asarray([[c, s], [-s, c]], dtype=np.float32).T


def _obb_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    local = np.asarray(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ],
        dtype=np.float32,
    )
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return local @ np.asarray([[c, s], [-s, c]], dtype=np.float32) + np.asarray([x, y])


def _load_row_data(row: dict[str, str], dataset_dir: Path) -> dict[str, np.ndarray]:
    shard_row = int(row["shard_row"])
    with np.load(dataset_dir / row["shard"], allow_pickle=False) as data:
        pair = {
            "trajectory": np.asarray(data["trajectory"][shard_row], dtype=np.float32),
            "valid_mask": np.asarray(data["valid_mask"][shard_row], dtype=bool),
            "size": np.asarray(data["agent_size_m"][shard_row], dtype=np.float32),
            "conflict_xy": np.asarray(data["conflict_xy"][shard_row], dtype=np.float32),
        }

    with np.load(row["source_path"], allow_pickle=False) as data:
        source = {
            "agents": np.asarray(data["agents"], dtype=np.float32),
            "agent_mask": np.asarray(data["agent_mask"], dtype=bool),
            "agent_ids": np.asarray(data["agent_ids"], dtype=np.int64),
            "map_polylines": np.asarray(data["map_polylines"], dtype=np.float32),
            "map_mask": np.asarray(data["map_mask"], dtype=bool),
        }

    first_id = int(row["first_agent_id"])
    found = np.flatnonzero(source["agent_mask"] & (source["agent_ids"] == first_id))
    if not len(found):
        raise RuntimeError(f"Cannot locate first agent id={first_id} in {row['source_path']}")
    frame_yaw = _interpolate_yaw(source["agents"][int(found[0])], float(row["primary_step_first"]))

    map_paths = []
    for polyline, mask in zip(source["map_polylines"], source["map_mask"]):
        points = polyline[mask, 0:2]
        if len(points) >= 2:
            map_paths.append(_transform(points, pair["conflict_xy"], frame_yaw))
    pair["map_paths"] = np.asarray(map_paths, dtype=object)
    return pair


def _square_bounds(trajectory: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float, float, float]:
    points = np.concatenate([trajectory[i, valid_mask[i], 0:2] for i in range(2)], axis=0)
    finite = points[np.isfinite(points).all(axis=1)]
    if not len(finite):
        return -40.0, 40.0, -40.0, 40.0
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    center = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 20.0)
    half = 0.5 * span + max(6.0, 0.08 * span)
    return center[0] - half, center[0] + half, center[1] - half, center[1] + half


def _path_touches_bounds(points: np.ndarray, bounds: tuple[float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax = bounds
    margin = 8.0
    return bool(
        (
            (points[:, 0] >= xmin - margin)
            & (points[:, 0] <= xmax + margin)
            & (points[:, 1] >= ymin - margin)
            & (points[:, 1] <= ymax + margin)
        ).any()
    )


def render_animation(
    row: dict[str, str],
    dataset_dir: Path,
    output_path: Path,
    *,
    fps: int,
    dpi: int,
) -> None:
    data = _load_row_data(row, dataset_dir)
    trajectory = data["trajectory"]
    valid_mask = data["valid_mask"]
    size = data["size"]
    bounds = _square_bounds(trajectory, valid_mask)
    num_steps = trajectory.shape[1]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=dpi)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#202020")

    for points in data["map_paths"]:
        points = np.asarray(points, dtype=np.float32)
        if _path_touches_bounds(points, bounds):
            ax.plot(points[:, 0], points[:, 1], color="#8b8b8b", linewidth=0.55, alpha=0.58, zorder=1)

    lines = []
    markers = []
    boxes = []
    ids = (int(row["first_agent_id"]), int(row["second_agent_id"]))
    types = (int(row["first_agent_type"]), int(row["second_agent_type"]))
    for agent in range(2):
        label = f"agent {agent + 1}: id={ids[agent]} ({TYPE_NAMES.get(types[agent], str(types[agent]))})"
        (line,) = ax.plot([], [], color=COLORS[agent], linewidth=3.0, alpha=0.92, label=label, zorder=5)
        (marker,) = ax.plot(
            [], [], marker="o", markersize=9, color=COLORS[agent], markeredgecolor="white",
            markeredgewidth=1.1, linestyle="none", zorder=8,
        )
        box = Polygon(np.zeros((4, 2), dtype=np.float32), closed=True, facecolor=COLORS[agent],
                      edgecolor="white", linewidth=0.9, alpha=0.35, zorder=7)
        box.set_visible(False)
        ax.add_patch(box)
        lines.append(line)
        markers.append(marker)
        boxes.append(box)

        valid_indices = np.flatnonzero(valid_mask[agent])
        if len(valid_indices):
            start = trajectory[agent, int(valid_indices[0]), 0:2]
            ax.scatter(start[0], start[1], marker="^", s=82, color=COLORS[agent], edgecolors="white",
                       linewidths=0.9, zorder=6)

    primary_steps = (float(row["primary_step_first"]), float(row["primary_step_second"]))
    closest_points = [_interpolate_xy(trajectory[i], primary_steps[i]) for i in range(2)]
    ax.plot(
        [closest_points[0][0], closest_points[1][0]],
        [closest_points[0][1], closest_points[1][1]],
        color="#f1f1f1", linewidth=1.2, linestyle="--", alpha=0.9, zorder=3,
        label="fallback closest-path pair",
    )
    for agent, point in enumerate(closest_points):
        ax.scatter(point[0], point[1], marker="*", s=180, facecolors="none", edgecolors=COLORS[agent],
                   linewidths=2.0, zorder=9)
        ax.annotate(
            f"closest @ {primary_steps[agent] * 0.1:.1f}s",
            xy=point,
            xytext=(7, 7 if agent == 0 else -14),
            textcoords="offset points",
            fontsize=8,
            color="#f4f4f4",
            bbox={"facecolor": "#111111", "alpha": 0.62, "edgecolor": "none", "pad": 2},
            zorder=10,
        )

    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#555555", alpha=0.28, linewidth=0.5)
    ax.set_xlabel("event-frame longitudinal position (m)", color="#dddddd")
    ax.set_ylabel("event-frame lateral position (m)", color="#dddddd")
    ax.tick_params(colors="#dddddd", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#888888")

    ax.legend(loc="best", fontsize=8, facecolor="#151515", edgecolor="#777777", labelcolor="white")
    title = ax.set_title("", color="white", fontsize=11, pad=12)
    time_text = ax.text(
        0.015, 0.015, "", transform=ax.transAxes, color="#f3f3f3", fontsize=9, va="bottom",
        bbox={"facecolor": "#111111", "alpha": 0.75, "edgecolor": "none", "pad": 4}, zorder=12,
    )

    def update(frame: int):
        for agent in range(2):
            visible = trajectory[agent, :, 0:2].copy()
            visible[~valid_mask[agent]] = np.nan
            visible[frame + 1 :] = np.nan
            lines[agent].set_data(visible[:, 0], visible[:, 1])
            if valid_mask[agent, frame]:
                x, y = trajectory[agent, frame, 0:2]
                markers[agent].set_data([x], [y])
                yaw = float(np.arctan2(trajectory[agent, frame, 4], trajectory[agent, frame, 5]))
                boxes[agent].set_xy(_obb_corners(x, y, yaw, float(size[agent, 0]), float(size[agent, 1])))
                boxes[agent].set_visible(True)
            else:
                markers[agent].set_data([], [])
                boxes[agent].set_visible(False)

        title.set_text(
            f"OOI closest fallback | {row['selection_role']}\n"
            f"scenario={row['scenario_id']}  |  t={frame * 0.1:.1f}s / {(num_steps - 1) * 0.1:.1f}s"
        )
        time_text.set_text(
            f"PET / arrival gap: {float(row['zone_pet_s']):.2f}s\n"
            f"estimated clearance: {float(row['min_clearance_m']):.2f}m\n"
            "triangle=start, circle=current, stars=fallback pair"
        )
        return [*lines, *markers, *boxes, title, time_text]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=range(num_steps),
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)


def make_gif(mp4_path: Path, gif_path: Path, fps: int) -> None:
    filter_graph = (
        f"fps={fps},scale=720:-2:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
    )
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4_path),
            "-filter_complex", filter_graph, "-loop", "0", str(gif_path),
        ],
        check=True,
    )


def write_gallery(path: Path, rendered: list[tuple[dict[str, str], Path, Path | None]]) -> None:
    cards = []
    for row, mp4_path, gif_path in rendered:
        original = "../" + Path(row["figure"]).name
        gif_link = f" · <a href='{html.escape(gif_path.name)}'>GIF</a>" if gif_path is not None else ""
        cards.append(
            "<article>"
            f"<video controls loop muted playsinline preload='metadata' poster='{html.escape(original)}'>"
            f"<source src='{html.escape(mp4_path.name)}' type='video/mp4'></video>"
            f"<h2>{html.escape(row['selection_role'])}</h2>"
            f"<p>scenario={html.escape(row['scenario_id'])}; PET={float(row['zone_pet_s']):.2f}s; "
            f"clearance={float(row['min_clearance_m']):.2f}m</p>"
            f"<p><a href='{html.escape(mp4_path.name)}'>MP4</a>{gif_link} · "
            f"<a href='{html.escape(original)}'>原静态图</a></p></article>"
        )
    path.write_text(
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>OOI fallback 轨迹动画</title><style>"
        "body{font-family:system-ui;margin:24px;background:#eee;color:#222}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px}"
        "article{background:white;padding:12px;border-radius:8px;box-shadow:0 1px 4px #bbb}"
        "video{width:100%;background:#111}h2{font-size:18px;margin:8px 0}p{margin:6px 0}"
        "</style></head><body><h1>OOI closest fallback 轨迹动画</h1>"
        "<p>两条轨迹按 10 Hz 同步延伸。三角形是起点，圆点是当前同步时刻，"
        "空心星号及虚线是 fallback 算法选出的两个异步最近轨迹点。</p>"
        f"<main>{''.join(cards)}</main></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gallery-dir",
        type=Path,
        default=Path("waymo/cache/interaction_full_pairs_50k_v2/visual_audit_36_final"),
    )
    parser.add_argument("--category", default="ooi_fallback")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--gif", action="store_true", help="Also convert each MP4 to a looping GIF.")
    args = parser.parse_args()

    gallery_dir = args.gallery_dir.resolve()
    config_path = gallery_dir / "gallery_config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    import json

    config = json.loads(config_path.read_text())
    dataset_dir = Path(config["dataset_dir"])
    with (gallery_dir / "selected_samples.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["audit_category"] == args.category]
    if not rows:
        raise RuntimeError(f"No rows found for category={args.category!r} in {gallery_dir}")

    output_dir = (args.output_dir or (gallery_dir / args.category / "animated")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[dict[str, str], Path, Path | None]] = []
    for index, row in enumerate(rows, start=1):
        stem = Path(row["figure"]).stem
        mp4_path = output_dir / f"{stem}.mp4"
        print(f"[{index}/{len(rows)}] rendering {mp4_path}", flush=True)
        render_animation(row, dataset_dir, mp4_path, fps=args.fps, dpi=args.dpi)
        gif_path = None
        if args.gif:
            gif_path = output_dir / f"{stem}.gif"
            print(f"[{index}/{len(rows)}] converting {gif_path}", flush=True)
            make_gif(mp4_path, gif_path, fps=args.fps)
        rendered.append((row, mp4_path, gif_path))

    index_path = output_dir / "index.html"
    write_gallery(index_path, rendered)
    print(f"saved {index_path}", flush=True)


if __name__ == "__main__":
    main()
