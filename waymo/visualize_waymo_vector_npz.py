"""Visualize a filtered Waymo vector NPZ as an MP4 video.

The NPZ is expected to come from `waymo_vector_filter.py` and contain:

- agents: (K, T, 8) with x, y, speed, vx, vy, valid, yaw, type
- map_polylines: (M, P, 6) with x, y, dir_x, dir_y, type, valid
- lights: (T, L, 4) with x, y, state, valid
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import cv2
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

    map_poly = item["map_polylines"]
    map_mask = item["map_mask"].astype(bool)
    if map_mask.any():
        points.append(map_poly[:, :, 0:2][map_mask])

    lights = item["lights"]
    light_mask = item["light_mask"].astype(bool)
    if light_mask.any():
        points.append(lights[:, :, 0:2][light_mask])

    if not points:
        return -50.0, 50.0, -50.0, 50.0

    xy = np.concatenate(points, axis=0)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return -50.0, 50.0, -50.0, 50.0

    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xmin -= margin_m
    ymin -= margin_m
    xmax += margin_m
    ymax += margin_m

    if xmax - xmin < 10.0:
        cx = 0.5 * (xmin + xmax)
        xmin, xmax = cx - 5.0, cx + 5.0
    if ymax - ymin < 10.0:
        cy = 0.5 * (ymin + ymax)
        ymin, ymax = cy - 5.0, cy + 5.0
    return float(xmin), float(xmax), float(ymin), float(ymax)


class Viewport:
    def __init__(self, bounds: Tuple[float, float, float, float], width: int, height: int, pad: int):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.width = int(width)
        self.height = int(height)
        self.pad = int(pad)
        sx = (self.width - 2 * self.pad) / max(1e-6, self.xmax - self.xmin)
        sy = (self.height - 2 * self.pad) / max(1e-6, self.ymax - self.ymin)
        self.scale = float(min(sx, sy))

        used_w = self.scale * (self.xmax - self.xmin)
        used_h = self.scale * (self.ymax - self.ymin)
        self.xoff = 0.5 * (self.width - used_w)
        self.yoff = 0.5 * (self.height - used_h)

    def xy_to_px(self, xy: np.ndarray) -> np.ndarray:
        arr = np.asarray(xy, dtype=np.float32)
        out = np.empty_like(arr, dtype=np.int32)
        out[..., 0] = np.round((arr[..., 0] - self.xmin) * self.scale + self.xoff).astype(np.int32)
        out[..., 1] = np.round(self.height - ((arr[..., 1] - self.ymin) * self.scale + self.yoff)).astype(np.int32)
        return out


def _agent_color(agent_type: int, is_ego: bool) -> Tuple[int, int, int]:
    if is_ego:
        return (80, 240, 80)
    # OpenCV uses BGR.
    colors = {
        1: (255, 170, 70),   # vehicle
        2: (220, 90, 220),   # pedestrian
        3: (80, 220, 255),   # cyclist
        4: (180, 180, 180),  # other
    }
    return colors.get(int(agent_type), (210, 210, 210))


def _light_color(state: int) -> Tuple[int, int, int]:
    # Waymo signal states: 1/4/7 stop, 2/5/8 caution, 3/6 go.
    if int(state) in (1, 4, 7):
        return (40, 40, 230)
    if int(state) in (2, 5, 8):
        return (30, 210, 240)
    if int(state) in (3, 6):
        return (60, 220, 60)
    return (160, 160, 160)


def _draw_map(frame: np.ndarray, item: Dict[str, np.ndarray], vp: Viewport) -> None:
    map_poly = item["map_polylines"]
    map_mask = item["map_mask"].astype(bool)
    for poly, mask in zip(map_poly, map_mask):
        pts = poly[mask, 0:2]
        if len(pts) < 2:
            continue
        px = vp.xy_to_px(pts).reshape((-1, 1, 2))
        cv2.polylines(frame, [px], isClosed=False, color=(92, 92, 92), thickness=1, lineType=cv2.LINE_AA)


def _draw_lights(frame: np.ndarray, item: Dict[str, np.ndarray], t: int, vp: Viewport) -> None:
    lights = item["lights"][t]
    light_mask = item["light_mask"][t].astype(bool)
    for light in lights[light_mask]:
        xy = light[0:2]
        state = int(light[2])
        px = tuple(vp.xy_to_px(xy).tolist())
        color = _light_color(state)
        cv2.circle(frame, px, 5, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, px, 7, (20, 20, 20), thickness=1, lineType=cv2.LINE_AA)


def _draw_agent(
    frame: np.ndarray,
    xy: np.ndarray,
    yaw: float,
    color: Tuple[int, int, int],
    vp: Viewport,
    label: str,
    ring_color: Tuple[int, int, int] | None = None,
) -> None:
    center = vp.xy_to_px(xy).astype(np.int32)
    px = tuple(center.tolist())
    radius = 6
    cv2.circle(frame, px, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(frame, px, radius + 1, (15, 15, 15), thickness=1, lineType=cv2.LINE_AA)
    if ring_color is not None:
        cv2.circle(frame, px, radius + 5, ring_color, thickness=2, lineType=cv2.LINE_AA)

    heading_len_m = 4.0
    tip_xy = xy + heading_len_m * np.asarray([np.cos(yaw), np.sin(yaw)], dtype=np.float32)
    tip = tuple(vp.xy_to_px(tip_xy).astype(np.int32).tolist())
    cv2.arrowedLine(frame, px, tip, color, thickness=2, line_type=cv2.LINE_AA, tipLength=0.35)

    if label:
        cv2.putText(frame, label, (px[0] + 8, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def _draw_agents(frame: np.ndarray, item: Dict[str, np.ndarray], t: int, vp: Viewport, trail: int) -> None:
    agents = item["agents"]
    agent_mask = item["agent_mask"].astype(bool)
    agent_ids = item.get("agent_ids", np.arange(len(agent_mask)))
    ooi_mask = item.get("agent_objects_of_interest", np.zeros_like(agent_mask)).astype(bool)
    ttp_mask = item.get("agent_tracks_to_predict", np.zeros_like(agent_mask)).astype(bool)

    for k, agent in enumerate(agents):
        if not agent_mask[k]:
            continue
        if agent[t, 5] <= 0.5:
            continue

        agent_type = int(round(float(agent[t, 7])))
        color = _agent_color(agent_type, is_ego=(k == 0))

        if trail > 0:
            start = max(0, t - trail)
            hist = agent[start : t + 1]
            hist = hist[hist[:, 5] > 0.5, 0:2]
            if len(hist) >= 2:
                px = vp.xy_to_px(hist).reshape((-1, 1, 2))
                cv2.polylines(frame, [px], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)

        label_parts = ["ego" if k == 0 else str(int(agent_ids[k]))]
        ring_color = None
        if ooi_mask[k]:
            label_parts.append("OOI")
            ring_color = (0, 165, 255)
        if ttp_mask[k]:
            label_parts.append("TTP")
            ring_color = ring_color or (255, 255, 255)
        label = ":".join(label_parts)
        _draw_agent(frame, agent[t, 0:2], float(agent[t, 6]), color, vp, label, ring_color=ring_color)


def _draw_origin(frame: np.ndarray, vp: Viewport) -> None:
    origin = tuple(vp.xy_to_px(np.asarray([0.0, 0.0], dtype=np.float32)).tolist())
    cv2.drawMarker(frame, origin, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=1)
    x_tip = tuple(vp.xy_to_px(np.asarray([10.0, 0.0], dtype=np.float32)).tolist())
    y_tip = tuple(vp.xy_to_px(np.asarray([0.0, 10.0], dtype=np.float32)).tolist())
    cv2.arrowedLine(frame, origin, x_tip, (240, 240, 240), 1, cv2.LINE_AA, tipLength=0.2)
    cv2.arrowedLine(frame, origin, y_tip, (200, 200, 200), 1, cv2.LINE_AA, tipLength=0.2)
    cv2.putText(frame, "ego frame", (origin[0] + 8, origin[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, item: Dict[str, np.ndarray], t: int, fps: int) -> None:
    scenario_id = _as_str(item.get("scenario_id", "unknown"))
    agents = item["agents"]
    map_mask = item["map_mask"]
    light_mask = item["light_mask"][t]
    valid_agents = int(((agents[:, t, 5] > 0.5) & item["agent_mask"].astype(bool)).sum())
    valid_map = int((map_mask.sum(axis=1) > 0).sum())
    valid_lights = int(light_mask.sum())

    lines = [
        f"scenario: {scenario_id}",
        f"t={t:02d}/90  time={(t - CURRENT_IDX) * 0.1:+.1f}s  fps={fps}",
        f"agents now={valid_agents}  map polylines={valid_map}  lights now={valid_lights}",
        "colors: ego green, vehicles blue, peds purple, cyclists yellow; orange ring=OOI, white ring=TTP",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        y += 22


def _render_frame(item: Dict[str, np.ndarray], t: int, vp: Viewport, width: int, height: int, fps: int, trail: int) -> np.ndarray:
    frame = np.full((height, width, 3), 28, dtype=np.uint8)
    _draw_map(frame, item, vp)
    _draw_origin(frame, vp)
    _draw_lights(frame, item, t, vp)
    _draw_agents(frame, item, t, vp, trail=trail)
    _draw_hud(frame, item, t, fps=fps)
    return frame


def _write_contact_sheet(frames: list[np.ndarray], output_path: str, cols: int) -> None:
    if len(frames) == 0:
        return
    cols = max(1, int(cols))
    rows = int(np.ceil(len(frames) / cols))
    h, w = frames[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 28, dtype=np.uint8)
    for i, frame in enumerate(frames):
        r = i // cols
        c = i % cols
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = frame
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, sheet)


def _write_gif_from_mp4(mp4_path: str, gif_path: str, fps: int) -> None:
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        mp4_path,
        "-vf",
        f"fps={fps},scale=900:-1:flags=lanczos",
        gif_path,
    ]
    subprocess.run(cmd, check=True)


def render_video(
    npz_path: str,
    output_path: str,
    fps: int = 10,
    width: int = 1200,
    height: int = 900,
    trail: int = 15,
    margin_m: float = 15.0,
    start: int = 0,
    end: int | None = None,
    preview_png: str | None = None,
    preview_frames: int = 12,
    preview_cols: int = 4,
    gif_output: str | None = None,
    gif_fps: int = 5,
) -> None:
    item = _load_npz(npz_path)
    agents = item["agents"]
    num_steps = int(agents.shape[1])
    start = max(0, int(start))
    end = num_steps if end is None else min(num_steps, int(end))
    if start >= end:
        raise ValueError(f"Invalid frame range start={start}, end={end}")

    bounds = _valid_xy_bounds(item, margin_m=margin_m)
    vp = Viewport(bounds, width=width, height=height, pad=36)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer for {output_path}")

    preview_ts = set()
    if preview_png is not None and preview_frames > 0:
        preview_ts = set(np.linspace(start, end - 1, num=min(preview_frames, end - start), dtype=np.int64).tolist())
    preview_images = []

    for t in range(start, end):
        frame = _render_frame(item, t, vp, width=width, height=height, fps=fps, trail=trail)
        writer.write(frame)
        if t in preview_ts:
            small_w = max(1, width // 2)
            small_h = max(1, height // 2)
            preview_images.append(cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA))

    writer.release()

    if preview_png is not None:
        _write_contact_sheet(preview_images, preview_png, cols=preview_cols)

    if gif_output is not None:
        _write_gif_from_mp4(output_path, gif_output, fps=gif_fps)


def _default_output_path(npz_path: str) -> str:
    path = Path(npz_path)
    return str(path.with_suffix(".mp4"))


def _default_preview_path(output_path: str) -> str:
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}_preview.png"))


def main() -> None:
    p = argparse.ArgumentParser(description="Render a filtered Waymo vector NPZ to MP4.")
    p.add_argument("npz", type=str)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--preview_png", type=str, default=None)
    p.add_argument("--no_preview_png", action="store_true")
    p.add_argument("--preview_frames", type=int, default=12)
    p.add_argument("--preview_cols", type=int, default=4)
    p.add_argument("--gif_output", type=str, default=None)
    p.add_argument("--gif_fps", type=int, default=5)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--trail", type=int, default=15)
    p.add_argument("--margin_m", type=float, default=15.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()

    output = args.output or _default_output_path(args.npz)
    preview_png = None if args.no_preview_png else (args.preview_png or _default_preview_path(output))
    render_video(
        npz_path=args.npz,
        output_path=output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        trail=args.trail,
        margin_m=args.margin_m,
        start=args.start,
        end=args.end,
        preview_png=preview_png,
        preview_frames=args.preview_frames,
        preview_cols=args.preview_cols,
        gif_output=args.gif_output,
        gif_fps=args.gif_fps,
    )
    print(output)
    if preview_png is not None:
        print(preview_png)
    if args.gif_output is not None:
        print(args.gif_output)


if __name__ == "__main__":
    main()
