"""Visualize Waymo vector-tokenizer reconstructions.

This script loads one or more tokenizer checkpoints, runs reconstruction on the
same validation sample, and writes a side-by-side PNG showing ground-truth
trajectories and reconstructed trajectories over the input map.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
EVAL_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, WAYMO_ROOT / "training", WAYMO_ROOT / "core", EVAL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_waymo_vector_tokenizer import build_model  # noqa: E402
from vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo_vector_dataset import WaymoVectorDataset  # noqa: E402


DEFAULT_ARGS = {
    "d_model": 128,
    "n_heads": 4,
    "depth": 3,
    "decoder_depth": 3,
    "n_latents": 8,
    "d_bottleneck": 32,
    "hidden_dim": 64,
    "dropout": 0.05,
    "mlp_ratio": 4.0,
    "time_every": 1,
    "scale_pos_embeds": True,
    "encoder_variant": "repeat_map",
    "map_depth": 2,
    "map_cross_every": 1,
    "map_query_tokens": "latent_agent",
    "bottleneck_output": "tanh",
    "decoder_use_agent_tokens": False,
    "agent_xy_loss": "smooth_l1",
    "agent_xy_parameterization": "absolute",
}

DEFAULT_COMPARE_CHECKPOINTS = [
    REPO_ROOT / "waymo/checkpoints/ooi50k_lat16_d256_ep200_2gpu_lossfix/latest.pt",
    REPO_ROOT / "waymo/checkpoints/ooi50k_lat16_d256_ep200_2a100_staticmap_v2_lossfix/latest.pt",
]
DEFAULT_COMPARE_LABELS = ["repeat_map", "staticmap_v2"]


def _checkpoint_args(ckpt: Dict) -> SimpleNamespace:
    values = dict(DEFAULT_ARGS)
    values.update(ckpt.get("args", {}))
    return SimpleNamespace(**values)


def _checkpoint_time_window(checkpoint: str) -> int:
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = _checkpoint_args(ckpt)
    return int(getattr(args, "time_window", 0))


def _as_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return str(value.item()) if value.shape == () else str(value)
    return str(value)


def _scenario_id_from_npz(path: str) -> str:
    with np.load(path, allow_pickle=False) as data:
        if "scenario_id" in data:
            return _as_str(data["scenario_id"])
    return ""


def _resolve_index(dataset: WaymoVectorDataset, *, index: int, scenario: str | None) -> int:
    if not scenario:
        return int(index)

    scenario = str(scenario)
    matches: List[int] = []
    for i, path in enumerate(dataset.paths):
        stem = Path(path).stem
        if scenario == stem or scenario in stem:
            matches.append(i)
            continue
        sid = _scenario_id_from_npz(path)
        if scenario == sid or scenario in sid:
            matches.append(i)

    if not matches:
        raise ValueError(f"No scenario matching {scenario!r} found under dataset roots.")
    if len(matches) > 1:
        preview = ", ".join(Path(dataset.paths[i]).name for i in matches[:5])
        raise ValueError(
            f"Scenario {scenario!r} matched {len(matches)} files. Use a fuller id or --npz. "
            f"First matches: {preview}"
        )
    return matches[0]


def _batch_from_item(item: Dict[str, torch.Tensor], time_window: int, device: torch.device) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
    if time_window > 0:
        k = batch["agent_mask"].shape[-1]
        if batch["agents"].shape[1] == k:
            batch["agents"] = batch["agents"][:, :, :time_window]
        else:
            batch["agents"] = batch["agents"][:, :time_window]
        batch["lights"] = batch["lights"][:, :time_window]
        batch["light_mask"] = batch["light_mask"][:, :time_window]
    return batch


def _batch_time_length(batch: Dict[str, torch.Tensor]) -> int:
    k = batch["agent_mask"].shape[-1]
    if batch["agents"].shape[1] == k:
        return int(batch["agents"].shape[2])
    return int(batch["agents"].shape[1])


def _slice_batch_time(batch: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    out = dict(batch)
    k = batch["agent_mask"].shape[-1]
    if batch["agents"].shape[1] == k:
        out["agents"] = batch["agents"][:, :, start:end]
    else:
        out["agents"] = batch["agents"][:, start:end]
    out["lights"] = batch["lights"][:, start:end]
    out["light_mask"] = batch["light_mask"][:, start:end]
    return out


def _agents_btkf(agents: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    k = agent_mask.shape[-1]
    if agents.shape[1] == k:
        return agents.transpose(1, 2).contiguous()
    return agents


def _load_model(checkpoint: str, sample: Dict[str, torch.Tensor], device: torch.device) -> tuple[torch.nn.Module, SimpleNamespace]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = _checkpoint_args(ckpt)
    n_agents = int(sample["agent_mask"].shape[-1])
    n_lights = int(sample["lights"].shape[1])
    model = build_model(args, n_agents=n_agents, n_lights=n_lights)
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError:
        incompatible = model.load_state_dict(ckpt["model"], strict=False)
        allowed_gmm_keys = {
            "decoder.agent_xy_gmm_head.weight",
            "decoder.agent_xy_gmm_head.bias",
        }
        if (
            not set(incompatible.missing_keys).issubset(allowed_gmm_keys)
            or not set(incompatible.unexpected_keys).issubset(allowed_gmm_keys)
            or getattr(args, "agent_xy_loss", "smooth_l1") == "gmm"
        ):
            raise
        print(f"loaded {checkpoint} with optional xy GMM head mismatch")
    model.to(device).eval()
    return model, args


def _uses_raw_agent_targets(checkpoint: str, args: SimpleNamespace) -> bool:
    value = getattr(args, "agent_target_scale", None)
    if value is not None:
        return float(value) == 1.0
    return "_raw_" in str(checkpoint)


@torch.no_grad()
def _reconstruct(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    agent_xy_loss: str = "smooth_l1",
    agent_xy_parameterization: str = "absolute",
    xy_scale: float = 100.0,
    speed_scale: float = 30.0,
) -> Dict[str, np.ndarray]:
    out = model(
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    pred = out.decoder.agent_continuous[0].detach().float().cpu()
    agents_btkf = _agents_btkf(batch["agents"], batch["agent_mask"])
    anchor_xy = agents_btkf[:, 0, :, 0:2] if agent_xy_parameterization == "delta" else None
    pred_xy_raw = decoder_agent_xy(
        out.decoder,
        agent_xy_loss,
        agent_xy_parameterization,
        anchor_xy=anchor_xy,
    )[0].detach().float().cpu()
    pred_xy = pred_xy_raw.numpy() * float(xy_scale)
    pred_speed = pred[..., 2].numpy() * float(speed_scale)
    pred_vel = pred[..., 3:5].numpy() * float(speed_scale)
    pred_yaw = torch.atan2(pred[..., 5], pred[..., 6]).numpy()
    pred_valid_prob = torch.sigmoid(out.decoder.agent_valid_logits[0].detach().float().cpu()).numpy()
    return {
        "xy": pred_xy,
        "speed": pred_speed,
        "vel": pred_vel,
        "yaw": pred_yaw,
        "valid_prob": pred_valid_prob,
        "steps": pred_xy.shape[0],
    }


def _concat_predictions(parts: List[Dict[str, np.ndarray]], *, chunk_window: int, input_steps: int) -> Dict[str, np.ndarray]:
    if not parts:
        raise ValueError(f"Input has {input_steps} steps; no chunks to reconstruct.")
    keys = ("xy", "speed", "vel", "yaw", "valid_prob")
    out = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
    out["steps"] = int(out["xy"].shape[0])
    out["chunk_window"] = int(chunk_window)
    out["num_chunks"] = int(len(parts))
    out["chunk_lengths"] = [int(part["steps"]) for part in parts]
    out["discarded_steps"] = int(input_steps - out["steps"])
    return out


@torch.no_grad()
def _reconstruct_chunked(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    chunk_window: int,
    agent_xy_loss: str = "smooth_l1",
    agent_xy_parameterization: str = "absolute",
    xy_scale: float = 100.0,
    speed_scale: float = 30.0,
) -> Dict[str, np.ndarray]:
    if chunk_window <= 0:
        raise ValueError(f"chunk_window must be positive, got {chunk_window}")
    input_steps = _batch_time_length(batch)
    parts = []
    for start in range(0, input_steps, chunk_window):
        end = min(start + chunk_window, input_steps)
        parts.append(
            _reconstruct(
                model,
                _slice_batch_time(batch, start, end),
                agent_xy_loss=agent_xy_loss,
                agent_xy_parameterization=agent_xy_parameterization,
                xy_scale=xy_scale,
                speed_scale=speed_scale,
            )
        )
    return _concat_predictions(parts, chunk_window=chunk_window, input_steps=input_steps)


def _prediction_metrics(gt_tkf: np.ndarray, pred: Dict[str, np.ndarray], agent_mask: np.ndarray) -> Dict[str, float]:
    steps = min(int(pred["xy"].shape[0]), int(gt_tkf.shape[0]))
    valid = (gt_tkf[:steps, :, 5] > 0.5) & agent_mask[None, :]
    pred_xy = pred["xy"][:steps]
    gt_xy = gt_tkf[:steps, :, 0:2]
    if valid.any():
        ade = float(np.linalg.norm(pred_xy[valid] - gt_xy[valid], axis=-1).mean())
    else:
        ade = float("nan")

    any_valid = valid.any(axis=0)
    fdes = []
    for k in np.flatnonzero(any_valid):
        idx = np.flatnonzero(valid[:, k])[-1]
        fdes.append(float(np.linalg.norm(pred_xy[idx, k] - gt_xy[idx, k])))
    fde = float(np.mean(fdes)) if fdes else float("nan")

    focus_valid = valid[:, 0] if valid.shape[1] > 0 else np.zeros((steps,), dtype=bool)
    if focus_valid.any():
        focus_err = np.linalg.norm(pred_xy[focus_valid, 0] - gt_xy[focus_valid, 0], axis=-1)
        focus_ade = float(focus_err.mean())
        focus_last = np.flatnonzero(focus_valid)[-1]
        focus_fde = float(np.linalg.norm(pred_xy[focus_last, 0] - gt_xy[focus_last, 0]))
    else:
        focus_ade = float("nan")
        focus_fde = float("nan")

    return {
        "steps": float(steps),
        "ade_m": ade,
        "fde_m": fde,
        "focus_ade_m": focus_ade,
        "focus_fde_m": focus_fde,
        "valid_points": float(valid.sum()),
        "valid_agents": float(any_valid.sum()),
    }


def _bounds_from_points(points: List[np.ndarray], margin: float) -> tuple[float, float, float, float]:
    if not points:
        return -80.0, 80.0, -80.0, 80.0
    xy = np.concatenate(points, axis=0)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return -80.0, 80.0, -80.0, 80.0
    lo = xy.min(axis=0) - margin
    hi = xy.max(axis=0) + margin
    width = float(max(hi[0] - lo[0], 1e-6))
    height = float(max(hi[1] - lo[1], 1e-6))
    min_span = 30.0
    if width < min_span:
        pad = 0.5 * (min_span - width)
        lo[0] -= pad
        hi[0] += pad
    if height < min_span:
        pad = 0.5 * (min_span - height)
        lo[1] -= pad
        hi[1] += pad
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _valid_bounds(
    gt_tkf: np.ndarray,
    map_polylines: np.ndarray,
    map_mask: np.ndarray,
    margin: float,
    *,
    selected_agents: List[int] | None = None,
    preds: List[tuple[str, Dict[str, np.ndarray]]] | None = None,
    bounds_source: str = "agents",
) -> tuple[float, float, float, float]:
    points: List[np.ndarray] = []
    gt_valid = gt_tkf[..., 5] > 0.5

    if bounds_source == "all":
        if gt_valid.any():
            points.append(gt_tkf[..., 0:2][gt_valid])
    else:
        for k in selected_agents or []:
            valid = gt_valid[:, k]
            if valid.any():
                points.append(gt_tkf[:, k, 0:2][valid])
                for _, pred in preds or []:
                    pred_steps = min(int(pred["xy"].shape[0]), int(gt_tkf.shape[0]))
                    pred_valid = gt_valid[:pred_steps, k]
                    if pred_valid.any():
                        points.append(pred["xy"][:pred_steps, k, :][pred_valid])

    if not points and gt_valid.any():
        points.append(gt_tkf[..., 0:2][gt_valid])

    map_valid = map_mask.astype(bool)
    if bounds_source == "all" and map_valid.any():
        points.append(map_polylines[..., 0:2][map_valid])

    return _bounds_from_points(points, margin=margin)


def _agent_color(k: int, agent_type: int) -> str:
    if k == 0:
        return "#2ca02c"
    return {
        1: "#1f77b4",
        2: "#d627b0",
        3: "#ffbf00",
    }.get(int(agent_type), "#bbbbbb")


def _draw_map(ax, map_polylines: np.ndarray, map_mask: np.ndarray) -> None:
    for poly, mask in zip(map_polylines, map_mask.astype(bool)):
        pts = poly[mask, 0:2]
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], color="#9a9a9a", linewidth=0.75, alpha=0.82, zorder=1)


def _agent_errors(gt_xy: np.ndarray, pred_xy: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    if not valid.any():
        return float("nan"), float("nan")
    err = np.linalg.norm(pred_xy[valid] - gt_xy[valid], axis=-1)
    valid_idx = np.flatnonzero(valid)
    fde = float(np.linalg.norm(pred_xy[valid_idx[-1]] - gt_xy[valid_idx[-1]]))
    return float(err.mean()), fde


def _select_agents(gt_tkf: np.ndarray, agent_mask: np.ndarray, max_agents: int) -> List[int]:
    valid_counts = ((gt_tkf[..., 5] > 0.5) & agent_mask[None, :]).sum(axis=0)
    order = [0] if agent_mask[0] else []
    rest = [int(k) for k in np.argsort(-valid_counts) if int(k) != 0 and agent_mask[int(k)] and valid_counts[int(k)] > 0]
    return (order + rest)[:max_agents]


def _draw_panel(
    ax,
    *,
    title: str,
    gt_tkf: np.ndarray,
    pred: Dict[str, np.ndarray] | None,
    agent_mask: np.ndarray,
    map_polylines: np.ndarray,
    map_mask: np.ndarray,
    agent_ids: np.ndarray,
    selected_agents: List[int],
    bounds: tuple[float, float, float, float],
    show_ids: bool,
) -> List[str]:
    ax.set_facecolor("#202020")
    _draw_map(ax, map_polylines, map_mask)
    table_lines: List[str] = []
    for k in selected_agents:
        steps = gt_tkf.shape[0] if pred is None else min(int(pred["xy"].shape[0]), int(gt_tkf.shape[0]))
        valid = (gt_tkf[:steps, k, 5] > 0.5) & bool(agent_mask[k])
        if not valid.any():
            continue
        gt_part = gt_tkf[:steps]
        agent_type = int(round(float(gt_part[valid, k, 7][0])))
        color = _agent_color(k, agent_type)
        gt_xy = gt_part[:, k, 0:2]
        ax.plot(gt_xy[valid, 0], gt_xy[valid, 1], color=color, linewidth=2.0 if k == 0 else 1.3, alpha=0.95, zorder=4)
        ax.scatter(gt_xy[valid, 0][0], gt_xy[valid, 1][0], s=18, color=color, edgecolors="black", linewidths=0.4, zorder=5)
        ax.scatter(gt_xy[valid, 0][-1], gt_xy[valid, 1][-1], s=32, marker="x", color=color, zorder=5)
        if pred is not None:
            pred_xy = pred["xy"][:steps, k, :]
            ax.plot(
                pred_xy[valid, 0],
                pred_xy[valid, 1],
                color=color,
                linewidth=1.4 if k == 0 else 1.0,
                linestyle="--",
                alpha=0.9,
                zorder=6,
            )
            ade, fde = _agent_errors(gt_xy, pred_xy, valid)
            table_lines.append(f"k{k:02d} id={int(agent_ids[k])}: ADE={ade:.2f}m FDE={fde:.2f}m")
        if show_ids:
            label = "focus" if k == 0 else f"k{k}"
            ax.text(gt_xy[valid, 0][-1] + 1.0, gt_xy[valid, 1][-1] + 1.0, label, color=color, fontsize=7, zorder=8)

    ax.scatter(0.0, 0.0, marker="+", s=90, color="white", linewidths=1.2, zorder=10)
    ax.arrow(0.0, 0.0, 8.0, 0.0, color="white", width=0.08, head_width=0.8, head_length=1.0, zorder=10)
    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#444444", alpha=0.35, linewidth=0.5)
    ax.set_title(title, color="white", fontsize=11)
    ax.tick_params(colors="#dddddd", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#888888")
    return table_lines


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _pixel_transform(bounds: tuple[float, float, float, float], panel_size: int, pad: int):
    xmin, xmax, ymin, ymax = bounds
    scale = min((panel_size - 2 * pad) / max(xmax - xmin, 1e-6), (panel_size - 2 * pad) / max(ymax - ymin, 1e-6))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    def to_px(xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float32)
        px = panel_size * 0.5 + (pts[..., 0] - cx) * scale
        py = panel_size * 0.5 - (pts[..., 1] - cy) * scale
        return np.stack([px, py], axis=-1)

    return to_px


def _draw_poly(draw: ImageDraw.ImageDraw, pts: np.ndarray, fill, width: int) -> None:
    if len(pts) < 2:
        return
    draw.line([tuple(map(float, p)) for p in pts], fill=fill, width=width, joint="curve")


def _draw_cross(draw: ImageDraw.ImageDraw, xy: np.ndarray, fill, size: int = 5, width: int = 2) -> None:
    x, y = float(xy[0]), float(xy[1])
    draw.line((x - size, y - size, x + size, y + size), fill=fill, width=width)
    draw.line((x - size, y + size, x + size, y - size), fill=fill, width=width)


def _draw_pil_panel(
    *,
    title: str,
    gt_tkf: np.ndarray,
    pred: Dict[str, np.ndarray] | None,
    agent_mask: np.ndarray,
    map_polylines: np.ndarray,
    map_mask: np.ndarray,
    agent_ids: np.ndarray,
    selected_agents: List[int],
    bounds: tuple[float, float, float, float],
    panel_size: int,
    show_ids: bool,
) -> tuple[Image.Image, List[str]]:
    img = Image.new("RGB", (panel_size, panel_size), (32, 32, 32))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    to_px = _pixel_transform(bounds, panel_size=panel_size, pad=46)

    for poly, mask in zip(map_polylines, map_mask.astype(bool)):
        pts = poly[mask, 0:2]
        if len(pts) >= 2:
            _draw_poly(draw, to_px(pts), fill=(145, 145, 145), width=max(2, panel_size // 450))

    table_lines: List[str] = []
    for k in selected_agents:
        steps = gt_tkf.shape[0] if pred is None else min(int(pred["xy"].shape[0]), int(gt_tkf.shape[0]))
        valid = (gt_tkf[:steps, k, 5] > 0.5) & bool(agent_mask[k])
        if not valid.any():
            continue
        gt_part = gt_tkf[:steps]
        agent_type = int(round(float(gt_part[valid, k, 7][0])))
        color = _hex_to_rgb(_agent_color(k, agent_type))
        gt_xy = gt_part[:, k, 0:2]
        gt_pts = to_px(gt_xy[valid])
        if pred is None:
            _draw_poly(draw, gt_pts, fill=color, width=5 if k == 0 else 3)
        else:
            _draw_poly(draw, gt_pts, fill=tuple(int(0.38 * c + 0.62 * 32) for c in color), width=4 if k == 0 else 2)
            pred_xy = pred["xy"][:steps, k, :]
            pred_pts = to_px(pred_xy[valid])
            _draw_poly(draw, pred_pts, fill=color, width=5 if k == 0 else 3)
            ade, fde = _agent_errors(gt_xy, pred_xy, valid)
            table_lines.append(f"k{k:02d} id={int(agent_ids[k])}: ADE={ade:.2f}m FDE={fde:.2f}m")

        r = 7 if k == 0 else 5
        draw.ellipse((gt_pts[0, 0] - r, gt_pts[0, 1] - r, gt_pts[0, 0] + r, gt_pts[0, 1] + r), fill=color, outline=(0, 0, 0))
        _draw_cross(draw, gt_pts[-1], fill=color, size=8 if k == 0 else 6, width=3 if k == 0 else 2)
        if show_ids:
            draw.text((float(gt_pts[-1, 0] + 5), float(gt_pts[-1, 1] + 5)), "focus" if k == 0 else f"k{k}", fill=color, font=font)

    origin = to_px(np.asarray([[0.0, 0.0], [8.0, 0.0]], dtype=np.float32))
    _draw_cross(draw, origin[0], fill=(245, 245, 245), size=7)
    draw.line((origin[0, 0], origin[0, 1], origin[1, 0], origin[1, 1]), fill=(245, 245, 245), width=2)
    _draw_cross(draw, origin[1], fill=(245, 245, 245), size=3)

    draw.rectangle((0, 0, panel_size - 1, panel_size - 1), outline=(140, 140, 140), width=1)
    draw.rectangle((0, 0, panel_size, 22), fill=(17, 17, 17))
    draw.text((8, 6), title, fill=(238, 238, 238), font=font)
    return img, table_lines


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def _panel_output_dir(args: argparse.Namespace, scenario_id: str, sample_path: str, focus_agent_id: int) -> Path:
    prefix = f"{_safe_name(scenario_id or Path(sample_path).stem)}_focus_{int(focus_agent_id)}"
    if args.output_dir:
        out_dir = Path(args.output_dir) / prefix
    elif args.output:
        out_dir = Path(args.output).with_suffix("")
    else:
        out_dir = WAYMO_ROOT / "evaluation/reports/reconstruction_compare" / prefix
    return out_dir


def _save_split_pil_panels(
    args: argparse.Namespace,
    *,
    panels: List[tuple[str, Image.Image, List[str]]],
    scenario_id: str,
    sample_path: str,
    focus_agent_id: int,
    item_index: int,
    model_summaries: List[str],
) -> None:
    out_dir = _panel_output_dir(args, scenario_id, sample_path, focus_agent_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for i, (name, panel, lines) in enumerate(panels):
        caption_h = 130 if lines else 92
        img = Image.new("RGB", (panel.width, panel.height + caption_h), (17, 17, 17))
        img.paste(panel, (0, 0))
        draw = ImageDraw.Draw(img)
        y = panel.height + 10
        draw.text((8, y), f"scenario={scenario_id or Path(sample_path).stem}  index={item_index}  focus={focus_agent_id}", fill=(238, 238, 238), font=font)
        y += 16
        draw.text((8, y), f"sample={sample_path}", fill=(210, 210, 210), font=font)
        y += 16
        if i > 0:
            draw.text((8, y), "muted=GT, bright=reconstruction. Dot=start, x=end.", fill=(210, 210, 210), font=font)
            y += 16
        summary_idx = max(0, i - 1)
        if i > 0 and summary_idx < len(model_summaries):
            draw.text((8, y), model_summaries[summary_idx], fill=(210, 210, 210), font=font)
            y += 16
        for line in lines[: args.max_table_lines]:
            draw.text((8, y), line, fill=(238, 238, 238), font=font)
            y += 14
        path = out_dir / f"{i + 1:02d}_{_safe_name(name)}.png"
        img.save(path)
        print(f"wrote {path}")


def _save_metrics_csv(
    args: argparse.Namespace,
    *,
    rows: List[Dict[str, object]],
    scenario_id: str,
    sample_path: str,
    focus_agent_id: int,
) -> None:
    if not rows:
        return
    if args.split_panels:
        path = _panel_output_dir(args, scenario_id, sample_path, focus_agent_id) / "metrics.csv"
    elif args.output:
        path = Path(args.output).with_suffix(".metrics.csv")
    else:
        path = WAYMO_ROOT / "evaluation/reports/reconstruction_compare.metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def _visualize_pil(
    args: argparse.Namespace,
    *,
    gt_tkf: np.ndarray,
    pred_items: List[tuple[str, Dict[str, np.ndarray]]],
    agent_mask: np.ndarray,
    map_polylines: np.ndarray,
    map_mask: np.ndarray,
    agent_ids: np.ndarray,
    selected_agents: List[int],
    bounds: tuple[float, float, float, float],
    item_index: int,
    sample_path: str,
    scenario_id: str,
    model_summaries: List[str],
) -> None:
    panel_size = int(args.panel_size)
    gutter = 12
    caption_h = 190
    w = panel_size * (1 + len(pred_items)) + gutter * len(pred_items)
    h = panel_size + caption_h
    canvas = Image.new("RGB", (w, h), (17, 17, 17))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    panels = [
        (
            "ground_truth",
            *_draw_pil_panel(
                title="Ground Truth",
                gt_tkf=gt_tkf,
                pred=None,
                agent_mask=agent_mask,
                map_polylines=map_polylines,
                map_mask=map_mask,
                agent_ids=agent_ids,
                selected_agents=selected_agents,
                bounds=bounds,
                panel_size=panel_size,
                show_ids=True,
            ),
        )
    ]
    for label, pred in pred_items:
        panels.append(
            (
                label,
                *_draw_pil_panel(
                    title=f"{label} reconstruction",
                    gt_tkf=gt_tkf,
                    pred=pred,
                    agent_mask=agent_mask,
                    map_polylines=map_polylines,
                    map_mask=map_mask,
                    agent_ids=agent_ids,
                    selected_agents=selected_agents,
                    bounds=bounds,
                    panel_size=panel_size,
                    show_ids=False,
                ),
            )
        )

    focus_agent_id = int(agent_ids[0]) if len(agent_ids) else -1
    if args.split_panels:
        _save_split_pil_panels(
            args,
            panels=panels,
            scenario_id=scenario_id,
            sample_path=sample_path,
            focus_agent_id=focus_agent_id,
            item_index=item_index,
            model_summaries=model_summaries,
        )
        return

    x = 0
    all_table_lines: List[str] = []
    for _, panel, lines in panels:
        canvas.paste(panel, (x, 0))
        x += panel_size + gutter
        all_table_lines.extend(lines[: args.max_table_lines])

    y = panel_size + 10
    draw.text((8, y), f"Tokenizer reconstruction | index={item_index} | scenario={scenario_id or Path(sample_path).stem}", fill=(238, 238, 238), font=font)
    y += 16
    draw.text((8, y), f"sample={sample_path}", fill=(210, 210, 210), font=font)
    y += 16
    draw.text((8, y), "Recon panels: muted line is GT, bright line is reconstruction. Dot=start, x=end.", fill=(210, 210, 210), font=font)
    y += 18
    for line in model_summaries[:6]:
        draw.text((8, y), line, fill=(210, 210, 210), font=font)
        y += 14

    if all_table_lines:
        tx = max(8, w - 410)
        ty = panel_size + 10
        for line in all_table_lines[:18]:
            draw.text((tx, ty), line, fill=(238, 238, 238), font=font)
            ty += 14

    if not args.output:
        args.output = str(WAYMO_ROOT / "evaluation/reports/reconstruction_compare.png")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"wrote {args.output}")


def visualize(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.time_window < 0 and not args.chunked_full_trajectory:
        args.time_window = _checkpoint_time_window(args.checkpoint[0]) if args.checkpoint else 0
    dataset = WaymoVectorDataset(args.npz if args.npz else args.data_dir)
    item_index = _resolve_index(dataset, index=args.index, scenario=args.scenario)
    item = dataset[item_index]
    batch_time_window = 0 if args.chunked_full_trajectory else args.time_window
    batch = _batch_from_item(item, batch_time_window, device)
    input_steps = _batch_time_length(batch)

    gt_tkf = _agents_btkf(batch["agents"], batch["agent_mask"])[0].detach().float().cpu().numpy()
    agent_mask = batch["agent_mask"][0].detach().cpu().numpy().astype(bool)
    if gt_tkf.shape[1] != agent_mask.shape[0]:
        raise ValueError(f"Agent layout mismatch: gt_tkf={gt_tkf.shape}, agent_mask={agent_mask.shape}")
    map_polylines = batch["map_polylines"][0].detach().float().cpu().numpy()
    map_mask = batch["map_mask"][0].detach().cpu().numpy().astype(bool)
    agent_ids = batch["agent_ids"][0].detach().cpu().numpy()
    selected_agents = _select_agents(gt_tkf, agent_mask, args.max_agents)

    labels = args.label or []
    if len(labels) < len(args.checkpoint):
        labels.extend([Path(p).parent.name for p in args.checkpoint[len(labels) :]])

    preds = []
    model_summaries = []
    metric_rows: List[Dict[str, object]] = []
    for ckpt_path, label in zip(args.checkpoint, labels):
        model, ckpt_args = _load_model(ckpt_path, item, device)
        agent_xy_loss = getattr(ckpt_args, "agent_xy_loss", "smooth_l1")
        agent_xy_parameterization = getattr(ckpt_args, "agent_xy_parameterization", "absolute")
        raw_targets = _uses_raw_agent_targets(ckpt_path, ckpt_args)
        xy_scale = 1.0 if raw_targets else 100.0
        speed_scale = 1.0 if raw_targets else 30.0
        if args.chunked_full_trajectory:
            chunk_window = int(args.time_window) if args.time_window > 0 else int(getattr(ckpt_args, "time_window", 0))
            pred = _reconstruct_chunked(
                model,
                batch,
                chunk_window=chunk_window,
                agent_xy_loss=agent_xy_loss,
                agent_xy_parameterization=agent_xy_parameterization,
                xy_scale=xy_scale,
                speed_scale=speed_scale,
            )
            mode = "chunked"
        else:
            chunk_window = int(_batch_time_length(batch))
            pred = _reconstruct(
                model,
                batch,
                agent_xy_loss=agent_xy_loss,
                agent_xy_parameterization=agent_xy_parameterization,
                xy_scale=xy_scale,
                speed_scale=speed_scale,
            )
            mode = "single"
        preds.append((label, pred))
        metrics = _prediction_metrics(gt_tkf, pred, agent_mask)
        metric_rows.append(
            {
                "label": label,
                "mode": mode,
                "checkpoint": str(ckpt_path),
                "input_steps": input_steps,
                "recon_steps": int(metrics["steps"]),
                "chunk_window": int(pred.get("chunk_window", chunk_window)),
                "num_chunks": int(pred.get("num_chunks", 1)),
                "chunk_lengths": "+".join(str(x) for x in pred.get("chunk_lengths", [int(metrics["steps"])])),
                "discarded_steps": int(pred.get("discarded_steps", max(0, input_steps - int(metrics["steps"])))),
                "ade_m": metrics["ade_m"],
                "fde_m": metrics["fde_m"],
                "focus_ade_m": metrics["focus_ade_m"],
                "focus_fde_m": metrics["focus_fde_m"],
                "valid_points": int(metrics["valid_points"]),
                "valid_agents": int(metrics["valid_agents"]),
            }
        )
        chunk_note = (
            f", chunkT={int(pred.get('chunk_window', chunk_window))}, chunks={int(pred.get('num_chunks', 1))}, "
            f"lens={'+'.join(str(x) for x in pred.get('chunk_lengths', [int(metrics['steps'])]))}, "
            f"used={int(metrics['steps'])}/{input_steps}, drop={int(pred.get('discarded_steps', max(0, input_steps - int(metrics['steps']))))}"
        )
        model_summaries.append(
            f"{label}: {ckpt_args.encoder_variant}, z={ckpt_args.n_latents}x{ckpt_args.d_bottleneck}, "
            f"D={ckpt_args.d_model}, depth={ckpt_args.depth}{chunk_note}, "
            f"ADE={metrics['ade_m']:.2f}m FDE={metrics['fde_m']:.2f}m focusFDE={metrics['focus_fde_m']:.2f}m"
        )

    bounds = _valid_bounds(
        gt_tkf,
        map_polylines,
        map_mask,
        margin=args.margin_m,
        selected_agents=selected_agents,
        preds=preds,
        bounds_source=args.bounds_source,
    )

    sample_path = str(item.get("path", "unknown"))
    scenario_id = _as_str(item.get("scenario_id", ""))
    focus_agent_id = int(agent_ids[0]) if len(agent_ids) else -1
    _save_metrics_csv(
        args,
        rows=metric_rows,
        scenario_id=scenario_id,
        sample_path=sample_path,
        focus_agent_id=focus_agent_id,
    )
    if args.backend == "pil" or plt is None:
        _visualize_pil(
            args,
            gt_tkf=gt_tkf,
            pred_items=preds,
            agent_mask=agent_mask,
            map_polylines=map_polylines,
            map_mask=map_mask,
            agent_ids=agent_ids,
            selected_agents=selected_agents,
            bounds=bounds,
            item_index=item_index,
            sample_path=sample_path,
            scenario_id=scenario_id,
            model_summaries=model_summaries,
        )
        return

    num_panels = 1 + len(preds)
    fig, axes = plt.subplots(1, num_panels, figsize=(7 * num_panels, 7), dpi=args.dpi)
    fig.patch.set_facecolor("#111111")
    axes_arr = np.asarray(axes).reshape(-1)

    _draw_panel(
        axes_arr[0],
        title="Ground Truth",
        gt_tkf=gt_tkf,
        pred=None,
        agent_mask=agent_mask,
        map_polylines=map_polylines,
        map_mask=map_mask,
        agent_ids=agent_ids,
        selected_agents=selected_agents,
        bounds=bounds,
        show_ids=True,
    )

    all_table_lines: List[str] = []
    for ax, (label, pred) in zip(axes_arr[1:], preds):
        lines = _draw_panel(
            ax,
            title=f"{label}: solid=GT, dashed=recon",
            gt_tkf=gt_tkf,
            pred=pred,
            agent_mask=agent_mask,
            map_polylines=map_polylines,
            map_mask=map_mask,
            agent_ids=agent_ids,
            selected_agents=selected_agents,
            bounds=bounds,
            show_ids=False,
        )
        all_table_lines.append(label)
        all_table_lines.extend(lines[: args.max_table_lines])

    fig.suptitle(
        f"Tokenizer reconstruction | index={item_index} | T={gt_tkf.shape[0]} | "
        f"scenario={scenario_id or Path(sample_path).stem}",
        color="white",
        fontsize=13,
    )
    note = "GT trajectory: solid. Reconstructed trajectory: dashed. Dot=start, x=end.\n" + "\n".join(model_summaries)
    fig.text(0.01, 0.02, note, color="#eeeeee", fontsize=9, va="bottom")
    if all_table_lines:
        fig.text(
            0.99,
            0.02,
            "\n".join(all_table_lines),
            color="#eeeeee",
            fontsize=8,
            ha="right",
            va="bottom",
            bbox={"facecolor": "#111111", "alpha": 0.75, "edgecolor": "#444444", "pad": 5},
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize tokenizer GT vs reconstructed trajectories.")
    p.add_argument("--data_dir", type=str, default=str(WAYMO_ROOT / "data/waymo_vector_dataset_ooi_centered_50k/val"))
    p.add_argument("--npz", type=str, default=None, help="Visualize one exact filtered scenario .npz file.")
    p.add_argument("--scenario", type=str, default=None, help="Scenario id or unique substring of the .npz filename/scenario_id.")
    p.add_argument("--checkpoint", type=str, nargs="+", default=[str(p) for p in DEFAULT_COMPARE_CHECKPOINTS])
    p.add_argument("--label", type=str, nargs="*", default=None)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--time_window", type=int, default=-1, help="Use checkpoint training window by default; pass 0 to render all timesteps.")
    p.add_argument(
        "--chunked_full_trajectory",
        action="store_true",
        help="Reconstruct the full input with non-overlapping time chunks; the final short chunk is decoded without padding.",
    )
    p.add_argument("--max_agents", type=int, default=8)
    p.add_argument("--max_table_lines", type=int, default=8)
    p.add_argument("--margin_m", type=float, default=12.0)
    p.add_argument("--bounds_source", choices=["agents", "all"], default="agents")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--backend", choices=["auto", "matplotlib", "pil"], default="auto")
    p.add_argument("--panel_size", type=int, default=1100, help="PIL backend panel size in pixels.")
    p.add_argument("--split_panels", action="store_true", help="Write GT and each reconstruction as separate PNGs in a scenario-named folder.")
    args = p.parse_args()
    if args.label is None and args.checkpoint == [str(p) for p in DEFAULT_COMPARE_CHECKPOINTS]:
        args.label = list(DEFAULT_COMPARE_LABELS)
    if args.backend == "auto":
        args.backend = "pil" if args.split_panels else ("matplotlib" if plt is not None else "pil")
    if args.backend == "matplotlib" and plt is None:
        raise ModuleNotFoundError("matplotlib is not installed; use --backend pil")
    if not args.split_panels and not args.output:
        args.output = str(WAYMO_ROOT / "evaluation/reports/reconstruction_compare.png")
    visualize(args)


if __name__ == "__main__":
    main()
