"""Visualize Waymo vector-tokenizer reconstructions.

This script loads one or more tokenizer checkpoints, runs reconstruction on the
same validation sample, and writes a side-by-side PNG showing ground-truth
trajectories and reconstructed trajectories over the input map.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WAYMO_ROOT = Path(__file__).resolve().parent
if str(WAYMO_ROOT) not in sys.path:
    sys.path.insert(0, str(WAYMO_ROOT))

from train_waymo_vector_tokenizer import build_model  # noqa: E402
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
}


def _checkpoint_args(ckpt: Dict) -> SimpleNamespace:
    values = dict(DEFAULT_ARGS)
    values.update(ckpt.get("args", {}))
    return SimpleNamespace(**values)


def _batch_from_item(item: Dict[str, torch.Tensor], time_window: int, device: torch.device) -> Dict[str, torch.Tensor]:
    batch: Dict[str, torch.Tensor] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
    if time_window > 0:
        batch["agents"] = batch["agents"][:, :, :time_window]
        batch["lights"] = batch["lights"][:, :time_window]
        batch["light_mask"] = batch["light_mask"][:, :time_window]
    return batch


def _load_model(checkpoint: str, sample: Dict[str, torch.Tensor], device: torch.device) -> tuple[torch.nn.Module, SimpleNamespace]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = _checkpoint_args(ckpt)
    n_agents = int(sample["agent_mask"].shape[-1])
    n_lights = int(sample["lights"].shape[1])
    model = build_model(args, n_agents=n_agents, n_lights=n_lights)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, args


@torch.no_grad()
def _reconstruct(model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    out = model(
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
    )
    pred = out.decoder.agent_continuous[0].detach().float().cpu()
    pred_xy = pred[..., 0:2].numpy() * 100.0
    pred_speed = pred[..., 2].numpy() * 30.0
    pred_vel = pred[..., 3:5].numpy() * 30.0
    pred_yaw = torch.atan2(pred[..., 5], pred[..., 6]).numpy()
    pred_valid_prob = torch.sigmoid(out.decoder.agent_valid_logits[0].detach().float().cpu()).numpy()
    return {
        "xy": pred_xy,
        "speed": pred_speed,
        "vel": pred_vel,
        "yaw": pred_yaw,
        "valid_prob": pred_valid_prob,
    }


def _valid_bounds(gt_tkf: np.ndarray, map_polylines: np.ndarray, map_mask: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    points = []
    gt_valid = gt_tkf[..., 5] > 0.5
    if gt_valid.any():
        points.append(gt_tkf[..., 0:2][gt_valid])
    map_valid = map_mask.astype(bool)
    if map_valid.any():
        points.append(map_polylines[..., 0:2][map_valid])
    if not points:
        return -80.0, 80.0, -80.0, 80.0
    xy = np.concatenate(points, axis=0)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return -80.0, 80.0, -80.0, 80.0
    lo = xy.min(axis=0) - margin
    hi = xy.max(axis=0) + margin
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


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
            ax.plot(pts[:, 0], pts[:, 1], color="#777777", linewidth=0.45, alpha=0.65, zorder=1)


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
        valid = (gt_tkf[:, k, 5] > 0.5) & bool(agent_mask[k])
        if not valid.any():
            continue
        agent_type = int(round(float(gt_tkf[valid, k, 7][0])))
        color = _agent_color(k, agent_type)
        gt_xy = gt_tkf[:, k, 0:2]
        ax.plot(gt_xy[valid, 0], gt_xy[valid, 1], color=color, linewidth=2.0 if k == 0 else 1.3, alpha=0.95, zorder=4)
        ax.scatter(gt_xy[valid, 0][0], gt_xy[valid, 1][0], s=18, color=color, edgecolors="black", linewidths=0.4, zorder=5)
        ax.scatter(gt_xy[valid, 0][-1], gt_xy[valid, 1][-1], s=32, marker="x", color=color, zorder=5)
        if pred is not None:
            pred_xy = pred["xy"][:, k, :]
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


def visualize(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = WaymoVectorDataset(args.data_dir)
    item = dataset[int(args.index)]
    batch = _batch_from_item(item, args.time_window, device)

    gt_ktf = batch["agents"][0].detach().float().cpu().numpy()
    gt_tkf = np.transpose(gt_ktf, (1, 0, 2))
    agent_mask = batch["agent_mask"][0].detach().cpu().numpy().astype(bool)
    map_polylines = batch["map_polylines"][0].detach().float().cpu().numpy()
    map_mask = batch["map_mask"][0].detach().cpu().numpy().astype(bool)
    agent_ids = batch["agent_ids"][0].detach().cpu().numpy()
    selected_agents = _select_agents(gt_tkf, agent_mask, args.max_agents)
    bounds = _valid_bounds(gt_tkf, map_polylines, map_mask, margin=args.margin_m)

    labels = args.label or []
    if len(labels) < len(args.checkpoint):
        labels.extend([Path(p).parent.name for p in args.checkpoint[len(labels) :]])

    preds = []
    model_summaries = []
    for ckpt_path, label in zip(args.checkpoint, labels):
        model, ckpt_args = _load_model(ckpt_path, item, device)
        preds.append((label, _reconstruct(model, batch)))
        model_summaries.append(
            f"{label}: {ckpt_args.encoder_variant}, z={ckpt_args.n_latents}x{ckpt_args.d_bottleneck}, "
            f"D={ckpt_args.d_model}, depth={ckpt_args.depth}"
        )

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

    sample_path = str(item.get("path", "unknown"))
    fig.suptitle(
        f"Tokenizer reconstruction | index={args.index} | T={gt_tkf.shape[0]} | sample={Path(sample_path).name}",
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
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, nargs="+", required=True)
    p.add_argument("--label", type=str, nargs="*", default=None)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--time_window", type=int, default=32)
    p.add_argument("--max_agents", type=int, default=8)
    p.add_argument("--max_table_lines", type=int, default=8)
    p.add_argument("--margin_m", type=float, default=20.0)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
