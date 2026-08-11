#!/usr/bin/env python3
"""Visualize and quantify stochastic diversity in one 32-rollout WOSAC NPZ."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


CURRENT_INDEX = 10
FUTURE_SLICE = slice(11, 91)
ROLLOUTS = 32
FUTURE_STEPS = 80


def world_to_local_xy(world_xy: np.ndarray, origin_xy: np.ndarray, heading: float) -> np.ndarray:
    delta = world_xy - origin_xy
    c = np.float32(math.cos(heading))
    s = np.float32(math.sin(heading))
    local = np.empty_like(delta, dtype=np.float32)
    local[..., 0] = c * delta[..., 0] + s * delta[..., 1]
    local[..., 1] = -s * delta[..., 0] + c * delta[..., 1]
    return local


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    """Return [R*(R-1)/2,T,A] pairwise Euclidean distances."""
    parts = []
    for first in range(xy.shape[0]):
        for second in range(first + 1, xy.shape[0]):
            parts.append(np.linalg.norm(xy[first] - xy[second], axis=-1))
    return np.stack(parts, axis=0)


def circular_std(angle: np.ndarray, axis: int = 0) -> np.ndarray:
    mean_cos = np.mean(np.cos(angle), axis=axis)
    mean_sin = np.mean(np.sin(angle), axis=axis)
    resultant = np.clip(np.hypot(mean_cos, mean_sin), 1e-8, 1.0)
    return np.sqrt(np.maximum(0.0, -2.0 * np.log(resultant)))


def draw_map(ax: plt.Axes, map_polylines: np.ndarray, map_mask: np.ndarray) -> None:
    for points, mask in zip(map_polylines, map_mask):
        valid = mask & np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
        if int(valid.sum()) >= 2:
            ax.plot(points[valid, 0], points[valid, 1], color="#777777", linewidth=0.7, alpha=0.55, zorder=0)


def setup_axis(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    ax.set_facecolor("#171717")
    ax.grid(color="#444444", linewidth=0.4, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.tick_params(colors="#dddddd", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#888888")


def scene_bounds(gt_agents: np.ndarray, slots: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    gt = np.transpose(gt_agents[slots], (1, 0, 2))
    valid = gt[..., 5] > 0.5
    points = gt[..., :2][valid]
    lo = np.nanmin(points, axis=0)
    hi = np.nanmax(points, axis=0)
    center = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0 + margin, 20.0)
    extent = float(max(half[0], half[1]))
    return center[0] - extent, center[0] + extent, center[1] - extent, center[1] + extent


def make_contact_sheet(images: list[Path], output: Path) -> None:
    thumb_size = 300
    label_height = 26
    columns = 8
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), (12, 12, 12))
    for index, path in enumerate(images):
        with Image.open(path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_size, thumb_size + label_height), (20, 20, 20))
        tile.paste(thumb, ((thumb_size - thumb.width) // 2, 0))
        ImageDraw.Draw(tile).text((8, thumb_size + 5), f"rollout {index:02d}", fill=(235, 235, 235))
        sheet.paste(tile, ((index % columns) * thumb_size, (index // columns) * (thumb_size + label_height)))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def visualize(args: argparse.Namespace) -> dict[str, object]:
    rollout_path = Path(args.rollout_npz).resolve()
    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "individual_rollouts"
    images_dir.mkdir(parents=True, exist_ok=True)

    with np.load(rollout_path, allow_pickle=False) as rollout:
        center_x = np.asarray(rollout["center_x"], dtype=np.float32)
        center_y = np.asarray(rollout["center_y"], dtype=np.float32)
        pred_heading_world = np.asarray(rollout["heading"], dtype=np.float32)
        valid_probability = np.asarray(rollout["valid_probability"], dtype=np.float32)
        agent_ids = np.asarray(rollout["agent_ids"], dtype=np.int64)
        current_valid = np.asarray(rollout["current_valid"], dtype=bool)
        focus_track_id = int(rollout["focus_track_id"])
        partner_track_id = int(rollout["partner_track_id"])
        scenario_id = str(rollout["scenario_id"])
        source_npz_path = Path(str(rollout["source_npz_path"])).resolve()
        oracle_focus = bool(rollout["oracle_focus"])

    expected = (ROLLOUTS, FUTURE_STEPS, agent_ids.shape[0])
    if center_x.shape != expected or center_y.shape != expected:
        raise ValueError(f"Expected rollout coordinates {expected}, got {center_x.shape} and {center_y.shape}")
    if not oracle_focus:
        raise ValueError("This visualization expects an oracle-focus rollout")

    with np.load(source_npz_path, allow_pickle=False) as source:
        gt_agents = np.asarray(source["agents"], dtype=np.float32)
        source_agent_ids = np.asarray(source["agent_ids"], dtype=np.int64)
        agent_mask = np.asarray(source["agent_mask"], dtype=bool)
        map_polylines = np.asarray(source["map_polylines"], dtype=np.float32)
        map_mask = np.asarray(source["map_mask"], dtype=bool)
        origin_xy = np.asarray(source["ego_origin_xy"], dtype=np.float32)
        frame_heading = float(source["ego_heading"])
    if not np.array_equal(agent_ids, source_agent_ids):
        raise ValueError("Rollout and source NPZ agent IDs do not match")

    pred_xy_world = np.stack([center_x, center_y], axis=-1)
    pred_xy = world_to_local_xy(pred_xy_world, origin_xy, frame_heading)
    pred_heading = wrap_angle(pred_heading_world - np.float32(frame_heading))
    slots = np.flatnonzero(current_valid & agent_mask)
    focus_matches = np.flatnonzero(agent_ids == focus_track_id)
    if focus_matches.size != 1:
        raise ValueError(f"Expected one focus ID {focus_track_id}, found slots {focus_matches.tolist()}")
    focus_slot = int(focus_matches[0])
    nonfocus_slots = slots[slots != focus_slot]
    if nonfocus_slots.size == 0:
        raise ValueError("No non-focus current-valid agents")

    xy = pred_xy[:, :, nonfocus_slots]
    yaw = pred_heading[:, :, nonfocus_slots]
    pairwise = pairwise_distances(xy)
    pair_indices = [
        (first, second)
        for first in range(xy.shape[0])
        for second in range(first + 1, xy.shape[0])
    ]
    pairwise_trajectory = pairwise.mean(axis=(1, 2))
    same_microbatch = np.asarray([
        first // int(args.rollout_batch_size) == second // int(args.rollout_batch_size)
        for first, second in pair_indices
    ])
    spatial_std = np.sqrt(np.var(xy, axis=0).sum(axis=-1))
    yaw_std = circular_std(yaw, axis=0)
    ensemble_mean = xy.mean(axis=0)
    rollout_distance_to_mean = np.linalg.norm(xy - ensemble_mean[None], axis=-1).mean(axis=(1, 2))

    gt_future = np.transpose(gt_agents[nonfocus_slots, FUTURE_SLICE], (1, 0, 2))
    gt_valid = gt_future[..., 5] > 0.5
    errors = np.linalg.norm(xy - gt_future[None, ..., :2], axis=-1)
    rollout_ade = np.asarray([
        float(error[gt_valid].mean()) if gt_valid.any() else float("nan")
        for error in errors
    ])

    focus_gt = gt_agents[focus_slot, FUTURE_SLICE, :2]
    focus_max_oracle_error = float(np.linalg.norm(pred_xy[:, :, focus_slot] - focus_gt[None], axis=-1).max())
    if focus_max_oracle_error > 1e-3:
        raise ValueError(f"Focus is not exactly oracle: max error={focus_max_oracle_error}")

    horizon_indices = {"1s": 9, "3s": 29, "5s": 49, "8s": 79}
    horizon_metrics = {}
    for label, index in horizon_indices.items():
        horizon_metrics[label] = {
            "mean_pairwise_distance_m": float(pairwise[:, index].mean()),
            "mean_spatial_std_m": float(spatial_std[index].mean()),
            "p90_agent_spatial_std_m": float(np.percentile(spatial_std[index], 90)),
            "max_agent_spatial_std_m": float(spatial_std[index].max()),
        }

    # Full-sequence decoder chunks are [0,32), [30,62), [59,91).  The
    # stitched future starts at original frame 11, so newly kept chunks begin
    # at future indices 21 and 51 (2.2 s and 5.2 s on a 10 Hz timeline).
    mean_std_curve = spatial_std.mean(axis=1)
    mean_step_displacement = np.linalg.norm(np.diff(xy, axis=1), axis=-1).mean(axis=(0, 2))
    chunk_seams = {}
    for seam_index in (21, 51):
        before_std = float(mean_std_curve[seam_index - 1])
        at_std = float(mean_std_curve[seam_index])
        previous_step = float(mean_step_displacement[seam_index - 2])
        seam_step = float(mean_step_displacement[seam_index - 1])
        chunk_seams[f"future_index_{seam_index}"] = {
            "original_frame_index_zero_based": seam_index + 11,
            "prediction_time_s": (seam_index + 1) / 10.0,
            "mean_spatial_std_before_m": before_std,
            "mean_spatial_std_at_seam_m": at_std,
            "spatial_std_jump_m": at_std - before_std,
            "spatial_std_jump_ratio": at_std / max(before_std, 1e-8),
            "mean_position_step_before_seam_m": previous_step,
            "mean_position_step_across_seam_m": seam_step,
            "position_step_jump_ratio": seam_step / max(previous_step, 1e-8),
        }

    per_agent = []
    for local_index, slot in enumerate(nonfocus_slots):
        endpoint_distances = pairwise[:, -1, local_index]
        endpoint_xy = xy[:, -1, local_index]
        covariance = np.cov(endpoint_xy.T, ddof=0)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        per_agent.append({
            "slot": int(slot),
            "agent_id": int(agent_ids[slot]),
            "mean_trajectory_pairwise_distance_m": float(pairwise[:, :, local_index].mean()),
            "mean_endpoint_pairwise_distance_m": float(endpoint_distances.mean()),
            "max_endpoint_pairwise_distance_m": float(endpoint_distances.max()),
            "endpoint_spatial_std_m": float(spatial_std[-1, local_index]),
            "endpoint_covariance_eigenvalues_m2": eigenvalues.astype(float).tolist(),
            "mean_yaw_circular_std_rad": float(yaw_std[:, local_index].mean()),
            "endpoint_yaw_circular_std_rad": float(yaw_std[-1, local_index]),
            "mean_valid_probability": float(valid_probability[:, :, slot].mean()),
            "valid_probability_std": float(valid_probability[:, :, slot].std()),
            "gt_valid_future_steps": int(gt_valid[:, local_index].sum()),
        })
    per_agent.sort(key=lambda row: row["mean_trajectory_pairwise_distance_m"], reverse=True)
    top_slots = [int(row["slot"]) for row in per_agent[: int(args.top_agents)]]
    top_ids = [int(row["agent_id"]) for row in per_agent[: int(args.top_agents)]]

    summary: dict[str, object] = {
        "scenario_id": scenario_id,
        "source_npz": str(source_npz_path),
        "rollout_npz": str(rollout_path),
        "protocol": "oracle_focus_local_32slot_32rollouts",
        "num_rollouts": ROLLOUTS,
        "future_steps": FUTURE_STEPS,
        "focus_track_id": focus_track_id,
        "partner_track_id": partner_track_id,
        "focus_slot": focus_slot,
        "focus_max_oracle_position_error_m": focus_max_oracle_error,
        "num_current_valid_agents": int(slots.size),
        "num_nonfocus_current_valid_agents": int(nonfocus_slots.size),
        "diversity_scope": "all nonfocus current-valid selected agents; predicted future is not validity-masked",
        "mean_trajectory_pairwise_distance_m": float(pairwise.mean()),
        "median_trajectory_pairwise_distance_m": float(np.median(pairwise)),
        "p90_trajectory_pairwise_distance_m": float(np.percentile(pairwise, 90)),
        "mean_endpoint_pairwise_distance_m": float(pairwise[:, -1].mean()),
        "p90_endpoint_pairwise_distance_m": float(np.percentile(pairwise[:, -1], 90)),
        "max_endpoint_pairwise_distance_m": float(pairwise[:, -1].max()),
        "mean_spatial_std_over_time_agents_m": float(spatial_std.mean()),
        "mean_endpoint_spatial_std_m": float(spatial_std[-1].mean()),
        "mean_yaw_circular_std_rad": float(yaw_std.mean()),
        "mean_endpoint_yaw_circular_std_rad": float(yaw_std[-1].mean()),
        "num_exact_unique_nonfocus_joint_trajectories": int(
            np.unique(xy.reshape(ROLLOUTS, -1), axis=0).shape[0]
        ),
        "pairwise_mean_trajectory_distance_by_rollout_pair_m": {
            "min": float(pairwise_trajectory.min()),
            "mean": float(pairwise_trajectory.mean()),
            "max": float(pairwise_trajectory.max()),
            "same_microbatch_mean": float(pairwise_trajectory[same_microbatch].mean()),
            "different_microbatch_mean": float(pairwise_trajectory[~same_microbatch].mean()),
            "rollout_batch_size": int(args.rollout_batch_size),
        },
        "rollout_nonfocus_ade_to_gt_m": {
            "mean": float(np.nanmean(rollout_ade)),
            "std": float(np.nanstd(rollout_ade)),
            "min": float(np.nanmin(rollout_ade)),
            "max": float(np.nanmax(rollout_ade)),
            "per_rollout": rollout_ade.astype(float).tolist(),
        },
        "rollout_mean_distance_to_ensemble_mean_m": rollout_distance_to_mean.astype(float).tolist(),
        "horizons": horizon_metrics,
        "decoder_chunk_seams": chunk_seams,
        "top_diverse_agent_ids": top_ids,
        "per_agent": per_agent,
    }

    bounds = scene_bounds(gt_agents, slots, float(args.margin_m))
    palette = plt.get_cmap("tab20")
    color_by_slot = {int(slot): palette(index % 20) for index, slot in enumerate(slots)}
    color_by_slot[focus_slot] = "#25d13b"
    partner_matches = np.flatnonzero(agent_ids == partner_track_id)
    partner_slot = int(partner_matches[0]) if partner_matches.size == 1 else -1
    if partner_slot >= 0:
        color_by_slot[partner_slot] = "#ff9d22"

    image_paths = []
    for rollout_index in range(ROLLOUTS):
        output = images_dir / f"rollout_{rollout_index:02d}.png"
        fig, ax = plt.subplots(figsize=(9, 9), dpi=int(args.dpi))
        fig.patch.set_facecolor("#111111")
        setup_axis(ax, bounds)
        draw_map(ax, map_polylines, map_mask)
        for slot in slots:
            slot = int(slot)
            color = color_by_slot[slot]
            history = gt_agents[slot, 1 : CURRENT_INDEX + 1]
            history_valid = history[:, 5] > 0.5
            future = gt_agents[slot, FUTURE_SLICE]
            future_valid = future[:, 5] > 0.5
            if history_valid.any():
                ax.plot(history[history_valid, 0], history[history_valid, 1], color=color, linewidth=1.1, alpha=0.7)
            if future_valid.any():
                ax.plot(future[future_valid, 0], future[future_valid, 1], color="#d8d8d8", linewidth=0.8, alpha=0.38, linestyle=":")
            width = 2.6 if slot in {focus_slot, partner_slot} else 1.25
            alpha = 1.0 if slot in {focus_slot, partner_slot} else 0.82
            ax.plot(pred_xy[rollout_index, :, slot, 0], pred_xy[rollout_index, :, slot, 1], color=color, linewidth=width, alpha=alpha, linestyle="--")
            ax.scatter(pred_xy[rollout_index, -1, slot, 0], pred_xy[rollout_index, -1, slot, 1], s=18, color=color, zorder=4)
            if slot in top_slots or slot in {focus_slot, partner_slot}:
                ax.text(pred_xy[rollout_index, -1, slot, 0], pred_xy[rollout_index, -1, slot, 1], str(int(agent_ids[slot])), color=color, fontsize=7)
        ax.set_title(
            f"WOSAC rollout {rollout_index:02d}/31 · scenario={scenario_id}\n"
            f"oracle focus={focus_track_id} (green), partner={partner_track_id} (orange) · nonfocus ADE={rollout_ade[rollout_index]:.2f}m",
            color="#eeeeee",
            fontsize=11,
        )
        fig.text(
            0.02,
            0.015,
            f"dashed=model, dotted gray=GT future | rollout distance to ensemble mean={rollout_distance_to_mean[rollout_index]:.2f}m",
            color="#eeeeee",
            fontsize=8,
        )
        fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        image_paths.append(output)

    overlay_path = output_dir / "all_32_rollouts_overlay.png"
    fig, ax = plt.subplots(figsize=(10, 10), dpi=int(args.dpi))
    fig.patch.set_facecolor("#111111")
    setup_axis(ax, bounds)
    draw_map(ax, map_polylines, map_mask)
    for slot in top_slots:
        color = color_by_slot[slot]
        local_index = int(np.flatnonzero(nonfocus_slots == slot)[0])
        for rollout_index in range(ROLLOUTS):
            ax.plot(xy[rollout_index, :, local_index, 0], xy[rollout_index, :, local_index, 1], color=color, linewidth=0.8, alpha=0.22)
            ax.scatter(xy[rollout_index, -1, local_index, 0], xy[rollout_index, -1, local_index, 1], color=color, s=8, alpha=0.45)
        gt = gt_agents[slot, FUTURE_SLICE]
        valid = gt[:, 5] > 0.5
        if valid.any():
            ax.plot(gt[valid, 0], gt[valid, 1], color=color, linewidth=2.5, linestyle="--", label=f"id={int(agent_ids[slot])}")
    focus = gt_agents[focus_slot, 1:91]
    focus_valid = focus[:, 5] > 0.5
    ax.plot(focus[focus_valid, 0], focus[focus_valid, 1], color="#25d13b", linewidth=3.0, label=f"oracle focus={focus_track_id}")
    ax.set_title(
        f"All 32 WOSAC rollouts · top-{len(top_slots)} most diverse nonfocus agents\n"
        f"mean pairwise trajectory distance={pairwise.mean():.2f}m · endpoint={pairwise[:, -1].mean():.2f}m",
        color="#eeeeee",
        fontsize=12,
    )
    legend = ax.legend(loc="best", fontsize=7, facecolor="#222222", edgecolor="#777777")
    for text in legend.get_texts():
        text.set_color("#eeeeee")
    fig.savefig(overlay_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    curve_path = output_dir / "diversity_over_time.png"
    seconds = (np.arange(FUTURE_STEPS) + 1) / 10.0
    fig, ax = plt.subplots(figsize=(9, 5), dpi=int(args.dpi))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#171717")
    mean_curve = spatial_std.mean(axis=1)
    p25 = np.percentile(spatial_std, 25, axis=1)
    p75 = np.percentile(spatial_std, 75, axis=1)
    p90 = np.percentile(spatial_std, 90, axis=1)
    ax.fill_between(seconds, p25, p75, color="#3299ff", alpha=0.22, label="agent p25-p75")
    ax.plot(seconds, mean_curve, color="#55aaff", linewidth=2.2, label="mean spatial std")
    ax.plot(seconds, p90, color="#ff9d22", linewidth=1.4, label="agent p90")
    for seam_index, seam_color in ((21, "#ff5e5e"), (51, "#d88cff")):
        seam_time = (seam_index + 1) / 10.0
        ax.axvline(seam_time, color=seam_color, linewidth=1.1, linestyle="--", alpha=0.85)
        ax.text(
            seam_time + 0.04,
            float(p90.max()) * 0.78,
            f"decoder seam {seam_time:.1f}s",
            color=seam_color,
            fontsize=8,
            rotation=90,
            va="center",
        )
    ax.set_xlabel("prediction horizon (s)", color="#eeeeee")
    ax.set_ylabel("spatial std across 32 rollouts (m)", color="#eeeeee")
    ax.set_title("Stochastic trajectory spread over time (nonfocus current-valid agents)", color="#eeeeee")
    ax.grid(color="#555555", alpha=0.4)
    ax.tick_params(colors="#dddddd")
    legend = ax.legend(facecolor="#222222", edgecolor="#777777")
    for text in legend.get_texts():
        text.set_color("#eeeeee")
    fig.savefig(curve_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    contact_sheet = output_dir / "contact_sheet_32_rollouts.png"
    make_contact_sheet(image_paths, contact_sheet)
    summary_path = output_dir / "randomness_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    per_agent_path = output_dir / "per_agent_randomness.csv"
    csv_fields = [
        "slot", "agent_id", "mean_trajectory_pairwise_distance_m",
        "mean_endpoint_pairwise_distance_m", "max_endpoint_pairwise_distance_m",
        "endpoint_spatial_std_m", "mean_yaw_circular_std_rad",
        "endpoint_yaw_circular_std_rad", "mean_valid_probability",
        "valid_probability_std", "gt_valid_future_steps",
    ]
    with per_agent_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_agent)

    print(f"wrote_individual_images={len(image_paths)} dir={images_dir}", flush=True)
    print(f"wrote_contact_sheet={contact_sheet}", flush=True)
    print(f"wrote_overlay={overlay_path}", flush=True)
    print(f"wrote_metrics={summary_path}", flush=True)
    print(json.dumps({key: summary[key] for key in (
        "mean_trajectory_pairwise_distance_m",
        "mean_endpoint_pairwise_distance_m",
        "mean_spatial_std_over_time_agents_m",
        "mean_endpoint_spatial_std_m",
        "top_diverse_agent_ids",
    )}, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize stochastic diversity in one local-WOSAC rollout NPZ.")
    parser.add_argument("--rollout_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_agents", type=int, default=8)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    parser.add_argument("--margin_m", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=140)
    return parser


if __name__ == "__main__":
    visualize(build_parser().parse_args())
