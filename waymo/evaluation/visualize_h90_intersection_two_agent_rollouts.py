#!/usr/bin/env python3
"""Select intersection-rich scenes and overlay one target agent's N rollouts.

The focus/ego follows its recorded future plan and is drawn exactly once.  A
single non-focus agent with a long, fully observed future is selected in each
scene; its ground truth is drawn once and all stochastic model rollouts are
overlaid in the same panel.  No other agent trajectories are displayed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.core.vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo.data_prep.analyze_waymo_vector_dataset import (  # noqa: E402
    analyze_file,
    default_thresholds,
)
from waymo.evaluation import eval_waymo_world_model_horizons as base_eval  # noqa: E402
from waymo.evaluation.visualize_vector_tokenizer_reconstruction import _draw_map  # noqa: E402
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


AGENT_TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}


def _path_stats(agent_tf: np.ndarray, start: int, end: int) -> dict[str, float | int]:
    """Motion statistics without joining gaps between disjoint valid spans."""
    state = agent_tf[start:end]
    valid = state[:, 5] > 0.5
    xy = state[:, 0:2]
    if len(state) > 1:
        adjacent = valid[:-1] & valid[1:]
        step_distance = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
        path_length = float(step_distance[adjacent].sum())
    else:
        path_length = 0.0
    valid_indices = np.flatnonzero(valid)
    displacement = (
        float(np.linalg.norm(xy[valid_indices[-1]] - xy[valid_indices[0]]))
        if len(valid_indices) >= 2
        else 0.0
    )
    return {
        "valid_steps": int(valid.sum()),
        "path_length_m": path_length,
        "displacement_m": displacement,
    }


def _select_target_agent(
    item: dict[str, np.ndarray],
    *,
    context_frames: int,
    total_frames: int,
    min_future_valid_steps: int,
) -> tuple[int, dict[str, float | int]] | None:
    agents = item["agents"]
    agent_mask = item["agent_mask"].astype(bool)
    current = int(context_frames) - 1
    candidates: list[tuple[tuple[float, ...], int, dict[str, float | int]]] = []
    for slot in range(1, int(agents.shape[0])):
        if not agent_mask[slot] or agents[slot, current, 5] <= 0.5:
            continue
        stats = _path_stats(agents[slot], current, total_frames)
        future_valid = int((agents[slot, context_frames:total_frames, 5] > 0.5).sum())
        if future_valid < int(min_future_valid_steps):
            continue
        stats["future_valid_steps"] = future_valid
        score = (
            float(stats["path_length_m"]),
            float(future_valid),
            float(stats["displacement_m"]),
        )
        candidates.append((score, slot, stats))
    if not candidates:
        return None
    _, slot, stats = max(candidates, key=lambda row: row[0])
    return slot, stats


def select_scenes(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(args.subset_manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    records = list(manifest["samples"])
    if len(records) != int(args.source_subset_size):
        raise ValueError(
            f"Expected {args.source_subset_size} source samples, found {len(records)} in {manifest_path}"
        )

    thresholds = default_thresholds()
    total_frames = int(args.eval_ctx) + int(args.horizon)
    ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    seen_scenarios: set[str] = set()
    for record in records:
        scenario_id = str(record["scenario_id"])
        if scenario_id in seen_scenarios:
            continue
        seen_scenarios.add(scenario_id)
        npz_path = Path(record["path"]).resolve()
        analysis = analyze_file(str(npz_path), thresholds)
        with np.load(npz_path, allow_pickle=False) as source:
            item = {key: source[key] for key in source.files}
        if int(item["agents"].shape[1]) < total_frames:
            continue
        ego_future_valid = int(
            (item["agents"][0, args.eval_ctx:total_frames, 5] > 0.5).sum()
        )
        if ego_future_valid < int(args.min_future_valid_steps):
            continue
        target = _select_target_agent(
            item,
            context_frames=args.eval_ctx,
            total_frames=total_frames,
            min_future_valid_steps=args.min_future_valid_steps,
        )
        if target is None:
            continue
        target_slot, target_stats = target
        target_id = int(item["agent_ids"][target_slot])
        target_type = int(round(float(item["agents"][target_slot, args.eval_ctx - 1, 7])))

        maneuver_is_turn = analysis["ego_maneuver_label"] in {
            "left_turn",
            "right_turn",
            "large_turn_or_uturn",
        }
        crossing_count = int(analysis["num_crossing_close_agents"])
        close_count = int(analysis["num_close_agents_20m"])
        # First require an intersection-like setting.  Among those, prioritize
        # traffic-controlled turns with crossing traffic, then a long target
        # trajectory so the ten rollout curves are visually informative.
        rank_key = (
            float(bool(analysis["intersection_like"])),
            float(bool(analysis["has_traffic_lights"] and maneuver_is_turn)),
            float(bool(crossing_count > 0)),
            float(bool(analysis["has_traffic_lights"])),
            float(maneuver_is_turn),
            float(min(crossing_count, 12)),
            float(min(close_count, 20)),
            float(target_stats["path_length_m"]),
        )
        selected_record = dict(record)
        selected_record.update(
            {
                "source_manifest": str(manifest_path),
                "intersection_like": bool(analysis["intersection_like"]),
                "has_traffic_lights": bool(analysis["has_traffic_lights"]),
                "ego_maneuver_label": str(analysis["ego_maneuver_label"]),
                "ego_heading_change_deg": float(analysis["ego_heading_change_deg"]),
                "num_crossing_close_agents": crossing_count,
                "num_close_agents_20m": close_count,
                "interaction_labels": str(analysis["interaction_labels"]),
                "ego_future_valid_steps": ego_future_valid,
                "target_slot": int(target_slot),
                "target_track_id": target_id,
                "target_type_id": target_type,
                "target_type": AGENT_TYPE_NAMES.get(target_type, f"type_{target_type}"),
                **{f"target_{key}": value for key, value in target_stats.items()},
            }
        )
        ranked.append((rank_key, selected_record))

    ranked.sort(key=lambda row: row[0], reverse=True)
    selected = [record for _, record in ranked[: int(args.num_scenes)]]
    if len(selected) != int(args.num_scenes):
        raise RuntimeError(
            f"Only {len(selected)} scenes satisfy the selection criteria; requested {args.num_scenes}"
        )
    selection_summary = {
        "source_manifest": str(manifest_path),
        "source_sample_count": len(records),
        "source_unique_scenario_count": len(seen_scenarios),
        "selection": (
            "unique scenarios ranked by intersection-like evidence, traffic-controlled ego turn, "
            "crossing/close traffic, then longest fully observed non-focus trajectory"
        ),
        "min_future_valid_steps": int(args.min_future_valid_steps),
        "selected_count": len(selected),
        "samples": selected,
    }
    return selected, selection_summary


def _square_bounds(point_sets: list[np.ndarray], margin: float) -> tuple[float, float, float, float]:
    finite_parts = []
    for points in point_sets:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        finite = np.isfinite(points).all(axis=-1)
        if finite.any():
            finite_parts.append(points[finite])
    if not finite_parts:
        return (-50.0, 50.0, -50.0, 50.0)
    all_points = np.concatenate(finite_parts, axis=0)
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    center = 0.5 * (low + high)
    half = 0.5 * max(float((high - low).max()), 20.0) + float(margin)
    return (
        float(center[0] - half),
        float(center[0] + half),
        float(center[1] - half),
        float(center[1] + half),
    )


def _rollout_metrics(
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    errors = np.linalg.norm(pred_xy - gt_xy[None], axis=-1)
    per_ade = errors[:, valid].mean(axis=1)
    last = int(np.flatnonzero(valid)[-1])
    per_fde = errors[:, last]
    pair_trajectory: list[float] = []
    pair_endpoint: list[float] = []
    for left in range(int(pred_xy.shape[0])):
        for right in range(left + 1, int(pred_xy.shape[0])):
            distance = np.linalg.norm(pred_xy[left] - pred_xy[right], axis=-1)
            pair_trajectory.append(float(distance[valid].mean()))
            pair_endpoint.append(float(distance[last]))
    return {
        "per_rollout_ade_m": per_ade.astype(float).tolist(),
        "per_rollout_fde_m": per_fde.astype(float).tolist(),
        "mean_ade_m": float(per_ade.mean()),
        "min_ade_m": float(per_ade.min()),
        "max_ade_m": float(per_ade.max()),
        "mean_fde_m": float(per_fde.mean()),
        "mean_pairwise_trajectory_distance_m": float(np.mean(pair_trajectory)),
        "mean_pairwise_endpoint_distance_m": float(np.mean(pair_endpoint)),
    }


def _plot_scene(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    gt_tkf: np.ndarray,
    pred_xy: np.ndarray,
    map_polylines: np.ndarray,
    map_mask: np.ndarray,
    agent_ids: np.ndarray,
    output_path: Path,
    bounds_override: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    context_frames = int(args.eval_ctx)
    target_slot = int(record["target_slot"])
    focus_slot = 0
    future_slice = slice(context_frames - 1, context_frames + int(args.horizon))
    ego_valid = gt_tkf[future_slice, focus_slot, 5] > 0.5
    target_valid = gt_tkf[future_slice, target_slot, 5] > 0.5
    ego_xy = gt_tkf[future_slice, focus_slot, 0:2]
    target_gt_xy = gt_tkf[future_slice, target_slot, 0:2]
    target_pred_xy = pred_xy[:, future_slice, target_slot]

    bounds = bounds_override or _square_bounds(
        [ego_xy[ego_valid], target_gt_xy[target_valid], target_pred_xy[:, target_valid]],
        margin=float(args.margin_m),
    )
    fig, ax = plt.subplots(figsize=(9.2, 9.2), dpi=int(args.dpi))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#202020")
    _draw_map(ax, map_polylines, map_mask)

    # The supplied ego plan and target ground truth each appear exactly once.
    ax.plot(
        ego_xy[ego_valid, 0],
        ego_xy[ego_valid, 1],
        color="#ff4d4d",
        linewidth=3.1,
        label="provided ego GT plan",
        zorder=8,
    )
    ax.plot(
        target_gt_xy[target_valid, 0],
        target_gt_xy[target_valid, 1],
        color="#ffffff",
        linewidth=2.8,
        linestyle="--",
        label=f"target GT (id={int(agent_ids[target_slot])})",
        zorder=9,
    )
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, int(pred_xy.shape[0])))
    for rollout_index, color in enumerate(colors):
        curve = target_pred_xy[rollout_index]
        ax.plot(
            curve[target_valid, 0],
            curve[target_valid, 1],
            color=color,
            linewidth=1.65,
            alpha=0.82,
            label="10 target-agent rollouts" if rollout_index == 0 else None,
            zorder=6,
        )

    current = 0
    ax.scatter(
        ego_xy[current, 0], ego_xy[current, 1], s=70, color="#ff4d4d",
        edgecolors="black", linewidths=0.8, zorder=12,
    )
    ax.scatter(
        target_gt_xy[current, 0], target_gt_xy[current, 1], s=70, color="#ffffff",
        edgecolors="black", linewidths=0.8, zorder=12,
    )
    ax.text(
        ego_xy[current, 0] + 1.2, ego_xy[current, 1] + 1.2, "ego @ context",
        color="#ff9a9a", fontsize=8, zorder=13,
    )
    ax.text(
        target_gt_xy[current, 0] + 1.2, target_gt_xy[current, 1] + 1.2,
        f"target k{target_slot}", color="white", fontsize=8, zorder=13,
    )

    xmin, xmax, ymin, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#555555", alpha=0.28, linewidth=0.5)
    ax.tick_params(colors="#dddddd", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#888888")
    ax.set_xlabel("local x (m)", color="#dddddd")
    ax.set_ylabel("local y (m)", color="#dddddd")
    model_label = str(getattr(args, "model_label", "")).strip()
    title_prefix = f"{model_label}\n" if model_label else ""
    ax.set_title(
        title_prefix
        + f"scenario {record['scenario_id']} | {record['ego_maneuver_label']} | "
        f"crossing={record['num_crossing_close_agents']}\n"
        f"recorded ego plan + target agent GT + {pred_xy.shape[0]} stochastic H{args.horizon} rollouts",
        color="white",
        fontsize=11,
    )
    legend = ax.legend(loc="best", facecolor="#161616", edgecolor="#777777", fontsize=8)
    for text_item in legend.get_texts():
        text_item.set_color("white")

    metrics = _rollout_metrics(target_gt_xy, target_pred_xy, target_valid)
    fig.text(
        0.02,
        0.012,
        f"target id={int(agent_ids[target_slot])} ({record['target_type']}) | "
        f"GT path={record['target_path_length_m']:.1f} m | "
        f"mean/min ADE={metrics['mean_ade_m']:.2f}/{metrics['min_ade_m']:.2f} m | "
        f"pairwise trajectory distance={metrics['mean_pairwise_trajectory_distance_m']:.2f} m",
        color="#eeeeee",
        fontsize=8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return metrics


def _make_contact_sheet(images: list[Path], output_path: Path) -> None:
    thumbs: list[Image.Image] = []
    for path in images:
        with Image.open(path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((620, 620), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (640, 660), (15, 15, 15))
        canvas.paste(thumb, ((canvas.width - thumb.width) // 2, 5))
        ImageDraw.Draw(canvas).text((10, 638), path.stem, fill=(235, 235, 235))
        thumbs.append(canvas)
    columns = 2
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 640, rows * 660), (10, 10, 10))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 640, (index // columns) * 660))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


@torch.no_grad()
def visualize(args: argparse.Namespace) -> None:
    if int(args.eval_ctx) != 11 or int(args.horizon) != 80:
        raise ValueError("This h90 visualization expects eval_ctx=11 and horizon=80")
    if int(args.eval_num_rollouts) != 10:
        raise ValueError("This visualization expects exactly 10 rollouts per scene")
    if args.eval_schedule != "shortcut" or float(args.eval_d) != 1.0:
        raise ValueError("This visualization uses the established shortcut D1 protocol")
    if not args.use_ego_actions or args.ego_action_source != "focus":
        raise ValueError("Pass --use_ego_actions --ego_action_source focus")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected, selection_summary = select_scenes(args)
    selection_path = output_dir / "selected_intersection10_manifest.json"
    selection_path.write_text(json.dumps(selection_summary, indent=2, sort_keys=True) + "\n")
    if args.selection_only:
        print(f"wrote_selection={selection_path}", flush=True)
        return

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(int(args.eval_multisample_seed))
    dataset = wm.WaymoVectorDataset(args.val_data_dir)
    indices = [int(record["dataset_index"]) for record in selected]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=int(args.num_workers) > 0,
        worker_init_fn=wm.worker_init_fn,
        collate_fn=wm._collate,
    )

    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    if isinstance(tokenizer, wm.FrozenWaymoFocusTokenizer):
        raise ValueError("Expected the vector tokenizer")
    n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
    d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % int(args.packing_factor):
        raise ValueError("n_latents must be divisible by packing_factor")
    args.n_spatial = n_latents // int(args.packing_factor)
    args.d_spatial = d_bottleneck * int(args.packing_factor)

    checkpoint = torch.load(args.eval_ckpt, map_location="cpu")
    dyn = base_eval.build_dynamics(
        args,
        d_bottleneck,
        device,
        map_memory_dim=(
            wm.tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None
        ),
    )
    base_eval.load_dynamics_state(dyn, args.eval_ckpt, ckpt=checkpoint)
    dyn.eval()
    schedule = wm.make_tau_schedule(
        k_max=args.k_max,
        schedule=args.eval_schedule,
        d=args.eval_d,
    )

    total_frames = int(args.eval_ctx) + int(args.horizon)
    repeats = int(args.eval_num_rollouts)
    image_paths: list[Path] = []
    result_rows: list[dict[str, Any]] = []
    for scene_index, (record, batch) in enumerate(zip(selected, loader)):
        scene_seed = int(args.eval_multisample_seed) + int(record["sample_order"])
        wm.seed_everything(scene_seed)
        batch = wm.slice_time_window(
            wm.move_batch(batch, device), total_frames, random_start=False
        )
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        if actions is None or act_mask is None or action_slots is None:
            raise RuntimeError("The checkpoint did not produce recorded focus actions")
        if int(action_slots[0]) != 0:
            raise RuntimeError(f"Expected focus slot 0, got {int(action_slots[0])}")

        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            tokenizer,
            batch,
            args,
            return_map=args.dynamics_attend_map,
        )
        z_gt_packed = wm.pack_bottleneck_to_spatial(
            z_gt,
            n_spatial=args.n_spatial,
            k=args.packing_factor,
        )

        def repeat(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.repeat_interleave(repeats, dim=0)

        rollout_batch = wm.repeat_batch_candidates(batch, repeats)
        z_pred_packed = wm.sample_autoregressive_packed_sequence(
            wm.unwrap_model(dyn),
            z_gt_packed=repeat(z_gt_packed),
            actions=repeat(actions),
            act_mask=repeat(act_mask),
            map_tokens=repeat(map_tokens),
            map_mask=repeat(map_mask),
            ctx_length=args.eval_ctx,
            horizon=args.horizon,
            k_max=args.k_max,
            sched=schedule,
            max_rollout_window=args.max_rollout_window,
        )
        z_pred = wm.unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
        decoded = wm.decode_batch_z_for_world_model(tokenizer, z_pred, rollout_batch, args)
        anchor_xy = None
        if args.agent_xy_parameterization == "delta":
            anchor_xy = wm.agents_to_btkf(
                rollout_batch["agents"], rollout_batch["agent_mask"]
            )[:, 0, :, 0:2]
        pred_xy = decoder_agent_xy(
            decoded,
            args.agent_xy_loss,
            args.agent_xy_parameterization,
            anchor_xy=anchor_xy,
        ).detach().float().cpu().numpy()

        gt_tkf = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])[0].detach().float().cpu().numpy()
        pred_xy[:, : args.eval_ctx] = gt_tkf[None, : args.eval_ctx, :, 0:2]
        agent_ids = batch["agent_ids"][0].detach().cpu().numpy()
        expected_target_id = int(record["target_track_id"])
        actual_target_id = int(agent_ids[int(record["target_slot"])])
        if actual_target_id != expected_target_id:
            raise RuntimeError(
                f"Target mismatch for {record['scenario_id']}: expected {expected_target_id}, "
                f"loaded {actual_target_id}"
            )
        map_polylines = batch["map_polylines"][0].detach().float().cpu().numpy()
        map_mask_np = batch["map_mask"][0].detach().cpu().numpy().astype(bool)
        image_path = output_dir / f"scene_{scene_index:02d}_{record['scenario_id']}.png"
        metrics = _plot_scene(
            args=args,
            record=record,
            gt_tkf=gt_tkf,
            pred_xy=pred_xy,
            map_polylines=map_polylines,
            map_mask=map_mask_np,
            agent_ids=agent_ids,
            output_path=image_path,
        )
        np.savez_compressed(
            output_dir / f"scene_{scene_index:02d}_{record['scenario_id']}_trajectories.npz",
            ego_gt_xy=gt_tkf[:, 0, 0:2],
            ego_gt_valid=gt_tkf[:, 0, 5] > 0.5,
            target_gt_xy=gt_tkf[:, int(record["target_slot"]), 0:2],
            target_gt_valid=gt_tkf[:, int(record["target_slot"]), 5] > 0.5,
            target_rollout_xy=pred_xy[:, :, int(record["target_slot"])],
            context_frames=np.asarray(int(args.eval_ctx)),
            horizon=np.asarray(int(args.horizon)),
            scene_seed=np.asarray(scene_seed),
        )
        image_paths.append(image_path)
        result_rows.append(
            {
                **record,
                "scene_rank": scene_index,
                "scene_seed": scene_seed,
                "image": str(image_path),
                **metrics,
            }
        )
        print(
            f"generated {scene_index + 1}/{len(selected)} scenario={record['scenario_id']} "
            f"target={actual_target_id} mean_ADE={metrics['mean_ade_m']:.3f}m "
            f"pairwise={metrics['mean_pairwise_trajectory_distance_m']:.3f}m",
            flush=True,
        )

    contact_sheet = output_dir / "contact_sheet_intersection10_n10.png"
    _make_contact_sheet(image_paths, contact_sheet)
    payload = {
        "eval_ckpt": str(Path(args.eval_ckpt).resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "protocol": "ctx11_h80_shortcut_d1_recorded_focus_plan_target_agent_n10",
        "provided_ego_plan": "recorded ground-truth focus trajectory",
        "ego_draw_count_per_figure": 1,
        "target_ground_truth_draw_count_per_figure": 1,
        "target_rollout_count_per_figure": repeats,
        "other_agent_trajectories_drawn": 0,
        "selection_manifest": str(selection_path),
        "contact_sheet": str(contact_sheet),
        "samples": result_rows,
    }
    metrics_path = output_dir / "rollout_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote_selection={selection_path}", flush=True)
    print(f"wrote_metrics={metrics_path}", flush=True)
    print(f"wrote_contact_sheet={contact_sheet}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = base_eval.add_eval_args(wm.build_argparser())
    parser.description = "Overlay ten h90 rollouts for one long target agent in ten intersections."
    parser.add_argument("--subset_manifest", required=True)
    parser.add_argument("--source_subset_size", type=int, default=128)
    parser.add_argument("--num_scenes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--min_future_valid_steps", type=int, default=80)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--margin_m", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--selection_only", action="store_true")
    return parser


if __name__ == "__main__":
    visualize(build_parser().parse_args())
