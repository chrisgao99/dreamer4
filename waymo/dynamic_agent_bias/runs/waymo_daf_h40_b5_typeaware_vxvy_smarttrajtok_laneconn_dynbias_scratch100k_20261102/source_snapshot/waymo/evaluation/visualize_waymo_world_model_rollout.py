#!/usr/bin/env python3
"""Visualize GT and legacy World Model rollout trajectories on a fixed manifest."""

from __future__ import annotations

import argparse
import json
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
from waymo.evaluation import eval_waymo_world_model_horizons as base_eval  # noqa: E402
from waymo.evaluation.visualize_vector_tokenizer_reconstruction import (  # noqa: E402
    _draw_panel,
    _select_agents,
    _valid_bounds,
)
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


def _trajectory_metrics(
    gt_tkf: np.ndarray,
    pred_xy: np.ndarray,
    agent_mask: np.ndarray,
    *,
    focus_slot: int,
    context_frames: int,
) -> dict[str, float]:
    steps = min(int(gt_tkf.shape[0]), int(pred_xy.shape[0]))
    score_start = min(max(1, int(context_frames)), steps)
    gt_future = gt_tkf[score_start:steps]
    pred_future = pred_xy[score_start:steps]
    valid = (gt_future[..., 5] > 0.5) & agent_mask[None]
    error = np.linalg.norm(pred_future - gt_future[..., 0:2], axis=-1)
    if valid.any():
        all_ade = float(error[valid].mean())
    else:
        all_ade = float("nan")
    focus_valid = valid[:, focus_slot]
    if focus_valid.any():
        valid_indices = np.flatnonzero(focus_valid)
        focus_ade = float(error[focus_valid, focus_slot].mean())
        final_index = int(valid_indices[-1])
        focus_fde = float(error[final_index, focus_slot])
    else:
        focus_ade = float("nan")
        focus_fde = float("nan")
    return {
        "all_agent_ade_m": all_ade,
        "focus_ade_m": focus_ade,
        "focus_fde_m": focus_fde,
    }


def _agent_metric_lines(
    gt_tkf: np.ndarray,
    pred_xy: np.ndarray,
    agent_mask: np.ndarray,
    agent_ids: np.ndarray,
    selected_agents: list[int],
    *,
    context_frames: int,
) -> list[str]:
    steps = min(int(gt_tkf.shape[0]), int(pred_xy.shape[0]))
    score_start = min(max(1, int(context_frames)), steps)
    lines = []
    for k in selected_agents:
        valid = (
            (gt_tkf[score_start:steps, k, 5] > 0.5)
            & bool(agent_mask[k])
        )
        if not valid.any():
            continue
        error = np.linalg.norm(
            pred_xy[score_start:steps, k] - gt_tkf[score_start:steps, k, 0:2],
            axis=-1,
        )
        valid_indices = np.flatnonzero(valid)
        ade = float(error[valid].mean())
        fde = float(error[int(valid_indices[-1])])
        lines.append(f"k{k:02d} id={int(agent_ids[k])}: ADE={ade:.2f}m FDE={fde:.2f}m")
    return lines


def _make_contact_sheet(images: list[Path], output: Path) -> None:
    if not images:
        return
    thumbs = []
    for path in images:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((520, 520), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (540, 560), (20, 20, 20))
            x = (canvas.width - thumb.width) // 2
            canvas.paste(thumb, (x, 10))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 532), path.stem, fill=(235, 235, 235))
            thumbs.append(canvas)
    columns = 5
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 540, rows * 560), (12, 12, 12))
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % columns) * 540, (index // columns) * 560))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


@torch.no_grad()
def visualize(args: argparse.Namespace) -> None:
    total_frames = int(args.horizon) + 1
    if args.eval_ctx < 1 or args.eval_ctx >= total_frames:
        raise ValueError(
            f"eval_ctx must be in [1, {total_frames - 1}], got {args.eval_ctx}"
        )
    if args.eval_schedule != "shortcut" or float(args.eval_d) != 1.0:
        raise ValueError("This visualization protocol requires shortcut D1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.subset_manifest).read_text())
    records = manifest["samples"][: int(args.num_samples)]
    if len(records) != int(args.num_samples):
        raise ValueError(f"Manifest has only {len(records)} requested records")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(args.seed)
    dataset = wm.WaymoVectorDataset(args.val_data_dir)
    indices = [int(record["dataset_index"]) for record in records]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=wm.worker_init_fn,
        collate_fn=wm._collate,
    )

    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    if isinstance(tokenizer, wm.FrozenWaymoFocusTokenizer):
        raise ValueError("This checkpoint visualization expects the vector tokenizer")
    n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
    d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor:
        raise ValueError("n_latents must be divisible by packing_factor")
    args.n_spatial = n_latents // args.packing_factor
    args.d_spatial = d_bottleneck * args.packing_factor

    checkpoint = torch.load(args.eval_ckpt, map_location="cpu")
    dyn = base_eval.build_dynamics(
        args,
        d_bottleneck,
        device,
        map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None,
    )
    base_eval.load_dynamics_state(dyn, args.eval_ckpt, ckpt=checkpoint)
    dyn.eval()
    schedule = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    if int(schedule["K"]) != 1:
        raise ValueError("Expected exactly one D1 solver sample per predicted frame")

    # Match the recorded evaluation RNG after model construction.
    wm.seed_everything(args.seed)
    image_paths: list[Path] = []
    metric_rows: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for sample_index, (record, batch) in enumerate(zip(records, loader)):
        batch = wm.slice_time_window(
            wm.move_batch(batch, device), total_frames, random_start=False
        )
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        if actions is None or act_mask is None:
            raise ValueError("World Model rollout visualization requires ego actions")
        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            tokenizer, batch, args, return_map=args.dynamics_attend_map
        )
        z_gt_packed = wm.pack_bottleneck_to_spatial(
            z_gt, n_spatial=args.n_spatial, k=args.packing_factor
        )
        z_pred_packed = wm.sample_autoregressive_packed_sequence(
            wm.unwrap_model(dyn),
            z_gt_packed=z_gt_packed,
            actions=actions,
            act_mask=act_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            ctx_length=args.eval_ctx,
            horizon=total_frames - args.eval_ctx,
            k_max=args.k_max,
            sched=schedule,
            max_rollout_window=args.max_rollout_window,
        )
        z_pred = wm.unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
        decoded = wm.decode_batch_z_for_world_model(tokenizer, z_pred, batch, args)

        gt_tkf = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])[0].detach().float().cpu().numpy()
        anchor_xy = None
        if args.agent_xy_parameterization == "delta":
            anchor_xy = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])[:, 0, :, 0:2]
        pred_xy = decoder_agent_xy(
            decoded,
            args.agent_xy_loss,
            args.agent_xy_parameterization,
            anchor_xy=anchor_xy,
        )[0].detach().float().cpu().numpy()
        # Context latents come from GT. Display their physical coordinates
        # exactly, then show the decoded free-running rollout after context.
        pred_xy[: args.eval_ctx] = gt_tkf[: args.eval_ctx, :, 0:2]
        agent_mask_np = batch["agent_mask"][0].detach().cpu().numpy().astype(bool)
        map_polylines = batch["map_polylines"][0].detach().float().cpu().numpy()
        map_mask_np = batch["map_mask"][0].detach().cpu().numpy().astype(bool)
        agent_ids = batch["agent_ids"][0].detach().cpu().numpy()
        focus_slot = int(action_slots[0].item())
        # Hide the dashed line over the GT context, retaining the final context
        # point as the visual anchor for the first predicted segment.
        draw_pred_xy = pred_xy.copy()
        draw_pred_xy[: max(0, args.eval_ctx - 1)] = np.nan
        pred = {"xy": draw_pred_xy, "steps": int(draw_pred_xy.shape[0])}
        selected_agents = _select_agents(gt_tkf, agent_mask_np, args.max_agents)
        if focus_slot not in selected_agents and agent_mask_np[focus_slot]:
            selected_agents = [focus_slot] + selected_agents[: max(0, args.max_agents - 1)]
        bounds = _valid_bounds(
            gt_tkf,
            map_polylines,
            map_mask_np,
            margin=args.margin_m,
            selected_agents=selected_agents,
            preds=[("rollout", pred)],
            bounds_source="agents",
        )
        metrics = _trajectory_metrics(
            gt_tkf,
            pred_xy,
            agent_mask_np,
            focus_slot=focus_slot,
            context_frames=args.eval_ctx,
        )
        scenario_id = str(record.get("scenario_id", Path(record["filename"]).stem))
        output = output_dir / f"sample_{sample_index:02d}_{scenario_id}.png"
        fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=args.dpi)
        fig.patch.set_facecolor("#111111")
        _draw_panel(
            ax,
            title=(
                f"Sample {sample_index:02d} | scenario={scenario_id}\n"
                f"GT solid · D1 rollout dashed · GT ctx={args.eval_ctx} → H{args.horizon}"
            ),
            gt_tkf=gt_tkf,
            pred=pred,
            agent_mask=agent_mask_np,
            map_polylines=map_polylines,
            map_mask=map_mask_np,
            agent_ids=agent_ids,
            selected_agents=selected_agents,
            bounds=bounds,
            show_ids=True,
        )
        table_lines = _agent_metric_lines(
            gt_tkf,
            pred_xy,
            agent_mask_np,
            agent_ids,
            selected_agents,
            context_frames=args.eval_ctx,
        )
        fig.text(
            0.02,
            0.015,
            (
                f"focus slot={focus_slot}, id={int(agent_ids[focus_slot])} | "
                f"ADE={metrics['focus_ade_m']:.2f} m | FDE={metrics['focus_fde_m']:.2f} m\n"
                + "\n".join(table_lines[:6])
            ),
            color="#eeeeee",
            fontsize=8,
            va="bottom",
        )
        fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        image_paths.append(output)
        row = {
            "sample_order": int(record["sample_order"]),
            "dataset_index": int(record["dataset_index"]),
            "scenario_id": scenario_id,
            "filename": record["filename"],
            "focus_slot": focus_slot,
            "focus_agent_id": int(agent_ids[focus_slot]),
            "image": str(output.resolve()),
            **metrics,
        }
        metric_rows.append(row)
        selected_records.append(record)
        print(
            f"visualized {sample_index + 1}/{len(records)} scenario={scenario_id} "
            f"focus_ADE={metrics['focus_ade_m']:.3f}m focus_FDE={metrics['focus_fde_m']:.3f}m",
            flush=True,
        )

    contact_sheet = output_dir / "contact_sheet_first10.png"
    _make_contact_sheet(image_paths, contact_sheet)
    payload = {
        "eval_ckpt": str(Path(args.eval_ckpt).resolve()),
        "ckpt_step": int(checkpoint.get("step", -1)),
        "protocol": (
            f"ctx{args.eval_ctx}_h{args.horizon}_shortcut_d1_chunk32_stride30"
        ),
        "ground_truth_context_frames": int(args.eval_ctx),
        "rollout_frames": int(total_frames - args.eval_ctx),
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "selection": "first_10_in_recorded_128_manifest",
        "samples": metric_rows,
    }
    (output_dir / "rollout_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output_dir / "selected_manifest_first10.json").write_text(
        json.dumps({"source_manifest": args.subset_manifest, "samples": selected_records}, indent=2) + "\n"
    )
    print(f"wrote contact sheet: {contact_sheet}", flush=True)
    print(f"wrote output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    parser = base_eval.add_eval_args(wm.build_argparser())
    parser.description = "Visualize GT and H90 rollout trajectories for a fixed validation subset."
    parser.add_argument("--subset_manifest", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--max_agents", type=int, default=8)
    parser.add_argument("--margin_m", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=150)
    visualize(parser.parse_args())
