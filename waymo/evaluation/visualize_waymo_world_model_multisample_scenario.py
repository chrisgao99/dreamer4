#!/usr/bin/env python3
"""Generate N individual world-model rollout plots for one exact Waymo NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

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
from waymo.evaluation.visualize_waymo_world_model_rollout import (  # noqa: E402
    _agent_metric_lines,
    _make_contact_sheet,
    _trajectory_metrics,
)
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


@torch.no_grad()
def visualize(args: argparse.Namespace) -> None:
    total_frames = int(args.eval_ctx) + int(args.eval_horizon)
    protocol = (int(args.eval_ctx), int(args.eval_horizon))
    if protocol not in {(1, 90), (11, 80)}:
        raise ValueError(
            "This comparison supports (eval_ctx, eval_horizon)=(1,90) or (11,80)"
        )
    if int(args.eval_num_rollouts) != 8:
        raise ValueError("This comparison requires eval_num_rollouts=8")
    if args.eval_schedule != "shortcut" or float(args.eval_d) != 1.0:
        raise ValueError("This comparison requires shortcut D1")

    if int(args.max_rollout_window) > 0:
        prediction_context_frames = min(
            int(args.eval_ctx),
            max(1, int(args.max_rollout_window) - 1),
        )
    else:
        prediction_context_frames = int(args.eval_ctx)
    prediction_context_start = int(args.eval_ctx) - prediction_context_frames + 1
    prediction_context_end = int(args.eval_ctx)
    context_usage_note = (
        "frame 1 is display-only"
        if prediction_context_start > 1
        else "all displayed context participates in prediction"
    )

    output_dir = Path(args.output_dir).resolve()
    images_dir = output_dir / "individual_rollouts"
    images_dir.mkdir(parents=True, exist_ok=True)
    scenario_npz = Path(args.scenario_npz).resolve()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(int(args.eval_multisample_seed))

    dataset = wm.WaymoVectorDataset(str(scenario_npz))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=wm._collate)
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
    schedule = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)

    # Match validation: reset after model construction and sample all eight
    # candidates in one vectorized call.
    wm.seed_everything(int(args.eval_multisample_seed))
    batch = next(iter(loader))
    batch = wm.slice_time_window(wm.move_batch(batch, device), total_frames, random_start=False)
    actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
    z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
        tokenizer, batch, args, return_map=args.dynamics_attend_map
    )
    z_gt_packed = wm.pack_bottleneck_to_spatial(
        z_gt, n_spatial=args.n_spatial, k=args.packing_factor
    )
    repeats = int(args.eval_num_rollouts)

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
        horizon=args.eval_horizon,
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
    agent_mask_np = batch["agent_mask"][0].detach().cpu().numpy().astype(bool)
    map_polylines = batch["map_polylines"][0].detach().float().cpu().numpy()
    map_mask_np = batch["map_mask"][0].detach().cpu().numpy().astype(bool)
    agent_ids = batch["agent_ids"][0].detach().cpu().numpy()
    focus_slot = int(action_slots[0].item())
    selected_agents = _select_agents(gt_tkf, agent_mask_np, int(args.max_agents))
    if focus_slot not in selected_agents and agent_mask_np[focus_slot]:
        selected_agents = [focus_slot] + selected_agents[: max(0, int(args.max_agents) - 1)]

    predictions = []
    for rollout_index in range(repeats):
        draw_xy = pred_xy[rollout_index].copy()
        draw_xy[: max(0, int(args.eval_ctx) - 1)] = np.nan
        predictions.append((f"rollout_{rollout_index:02d}", {"xy": draw_xy, "steps": total_frames}))
    bounds = _valid_bounds(
        gt_tkf,
        map_polylines,
        map_mask_np,
        margin=float(args.margin_m),
        selected_agents=selected_agents,
        preds=predictions,
        bounds_source="agents",
    )

    scenario_value = batch.get("scenario_id", [scenario_npz.stem])
    if isinstance(scenario_value, (list, tuple)):
        scenario_value = scenario_value[0]
    scenario_id = str(scenario_value).split("_focus_", 1)[0]
    image_paths: list[Path] = []
    rows = []
    for rollout_index, (_, pred) in enumerate(predictions):
        metrics = _trajectory_metrics(
            gt_tkf,
            pred_xy[rollout_index],
            agent_mask_np,
            focus_slot=focus_slot,
            context_frames=args.eval_ctx,
        )
        output = images_dir / f"rollout_{rollout_index:02d}.png"
        fig, ax = plt.subplots(figsize=(8.7, 8.7), dpi=int(args.dpi))
        fig.patch.set_facecolor("#111111")
        _draw_panel(
            ax,
            title=(
                f"{args.model_label} · rollout {rollout_index + 1}/8\n"
                f"scenario={scenario_id} · GT solid · model dashed · "
                f"context={args.eval_ctx} → future={args.eval_horizon}\n"
                f"first prediction reads context frames "
                f"{prediction_context_start}–{prediction_context_end}; "
                f"{context_usage_note}"
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
            pred_xy[rollout_index],
            agent_mask_np,
            agent_ids,
            selected_agents,
            context_frames=args.eval_ctx,
        )
        fig.text(
            0.02,
            0.015,
            (
                f"all-agent ADE={metrics['all_agent_ade_m']:.2f}m | "
                f"focus ADE={metrics['focus_ade_m']:.2f}m | "
                f"focus FDE={metrics['focus_fde_m']:.2f}m\n"
                + "\n".join(table_lines[:5])
            ),
            color="#eeeeee",
            fontsize=8,
            va="bottom",
        )
        fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        image_paths.append(output)
        rows.append({"rollout_index": rollout_index, "image": str(output), **metrics})

    contact_sheet = output_dir / "contact_sheet_8_rollouts.png"
    _make_contact_sheet(image_paths, contact_sheet)
    payload = {
        "model_label": args.model_label,
        "eval_ckpt": str(Path(args.eval_ckpt).resolve()),
        "ckpt_step": int(checkpoint.get("step", -1)),
        "scenario_id": scenario_id,
        "scenario_npz": str(scenario_npz),
        "focus_slot": focus_slot,
        "focus_track_id": int(agent_ids[focus_slot]),
        "protocol": (
            f"ctx{args.eval_ctx}_future{args.eval_horizon}_shortcut_d1_"
            f"n{repeats}_seed{args.eval_multisample_seed}"
        ),
        "displayed_context_frames_1based": [1, int(args.eval_ctx)],
        "initial_prediction_context_frames_1based": [
            prediction_context_start,
            prediction_context_end,
        ],
        "max_rollout_window_including_prediction_token": int(args.max_rollout_window),
        "num_rollouts": repeats,
        "rollouts": rows,
        "contact_sheet": str(contact_sheet),
    }
    (output_dir / "rollout_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote_individual_images={len(image_paths)} dir={images_dir}", flush=True)
    print(f"wrote_contact_sheet={contact_sheet}", flush=True)


if __name__ == "__main__":
    parser = base_eval.add_eval_args(wm.build_argparser())
    parser.description = "Visualize eight stochastic rollouts for one exact Waymo scenario."
    parser.add_argument("--scenario_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--max_agents", type=int, default=8)
    parser.add_argument("--margin_m", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=150)
    visualize(parser.parse_args())
