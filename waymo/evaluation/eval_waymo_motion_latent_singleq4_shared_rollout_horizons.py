"""Evaluate SingleQ4 MotionLatent checkpoints on shared rollout prefixes.

Every selected validation sample is rolled out once from the supplied context
through the largest requested horizon.  Metrics for shorter horizons are
computed from prefixes of that same stochastic latent trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.evaluation import eval_waymo_motion_latent_shared_rollout_horizons as shared_eval  # noqa: E402
from waymo.evaluation import eval_waymo_world_model_horizons as base_eval  # noqa: E402
from waymo.training.world_model import train_waymo_motion_latent_singleq4 as singleq4_train  # noqa: E402
from waymo.training.world_model import train_waymo_motion_latent_v1 as motion_latent_v1_train  # noqa: E402
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402
from waymo.training.world_model.motion_latent_singleq4 import MotionLatentSingleQ4  # noqa: E402


def build_singleq4_dynamics(
    ckpt: dict[str, Any],
    tokenizer: torch.nn.Module,
    tok_args: dict[str, Any],
    d_bottleneck: int,
    n_spatial: int,
    d_spatial: int,
    device: torch.device,
) -> MotionLatentSingleQ4:
    model_args = ckpt.get("args", {})
    if not hasattr(tokenizer, "decoder"):
        raise ValueError("SingleQ4 evaluation requires the vector tokenizer")
    return MotionLatentSingleQ4(
        d_model=int(model_args.get("d_model", 512)),
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=int(model_args.get("n_register", 8)),
        n_agents=int(tokenizer.decoder.n_agents),
        n_heads=int(model_args.get("n_heads", 8)),
        depth=int(model_args.get("depth", 8)),
        k_max=int(model_args.get("k_max", 64)),
        dropout=float(model_args.get("dropout", 0.0)),
        mlp_ratio=float(model_args.get("mlp_ratio", tok_args.get("mlp_ratio", 4.0))),
        time_every=int(model_args.get("time_every", 1)),
        scale_pos_embeds=True,
        action_clamp_inputs=False,
        map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer),
        map_cross_every=int(model_args.get("map_cross_every", 1)),
    ).to(device)


@torch.no_grad()
def sample_singleq4_sequence(
    dyn: MotionLatentSingleQ4,
    *,
    z_gt_packed: torch.Tensor,
    q_gt: torch.Tensor,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    action_slots: torch.Tensor,
    agent_mask: torch.Tensor,
    map_tokens: torch.Tensor | None,
    map_mask: torch.Tensor | None,
    ctx_length: int,
    horizon: int,
    max_context: int,
    kinematic_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out exactly one q integration and four latent shortcuts per frame."""
    total = int(z_gt_packed.shape[1])
    ctx_length = max(1, min(int(ctx_length), total - 1))
    horizon = min(int(horizon), total - ctx_length)
    z_history = [z_gt_packed[:, index] for index in range(ctx_length)]
    q_history = [q_gt[:, index] for index in range(ctx_length)]
    time_history = list(range(ctx_length))
    z_outputs = list(z_history)
    q_outputs = list(q_history)
    shortcut_steps = int(dyn.shortcut_steps)
    if shortcut_steps != 4 or dyn.k_max % shortcut_steps:
        raise ValueError(
            f"SingleQ4 requires four shortcut steps dividing k_max, got {shortcut_steps} and {dyn.k_max}"
        )
    solver_dt = 1.0 / shortcut_steps
    signal_scale = dyn.k_max // shortcut_steps

    for target_time in range(ctx_length, ctx_length + horizon):
        keep = min(int(max_context), len(z_history))
        past_packed = torch.stack(z_history[-keep:], dim=1)
        past_q = torch.stack(q_history[-keep:], dim=1)
        past_times = time_history[-keep:]
        indices = torch.tensor(
            past_times + [target_time], device=actions.device, dtype=torch.long
        )
        actions_sequence = actions.index_select(1, indices)
        act_mask_sequence = act_mask.index_select(1, indices)
        _, q_next = dyn.predict_single_q(
            past_packed=past_packed,
            past_q=past_q,
            q_current=q_history[-1],
            actions_sequence=actions_sequence,
            act_mask_sequence=act_mask_sequence,
            action_slots=action_slots,
            agent_mask=agent_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            kinematic_dt=kinematic_dt,
        )

        z_tau = torch.randn_like(z_gt_packed[:, target_time])
        for solver_index in range(shortcut_steps):
            tau = solver_index / shortcut_steps
            z_endpoint = dyn.predict_latent_endpoint(
                past_packed=past_packed,
                past_q=past_q,
                z_tau=z_tau,
                q_condition=q_next,
                actions_sequence=actions_sequence,
                act_mask_sequence=act_mask_sequence,
                agent_mask=agent_mask,
                map_tokens=map_tokens,
                map_mask=map_mask,
                tau_index=solver_index * signal_scale,
            )
            velocity = (z_endpoint.float() - z_tau.float()) / max(1e-4, 1.0 - tau)
            z_tau = (z_tau.float() + solver_dt * velocity).to(z_endpoint.dtype)

        z_next = z_tau
        z_outputs.append(z_next)
        q_outputs.append(q_next)
        z_history.append(z_next)
        q_history.append(q_next)
        time_history.append(target_time)
        if len(z_history) > max_context:
            z_history = z_history[-max_context:]
            q_history = q_history[-max_context:]
            time_history = time_history[-max_context:]

    return torch.stack(z_outputs, dim=1), torch.stack(q_outputs, dim=1)


def q_prefix_metrics(
    q_pred: torch.Tensor,
    q_gt: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    """Physical-q errors over all agents, evaluated only where GT is valid."""
    pred = q_pred[:, start:end]
    target = q_gt[:, start:end]
    eligible = agent_mask.to(torch.bool)
    state_mask = eligible[:, None] & (target[..., 5] > 0.5)

    xy = torch.linalg.vector_norm(pred[..., 0:2] - target[..., 0:2], dim=-1)
    velocity = torch.linalg.vector_norm(pred[..., 3:5] - target[..., 3:5], dim=-1)
    yaw_delta = pred[..., 6] - target[..., 6]
    yaw = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).abs()

    # Match rollout-stream training aggregation: average agents within each
    # frame, then give every future frame equal weight.
    per_frame_denom = state_mask.sum(dim=(0, 2)).clamp_min(1).to(xy.dtype)

    def prefix_mean(error: torch.Tensor) -> torch.Tensor:
        return ((error * state_mask).sum(dim=(0, 2)) / per_frame_denom).mean()

    final_mask = state_mask[:, -1]
    final_denom = final_mask.sum().clamp_min(1).to(xy.dtype)
    return {
        "q_xy_mae_m": prefix_mean(xy),
        "q_velocity_mae_mps": prefix_mean(velocity),
        "q_yaw_mae_rad": prefix_mean(yaw),
        "q_xy_fde_m": (xy[:, -1] * final_mask).sum() / final_denom,
        "q_velocity_fde_mps": (velocity[:, -1] * final_mask).sum() / final_denom,
        "q_yaw_fde_rad": (yaw[:, -1] * final_mask).sum() / final_denom,
    }


def semantic_reader_prefix_metrics(
    semantic_pred: Any,
    q_pred: torch.Tensor,
    q_gt: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    """Measure reader-to-q and reader-to-GT gaps in physical units.

    These diagnostics deliberately use target-valid masks, matching the
    continuous part of semantic-reader training.  ``q_gtvalid_*`` uses the
    same all-agent GT-valid mask as decoder ``agent_xy_mae_m``.
    """
    reader_cont = semantic_pred.continuous[:, start:end].float()
    reader_valid = torch.sigmoid(semantic_pred.valid_logits[:, start:end].float()) > 0.5
    q_state = q_pred[:, start:end].float()
    gt_state = q_gt[:, start:end].float()
    slots = agent_mask.to(torch.bool)[:, None].expand_as(reader_valid)
    q_valid = slots & (q_state[..., 5] > 0.5)
    gt_valid = slots & (gt_state[..., 5] > 0.5)

    reader_yaw = torch.atan2(reader_cont[..., 5], reader_cont[..., 6])

    def errors(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = torch.linalg.vector_norm(reader_cont[..., 0:2] - target[..., 0:2], dim=-1)
        velocity = torch.linalg.vector_norm(reader_cont[..., 3:5] - target[..., 3:5], dim=-1)
        yaw_delta = reader_yaw - target[..., 6]
        yaw = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).abs()
        return xy, velocity, yaw

    reader_q_xy, reader_q_velocity, reader_q_yaw = errors(q_state)
    reader_gt_xy, reader_gt_velocity, reader_gt_yaw = errors(gt_state)
    q_gt_xy = torch.linalg.vector_norm(q_state[..., 0:2] - gt_state[..., 0:2], dim=-1)
    q_gt_velocity = torch.linalg.vector_norm(q_state[..., 3:5] - gt_state[..., 3:5], dim=-1)
    q_gt_yaw_delta = q_state[..., 6] - gt_state[..., 6]
    q_gt_yaw = torch.atan2(torch.sin(q_gt_yaw_delta), torch.cos(q_gt_yaw_delta)).abs()

    def prefix_mean(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=(0, 2)).clamp_min(1).to(error.dtype)
        return ((error * mask).sum(dim=(0, 2)) / denom).mean()

    def endpoint_mean(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        final_mask = mask[:, -1]
        denom = final_mask.sum().clamp_min(1).to(error.dtype)
        return (error[:, -1] * final_mask).sum() / denom

    q_slot_values = (reader_valid == (q_state[..., 5] > 0.5))[slots]
    gt_slot_values = (reader_valid == (gt_state[..., 5] > 0.5))[slots]
    return {
        "reader_q_xy_mae_m": prefix_mean(reader_q_xy, q_valid),
        "reader_q_velocity_mae_mps": prefix_mean(reader_q_velocity, q_valid),
        "reader_q_yaw_mae_rad": prefix_mean(reader_q_yaw, q_valid),
        "reader_q_xy_fde_m": endpoint_mean(reader_q_xy, q_valid),
        "reader_q_velocity_fde_mps": endpoint_mean(reader_q_velocity, q_valid),
        "reader_q_yaw_fde_rad": endpoint_mean(reader_q_yaw, q_valid),
        "reader_q_valid_acc": q_slot_values.float().mean(),
        "reader_gt_xy_mae_m": prefix_mean(reader_gt_xy, gt_valid),
        "reader_gt_velocity_mae_mps": prefix_mean(reader_gt_velocity, gt_valid),
        "reader_gt_yaw_mae_rad": prefix_mean(reader_gt_yaw, gt_valid),
        "reader_gt_xy_fde_m": endpoint_mean(reader_gt_xy, gt_valid),
        "reader_gt_velocity_fde_mps": endpoint_mean(reader_gt_velocity, gt_valid),
        "reader_gt_yaw_fde_rad": endpoint_mean(reader_gt_yaw, gt_valid),
        "reader_gt_valid_acc": gt_slot_values.float().mean(),
        "q_gtvalid_xy_mae_m": prefix_mean(q_gt_xy, gt_valid),
        "q_gtvalid_velocity_mae_mps": prefix_mean(q_gt_velocity, gt_valid),
        "q_gtvalid_yaw_mae_rad": prefix_mean(q_gt_yaw, gt_valid),
        "q_gtvalid_xy_fde_m": endpoint_mean(q_gt_xy, gt_valid),
        "q_gtvalid_velocity_fde_mps": endpoint_mean(q_gt_velocity, gt_valid),
        "q_gtvalid_yaw_fde_rad": endpoint_mean(q_gt_yaw, gt_valid),
    }


def apply_semantic_reader_framewise(
    semantic_reader: torch.nn.Module,
    z: torch.Tensor,
    agent_mask: torch.Tensor,
) -> Any:
    """Apply P(q|z) exactly as trained: independently with T=1 per frame."""
    bsz, time_steps, n_latents, d_bottleneck = z.shape
    n_agents = int(agent_mask.shape[-1])
    flat_z = z.reshape(bsz * time_steps, 1, n_latents, d_bottleneck)
    flat_mask = (
        agent_mask[:, None]
        .expand(bsz, time_steps, n_agents)
        .reshape(bsz * time_steps, n_agents)
    )
    flat = semantic_reader(flat_z, agent_mask=flat_mask)
    return type(flat)(
        continuous=flat.continuous.reshape(bsz, time_steps, n_agents, -1),
        valid_logits=flat.valid_logits.reshape(bsz, time_steps, n_agents),
        agent_tokens=flat.agent_tokens.reshape(bsz, time_steps, n_agents, -1),
    )


@torch.no_grad()
def evaluate_shared_rollout(
    dyn: MotionLatentSingleQ4,
    tokenizer: torch.nn.Module,
    semantic_reader: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt: dict[str, Any],
) -> dict[str, dict[str, float]]:
    dyn.eval()
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    max_horizon = max(horizons)
    model_args = ckpt.get("args", {})
    max_context = int(model_args.get("max_context", args.max_rollout_window))
    kinematic_dt = float(model_args.get("kinematic_dt", args.kinematic_dt))
    totals: dict[int, dict[str, float]] = {horizon: {} for horizon in horizons}
    counts = {horizon: 0 for horizon in horizons}
    wm.seed_everything(args.seed)

    for batch_index, batch in enumerate(loader, start=1):
        batch = wm.slice_time_window(
            wm.move_batch(batch, device), args.eval_seq_len, random_start=False
        )
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        if actions is None or act_mask is None:
            raise ValueError("SingleQ4 evaluation requires --use_ego_actions")
        q_gt = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            tokenizer, batch, args, return_map=True
        )
        required = int(args.eval_ctx) + max_horizon
        if z_gt.shape[1] < required:
            raise ValueError(f"Need at least {required} frames, got {z_gt.shape[1]}")
        z_gt_packed = wm.pack_bottleneck_to_spatial(
            z_gt, n_spatial=args.n_spatial, k=args.packing_factor
        )
        z_pred_packed, q_pred = sample_singleq4_sequence(
            dyn,
            z_gt_packed=z_gt_packed,
            q_gt=q_gt,
            actions=actions,
            act_mask=act_mask,
            action_slots=action_slots,
            agent_mask=batch["agent_mask"],
            map_tokens=map_tokens,
            map_mask=map_mask,
            ctx_length=args.eval_ctx,
            horizon=max_horizon,
            max_context=max_context,
            kinematic_dt=kinematic_dt,
        )
        z_pred = wm.unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
        semantic_pred = None
        if semantic_reader is not None:
            semantic_pred = apply_semantic_reader_framewise(
                semantic_reader, z_pred, batch["agent_mask"]
            )
        z_decode = z_pred
        if z_pred.shape[1] < z_gt.shape[1]:
            z_decode = torch.cat([z_pred, z_gt[:, z_pred.shape[1] :]], dim=1)
        decoded = wm.decode_batch_z_for_world_model(tokenizer, z_decode, batch, args)

        score_start = int(args.eval_ctx)
        for horizon in horizons:
            score_end = score_start + horizon
            decoded_future = wm.slice_decoder_output(decoded, score_start, score_end)
            batch_future = wm.slice_future_batch(batch, score_start, score_end)
            future_weight = wm.build_agent_loss_weight_multiplier(
                batch_future, args, action_slots=action_slots
            )
            metrics = wm.reconstruction_metrics(
                tokenizer,
                decoded_future,
                batch_future,
                args,
                agent_loss_weight_multiplier=future_weight,
            )
            metrics["latent_mse_future"] = (
                z_pred_packed[:, score_start:score_end].float()
                - z_gt_packed[:, score_start:score_end].float()
            ).pow(2).mean()
            metrics.update(
                q_prefix_metrics(
                    q_pred,
                    q_gt,
                    batch["agent_mask"],
                    start=score_start,
                    end=score_end,
                )
            )
            if semantic_pred is not None:
                metrics.update(
                    semantic_reader_prefix_metrics(
                        semantic_pred,
                        q_pred,
                        q_gt,
                        batch["agent_mask"],
                        start=score_start,
                        end=score_end,
                    )
                )
            for name, value in wm.tensor_metrics(metrics).items():
                totals[horizon][name] = totals[horizon].get(name, 0.0) + value
            counts[horizon] += 1

        if batch_index == 1 or batch_index % 16 == 0 or batch_index == len(loader):
            print(f"shared rollout progress {batch_index}/{len(loader)}", flush=True)

    return {
        f"h{horizon}": {
            name: float(total / max(1, counts[horizon]))
            for name, total in totals[horizon].items()
        }
        for horizon in horizons
    }


def main(args: argparse.Namespace) -> None:
    if args.eval_batch_size != 1:
        raise ValueError("The recorded 128-sample protocol requires --eval_batch_size 1")
    if int(args.eval_ctx) != 1:
        raise ValueError("This protocol requires --eval_ctx 1")
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    supported_horizon_sets = ([10, 20, 30, 50, 80], [10, 20, 30, 50, 80, 90])
    if horizons not in supported_horizon_sets:
        raise ValueError(
            "This protocol requires --horizons 10 20 30 50 80 "
            "or --horizons 10 20 30 50 80 90"
        )
    required = int(args.eval_ctx) + max(horizons)
    if int(args.eval_seq_len) != required:
        raise ValueError(f"This protocol requires --eval_seq_len {required}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(args.seed)
    eval_ds = wm.WaymoVectorDataset(args.val_data_dir)
    subset_indices, subset_payload = shared_eval.load_or_create_subset_manifest(
        eval_ds,
        path=Path(args.subset_manifest),
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
    )
    eval_loader = DataLoader(
        Subset(eval_ds, subset_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=wm.worker_init_fn,
        collate_fn=wm._collate,
    )

    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
    d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor:
        raise ValueError("n_latents must be divisible by packing_factor")
    args.n_spatial = n_latents // args.packing_factor
    args.d_spatial = d_bottleneck * args.packing_factor

    ckpt = torch.load(args.eval_ckpt, map_location="cpu")
    if ckpt.get("format") != singleq4_train.CHECKPOINT_FORMAT:
        raise ValueError(
            f"Expected {singleq4_train.CHECKPOINT_FORMAT}, got {ckpt.get('format', 'legacy')}"
        )
    dyn = build_singleq4_dynamics(
        ckpt,
        tokenizer,
        tok_args,
        d_bottleneck,
        args.n_spatial,
        args.d_spatial,
        device,
    )
    dyn.load_state_dict(ckpt["model"], strict=True)
    dyn.eval()
    semantic_reader = None
    if args.semantic_reader_ckpt:
        semantic_reader = motion_latent_v1_train.load_semantic_reader(
            args.semantic_reader_ckpt,
            tokenizer=tokenizer,
            tok_args=tok_args,
            device=device,
        )

    print(f"eval_ckpt={args.eval_ckpt}", flush=True)
    print(
        f"ckpt_step={int(ckpt.get('step', -1))} ckpt_epoch={int(ckpt.get('epoch', -1))}",
        flush=True,
    )
    print(
        f"rollout_mode=motion_latent_singleq4 shared_rollout_horizon={max(horizons)} "
        "physical_integrations_per_frame=1 latent_shortcut_steps=4",
        flush=True,
    )
    print(
        f"val_size={len(eval_ds)} subset_size={len(subset_indices)} subset_seed={args.subset_seed} "
        f"subset_manifest={args.subset_manifest}",
        flush=True,
    )
    print(
        f"tokenizer_chunk_window={args.tokenizer_chunk_window} "
        f"tokenizer_chunk_stride={args.tokenizer_chunk_stride}",
        flush=True,
    )
    print(f"eval_ctx={args.eval_ctx} horizons={' '.join(map(str, horizons))}", flush=True)

    results = evaluate_shared_rollout(
        dyn, tokenizer, semantic_reader, eval_loader, device, args, ckpt
    )
    for horizon in sorted(results, key=lambda name: int(name[1:])):
        print(f"eval horizon={horizon[1:]} {wm.format_metrics(results[horizon])}", flush=True)

    output = {
        "eval_ckpt": args.eval_ckpt,
        "checkpoint_format": ckpt.get("format"),
        "ckpt_step": int(ckpt.get("step", -1)),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "rollout_mode": "motion_latent_singleq4",
        "physical_integrations_per_predicted_frame": 1,
        "latent_shortcut_steps_per_predicted_frame": 4,
        "shared_rollout_horizon": max(horizons),
        "metrics_are_prefixes_of_same_rollout": True,
        "eval_ctx": int(args.eval_ctx),
        "horizons": horizons,
        "val_size": len(eval_ds),
        "subset_size": len(subset_indices),
        "subset_seed": int(args.subset_seed),
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "subset_selection": subset_payload["selection"],
        "tokenizer_chunk_window": int(args.tokenizer_chunk_window),
        "tokenizer_chunk_stride": int(args.tokenizer_chunk_stride),
        "semantic_reader_ckpt": (
            str(Path(args.semantic_reader_ckpt).resolve()) if args.semantic_reader_ckpt else None
        ),
        "metrics": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote metrics: {output_path}", flush=True)


if __name__ == "__main__":
    parser = base_eval.add_eval_args(wm.build_argparser())
    parser.description = "Evaluate SingleQ4 MotionLatent shared rollout prefixes."
    parser.add_argument("--subset_manifest", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--semantic_reader_ckpt", type=str, default=None)
    main(parser.parse_args())
