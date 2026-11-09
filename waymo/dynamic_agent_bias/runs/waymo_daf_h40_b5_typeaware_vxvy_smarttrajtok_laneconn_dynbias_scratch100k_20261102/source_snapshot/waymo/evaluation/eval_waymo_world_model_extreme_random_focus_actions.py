"""Paired stress test of extreme random focus actions on a Waymo world model.

For every validation scene this evaluator runs the same ctx->H rollout twice:

1. with the recorded focus-agent actions (the control), and
2. with every future focus action replaced by an extreme random action.

The PyTorch RNG state is restored between the two rollouts, so their diffusion
noise is identical.  Metrics therefore separate action sensitivity from normal
sampling variation.  Short horizons are prefixes of the same H=max(horizons)
pair of rollouts.
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

from waymo.core.vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo.evaluation import eval_waymo_world_model_horizons as base_eval  # noqa: E402
from waymo.evaluation.eval_waymo_motion_latent_shared_rollout_horizons import (  # noqa: E402
    load_or_create_subset_manifest,
)
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base_eval.add_eval_args(parser)
    parser.description = "Paired extreme-random focus-action sensitivity evaluation."
    parser.add_argument("--subset_manifest", type=str, required=True)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--random_action_seed", type=int, default=20260807)
    parser.add_argument(
        "--random_delta_xy_max_m",
        type=float,
        default=50.0,
        help="Each of delta-x and delta-y is uniform in [-max,+max] meters per 0.1 s tick.",
    )
    parser.add_argument(
        "--random_delta_yaw_max_rad",
        type=float,
        default=math.pi,
        help="Delta-yaw is uniform in [-max,+max] radians per tick.",
    )
    parser.add_argument(
        "--random_speed_abs_max_mps",
        type=float,
        default=200.0,
        help="Signed speed is uniform in [-max,+max] m/s.",
    )
    parser.add_argument(
        "--random_velocity_abs_max_mps",
        type=float,
        default=200.0,
        help="Each of vx and vy is uniform in [-max,+max] m/s.",
    )
    parser.add_argument("--progress_every", type=int, default=8)
    return parser


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return values.masked_fill(~mask, float("-inf")).amax()
    return values.sum() * 0.0


def _last_valid_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    selected_agents: torch.Tensor,
) -> torch.Tensor:
    """Average values at each selected agent's last valid frame."""
    time_index = torch.arange(valid.shape[1], device=valid.device).view(1, -1, 1)
    last_index = torch.where(valid, time_index, torch.zeros_like(time_index)).amax(dim=1)
    gathered = values.gather(dim=1, index=last_index[:, None, :]).squeeze(1)
    has_valid = valid.any(dim=1) & selected_agents
    return _masked_mean(gathered, has_valid)


def _prefix_standard_metrics(
    tokenizer: torch.nn.Module,
    decoded: Any,
    batch: dict[str, Any],
    args: argparse.Namespace,
    action_slots: torch.Tensor,
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    decoded_future = wm.slice_decoder_output(decoded, start, end)
    batch_future = wm.slice_future_batch(batch, start, end)
    future_weight = wm.build_agent_loss_weight_multiplier(batch_future, args, action_slots=action_slots)
    return wm.reconstruction_metrics(
        tokenizer,
        decoded_future,
        batch_future,
        args,
        agent_loss_weight_multiplier=future_weight,
    )


def make_extreme_random_actions(
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    *,
    ctx_length: int,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace all future conditioning fields while keeping valid forced on."""
    randomized = actions.clone()
    randomized_mask = act_mask.clone()
    future_steps = int(actions.shape[1]) - int(ctx_length)
    shape = (int(actions.shape[0]), future_steps)

    def symmetric_uniform(max_abs: float, tail: int) -> torch.Tensor:
        sample = torch.rand((*shape, tail), generator=generator, dtype=torch.float32)
        return (sample * 2.0 - 1.0) * float(max_abs)

    future = randomized[:, ctx_length:]
    future[..., 0:2] = symmetric_uniform(args.random_delta_xy_max_m, 2).to(
        device=actions.device, dtype=actions.dtype
    )
    future[..., 2:3] = symmetric_uniform(args.random_delta_yaw_max_rad, 1).to(
        device=actions.device, dtype=actions.dtype
    )
    future[..., 3:4] = symmetric_uniform(args.random_speed_abs_max_mps, 1).to(
        device=actions.device, dtype=actions.dtype
    )
    future[..., 4:6] = symmetric_uniform(args.random_velocity_abs_max_mps, 2).to(
        device=actions.device, dtype=actions.dtype
    )
    future[..., 6] = 1.0
    future[..., 7:] = 0.0
    randomized_mask[:, ctx_length:, :] = 0.0
    randomized_mask[:, ctx_length:, 0:7] = 1.0
    return randomized, randomized_mask


def _capture_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _restore_rng_state(device: torch.device, state: tuple[torch.Tensor, torch.Tensor | None]) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _decoded_state(
    decoded: Any,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    agents = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
    anchor_xy = agents[:, 0, :, 0:2] if args.agent_xy_parameterization == "delta" else None
    xy = decoder_agent_xy(
        decoded,
        agent_xy_loss=args.agent_xy_loss,
        agent_xy_parameterization=args.agent_xy_parameterization,
        anchor_xy=anchor_xy,
    )
    speed = decoded.agent_continuous[..., 2]
    yaw = torch.atan2(decoded.agent_continuous[..., 5], decoded.agent_continuous[..., 6])
    valid_probability = decoded.agent_valid_logits.sigmoid()
    return xy, speed, yaw, valid_probability


def _group_masks(
    batch: dict[str, Any],
    action_slots: torch.Tensor,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    agents = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
    valid = (agents[:, start:end, :, 5] > 0.5) & batch["agent_mask"][:, None, :].bool()
    focus = torch.nn.functional.one_hot(
        action_slots,
        num_classes=int(batch["agent_mask"].shape[1]),
    ).bool()
    others = batch["agent_mask"].bool() & ~focus
    return valid, focus, others


def _comparison_metrics(
    baseline_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    random_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, Any],
    action_slots: torch.Tensor,
    z_baseline: torch.Tensor,
    z_random: torch.Tensor,
    random_actions: torch.Tensor,
    *,
    start: int,
    end: int,
) -> dict[str, torch.Tensor]:
    baseline_xy, baseline_speed, baseline_yaw, baseline_valid_prob = baseline_state
    random_xy, random_speed, random_yaw, random_valid_prob = random_state
    gt = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
    valid, focus_agents, other_agents = _group_masks(batch, action_slots, start, end)
    focus_valid = valid & focus_agents[:, None, :]
    other_valid = valid & other_agents[:, None, :]

    baseline_xy = baseline_xy[:, start:end]
    random_xy = random_xy[:, start:end]
    baseline_speed = baseline_speed[:, start:end]
    random_speed = random_speed[:, start:end]
    baseline_yaw = baseline_yaw[:, start:end]
    random_yaw = random_yaw[:, start:end]
    baseline_valid_prob = baseline_valid_prob[:, start:end]
    random_valid_prob = random_valid_prob[:, start:end]
    gt_xy = gt[:, start:end, :, 0:2]

    baseline_gt_error = torch.linalg.vector_norm(baseline_xy - gt_xy, dim=-1)
    random_gt_error = torch.linalg.vector_norm(random_xy - gt_xy, dim=-1)
    response_xy = torch.linalg.vector_norm(random_xy - baseline_xy, dim=-1)
    response_speed = (random_speed - baseline_speed).abs()
    response_yaw_deg = wm.wrap_angle_rad(random_yaw - baseline_yaw).abs() * (180.0 / math.pi)
    response_valid_probability = (random_valid_prob - baseline_valid_prob).abs()

    metrics: dict[str, torch.Tensor] = {}
    for group_name, group_valid, selected_agents in (
        ("focus", focus_valid, focus_agents),
        ("other", other_valid, other_agents),
    ):
        metrics[f"baseline_{group_name}_ade_gt_m"] = _masked_mean(baseline_gt_error, group_valid)
        metrics[f"extreme_{group_name}_ade_gt_m"] = _masked_mean(random_gt_error, group_valid)
        metrics[f"baseline_{group_name}_fde_gt_m"] = _last_valid_mean(
            baseline_gt_error, valid, selected_agents
        )
        metrics[f"extreme_{group_name}_fde_gt_m"] = _last_valid_mean(
            random_gt_error, valid, selected_agents
        )
        metrics[f"{group_name}_response_ade_m"] = _masked_mean(response_xy, group_valid)
        metrics[f"{group_name}_response_fde_m"] = _last_valid_mean(response_xy, valid, selected_agents)
        metrics[f"{group_name}_response_speed_mae_mps"] = _masked_mean(response_speed, group_valid)
        metrics[f"{group_name}_response_yaw_mae_deg"] = _masked_mean(response_yaw_deg, group_valid)
        metrics[f"{group_name}_response_valid_probability"] = _masked_mean(
            response_valid_probability,
            group_valid,
        )

    metrics["other_response_max_m"] = _masked_max(response_xy, other_valid)
    metrics["other_response_fraction_gt_1m"] = _masked_mean((response_xy > 1.0).float(), other_valid)
    metrics["other_response_fraction_gt_5m"] = _masked_mean((response_xy > 5.0).float(), other_valid)
    metrics["extreme_minus_baseline_focus_ade_gt_m"] = (
        metrics["extreme_focus_ade_gt_m"] - metrics["baseline_focus_ade_gt_m"]
    )
    metrics["extreme_minus_baseline_focus_fde_gt_m"] = (
        metrics["extreme_focus_fde_gt_m"] - metrics["baseline_focus_fde_gt_m"]
    )
    metrics["extreme_minus_baseline_other_ade_gt_m"] = (
        metrics["extreme_other_ade_gt_m"] - metrics["baseline_other_ade_gt_m"]
    )
    metrics["extreme_minus_baseline_other_fde_gt_m"] = (
        metrics["extreme_other_fde_gt_m"] - metrics["baseline_other_fde_gt_m"]
    )
    metrics["other_to_focus_response_ade_ratio"] = (
        metrics["other_response_ade_m"] / metrics["focus_response_ade_m"].clamp_min(1e-8)
    )
    metrics["latent_response_rmse"] = (
        z_random[:, start:end].float() - z_baseline[:, start:end].float()
    ).pow(2).mean().sqrt()

    action_prefix = random_actions[:, start:end]
    action_delta_xy = torch.linalg.vector_norm(action_prefix[..., 0:2], dim=-1)
    action_vxvy = torch.linalg.vector_norm(action_prefix[..., 4:6], dim=-1)
    metrics["random_action_delta_xy_mean_m"] = action_delta_xy.mean()
    metrics["random_action_delta_xy_max_m"] = action_delta_xy.amax()
    metrics["random_action_delta_yaw_mean_abs_deg"] = action_prefix[..., 2].abs().mean() * (180.0 / math.pi)
    metrics["random_action_speed_mean_abs_mps"] = action_prefix[..., 3].abs().mean()
    metrics["random_action_speed_max_abs_mps"] = action_prefix[..., 3].abs().amax()
    metrics["random_action_vxvy_mean_norm_mps"] = action_vxvy.mean()
    metrics["random_action_vxvy_max_norm_mps"] = action_vxvy.amax()
    return metrics


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in rows[0]
    }


def _normal_ci(rows: list[dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    """Scene-level mean and normal-approximation 95% CI."""
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
        mean = float(values.mean())
        if values.numel() > 1:
            half_width = float(1.96 * values.std(unbiased=True) / math.sqrt(values.numel()))
        else:
            half_width = 0.0
        result[key] = {
            "mean": mean,
            "ci95_low": mean - half_width,
            "ci95_high": mean + half_width,
            "n_scenes": int(values.numel()),
        }
    return result


@torch.no_grad()
def evaluate(
    dyn: torch.nn.Module,
    tokenizer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    dyn.eval()
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    max_horizon = max(horizons)
    schedule = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(int(args.random_action_seed))
    rows: dict[int, list[dict[str, Any]]] = {horizon: [] for horizon in horizons}

    # Model construction consumes RNG; standardize rollout noise after loading.
    wm.seed_everything(args.seed)
    for batch_index, batch in enumerate(loader, start=1):
        batch = wm.slice_time_window(wm.move_batch(batch, device), args.eval_seq_len, random_start=False)
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        if actions is None or act_mask is None or action_slots is None:
            raise ValueError("This evaluation requires --use_ego_actions --ego_action_source focus")
        random_actions, random_act_mask = make_extreme_random_actions(
            actions,
            act_mask,
            ctx_length=args.eval_ctx,
            generator=random_generator,
            args=args,
        )
        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
            tokenizer,
            batch,
            args,
            return_map=args.dynamics_attend_map,
        )
        required = int(args.eval_ctx) + max_horizon
        if z_gt.shape[1] < required:
            raise ValueError(f"Need at least {required} frames, got {z_gt.shape[1]}")
        z_gt_packed = wm.pack_bottleneck_to_spatial(
            z_gt,
            n_spatial=args.n_spatial,
            k=args.packing_factor,
        )

        paired_rng_state = _capture_rng_state(device)
        z_baseline_packed = wm.sample_autoregressive_packed_sequence(
            wm.unwrap_model(dyn),
            z_gt_packed=z_gt_packed,
            actions=actions,
            act_mask=act_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            ctx_length=args.eval_ctx,
            horizon=max_horizon,
            k_max=args.k_max,
            sched=schedule,
            max_rollout_window=args.max_rollout_window,
        )
        _restore_rng_state(device, paired_rng_state)
        z_random_packed = wm.sample_autoregressive_packed_sequence(
            wm.unwrap_model(dyn),
            z_gt_packed=z_gt_packed,
            actions=random_actions,
            act_mask=random_act_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            ctx_length=args.eval_ctx,
            horizon=max_horizon,
            k_max=args.k_max,
            sched=schedule,
            max_rollout_window=args.max_rollout_window,
        )

        z_baseline = wm.unpack_spatial_to_bottleneck(z_baseline_packed, k=args.packing_factor)
        z_random = wm.unpack_spatial_to_bottleneck(z_random_packed, k=args.packing_factor)
        decoded_baseline = wm.decode_batch_z_for_world_model(tokenizer, z_baseline, batch, args)
        decoded_random = wm.decode_batch_z_for_world_model(tokenizer, z_random, batch, args)
        baseline_state = _decoded_state(decoded_baseline, batch, args)
        random_state = _decoded_state(decoded_random, batch, args)

        score_start = int(args.eval_ctx)
        for horizon in horizons:
            score_end = score_start + horizon
            baseline_metrics = _prefix_standard_metrics(
                tokenizer,
                decoded_baseline,
                batch,
                args,
                action_slots,
                score_start,
                score_end,
            )
            random_metrics = _prefix_standard_metrics(
                tokenizer,
                decoded_random,
                batch,
                args,
                action_slots,
                score_start,
                score_end,
            )
            baseline_metrics["latent_mse_future"] = (
                z_baseline_packed[:, score_start:score_end].float()
                - z_gt_packed[:, score_start:score_end].float()
            ).pow(2).mean()
            random_metrics["latent_mse_future"] = (
                z_random_packed[:, score_start:score_end].float()
                - z_gt_packed[:, score_start:score_end].float()
            ).pow(2).mean()
            comparison = _comparison_metrics(
                baseline_state,
                random_state,
                batch,
                action_slots,
                z_baseline_packed,
                z_random_packed,
                random_actions,
                start=score_start,
                end=score_end,
            )
            rows[horizon].append(
                {
                    "sample_order": batch_index - 1,
                    "baseline_vs_gt": wm.tensor_metrics(baseline_metrics),
                    "extreme_random_vs_gt": wm.tensor_metrics(random_metrics),
                    "causal_response": wm.tensor_metrics(comparison),
                }
            )

        if batch_index == 1 or (
            args.progress_every > 0 and batch_index % int(args.progress_every) == 0
        ) or batch_index == len(loader):
            print(f"paired extreme-action progress {batch_index}/{len(loader)}", flush=True)
        if args.eval_max_batches > 0 and batch_index >= int(args.eval_max_batches):
            break

    results: dict[str, Any] = {}
    ci_keys = [
        "focus_response_ade_m",
        "focus_response_fde_m",
        "other_response_ade_m",
        "other_response_fde_m",
        "other_response_max_m",
        "extreme_minus_baseline_focus_ade_gt_m",
        "extreme_minus_baseline_focus_fde_gt_m",
        "extreme_minus_baseline_other_ade_gt_m",
        "extreme_minus_baseline_other_fde_gt_m",
    ]
    for horizon in horizons:
        horizon_rows = rows[horizon]
        baseline_rows = [row["baseline_vs_gt"] for row in horizon_rows]
        random_rows = [row["extreme_random_vs_gt"] for row in horizon_rows]
        comparison_rows = [row["causal_response"] for row in horizon_rows]
        results[f"h{horizon}"] = {
            "baseline_vs_gt": _mean_dict(baseline_rows),
            "extreme_random_vs_gt": _mean_dict(random_rows),
            "causal_response": _mean_dict(comparison_rows),
            "scene_level_ci95": _normal_ci(comparison_rows, ci_keys),
            "per_sample": horizon_rows,
        }
    return results


def main(args: argparse.Namespace) -> None:
    if args.eval_batch_size != 1:
        raise ValueError("Paired scene-level statistics require --eval_batch_size 1")
    if args.eval_schedule != "shortcut":
        raise ValueError("This checkpoint protocol requires --eval_schedule shortcut")
    if not args.use_ego_actions or args.ego_action_source != "focus":
        raise ValueError("Pass --use_ego_actions --ego_action_source focus")
    if args.ego_action_normalization != "raw" or args.ego_action_clamp:
        raise ValueError("The requested checkpoint requires raw, unclamped actions")
    horizons = sorted(set(int(horizon) for horizon in args.horizons))
    if not horizons or horizons[0] < 1:
        raise ValueError("Horizons must be positive")
    if int(args.eval_seq_len) < int(args.eval_ctx) + max(horizons):
        raise ValueError("eval_seq_len must cover eval_ctx + max(horizons)")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(args.seed)
    eval_ds = wm.WaymoVectorDataset(args.val_data_dir)
    subset_indices, subset_payload = load_or_create_subset_manifest(
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
    if isinstance(tokenizer, wm.FrozenWaymoFocusTokenizer):
        n_latents = tokenizer.n_latents
        d_bottleneck = tokenizer.d_bottleneck
    else:
        n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
        d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor != 0:
        raise ValueError(f"n_latents={n_latents} must be divisible by packing_factor={args.packing_factor}")
    args.n_spatial = n_latents // args.packing_factor
    args.d_spatial = d_bottleneck * args.packing_factor

    ckpt = torch.load(args.eval_ckpt, map_location="cpu")
    if ckpt.get("format") == base_eval.MOTION_LATENT_V1_FORMAT:
        raise ValueError("Expected a legacy World Model checkpoint, got MotionLatent V1")
    dyn = base_eval.build_dynamics(
        args,
        d_bottleneck,
        device,
        map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None,
    )
    base_eval.load_dynamics_state(dyn, args.eval_ckpt, ckpt=ckpt)
    dyn.eval()

    schedule = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    action_config = {
        "seed": int(args.random_action_seed),
        "delta_x_each_m_uniform": [-float(args.random_delta_xy_max_m), float(args.random_delta_xy_max_m)],
        "delta_y_each_m_uniform": [-float(args.random_delta_xy_max_m), float(args.random_delta_xy_max_m)],
        "delta_yaw_rad_uniform": [
            -float(args.random_delta_yaw_max_rad),
            float(args.random_delta_yaw_max_rad),
        ],
        "signed_speed_mps_uniform": [
            -float(args.random_speed_abs_max_mps),
            float(args.random_speed_abs_max_mps),
        ],
        "vx_each_mps_uniform": [
            -float(args.random_velocity_abs_max_mps),
            float(args.random_velocity_abs_max_mps),
        ],
        "vy_each_mps_uniform": [
            -float(args.random_velocity_abs_max_mps),
            float(args.random_velocity_abs_max_mps),
        ],
        "future_valid_forced_to_one": True,
        "future_action_mask_fields_0_through_6_forced_to_one": True,
        "fields_are_independent_across_time_and_dimension": True,
    }
    print(f"eval_mode=paired_extreme_random_focus_actions", flush=True)
    print(f"eval_ckpt={args.eval_ckpt}", flush=True)
    print(f"ckpt_step={int(ckpt.get('step', -1))} ckpt_epoch={int(ckpt.get('epoch', -1))}", flush=True)
    print(
        f"paired_noise=true shared_rollout_horizon={max(horizons)} "
        f"solver_steps_per_frame={int(schedule['K'])} horizons={' '.join(map(str, horizons))}",
        flush=True,
    )
    print(f"extreme_random_action_config={json.dumps(action_config, sort_keys=True)}", flush=True)
    print(
        f"val_size={len(eval_ds)} subset_size={len(subset_indices)} "
        f"subset_manifest={args.subset_manifest}",
        flush=True,
    )

    results = evaluate(dyn, tokenizer, eval_loader, device, args)
    for horizon in horizons:
        metrics = results[f"h{horizon}"]["causal_response"]
        print(
            f"h={horizon} focus_response_ADE={metrics['focus_response_ade_m']:.4f}m "
            f"focus_response_FDE={metrics['focus_response_fde_m']:.4f}m "
            f"other_response_ADE={metrics['other_response_ade_m']:.4f}m "
            f"other_response_FDE={metrics['other_response_fde_m']:.4f}m "
            f"focus_gt_ADE_delta={metrics['extreme_minus_baseline_focus_ade_gt_m']:+.4f}m "
            f"other_gt_ADE_delta={metrics['extreme_minus_baseline_other_ade_gt_m']:+.4f}m",
            flush=True,
        )

    output = {
        "eval_mode": "paired_extreme_random_focus_actions",
        "eval_ckpt": args.eval_ckpt,
        "checkpoint_format": ckpt.get("format", "legacy"),
        "ckpt_step": int(ckpt.get("step", -1)),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "paired_rollout_noise": True,
        "control_action": "recorded_focus_action",
        "treatment_action": "independent_extreme_random_focus_action_every_future_frame",
        "random_action_config": action_config,
        "eval_ctx": int(args.eval_ctx),
        "shared_rollout_horizon": max(horizons),
        "horizons": horizons,
        "metrics_are_prefixes_of_same_rollout": True,
        "solver_steps_per_predicted_frame": int(schedule["K"]),
        "eval_schedule": str(args.eval_schedule),
        "eval_d": float(schedule["d"]),
        "val_size": len(eval_ds),
        "subset_size": len(subset_indices),
        "subset_seed": int(args.subset_seed),
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "subset_selection": subset_payload["selection"],
        "tokenizer_chunk_window": int(args.tokenizer_chunk_window),
        "tokenizer_chunk_stride": int(args.tokenizer_chunk_stride),
        "results": results,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote metrics: {output_path}", flush=True)


if __name__ == "__main__":
    main(add_args(wm.build_argparser()).parse_args())
