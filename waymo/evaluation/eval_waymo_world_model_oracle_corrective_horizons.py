"""Evaluate map-conditioned world-model rollouts with oracle corrective focus actions.

At every predicted timestep this evaluator decodes the current imagined latent
prefix, measures the offset from the decoded focus state to the next GT focus
state, and uses that offset as the next action's delta-XY/delta-yaw fields.
This intentionally leaks the next GT focus state and is therefore an oracle
controllability diagnostic, not a deployable forecasting metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.core.vector_tokenizer_decoder import decoder_agent_xy  # noqa: E402
from waymo.evaluation import eval_waymo_world_model_horizons as base  # noqa: E402

wm = base.wm


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = base.add_eval_args(parser)
    parser.description = "Evaluate oracle corrective-action world-model rollouts."
    parser.add_argument(
        "--oracle_decode_window",
        type=int,
        default=0,
        help="Latent prefix length used to decode the imagined focus state; 0 keeps the full prefix.",
    )
    parser.add_argument("--progress_every", type=int, default=8, help="Print progress every N batches; 0 disables it.")
    return parser


@torch.no_grad()
def _decoder_static_kwargs(tokenizer: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Encode static decoder map memory once per batch instead of once per rollout step."""
    if isinstance(tokenizer, wm.FrozenWaymoFocusTokenizer):
        return {}
    return wm.decoder_map_kwargs(tokenizer, batch)


@torch.no_grad()
def _decode_prefix(
    tokenizer: torch.nn.Module,
    z_prefix: torch.Tensor,
    batch: Dict[str, Any],
    decoder_static_kwargs: Dict[str, torch.Tensor],
) -> Any:
    if isinstance(tokenizer, wm.FrozenWaymoFocusTokenizer):
        return wm.decode_batch_z(tokenizer, z_prefix, batch)
    return tokenizer.decoder(
        z_prefix,
        agent_mask=batch["agent_mask"],
        light_mask=batch["light_mask"][:, : z_prefix.shape[1]],
        **decoder_static_kwargs,
    )


def _gather_slot(values: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Gather (B,T,K,...) values at one slot per batch item."""
    bsz, steps = values.shape[:2]
    tail = values.shape[3:]
    index = slots.view(bsz, 1, 1, *([1] * len(tail))).expand(bsz, steps, 1, *tail)
    return values.gather(dim=2, index=index).squeeze(2)


@torch.no_grad()
def _decoded_focus_state(
    tokenizer: torch.nn.Module,
    z_prefix: torch.Tensor,
    batch: Dict[str, Any],
    args: argparse.Namespace,
    action_slots: torch.Tensor,
    decoder_static_kwargs: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = _decode_prefix(tokenizer, z_prefix, batch, decoder_static_kwargs)
    agents_btkf = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
    anchor_xy = agents_btkf[:, 0, :, 0:2] if args.agent_xy_parameterization == "delta" else None
    pred_xy = decoder_agent_xy(
        pred,
        args.agent_xy_loss,
        args.agent_xy_parameterization,
        anchor_xy=anchor_xy,
    )
    pred_yaw = torch.atan2(pred.agent_continuous[..., 5], pred.agent_continuous[..., 6])
    return _gather_slot(pred_xy, action_slots)[:, -1], _gather_slot(pred_yaw[..., None], action_slots)[:, -1, 0]


@torch.no_grad()
def oracle_corrective_rollout(
    dyn: torch.nn.Module,
    tokenizer: torch.nn.Module,
    *,
    z_gt_packed: torch.Tensor,
    batch: Dict[str, Any],
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    action_slots: torch.Tensor,
    map_tokens: torch.Tensor | None,
    map_mask: torch.Tensor | None,
    args: argparse.Namespace,
    horizon: int,
    sched: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = int(z_gt_packed.shape[1])
    ctx_length = max(1, min(int(args.eval_ctx), total - 1))
    horizon = min(int(horizon), total - ctx_length)
    outs = [z_gt_packed[:, t] for t in range(ctx_length)]
    corrective_actions = actions.clone()
    corrective_mask = act_mask.clone()

    agents_btkf = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
    gt_focus = _gather_slot(agents_btkf, action_slots)
    slot_valid = batch["agent_mask"].gather(1, action_slots[:, None]).squeeze(1).bool()
    decoder_static = _decoder_static_kwargs(tokenizer, batch)
    correction_xy_steps = []
    correction_yaw_steps = []

    for _ in range(horizon):
        past_packed = torch.stack(outs, dim=1)
        next_t = int(past_packed.shape[1])
        z_prefix = wm.unpack_spatial_to_bottleneck(past_packed, k=args.packing_factor)
        if args.oracle_decode_window > 0 and z_prefix.shape[1] > args.oracle_decode_window:
            z_prefix = z_prefix[:, -int(args.oracle_decode_window) :]

        pred_xy, pred_yaw = _decoded_focus_state(
            tokenizer,
            z_prefix,
            batch,
            args,
            action_slots,
            decoder_static,
        )
        target = gt_focus[:, next_t]
        correction_xy = target[:, 0:2] - pred_xy
        correction_yaw = wm.wrap_angle_rad(target[:, 6] - pred_yaw)

        action_xy = correction_xy
        action_yaw = correction_yaw
        if args.ego_action_normalization == "scaled":
            action_xy = action_xy / float(args.ego_action_xy_scale)
            action_yaw = action_yaw / float(args.ego_action_yaw_scale)
        corrective_actions[:, next_t, 0:2] = action_xy
        corrective_actions[:, next_t, 2] = action_yaw

        target_valid = (target[:, 5] > 0.5) & slot_valid
        corrective_mask[:, next_t, 0:3] = target_valid[:, None].to(corrective_mask.dtype)
        corrective_actions[:, next_t] *= target_valid[:, None].to(corrective_actions.dtype)

        correction_xy_steps.append(correction_xy.norm(dim=-1))
        correction_yaw_steps.append(correction_yaw.abs() * (180.0 / math.pi))

        z_next = wm.sample_one_timestep_packed(
            wm.unwrap_model(dyn),
            past_packed=past_packed,
            actions_seq=corrective_actions[:, : next_t + 1],
            act_mask_seq=corrective_mask[:, : next_t + 1],
            map_tokens=map_tokens,
            map_mask=map_mask,
            k_max=args.k_max,
            sched=sched,
            max_rollout_window=args.max_rollout_window,
        )
        outs.append(z_next)

    return (
        torch.stack(outs, dim=1),
        torch.stack(correction_xy_steps, dim=1),
        torch.stack(correction_yaw_steps, dim=1),
    )


def _average_totals(totals: Dict[str, float], count: int, device: torch.device, ddp: bool) -> Dict[str, float]:
    names = wm.metric_order(totals)
    packed = torch.tensor(
        [float(count)] + [totals.get(name, 0.0) for name in names],
        device=device,
        dtype=torch.float64,
    )
    if ddp:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(packed[0].item()))
    return {name: float(packed[i + 1].item() / total_count) for i, name in enumerate(names)}


@torch.no_grad()
def evaluate_oracle_corrective_horizons(
    dyn: torch.nn.Module,
    tokenizer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    ddp: bool,
) -> Dict[str, Dict[str, float]]:
    was_training = dyn.training
    dyn.eval()
    horizons = sorted(set(int(h) for h in args.horizons))
    max_horizon = max(horizons)
    sched = wm.make_tau_schedule(k_max=args.k_max, schedule=args.eval_schedule, d=args.eval_d)
    totals = {h: {} for h in horizons}
    count = 0
    start_time = time.time()

    for batch in loader:
        batch = wm.slice_time_window(wm.move_batch(batch, device), args.eval_seq_len, random_start=False)
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        if actions is None or act_mask is None or action_slots is None:
            raise ValueError("Oracle corrective evaluation requires --use_ego_actions.")
        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs(
            tokenizer,
            batch,
            return_map=args.dynamics_attend_map,
        )
        z_gt_packed = wm.pack_bottleneck_to_spatial(z_gt, n_spatial=args.n_spatial, k=args.packing_factor)
        z_pred_packed, correction_xy, correction_yaw = oracle_corrective_rollout(
            dyn,
            tokenizer,
            z_gt_packed=z_gt_packed,
            batch=batch,
            actions=actions,
            act_mask=act_mask,
            action_slots=action_slots,
            map_tokens=map_tokens,
            map_mask=map_mask,
            args=args,
            horizon=max_horizon,
            sched=sched,
        )
        z_pred = wm.unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
        decoder_static = _decoder_static_kwargs(tokenizer, batch)
        decoded = _decode_prefix(tokenizer, z_pred, batch, decoder_static)

        score_start = int(args.eval_ctx)
        for horizon in horizons:
            score_end = score_start + horizon
            decoded_future = wm.slice_decoder_output(decoded, score_start, score_end)
            batch_future = wm.slice_future_batch(batch, score_start, score_end)
            future_weight = wm.build_agent_loss_weight_multiplier(batch_future, args, action_slots=action_slots)
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
            metrics["oracle_correction_xy_mean_m"] = correction_xy[:, :horizon].mean()
            metrics["oracle_correction_xy_at_h_m"] = correction_xy[:, horizon - 1].mean()
            metrics["oracle_correction_yaw_mean_deg"] = correction_yaw[:, :horizon].mean()
            values = wm.tensor_metrics(metrics)
            for key, value in values.items():
                totals[horizon][key] = totals[horizon].get(key, 0.0) + value

        count += 1
        if wm.is_rank0() and args.progress_every > 0 and count % int(args.progress_every) == 0:
            elapsed = time.time() - start_time
            print(f"oracle progress batches={count} elapsed_min={elapsed / 60.0:.1f}", flush=True)
        if args.eval_max_batches > 0 and count >= args.eval_max_batches:
            break

    results = {f"h{h}": _average_totals(totals[h], count, device, ddp) for h in horizons}
    if was_training:
        dyn.train()
    return results


def main(args: argparse.Namespace) -> None:
    ddp, rank, world_size, local_rank = wm.init_distributed()
    if ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(args.seed + rank)

    try:
        if args.val_data_dir is not None:
            eval_ds = wm.WaymoVectorDataset(args.val_data_dir)
        else:
            dataset = wm.WaymoVectorDataset(args.data_dir)
            _, eval_ds = wm.make_splits(dataset, args.val_fraction, args.seed)
            if eval_ds is None:
                raise ValueError("No validation split available. Pass --val_data_dir.")
        eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.eval_batch_size,
            sampler=eval_sampler,
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

        max_available_horizon = int(args.eval_seq_len) - int(args.eval_ctx)
        if max(args.horizons) > max_available_horizon:
            raise ValueError(
                f"Maximum horizon {max(args.horizons)} exceeds eval_seq_len-eval_ctx={max_available_horizon}."
            )
        dyn = base.build_dynamics(
            args,
            d_bottleneck,
            device,
            map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None,
        )
        ckpt = base.load_dynamics_state(dyn, args.eval_ckpt)
        dyn.eval()
        if ddp:
            dyn = torch.nn.parallel.DistributedDataParallel(
                dyn,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                broadcast_buffers=False,
            )

        if wm.is_rank0():
            print(f"eval_mode=oracle_corrective_focus", flush=True)
            print(f"eval_ckpt={args.eval_ckpt}", flush=True)
            print(f"ckpt_step={int(ckpt.get('step', -1))} ckpt_epoch={int(ckpt.get('epoch', -1))}", flush=True)
            print(
                f"device={device} ddp={ddp} world_size={world_size} val={len(eval_ds)} "
                f"eval_batch_size={args.eval_batch_size} eval_max_batches={args.eval_max_batches}",
                flush=True,
            )
            print(
                f"eval_seq_len={args.eval_seq_len} eval_ctx={args.eval_ctx} "
                f"horizons={' '.join(str(h) for h in args.horizons)} "
                f"oracle_decode_window={args.oracle_decode_window}",
                flush=True,
            )

        results = evaluate_oracle_corrective_horizons(dyn, tokenizer, eval_loader, device, args, ddp=ddp)
        if wm.is_rank0():
            for horizon in args.horizons:
                print(f"eval horizon={horizon} {wm.format_metrics(results[f'h{horizon}'])}", flush=True)

        if wm.is_rank0() and args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "eval_mode": "oracle_corrective_focus",
                "oracle_uses_next_gt_focus_state": True,
                "oracle_decode_window": int(args.oracle_decode_window),
                "single_rollout_to_max_horizon": True,
                "eval_ckpt": args.eval_ckpt,
                "ckpt_step": int(ckpt.get("step", -1)),
                "ckpt_epoch": int(ckpt.get("epoch", -1)),
                "val_size": len(eval_ds),
                "eval_batch_size": args.eval_batch_size,
                "eval_max_batches": args.eval_max_batches,
                "eval_seq_len": args.eval_seq_len,
                "eval_ctx": args.eval_ctx,
                "horizons": args.horizons,
                "use_ego_actions": bool(args.use_ego_actions),
                "ego_action_source": args.ego_action_source,
                "ego_action_normalization": args.ego_action_normalization,
                "ego_action_clamp": bool(args.ego_action_clamp),
                "dynamics_attend_map": bool(args.dynamics_attend_map),
                "map_cross_every": int(args.map_cross_every),
                "metrics": results,
            }
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"wrote metrics: {out_path}", flush=True)
    finally:
        wm.cleanup_distributed(ddp, device)


if __name__ == "__main__":
    parser = add_args(wm.build_argparser())
    main(parser.parse_args())
