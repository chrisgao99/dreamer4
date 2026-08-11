#!/usr/bin/env python3
"""Train motion-head + latent-head Waymo V1 with full on-policy rollouts."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

from dreamer4.model import pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck
from waymo.core.vector_tokenizer_encoder import _collate
from waymo.core.waymo_vector_dataset import WaymoVectorDataset
from waymo.training.world_model import train_waymo_world_model as wm
from waymo.training.world_model.motion_latent_v1 import (
    LightweightAgentSemanticReader,
    MotionLatentDynamicsV1,
    latent_q_consistency_loss,
    motion_residual_loss,
    motion_targets,
)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def bootstrap_weight(step: int, *, start: int, ramp_end: int, maximum: float) -> float:
    if step < start:
        return 0.0
    if step >= ramp_end:
        return float(maximum)
    return float(maximum) * float(step - start) / max(1, ramp_end - start)


def context_one_probability(step: int) -> float:
    if step < 30_000:
        return 0.0
    if step < 150_000:
        return 0.8 * float(step - 30_000) / 120_000.0
    return 0.8


def choose_context(step: int, device: torch.device, ddp: bool) -> tuple[int, float]:
    probability = context_one_probability(step)
    if wm.is_rank0():
        value = 1 if torch.rand(()) < probability else 11
        tensor = torch.tensor(value, device=device, dtype=torch.long)
    else:
        tensor = torch.zeros((), device=device, dtype=torch.long)
    if ddp:
        dist.broadcast(tensor, src=0)
    return int(tensor.item()), probability


def load_semantic_reader(
    path: str,
    *,
    tokenizer: torch.nn.Module,
    tok_args: dict[str, Any],
    device: torch.device,
) -> LightweightAgentSemanticReader:
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != "waymo_lightweight_agent_semantic_reader_v1":
        raise ValueError(f"Unexpected semantic reader format in {path}")
    decoder = tokenizer.decoder
    reader_args = checkpoint.get("args", {})
    reader = LightweightAgentSemanticReader(
        d_bottleneck=int(tok_args.get("d_bottleneck", decoder.up_proj.in_features)),
        d_model=int(decoder.d_model),
        n_heads=int(tok_args.get("n_heads", 4)),
        n_latents=int(tok_args.get("n_latents", decoder.n_latents)),
        n_agents=int(decoder.n_agents),
        depth=int(reader_args.get("reader_depth", 2)),
        dropout=float(reader_args.get("dropout", 0.05)),
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        scale_pos_embeds=bool(tok_args.get("scale_pos_embeds", True)),
    ).to(device)
    reader.load_state_dict(checkpoint["reader"], strict=True)
    reader.eval()
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    return reader


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "waymo_motion_latent_world_model_v1",
            "step": int(step),
            "epoch": int(epoch),
            "args": vars(args),
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    unwrap(model).load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("step", 0)), int(checkpoint.get("epoch", 0))


def load_model_weights(path: str, *, model: torch.nn.Module) -> None:
    """Initialize model weights without restoring optimizer or training step."""
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != "waymo_motion_latent_world_model_v1":
        raise ValueError(f"Unexpected motion-latent checkpoint format in {path}")
    unwrap(model).load_state_dict(checkpoint["model"], strict=True)


def make_prediction_inputs(
    z_history: list[torch.Tensor],
    q_history: list[torch.Tensor],
    time_history: list[int],
    *,
    target_time: int,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    max_context: int,
    k_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = min(int(max_context), len(z_history))
    past_z = torch.stack(z_history[-keep:], dim=1)
    past_q = torch.stack(q_history[-keep:], dim=1)
    past_times = time_history[-keep:]
    noise = torch.randn_like(past_z[:, :1])
    packed = torch.cat([past_z, noise], dim=1)
    q_sequence = torch.cat([past_q, past_q[:, -1:]], dim=1)
    indices = torch.tensor(past_times + [int(target_time)], device=actions.device, dtype=torch.long)
    actions_sequence = actions.index_select(1, indices)
    mask_sequence = act_mask.index_select(1, indices)
    emax = int(round(math.log2(k_max)))
    step_idxs = torch.full(packed.shape[:2], emax, device=packed.device, dtype=torch.long)
    signal_idxs = torch.full(packed.shape[:2], k_max - 1, device=packed.device, dtype=torch.long)
    step_idxs[:, -1] = 0
    signal_idxs[:, -1] = 0
    return packed, q_sequence, actions_sequence, mask_sequence, step_idxs, signal_idxs


@torch.no_grad()
def two_half_step_bootstrap_target(
    model: MotionLatentDynamicsV1,
    *,
    packed_main: torch.Tensor,
    q_sequence: torch.Tensor,
    actions_sequence: torch.Tensor,
    mask_sequence: torch.Tensor,
    step_main: torch.Tensor,
    signal_main: torch.Tensor,
    q_current: torch.Tensor,
    action_slots: torch.Tensor,
    agent_mask: torch.Tensor,
    map_tokens: Optional[torch.Tensor],
    map_mask: Optional[torch.Tensor],
    subset: int,
    k_max: int,
    kinematic_dt: float,
) -> torch.Tensor:
    """Detached two-half-step teacher for the auxiliary 50% bootstrap subset."""
    sl = slice(0, subset)
    packed0 = packed_main[sl]
    qseq = q_sequence[sl]
    actionseq = actions_sequence[sl]
    maskseq = mask_sequence[sl]
    step_half = step_main[sl].clone()
    signal0 = signal_main[sl].clone()
    step_half[:, -1] = 1
    signal0[:, -1] = 0
    kwargs = dict(
        act_mask=maskseq,
        agent_mask=agent_mask[sl],
        map_tokens=None if map_tokens is None else map_tokens[sl],
        map_mask=None if map_mask is None else map_mask[sl],
        q_current=q_current[sl],
        action_slots=action_slots[sl],
        kinematic_dt=kinematic_dt,
    )
    half1, _, _ = model(actionseq, step_half, signal0, packed0, qseq, **kwargs)
    z0 = packed0[:, -1]
    first_endpoint = half1[:, -1]
    midpoint = z0 + 0.5 * (first_endpoint - z0)
    packed_mid = torch.cat([packed0[:, :-1], midpoint[:, None]], dim=1)
    signal_half = signal0.clone()
    signal_half[:, -1] = k_max // 2
    half2, _, _ = model(actionseq, step_half, signal_half, packed_mid, qseq, **kwargs)
    second_endpoint = half2[:, -1]
    velocity_first = first_endpoint - z0
    velocity_second = (second_endpoint - midpoint) / 0.5
    return 0.5 * (velocity_first + velocity_second)


def exact_rollout_loss(
    model: torch.nn.Module,
    reader: LightweightAgentSemanticReader,
    *,
    z_gt_packed: torch.Tensor,
    q_gt: torch.Tensor,
    actions: torch.Tensor,
    act_mask: torch.Tensor,
    action_slots: torch.Tensor,
    agent_mask: torch.Tensor,
    map_tokens: Optional[torch.Tensor],
    map_mask: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fixed-parameter ctx1 rollout matching the V1 evaluation path."""
    z_history = [z_gt_packed[:, 0]]
    q_history = [q_gt[:, 0]]
    time_history = [0]
    losses: list[torch.Tensor] = []
    metric_sums: dict[str, torch.Tensor] = {}

    for target_time in range(1, args.rollout_end + 1):
        packed, q_sequence, action_sequence, mask_sequence, step_idxs, signal_idxs = make_prediction_inputs(
            z_history,
            q_history,
            time_history,
            target_time=target_time,
            actions=actions,
            act_mask=act_mask,
            max_context=args.max_context,
            k_max=args.k_max,
        )
        q_current = q_history[-1]
        latent_full, motion_full, q_next = model(
            action_sequence,
            step_idxs,
            signal_idxs,
            packed,
            q_sequence,
            act_mask=mask_sequence,
            agent_mask=agent_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            q_current=q_current,
            action_slots=action_slots,
            kinematic_dt=args.kinematic_dt,
        )
        if q_next is None:
            raise RuntimeError("Model did not return q_next")
        z_next = latent_full[:, -1]
        latent_loss = F.mse_loss(z_next.float(), z_gt_packed[:, target_time].float())
        target_motion = motion_targets(q_current, q_gt[:, target_time], dt=args.kinematic_dt)
        motion_loss, motion_metrics = motion_residual_loss(
            motion_full[:, -1],
            target_motion,
            q_current,
            q_gt[:, target_time],
            agent_mask,
            action_slots,
            validity_weight=args.motion_validity_weight,
        )
        z_next_unpacked = unpack_spatial_to_bottleneck(z_next[:, None], k=args.packing_factor)
        semantic_pred = reader(z_next_unpacked, agent_mask=agent_mask)
        consistency_loss, consistency_metrics = latent_q_consistency_loss(semantic_pred, q_next, agent_mask)
        step_loss = latent_loss + args.motion_weight * motion_loss + args.consistency_weight * consistency_loss
        losses.append(step_loss)

        step_metrics = {
            "latent_mse": latent_loss.detach(),
            "motion_loss": motion_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            **motion_metrics,
            "consistency_continuous": consistency_metrics["loss_continuous"],
            "consistency_valid": consistency_metrics["loss_valid"],
        }
        for name, value in step_metrics.items():
            metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value.detach()

        # Do not detach: later losses backpropagate through the same predicted
        # z/q history used by the fixed-checkpoint evaluation rollout.
        z_history.append(z_next)
        q_history.append(q_next)
        time_history.append(target_time)
        if len(z_history) > args.max_context:
            z_history = z_history[-args.max_context :]
            q_history = q_history[-args.max_context :]
            time_history = time_history[-args.max_context :]

    loss = torch.stack(losses).mean()
    count = float(len(losses))
    metrics = {name: value / count for name, value in metric_sums.items()}
    metrics.update(
        {
            "loss_total": loss.detach(),
            "ctx": torch.tensor(1.0, device=loss.device),
            "rollout_target_time": torch.tensor(float(args.rollout_end), device=loss.device),
        }
    )
    return loss, metrics


def train(args: argparse.Namespace) -> None:
    ddp, rank, world_size, local_rank = wm.init_distributed()
    device = torch.device(f"cuda:{local_rank}" if ddp and torch.cuda.is_available() else (args.device or "cuda"))
    wm.seed_everything(args.seed + rank)

    train_set = WaymoVectorDataset(args.data_dir)
    sampler = DistributedSampler(train_set, world_size, rank, shuffle=True) if ddp else None
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=wm.worker_init_fn,
        collate_fn=_collate,
    )

    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    if not hasattr(tokenizer, "decoder"):
        raise ValueError("Motion-latent V1 requires the vector tokenizer")
    reader = load_semantic_reader(args.semantic_reader_ckpt, tokenizer=tokenizer, tok_args=tok_args, device=device)
    n_latents = int(tok_args.get("n_latents", tokenizer.decoder.n_latents))
    d_bottleneck = int(tok_args.get("d_bottleneck", tokenizer.decoder.up_proj.in_features))
    if n_latents % args.packing_factor:
        raise ValueError("n_latents must be divisible by packing_factor")
    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    model = MotionLatentDynamicsV1(
        d_model=args.d_model,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=args.n_register,
        n_agents=tokenizer.decoder.n_agents,
        n_heads=args.n_heads,
        depth=args.depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        scale_pos_embeds=True,
        action_clamp_inputs=False,
        map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer),
        map_cross_every=args.map_cross_every,
    ).to(device)
    if args.init_ckpt and args.resume:
        raise ValueError("--init_ckpt and --resume are mutually exclusive")
    if args.init_ckpt:
        load_model_weights(args.init_ckpt, model=model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": torch.float32}[args.amp_dtype]
    scaler = GradScaler(device="cuda", enabled=use_amp and amp_dtype == torch.float16)

    step, start_epoch = 0, 0
    if args.resume:
        step, start_epoch = load_checkpoint(args.resume, model=model, optimizer=optimizer, scaler=scaler)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    wandb_run = None
    if args.wandb and wm.is_rank0():
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    if wm.is_rank0():
        trainable = sum(p.numel() for p in unwrap(model).parameters() if p.requires_grad)
        reader_params = sum(p.numel() for p in reader.parameters())
        print(
            f"motion_latent_v1 device={device} ddp={ddp} world_size={world_size} train={len(train_set)} "
            f"local_batch={args.batch_size} global_batch={args.batch_size * world_size} "
            f"model_params={trainable:,} frozen_reader_params={reader_params:,}"
        )
        if args.train_mode == "exact_rollout":
            print(
                f"max_steps={args.max_steps} exact_rollout_ctx=1 exact_rollout_horizon={args.rollout_end} "
                f"max_context={args.max_context} init_ckpt={args.init_ckpt} "
                f"tokenizer_chunk_window={args.tokenizer_chunk_window} "
                f"tokenizer_chunk_stride={args.tokenizer_chunk_stride} "
                f"motion_weight={args.motion_weight} consistency_weight={args.consistency_weight}"
            )
        else:
            print(
                f"max_steps={args.max_steps} full_rollout_to={args.rollout_end} max_context={args.max_context} "
                "curriculum=ctx11@0-30k,linear_to_ctx1_80%@150k,ctx1/ctx11=80/20@150k-1m "
                f"bootstrap=aux50% start={args.bootstrap_start} ramp_end={args.bootstrap_ramp_end} "
                f"max_weight={args.bootstrap_weight} consistency_weight={args.consistency_weight}"
            )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    start_time = time.time()
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= args.max_steps:
                break
            batch = wm.slice_time_window(wm.move_batch(batch, device), args.rollout_end + 1, random_start=False)
            q_gt = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
            if q_gt.shape[1] < args.rollout_end + 1:
                raise ValueError(f"Scene has only {q_gt.shape[1]} frames, need {args.rollout_end + 1}")
            actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
            with torch.no_grad():
                z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs_for_world_model(
                    tokenizer,
                    batch,
                    args,
                    return_map=True,
                )
                z_gt_packed = pack_bottleneck_to_spatial(z_gt, n_spatial=n_spatial, k=args.packing_factor)

            if args.train_mode == "exact_rollout":
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss, metrics = exact_rollout_loss(
                        model,
                        reader,
                        z_gt_packed=z_gt_packed,
                        q_gt=q_gt,
                        actions=actions,
                        act_mask=act_mask,
                        action_slots=action_slots,
                        agent_mask=batch["agent_mask"],
                        map_tokens=map_tokens,
                        map_mask=map_mask,
                        args=args,
                    )
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step == 1 or (args.log_every > 0 and step % args.log_every == 0):
                    values = wm.reduce_metric_dict(metrics, device, ddp)
                    if wm.is_rank0():
                        elapsed = max(1e-6, time.time() - start_time)
                        print(f"step={step} epoch={epoch + 1} {wm.format_metrics(values)} steps_per_sec={step / elapsed:.3f}")
                        if wandb_run is not None:
                            wandb_run.log({f"train/{k}": v for k, v in values.items()}, step=step)
                if wm.is_rank0() and args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(
                        ckpt_dir / f"step_{step:08d}.pt", model=model, optimizer=optimizer,
                        scaler=scaler, args=args, step=step, epoch=epoch,
                    )
                    save_checkpoint(
                        latest, model=model, optimizer=optimizer, scaler=scaler,
                        args=args, step=step, epoch=epoch,
                    )
                continue

            context, ctx1_probability = choose_context(step, device, ddp)
            z_history = [z_gt_packed[:, index].detach() for index in range(context)]
            q_history = [q_gt[:, index].detach() for index in range(context)]
            time_history = list(range(context))

            for target_time in range(context, args.rollout_end + 1):
                if step >= args.max_steps:
                    break
                packed, q_sequence, action_sequence, mask_sequence, step_idxs, signal_idxs = make_prediction_inputs(
                    z_history,
                    q_history,
                    time_history,
                    target_time=target_time,
                    actions=actions,
                    act_mask=act_mask,
                    max_context=args.max_context,
                    k_max=args.k_max,
                )
                q_current = q_history[-1]
                lambda_boot = bootstrap_weight(
                    step,
                    start=args.bootstrap_start,
                    ramp_end=args.bootstrap_ramp_end,
                    maximum=args.bootstrap_weight,
                )
                subset = packed.shape[0] // 2
                bootstrap_target = None
                if lambda_boot > 0 and subset > 0:
                    with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        bootstrap_target = two_half_step_bootstrap_target(
                            unwrap(model),
                            packed_main=packed,
                            q_sequence=q_sequence,
                            actions_sequence=action_sequence,
                            mask_sequence=mask_sequence,
                            step_main=step_idxs,
                            signal_main=signal_idxs,
                            q_current=q_current,
                            action_slots=action_slots,
                            agent_mask=batch["agent_mask"],
                            map_tokens=map_tokens,
                            map_mask=map_mask,
                            subset=subset,
                            k_max=args.k_max,
                            kinematic_dt=args.kinematic_dt,
                        )

                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    latent_full, motion_full, q_next = model(
                        action_sequence,
                        step_idxs,
                        signal_idxs,
                        packed,
                        q_sequence,
                        act_mask=mask_sequence,
                        agent_mask=batch["agent_mask"],
                        map_tokens=map_tokens,
                        map_mask=map_mask,
                        q_current=q_current,
                        action_slots=action_slots,
                        kinematic_dt=args.kinematic_dt,
                    )
                    if q_next is None:
                        raise RuntimeError("Model did not return q_next")
                    z_next = latent_full[:, -1]
                    latent_loss = F.mse_loss(z_next.float(), z_gt_packed[:, target_time].float())
                    target_motion = motion_targets(q_current, q_gt[:, target_time], dt=args.kinematic_dt)
                    motion_loss, motion_metrics = motion_residual_loss(
                        motion_full[:, -1],
                        target_motion,
                        q_current,
                        q_gt[:, target_time],
                        batch["agent_mask"],
                        action_slots,
                        validity_weight=args.motion_validity_weight,
                    )
                    z_next_unpacked = unpack_spatial_to_bottleneck(z_next[:, None], k=args.packing_factor)
                    semantic_pred = reader(z_next_unpacked, agent_mask=batch["agent_mask"])
                    consistency_loss, consistency_metrics = latent_q_consistency_loss(
                        semantic_pred, q_next, batch["agent_mask"]
                    )
                    bootstrap_loss = z_next.new_zeros((), dtype=torch.float32)
                    if bootstrap_target is not None:
                        z0 = packed[:subset, -1]
                        main_velocity = z_next[:subset] - z0
                        bootstrap_loss = F.mse_loss(main_velocity.float(), bootstrap_target.float())
                    loss = (
                        latent_loss
                        + args.motion_weight * motion_loss
                        + args.consistency_weight * consistency_loss
                        + lambda_boot * bootstrap_loss
                    )

                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                z_history.append(z_next.detach())
                q_history.append(q_next.detach())
                time_history.append(target_time)
                if len(z_history) > args.max_context:
                    z_history = z_history[-args.max_context :]
                    q_history = q_history[-args.max_context :]
                    time_history = time_history[-args.max_context :]

                metrics = {
                    "loss_total": loss.detach(),
                    "latent_mse": latent_loss.detach(),
                    "bootstrap_mse": bootstrap_loss.detach(),
                    "lambda_bootstrap": torch.tensor(lambda_boot, device=device),
                    "motion_loss": motion_loss.detach(),
                    "consistency_loss": consistency_loss.detach(),
                    "ctx": torch.tensor(float(context), device=device),
                    "ctx1_probability": torch.tensor(ctx1_probability, device=device),
                    "rollout_target_time": torch.tensor(float(target_time), device=device),
                    **motion_metrics,
                    "consistency_continuous": consistency_metrics["loss_continuous"],
                    "consistency_valid": consistency_metrics["loss_valid"],
                }
                if step == 1 or (args.log_every > 0 and step % args.log_every == 0):
                    values = wm.reduce_metric_dict(metrics, device, ddp)
                    if wm.is_rank0():
                        elapsed = max(1e-6, time.time() - start_time)
                        print(f"step={step} epoch={epoch + 1} {wm.format_metrics(values)} steps_per_sec={step / elapsed:.3f}")
                        if wandb_run is not None:
                            wandb_run.log({f"train/{k}": v for k, v in values.items()}, step=step)

                if wm.is_rank0() and args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(
                        ckpt_dir / f"step_{step:08d}.pt", model=model, optimizer=optimizer,
                        scaler=scaler, args=args, step=step, epoch=epoch,
                    )
                    save_checkpoint(
                        latest, model=model, optimizer=optimizer, scaler=scaler,
                        args=args, step=step, epoch=epoch,
                    )
        epoch += 1

    if wm.is_rank0():
        save_checkpoint(
            ckpt_dir / f"final_step_{step:08d}.pt", model=model, optimizer=optimizer,
            scaler=scaler, args=args, step=step, epoch=epoch,
        )
        save_checkpoint(
            latest, model=model, optimizer=optimizer, scaler=scaler,
            args=args, step=step, epoch=epoch,
        )
    if wandb_run is not None:
        wandb_run.finish()
    wm.cleanup_distributed(ddp, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", nargs="+", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--semantic_reader_ckpt", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init_ckpt", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch; two GPUs give global batch 8")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1_000_000)
    parser.add_argument("--train_mode", choices=("online_step", "exact_rollout"), default="online_step")
    parser.add_argument("--rollout_end", type=int, default=90)
    parser.add_argument("--max_context", type=int, default=11)
    parser.add_argument(
        "--tokenizer_chunk_window",
        type=int,
        default=32,
        help="Encode sequences as overlapping windows (default: 32 timesteps; set <=0 to disable).",
    )
    parser.add_argument(
        "--tokenizer_chunk_stride",
        type=int,
        default=30,
        help="Stride for tokenizer chunks; the final chunk is shifted to end at the sequence end.",
    )
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--time_every", type=int, default=4)
    parser.add_argument("--map_cross_every", type=int, default=1)
    parser.add_argument("--packing_factor", type=int, default=2)
    parser.add_argument("--n_register", type=int, default=8)
    parser.add_argument("--k_max", type=int, default=64)
    parser.add_argument("--kinematic_dt", type=float, default=0.1)
    parser.add_argument("--motion_weight", type=float, default=1.0)
    parser.add_argument("--motion_validity_weight", type=float, default=0.2)
    parser.add_argument("--consistency_weight", type=float, default=0.1)
    parser.add_argument("--bootstrap_start", type=int, default=20_000)
    parser.add_argument("--bootstrap_ramp_end", type=int, default=60_000)
    parser.add_argument("--bootstrap_weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--use_ego_actions", action="store_true", default=True)
    parser.add_argument("--ego_action_source", choices=("focus", "sdc"), default="focus")
    parser.add_argument("--ego_action_normalization", choices=("raw", "scaled"), default="raw")
    parser.add_argument("--ego_action_xy_scale", type=float, default=5.0)
    parser.add_argument("--ego_action_yaw_scale", type=float, default=math.pi)
    parser.add_argument("--ego_action_speed_scale", type=float, default=30.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="waymo-world-model")
    parser.add_argument("--wandb_run_name", default="waymo_motion_latent_v1_ctx_curriculum_b8_1m")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
