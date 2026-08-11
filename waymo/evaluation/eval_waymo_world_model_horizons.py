"""Evaluate a trained Waymo world model on multiple rollout horizons."""

from __future__ import annotations

import argparse
import json
import math
import sys
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

from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402
from waymo.training.world_model.motion_latent_v1 import MotionLatentDynamicsV1  # noqa: E402


MOTION_LATENT_V1_FORMAT = "waymo_motion_latent_world_model_v1"


def parse_horizons(value: str) -> list[int]:
    horizons = []
    for part in value.replace(",", " ").split():
        horizon = int(part)
        if horizon <= 0:
            raise argparse.ArgumentTypeError("horizons must be positive integers")
        horizons.append(horizon)
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    return horizons


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.description = "Evaluate a trained Waymo latent-space world model."
    parser.add_argument("--eval_ckpt", type=str, required=True, help="World model checkpoint to evaluate.")
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=parse_horizons("10 30 50 80"),
        help="Space- or comma-separated rollout horizons, e.g. '10 30 50 80'.",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Optional path for saved metrics.")
    return parser


def build_dynamics(
    args: argparse.Namespace,
    d_bottleneck: int,
    device: torch.device,
    *,
    map_memory_dim: int | None = None,
) -> torch.nn.Module:
    if args.dynamics_variant == "focus_film":
        return wm.FocusFiLMDynamics(
            d_model=args.d_model_dyn,
            d_bottleneck=d_bottleneck,
            d_spatial=args.d_spatial,
            n_spatial=args.n_spatial,
            n_register=args.n_register,
            n_heads=args.n_heads,
            depth=args.dyn_depth,
            k_max=args.k_max,
            dropout=args.dropout,
            mlp_ratio=args.mlp_ratio,
            scale_pos_embeds=args.scale_pos_embeds,
        ).to(device)
    dyn = wm.Dynamics(
        d_model=args.d_model_dyn,
        d_bottleneck=d_bottleneck,
        d_spatial=args.d_spatial,
        n_spatial=args.n_spatial,
        n_register=args.n_register,
        n_agent=0,
        n_heads=args.n_heads,
        depth=args.dyn_depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        space_mode="wm_agent_isolated",
        scale_pos_embeds=args.scale_pos_embeds,
        action_clamp_inputs=args.ego_action_clamp,
        map_memory_dim=map_memory_dim if args.dynamics_attend_map else None,
        map_cross_every=args.map_cross_every if args.dynamics_attend_map else 0,
    ).to(device)
    wm.freeze_unused_action_mlp(dyn)
    return dyn


def load_dynamics_state(
    dyn: torch.nn.Module,
    ckpt_path: str,
    *,
    ckpt: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if ckpt is None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state_key = "model" if ckpt.get("format") == MOTION_LATENT_V1_FORMAT else "dynamics"
    dyn.load_state_dict(ckpt[state_key], strict=True)
    return ckpt


def build_motion_latent_v1_dynamics(
    ckpt: Dict[str, Any],
    tokenizer: torch.nn.Module,
    tok_args: Dict[str, Any],
    d_bottleneck: int,
    n_spatial: int,
    d_spatial: int,
    device: torch.device,
) -> MotionLatentDynamicsV1:
    """Reconstruct the explicit-q MotionLatent V1 architecture from its checkpoint."""
    model_args = ckpt.get("args", {})
    if not hasattr(tokenizer, "decoder"):
        raise ValueError("MotionLatent V1 evaluation requires the vector tokenizer")
    return MotionLatentDynamicsV1(
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
        time_every=int(model_args.get("time_every", 4)),
        scale_pos_embeds=True,
        action_clamp_inputs=False,
        map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer),
        map_cross_every=int(model_args.get("map_cross_every", 1)),
    ).to(device)


@torch.no_grad()
def sample_motion_latent_v1_sequence(
    dyn: MotionLatentDynamicsV1,
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
    k_max: int,
    kinematic_dt: float,
) -> torch.Tensor:
    """Roll out V1 exactly as trained: one direct d=1 transition per future frame."""
    total = int(z_gt_packed.shape[1])
    ctx_length = max(1, min(int(ctx_length), total - 1))
    horizon = min(int(horizon), total - ctx_length)
    z_history = [z_gt_packed[:, index] for index in range(ctx_length)]
    q_history = [q_gt[:, index] for index in range(ctx_length)]
    time_history = list(range(ctx_length))
    outputs = list(z_history)
    emax = int(round(math.log2(k_max)))

    for target_time in range(ctx_length, ctx_length + horizon):
        keep = min(int(max_context), len(z_history))
        past_z = torch.stack(z_history[-keep:], dim=1)
        past_q = torch.stack(q_history[-keep:], dim=1)
        past_times = time_history[-keep:]
        packed = torch.cat([past_z, torch.randn_like(past_z[:, :1])], dim=1)
        q_sequence = torch.cat([past_q, past_q[:, -1:]], dim=1)
        indices = torch.tensor(past_times + [target_time], device=actions.device, dtype=torch.long)
        action_sequence = actions.index_select(1, indices)
        mask_sequence = act_mask.index_select(1, indices)
        step_idxs = torch.full(packed.shape[:2], emax, device=packed.device, dtype=torch.long)
        signal_idxs = torch.full(packed.shape[:2], k_max - 1, device=packed.device, dtype=torch.long)
        step_idxs[:, -1] = 0
        signal_idxs[:, -1] = 0

        latent_full, _, q_next = dyn(
            action_sequence,
            step_idxs,
            signal_idxs,
            packed,
            q_sequence,
            act_mask=mask_sequence,
            agent_mask=agent_mask,
            map_tokens=map_tokens,
            map_mask=map_mask,
            q_current=q_history[-1],
            action_slots=action_slots,
            kinematic_dt=kinematic_dt,
        )
        if q_next is None:
            raise RuntimeError("MotionLatent V1 did not return q_next during rollout")
        z_next = latent_full[:, -1]
        outputs.append(z_next)
        z_history.append(z_next)
        q_history.append(q_next)
        time_history.append(target_time)
        if len(z_history) > max_context:
            z_history = z_history[-max_context:]
            q_history = q_history[-max_context:]
            time_history = time_history[-max_context:]

    return torch.stack(outputs, dim=1)


@torch.no_grad()
def evaluate_motion_latent_v1(
    dyn: torch.nn.Module,
    tokenizer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    ckpt: Dict[str, Any],
    *,
    ddp: bool,
) -> Dict[str, float]:
    """Use the existing decoder metrics with the explicit-q V1 rollout path."""
    was_training = dyn.training
    dyn.eval()
    model_args = ckpt.get("args", {})
    max_context = int(model_args.get("max_context", args.max_rollout_window))
    k_max = int(model_args.get("k_max", args.k_max))
    kinematic_dt = float(model_args.get("kinematic_dt", args.kinematic_dt))
    totals: Dict[str, float] = {}
    count = 0

    for batch in loader:
        batch = wm.slice_time_window(wm.move_batch(batch, device), args.eval_seq_len, random_start=False)
        actions, act_mask, action_slots = wm.build_ego_action_features(batch, args)
        if actions is None or act_mask is None:
            raise ValueError("MotionLatent V1 evaluation requires --use_ego_actions")
        q_gt = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])
        z_gt, map_tokens, map_mask = wm.encode_batch_dynamics_inputs(tokenizer, batch, return_map=True)
        z_gt_packed = wm.pack_bottleneck_to_spatial(
            z_gt, n_spatial=args.n_spatial, k=args.packing_factor
        )
        z_pred_packed = sample_motion_latent_v1_sequence(
            wm.unwrap_model(dyn),
            z_gt_packed=z_gt_packed,
            q_gt=q_gt,
            actions=actions,
            act_mask=act_mask,
            action_slots=action_slots,
            agent_mask=batch["agent_mask"],
            map_tokens=map_tokens,
            map_mask=map_mask,
            ctx_length=args.eval_ctx,
            horizon=args.eval_horizon,
            max_context=max_context,
            k_max=k_max,
            kinematic_dt=kinematic_dt,
        )
        z_pred = wm.unpack_spatial_to_bottleneck(z_pred_packed, k=args.packing_factor)
        z_decode = z_pred
        if z_pred.shape[1] < z_gt.shape[1]:
            z_decode = torch.cat([z_pred, z_gt[:, z_pred.shape[1] :]], dim=1)
        decoded = wm.decode_batch_z(tokenizer, z_decode, batch)

        score_start = min(int(args.eval_ctx), int(z_pred.shape[1]) - 1)
        score_end = int(z_pred.shape[1])
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
        values = wm.tensor_metrics(metrics)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if args.eval_max_batches > 0 and count >= args.eval_max_batches:
            break

    names = wm.metric_order(totals)
    packed = torch.tensor(
        [float(count)] + [totals.get(name, 0.0) for name in names],
        device=device,
        dtype=torch.float64,
    )
    if ddp:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(packed[0].item()))
    if was_training:
        dyn.train()
    return {name: float(packed[index + 1].item() / total_count) for index, name in enumerate(names)}


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
                raise ValueError("No validation split available. Pass --val_data_dir for full val evaluation.")

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

        ckpt = torch.load(args.eval_ckpt, map_location="cpu")
        is_motion_latent_v1 = ckpt.get("format") == MOTION_LATENT_V1_FORMAT
        if is_motion_latent_v1:
            dyn = build_motion_latent_v1_dynamics(
                ckpt,
                tokenizer,
                tok_args,
                d_bottleneck,
                args.n_spatial,
                args.d_spatial,
                device,
            )
        else:
            dyn = build_dynamics(
                args,
                d_bottleneck,
                device,
                map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None,
            )
        ckpt = load_dynamics_state(dyn, args.eval_ckpt, ckpt=ckpt)
        dyn.eval()
        if ddp:
            dyn = torch.nn.parallel.DistributedDataParallel(
                dyn,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                broadcast_buffers=False,
            )

        if wm.is_rank0():
            print(f"eval_ckpt={args.eval_ckpt}", flush=True)
            print(f"ckpt_step={int(ckpt.get('step', -1))} ckpt_epoch={int(ckpt.get('epoch', -1))}", flush=True)
            print(
                f"checkpoint_format={ckpt.get('format', 'legacy')} "
                f"rollout_mode={'motion_latent_v1_direct_d1' if is_motion_latent_v1 else 'legacy_flow'}",
                flush=True,
            )
            print(
                f"device={device} ddp={ddp} world_size={world_size} val={len(eval_ds)} "
                f"eval_batch_size={args.eval_batch_size} eval_max_batches={args.eval_max_batches}",
                flush=True,
            )
            print(
                f"eval_seq_len={args.eval_seq_len} eval_ctx={args.eval_ctx} "
                f"horizons={' '.join(str(h) for h in args.horizons)}",
                flush=True,
            )

        results: Dict[str, Dict[str, float]] = {}
        for horizon in args.horizons:
            args.eval_horizon = int(horizon)
            if is_motion_latent_v1:
                metrics = evaluate_motion_latent_v1(dyn, tokenizer, eval_loader, device, args, ckpt, ddp=ddp)
            else:
                metrics = wm.evaluate(dyn, tokenizer, eval_loader, device, args, ddp=ddp)
            results[f"h{horizon}"] = metrics
            if wm.is_rank0():
                print(f"eval horizon={horizon} {wm.format_metrics(metrics)}", flush=True)

        if wm.is_rank0() and args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "eval_ckpt": args.eval_ckpt,
                "checkpoint_format": ckpt.get("format", "legacy"),
                "rollout_mode": "motion_latent_v1_direct_d1" if is_motion_latent_v1 else "legacy_flow",
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
                "agent_far_weight": float(args.agent_far_weight),
                "agent_distance_source": args.agent_distance_source,
                "metrics": results,
            }
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(f"wrote metrics: {out_path}", flush=True)
    finally:
        wm.cleanup_distributed(ddp, device)


if __name__ == "__main__":
    parser = add_eval_args(wm.build_argparser())
    main(parser.parse_args())
