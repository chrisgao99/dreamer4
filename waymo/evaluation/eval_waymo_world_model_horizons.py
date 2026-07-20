"""Evaluate a trained Waymo world model on multiple rollout horizons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, DistributedSampler

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


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


def load_dynamics_state(dyn: torch.nn.Module, ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    dyn.load_state_dict(ckpt["dynamics"], strict=True)
    return ckpt


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

        dyn = build_dynamics(
            args,
            d_bottleneck,
            device,
            map_memory_dim=wm.tokenizer_map_memory_dim(tokenizer) if args.dynamics_attend_map else None,
        )
        ckpt = load_dynamics_state(dyn, args.eval_ckpt)
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
            metrics = wm.evaluate(dyn, tokenizer, eval_loader, device, args, ddp=ddp)
            results[f"h{horizon}"] = metrics
            if wm.is_rank0():
                print(f"eval horizon={horizon} {wm.format_metrics(metrics)}", flush=True)

        if wm.is_rank0() and args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
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
