#!/usr/bin/env python3
"""Pretrain the frozen lightweight P(q|z) used by motion-latent V1."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

from waymo.core.vector_tokenizer_encoder import _collate
from waymo.core.waymo_vector_dataset import WaymoVectorDataset
from waymo.training.world_model import train_waymo_world_model as wm
from waymo.training.world_model.motion_latent_v1 import (
    LightweightAgentSemanticReader,
    agent_semantic_targets,
    semantic_reader_loss,
)


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def save_checkpoint(
    path: Path,
    *,
    reader: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    best_val: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "waymo_lightweight_agent_semantic_reader_v1",
            "reader": unwrap(reader).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "step": int(step),
            "epoch": int(epoch),
            "best_val": float(best_val),
        },
        tmp,
    )
    tmp.replace(path)


def select_reader_frame(
    z: torch.Tensor,
    agents: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    random_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = wm.agents_to_btkf(agents, agent_mask)
    bsz, time_steps = z.shape[:2]
    if random_frame:
        frame = torch.randint(0, time_steps, (bsz,), device=z.device)
    else:
        frame = torch.full((bsz,), time_steps - 1, device=z.device, dtype=torch.long)
    rows = torch.arange(bsz, device=z.device)
    return z[rows, frame][:, None], q[rows, frame][:, None]


@torch.no_grad()
def evaluate(
    reader: torch.nn.Module,
    tokenizer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    ddp: bool,
) -> dict[str, float]:
    reader.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.eval_batches:
            break
        batch = wm.slice_time_window(wm.move_batch(batch, device), args.reader_context, random_start=False)
        z = wm.encode_batch_z(tokenizer, batch)
        z_frame, q_frame = select_reader_frame(z, batch["agents"], batch["agent_mask"], random_frame=False)
        target_cont, target_valid = agent_semantic_targets(q_frame)
        pred = reader(z_frame, agent_mask=batch["agent_mask"])
        _, metrics = semantic_reader_loss(pred, target_cont, target_valid, batch["agent_mask"])
        reduced = wm.reduce_metric_dict(metrics, device, ddp)
        for key, value in reduced.items():
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
    reader.train()
    return {key: value / max(1, count) for key, value in sums.items()}


def train(args: argparse.Namespace) -> None:
    ddp, rank, world_size, local_rank = wm.init_distributed()
    device = torch.device(f"cuda:{local_rank}" if ddp and torch.cuda.is_available() else (args.device or "cuda"))
    wm.seed_everything(args.seed + rank)

    train_set = WaymoVectorDataset(args.data_dir)
    val_set = WaymoVectorDataset(args.val_data_dir)
    train_sampler = DistributedSampler(train_set, world_size, rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_set, world_size, rank, shuffle=False) if ddp else None
    loader_kwargs: dict[str, Any] = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=wm.worker_init_fn,
        collate_fn=_collate,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.eval_batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    if not hasattr(tokenizer, "decoder"):
        raise ValueError("Semantic reader V1 requires the vector tokenizer with a decoder")
    decoder = tokenizer.decoder
    reader = LightweightAgentSemanticReader(
        d_bottleneck=int(tok_args.get("d_bottleneck", decoder.up_proj.in_features)),
        d_model=int(decoder.d_model),
        n_heads=int(tok_args.get("n_heads", 4)),
        n_latents=int(tok_args.get("n_latents", decoder.n_latents)),
        n_agents=int(decoder.n_agents),
        depth=args.reader_depth,
        dropout=args.dropout,
        mlp_ratio=float(tok_args.get("mlp_ratio", 4.0)),
        scale_pos_embeds=bool(tok_args.get("scale_pos_embeds", True)),
    ).to(device)
    reader.init_from_tokenizer_decoder(decoder)

    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": torch.float32}[args.amp_dtype]
    scaler = GradScaler(device="cuda", enabled=use_amp and amp_dtype == torch.float16)
    step, start_epoch, best_val = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        reader.load_state_dict(checkpoint["reader"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("best_val", best_val))

    if ddp:
        reader = torch.nn.parallel.DistributedDataParallel(
            reader,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

    wandb_run = None
    if args.wandb and wm.is_rank0():
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    if wm.is_rank0():
        params = sum(p.numel() for p in unwrap(reader).parameters() if p.requires_grad)
        print(
            f"semantic_reader device={device} ddp={ddp} world_size={world_size} "
            f"train={len(train_set)} val={len(val_set)} params={params:,} "
            f"local_batch={args.batch_size} global_batch={args.batch_size * world_size} "
            f"context={args.reader_context} max_steps={args.max_steps}"
        )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    epoch = start_epoch
    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        reader.train()
        for batch in train_loader:
            if step >= args.max_steps:
                break
            batch = wm.slice_time_window(
                wm.move_batch(batch, device), args.reader_context, random_start=True
            )
            with torch.no_grad():
                z = wm.encode_batch_z(tokenizer, batch)
                z_frame, q_frame = select_reader_frame(
                    z, batch["agents"], batch["agent_mask"], random_frame=True
                )
                target_cont, target_valid = agent_semantic_targets(q_frame)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = reader(z_frame, agent_mask=batch["agent_mask"])
                loss, metrics = semantic_reader_loss(pred, target_cont, target_valid, batch["agent_mask"])
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(reader.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 1 or (args.log_every > 0 and step % args.log_every == 0):
                values = wm.reduce_metric_dict(metrics, device, ddp)
                if wm.is_rank0():
                    elapsed = max(1e-6, time.time() - start_time)
                    print(f"reader step={step} {wm.format_metrics(values)} steps_per_sec={step / elapsed:.3f}")
                    if wandb_run is not None:
                        wandb_run.log({f"train/{k}": v for k, v in values.items()}, step=step)

            if args.eval_every > 0 and step % args.eval_every == 0:
                val = evaluate(reader, tokenizer, val_loader, device, args, ddp=ddp)
                if wm.is_rank0():
                    print(f"reader eval step={step} {wm.format_metrics(val)}")
                    if wandb_run is not None:
                        wandb_run.log({f"val/{k}": v for k, v in val.items()}, step=step)
                    if val.get("loss_total", float("inf")) < best_val:
                        best_val = val["loss_total"]
                        save_checkpoint(
                            ckpt_dir / "best.pt", reader=reader, optimizer=optimizer, scaler=scaler,
                            args=args, step=step, epoch=epoch, best_val=best_val,
                        )

            if wm.is_rank0() and args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{step:08d}.pt", reader=reader, optimizer=optimizer,
                    scaler=scaler, args=args, step=step, epoch=epoch, best_val=best_val,
                )
                save_checkpoint(
                    ckpt_dir / "latest.pt", reader=reader, optimizer=optimizer,
                    scaler=scaler, args=args, step=step, epoch=epoch, best_val=best_val,
                )
        epoch += 1

    if wm.is_rank0():
        save_checkpoint(
            ckpt_dir / "latest.pt", reader=reader, optimizer=optimizer, scaler=scaler,
            args=args, step=step, epoch=epoch, best_val=best_val,
        )
        if not (ckpt_dir / "best.pt").exists():
            save_checkpoint(
                ckpt_dir / "best.pt", reader=reader, optimizer=optimizer, scaler=scaler,
                args=args, step=step, epoch=epoch, best_val=best_val,
            )
    if wandb_run is not None:
        wandb_run.finish()
    wm.cleanup_distributed(ddp, device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", nargs="+", required=True)
    parser.add_argument("--val_data_dir", nargs="+", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--reader_context", type=int, default=32)
    parser.add_argument("--reader_depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=20_000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="waymo-world-model")
    parser.add_argument("--wandb_run_name", default="waymo_semantic_reader_v1")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
