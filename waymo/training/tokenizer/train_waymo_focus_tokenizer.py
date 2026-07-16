"""Train focus-only Waymo tokenizer experiments on one GPU."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

WAYMO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WAYMO_ROOT.parent
CORE_ROOT = WAYMO_ROOT / "core"
for path in (REPO_ROOT, CORE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from focus_agent_tokenizer import FocusAgentTokenizer, focus_tokenizer_loss
    from vector_tokenizer_encoder import _collate
    from waymo_vector_dataset import WaymoVectorDataset
except ModuleNotFoundError:
    from waymo.core.focus_agent_tokenizer import FocusAgentTokenizer, focus_tokenizer_loss
    from waymo.core.vector_tokenizer_encoder import _collate
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset


def seed_everything(seed: int) -> None:
    seed = int(seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed_everything(info.seed)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()}


def slice_time_window(batch: Dict[str, Any], window: int, *, random_start: bool) -> Dict[str, Any]:
    if window <= 0:
        return batch
    agents = batch["agents"]
    k = int(batch["agent_mask"].shape[-1])
    agent_time_dim = 2 if agents.shape[1] == k else 1
    total = int(agents.shape[agent_time_dim])
    if total <= int(window):
        return batch
    start = int(torch.randint(0, total - int(window) + 1, (1,)).item()) if random_start else 0
    end = start + int(window)
    out = dict(batch)
    out["agents"] = agents[:, :, start:end] if agent_time_dim == 2 else agents[:, start:end]
    return out


def build_model(args: argparse.Namespace) -> FocusAgentTokenizer:
    return FocusAgentTokenizer(
        representation=args.representation,
        d_model=args.d_model,
        d_latent=args.d_latent,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        map_depth=args.map_depth,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        scale_pos_embeds=args.scale_pos_embeds,
    )


def compute_loss(model_out, batch: Dict[str, Any], args: argparse.Namespace):
    return focus_tokenizer_loss(
        model_out,
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        xy_weight=args.xy_weight,
        velocity_weight=args.velocity_weight,
        yaw_weight=args.yaw_weight,
        valid_weight=args.valid_weight,
        delta_xy_weight=args.delta_xy_weight,
        kinematic_xy_weight=args.kinematic_xy_weight,
        speed_yaw_kinematic_weight=args.speed_yaw_kinematic_weight,
        kinematic_dt=args.kinematic_dt,
    )


def metric_values(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value.detach().float().item()) for key, value in metrics.items()}


def format_metrics(metrics: Dict[str, float]) -> str:
    preferred = [
        "loss_total",
        "focus_xy_mae_m",
        "focus_fde_m",
        "focus_speed_mae_mps",
        "focus_vxvy_mae_mps",
        "focus_yaw_mae_deg",
        "focus_valid_acc",
        "representation_rms",
    ]
    names = [name for name in preferred if name in metrics]
    names += sorted(set(metrics) - set(names))
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in names)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = slice_time_window(move_batch(batch, device), args.time_window, random_start=False)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(
                agents=batch["agents"],
                agent_mask=batch["agent_mask"],
                map_polylines=batch["map_polylines"],
                map_mask=batch["map_mask"],
            )
            _, metrics = compute_loss(out, batch, args)
        for key, value in metric_values(metrics).items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if args.eval_max_batches > 0 and count >= args.eval_max_batches:
            break
    if was_training:
        model.train()
    return {key: value / max(1, count) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "waymo_focus_tokenizer_v1",
            "representation": args.representation,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "args": vars(args),
            "step": int(step),
            "epoch": int(epoch),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "waymo_focus_tokenizer_v1":
        raise ValueError(f"Not a focus tokenizer checkpoint: {path}")
    if ckpt.get("representation") != model.representation_type:
        raise ValueError(
            f"Checkpoint representation={ckpt.get('representation')} does not match model={model.representation_type}"
        )
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["opt"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def make_loader(
    dataset: Iterable,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        collate_fn=_collate,
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        # Each tmux worker exposes exactly one physical GPU through
        # CUDA_VISIBLE_DEVICES, so an unindexed "cuda" means logical cuda:0.
        device = torch.device(f"cuda:{0 if device.index is None else device.index}")
        torch.cuda.set_device(device)
    seed_everything(args.seed)

    train_ds = WaymoVectorDataset(args.data_dir)
    val_ds = WaymoVectorDataset(args.val_data_dir) if args.val_data_dir else None
    train_loader = make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        drop_last=True,
    )
    val_loader = None if val_ds is None else make_loader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
        drop_last=False,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    step = 0
    start_epoch = 0
    if args.resume:
        step, start_epoch = load_checkpoint(Path(args.resume), model=model, optimizer=optimizer)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": torch.float32}[args.amp_dtype]
    use_amp = device.type == "cuda" and args.amp_dtype != "none"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is not installed; omit --wandb or install it") from exc
        wandb_run = wandb
        wandb_run.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity, config=vars(args))

    params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(
        f"device={device} representation={args.representation} raw_features=True "
        f"train={len(train_ds)} val={0 if val_ds is None else len(val_ds)} params={params:,}"
    )
    print(
        f"d_model={args.d_model} d_latent={args.d_latent if args.representation != 'agent_token' else 'none'} "
        f"depth={args.depth} decoder_depth={args.decoder_depth} map_depth={args.map_depth}"
    )

    ckpt_dir = Path(args.ckpt_dir)
    latest = ckpt_dir / "latest.pt"
    t0 = time.time()
    stop = False
    last_epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch + 1
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = slice_time_window(move_batch(batch, device), args.time_window, random_start=args.random_time_window_start)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    agents=batch["agents"],
                    agent_mask=batch["agent_mask"],
                    map_polylines=batch["map_polylines"],
                    map_mask=batch["map_mask"],
                )
                loss, metrics = compute_loss(out, batch, args)
                scaled_loss = loss / max(1, args.grad_accum)
            scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % max(1, args.grad_accum) != 0:
                continue
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step == 1 or (args.log_every > 0 and step % args.log_every == 0):
                values = metric_values(metrics)
                elapsed = max(1e-6, time.time() - t0)
                print(f"step={step} epoch={epoch + 1} {format_metrics(values)} steps_per_sec={step / elapsed:.3f}")
                if wandb_run is not None:
                    wandb_run.log({f"train/{key}": value for key, value in values.items()}, step=step)

            if val_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                values = evaluate(model, val_loader, device, args, amp_dtype=amp_dtype, use_amp=use_amp)
                print(f"eval step={step} {format_metrics(values)}")
                if wandb_run is not None:
                    wandb_run.log({f"val/{key}": value for key, value in values.items()}, step=step)

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"step_{step:08d}.pt",
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    step=step,
                    epoch=epoch,
                )
                save_checkpoint(latest, model=model, optimizer=optimizer, args=args, step=step, epoch=epoch)

            if args.max_steps > 0 and step >= args.max_steps:
                stop = True
                break

        save_checkpoint(latest, model=model, optimizer=optimizer, args=args, step=step, epoch=epoch + 1)
        if stop:
            break

    save_checkpoint(
        ckpt_dir / f"final_step_{step:08d}.pt",
        model=model,
        optimizer=optimizer,
        args=args,
        step=step,
        epoch=last_epoch,
    )
    if wandb_run is not None:
        wandb_run.finish()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train focus-only Waymo agent-token or bottleneck-z tokenizer")
    parser.add_argument("--representation", choices=["agent_token", "latent_z16", "latent_z64"], required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=150_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--time_window", type=int, default=32)
    parser.add_argument("--random_time_window_start", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_latent", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--map_depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--scale_pos_embeds", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--xy_weight", type=float, default=1.0)
    parser.add_argument("--velocity_weight", type=float, default=0.5)
    parser.add_argument("--yaw_weight", type=float, default=0.5)
    parser.add_argument("--valid_weight", type=float, default=0.2)
    parser.add_argument("--delta_xy_weight", type=float, default=0.0)
    parser.add_argument("--kinematic_xy_weight", type=float, default=5.0)
    parser.add_argument("--speed_yaw_kinematic_weight", type=float, default=2.0)
    parser.add_argument("--kinematic_dt", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_max_batches", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=500)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="waymo-focus-tokenizer")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
