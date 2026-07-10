"""Train the Waymo vector tokenizer on filtered NPZ scenes.

This trains Decoder 1 with either Encoder A or Encoder B:

- Encoder A input: selected agents, repeated static map tokens, traffic lights
- Encoder B input: selected agents/lights as dynamic tokens, static map memory queried by agents/latents
- bottleneck: z with shape (B,T,n_latents,d_bottleneck)
- decoder input: z plus learned agent/light query tokens
- targets: selected-agent state reconstruction and traffic-light state/validity
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler, random_split

WAYMO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = WAYMO_ROOT.parent
CORE_ROOT = WAYMO_ROOT / "core"
for path in (REPO_ROOT, CORE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from vector_tokenizer_decoder import (
        VectorBlockCausalTokenizerDecoder,
        VectorTokenizer,
        _slice_time_window,
        vector_tokenizer_reconstruction_loss,
    )
    from waymo_vector_dataset import WaymoVectorDataset
except ModuleNotFoundError:
    from waymo.core.vector_tokenizer_encoder import VectorBlockCausalEncoder, VectorStaticMapQueryEncoder, _collate
    from waymo.core.vector_tokenizer_decoder import (
        VectorBlockCausalTokenizerDecoder,
        VectorTokenizer,
        _slice_time_window,
        vector_tokenizer_reconstruction_loss,
    )
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset


def seed_everything(seed: int) -> None:
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def get_dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


def cleanup_distributed(ddp: bool, device: torch.device) -> None:
    if ddp:
        if device.type == "cuda":
            dist.barrier(device_ids=[int(device.index if device.index is not None else torch.cuda.current_device())])
        else:
            dist.barrier()
        dist.destroy_process_group()


def worker_init_fn(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        seed_everything(info.seed)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_splits(dataset: WaymoVectorDataset, val_fraction: float, seed: int) -> tuple[Iterable, Iterable | None]:
    if val_fraction <= 0.0 or len(dataset) < 2:
        return dataset, None
    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_val = min(n_val, len(dataset) - 1)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds


def slice_time_window(
    batch: Dict[str, torch.Tensor],
    time_window: int,
    *,
    random_start: bool = False,
) -> Dict[str, torch.Tensor]:
    if time_window <= 0:
        return batch
    total_steps = int(batch["lights"].shape[1])
    if total_steps <= time_window:
        return _slice_time_window(batch, time_window)
    if random_start:
        start = int(torch.randint(0, total_steps - int(time_window) + 1, (1,)).item())
    else:
        start = 0
    end = start + int(time_window)

    out = dict(batch)
    k = batch["agent_mask"].shape[-1]
    if batch["agents"].shape[1] == k:
        out["agents"] = batch["agents"][:, :, start:end]
    else:
        out["agents"] = batch["agents"][:, start:end]
    out["lights"] = batch["lights"][:, start:end]
    out["light_mask"] = batch["light_mask"][:, start:end]
    return out


def build_model(args: argparse.Namespace, n_agents: int, n_lights: int) -> VectorTokenizer:
    encoder_kwargs = dict(
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        n_latents=args.n_latents,
        d_bottleneck=args.d_bottleneck,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        scale_pos_embeds=args.scale_pos_embeds,
        bottleneck_output=args.bottleneck_output,
    )
    if args.encoder_variant == "repeat_map":
        encoder = VectorBlockCausalEncoder(**encoder_kwargs)
    elif args.encoder_variant == "static_map_query":
        encoder = VectorStaticMapQueryEncoder(
            **encoder_kwargs,
            map_depth=args.map_depth,
            map_cross_every=args.map_cross_every,
            map_query_tokens=args.map_query_tokens,
        )
    else:
        raise ValueError(f"Unknown encoder_variant: {args.encoder_variant}")
    decoder = VectorBlockCausalTokenizerDecoder(
        d_bottleneck=args.d_bottleneck,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.decoder_depth,
        n_latents=args.n_latents,
        n_agents=n_agents,
        n_lights=n_lights,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        scale_pos_embeds=args.scale_pos_embeds,
        use_agent_tokens=args.decoder_use_agent_tokens,
        agent_token_mode=args.decoder_agent_token_mode,
        attend_map=args.decoder_attend_map,
        map_cross_every=args.decoder_map_cross_every,
        map_query_tokens=args.decoder_map_query_tokens,
        predict_agent_xy_gmm=args.agent_xy_loss == "gmm",
    )
    return VectorTokenizer(encoder=encoder, decoder=decoder)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    model = model.module if hasattr(model, "module") else model
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def compute_loss(out, batch: Dict[str, torch.Tensor], args: argparse.Namespace):
    return vector_tokenizer_reconstruction_loss(
        out.decoder,
        agents=batch["agents"],
        agent_mask=batch["agent_mask"],
        lights=batch["lights"],
        light_mask=batch["light_mask"],
        agent_xy_weight=args.agent_xy_weight,
        agent_vel_weight=args.agent_vel_weight,
        agent_yaw_weight=args.agent_yaw_weight,
        agent_valid_weight=args.agent_valid_weight,
        light_state_weight=args.light_state_weight,
        light_valid_weight=args.light_valid_weight,
        agent_delta_xy_weight=args.agent_delta_xy_weight,
        agent_fde_xy_weight=args.agent_fde_xy_weight,
        agent_kinematic_xy_weight=args.agent_kinematic_xy_weight,
        agent_speed_yaw_kinematic_weight=args.agent_speed_yaw_kinematic_weight,
        kinematic_dt=args.kinematic_dt,
        focus_agent_weight=args.focus_agent_weight,
        agent_xy_loss=args.agent_xy_loss,
        agent_xy_parameterization=args.agent_xy_parameterization,
    )


def metric_values(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().float().item()) for k, v in metrics.items()}


def format_metrics(metrics: Dict[str, float]) -> str:
    names = [
        "loss_total",
        "loss_agent_xy",
        "loss_agent_vel",
        "loss_agent_yaw",
        "loss_agent_valid",
        "loss_agent_delta_xy",
        "loss_agent_fde_xy",
        "loss_agent_kinematic_xy",
        "loss_agent_speed_yaw_kinematic",
        "loss_light_state",
        "loss_light_valid",
        "agent_xy_mae_m",
        "agent_delta_xy_mae_m",
        "agent_fde_mae_m",
        "agent_kinematic_xy_mae_m",
        "agent_speed_yaw_kinematic_mae_m",
        "focus_agent_xy_mae_m",
        "focus_agent_fde_m",
        "agent_speed_mae_mps",
        "agent_vxvy_mae_mps",
        "agent_yaw_mae_deg",
        "agent_valid_acc",
        "light_state_acc",
        "light_valid_acc",
    ]
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in names if name in metrics)


def ordered_metric_names(metrics: Dict[str, torch.Tensor] | Dict[str, float]) -> list[str]:
    preferred = [
        "loss_agent_valid",
        "loss_agent_vel",
        "loss_agent_xy",
        "loss_agent_yaw",
        "loss_agent_delta_xy",
        "loss_agent_fde_xy",
        "loss_agent_kinematic_xy",
        "loss_agent_speed_yaw_kinematic",
        "loss_light_state",
        "loss_light_valid",
        "loss_total",
        "agent_xy_mae_m",
        "agent_delta_xy_mae_m",
        "agent_fde_mae_m",
        "agent_kinematic_xy_mae_m",
        "agent_speed_yaw_kinematic_mae_m",
        "focus_agent_xy_mae_m",
        "focus_agent_fde_m",
        "agent_speed_mae_mps",
        "agent_vxvy_mae_mps",
        "agent_yaw_mae_deg",
        "agent_valid_acc",
        "light_state_acc",
        "light_valid_acc",
    ]
    return [name for name in preferred if name in metrics] + sorted(set(metrics) - set(preferred))


def average_metrics(metrics: Dict[str, torch.Tensor], device: torch.device, ddp: bool) -> Dict[str, float]:
    names = ordered_metric_names(metrics)
    values = torch.tensor([float(metrics[name].detach().float().item()) for name in names], device=device)
    if ddp:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {name: float(value.item()) for name, value in zip(names, values)}


@torch.no_grad()
def evaluate(
    model: VectorTokenizer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    ddp: bool = False,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = slice_time_window(move_batch(batch, device), args.time_window, random_start=args.eval_random_time_window_start)
        out = model(
            agents=batch["agents"],
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_mask=batch["map_mask"],
            lights=batch["lights"],
            light_mask=batch["light_mask"],
        )
        _, metrics = compute_loss(out, batch, args)
        values = metric_values(metrics)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    names = ordered_metric_names(totals)
    packed = torch.tensor([float(count)] + [totals.get(name, 0.0) for name in names], device=device, dtype=torch.float64)
    if ddp:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_count = max(1.0, float(packed[0].item()))
    if was_training:
        model.train()
    return {name: float(packed[i + 1].item() / total_count) for i, name in enumerate(names)}


def save_ckpt(
    path: Path,
    *,
    model: VectorTokenizer,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "step": int(step),
            "epoch": int(epoch),
        },
        tmp,
    )
    tmp.replace(path)


def load_ckpt(path: Path, model: VectorTokenizer, opt: torch.optim.Optimizer, scaler: GradScaler) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_agent_xy_loss = getattr(ckpt_args, "agent_xy_loss", "smooth_l1")
    elif isinstance(ckpt_args, dict):
        ckpt_agent_xy_loss = ckpt_args.get("agent_xy_loss", "smooth_l1")
    else:
        ckpt_agent_xy_loss = "smooth_l1"
    model_to_load = unwrap_model(model)
    model_changed = False
    try:
        model_to_load.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError:
        incompatible = model_to_load.load_state_dict(ckpt["model"], strict=False)
        allowed_gmm_keys = {
            "decoder.agent_xy_gmm_head.weight",
            "decoder.agent_xy_gmm_head.bias",
        }
        if (
            not set(incompatible.missing_keys).issubset(allowed_gmm_keys)
            or not set(incompatible.unexpected_keys).issubset(allowed_gmm_keys)
            or ckpt_agent_xy_loss == "gmm"
        ):
            raise
        model_changed = True
        print(f"Loaded {path} without optimizer state because optional xy GMM head differs.")
    if not model_changed:
        opt.load_state_dict(ckpt["opt"])
    if not model_changed and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def train(args: argparse.Namespace) -> None:
    ddp, rank, world_size, local_rank = init_distributed()
    if ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed + rank)
    use_amp = (device.type == "cuda") and (not args.no_amp)

    dataset = WaymoVectorDataset(args.data_dir)
    sample = dataset[0]
    n_agents = int(sample["agent_mask"].shape[-1])
    n_lights = int(sample["lights"].shape[1])
    if args.val_data_dir is not None:
        train_ds = dataset
        val_ds = WaymoVectorDataset(args.val_data_dir)
    else:
        train_ds, val_ds = make_splits(dataset, args.val_fraction, args.seed)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
        collate_fn=_collate,
    )
    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=worker_init_fn,
            collate_fn=_collate,
        )

    model = build_model(args, n_agents=n_agents, n_lights=n_lights).to(device)
    if args.compile:
        model = torch.compile(model)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device="cuda", enabled=use_amp)

    step = 0
    start_epoch = 0
    if args.resume:
        step, start_epoch = load_ckpt(Path(args.resume), model, opt, scaler)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("wandb is not installed; run without --wandb or install wandb.") from exc
        if is_rank0():
            wandb_run = wandb
            wandb_run.init(project=args.wandb_project, name=args.wandb_run_name, entity=args.wandb_entity, config=vars(args))

    if is_rank0():
        param_count = sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad)
        print(
            f"device={device} amp={use_amp} ddp={ddp} world_size={world_size} "
            f"train={len(train_ds)} val={0 if val_ds is None else len(val_ds)}"
        )
        print(
            f"n_agents={n_agents} n_lights={n_lights} n_latents={args.n_latents} "
            f"d_bottleneck={args.d_bottleneck} decoder_agent_token_mode={args.decoder_agent_token_mode} "
            f"decoder_attend_map={args.decoder_attend_map} decoder_map_query_tokens={args.decoder_map_query_tokens}"
        )
        print(
            f"encoder_variant={args.encoder_variant} depth={args.depth} decoder_depth={args.decoder_depth} "
            f"map_depth={args.map_depth} map_cross_every={args.map_cross_every} map_query_tokens={args.map_query_tokens}"
        )
        print(f"learnable parameters: {param_count:,}")

    t0 = time.time()
    latest = Path(args.ckpt_dir) / "latest.pt"
    stop = False
    last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch + 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = slice_time_window(move_batch(batch, device), args.time_window, random_start=args.random_time_window_start)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                out = model(
                    agents=batch["agents"],
                    agent_mask=batch["agent_mask"],
                    map_polylines=batch["map_polylines"],
                    map_mask=batch["map_mask"],
                    lights=batch["lights"],
                    light_mask=batch["light_mask"],
                )
                loss, metrics = compute_loss(out, batch, args)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            step += 1

            should_log = step % args.log_every == 0 or step == 1
            if ddp and (should_log or args.wandb):
                values = average_metrics(metrics, device, ddp=True)
            else:
                values = metric_values(metrics)
            if is_rank0() and should_log:
                elapsed = max(1e-6, time.time() - t0)
                print(f"step={step} epoch={epoch + 1} {format_metrics(values)} steps_per_sec={step / elapsed:.3f}")
            if is_rank0() and wandb_run is not None:
                wandb_run.log({f"train/{k}": v for k, v in values.items()}, step=step)

            if val_loader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                val_metrics = evaluate(model, val_loader, device, args, ddp=ddp)
                if is_rank0():
                    print(f"eval step={step} {format_metrics(val_metrics)}")
                if is_rank0() and wandb_run is not None:
                    wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)

            if is_rank0() and args.save_every > 0 and step % args.save_every == 0:
                save_ckpt(Path(args.ckpt_dir) / f"step_{step:08d}.pt", model=model, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch)
                save_ckpt(latest, model=model, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch)

            if args.max_steps > 0 and step >= args.max_steps:
                stop = True
                break

        if is_rank0():
            save_ckpt(latest, model=model, opt=opt, scaler=scaler, args=args, step=step, epoch=epoch + 1)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args, ddp=ddp)
            if is_rank0():
                print(f"epoch={epoch + 1} val {format_metrics(val_metrics)}")
            if is_rank0() and wandb_run is not None:
                wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
        if stop:
            break

    if is_rank0():
        save_ckpt(Path(args.ckpt_dir) / f"final_step_{step:08d}.pt", model=model, opt=opt, scaler=scaler, args=args, step=step, epoch=last_epoch)
    if is_rank0() and wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(ddp, device)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Waymo vector tokenizer.")
    p.add_argument("--data_dir", type=str, nargs="+", default=["/p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset"])
    p.add_argument("--val_data_dir", type=str, nargs="+", default=None)
    p.add_argument("--ckpt_dir", type=str, default="/p/yufeng/tri30/dreamer4/waymo/checkpoints/vector_tokenizer")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--time_window", type=int, default=32)
    p.add_argument("--random_time_window_start", action="store_true", help="Sample a random contiguous time window during training.")
    p.add_argument("--eval_random_time_window_start", action="store_true", help="Also sample random windows during validation/eval logging.")
    p.add_argument("--val_fraction", type=float, default=0.1)

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--decoder_depth", type=int, default=3)
    p.add_argument("--n_latents", type=int, default=8)
    p.add_argument("--d_bottleneck", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)
    p.add_argument("--scale_pos_embeds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--encoder_variant", choices=["repeat_map", "static_map_query"], default="repeat_map")
    p.add_argument("--map_depth", type=int, default=2)
    p.add_argument("--map_cross_every", type=int, default=1)
    p.add_argument("--map_query_tokens", choices=["latent", "agent", "latent_agent", "all"], default="latent_agent")
    p.add_argument("--bottleneck_output", choices=["tanh", "layernorm"], default="tanh")
    p.add_argument("--decoder_use_agent_tokens", action="store_true")
    p.add_argument("--decoder_agent_token_mode", choices=["none", "all", "focus"], default="none")
    p.add_argument("--decoder_attend_map", action="store_true")
    p.add_argument("--decoder_map_cross_every", type=int, default=1)
    p.add_argument(
        "--decoder_map_query_tokens",
        choices=["latent", "agent", "light", "latent_agent", "agent_light", "all"],
        default="all",
    )

    p.add_argument("--agent_xy_weight", type=float, default=1.0)
    p.add_argument("--agent_xy_loss", choices=["smooth_l1", "gmm"], default="smooth_l1")
    p.add_argument("--agent_xy_parameterization", choices=["absolute", "delta"], default="absolute")
    p.add_argument("--agent_vel_weight", type=float, default=0.5)
    p.add_argument("--agent_yaw_weight", type=float, default=0.5)
    p.add_argument("--agent_valid_weight", type=float, default=0.2)
    p.add_argument("--light_state_weight", type=float, default=0.5)
    p.add_argument("--light_valid_weight", type=float, default=0.1)
    p.add_argument("--agent_delta_xy_weight", type=float, default=0.0)
    p.add_argument("--agent_fde_xy_weight", type=float, default=0.0)
    p.add_argument("--agent_kinematic_xy_weight", type=float, default=0.0)
    p.add_argument("--agent_speed_yaw_kinematic_weight", type=float, default=0.0)
    p.add_argument("--kinematic_dt", type=float, default=0.1)
    p.add_argument("--focus_agent_weight", type=float, default=1.0)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")

    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=500)

    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="waymo-vector-tokenizer")
    p.add_argument("--wandb_run_name", type=str, default="baseline")
    p.add_argument("--wandb_entity", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.decoder_use_agent_tokens and args.decoder_agent_token_mode == "none":
        args.decoder_agent_token_mode = "all"
    if args.decoder_agent_token_mode != "none":
        args.decoder_use_agent_tokens = True
    train(args)


if __name__ == "__main__":
    main()
