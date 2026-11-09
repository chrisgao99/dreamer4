#!/usr/bin/env python3
"""Multi-GPU BC pretraining of the direct-action-flow model on the whole-scene cache.

Launch with torchrun (see ``launch_train_tmux.sh``). Model, loss, optimizer and
schedule are those of ``train_waymo_direct_action_flow.py``; hyperparameters
default to the args stored in ``--base_config`` (a checkpoint of the original
laneconn run), so only data, agent selection, attention masks and DDP differ:

* data: ``SceneCacheDataset`` on the full WOMD training split; center = SDC
  with probability ``p_sdc`` else a random tracks_to_predict agent; current-
  valid agents within ``train_agent_radius_m`` (nearest ``max_agents``); map
  cropped to ``map_crop_m`` around them.
* loss: on ``loss_agents`` agents per scene (center, SDC, tracks_to_predict and
  objects of interest always, rest random). All agents are still denoised
  jointly as at inference; non-loss agents only lack supervision.
* model: kNN agent attention (``agent_knn`` within ``agent_radius_m``) and SDC
  factorization in the decoder (``sparse_agent_attention.py``).
* DDP: gradients are all-reduced explicitly because the training code calls
  ``encode_scene``/``decode_velocity`` rather than ``forward``.

Validation (all ranks, sharded): fixed random official-validation scenes,
closed-loop H80/B5 rollouts (``eval_scene_ade.evaluate_batch``); ``best.pt``
tracks the WOSAC-style ``minade_eval_agent``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import autocast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waymo.scene_cache.dataset import SceneCacheDataset, collate_scenes, read_manifest  # noqa: E402
from waymo.scene_cache.eval_scene_ade import evaluate_batch, make_batches, summarize  # noqa: E402
from waymo.scene_cache.sparse_agent_attention import patch_from_args, set_sdc_mask  # noqa: E402
from waymo.training.world_model import train_waymo_direct_action_flow as base  # noqa: E402
from waymo.training.world_model.direct_action_flow import flow_matching_loss  # noqa: E402

SCENE_CACHE = "/p/yufeng/waymo_scene_prep/data"


class ShardedEpochSampler(torch.utils.data.Sampler):
    """Per-epoch permutation, sharded by rank, resumable at a batch offset."""

    def __init__(self, size, batch_size, rank, world, seed):
        self.size, self.batch_size, self.rank, self.world, self.seed = size, batch_size, rank, world, seed
        self.batches_per_epoch = size // (batch_size * world)
        self.epoch, self.start_batch = 0, 0

    def set_epoch(self, epoch, start_batch=0):
        self.epoch, self.start_batch = int(epoch), int(start_batch)

    def __iter__(self):
        order = np.random.default_rng([self.seed, self.epoch]).permutation(self.size)
        order = order[: self.batches_per_epoch * self.batch_size * self.world]
        mine = order[self.rank::self.world].reshape(self.batches_per_epoch, self.batch_size)
        for batch in mine[self.start_batch:]:
            yield batch.tolist()

    def __len__(self):
        return self.batches_per_epoch - self.start_batch


def build_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_config", default="/p/yufeng/tri30/dreamer4/waymo/checkpoints/"
                   "waymo_daf_laneconn90k_mon8_gt_unroll80_b5_detach0_10k_20261009/best_ade.pt")
    p.add_argument("--train_manifest", default=f"{SCENE_CACHE}/manifest_training.jsonl")
    p.add_argument("--val_manifest", default=f"{SCENE_CACHE}/manifest_validation.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume", default="")
    p.add_argument("--epochs", type=float, default=30)
    p.add_argument("--max_steps", type=int, default=0, help="Override epochs (for smoke tests)")
    p.add_argument("--batch_size", type=int, default=8, help="Per GPU")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=8, help="Per GPU")
    p.add_argument("--seed", type=int, default=0)
    # Data / agents.
    p.add_argument("--p_sdc", type=float, default=0.5)
    p.add_argument("--train_agent_radius_m", type=float, default=150.0)
    p.add_argument("--max_agents", type=int, default=128)
    p.add_argument("--map_crop_m", type=float, default=120.0)
    p.add_argument("--loss_agents", type=int, default=48)
    # Model patches.
    p.add_argument("--agent_knn", type=int, default=48)
    p.add_argument("--agent_radius_m", type=float, default=80.0)
    p.add_argument("--no_sdc_factorized", action="store_true")
    # Logging / eval / checkpoints.
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=20000)
    p.add_argument("--val_scenes", type=int, default=256)
    p.add_argument("--val_rollouts", type=int, default=4)
    p.add_argument("--save_every", type=int, default=5000, help="latest.pt")
    p.add_argument("--keep_every", type=int, default=25000, help="numbered checkpoints")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="waymo-world-model")
    return p.parse_args()


def make_train_args(cli, world):
    base_ckpt = torch.load(cli.base_config, map_location="cpu", weights_only=False, mmap=True)
    ta = argparse.Namespace(**base_ckpt["args"])
    stats = base_ckpt["action_stats"]
    del base_ckpt
    ta.data_dir, ta.val_data_dir = cli.train_manifest, cli.val_manifest
    ta.ckpt_dir = cli.output_dir
    ta.batch_size, ta.grad_accum_steps, ta.world_size = cli.batch_size, cli.grad_accum_steps, world
    ta.lr, ta.warmup_steps, ta.min_lr_ratio, ta.seed = cli.lr, cli.warmup_steps, cli.min_lr_ratio, cli.seed
    ta.condition_focus_actions = False
    for key in ("p_sdc", "train_agent_radius_m", "max_agents", "map_crop_m", "loss_agents",
                "agent_knn", "agent_radius_m", "epochs"):
        setattr(ta, key, getattr(cli, key))
    ta.sdc_factorized = not cli.no_sdc_factorized
    ta.scene_cache = True
    ta.base_config = cli.base_config
    return ta, stats


def all_reduce_gradients(model, world):
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for p, g in zip(model.parameters(), grads):
        n = g.numel()
        chunk = flat[offset:offset + n].view_as(g).to(p.dtype)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


@torch.no_grad()
def validate(model, normalizer, ta, val_rows, rank, world, device, cli):
    model.eval()
    shard = val_rows[rank::world]
    batches = make_batches(shard, agent_budget=40000, max_batch=8)
    dataset = SceneCacheDataset(shard)
    index = {r["file"]: i for i, r in enumerate(shard)}
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=[[index[r["file"]] for r in b] for b in batches],
                                         collate_fn=collate_scenes, num_workers=4)
    rows, flow_losses = [], []
    cuda_state, cpu_state = torch.cuda.get_rng_state(device), torch.get_rng_state()
    for bi, raw in enumerate(loader):
        batch = base.move_batch(raw, device)
        with autocast("cuda", dtype=torch.bfloat16):
            rows += evaluate_batch(model, normalizer, ta, batch, cli.val_rollouts, 8, 12346 + 1000 * rank + bi * 10)
            set_sdc_mask(model, batch["agent_is_sdc"])
            prepared = base.prepare_batch(batch, normalizer, ta, random_start=False)
            scene = model.encode_scene(**base.scene_kwargs(batch, prepared))
            torch.manual_seed(1000 + 1000 * rank + bi)
            loss, _ = flow_matching_loss(model, scene, prepared.normalized_actions, prepared.targets.valid,
                                         loss_type=ta.flow_loss_type, huber_beta=ta.flow_huber_beta,
                                         condition_focus_actions=False)
        flow_losses.append(float(loss))
    torch.cuda.set_rng_state(cuda_state, device)
    torch.set_rng_state(cpu_state)
    gathered = [None] * world
    dist.all_gather_object(gathered, (rows, flow_losses))
    all_rows = [r for g in gathered for r in g[0]]
    summary = summarize(all_rows)
    record = {f"val_{k}": v for k, v in summary["all"].items() if k != "scenes"}
    record["val_flow_loss"] = float(np.mean([x for g in gathered for x in g[1]]))
    record["val_scenes"] = len(all_rows)
    for bucket, values in summary.items():
        if bucket != "all" and values.get("scenes"):
            record[f"val_{bucket}_minade_eval_agent"] = values["minade_eval_agent"]
    return record


def main():
    cli = build_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60), device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_float32_matmul_precision("high")

    ta, stats = make_train_args(cli, world)
    base.resolve_kinematics_args(ta)
    out = Path(cli.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if not cli.resume and (out / "latest.pt").exists():
            raise FileExistsError(f"{out} has a run; pass --resume {out/'latest.pt'}")
    dist.barrier()

    train_rows = read_manifest(cli.train_manifest)
    val_rows = random.Random(0).sample(read_manifest(cli.val_manifest), cli.val_scenes)
    global_batch = cli.batch_size * cli.grad_accum_steps * world
    sampler = ShardedEpochSampler(len(train_rows), cli.batch_size, rank, world, cli.seed)
    steps_per_epoch = sampler.batches_per_epoch // cli.grad_accum_steps
    ta.max_steps = cli.max_steps or int(math.ceil(cli.epochs * steps_per_epoch))
    ta.lr_decay_steps = ta.max_steps

    base.seed_everything(cli.seed)  # identical init on every rank
    model = base.create_model(ta).to(device)
    ema = base.ModelEMA(model, ta.ema_decay)  # deepcopy before patching
    patch_from_args(model, ta)
    patch_from_args(ema.model, ta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=ta.lr, betas=(0.9, 0.95), weight_decay=ta.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16
    normalizer = base.build_normalizer(stats, device)

    step, epoch, best = 0, 0, math.inf
    if cli.resume:
        step, epoch, best = base.load_checkpoint(Path(cli.resume), model=model, ema=ema,
                                                 optimizer=optimizer, scaler=scaler)
    for p in model.parameters():
        dist.broadcast(p.data, 0)
    base.seed_everything(cli.seed + 7919 * rank + step)

    dataset = SceneCacheDataset(train_rows, center="mixed", p_sdc=cli.p_sdc,
                                agent_radius_m=cli.train_agent_radius_m, max_agents=cli.max_agents,
                                map_crop_m=cli.map_crop_m, loss_agents=cli.loss_agents, seed=cli.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=cli.num_workers,
                                         collate_fn=collate_scenes, pin_memory=True, prefetch_factor=4,
                                         persistent_workers=False)
    metrics_path = out / "metrics.jsonl"
    wandb_run = None
    if rank == 0:
        (out / "config.json").write_text(json.dumps({"cli": vars(cli), "train_args": vars(ta)}, indent=1, default=str))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"world={world} global_batch={global_batch} steps/epoch={steps_per_epoch} max_steps={ta.max_steps} "
              f"train={len(train_rows)} val={len(val_rows)} params={n_params:,} knn={ta.agent_knn} "
              f"r={ta.agent_radius_m} sdc_factorized={ta.sdc_factorized} loss_agents={ta.loss_agents}", flush=True)
        if cli.wandb:
            try:
                import wandb
                wandb_run = wandb.init(project=cli.wandb_project, name=out.name, config=vars(ta),
                                       resume="allow", id=out.name)
            except Exception as exc:
                print(f"wandb disabled: {exc}", flush=True)

    def save(name):
        if rank == 0:
            base.save_checkpoint(out / name, model=model, ema=ema, optimizer=optimizer, scaler=scaler,
                                 args=ta, stats=stats, step=step, epoch=epoch, best_val=best)

    model.train()
    for group in optimizer.param_groups:
        group["lr"] = ta.lr * base.lr_multiplier(step, ta)
    rolling, count, tic, data_time = {}, 0, time.monotonic(), 0.0
    while step < ta.max_steps:
        epoch = step // steps_per_epoch
        sampler.set_epoch(epoch, (step % steps_per_epoch) * cli.grad_accum_steps)
        wait = time.monotonic()
        for micro, raw in enumerate(loader):
            data_time += time.monotonic() - wait
            batch = base.move_batch(raw, device)
            set_sdc_mask(model, batch["agent_is_sdc"])
            prepared = base.prepare_batch(batch, normalizer, ta, random_start=True)
            actions = prepared.normalized_actions
            if ta.train_normalized_action_clip > 0:
                actions = actions.clamp(-ta.train_normalized_action_clip, ta.train_normalized_action_clip)
            loss_valid = prepared.targets.valid & batch["agent_loss_mask"][:, :, None]
            with autocast("cuda", dtype=torch.bfloat16):
                scene = model.encode_scene(**base.scene_kwargs(batch, prepared))
                loss, metrics = flow_matching_loss(model, scene, actions, loss_valid,
                                                   flow_time_max=ta.train_flow_time_max,
                                                   loss_type=ta.flow_loss_type, huber_beta=ta.flow_huber_beta,
                                                   condition_focus_actions=False)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step} rank {rank}")
            (loss / cli.grad_accum_steps).backward()
            for k, v in metrics.items():
                rolling[k] = rolling.get(k, 0.0) + float(v)
            rolling["agents_per_scene"] = rolling.get("agents_per_scene", 0.0) + float(batch["agent_mask"].sum(1).float().mean())
            rolling["map_segments"] = rolling.get("map_segments", 0.0) + float(batch["map_ids"].shape[1])
            count += 1
            wait = time.monotonic()
            if (micro + 1) % cli.grad_accum_steps:
                continue

            all_reduce_gradients(model, world)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ta.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = ta.lr * base.lr_multiplier(step, ta)
            ema.update(model)

            if step % cli.log_every == 0 and rank == 0:
                elapsed = time.monotonic() - tic
                record = dict(step=step, epoch=round(step / steps_per_epoch, 4), lr=optimizer.param_groups[0]["lr"],
                              grad_norm=float(grad_norm), steps_per_second=cli.log_every / elapsed,
                              scenes_per_second=cli.log_every * global_batch / elapsed,
                              data_wait_fraction=data_time / elapsed,
                              cuda_peak_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                              **{f"train_{k}": v / max(1, count) for k, v in rolling.items()})
                eta_h = (ta.max_steps - step) / record["steps_per_second"] / 3600
                print(" ".join(f"{k}={v:.5g}" for k, v in record.items()) + f" eta_h={eta_h:.1f}", flush=True)
                base.append_metrics(metrics_path, record)
                if wandb_run:
                    wandb_run.log(record, step=step)
            if step % cli.log_every == 0:
                rolling, count, tic, data_time = {}, 0, time.monotonic(), 0.0

            if step % cli.eval_every == 0 or step == ta.max_steps:
                record = dict(step=step, epoch=round(step / steps_per_epoch, 4),
                              **validate(ema.model, normalizer, ta, val_rows, rank, world, device, cli))
                if rank == 0:
                    print("validation " + " ".join(f"{k}={v:.5g}" for k, v in record.items()), flush=True)
                    base.append_metrics(metrics_path, record)
                    if wandb_run:
                        wandb_run.log(record, step=step)
                if record["val_minade_eval_agent"] < best:
                    best = record["val_minade_eval_agent"]
                    save("best.pt")
                tic, data_time = time.monotonic(), 0.0
            if step % cli.save_every == 0 or step == ta.max_steps:
                save("latest.pt")
            if step % cli.keep_every == 0:
                save(f"step_{step:09d}.pt")
            if step >= ta.max_steps or step % steps_per_epoch == 0:
                break
            wait = time.monotonic()  # exclude validation/checkpoint time from data wait
    save(f"final_step_{step:09d}.pt")
    if rank == 0:
        print(f"training complete at step {step}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
