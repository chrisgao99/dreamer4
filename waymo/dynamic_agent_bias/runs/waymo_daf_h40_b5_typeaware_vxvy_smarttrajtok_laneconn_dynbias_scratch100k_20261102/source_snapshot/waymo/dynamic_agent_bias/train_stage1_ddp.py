#!/usr/bin/env python3
"""Stage one (flow matching) of the laneconn baseline with the dynamic agent bias, on several GPUs.

Same data, arguments, schedule, evaluation and checkpoints as
waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k_20261007 (base.train), except
``dynamic_agent_bias=True``. The global batch of 16 per optimizer step is split across ranks
(6/5/5 on 3 GPUs) instead of 4x4 gradient accumulation; each rank's mean loss is weighted by its
share and gradients are summed, so one update equals the single-GPU update. Rank 0 evaluates and
saves. Launch with torchrun.
"""
import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Sampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from waymo.dynamic_agent_bias.common import (GLOBAL_BATCH, broadcast_parameters, max_parameter_divergence,  # noqa: E402
                                             rank_batch_sizes, stage1_args, sum_gradients)
from waymo.lane_connectivity.dataset import ConnectivityDataset  # noqa: E402
from waymo.training.world_model import train_waymo_direct_action_flow as base  # noqa: E402


class GlobalStepBatches(Sampler):
    """Per-epoch seeded permutation cut into global batches; this rank's slice of each.

    Indexed by global step, so a resume continues the exact sequence.
    """

    def __init__(self, size, sizes, rank, seed, start_step, stop_step):
        self.size, self.sizes, self.rank, self.seed = size, sizes, rank, seed
        self.start, self.stop = start_step, stop_step
        self.total = sum(sizes)
        self.steps_per_epoch = size // self.total
        self.offset = sum(sizes[:rank])

    def epoch_of(self, step):
        return step // self.steps_per_epoch

    def __iter__(self):
        cached_epoch, perm = None, None
        for step in range(self.start, self.stop):
            epoch = self.epoch_of(step)
            if epoch != cached_epoch:
                perm = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
                cached_epoch = epoch
            start = (step % self.steps_per_epoch) * self.total + self.offset
            yield perm[start:start + self.sizes[self.rank]]

    def __len__(self):
        return self.stop - self.start


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_name', required=True)
    p.add_argument('--num_workers', type=int, default=6, help='data workers per rank')
    p.add_argument('--resume', action='store_true', help='continue from <ckpt_dir>/latest.pt')
    p.add_argument('--ckpt_dir', default=None, help='override (smoke tests)')
    p.add_argument('--override', nargs='*', default=[], help='key=json overrides (smoke tests only)')
    return p.parse_args()


def main():
    options = parse()
    dist.init_process_group('nccl', timeout=timedelta(hours=3))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    args = stage1_args(options.run_name)
    args.num_workers = options.num_workers
    if options.ckpt_dir:
        args.ckpt_dir = options.ckpt_dir
    for item in options.override:
        key, value = item.split('=', 1)
        if not hasattr(args, key):
            raise KeyError(key)
        setattr(args, key, json.loads(value))
    sizes = rank_batch_sizes(world)
    args.ddp_world_size, args.ddp_rank_batch_sizes = world, sizes
    if args.horizon % args.commitment or args.amp_dtype != 'bf16' or args.condition_focus_actions:
        raise ValueError('Expected the bf16 generate-all baseline settings')
    if getattr(args, 'balance_dynamic_loss', False) or getattr(args, 'map_aux_losses', False):
        raise ValueError('Baseline uses neither dynamic balance nor map losses')

    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    base.seed_everything(args.seed)
    base.resolve_kinematics_args(args)
    stats = base.load_or_compute_action_statistics(args)
    normalizer = base.build_normalizer(stats, device)
    train_dataset = ConnectivityDataset(args.data_dir, args.map_cache_dir, args.lane_topology_dir)
    val_dataset = ConnectivityDataset(args.val_data_dir, args.map_cache_dir, args.lane_topology_dir)

    model = base.create_model(args).to(device)
    model.set_action_normalizer(normalizer)
    broadcast_parameters(model)
    ema = base.ModelEMA(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = GradScaler('cuda', enabled=False)  # bf16: kept only for the checkpoint format
    ckpt_dir = Path(args.ckpt_dir)
    step, epoch, best_val = 0, 0, math.inf
    if options.resume:
        step, epoch, best_val = base.load_checkpoint(ckpt_dir / 'latest.pt', model=model, ema=ema,
                                                     optimizer=optimizer, scaler=scaler)
        log(f'resumed step={step} best_val={best_val:.6f}')
    elif rank == 0 and ckpt_dir.exists() and any(ckpt_dir.glob('*.pt')):
        raise FileExistsError(f'{ckpt_dir} has checkpoints; use --resume')
    for group in optimizer.param_groups:
        group['lr'] = args.lr * base.lr_multiplier(step, args)
    # Model initialization is shared; training randomness (windows, noise, flow time) is per rank.
    base.seed_everything(args.seed + 7919 * rank + 104729 * step)

    sampler = GlobalStepBatches(len(train_dataset), sizes, rank, args.seed, step, args.max_steps)
    loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers else None,
                        worker_init_fn=base.worker_init_fn, collate_fn=base.collate_vector_batch)
    val_loader = base.make_loader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                  num_workers=4, device=device) if rank == 0 else None
    weight = sizes[rank] / float(GLOBAL_BATCH)

    metrics_path = ckpt_dir / 'metrics.jsonl'
    wandb_run = None
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / 'resolved_train_args.json').write_text(json.dumps(vars(args), indent=2) + '\n')
        log(f'world={world} rank_batches={sizes} train={len(train_dataset)} val={len(val_dataset)} '
            f'parameters={sum(p.numel() for p in model.parameters()):,} steps/epoch={sampler.steps_per_epoch} '
            f'dynamic_agent_bias={args.dynamic_agent_bias}')
        if args.wandb:
            try:
                import wandb
                wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                                       entity=args.wandb_entity, config=vars(args),
                                       resume='allow' if options.resume else None)
            except Exception as exc:
                log(f'wandb disabled after initialization error: {exc}')

    def save(path):
        base.save_checkpoint(path, model=model, ema=ema, optimizer=optimizer, scaler=scaler, args=args,
                             stats=stats, step=step, epoch=epoch, best_val=best_val)

    model.train()
    rolling, rolling_count = {}, 0
    last_log_time = time.monotonic()
    for raw_batch in loader:
        epoch = sampler.epoch_of(step) + 1
        batch = base.move_batch(raw_batch, device)
        prepared = base.prepare_batch(batch, normalizer, args, random_start=True)
        actions = prepared.normalized_actions
        if args.train_normalized_action_clip > 0.0:
            actions = actions.clamp(-args.train_normalized_action_clip, args.train_normalized_action_clip)
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            scene = model.encode_scene(**base.scene_kwargs(batch, prepared))
            loss, metrics = base.flow_matching_loss(
                model, scene, actions, prepared.targets.valid, flow_time_max=args.train_flow_time_max,
                loss_type=args.flow_loss_type, huber_beta=args.flow_huber_beta,
                condition_focus_actions=args.condition_focus_actions)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f'Non-finite training loss at step={step} rank={rank}: {float(loss)}')
        (loss * weight).backward()
        sum_gradients(model.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        for group in optimizer.param_groups:
            group['lr'] = args.lr * base.lr_multiplier(step, args)
        ema.update(model)
        rolling_count += 1
        for name, value in metrics.items():
            rolling[name] = rolling.get(name, 0.0) + float(value) * weight

        if step % args.log_every == 0:
            names = sorted(rolling)
            totals = torch.tensor([rolling[n] for n in names], device=device, dtype=torch.float64)
            dist.all_reduce(totals)
            elapsed = max(1e-6, time.monotonic() - last_log_time)
            record = {'step': step, 'epoch': epoch, 'lr': optimizer.param_groups[0]['lr'],
                      'grad_norm': float(grad_norm), 'steps_per_second': args.log_every / elapsed,
                      **{f'train_{n}': float(v) / rolling_count for n, v in zip(names, totals.tolist())}}
            if rank == 0:
                log(' '.join(f'{k}={v:.6g}' for k, v in record.items()))
                base.append_metrics(metrics_path, record)
                if wandb_run is not None:
                    wandb_run.log(record, step=step)
            rolling.clear()
            rolling_count = 0
            last_log_time = time.monotonic()

        if step % 1000 == 0:
            divergence = max_parameter_divergence(model)
            if divergence > 0:
                log(f'parameter divergence {divergence:.3g} at step {step}; resynchronizing from rank 0')
                broadcast_parameters(model)

        should_eval = step % args.eval_every == 0 or step == args.max_steps
        should_save = step % args.save_every == 0 or step == args.max_steps
        if (should_eval or should_save) and rank == 0:
            if should_eval:
                validation = base.evaluate_flow(ema.model, val_loader, normalizer, device, args)
                if step % args.sample_eval_every == 0 or step == args.max_steps:
                    validation.update(base.evaluate_samples(ema.model, val_loader, normalizer, device, args,
                                                            seed=args.eval_seed))
                if args.receding_eval_every > 0 and (step % args.receding_eval_every == 0 or step == args.max_steps):
                    validation.update(base.evaluate_receding_rollout(ema.model, val_loader, normalizer, device,
                                                                     args, seed=args.eval_seed + 1))
                record = {'step': step, 'epoch': epoch, **validation}
                log('validation ' + ' '.join(f'{k}={v:.6g}' for k, v in record.items()))
                base.append_metrics(metrics_path, record)
                if wandb_run is not None:
                    wandb_run.log(validation, step=step)
                model.train()
                if validation['val_flow_loss'] < best_val:
                    best_val = validation['val_flow_loss']
                    save(ckpt_dir / 'best.pt')
                    log(f'new best val_flow_loss={best_val:.6f} at step={step}')
            if should_save:
                numbered = ckpt_dir / f'step_{step:09d}.pt'
                save(numbered)
                shutil.copyfile(numbered, ckpt_dir / 'latest.pt')
                log(f'saved {numbered}')
        if should_eval or should_save:
            best = torch.tensor([best_val], device=device, dtype=torch.float64)
            dist.broadcast(best, 0)
            best_val = float(best)
            dist.barrier()
        if step >= args.max_steps:
            break

    if rank == 0:
        final = ckpt_dir / f'final_step_{step:09d}.pt'
        save(final)
        log(f'training complete: {final}')
        if wandb_run is not None:
            wandb_run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
