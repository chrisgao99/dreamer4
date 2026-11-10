#!/usr/bin/env python3
"""Experiment 1 training (plan §8.2): frozen stage-1 85k WM (EMA weights, eval mode), train only g.

Loss: the stage-1 flow loss (``flow_matching_loss``; random window start, bf16,
flow_time_max 0.9, Huber beta 1, action clip 5) with teacher-forced log z at the
anchor. Only samples with >= 1 labelled pair are used (others give g no gradient).

Validation every ``--eval_every`` steps on a fixed set (fixed anchors, noise and
flow times): per-agent flow loss with true z vs unknown z, for all agents, the
agents in active pairs (yielders / priority cars), split by whether the second
car's logged entry is inside the 4 s plan window; plus bias statistics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from waymo.pair_order_z import compat  # noqa: E402
from waymo.pair_order_z import z_bias as zb  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoint", default=str(compat.STAGE1_CKPT))
    p.add_argument("--labels_dir", default=str(compat.DATA / "pair_order_z_labels_v1"))
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=60)
    p.add_argument("--num_workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_minutes", type=float, default=0.0, help="Stop early (saving) after this wall time.")
    return p.parse_args()


def load_frozen_wm(path: str, device: torch.device):
    from waymo.training.world_model.train_waymo_direct_action_flow import build_normalizer, create_model
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    train_args = argparse.Namespace(**ckpt["args"])
    model = create_model(train_args).to(device)
    model.load_state_dict(ckpt["ema_model"], strict=True)
    model.eval()
    model.requires_grad_(False)
    normalizer = build_normalizer(ckpt["action_stats"], device)
    model.set_action_normalizer(normalizer)
    info = {"path": str(path), "step": int(ckpt.get("step", -1)), "best_val": ckpt.get("best_val")}
    del ckpt
    return model, normalizer, train_args, info


def make_dataset(split: str, labels: zb.PairLabels):
    from waymo.lane_connectivity.dataset import ConnectivityDataset
    root = compat.TRAIN_DIR if split == "train" else compat.VAL_DIR
    ds = zb.with_z_pairs(ConnectivityDataset, labels)(str(root), str(compat.MAP_CACHE), str(compat.LANE_TOPOLOGY))
    keep = [k for k, p in enumerate(ds.paths) if labels.has_pairs(Path(p).name)]
    return Subset(ds, keep), len(ds)


def lr_at(step: int, a) -> float:
    if step < a.warmup_steps:
        return a.lr * (step + 1) / a.warmup_steps
    progress = min(1.0, (step - a.warmup_steps) / max(1, a.steps - a.warmup_steps))
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def biased_scene(g, scene, batch, prepared):
    z, codes, priority, yielder = zb.training_z(batch["z_pairs"], prepared.anchors, scene.agent_mask)
    bias = g(z, scene.agent_pose, prepared.history[:, :, -1, 2], scene.agent_mask)
    return (scene if bias is None else replace(scene, chunk_bias=bias)), bias, z, codes, priority, yielder


def per_agent_flow_loss(model, scene, actions, valid, x0, flow_time):
    """flow_matching_loss's Huber term (generate-all), per agent: (sum over H,3; count of valid steps)."""
    lam = flow_time[:, None, None, None].to(actions.dtype)
    x_lambda = (1.0 - lam) * x0 + lam * actions
    mask = scene.agent_mask[:, :, None].expand_as(valid)
    pred = model.decode_velocity(x_lambda, flow_time, scene, mask)
    per = F.smooth_l1_loss(pred.float(), (actions - x0).float(), reduction="none", beta=1.0).sum(-1)
    per = per * valid.float()
    return per.sum(-1), valid.sum(-1).float()


@torch.no_grad()
def validate(model, g, normalizer, train_args, val_batches, device):
    from waymo.training.world_model.train_waymo_direct_action_flow import prepare_batch, scene_kwargs
    groups = ("all", "active", "yielder", "priority", "active_in_window", "active_beyond_window")
    acc = {k: {"z": 0.0, "u": 0.0, "n": 0.0} for k in groups}
    stats = zb.BiasStats()
    code_counts = np.zeros(9, dtype=np.int64)
    for k, batch in enumerate(val_batches):
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(1000 + k)
            prepared = prepare_batch(batch, normalizer, train_args, random_start=True)
            actions = prepared.normalized_actions.clamp(-5.0, 5.0)
            x0 = torch.randn_like(actions)
            flow_time = torch.rand((actions.shape[0],), device=device) * 0.9
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scene = model.encode_scene(**scene_kwargs(batch, prepared))
            zscene, bias, z, codes, priority, yielder = biased_scene(g, scene, batch, prepared)
            lz, cnt = per_agent_flow_loss(model, zscene, actions, prepared.targets.valid, x0, flow_time)
            lu, _ = per_agent_flow_loss(model, scene, actions, prepared.targets.valid, x0, flow_time)
        code_counts += np.bincount(codes.flatten().cpu().numpy(), minlength=9)[:9]
        if bias is not None:
            stats.update(bias, z)
        member = torch.zeros_like(scene.agent_mask, dtype=torch.bool)
        yield_m, prio_m, inwin_m = member.clone(), member.clone(), member.clone()
        active = priority >= 0
        if active.any():
            b_idx = torch.arange(priority.shape[0], device=device)[:, None].expand_as(priority)
            second_enter = torch.maximum(batch["z_pairs"][..., 2], batch["z_pairs"][..., 4])
            in_window = second_enter <= prepared.anchors[:, None] + int(train_args.horizon)
            yield_m[b_idx[active], yielder[active]] = True
            prio_m[b_idx[active], priority[active]] = True
            sel = active & in_window
            inwin_m[b_idx[sel], yielder[sel]] = True
            inwin_m[b_idx[sel], priority[sel]] = True
        active_m = yield_m | prio_m
        masks = {"all": scene.agent_mask.bool(), "active": active_m, "yielder": yield_m,
                 "priority": prio_m, "active_in_window": inwin_m,
                 "active_beyond_window": active_m & ~inwin_m}
        for name, m in masks.items():
            mf = m.float()
            acc[name]["z"] += float((lz * mf).sum())
            acc[name]["u"] += float((lu * mf).sum())
            acc[name]["n"] += float((cnt * mf).sum() * 3)
    out = {}
    for name, v in acc.items():
        n = max(1.0, v["n"])
        out[f"val_{name}_loss_z"] = v["z"] / n
        out[f"val_{name}_loss_unknown"] = v["u"] / n
        out[f"val_{name}_gain_pct"] = 100.0 * (v["u"] - v["z"]) / max(1e-12, v["u"])
        out[f"val_{name}_points"] = v["n"] / 3
    out["val_pair_codes"] = {zb.CODE_NAMES[c]: int(code_counts[c]) for c in range(1, 9)}
    out["val_bias"] = stats.summary()
    return out


def main() -> None:
    a = parse_args()
    compat.patch_map_cache_identity()
    from waymo.training.world_model.train_waymo_direct_action_flow import (
        flow_matching_loss, make_loader, move_batch, prepare_batch, scene_kwargs, seed_everything)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    seed_everything(a.seed)

    model, normalizer, train_args, wm_info = load_frozen_wm(a.checkpoint, device)
    labels_train = zb.PairLabels(str(Path(a.labels_dir) / "train.npz"))
    labels_val = zb.PairLabels(str(Path(a.labels_dir) / "val.npz"))
    train_ds, train_total = make_dataset("train", labels_train)
    val_ds, val_total = make_dataset("val", labels_val)
    train_loader = make_loader(train_ds, batch_size=a.batch_size, shuffle=True,
                               num_workers=a.num_workers, device=device)
    val_loader = make_loader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=2, device=device)
    val_batches = []
    for k, raw in enumerate(val_loader):
        if k >= a.val_batches:
            break
        val_batches.append(move_batch(raw, device))
    del val_loader

    g = zb.ZBias(n_heads=int(train_args.n_heads), hidden=a.hidden).to(device)
    opt = torch.optim.AdamW(g.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_params = sum(p.numel() for p in g.parameters())
    config = {**vars(a), "wm": wm_info, "g_params": n_params, "g_config": g.config,
              "train_samples_with_pairs": len(train_ds), "train_samples_total": train_total,
              "val_samples_with_pairs": len(val_ds), "val_samples_total": val_total,
              "val_samples_used": sum(int(b["agents"].shape[0]) for b in val_batches)}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    print("config " + json.dumps(config, default=str), flush=True)
    metrics_path = out / "metrics.jsonl"

    def log(record):
        with metrics_path.open("a") as h:
            h.write(json.dumps(record, default=str) + "\n")

    def save(name, step, val):
        tmp = out / f"{name}.tmp"
        torch.save({"state_dict": g.state_dict(), "config": g.config, "step": step, "val": val,
                    "train_config": config}, tmp)
        tmp.replace(out / name)

    g.eval()  # no dropout in g; WM stays in eval mode throughout
    val = validate(model, g, normalizer, train_args, val_batches, device)
    print("validation step=0 " + json.dumps(val), flush=True)
    log({"step": 0, **val})
    best = val["val_all_loss_z"]
    save("zbias_best.pt", 0, val)

    step, epoch = 0, 0
    started = time.monotonic()
    window = {"loss": 0.0, "n": 0, "skipped": 0, "t": time.monotonic()}
    stop = False
    while not stop:
        epoch += 1
        for raw in train_loader:
            batch = move_batch(raw, device)
            with torch.no_grad():
                prepared = prepare_batch(batch, normalizer, train_args, random_start=True)
            actions = prepared.normalized_actions.clamp(-5.0, 5.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    scene = model.encode_scene(**scene_kwargs(batch, prepared))
                zscene, bias, *_ = biased_scene(g, scene, batch, prepared)
                if bias is None:
                    window["skipped"] += 1
                    continue
                # Memory-efficient SDPA backward fails here ("LSE is not correctly aligned
                # (strideH)") once gradients flow through float-masked attention; use math SDPA.
                with sdpa_kernel([SDPBackend.MATH]):
                    loss, _ = flow_matching_loss(model, zscene, actions, prepared.targets.valid,
                                                 flow_time_max=0.9, loss_type="huber", huber_beta=1.0,
                                                 condition_focus_actions=False)
            for group in opt.param_groups:
                group["lr"] = lr_at(step, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(g.parameters(), a.grad_clip)
            opt.step()
            step += 1
            window["loss"] += float(loss.detach())
            window["n"] += 1
            if step % a.log_every == 0:
                dt = time.monotonic() - window["t"]
                rec = {"step": step, "epoch": epoch, "lr": opt.param_groups[0]["lr"],
                       "train_flow_loss": window["loss"] / max(1, window["n"]),
                       "grad_norm": float(grad_norm), "steps_per_s": window["n"] / max(1e-6, dt),
                       "skipped_batches": window["skipped"], "elapsed_min": (time.monotonic() - started) / 60}
                print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
                      flush=True)
                log(rec)
                window = {"loss": 0.0, "n": 0, "skipped": 0, "t": time.monotonic()}
            timed_out = a.max_minutes > 0 and (time.monotonic() - started) / 60 > a.max_minutes
            if step % a.eval_every == 0 or step >= a.steps or timed_out:
                val = validate(model, g, normalizer, train_args, val_batches, device)
                print(f"validation step={step} " + json.dumps(val), flush=True)
                log({"step": step, **val})
                save("zbias_last.pt", step, val)
                if val["val_all_loss_z"] < best:
                    best = val["val_all_loss_z"]
                    save("zbias_best.pt", step, val)
                    print(f"new best val_all_loss_z={best:.6f} step={step}", flush=True)
            if step >= a.steps or timed_out:
                stop = True
                break
    print(f"training complete step={step} best_val_all_loss_z={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
