#!/usr/bin/env python3
"""Experiment 1b-v2: z v2 labels, denoiser unfrozen, loss focused on agents that have z.

Start: stage-1 85k ``best.pt`` (EMA weights). Trained: g v2 (zero-init, lr ``--lr_g``) and every
parameter outside ``scene_encoder`` (the denoiser: action layers, step layers, heads; lr ``--lr_wm``).
The scene encoder stays frozen (z only enters the denoiser's agent_attn). Dropout off (eval mode).

Per step (stage-1 flow loss otherwise: uniform flow time in [0, 0.9], Huber beta 1, action clip 5):
  * anchors: the stage-1 window rule (random valid start), restricted by rejection sampling to
    anchors where >= 1 pair has z on (codes 1/2/3/10 of z_bias_v2 at the logged state);
  * z-dropout: with prob ``--z_dropout`` a sample's z is set to unknown (bias 0);
  * loss weight per agent: 1 for agents in a pair with z on (whether or not z was dropped for the
    sample), ``--other_weight`` for the rest.
Validation (EMA weights, fixed anchors / noise / flow times): per-agent flow loss with true z vs
unknown z, for all agents and the z agents (crossing / same direction, yielders / priority cars);
``val_all_loss_unknown`` over steps shows drift of the unconditioned model.
Outputs: wm_{last,final}.pt (stage-1 checkpoint format, loadable by the eval scripts),
zbias_{last,final}.pt, metrics.jsonl, config.json.
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
from waymo.pair_order_z import z_bias_v2 as zb  # noqa: E402

_CTX = {"z_pairs": None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoint", default=str(compat.STAGE1_CKPT))
    p.add_argument("--labels_dir", default=str(compat.DATA / "pair_order_z_labels_v2"))
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr_g", type=float, default=1e-3)
    p.add_argument("--lr_wm", type=float, default=1e-5)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--z_dropout", type=float, default=0.3)
    p.add_argument("--other_weight", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--anchor_tries", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--val_batches", type=int, default=40)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_minutes", type=float, default=0.0)
    p.add_argument("--train_split", default="train", choices=("train", "val"), help="val only for smoke tests")
    return p.parse_args()


def install_anchor_rule(twm, tries: int):
    """Wrap select_window_anchors: among valid random starts prefer one where some pair has z on."""
    original = twm.select_window_anchors

    def select(agents_bntf, agent_mask, *, random_start, **kw):
        first = original(agents_bntf, agent_mask, random_start=random_start, **kw)
        zp = _CTX["z_pairs"]
        if not random_start or zp is None:
            return first
        chosen, done = first.clone(), torch.zeros_like(first, dtype=torch.bool)
        b = torch.arange(first.shape[0], device=first.device)
        for k in range(tries):
            cand = first if k == 0 else original(agents_bntf, agent_mask, random_start=True, **kw)
            frame = agents_bntf[b, :, cand]                                         # (B,N,F)
            alive = agent_mask.bool() & (frame[..., 5] > 0.5)
            pose = torch.stack((frame[..., 0], frame[..., 1], frame[..., 6]), -1)
            progress = cand[:, None].expand(-1, agent_mask.shape[1]).long()
            codes, _, _ = zb.pair_codes_v2(zp, progress, alive, pose, frame[..., 2])
            on = ((codes == 1) | (codes == 2) | (codes == 3) | (codes == 10)).any(1)
            take = on & ~done
            chosen = torch.where(take, cand, chosen)
            done |= on
            if bool(done.all()):
                break
        return chosen

    twm.select_window_anchors = select


def lr_scale(step: int, a) -> float:
    if step < a.warmup_steps:
        return (step + 1) / a.warmup_steps
    progress = min(1.0, (step - a.warmup_steps) / max(1, a.steps - a.warmup_steps))
    return a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def z_inputs(batch, prepared, scene):
    zp = batch["z_pairs"]
    speed = prepared.history[:, :, -1, 2]
    z, codes, priority, yielder = zb.training_z_v2(zp, prepared.anchors, scene.agent_mask, scene.agent_pose, speed)
    kind = zb.kind_matrix(zp, z.shape[1])
    return z, codes, priority, yielder, kind, speed


def members(priority, yielder, zp, n):
    """(B,N) masks: agents in an on pair; split by pair kind and by role."""
    bsz = priority.shape[0]
    out = {k: torch.zeros((bsz, n), dtype=torch.bool, device=priority.device)
           for k in ("z", "crossing", "same_dir", "yielder", "priority")}
    on = priority >= 0
    if on.any():
        b = torch.arange(bsz, device=priority.device)[:, None].expand_as(priority)
        same = zp[..., 7] == zb.KIND_SAME
        for role, idx in (("priority", priority), ("yielder", yielder)):
            out[role][b[on], idx[on]] = True
            out["z"][b[on], idx[on]] = True
            out["same_dir"][b[on & same], idx[on & same]] = True
            out["crossing"][b[on & ~same], idx[on & ~same]] = True
    return out


def per_agent_loss(model, scene, actions, valid, x0, flow_time):
    lam = flow_time[:, None, None, None].to(actions.dtype)
    x_l = (1.0 - lam) * x0 + lam * actions
    mask = scene.agent_mask[:, :, None].expand_as(valid)
    pred = model.decode_velocity(x_l, flow_time, scene, mask)
    per = F.smooth_l1_loss(pred.float(), (actions - x0).float(), reduction="none", beta=1.0).sum(-1)
    per = per * valid.float()
    return per.sum(-1), valid.sum(-1).float()              # (B,N) sum over H and 3 dims; valid steps


@torch.no_grad()
def validate(model, g, normalizer, train_args, val_batches, device):
    from waymo.training.world_model.train_waymo_direct_action_flow import prepare_batch, scene_kwargs
    groups = ("all", "z", "crossing", "same_dir", "yielder", "priority", "other")
    acc = {k: {"z": 0.0, "u": 0.0, "n": 0.0} for k in groups}
    stats = zb.BiasStats()
    code_counts = np.zeros(11, dtype=np.int64)
    for k, batch in enumerate(val_batches):
        _CTX["z_pairs"] = batch["z_pairs"]
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(1000 + k)
            prepared = prepare_batch(batch, normalizer, train_args, random_start=True)
            actions = prepared.normalized_actions.clamp(-5.0, 5.0)
            x0 = torch.randn_like(actions)
            flow_time = torch.rand((actions.shape[0],), device=device) * 0.9
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scene = model.encode_scene(**scene_kwargs(batch, prepared))
            z, codes, prio, yl, kind, speed = z_inputs(batch, prepared, scene)
            bias = g(z, scene.agent_pose, speed, scene.agent_mask, kind)
            zscene = scene if bias is None else replace(scene, chunk_bias=bias)
            lz, cnt = per_agent_loss(model, zscene, actions, prepared.targets.valid, x0, flow_time)
            lu, _ = per_agent_loss(model, scene, actions, prepared.targets.valid, x0, flow_time)
        code_counts += np.bincount(codes.flatten().cpu().numpy(), minlength=11)[:11]
        if bias is not None:
            stats.update(bias, z)
        m = members(prio, yl, batch["z_pairs"], z.shape[1])
        m["all"] = scene.agent_mask.bool()
        m["other"] = m["all"] & ~m["z"]
        for name in groups:
            mf = m[name].float()
            acc[name]["z"] += float((lz * mf).sum())
            acc[name]["u"] += float((lu * mf).sum())
            acc[name]["n"] += float((cnt * mf).sum() * 3)
    _CTX["z_pairs"] = None
    out = {}
    for name, v in acc.items():
        n = max(1.0, v["n"])
        out[f"val_{name}_loss_z"] = v["z"] / n
        out[f"val_{name}_loss_unknown"] = v["u"] / n
        out[f"val_{name}_gain_pct"] = 100.0 * (v["u"] - v["z"]) / max(1e-12, v["u"])
        out[f"val_{name}_points"] = v["n"] / 3
    out["val_pair_codes"] = {zb.CODE_NAMES[c]: int(code_counts[c]) for c in range(1, 11)}
    out["val_bias"] = stats.summary()
    return out


def main() -> None:
    a = parse_args()
    compat.patch_map_cache_identity()
    import waymo.training.world_model.train_waymo_direct_action_flow as twm
    from waymo.lane_connectivity.dataset import ConnectivityDataset
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    twm.seed_everything(a.seed)
    install_anchor_rule(twm, a.anchor_tries)

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    train_args = argparse.Namespace(**ckpt["args"])
    model = twm.create_model(train_args).to(device)
    model.load_state_dict(ckpt["ema_model"], strict=True)
    stats_payload = ckpt["action_stats"]
    wm_info = {"path": a.checkpoint, "step": int(ckpt.get("step", -1)), "best_val": ckpt.get("best_val")}
    del ckpt
    normalizer = twm.build_normalizer(stats_payload, device)
    model.set_action_normalizer(normalizer)
    model.eval()
    for name, prm in model.named_parameters():
        prm.requires_grad_(not name.startswith("scene_encoder."))
    wm_params = [prm for prm in model.parameters() if prm.requires_grad]
    ema = twm.ModelEMA(model, a.ema_decay)

    labels_train = zb.PairLabelsV2(str(Path(a.labels_dir) / f"{a.train_split}.npz"))
    labels_val = zb.PairLabelsV2(str(Path(a.labels_dir) / "val.npz"))

    def make(split, labels):
        root = compat.TRAIN_DIR if split == "train" else compat.VAL_DIR
        ds = zb.with_z_pairs_v2(ConnectivityDataset, labels)(str(root), str(compat.MAP_CACHE), str(compat.LANE_TOPOLOGY))
        keep = [k for k, pth in enumerate(ds.paths) if labels.has_pairs(Path(pth).name)]
        return Subset(ds, keep), len(ds)

    train_ds, train_total = make(a.train_split, labels_train)
    val_ds, val_total = make("val", labels_val)
    train_loader = twm.make_loader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, device=device)
    val_loader = twm.make_loader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=2, device=device)
    val_batches = []
    for k, raw in enumerate(val_loader):
        if k >= a.val_batches:
            break
        val_batches.append(twm.move_batch(raw, device))
    del val_loader

    g = zb.ZBiasV2(n_heads=int(train_args.n_heads), hidden=a.hidden).to(device)
    opt = torch.optim.AdamW([{"params": list(g.parameters()), "lr": a.lr_g, "base_lr": a.lr_g, "weight_decay": 0.0},
                             {"params": wm_params, "lr": a.lr_wm, "base_lr": a.lr_wm, "weight_decay": 0.0}])
    config = {**vars(a), "wm": wm_info, "g_params": sum(p.numel() for p in g.parameters()),
              "wm_trainable_params": sum(p.numel() for p in wm_params), "g_config": g.config,
              "train_samples_with_pairs": len(train_ds), "train_samples_total": train_total,
              "val_samples_with_pairs": len(val_ds), "val_samples_total": val_total,
              "val_samples_used": sum(int(b["agents"].shape[0]) for b in val_batches),
              "p_max": {"train": labels_train.p_max, "val": labels_val.p_max}}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    print("config " + json.dumps(config, default=str), flush=True)
    metrics_path = out / "metrics.jsonl"

    def log(record):
        with metrics_path.open("a") as h:
            h.write(json.dumps(record, default=str) + "\n")

    def save(tag, step, val):
        for name, payload in ((f"zbias_{tag}.pt", {"state_dict": g.state_dict(), "config": g.config, "step": step,
                                                   "val": val, "train_config": config}),
                              (f"wm_{tag}.pt", {"model": model.state_dict(), "ema_model": ema.model.state_dict(),
                                                "args": vars(train_args), "action_stats": stats_payload,
                                                "step": int(wm_info["step"]) + step, "epoch": 0,
                                                "best_val": float("nan"), "finetune_step": step,
                                                "finetune_config": config})):
            tmp = out / f"{name}.tmp"
            torch.save(payload, tmp)
            tmp.replace(out / name)

    val = validate(ema.model, g, normalizer, train_args, val_batches, device)
    print("validation step=0 " + json.dumps(val), flush=True)
    log({"step": 0, **val})

    step, epoch = 0, 0
    started = time.monotonic()
    window = {"loss": 0.0, "n": 0, "skipped": 0, "zfrac": 0.0, "t": time.monotonic()}
    stop = False
    while not stop:
        epoch += 1
        for raw in train_loader:
            batch = twm.move_batch(raw, device)
            _CTX["z_pairs"] = batch["z_pairs"]
            with torch.no_grad():
                prepared = twm.prepare_batch(batch, normalizer, train_args, random_start=True)
            _CTX["z_pairs"] = None
            actions = prepared.normalized_actions.clamp(-5.0, 5.0)
            valid = prepared.targets.valid
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    scene = model.encode_scene(**twm.scene_kwargs(batch, prepared))
                z, codes, prio, yl, kind, speed = z_inputs(batch, prepared, scene)
                mem = members(prio, yl, batch["z_pairs"], z.shape[1])["z"]
                keep = torch.rand((z.shape[0],), device=device) >= a.z_dropout
                bias = g(z * keep[:, None, None].long(), scene.agent_pose, speed, scene.agent_mask, kind)
                zscene = scene if bias is None else replace(scene, chunk_bias=bias)
                x0 = torch.randn_like(actions)
                flow_time = torch.rand((actions.shape[0],), device=device) * 0.9
                w = torch.where(mem, 1.0, float(a.other_weight)) * scene.agent_mask.float()       # (B,N)
                with sdpa_kernel([SDPBackend.MATH]):    # mem-efficient SDPA backward fails with float masks
                    lsum, cnt = per_agent_loss(model, zscene, actions, valid, x0, flow_time)
                loss = (lsum * w).sum() / ((cnt * w).sum() * 3).clamp_min(1.0)
            scale = lr_scale(step, a)
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn_g = torch.nn.utils.clip_grad_norm_(g.parameters(), a.grad_clip)
            gn_wm = torch.nn.utils.clip_grad_norm_(wm_params, a.grad_clip)
            opt.step()
            ema.update(model)
            step += 1
            window["loss"] += float(loss.detach())
            window["n"] += 1
            window["zfrac"] += float(mem.float().sum() / scene.agent_mask.float().sum().clamp_min(1))
            window["skipped"] += int(bias is None)
            if step % a.log_every == 0:
                dt = time.monotonic() - window["t"]
                rec = {"step": step, "epoch": epoch, "lr_g": opt.param_groups[0]["lr"], "lr_wm": opt.param_groups[1]["lr"],
                       "train_loss": window["loss"] / window["n"], "grad_norm_g": float(gn_g), "grad_norm_wm": float(gn_wm),
                       "z_agent_fraction": window["zfrac"] / window["n"], "batches_without_bias": window["skipped"],
                       "steps_per_s": window["n"] / max(1e-6, dt), "elapsed_min": (time.monotonic() - started) / 60}
                print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
                log(rec)
                window = {"loss": 0.0, "n": 0, "skipped": 0, "zfrac": 0.0, "t": time.monotonic()}
            timed_out = a.max_minutes > 0 and (time.monotonic() - started) / 60 > a.max_minutes
            if step % a.eval_every == 0 or step >= a.steps or timed_out:
                val = validate(ema.model, g, normalizer, train_args, val_batches, device)
                print(f"validation step={step} " + json.dumps(val), flush=True)
                log({"step": step, **val})
                save("last", step, val)
            if step >= a.steps or timed_out:
                stop = True
                break
    save("final", step, val)
    print(f"training complete step={step}", flush=True)


if __name__ == "__main__":
    main()
