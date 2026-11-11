#!/usr/bin/env python3
"""How big is g's bias, how much of it depends on z's direction, and how much does it move attention?

On val samples with labelled pairs (random anchors, fixed seeds, as in train_zbias_frozen.validate):
  b(z)       g output with the oracle z at the anchor
  b(flip z)  same state, order reversed (1<->2), i.e. what reversed mode feeds
  direction part  b(z) - b(flip z);  symmetric part (b(z) + b(flip z)) / 2
  base       the WM's own relative_bias on the same edges
  attention  per agent_attn layer, softmax weight of the pair edge with no z / oracle z / reversed z,
             plus the spread of the WM's own logits over keys for the same queries.
Edges: "yield" = query is the log yielder, key the log priority car (z == 2); "priority" = the reverse.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from waymo.pair_order_z import compat  # noqa: E402
from waymo.pair_order_z import z_bias as zb  # noqa: E402
from waymo.pair_order_z.train_zbias_frozen import load_frozen_wm, make_dataset  # noqa: E402


def q(x, ps=(50, 90, 99)):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": round(float(x.mean()), 3), **{f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps},
            "max": round(float(x.max()), 3), "n": int(x.size)}


class AttnRecorder:
    """Forward hooks on every agent_attn: store softmax weights (B*C, H, N, N) and raw logits."""

    def __init__(self, model):
        self.store, self.handles = [], []
        for layer in model.action_layers:
            self.handles.append(layer.agent_attn.register_forward_hook(self.hook, with_kwargs=True))

    def hook(self, mod, args, kwargs, out):
        x = args[0]
        b, n, _ = x.shape
        qq, kk, _ = mod.qkv(x).chunk(3, dim=-1)
        qq = qq.view(b, n, mod.n_heads, mod.head_dim).transpose(1, 2).float()
        kk = kk.view(b, n, mod.n_heads, mod.head_dim).transpose(1, 2).float()
        logits = qq @ kk.transpose(-1, -2) / mod.head_dim ** 0.5
        if kwargs.get("bias") is not None:
            logits = logits + kwargs["bias"].float()
        km = kwargs["key_mask"].bool()
        logits = logits.masked_fill(~km[:, None, None, :], float("-inf"))
        self.store.append((torch.softmax(logits, -1), logits, kwargs["query_mask"].bool(), km))

    def pop(self):
        s, self.store = self.store, []
        return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zbias_ckpt", required=True)
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    compat.patch_map_cache_identity()
    from waymo.training.world_model.train_waymo_direct_action_flow import (
        make_loader, move_batch, prepare_batch, scene_kwargs)
    device = torch.device("cuda")
    model, normalizer, train_args, _ = load_frozen_wm(str(compat.STAGE1_CKPT), device)
    g, ginfo = zb.load_zbias(a.zbias_ckpt, device)
    labels = zb.PairLabels(str(compat.DATA / "pair_order_z_labels_v1" / "val.npz"))
    ds, _ = make_dataset("val", labels)
    loader = make_loader(ds, batch_size=a.batch_size, shuffle=False, num_workers=2, device=device)
    rec = AttnRecorder(model)

    E = {k: {f: [] for f in ("b", "bflip", "dir", "sym", "base", "maxhead_b", "maxhead_dir")}
         for k in ("yield", "priority")}
    A = {k: {f: [] for f in ("w_none", "w_oracle", "w_rev", "uniform", "logit_std", "logit_edge_minus_mean")}
         for k in ("yield", "priority")}
    for k, raw in enumerate(loader):
        if k >= a.batches:
            break
        batch = move_batch(raw, device)
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(1000 + k)
            prepared = prepare_batch(batch, normalizer, train_args, random_start=True)
            actions = prepared.normalized_actions.clamp(-5.0, 5.0)
            x0 = torch.randn_like(actions)
            flow_time = torch.rand((actions.shape[0],), device=device) * 0.9
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scene = model.encode_scene(**scene_kwargs(batch, prepared))
            z, codes, prio, yl = zb.training_z(batch["z_pairs"], prepared.anchors, scene.agent_mask)
            zflip = torch.where(z == 1, 2, torch.where(z == 2, 1, z))
            speed = prepared.history[:, :, -1, 2]
            b = g(z, scene.agent_pose, speed, scene.agent_mask)
            if b is None:
                continue
            bf = g(zflip, scene.agent_pose, speed, scene.agent_mask)
            lam = flow_time[:, None, None, None].to(actions.dtype)
            x_l = (1.0 - lam) * x0 + lam * actions
            mask = scene.agent_mask[:, :, None].expand_as(prepared.targets.valid)
            weights = {}
            for name, sc in (("none", scene), ("oracle", replace(scene, chunk_bias=b)),
                             ("rev", replace(scene, chunk_bias=bf))):
                model.decode_velocity(x_l, flow_time, sc, mask)
                weights[name] = rec.pop()
        B, N = z.shape[0], z.shape[1]
        b0, bf0 = b[:, 0].float(), bf[:, 0].float()                   # (B,H,N,N)
        base = scene.relative_bias.float()                            # (B,H,N,N)
        for kind, zval in (("yield", 2), ("priority", 1)):
            bi, qi, ki = torch.nonzero(z == zval, as_tuple=True)
            if len(bi) == 0:
                continue
            eb, ef = b0[bi, :, qi, ki], bf0[bi, :, qi, ki]            # (E,H)
            E[kind]["b"].append(eb.abs().flatten().cpu().numpy())
            E[kind]["bflip"].append(ef.abs().flatten().cpu().numpy())
            E[kind]["dir"].append((eb - ef).abs().flatten().cpu().numpy())
            E[kind]["sym"].append(((eb + ef) / 2).abs().flatten().cpu().numpy())
            E[kind]["base"].append(base[bi, :, qi, ki].abs().flatten().cpu().numpy())
            E[kind]["maxhead_b"].append(eb.abs().amax(1).cpu().numpy())
            E[kind]["maxhead_dir"].append((eb - ef).abs().amax(1).cpu().numpy())
            for layer in range(len(weights["none"])):
                wn, logit, qm, km = weights["none"][layer]
                C = wn.shape[0] // B
                sel = lambda t: t.view(B, C, *t.shape[1:])[bi, :, :, qi, ki]          # (E,C,H)
                valid = (qm.view(B, C, N)[bi, :, qi] & km.view(B, C, N)[bi, :, ki])  # (E,C)
                vm = valid[..., None].expand(-1, -1, wn.shape[1])
                for tag in ("none", "oracle", "rev"):
                    A[kind][f"w_{tag if tag != 'rev' else 'rev'}"].append(sel(weights[tag][layer][0])[vm].cpu().numpy())
                nk = km.view(B, C, N)[bi].sum(-1).float()                            # (E,C)
                A[kind]["uniform"].append((1.0 / nk)[..., None].expand_as(vm)[vm].cpu().numpy())
                lg = logit.view(B, C, wn.shape[1], N, N)[bi, :, :, qi]                # (E,C,H,N)
                kmask = km.view(B, C, N)[bi][:, :, None, :].expand_as(lg)
                lgm = torch.where(kmask, lg, torch.nan)
                std = torch.from_numpy(np.nanstd(lgm.cpu().numpy(), -1)).to(lg.device)
                mean = torch.nanmean(lgm, -1)
                A[kind]["logit_std"].append(std[vm].cpu().numpy())
                A[kind]["logit_edge_minus_mean"].append((sel(logit) - mean)[vm].cpu().numpy())
        print(f"batch {k + 1}/{a.batches}", flush=True)

    out = {"g": ginfo["path"], "g_step": ginfo["step"], "edges": {}, "attention": {}}
    for kind in E:
        cat = {f: np.concatenate(v) for f, v in E[kind].items() if v}
        out["edges"][kind] = {f: q(v) for f, v in cat.items()}
        catA = {f: np.concatenate(v) for f, v in A[kind].items() if v}
        out["attention"][kind] = {f: q(v) for f, v in catA.items()}
        wn, wo, wr = catA["w_none"], catA["w_oracle"], catA["w_rev"]
        out["attention"][kind]["ratio_oracle_over_none"] = q(wo / np.maximum(wn, 1e-9), (10, 50, 90))
        out["attention"][kind]["ratio_rev_over_none"] = q(wr / np.maximum(wn, 1e-9), (10, 50, 90))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
