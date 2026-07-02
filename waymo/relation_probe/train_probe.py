"""Train and evaluate current-state z[31] relation probes."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import json_dump, seed_everything
from model import RelationProbe, masked_bce_with_logits, masked_smooth_l1


class RelationCache:
    def __init__(self, path: str):
        self.path = str(path)
        # .npz files are zip-backed. Keeping the NpzFile handle open and then
        # forking DataLoader workers can share a mutable zip reader across
        # processes, which may raise BadZipFile/CRC errors. Load the arrays into
        # ordinary memory once at startup instead.
        with np.load(self.path, allow_pickle=False) as data:
            self.data = {key: data[key] for key in data.files}
        self.feature_names = [str(x) for x in self.data["feature_names"]]
        self.regression_names = [str(x) for x in self.data["regression_names"]]
        self.binary_names = [str(x) for x in self.data["binary_names"]]

    @property
    def num_pairs(self) -> int:
        return int(self.data["pair_raw"].shape[0])

    @property
    def z_dim(self) -> int:
        return int(self.data["z_current"].shape[-1])

    @property
    def n_latents(self) -> int:
        return int(self.data["z_current"].shape[1])

    def close(self) -> None:
        return None


class PairDataset(Dataset):
    def __init__(self, cache: RelationCache, reg_mean: np.ndarray, reg_std: np.ndarray):
        self.cache = cache
        self.reg_mean = reg_mean.astype(np.float32)
        self.reg_std = reg_std.astype(np.float32)

    def __len__(self) -> int:
        return self.cache.num_pairs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self.cache.data
        reg = (d["reg_targets"][idx].astype(np.float32) - self.reg_mean) / self.reg_std
        return {
            "pair_raw": torch.from_numpy(d["pair_raw"][idx].astype(np.float32)),
            "scene_index": torch.as_tensor(int(d["pair_scene_index"][idx]), dtype=torch.long),
            "reg_targets": torch.from_numpy(reg),
            "reg_targets_raw": torch.from_numpy(d["reg_targets"][idx].astype(np.float32)),
            "reg_masks": torch.from_numpy(d["reg_masks"][idx].astype(np.float32)),
            "bin_targets": torch.from_numpy(d["bin_targets"][idx].astype(np.float32)),
            "bin_masks": torch.from_numpy(d["bin_masks"][idx].astype(np.float32)),
        }


def compute_reg_stats(cache: RelationCache) -> tuple[np.ndarray, np.ndarray]:
    y = cache.data["reg_targets"].astype(np.float64)
    m = cache.data["reg_masks"].astype(np.float64)
    denom = np.maximum(m.sum(axis=0), 1.0)
    mean = (y * m).sum(axis=0) / denom
    var = (((y - mean[None, :]) ** 2) * m).sum(axis=0) / denom
    std = np.sqrt(np.maximum(var, 1e-4))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_pos_weight(cache: RelationCache) -> np.ndarray:
    y = cache.data["bin_targets"].astype(np.float64)
    m = cache.data["bin_masks"].astype(np.float64)
    pos = (y * m).sum(axis=0)
    total = m.sum(axis=0)
    neg = np.maximum(total - pos, 0.0)
    weight = neg / np.maximum(pos, 1.0)
    return np.clip(weight, 1.0, 50.0).astype(np.float32)


def collate(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def _binary_ap(y_true: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float:
    keep = mask > 0.5
    y = y_true[keep] > 0.5
    s = score[keep]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(len(y_sorted)) + 1.0)
    return float((precision * y_sorted).sum() / n_pos)


def _binary_auroc(y_true: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float:
    keep = mask > 0.5
    y = y_true[keep] > 0.5
    s = score[keep]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    pos_rank_sum = ranks[y].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if np.isfinite(arr).any():
        return float(np.nanmean(arr))
    return float("nan")


@torch.no_grad()
def evaluate(
    model: RelationProbe,
    cache: RelationCache,
    loader: DataLoader,
    *,
    z_tensor: torch.Tensor,
    device: torch.device,
    reg_mean: torch.Tensor,
    reg_std: torch.Tensor,
    pos_weight: torch.Tensor,
) -> Dict[str, Any]:
    model.eval()
    loss_total = 0.0
    loss_reg_total = 0.0
    loss_bin_total = 0.0
    count = 0
    reg_preds = []
    reg_targets = []
    reg_masks = []
    bin_scores = []
    bin_targets = []
    bin_masks = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        z = None
        if model.mode != "raw_only":
            z = z_tensor[batch["scene_index"]]
            if model.mode == "raw_shuffled_z":
                z = z[torch.randperm(z.shape[0], device=device)]
        pred_reg, pred_bin, _ = model(batch["pair_raw"], z)
        loss_reg = masked_smooth_l1(pred_reg, batch["reg_targets"], batch["reg_masks"])
        loss_bin = masked_bce_with_logits(pred_bin, batch["bin_targets"], batch["bin_masks"], pos_weight=pos_weight)
        loss = loss_reg + loss_bin

        bsz = int(batch["pair_raw"].shape[0])
        loss_total += float(loss.item()) * bsz
        loss_reg_total += float(loss_reg.item()) * bsz
        loss_bin_total += float(loss_bin.item()) * bsz
        count += bsz

        pred_reg_raw = pred_reg * reg_std[None, :] + reg_mean[None, :]
        reg_preds.append(pred_reg_raw.detach().cpu().numpy())
        reg_targets.append(batch["reg_targets_raw"].detach().cpu().numpy())
        reg_masks.append(batch["reg_masks"].detach().cpu().numpy())
        bin_scores.append(torch.sigmoid(pred_bin).detach().cpu().numpy())
        bin_targets.append(batch["bin_targets"].detach().cpu().numpy())
        bin_masks.append(batch["bin_masks"].detach().cpu().numpy())

    reg_pred_np = np.concatenate(reg_preds, axis=0)
    reg_target_np = np.concatenate(reg_targets, axis=0)
    reg_mask_np = np.concatenate(reg_masks, axis=0)
    bin_score_np = np.concatenate(bin_scores, axis=0)
    bin_target_np = np.concatenate(bin_targets, axis=0)
    bin_mask_np = np.concatenate(bin_masks, axis=0)

    metrics: Dict[str, Any] = {
        "loss": loss_total / max(1, count),
        "loss_reg": loss_reg_total / max(1, count),
        "loss_bin": loss_bin_total / max(1, count),
        "num_pairs": int(count),
        "regression": {},
        "binary": {},
    }

    reg_maes = []
    for idx, name in enumerate(cache.regression_names):
        m = reg_mask_np[:, idx] > 0.5
        mae = float(np.abs(reg_pred_np[m, idx] - reg_target_np[m, idx]).mean()) if m.any() else float("nan")
        metrics["regression"][name] = {"mae": mae}
        reg_maes.append(mae)

    aps = []
    aucs = []
    for idx, name in enumerate(cache.binary_names):
        ap = _binary_ap(bin_target_np[:, idx], bin_score_np[:, idx], bin_mask_np[:, idx])
        auc = _binary_auroc(bin_target_np[:, idx], bin_score_np[:, idx], bin_mask_np[:, idx])
        pos_rate = float(((bin_target_np[:, idx] * bin_mask_np[:, idx]).sum()) / max(bin_mask_np[:, idx].sum(), 1.0))
        metrics["binary"][name] = {"ap": ap, "auroc": auc, "pos_rate": pos_rate}
        aps.append(ap)
        aucs.append(auc)

    metrics["summary"] = {
        "mean_reg_mae": _nanmean(reg_maes),
        "mean_bin_ap": _nanmean(aps),
        "mean_bin_auroc": _nanmean(aucs),
    }
    return metrics


def save_checkpoint(path: Path, model: RelationProbe, opt: torch.optim.Optimizer, args: argparse.Namespace, step: int, stats: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
            "step": int(step),
            "stats": stats,
        },
        tmp,
    )
    tmp.replace(path)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cache = RelationCache(args.train_cache)
    val_cache = RelationCache(args.val_cache)
    reg_mean_np, reg_std_np = compute_reg_stats(train_cache)
    pos_weight_np = compute_pos_weight(train_cache)

    train_ds = PairDataset(train_cache, reg_mean_np, reg_std_np)
    val_ds = PairDataset(val_cache, reg_mean_np, reg_std_np)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate,
        persistent_workers=(args.num_workers > 0),
    )

    z_train = torch.from_numpy(train_cache.data["z_current"].astype(np.float32)).to(device)
    z_val = torch.from_numpy(val_cache.data["z_current"].astype(np.float32)).to(device)
    reg_mean = torch.from_numpy(reg_mean_np).to(device)
    reg_std = torch.from_numpy(reg_std_np).to(device)
    pos_weight = torch.from_numpy(pos_weight_np).to(device)

    model = RelationProbe(
        mode=args.mode,
        pair_dim=int(train_cache.data["pair_raw"].shape[1]),
        z_dim=train_cache.z_dim,
        n_reg=len(train_cache.regression_names),
        n_bin=len(train_cache.binary_names),
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    stats = {
        "reg_mean": reg_mean_np.tolist(),
        "reg_std": reg_std_np.tolist(),
        "pos_weight": pos_weight_np.tolist(),
        "feature_names": train_cache.feature_names,
        "regression_names": train_cache.regression_names,
        "binary_names": train_cache.binary_names,
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
    }
    json_dump(run_dir / "stats.json", stats)
    json_dump(run_dir / "args.json", vars(args))

    best_score = -math.inf
    step = 0
    start = time.time()
    while step < args.max_steps:
        model.train()
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            z = None
            if args.mode != "raw_only":
                z = z_train[batch["scene_index"]]
                if args.mode == "raw_shuffled_z":
                    z = z[torch.randperm(z.shape[0], device=device)]

            pred_reg, pred_bin, _ = model(batch["pair_raw"], z)
            loss_reg = masked_smooth_l1(pred_reg, batch["reg_targets"], batch["reg_masks"])
            loss_bin = masked_bce_with_logits(pred_bin, batch["bin_targets"], batch["bin_masks"], pos_weight=pos_weight)
            loss = loss_reg + loss_bin

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if step % args.log_every == 0:
                elapsed = time.time() - start
                print(
                    f"step={step} loss={float(loss.item()):.4f} reg={float(loss_reg.item()):.4f} "
                    f"bin={float(loss_bin.item()):.4f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if step % args.eval_every == 0 or step == args.max_steps:
                metrics = evaluate(
                    model,
                    val_cache,
                    val_loader,
                    z_tensor=z_val,
                    device=device,
                    reg_mean=reg_mean,
                    reg_std=reg_std,
                    pos_weight=pos_weight,
                )
                metrics["step"] = int(step)
                json_dump(run_dir / "latest_metrics.json", metrics)
                score = float(metrics["summary"]["mean_bin_ap"])
                print(
                    f"eval step={step} loss={metrics['loss']:.4f} mean_reg_mae={metrics['summary']['mean_reg_mae']:.4f} "
                    f"mean_bin_ap={metrics['summary']['mean_bin_ap']:.4f} mean_bin_auroc={metrics['summary']['mean_bin_auroc']:.4f}",
                    flush=True,
                )
                save_checkpoint(run_dir / "latest.pt", model, opt, args, step, stats)
                if score > best_score:
                    best_score = score
                    save_checkpoint(run_dir / "best.pt", model, opt, args, step, stats)
                    json_dump(run_dir / "best_metrics.json", metrics)

            if step >= args.max_steps:
                break

    final_metrics = evaluate(
        model,
        val_cache,
        val_loader,
        z_tensor=z_val,
        device=device,
        reg_mean=reg_mean,
        reg_std=reg_std,
        pos_weight=pos_weight,
    )
    final_metrics["step"] = int(step)
    json_dump(run_dir / "final_metrics.json", final_metrics)
    save_checkpoint(run_dir / "final.pt", model, opt, args, step, stats)
    print(f"saved final metrics: {run_dir / 'final_metrics.json'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train z[31] current-state relation probe.")
    p.add_argument("--train_cache", type=str, required=True)
    p.add_argument("--val_cache", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["raw_only", "raw_z", "raw_shuffled_z"], required=True)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
