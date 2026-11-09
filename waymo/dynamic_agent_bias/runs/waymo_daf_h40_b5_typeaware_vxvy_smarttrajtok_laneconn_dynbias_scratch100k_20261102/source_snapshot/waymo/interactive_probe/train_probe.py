"""Train and evaluate future-interaction relation probes."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

WAYMO_ROOT = Path(__file__).resolve().parents[1]
RELATION_ROOT = WAYMO_ROOT / "relation_probe"
if str(RELATION_ROOT) not in sys.path:
    sys.path.append(str(RELATION_ROOT))

from common import json_dump, seed_everything  # noqa: E402
from model import InteractiveProbe, masked_bce_with_logits, masked_cross_entropy, masked_smooth_l1  # noqa: E402


class InteractiveCache:
    def __init__(self, path: str):
        self.path = str(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.data = {key: data[key] for key in data.files}
        self.feature_names = [str(x) for x in self.data["feature_names"]]
        self.type_names = [str(x) for x in self.data["type_names"]]
        self.response_binary_names = [str(x) for x in self.data["response_binary_names"]]
        self.response_regression_names = [str(x) for x in self.data["response_regression_names"]]

    @property
    def num_pairs(self) -> int:
        return int(self.data["pair_raw"].shape[0])

    @property
    def z_dim(self) -> int:
        return int(self.data["z_current"].shape[-1])

    @property
    def n_latents(self) -> int:
        return int(self.data["z_current"].shape[1])


class PairDataset(Dataset):
    def __init__(self, cache: InteractiveCache, reg_mean: np.ndarray, reg_std: np.ndarray):
        self.cache = cache
        self.reg_mean = reg_mean.astype(np.float32)
        self.reg_std = reg_std.astype(np.float32)

    def __len__(self) -> int:
        return self.cache.num_pairs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self.cache.data
        response_reg = (d["response_reg_targets"][idx].astype(np.float32) - self.reg_mean) / self.reg_std
        return {
            "pair_raw": torch.from_numpy(d["pair_raw"][idx].astype(np.float32)),
            "scene_index": torch.as_tensor(int(d["pair_scene_index"][idx]), dtype=torch.long),
            "relevance_targets": torch.from_numpy(d["relevance_targets"][idx].astype(np.float32)),
            "relevance_masks": torch.from_numpy(d["relevance_masks"][idx].astype(np.float32)),
            "type_targets": torch.as_tensor(int(d["type_targets"][idx]), dtype=torch.long),
            "type_masks": torch.as_tensor(float(d["type_masks"][idx]), dtype=torch.float32),
            "response_bin_targets": torch.from_numpy(d["response_bin_targets"][idx].astype(np.float32)),
            "response_bin_masks": torch.from_numpy(d["response_bin_masks"][idx].astype(np.float32)),
            "response_reg_targets": torch.from_numpy(response_reg),
            "response_reg_targets_raw": torch.from_numpy(d["response_reg_targets"][idx].astype(np.float32)),
            "response_reg_masks": torch.from_numpy(d["response_reg_masks"][idx].astype(np.float32)),
        }


def collate(batch: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def compute_reg_stats(cache: InteractiveCache) -> tuple[np.ndarray, np.ndarray]:
    y = cache.data["response_reg_targets"].astype(np.float64)
    m = cache.data["response_reg_masks"].astype(np.float64)
    denom = np.maximum(m.sum(axis=0), 1.0)
    mean = (y * m).sum(axis=0) / denom
    var = (((y - mean[None, :]) ** 2) * m).sum(axis=0) / denom
    std = np.sqrt(np.maximum(var, 1e-4))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_pos_weight(y: np.ndarray, m: np.ndarray) -> np.ndarray:
    yy = y.astype(np.float64)
    mm = m.astype(np.float64)
    pos = (yy * mm).sum(axis=0)
    total = mm.sum(axis=0)
    neg = np.maximum(total - pos, 0.0)
    weight = neg / np.maximum(pos, 1.0)
    return np.clip(weight, 1.0, 50.0).astype(np.float32)


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


def _type_metrics(target: np.ndarray, logits: np.ndarray, mask: np.ndarray, type_names: list[str]) -> Dict[str, Any]:
    keep = mask > 0.5
    if not keep.any():
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "num_pairs": 0,
            "per_type": {name: {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")} for name in type_names},
        }
    pred = np.argmax(logits[keep], axis=1)
    tgt = target[keep]
    per_type = {}
    f1s = []
    for idx, name in enumerate(type_names):
        tp = int(((pred == idx) & (tgt == idx)).sum())
        fp = int(((pred == idx) & (tgt != idx)).sum())
        fn = int(((pred != idx) & (tgt == idx)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        per_type[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int((tgt == idx).sum())}
        if int((tgt == idx).sum()) > 0:
            f1s.append(f1)
    return {
        "accuracy": float((pred == tgt).mean()),
        "macro_f1": _nanmean(f1s),
        "num_pairs": int(keep.sum()),
        "per_type": per_type,
    }


def compute_losses(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    *,
    relevance_pos_weight: torch.Tensor,
    response_pos_weight: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    loss_relevance = masked_bce_with_logits(
        pred["relevance"],
        batch["relevance_targets"],
        batch["relevance_masks"],
        pos_weight=relevance_pos_weight,
    )
    loss_type = masked_cross_entropy(pred["type"], batch["type_targets"], batch["type_masks"])
    loss_response_bin = masked_bce_with_logits(
        pred["response_bin"],
        batch["response_bin_targets"],
        batch["response_bin_masks"],
        pos_weight=response_pos_weight,
    )
    loss_response_reg = masked_smooth_l1(
        pred["response_reg"],
        batch["response_reg_targets"],
        batch["response_reg_masks"],
    )
    loss = loss_relevance + loss_type + loss_response_bin + loss_response_reg
    return {
        "loss": loss,
        "loss_relevance": loss_relevance,
        "loss_type": loss_type,
        "loss_response_bin": loss_response_bin,
        "loss_response_reg": loss_response_reg,
    }


@torch.no_grad()
def evaluate(
    model: InteractiveProbe,
    cache: InteractiveCache,
    loader: DataLoader,
    *,
    z_tensor: torch.Tensor,
    device: torch.device,
    reg_mean: torch.Tensor,
    reg_std: torch.Tensor,
    relevance_pos_weight: torch.Tensor,
    response_pos_weight: torch.Tensor,
) -> Dict[str, Any]:
    model.eval()
    totals = {key: 0.0 for key in ("loss", "loss_relevance", "loss_type", "loss_response_bin", "loss_response_reg")}
    count = 0
    rel_scores = []
    rel_targets = []
    rel_masks = []
    type_logits = []
    type_targets = []
    type_masks = []
    resp_bin_scores = []
    resp_bin_targets = []
    resp_bin_masks = []
    resp_reg_preds = []
    resp_reg_targets = []
    resp_reg_masks = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        z = None
        if model.mode != "raw_only":
            z = z_tensor[batch["scene_index"]]
            if model.mode == "raw_shuffled_z":
                z = z[torch.randperm(z.shape[0], device=device)]
        pred = model(batch["pair_raw"], z)
        losses = compute_losses(
            pred,
            batch,
            relevance_pos_weight=relevance_pos_weight,
            response_pos_weight=response_pos_weight,
        )
        bsz = int(batch["pair_raw"].shape[0])
        for key in totals:
            totals[key] += float(losses[key].item()) * bsz
        count += bsz

        rel_scores.append(torch.sigmoid(pred["relevance"]).detach().cpu().numpy())
        rel_targets.append(batch["relevance_targets"].detach().cpu().numpy())
        rel_masks.append(batch["relevance_masks"].detach().cpu().numpy())
        type_logits.append(pred["type"].detach().cpu().numpy())
        type_targets.append(batch["type_targets"].detach().cpu().numpy())
        type_masks.append(batch["type_masks"].detach().cpu().numpy())
        resp_bin_scores.append(torch.sigmoid(pred["response_bin"]).detach().cpu().numpy())
        resp_bin_targets.append(batch["response_bin_targets"].detach().cpu().numpy())
        resp_bin_masks.append(batch["response_bin_masks"].detach().cpu().numpy())
        pred_reg_raw = pred["response_reg"] * reg_std[None, :] + reg_mean[None, :]
        resp_reg_preds.append(pred_reg_raw.detach().cpu().numpy())
        resp_reg_targets.append(batch["response_reg_targets_raw"].detach().cpu().numpy())
        resp_reg_masks.append(batch["response_reg_masks"].detach().cpu().numpy())

    rel_scores_np = np.concatenate(rel_scores, axis=0)
    rel_targets_np = np.concatenate(rel_targets, axis=0)
    rel_masks_np = np.concatenate(rel_masks, axis=0)
    type_logits_np = np.concatenate(type_logits, axis=0)
    type_targets_np = np.concatenate(type_targets, axis=0)
    type_masks_np = np.concatenate(type_masks, axis=0)
    resp_bin_scores_np = np.concatenate(resp_bin_scores, axis=0)
    resp_bin_targets_np = np.concatenate(resp_bin_targets, axis=0)
    resp_bin_masks_np = np.concatenate(resp_bin_masks, axis=0)
    resp_reg_preds_np = np.concatenate(resp_reg_preds, axis=0)
    resp_reg_targets_np = np.concatenate(resp_reg_targets, axis=0)
    resp_reg_masks_np = np.concatenate(resp_reg_masks, axis=0)

    relevance = {
        "ap": _binary_ap(rel_targets_np[:, 0], rel_scores_np[:, 0], rel_masks_np[:, 0]),
        "auroc": _binary_auroc(rel_targets_np[:, 0], rel_scores_np[:, 0], rel_masks_np[:, 0]),
        "pos_rate": float((rel_targets_np * rel_masks_np).sum() / max(1.0, rel_masks_np.sum())),
    }
    type_metrics = _type_metrics(type_targets_np, type_logits_np, type_masks_np, cache.type_names)

    response_binary = {}
    response_aps = []
    for idx, name in enumerate(cache.response_binary_names):
        ap = _binary_ap(resp_bin_targets_np[:, idx], resp_bin_scores_np[:, idx], resp_bin_masks_np[:, idx])
        auroc = _binary_auroc(resp_bin_targets_np[:, idx], resp_bin_scores_np[:, idx], resp_bin_masks_np[:, idx])
        pos_rate = float((resp_bin_targets_np[:, idx] * resp_bin_masks_np[:, idx]).sum() / max(1.0, resp_bin_masks_np[:, idx].sum()))
        response_binary[name] = {"ap": ap, "auroc": auroc, "pos_rate": pos_rate, "num_pairs": int((resp_bin_masks_np[:, idx] > 0.5).sum())}
        response_aps.append(ap)

    response_regression = {}
    reg_maes = []
    abs_err = np.abs(resp_reg_preds_np - resp_reg_targets_np)
    for idx, name in enumerate(cache.response_regression_names):
        keep = resp_reg_masks_np[:, idx] > 0.5
        mae = float(abs_err[keep, idx].mean()) if keep.any() else float("nan")
        response_regression[name] = {"mae": mae, "num_pairs": int(keep.sum())}
        reg_maes.append(mae)

    metrics: Dict[str, Any] = {key: value / max(1, count) for key, value in totals.items()}
    metrics.update(
        {
            "num_pairs": int(cache.num_pairs),
            "relevance": relevance,
            "type": type_metrics,
            "response_binary": response_binary,
            "response_regression": response_regression,
            "summary": {
                "relevance_ap": relevance["ap"],
                "relevance_auroc": relevance["auroc"],
                "type_macro_f1": type_metrics["macro_f1"],
                "type_accuracy": type_metrics["accuracy"],
                "mean_response_bin_ap": _nanmean(response_aps),
                "mean_response_reg_mae": _nanmean(reg_maes),
            },
        }
    )
    return metrics


def save_checkpoint(path: Path, model: InteractiveProbe, opt: torch.optim.Optimizer, step: int, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "metrics": metrics}, path)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cache = InteractiveCache(args.train_cache)
    val_cache = InteractiveCache(args.val_cache)
    reg_mean_np, reg_std_np = compute_reg_stats(train_cache)
    relevance_pos_weight_np = compute_pos_weight(train_cache.data["relevance_targets"], train_cache.data["relevance_masks"])
    response_pos_weight_np = compute_pos_weight(train_cache.data["response_bin_targets"], train_cache.data["response_bin_masks"])

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
    relevance_pos_weight = torch.from_numpy(relevance_pos_weight_np).to(device)
    response_pos_weight = torch.from_numpy(response_pos_weight_np).to(device)

    model = InteractiveProbe(
        mode=args.mode,
        pair_dim=int(train_cache.data["pair_raw"].shape[1]),
        z_dim=train_cache.z_dim,
        n_types=len(train_cache.type_names),
        n_response_bin=len(train_cache.response_binary_names),
        n_response_reg=len(train_cache.response_regression_names),
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    stats = {
        "reg_mean": reg_mean_np.tolist(),
        "reg_std": reg_std_np.tolist(),
        "relevance_pos_weight": relevance_pos_weight_np.tolist(),
        "response_pos_weight": response_pos_weight_np.tolist(),
        "feature_names": train_cache.feature_names,
        "type_names": train_cache.type_names,
        "response_binary_names": train_cache.response_binary_names,
        "response_regression_names": train_cache.response_regression_names,
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
    }
    json_dump(run_dir / "stats.json", stats)
    json_dump(run_dir / "args.json", vars(args))

    best_loss = math.inf
    step = 0
    start_time = time.time()
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

            pred = model(batch["pair_raw"], z)
            losses = compute_losses(
                pred,
                batch,
                relevance_pos_weight=relevance_pos_weight,
                response_pos_weight=response_pos_weight,
            )

            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if step % args.log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"step={step} loss={losses['loss'].item():.4f} "
                    f"rel={losses['loss_relevance'].item():.4f} type={losses['loss_type'].item():.4f} "
                    f"resp_bin={losses['loss_response_bin'].item():.4f} "
                    f"resp_reg={losses['loss_response_reg'].item():.4f} elapsed={elapsed:.1f}s",
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
                    relevance_pos_weight=relevance_pos_weight,
                    response_pos_weight=response_pos_weight,
                )
                metrics["step"] = step
                summary = metrics["summary"]
                print(
                    f"eval step={step} loss={metrics['loss']:.4f} "
                    f"rel_ap={summary['relevance_ap']:.4f} type_f1={summary['type_macro_f1']:.4f} "
                    f"resp_ap={summary['mean_response_bin_ap']:.4f} "
                    f"delta_mae={summary['mean_response_reg_mae']:.4f}",
                    flush=True,
                )
                json_dump(run_dir / "latest_metrics.json", metrics)
                save_checkpoint(run_dir / "latest.pt", model, opt, step, metrics)
                if metrics["loss"] < best_loss:
                    best_loss = float(metrics["loss"])
                    json_dump(run_dir / "best_metrics.json", metrics)
                    save_checkpoint(run_dir / "best.pt", model, opt, step, metrics)

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
        relevance_pos_weight=relevance_pos_weight,
        response_pos_weight=response_pos_weight,
    )
    final_metrics["step"] = step
    json_dump(run_dir / "final_metrics.json", final_metrics)
    save_checkpoint(run_dir / "final.pt", model, opt, step, final_metrics)
    print(f"saved final metrics: {run_dir / 'final_metrics.json'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train future-interaction relation probe.")
    p.add_argument("--train_cache", type=str, required=True)
    p.add_argument("--val_cache", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["raw_only", "raw_z", "raw_shuffled_z"], required=True)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--eval_batch_size", type=int, default=8192)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())

