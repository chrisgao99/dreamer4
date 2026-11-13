#!/usr/bin/env python3
"""Train the per-directed-pair bias head on the search labels (plan BIAS_CONTROLLER_PLAN_2026_11_12 §2.4, §3).

Input: <labels_dir>/batches/*.npz from make_bias_labels.py (reads whatever batches exist).
Split by scenario (10% val). Features standardized with train statistics (stored in the model buffers).
3-class cross entropy (delta in {0, 2, 4}) with class weights ~ 1/sqrt(frequency).
Output: <out_dir>/bias_head.pt (best val loss), train_summary.{json,md}.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from waymo.pair_order_z import bias_head_lib as L  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load(labels_dir):
    X, Y, M, V = [], [], [], []
    files = sorted(glob.glob(str(Path(labels_dir) / "batches" / "*.npz")))
    rolls = dict(has_target=[], chosen=[], kind=[])
    for f in files:
        d = np.load(f)
        sc = {int(i): str(s) for i, s in zip(d["dataset_index"], d["scenario_ids"])}
        X.append(d["feats"].astype(np.float32)); Y.append(d["labels"].astype(np.int64)); M.append(d["meta"])
        V.append(np.array([sc[int(i)] for i in d["meta"][:, 0]]))
        for k in rolls:
            rolls[k].append(d[f"roll_{k}"])
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(M), np.concatenate(V),
            {k: np.concatenate(v) for k, v in rolls.items()}, len(files))


def is_val(scenarios, frac):
    return np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 1000 < frac * 1000 for s in scenarios])


@torch.no_grad()
def evaluate(head, X, Y, M, w, device):
    head.eval()
    logits = torch.cat([head(X[i:i + 65536].to(device)) for i in range(0, len(X), 65536)]).cpu()
    loss = float(torch.nn.functional.cross_entropy(logits, Y, weight=w))
    p = torch.softmax(logits, -1)
    pnz = 1 - p[:, 0]
    out = dict(loss=loss, n=int(len(Y)), n_nonzero=int((Y > 0).sum()))
    y_nz = (Y > 0).numpy()
    for tau in (0.3, 0.5, 0.7, 0.9):
        pred = (pnz > tau).numpy()
        tp = int((pred & y_nz).sum())
        out[f"tau{tau}"] = dict(pred_nonzero=int(pred.sum()), precision=tp / max(1, pred.sum()), recall=tp / max(1, y_nz.sum()))
    pred_cls = torch.where(pnz > 0.5, torch.where(p[:, 2] > p[:, 1], 2, 1), 0)
    cm = np.zeros((3, 3), int)
    np.add.at(cm, (Y.numpy(), pred_cls.numpy()), 1)
    out["confusion_tau0.5"] = cm.tolist()
    # ranking quality of P(nonzero)
    order = np.argsort(-pnz.numpy())
    ranks = np.empty(len(order)); ranks[order] = np.arange(len(order))
    npos, nneg = y_nz.sum(), (~y_nz).sum()
    out["auc_nonzero"] = float(1 - (ranks[y_nz].sum() - npos * (npos - 1) / 2) / max(1, npos * nneg)) if npos and nneg else None
    tgt = M[:, 5] == 1
    out["target_pair_instances"] = int(tgt.sum())
    if tgt.any():
        out["target_pair_acc_tau0.5"] = float((pred_cls[tgt] == Y[tgt]).float().mean())
    return out


def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    X, Y, M, V, rolls, nfiles = load(a.labels_dir)
    val = is_val(V, a.val_frac)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt, Yt, Xv, Yv = torch.from_numpy(X[~val]), torch.from_numpy(Y[~val]), torch.from_numpy(X[val]), torch.from_numpy(Y[val])
    counts = np.bincount(Y[~val], minlength=3)
    w = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32)
    w = w / w[0]
    print(f"{nfiles} batch files; instances train {len(Yt)} val {len(Yv)}; train class counts {counts.tolist()}; weights {w.tolist()}", flush=True)
    head = L.BiasHead(hidden=a.hidden).to(device)
    head.mean.copy_(Xt.mean(0).to(device)); head.std.copy_(Xt.std(0).clamp_min(1e-3).to(device))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(Yt) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, steps), pct_start=0.1)
    wd = w.to(device)
    best, hist = None, []
    t0 = time.time()
    for ep in range(a.epochs):
        head.train()
        perm = torch.randperm(len(Yt))
        tl = 0.0
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            loss = torch.nn.functional.cross_entropy(head(Xt[idx].to(device)), Yt[idx].to(device), weight=wd)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tl += float(loss) * len(idx)
        ev = evaluate(head, Xv, Yv, M[val], w, device)
        hist.append(dict(epoch=ep, train_loss=tl / len(perm), **{k: ev[k] for k in ("loss", "auc_nonzero")}))
        print(f"epoch {ep:2d} train {tl / len(perm):.4f} val {ev['loss']:.4f} auc {ev['auc_nonzero']} "
              f"tau0.5 P {ev['tau0.5']['precision']:.2f} R {ev['tau0.5']['recall']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or ev["loss"] < best[0]:
            best = (ev["loss"], ep, ev)
            torch.save(dict(state_dict=head.state_dict(), hidden=a.hidden, features=L.FEATURES, deltas=L.DELTAS,
                            class_weights=w.tolist(), epoch=ep, val=ev, labels_dir=str(a.labels_dir), n_files=nfiles),
                       out / "bias_head.pt")
    tr = rolls["has_target"] == 1
    summary = dict(n_files=nfiles, rollouts=int(len(tr)), rollouts_with_target=int(tr.sum()),
                   chosen_combo_counts={str(L.COMBOS[c]): int((rolls["chosen"][tr] == c).sum()) for c in range(len(L.COMBOS))},
                   train_instances=int(len(Yt)), val_instances=int(len(Yv)), train_class_counts=counts.tolist(),
                   class_weights=w.tolist(), best_epoch=best[1], val=best[2], history=hist)
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    v = best[2]
    md = ["# bias head 训练（v1）", "",
          f"- 标签：{nfiles} 个 batch 文件，{len(tr)} 条 rollout，其中 {int(tr.sum())} 条有目标车对。选中的组合：{json.dumps(summary['chosen_combo_counts'])}",
          f"- 样本：训练 {len(Yt)}，验证 {len(Yv)}（按场景划分）。训练集类别 {counts.tolist()}，权重 {[round(x, 2) for x in w.tolist()]}",
          f"- 最好的 epoch {best[1]}：验证 loss {v['loss']:.4f}，P(δ≠0) 的 AUC {v['auc_nonzero']}", "",
          "| τ | 预测非零 | 精确率 | 召回率 |", "|---|---|---|---|"]
    md += [f"| {t} | {v[f'tau{t}']['pred_nonzero']} | {v[f'tau{t}']['precision']:.3f} | {v[f'tau{t}']['recall']:.3f} |" for t in (0.3, 0.5, 0.7, 0.9)]
    md += ["", f"混淆矩阵（τ = 0.5，行 = 标签 0/2/4，列 = 预测）：{v['confusion_tau0.5']}", ""]
    (out / "train_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
