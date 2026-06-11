"""Compare vector-tokenizer reconstruction metrics on a Waymo validation split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Subset

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT / "training", WAYMO_ROOT / "core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_waymo_vector_tokenizer import build_argparser, build_model, compute_loss, metric_values  # noqa: E402
from vector_tokenizer_decoder import VectorDecoderOutput, _collate, _slice_time_window  # noqa: E402
from waymo_vector_dataset import WaymoVectorDataset  # noqa: E402


DEFAULT_MODELS = [
    (
        "fulltraj_trajloss_full91",
        str(WAYMO_ROOT / "checkpoints/ooi50k_lat16_d256_ep200_2a100_staticmap_v2_fulltraj_trajloss/latest.pt"),
        "full",
    ),
    (
        "staticmap_v2_chunk32",
        str(WAYMO_ROOT / "checkpoints/ooi50k_lat16_d256_ep200_2a100_staticmap_v2_lossfix/latest.pt"),
        "chunked",
    ),
    (
        "repeatmap_chunk32",
        str(WAYMO_ROOT / "checkpoints/ooi50k_lat16_d256_ep200_2gpu_lossfix/latest.pt"),
        "chunked",
    ),
]


def _checkpoint_args(ckpt: Dict) -> SimpleNamespace:
    defaults = vars(build_argparser().parse_args([]))
    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)
    defaults.update(ckpt_args)
    return SimpleNamespace(**defaults)


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _agents_btkf(agents: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    k = agent_mask.shape[-1]
    if agents.shape[1] == k:
        return agents.transpose(1, 2).contiguous()
    return agents


def _batch_time_length(batch: Dict[str, torch.Tensor]) -> int:
    k = batch["agent_mask"].shape[-1]
    if batch["agents"].shape[1] == k:
        return int(batch["agents"].shape[2])
    return int(batch["agents"].shape[1])


def _slice_batch_time(batch: Dict[str, torch.Tensor], start: int, end: int) -> Dict[str, torch.Tensor]:
    out = dict(batch)
    k = batch["agent_mask"].shape[-1]
    if batch["agents"].shape[1] == k:
        out["agents"] = batch["agents"][:, :, start:end]
    else:
        out["agents"] = batch["agents"][:, start:end]
    out["lights"] = batch["lights"][:, start:end]
    out["light_mask"] = batch["light_mask"][:, start:end]
    return out


def _concat_decoder_outputs(parts: List[VectorDecoderOutput]) -> VectorDecoderOutput:
    return VectorDecoderOutput(
        agent_continuous=torch.cat([p.agent_continuous for p in parts], dim=1),
        agent_valid_logits=torch.cat([p.agent_valid_logits for p in parts], dim=1),
        light_state_logits=torch.cat([p.light_state_logits for p in parts], dim=1),
        light_valid_logits=torch.cat([p.light_valid_logits for p in parts], dim=1),
        agent_tokens=torch.cat([p.agent_tokens for p in parts], dim=1),
        light_tokens=torch.cat([p.light_tokens for p in parts], dim=1),
        token_mask=torch.cat([p.token_mask for p in parts], dim=1),
    )


@torch.no_grad()
def _decode(model: torch.nn.Module, batch: Dict[str, torch.Tensor], *, mode: str, chunk_window: int) -> VectorDecoderOutput:
    if mode == "full":
        return model(
            agents=batch["agents"],
            agent_mask=batch["agent_mask"],
            map_polylines=batch["map_polylines"],
            map_mask=batch["map_mask"],
            lights=batch["lights"],
            light_mask=batch["light_mask"],
        ).decoder
    if mode == "window":
        windowed = _slice_time_window(batch, chunk_window)
        return model(
            agents=windowed["agents"],
            agent_mask=windowed["agent_mask"],
            map_polylines=windowed["map_polylines"],
            map_mask=windowed["map_mask"],
            lights=windowed["lights"],
            light_mask=windowed["light_mask"],
        ).decoder
    if mode != "chunked":
        raise ValueError(f"Unknown mode {mode!r}; expected full, chunked, or window.")
    if chunk_window <= 0:
        raise ValueError(f"chunk_window must be positive for chunked mode, got {chunk_window}")

    parts = []
    input_steps = _batch_time_length(batch)
    for start in range(0, input_steps, chunk_window):
        end = min(start + chunk_window, input_steps)
        chunk = _slice_batch_time(batch, start, end)
        parts.append(
            model(
                agents=chunk["agents"],
                agent_mask=chunk["agent_mask"],
                map_polylines=chunk["map_polylines"],
                map_mask=chunk["map_mask"],
                lights=chunk["lights"],
                light_mask=chunk["light_mask"],
            ).decoder
        )
    return _concat_decoder_outputs(parts)


def _sum_masked(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if values.dim() > mask.dim():
        mask = mask.unsqueeze(-1).expand_as(values)
    values = values[mask]
    if values.numel() == 0:
        return 0.0, 0.0
    return float(values.detach().float().sum().item()), float(values.numel())


def _wrapped_angle_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b)).abs()


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, float] = {}

    def add_mean(self, name: str, total: float, count: float) -> None:
        self.sums[name] = self.sums.get(name, 0.0) + float(total)
        self.counts[name] = self.counts.get(name, 0.0) + float(count)

    def add_scalar(self, name: str, value: float) -> None:
        self.add_mean(name, value, 1.0)

    def update_physical(
        self,
        pred: VectorDecoderOutput,
        batch: Dict[str, torch.Tensor],
        *,
        chunk_window: int,
    ) -> None:
        agents_btkf = _agents_btkf(batch["agents"], batch["agent_mask"])
        steps = min(int(agents_btkf.shape[1]), int(pred.agent_continuous.shape[1]))
        agents_btkf = agents_btkf[:, :steps]
        lights = batch["lights"][:, :steps]
        light_mask = batch["light_mask"][:, :steps].to(dtype=torch.bool)

        agent_slot_mask = batch["agent_mask"][:, None, :].to(device=agents_btkf.device, dtype=torch.bool)
        valid = (agents_btkf[..., 5] > 0.5) & agent_slot_mask
        pred_xy_m = pred.agent_continuous[:, :steps, :, 0:2] * 100.0
        target_xy_m = agents_btkf[..., 0:2]
        xy_err = (pred_xy_m - target_xy_m).norm(dim=-1)
        total, count = _sum_masked(xy_err, valid)
        self.add_mean("agent_xy_ade_m", total, count)
        self.add_mean("valid_agent_points", count, 1.0)

        if valid.shape[-1] > 0:
            focus_valid = valid[..., 0]
            focus_err = xy_err[..., 0]
            total, count = _sum_masked(focus_err, focus_valid)
            self.add_mean("focus_agent_xy_ade_m", total, count)

        any_valid = valid.any(dim=1)
        time_idx = torch.arange(steps, device=valid.device).view(1, -1, 1)
        last_idx = torch.where(valid, time_idx, torch.zeros_like(time_idx)).amax(dim=1)
        gather_idx = last_idx[:, None, :, None].expand(-1, 1, -1, 2)
        pred_final_xy = pred_xy_m.gather(dim=1, index=gather_idx).squeeze(1)
        target_final_xy = target_xy_m.gather(dim=1, index=gather_idx).squeeze(1)
        fde = (pred_final_xy - target_final_xy).norm(dim=-1)
        total, count = _sum_masked(fde, any_valid)
        self.add_mean("agent_fde_m", total, count)
        if any_valid.shape[-1] > 0:
            total, count = _sum_masked(fde[..., 0], any_valid[..., 0])
            self.add_mean("focus_agent_fde_m", total, count)

        if steps > 1:
            consecutive_valid = valid[:, 1:, :] & valid[:, :-1, :]
            pred_delta = pred_xy_m[:, 1:] - pred_xy_m[:, :-1]
            target_delta = target_xy_m[:, 1:] - target_xy_m[:, :-1]
            delta_err = (pred_delta - target_delta).norm(dim=-1)
            total, count = _sum_masked(delta_err, consecutive_valid)
            self.add_mean("agent_delta_xy_mae_m", total, count)

            if chunk_window > 0:
                boundary_indices = [i for i in range(chunk_window, steps, chunk_window)]
                if boundary_indices:
                    idx = torch.tensor([i - 1 for i in boundary_indices], device=delta_err.device, dtype=torch.long)
                    boundary_err = delta_err.index_select(dim=1, index=idx)
                    boundary_valid = consecutive_valid.index_select(dim=1, index=idx)
                    total, count = _sum_masked(boundary_err, boundary_valid)
                    self.add_mean("chunk_boundary_delta_xy_mae_m", total, count)

        pred_speed = pred.agent_continuous[:, :steps, :, 2] * 30.0
        speed_err = (pred_speed - agents_btkf[..., 2]).abs()
        total, count = _sum_masked(speed_err, valid)
        self.add_mean("agent_speed_mae_mps", total, count)

        pred_vel = pred.agent_continuous[:, :steps, :, 3:5] * 30.0
        vel_err = (pred_vel - agents_btkf[..., 3:5]).norm(dim=-1)
        total, count = _sum_masked(vel_err, valid)
        self.add_mean("agent_vxvy_mae_mps", total, count)

        pred_yaw = torch.atan2(pred.agent_continuous[:, :steps, :, 5], pred.agent_continuous[:, :steps, :, 6])
        yaw_err = _wrapped_angle_error(pred_yaw, agents_btkf[..., 6]) * (180.0 / math.pi)
        total, count = _sum_masked(yaw_err, valid)
        self.add_mean("agent_yaw_mae_deg", total, count)

        valid_pred = pred.agent_valid_logits[:, :steps] > 0.0
        valid_acc = (valid_pred == valid).float()
        valid_acc_mask = agent_slot_mask.expand_as(valid)
        total, count = _sum_masked(valid_acc, valid_acc_mask)
        self.add_mean("agent_valid_acc", total, count)

        light_valid_pred = pred.light_valid_logits[:, :steps] > 0.0
        light_valid_acc = (light_valid_pred == light_mask).float()
        self.add_mean("light_valid_acc", float(light_valid_acc.sum().item()), float(light_valid_acc.numel()))

        if light_mask.any():
            light_state_target = lights[..., 2].long().clamp(min=0, max=pred.light_state_logits.shape[-1] - 1)
            light_state_pred = pred.light_state_logits[:, :steps].argmax(dim=-1)
            light_state_acc = (light_state_pred == light_state_target).float()
            total, count = _sum_masked(light_state_acc, light_mask)
            self.add_mean("light_state_acc", total, count)

        self.add_mean("recon_steps", float(steps) * float(agents_btkf.shape[0]), float(agents_btkf.shape[0]))

    def to_dict(self) -> Dict[str, float]:
        out = {}
        for key, total in sorted(self.sums.items()):
            denom = self.counts.get(key, 0.0)
            out[key] = float("nan") if denom <= 0 else total / denom
        return out


def _load_model(checkpoint: Path, sample: Dict[str, torch.Tensor], device: torch.device) -> tuple[torch.nn.Module, SimpleNamespace, Dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    args = _checkpoint_args(ckpt)
    n_agents = int(sample["agent_mask"].shape[-1])
    n_lights = int(sample["lights"].shape[1])
    model = build_model(args, n_agents=n_agents, n_lights=n_lights)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, args, ckpt


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_models(values: Iterable[List[str]] | None) -> List[tuple[str, str, str]]:
    models = DEFAULT_MODELS if values is None else [tuple(v) for v in values]
    out = []
    for label, checkpoint, mode in models:
        if mode not in {"full", "chunked", "window"}:
            raise ValueError(f"Model {label}: mode must be full, chunked, or window, got {mode!r}")
        out.append((label, checkpoint, mode))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Compare vector-tokenizer reconstruction metrics on Waymo val data.")
    p.add_argument("--data_dir", type=str, default=str(WAYMO_ROOT / "data/waymo_vector_dataset_ooi_centered_50k/val"))
    p.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("LABEL", "CHECKPOINT", "MODE"),
        help="Model spec. MODE is full, chunked, or window. Defaults compare the three current checkpoints.",
    )
    p.add_argument("--chunk_window", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=0, help="Optional smoke-test limit before evaluating the full val set.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--summary_csv",
        type=str,
        default=str(WAYMO_ROOT / "evaluation/reports/val_reconstruction_compare/summary.csv"),
    )
    p.add_argument(
        "--summary_json",
        type=str,
        default=str(WAYMO_ROOT / "evaluation/reports/val_reconstruction_compare/summary.json"),
    )
    p.add_argument("--progress_every", type=int, default=50)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = WaymoVectorDataset(args.data_dir)
    if args.max_samples > 0:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    sample = dataset[0]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        collate_fn=_collate,
    )

    rows: List[Dict[str, object]] = []
    for label, checkpoint_str, mode in _parse_models(args.model):
        checkpoint = Path(checkpoint_str)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{label}: checkpoint not found: {checkpoint}")
        print(f"Evaluating {label}: mode={mode}, checkpoint={checkpoint}")
        model, ckpt_args, ckpt = _load_model(checkpoint, sample, device)
        chunk_window = int(args.chunk_window if mode == "chunked" else getattr(ckpt_args, "time_window", args.chunk_window))
        if mode == "window" and chunk_window <= 0:
            chunk_window = int(args.chunk_window)

        accum = MetricAccumulator()
        loss_totals: Dict[str, float] = {}
        loss_count = 0
        started = time.time()
        for batch_idx, batch in enumerate(loader, start=1):
            batch = _move_batch(batch, device)
            pred = _decode(model, batch, mode=mode, chunk_window=chunk_window)
            eval_batch = batch if mode != "window" else _slice_time_window(batch, chunk_window)
            accum.update_physical(pred, eval_batch, chunk_window=(chunk_window if mode == "chunked" else 0))
            loss, loss_metrics = compute_loss(SimpleNamespace(decoder=pred), eval_batch, ckpt_args)
            values = metric_values(loss_metrics)
            values["loss_total"] = float(loss.detach().float().item())
            for key, value in values.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + value
            loss_count += 1
            if args.progress_every > 0 and batch_idx % args.progress_every == 0:
                elapsed = time.time() - started
                print(f"  {label}: batches={batch_idx}, samples~={batch_idx * args.batch_size}, elapsed={elapsed:.1f}s")

        metrics = accum.to_dict()
        row: Dict[str, object] = {
            "label": label,
            "mode": mode,
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(ckpt.get("step", -1)),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "encoder_variant": getattr(ckpt_args, "encoder_variant", ""),
            "time_window": int(getattr(ckpt_args, "time_window", 0)),
            "eval_chunk_window": int(chunk_window),
            "num_samples": len(dataset),
        }
        for key, value in metrics.items():
            row[key] = value
        for key, total in sorted(loss_totals.items()):
            row[f"batchavg_{key}"] = total / max(1, loss_count)
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(Path(args.summary_csv), rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.summary_csv}")
    print(f"wrote {args.summary_json}")

    key_metrics = ["agent_xy_ade_m", "agent_fde_m", "focus_agent_xy_ade_m", "focus_agent_fde_m", "agent_delta_xy_mae_m"]
    print()
    print("Key metrics, lower is better:")
    for row in rows:
        parts = [f"{key}={float(row[key]):.4f}" for key in key_metrics if key in row and row[key] == row[key]]
        print(f"  {row['label']}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
