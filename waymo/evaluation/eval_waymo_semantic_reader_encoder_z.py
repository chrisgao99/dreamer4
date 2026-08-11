"""Compare the lightweight semantic reader and full decoder on encoder latents.

Both heads receive the exact same tokenizer-encoder z.  Continuous metrics use
all agent slots that are GT-valid (including the focus agent), while validity
accuracy is evaluated on every real agent slot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

WAYMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WAYMO_ROOT.parent
for path in (REPO_ROOT, WAYMO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from waymo.evaluation import eval_waymo_motion_latent_shared_rollout_horizons as shared_eval  # noqa: E402
from waymo.training.world_model import train_waymo_motion_latent_v1 as reader_train  # noqa: E402
from waymo.training.world_model import train_waymo_world_model as wm  # noqa: E402


class MetricSums:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.counts: dict[str, float] = {}

    def add(self, name: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        selected = values.float()[mask]
        self.sums[name] = self.sums.get(name, 0.0) + float(selected.sum().item())
        self.counts[name] = self.counts.get(name, 0.0) + float(selected.numel())

    def result(self) -> dict[str, float]:
        return {
            name: self.sums[name] / max(1.0, self.counts[name])
            for name in sorted(self.sums)
        }


def wrapped_yaw_error(pred_cont: torch.Tensor, target_cont: torch.Tensor) -> torch.Tensor:
    pred_yaw = torch.atan2(pred_cont[..., 5], pred_cont[..., 6])
    target_yaw = torch.atan2(target_cont[..., 5], target_cont[..., 6])
    delta = pred_yaw - target_yaw
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs()


def apply_reader_framewise(
    reader: torch.nn.Module, z: torch.Tensor, agent_mask: torch.Tensor
) -> Any:
    """Match reader training/consistency-loss usage, where every call has T=1."""
    bsz, time_steps, n_latents, d_bottleneck = z.shape
    n_agents = int(agent_mask.shape[-1])
    flat_z = z.reshape(bsz * time_steps, 1, n_latents, d_bottleneck)
    flat_mask = (
        agent_mask[:, None]
        .expand(bsz, time_steps, n_agents)
        .reshape(bsz * time_steps, n_agents)
    )
    flat = reader(flat_z, agent_mask=flat_mask)
    return type(flat)(
        continuous=flat.continuous.reshape(bsz, time_steps, n_agents, -1),
        valid_logits=flat.valid_logits.reshape(bsz, time_steps, n_agents),
        agent_tokens=flat.agent_tokens.reshape(bsz, time_steps, n_agents, -1),
    )


def apply_decoder_framewise(
    tokenizer: torch.nn.Module,
    z: torch.Tensor,
    agent_mask: torch.Tensor,
    light_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode each z with T=1, as required by a drop-in consistency head."""
    bsz, time_steps, n_latents, d_bottleneck = z.shape
    n_agents = int(agent_mask.shape[-1])
    n_lights = int(light_mask.shape[-1])
    flat_z = z.reshape(bsz * time_steps, 1, n_latents, d_bottleneck)
    flat_agent_mask = (
        agent_mask[:, None]
        .expand(bsz, time_steps, n_agents)
        .reshape(bsz * time_steps, n_agents)
    )
    flat_light_mask = light_mask.reshape(bsz * time_steps, 1, n_lights)
    flat = tokenizer.decoder(
        flat_z,
        agent_mask=flat_agent_mask,
        light_mask=flat_light_mask,
    )
    return (
        flat.agent_continuous.reshape(bsz, time_steps, n_agents, -1),
        flat.agent_valid_logits.reshape(bsz, time_steps, n_agents),
    )


def add_comparison_metrics(
    sums: MetricSums,
    *,
    prefix: str,
    pred_cont: torch.Tensor,
    target_cont: torch.Tensor,
    pred_valid: torch.Tensor,
    target_valid: torch.Tensor,
    gt_valid_mask: torch.Tensor,
    slot_mask: torch.Tensor,
) -> None:
    sums.add(
        f"{prefix}_xy_mae_m",
        torch.linalg.vector_norm(pred_cont[..., 0:2] - target_cont[..., 0:2], dim=-1),
        gt_valid_mask,
    )
    sums.add(
        f"{prefix}_velocity_mae_mps",
        torch.linalg.vector_norm(pred_cont[..., 3:5] - target_cont[..., 3:5], dim=-1),
        gt_valid_mask,
    )
    sums.add(f"{prefix}_yaw_mae_rad", wrapped_yaw_error(pred_cont, target_cont), gt_valid_mask)
    sums.add(f"{prefix}_valid_acc", (pred_valid == target_valid).float(), slot_mask)


@torch.no_grad()
def evaluate(
    tokenizer: torch.nn.Module,
    reader: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    tokenizer.eval()
    reader.eval()
    horizons = sorted(set(int(value) for value in args.horizons))
    totals = {horizon: MetricSums() for horizon in horizons}

    for batch_index, batch in enumerate(loader, start=1):
        batch = wm.slice_time_window(wm.move_batch(batch, device), args.eval_seq_len, random_start=False)
        z_enc, _, _ = wm.encode_batch_dynamics_inputs_for_world_model(tokenizer, batch, args)
        reader_pred = apply_reader_framewise(reader, z_enc, batch["agent_mask"])
        decoder_pred = wm.decode_batch_z_for_world_model(tokenizer, z_enc, batch, args)
        decoder_frame_cont, decoder_frame_logits = apply_decoder_framewise(
            tokenizer, z_enc, batch["agent_mask"], batch["light_mask"]
        )
        q_gt = wm.agents_to_btkf(batch["agents"], batch["agent_mask"])

        gt_yaw = q_gt[..., 6]
        gt_cont = torch.cat(
            [q_gt[..., 0:5], torch.sin(gt_yaw)[..., None], torch.cos(gt_yaw)[..., None]],
            dim=-1,
        ).float()
        gt_valid = q_gt[..., 5] > 0.5
        slot_mask = batch["agent_mask"].to(torch.bool)[:, None].expand_as(gt_valid)
        gt_valid_mask = slot_mask & gt_valid

        reader_cont = reader_pred.continuous.float()
        reader_valid = reader_pred.valid_logits > 0.0
        decoder_cont = decoder_pred.agent_continuous.float()
        decoder_valid = decoder_pred.agent_valid_logits > 0.0
        decoder_frame_cont = decoder_frame_cont.float()
        decoder_frame_valid = decoder_frame_logits > 0.0

        for horizon in horizons:
            start = int(args.eval_ctx)
            end = start + horizon
            region = (slice(None), slice(start, end), slice(None))
            endpoint = (slice(None), slice(end - 1, end), slice(None))
            sums = totals[horizon]
            for suffix, index in (("", region), ("_fde", endpoint)):
                add_comparison_metrics(
                    sums,
                    prefix=f"reader_gt{suffix}",
                    pred_cont=reader_cont[index],
                    target_cont=gt_cont[index],
                    pred_valid=reader_valid[index],
                    target_valid=gt_valid[index],
                    gt_valid_mask=gt_valid_mask[index],
                    slot_mask=slot_mask[index],
                )
                add_comparison_metrics(
                    sums,
                    prefix=f"decoder_gt{suffix}",
                    pred_cont=decoder_cont[index],
                    target_cont=gt_cont[index],
                    pred_valid=decoder_valid[index],
                    target_valid=gt_valid[index],
                    gt_valid_mask=gt_valid_mask[index],
                    slot_mask=slot_mask[index],
                )
                add_comparison_metrics(
                    sums,
                    prefix=f"decoder_framewise_gt{suffix}",
                    pred_cont=decoder_frame_cont[index],
                    target_cont=gt_cont[index],
                    pred_valid=decoder_frame_valid[index],
                    target_valid=gt_valid[index],
                    gt_valid_mask=gt_valid_mask[index],
                    slot_mask=slot_mask[index],
                )
                add_comparison_metrics(
                    sums,
                    prefix=f"reader_decoder{suffix}",
                    pred_cont=reader_cont[index],
                    target_cont=decoder_cont[index],
                    pred_valid=reader_valid[index],
                    target_valid=decoder_valid[index],
                    gt_valid_mask=gt_valid_mask[index],
                    slot_mask=slot_mask[index],
                )

        if batch_index == 1 or batch_index % 16 == 0 or batch_index == len(loader):
            print(f"encoder-z comparison progress {batch_index}/{len(loader)}", flush=True)

    return {f"h{horizon}": totals[horizon].result() for horizon in horizons}


def main(args: argparse.Namespace) -> None:
    required = int(args.eval_ctx) + max(args.horizons)
    if int(args.eval_seq_len) != required:
        raise ValueError(f"eval_seq_len must equal eval_ctx + max(horizons) = {required}")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    wm.seed_everything(args.seed)

    dataset = wm.WaymoVectorDataset(args.val_data_dir)
    indices, manifest = shared_eval.load_or_create_subset_manifest(
        dataset,
        path=Path(args.subset_manifest),
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
    )
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=wm.worker_init_fn,
        collate_fn=wm._collate,
    )
    tokenizer, tok_args = wm.load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    reader = reader_train.load_semantic_reader(
        args.semantic_reader_ckpt,
        tokenizer=tokenizer,
        tok_args=tok_args,
        device=device,
    )
    metrics = evaluate(tokenizer, reader, loader, device, args)
    for horizon, values in metrics.items():
        print(f"{horizon} " + ", ".join(f"{key}={value:.4f}" for key, value in values.items()), flush=True)

    payload: dict[str, Any] = {
        "protocol": "same_encoder_z_reader_vs_full_decoder",
        "reader_application": "framewise_t1_matching_reader_training_and_consistency_loss",
        "continuous_mask": "all agent slots, including focus, where GT valid",
        "validity_mask": "all real agent slots",
        "val_data_dir": str(Path(args.val_data_dir).resolve()),
        "subset_manifest": str(Path(args.subset_manifest).resolve()),
        "subset_selection": manifest["selection"],
        "subset_size": len(indices),
        "subset_seed": int(args.subset_seed),
        "eval_ctx": int(args.eval_ctx),
        "eval_seq_len": int(args.eval_seq_len),
        "horizons": sorted(set(int(value) for value in args.horizons)),
        "tokenizer_ckpt": str(Path(args.tokenizer_ckpt).resolve()),
        "semantic_reader_ckpt": str(Path(args.semantic_reader_ckpt).resolve()),
        "tokenizer_chunk_window": int(args.tokenizer_chunk_window),
        "tokenizer_chunk_stride": int(args.tokenizer_chunk_stride),
        "metrics": metrics,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote metrics: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data_dir", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--semantic_reader_ckpt", required=True)
    parser.add_argument("--subset_manifest", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_ctx", type=int, default=1)
    parser.add_argument("--eval_seq_len", type=int, default=91)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 30, 50, 80, 90])
    parser.add_argument("--tokenizer_chunk_window", type=int, default=32)
    parser.add_argument("--tokenizer_chunk_stride", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    main(parser.parse_args())
