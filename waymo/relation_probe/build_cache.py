"""Build z[31] current-state relation probe caches.

The cache contains one scene-level latent array and one row per valid
focus-candidate pair.  It is shared by raw_only, raw_z, and raw_shuffled_z
probe runs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    WaymoVectorDataset,
    _collate,
    encode_current_z,
    json_dump,
    load_frozen_waymo_vector_tokenizer,
    move_batch,
    seed_everything,
    slice_context_window,
    worker_init_fn,
)
from labels import BINARY_NAMES, FEATURE_NAMES, REGRESSION_NAMES, build_scene_pair_labels, label_metadata


def _tensor_to_numpy_scene_agents(batch: Dict[str, Any], index: int) -> tuple[np.ndarray, np.ndarray]:
    agents = batch["agents"][index].detach().cpu().numpy()
    agent_mask = batch["agent_mask"][index].detach().cpu().numpy().astype(bool)
    return agents, agent_mask


def build_cache(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = WaymoVectorDataset(args.data_dir)
    if args.max_scenes > 0:
        indices = list(range(min(args.max_scenes, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
        collate_fn=_collate,
    )

    tokenizer, tok_args = load_frozen_waymo_vector_tokenizer(args.tokenizer_ckpt, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_paths: List[str] = []
    z_current: List[np.ndarray] = []
    pair_scene_index: List[np.ndarray] = []
    candidate_index: List[np.ndarray] = []
    pair_raw: List[np.ndarray] = []
    reg_targets: List[np.ndarray] = []
    reg_masks: List[np.ndarray] = []
    bin_targets: List[np.ndarray] = []
    bin_masks: List[np.ndarray] = []

    scene_offset = 0
    pair_count = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        batch_ctx = slice_context_window(batch, args.context_steps)
        with torch.no_grad():
            z = encode_current_z(tokenizer, move_batch(batch_ctx, device), args.query_step).detach().cpu().to(torch.float16).numpy()

        bsz = int(z.shape[0])
        for i in range(bsz):
            labels = build_scene_pair_labels(
                *_tensor_to_numpy_scene_agents(batch, i),
                query_step=args.query_step,
                focus_index=args.focus_index,
            )
            scene_paths.append(str(getattr(dataset, "dataset", dataset).paths[scene_offset + i] if hasattr(getattr(dataset, "dataset", dataset), "paths") else scene_offset + i))
            z_current.append(z[i])
            n_pairs = int(labels.candidate_index.shape[0])
            if n_pairs > 0:
                pair_scene_index.append(np.full((n_pairs,), scene_offset + i, dtype=np.int64))
                candidate_index.append(labels.candidate_index.astype(np.int64))
                pair_raw.append(labels.pair_raw.astype(np.float32))
                reg_targets.append(labels.reg_targets.astype(np.float32))
                reg_masks.append(labels.reg_masks.astype(np.float32))
                bin_targets.append(labels.bin_targets.astype(np.float32))
                bin_masks.append(labels.bin_masks.astype(np.float32))
                pair_count += n_pairs
        scene_offset += bsz

        if (batch_idx + 1) % args.log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"batch={batch_idx + 1} scenes={scene_offset} pairs={pair_count} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if pair_count == 0:
        raise RuntimeError("No valid focus-candidate pairs were found.")

    arrays = dict(
        scene_paths=np.asarray(scene_paths),
        z_current=np.stack(z_current, axis=0).astype(np.float16),
        pair_scene_index=np.concatenate(pair_scene_index, axis=0),
        candidate_index=np.concatenate(candidate_index, axis=0),
        pair_raw=np.concatenate(pair_raw, axis=0).astype(np.float32),
        reg_targets=np.concatenate(reg_targets, axis=0).astype(np.float32),
        reg_masks=np.concatenate(reg_masks, axis=0).astype(np.float32),
        bin_targets=np.concatenate(bin_targets, axis=0).astype(np.float32),
        bin_masks=np.concatenate(bin_masks, axis=0).astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        regression_names=np.asarray(REGRESSION_NAMES),
        binary_names=np.asarray(BINARY_NAMES),
    )
    cache_path = out_dir / f"{args.split}_cache.npz"
    np.savez(cache_path, **arrays)

    meta = {
        "split": args.split,
        "data_dir": args.data_dir,
        "tokenizer_ckpt": args.tokenizer_ckpt,
        "context_steps": args.context_steps,
        "query_step": args.query_step,
        "focus_index": args.focus_index,
        "num_scenes": int(arrays["z_current"].shape[0]),
        "num_pairs": int(arrays["pair_raw"].shape[0]),
        "z_shape": list(arrays["z_current"].shape),
        "pair_raw_shape": list(arrays["pair_raw"].shape),
        "reg_targets_shape": list(arrays["reg_targets"].shape),
        "bin_targets_shape": list(arrays["bin_targets"].shape),
        "tokenizer_args": tok_args,
        "labels": label_metadata(),
    }
    json_dump(out_dir / f"{args.split}_metadata.json", meta)
    print(f"saved cache: {cache_path}", flush=True)
    print(f"saved metadata: {out_dir / f'{args.split}_metadata.json'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build z[31] current-state relation probe cache.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tokenizer_ckpt", type=str, required=True)
    p.add_argument("--context_steps", type=int, default=32)
    p.add_argument("--query_step", type=int, default=31)
    p.add_argument("--focus_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    return p


if __name__ == "__main__":
    build_cache(build_argparser().parse_args())

