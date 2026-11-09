"""Build future-interaction probe caches.

This script collects current pair inputs, future-derived interaction labels,
and one z[query_step] latent array per scene.  It does not train a probe.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

WAYMO_ROOT = Path(__file__).resolve().parents[1]
RELATION_ROOT = WAYMO_ROOT / "relation_probe"
if str(RELATION_ROOT) not in sys.path:
    sys.path.append(str(RELATION_ROOT))

from common import (  # noqa: E402
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
from labels import (  # noqa: E402
    DIAGNOSTIC_NAMES,
    FEATURE_NAMES,
    RESPONSE_BINARY_NAMES,
    RESPONSE_REGRESSION_NAMES,
    TYPE_NAMES,
    InteractiveLabelConfig,
    build_scene_interactive_labels,
    label_metadata,
)


def _tensor_to_numpy_scene_agents(batch: Dict[str, Any], index: int) -> tuple[np.ndarray, np.ndarray]:
    agents = batch["agents"][index].detach().cpu().numpy()
    agent_mask = batch["agent_mask"][index].detach().cpu().numpy().astype(bool)
    return agents, agent_mask


def _label_config_from_args(args: argparse.Namespace) -> InteractiveLabelConfig:
    return InteractiveLabelConfig(
        dt=args.dt,
        future_steps=args.future_steps,
        relevance_dist_m=args.relevance_dist_m,
        path_overlap_dist_m=args.path_overlap_dist_m,
        pet_relevant_s=args.pet_relevant_s,
        same_direction_deg=args.same_direction_deg,
        crossing_heading_deg=args.crossing_heading_deg,
        oncoming_heading_deg=args.oncoming_heading_deg,
        same_corridor_lateral_m=args.same_corridor_lateral_m,
        following_headway_m=args.following_headway_m,
        following_relevant_headway_m=args.following_relevant_headway_m,
        converging_current_lateral_m=args.converging_current_lateral_m,
        converging_future_lateral_m=args.converging_future_lateral_m,
        priority_pet_s=args.priority_pet_s,
        priority_time_margin_s=args.priority_time_margin_s,
        yield_time_margin_s=args.yield_time_margin_s,
        speed_drop_mps=args.speed_drop_mps,
        decel_mps2=args.decel_mps2,
    )


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
    label_cfg = _label_config_from_args(args)

    scene_paths: List[str] = []
    z_current: List[np.ndarray] = []
    pair_scene_index: List[np.ndarray] = []
    candidate_index: List[np.ndarray] = []
    pair_raw: List[np.ndarray] = []
    relevance_targets: List[np.ndarray] = []
    relevance_masks: List[np.ndarray] = []
    type_targets: List[np.ndarray] = []
    type_masks: List[np.ndarray] = []
    response_bin_targets: List[np.ndarray] = []
    response_bin_masks: List[np.ndarray] = []
    response_reg_targets: List[np.ndarray] = []
    response_reg_masks: List[np.ndarray] = []
    diagnostics: List[np.ndarray] = []

    type_counts = np.zeros((len(TYPE_NAMES),), dtype=np.int64)
    response_bin_counts = np.zeros((len(RESPONSE_BINARY_NAMES),), dtype=np.int64)
    response_bin_denoms = np.zeros((len(RESPONSE_BINARY_NAMES),), dtype=np.int64)
    relevance_pos = 0
    relevance_total = 0

    scene_offset = 0
    pair_count = 0
    start_time = time.time()
    base_dataset = getattr(dataset, "dataset", dataset)
    for batch_idx, batch in enumerate(loader):
        batch_ctx = slice_context_window(batch, args.context_steps)
        with torch.no_grad():
            z = encode_current_z(tokenizer, move_batch(batch_ctx, device), args.query_step).detach().cpu().to(torch.float16).numpy()

        bsz = int(z.shape[0])
        for i in range(bsz):
            labels = build_scene_interactive_labels(
                *_tensor_to_numpy_scene_agents(batch, i),
                query_step=args.query_step,
                focus_index=args.focus_index,
                cfg=label_cfg,
            )
            if hasattr(base_dataset, "paths"):
                scene_paths.append(str(base_dataset.paths[scene_offset + i]))
            else:
                scene_paths.append(str(scene_offset + i))
            z_current.append(z[i])
            n_pairs = int(labels.candidate_index.shape[0])
            if n_pairs > 0:
                pair_scene_index.append(np.full((n_pairs,), scene_offset + i, dtype=np.int64))
                candidate_index.append(labels.candidate_index.astype(np.int64))
                pair_raw.append(labels.pair_raw.astype(np.float32))
                relevance_targets.append(labels.relevance_targets.astype(np.float32))
                relevance_masks.append(labels.relevance_masks.astype(np.float32))
                type_targets.append(labels.type_targets.astype(np.int64))
                type_masks.append(labels.type_masks.astype(np.float32))
                response_bin_targets.append(labels.response_bin_targets.astype(np.float32))
                response_bin_masks.append(labels.response_bin_masks.astype(np.float32))
                response_reg_targets.append(labels.response_reg_targets.astype(np.float32))
                response_reg_masks.append(labels.response_reg_masks.astype(np.float32))
                diagnostics.append(labels.diagnostics.astype(np.float32))

                relevance_pos += int(labels.relevance_targets.sum())
                relevance_total += int(labels.relevance_masks.sum())
                for type_idx in range(len(TYPE_NAMES)):
                    type_counts[type_idx] += int(((labels.type_targets == type_idx) & (labels.type_masks > 0.5)).sum())
                response_bin_counts += ((labels.response_bin_targets > 0.5) & (labels.response_bin_masks > 0.5)).sum(axis=0).astype(np.int64)
                response_bin_denoms += (labels.response_bin_masks > 0.5).sum(axis=0).astype(np.int64)
                pair_count += n_pairs
        scene_offset += bsz

        if (batch_idx + 1) % args.log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"batch={batch_idx + 1} scenes={scene_offset} pairs={pair_count} "
                f"relevance_pos={relevance_pos}/{max(1, relevance_total)} elapsed={elapsed:.1f}s",
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
        relevance_targets=np.concatenate(relevance_targets, axis=0).astype(np.float32),
        relevance_masks=np.concatenate(relevance_masks, axis=0).astype(np.float32),
        type_targets=np.concatenate(type_targets, axis=0).astype(np.int64),
        type_masks=np.concatenate(type_masks, axis=0).astype(np.float32),
        response_bin_targets=np.concatenate(response_bin_targets, axis=0).astype(np.float32),
        response_bin_masks=np.concatenate(response_bin_masks, axis=0).astype(np.float32),
        response_reg_targets=np.concatenate(response_reg_targets, axis=0).astype(np.float32),
        response_reg_masks=np.concatenate(response_reg_masks, axis=0).astype(np.float32),
        diagnostics=np.concatenate(diagnostics, axis=0).astype(np.float32),
        feature_names=np.asarray(FEATURE_NAMES),
        type_names=np.asarray(TYPE_NAMES),
        response_binary_names=np.asarray(RESPONSE_BINARY_NAMES),
        response_regression_names=np.asarray(RESPONSE_REGRESSION_NAMES),
        diagnostic_names=np.asarray(DIAGNOSTIC_NAMES),
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
        "relevance_shape": list(arrays["relevance_targets"].shape),
        "type_shape": list(arrays["type_targets"].shape),
        "response_bin_shape": list(arrays["response_bin_targets"].shape),
        "response_reg_shape": list(arrays["response_reg_targets"].shape),
        "label_config": vars(label_cfg),
        "label_counts": {
            "relevance_positive": int(relevance_pos),
            "relevance_total": int(relevance_total),
            "type_counts": {name: int(type_counts[i]) for i, name in enumerate(TYPE_NAMES)},
            "response_binary_positive": {name: int(response_bin_counts[i]) for i, name in enumerate(RESPONSE_BINARY_NAMES)},
            "response_binary_total": {name: int(response_bin_denoms[i]) for i, name in enumerate(RESPONSE_BINARY_NAMES)},
        },
        "tokenizer_args": tok_args,
        "labels": label_metadata(),
    }
    json_dump(out_dir / f"{args.split}_metadata.json", meta)
    print(f"saved cache: {cache_path}", flush=True)
    print(f"saved metadata: {out_dir / f'{args.split}_metadata.json'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build future-interaction relation probe cache.")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--tokenizer_ckpt", type=str, required=True)
    p.add_argument("--context_steps", type=int, default=32)
    p.add_argument("--query_step", type=int, default=31)
    p.add_argument("--future_steps", type=int, default=50)
    p.add_argument("--focus_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_scenes", type=int, default=0)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--relevance_dist_m", type=float, default=8.0)
    p.add_argument("--path_overlap_dist_m", type=float, default=4.0)
    p.add_argument("--pet_relevant_s", type=float, default=3.0)
    p.add_argument("--same_direction_deg", type=float, default=45.0)
    p.add_argument("--crossing_heading_deg", type=float, default=60.0)
    p.add_argument("--oncoming_heading_deg", type=float, default=135.0)
    p.add_argument("--same_corridor_lateral_m", type=float, default=4.5)
    p.add_argument("--following_headway_m", type=float, default=30.0)
    p.add_argument("--following_relevant_headway_m", type=float, default=20.0)
    p.add_argument("--converging_current_lateral_m", type=float, default=2.0)
    p.add_argument("--converging_future_lateral_m", type=float, default=3.5)
    p.add_argument("--priority_pet_s", type=float, default=4.0)
    p.add_argument("--priority_time_margin_s", type=float, default=0.2)
    p.add_argument("--yield_time_margin_s", type=float, default=0.5)
    p.add_argument("--speed_drop_mps", type=float, default=1.5)
    p.add_argument("--decel_mps2", type=float, default=1.0)
    return p


if __name__ == "__main__":
    build_cache(build_argparser().parse_args())
