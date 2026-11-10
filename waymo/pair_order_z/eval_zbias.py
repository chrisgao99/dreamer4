#!/usr/bin/env python3
"""512-scene closed-loop eval with a z source (plan §8.3): unknown / oracle / reversed.

Runs the unchanged ``eval_waymo_direct_action_flow_multisample.evaluate`` (same
metrics, outputs, seeds). Differences from a plain run:
  * compat: map-cache identity check maps the old server's path prefix;
  * oracle/reversed: the dataset adds padded ``z_pairs``; every rollout's
    ``model.encode_scene`` is wrapped so each replan adds g's bias as
    ``scene.chunk_bias`` (denoiser agent_attn only).
``--mode unknown`` installs nothing but compat: it is the baseline.
``--zero_g`` uses an untrained (zero-output) g: must reproduce the baseline exactly.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("unknown", "oracle", "reversed"), required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--zbias_ckpt", default=None)
    p.add_argument("--zero_g", action="store_true")
    p.add_argument("--labels", default=str(compat.DATA / "pair_order_z_labels_v1" / "val.npz"))
    p.add_argument("--checkpoint", default=str(compat.STAGE1_CKPT))
    p.add_argument("--max_deviation_m", type=float, default=2.0)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_max_batches", type=int, default=128)
    p.add_argument("--start_batch", type=int, default=0)
    p.add_argument("--num_rollouts", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=12346)
    p.add_argument("--no_physical_metrics", action="store_true")
    return p.parse_args()


def eval_namespace(a: argparse.Namespace) -> argparse.Namespace:
    """Same arguments as launch_eval_mapadapt_laneconn_512_tmux.sh."""
    out = Path(a.out_dir)
    return argparse.Namespace(
        checkpoint=a.checkpoint, val_data_dir=str(compat.VAL_DIR),
        map_cache_dir=str(compat.MAP_CACHE), lane_topology_dir=str(compat.LANE_TOPOLOGY),
        output_json=str(out / "result.json"), rollout_ade_csv=str(out / "rollout_ade.csv"),
        scene_metrics_csv=str(out / "scene_metrics.csv"), details_npz=str(out / "details.npz"),
        start_batch=a.start_batch, metric_validity="holonomic",
        physical_metrics=not a.no_physical_metrics, device="cuda", weights="ema",
        focus_mode="generate_all", eval_batch_size=a.eval_batch_size,
        eval_max_batches=a.eval_max_batches, num_workers=a.num_workers,
        num_rollouts=a.num_rollouts, rollout_steps=80, commitment_steps=5, solver_steps=8,
        seed=a.seed, log_every=1, cpd_type_scales=list(compat.CPD_SCALES),
    )


def install_z_hooks(ev, a: argparse.Namespace, out_dir: Path):
    from waymo.pair_order_z import z_bias as zb
    import waymo.lane_connectivity.dataset as lcd

    labels = zb.PairLabels(a.labels)
    lcd.ConnectivityDataset = zb.with_z_pairs(lcd.ConnectivityDataset, labels)
    device = torch.device("cuda")
    if a.zero_g:
        g = zb.ZBias().to(device).eval()
        g_info = {"zero_g": True}
    else:
        if not a.zbias_ckpt:
            raise ValueError("--zbias_ckpt is required unless --zero_g")
        g, g_info = zb.load_zbias(a.zbias_ckpt, device)
    state_dir = out_dir / "z_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    stash = {"batch": None, "index": a.start_batch - 1, "codes": [], "bias": zb.BiasStats()}

    def flush():
        if stash["codes"]:
            # (R, K, B, P) int8 pair codes per rollout and replan; see z_bias.CODE_NAMES.
            codes = np.stack([np.stack(r) for r in stash["codes"]])
            zb.atomic_save_npz(state_dir / f"batch_{stash['index']:05d}.npz", codes=codes,
                               z_pairs=stash["batch"]["z_pairs"].cpu().numpy())
            stash["codes"] = []

    original_move = ev.move_batch

    def move_batch(raw, dev):
        flush()
        batch = original_move(raw, dev)
        stash["batch"] = batch
        stash["index"] += 1
        return batch

    original_rollout = ev.rollout_receding_horizon

    def rollout(model, normalizer, **kw):
        batch = stash["batch"]
        oracle = zb.OracleZ(batch["agents"], batch["agent_mask"], batch["z_pairs"],
                            reversed_order=(a.mode == "reversed"),
                            max_deviation_m=a.max_deviation_m)
        codes_this = []
        encode = model.encode_scene

        def encode_scene(**skw):
            scene = encode(**skw)
            speed = skw["history"][:, :, -1, 2]
            zcls, codes = oracle.step(scene.agent_pose, scene.agent_mask)
            codes_this.append(codes.to(torch.int8).cpu().numpy())
            bias = g(zcls, scene.agent_pose, speed, scene.agent_mask)
            if bias is None:
                return scene
            stash["bias"].update(bias, zcls)
            return replace(scene, chunk_bias=bias)

        model.encode_scene = encode_scene
        try:
            return original_rollout(model, normalizer, **kw)
        finally:
            del model.encode_scene
            stash["codes"].append(codes_this)

    ev.move_batch = move_batch
    ev.rollout_receding_horizon = rollout
    return flush, stash, g_info


def main() -> None:
    a = parse_args()
    compat.patch_map_cache_identity()
    from waymo.evaluation import eval_waymo_direct_action_flow_multisample as ev

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flush = stash = None
    g_info = {}
    if a.mode != "unknown":
        flush, stash, g_info = install_z_hooks(ev, a, out_dir)
    result = ev.evaluate(eval_namespace(a))
    extra = {"z_mode": a.mode, "zbias": g_info, "labels": a.labels if a.mode != "unknown" else None,
             "max_deviation_m": a.max_deviation_m}
    if flush is not None:
        flush()
        extra["bias_stats"] = stash["bias"].summary()
    (out_dir / "z_eval_info.json").write_text(json.dumps(extra, indent=2) + "\n")
    print("z_eval_info " + json.dumps(extra), flush=True)
    print(f"saved {out_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
