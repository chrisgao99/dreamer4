#!/usr/bin/env python3
"""512-scene closed-loop eval for 1b-v2: fine-tuned WM, z v2 (unknown / oracle / reversed).

Same as eval_zbias.py (unchanged ``eval_waymo_direct_action_flow_multisample.evaluate``, same metrics
and seed formula), with the v2 pieces: labels v2 records, OracleZV2 (same-direction rules on the
rollout's current poses and speeds) and g v2 (same-direction edge flag). ``--mode unknown`` installs
only compat, i.e. the checkpoint's own unconditioned behaviour.
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
from waymo.pair_order_z.eval_zbias import eval_namespace  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("unknown", "oracle", "reversed"), required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--zbias_ckpt", default=None)
    p.add_argument("--labels", default=str(compat.DATA / "pair_order_z_labels_v2" / "val.npz"))
    p.add_argument("--max_deviation_m", type=float, default=2.0)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_max_batches", type=int, default=128)
    p.add_argument("--start_batch", type=int, default=0)
    p.add_argument("--num_rollouts", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=12346)
    p.add_argument("--no_physical_metrics", action="store_true")
    return p.parse_args()


def install_z_hooks(ev, a, out_dir: Path):
    from waymo.pair_order_z import z_bias_v2 as zb
    import waymo.lane_connectivity.dataset as lcd

    labels = zb.PairLabelsV2(a.labels)
    lcd.ConnectivityDataset = zb.with_z_pairs_v2(lcd.ConnectivityDataset, labels)
    device = torch.device("cuda")
    g, g_info = zb.load_zbias_v2(a.zbias_ckpt, device)
    state_dir = out_dir / "z_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    stash = {"batch": None, "index": a.start_batch - 1, "codes": [], "bias": zb.BiasStats()}

    def flush():
        if stash["codes"]:
            codes = np.stack([np.stack(r) for r in stash["codes"]])        # (R, K, B, P) int8
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
        oracle = zb.OracleZV2(batch["agents"], batch["agent_mask"], batch["z_pairs"],
                              reversed_order=(a.mode == "reversed"), max_deviation_m=a.max_deviation_m)
        codes_this = []
        encode = model.encode_scene

        def encode_scene(**skw):
            scene = encode(**skw)
            speed = skw["history"][:, :, -1, 2]
            zcls, codes = oracle.step(scene.agent_pose, scene.agent_mask, speed)
            codes_this.append(codes.to(torch.int8).cpu().numpy())
            bias = g(zcls, scene.agent_pose, speed, scene.agent_mask, oracle.kind)
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
        if not a.zbias_ckpt:
            raise ValueError("--zbias_ckpt is required for oracle / reversed")
        flush, stash, g_info = install_z_hooks(ev, a, out_dir)
    result = ev.evaluate(eval_namespace(a))
    extra = {"z_mode": a.mode, "zbias": g_info, "labels": a.labels if a.mode != "unknown" else None,
             "checkpoint": a.checkpoint, "max_deviation_m": a.max_deviation_m, "num_rollouts": a.num_rollouts}
    if flush is not None:
        flush()
        extra["bias_stats"] = stash["bias"].summary()
    (out_dir / "z_eval_info.json").write_text(json.dumps(extra, indent=2) + "\n")
    print("z_eval_info " + json.dumps(extra), flush=True)
    print(f"saved {out_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
