#!/usr/bin/env python3
"""Closed-loop ADE of a direct-action-flow checkpoint on the whole-scene cache.

Protocol = the ERD/MON validation (``train_waymo_direct_action_flow_erd.validate``):
EMA weights, bf16, history 11, rollout 80 steps, generate H / execute B, K
rollouts with the same per-rollout seeds. Differences: every agent valid at
the current frame is simulated (SDC in slot 0, SDC-centered), and results are
reported per scene so they can be bucketed by agent count.

Metrics per scene (K rollouts):
  ade_chain / minade_chain   ERD definition: mean over the chain-valid inverse
                             targets of all agents; mean / min over rollouts.
                             Comparable to the checkpoint's ``val_ade_m``.
  ade_gt / minade_gt_agent   every step with a valid GT pose (WOSAC-style mask);
                             minade_gt_agent = per-agent min over rollouts, averaged.
  ade_gt_eval                ade_gt restricted to SDC + tracks_to_predict.
  minade_eval_agent          WOSAC-style: SDC + tracks_to_predict, per-agent ADE,
                             min over rollouts, averaged over agents (``ade_eval_agent``
                             is the same with the mean over rollouts).
  ade_gt_moving              ade_gt restricted to agents whose GT moves > 2 m.
  jump_gt5_fraction          executed steps moving > 5 m in 0.1 s.

Also runs the original legacy validation (focus-centered 32-agent NPZs) with
the unchanged ERD evaluator as a reproduction check of ``val_ade_m``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waymo.scene_cache.dataset import SceneCacheDataset, collate_scenes, read_manifest  # noqa: E402
from waymo.scene_cache.sparse_agent_attention import patch_from_args, patch_model, set_sdc_mask  # noqa: E402
from waymo.training.world_model import train_waymo_direct_action_flow as base  # noqa: E402
from waymo.training.world_model.action_kinematics import inverse_model_actions  # noqa: E402
from waymo.training.world_model.direct_action_flow import (  # noqa: E402
    agents_to_bntf, gather_agent_window, rollout_receding_horizon)

BUCKETS = ((1, 32), (33, 64), (65, 128), (129, 10**6))


def load_model(path, device, variant):
    """``variant``: ``ckpt`` (patches recorded in the checkpoint args), ``dense``
    or ``knn<K>_r<radius_m>`` (agent attention override, no SDC factorization)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ta = argparse.Namespace(**ckpt["args"])
    model = base.create_model(ta).to(device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    if variant == "ckpt":
        patch_from_args(model, ta)
    elif variant == "dense":
        patch_model(model, knn=0)
    else:
        knn_part, radius_part = variant.split("_")
        patch_model(model, knn=int(knn_part[3:]), radius_m=float(radius_part[1:]))
    normalizer = base.build_normalizer(ckpt["action_stats"], device)
    model.set_action_normalizer(normalizer)
    meta = dict(step=ckpt.get("step"), best_val=ckpt.get("best_val"), validation=ckpt.get("validation"))
    return model, normalizer, ta, ckpt.get("mon_args") or {}, meta


def select_scenes(manifest, n_random, n_large, large_min, seed):
    rows = read_manifest(manifest)
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n_random, len(rows)))
    taken = {r["scenario_id"] for r in picked}
    large = [r for r in rows if r["num_current_valid"] > large_min and r["scenario_id"] not in taken]
    picked += rng.sample(large, min(n_large, len(large)))
    return picked


def make_batches(rows, agent_budget, max_batch):
    """Sort by agent count; cap sum(N^2) per batch to bound pairwise memory."""
    rows = sorted(rows, key=lambda r: r["num_current_valid"])
    batches, current = [], []
    for row in rows:
        n = row["num_current_valid"]
        cost = (len(current) + 1) * max(n, current[-1]["num_current_valid"] if current else 0) ** 2
        if current and (cost > agent_budget or len(current) >= max_batch):
            batches.append(current)
            current = []
        current.append(row)
    if current:
        batches.append(current)
    return batches


@torch.no_grad()
def evaluate_batch(model, normalizer, ta, batch, rollouts, solver_steps, seed_base):
    agents = agents_to_bntf(batch["agents"], batch["agent_mask"])
    set_sdc_mask(model, batch.get("agent_is_sdc"))
    anchor = ta.history_length - 1
    horizon = min(80, int(agents.shape[2]) - anchor - 1)
    anchors = torch.full((agents.shape[0],), anchor, device=agents.device, dtype=torch.long)
    history, future = gather_agent_window(agents, anchors, history_length=ta.history_length, horizon=horizon)
    targets = inverse_model_actions(model, history, future, batch["agent_mask"],
                                    agent_lengths=batch.get("agent_lengths"),
                                    max_displacement_m=ta.physical_max_displacement_m,
                                    max_yaw_delta_rad=ta.physical_max_yaw_delta_rad)
    poses = []
    for k in range(rollouts):
        gen = torch.Generator(device=agents.device).manual_seed(seed_base + k)
        poses.append(rollout_receding_horizon(
            model, normalizer, initial_history=history, agent_lengths=batch.get("agent_lengths"),
            agent_mask=batch["agent_mask"], map_polylines=batch["map_polylines"], map_mask=batch["map_mask"],
            current_light_sequence=batch["lights"][:, anchor:anchor + horizon],
            current_light_mask_sequence=batch["light_mask"][:, anchor:anchor + horizon],
            map_ids=batch["map_ids"], map_is_lane=batch["map_is_lane"], map_stop_sign=batch["map_stop_sign"],
            map_stop_point=batch["map_stop_point"], light_id_sequence=batch["light_ids"][:, anchor:anchor + horizon],
            lane_edges=batch["lane_edges"], lane_edge_types=batch["lane_edge_types"],
            focus_action_sequence=None, focus_action_valid=None, rollout_steps=horizon,
            commitment=ta.commitment, solver_steps=solver_steps, generator=gen))
    poses = torch.stack(poses, 1).float()  # (B, K, N, T, 3)

    alive = batch["agent_mask"] & (history[:, :, -1, 5] > 0.5)
    chain = targets.valid.float()  # (B, N, T)
    gt_valid = ((future[..., 5] > 0.5) & alive[..., None]).float()
    eval_valid = gt_valid * batch["agent_eval_mask"][..., None].float()
    # Moving agents: GT pose ever more than 2 m from the current pose (parked cars
    # and standing pedestrians dominate crowded scenes and make all-agent ADE small).
    moved = ((future[..., 0:2] - history[:, :, -1:, 0:2]).norm(dim=-1) * gt_valid).amax(-1) > 2.0
    moving_valid = gt_valid * moved[..., None].float()
    d_chain = (poses[..., :2] - targets.future_pose[:, None, ..., :2].float()).norm(dim=-1)
    d_gt = (poses[..., :2] - future[:, None, ..., 0:2].float()).norm(dim=-1)

    def scene_mean(dist, mask):
        return (dist * mask[:, None]).sum((2, 3)) / mask.sum((1, 2))[:, None].clamp_min(1)

    ade_chain = scene_mean(d_chain, chain)
    ade_gt = scene_mean(d_gt, gt_valid)
    ade_eval = scene_mean(d_gt, eval_valid)
    ade_moving = scene_mean(d_gt, moving_valid)
    agent_ade = (d_gt * gt_valid[:, None]).sum(3) / gt_valid.sum(2)[:, None].clamp_min(1)  # (B, K, N)
    has_gt = gt_valid.sum(2) > 0
    minade_agent = (agent_ade.min(1).values * has_gt).sum(1) / has_gt.sum(1).clamp_min(1)
    # WOSAC-style minADE: evaluation agents only, per-agent min over rollouts.
    has_eval = has_gt & batch["agent_eval_mask"]
    minade_eval = (agent_ade.min(1).values * has_eval).sum(1) / has_eval.sum(1).clamp_min(1)
    ade_eval_agent = (agent_ade.mean(1) * has_eval).sum(1) / has_eval.sum(1).clamp_min(1)
    context = history[:, None, :, -1:, :2].float().expand(-1, rollouts, -1, -1, -1)
    step = torch.cat((context, poses[..., :2]), 3).diff(dim=3).norm(dim=-1)
    jumps = ((step > 5.0).float() * gt_valid[:, None]).sum((1, 2, 3)) / (gt_valid.sum((1, 2)) * rollouts).clamp_min(1)

    out = []
    for i in range(poses.shape[0]):
        out.append(dict(
            scenario_id=batch["scenario_id"][i], num_agents=int(alive[i].sum()),
            ade_chain=float(ade_chain[i].mean()), minade_chain=float(ade_chain[i].min()),
            ade_gt=float(ade_gt[i].mean()), minade_gt_agent=float(minade_agent[i]),
            ade_gt_eval=float(ade_eval[i].mean()),
            minade_eval_agent=float(minade_eval[i]), ade_eval_agent=float(ade_eval_agent[i]),
            ade_gt_moving=float(ade_moving[i].mean()) if moved[i].any() else float("nan"),
            num_moving=int(moved[i].sum()), jump_gt5_fraction=float(jumps[i]),
            chain_valid_fraction=float(chain[i].sum() / gt_valid[i].sum().clamp_min(1))))
    return out


def summarize(rows):
    keys = ("ade_chain", "minade_chain", "ade_gt", "minade_gt_agent", "ade_gt_eval",
            "minade_eval_agent", "ade_eval_agent", "ade_gt_moving",
            "num_moving", "jump_gt5_fraction", "chain_valid_fraction")
    def agg(sub):
        return dict(scenes=len(sub), mean_agents=round(float(np.mean([r["num_agents"] for r in sub])), 1),
                    **{k: round(float(np.nanmean([r[k] for r in sub])), 4) for k in keys}) if sub else dict(scenes=0)
    out = {"all": agg(rows)}
    for lo, hi in BUCKETS:
        out[f"agents_{lo}-{hi if hi < 10**6 else 'max'}"] = agg([r for r in rows if lo <= r["num_agents"] <= hi])
    return out


def run_legacy(model, normalizer, ta, mon_args, device, args):
    from waymo.training.world_model import train_waymo_direct_action_flow_erd as erd
    from waymo.lane_connectivity.dataset import ConnectivityDataset
    val = ConnectivityDataset(ta.val_data_dir, ta.map_cache_dir, ta.lane_topology_dir)
    loader = base.make_loader(val, batch_size=mon_args.get("eval_batch_size", 4), shuffle=False,
                              num_workers=4, device=device)
    cfg = argparse.Namespace(eval_batches=mon_args.get("eval_batches", 8), eval_rollouts=args.rollouts,
                             solver_steps=args.solver_steps, cpd_scales=mon_args.get("cpd_scales"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return erd.validate(model, normalizer, loader, ta, device, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="/p/yufeng/waymo_scene_prep/data/manifest_validation.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants", nargs="+", default=["dense", "knn32_r60"],
                        help="ckpt (checkpoint's own patches) | dense | knn<K>_r<radius_m>")
    parser.add_argument("--n_random", type=int, default=256)
    parser.add_argument("--n_large", type=int, default=256)
    parser.add_argument("--large_min", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--solver_steps", type=int, default=8)
    parser.add_argument("--agent_budget", type=int, default=40000, help="max batch_size * N^2")
    parser.add_argument("--max_batch", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_legacy", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = select_scenes(args.manifest, args.n_random, args.n_large, args.large_min, args.seed)
    with (out_dir / "scenes.jsonl").open("w") as handle:
        for row in scenes:
            handle.write(json.dumps(row) + "\n")
    batches = make_batches(scenes, args.agent_budget, args.max_batch)
    print(f"{len(scenes)} scenes in {len(batches)} batches; agents max {max(r['num_current_valid'] for r in scenes)}",
          flush=True)

    report = dict(checkpoint=args.checkpoint, manifest=args.manifest, args=vars(args), variants={})
    for variant in args.variants:
        model, normalizer, ta, mon_args, meta = load_model(args.checkpoint, device, variant)
        report["checkpoint_meta"] = meta
        result = {}
        if not args.skip_legacy:
            start = time.time()
            result["legacy_val_reproduction"] = run_legacy(model, normalizer, ta, mon_args, device, args)
            print(f"[{variant}] legacy val: {result['legacy_val_reproduction']} ({time.time()-start:.0f}s)", flush=True)

        dataset = SceneCacheDataset(scenes)
        index = {r["file"]: i for i, r in enumerate(scenes)}
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=[[index[r["file"]] for r in b] for b in batches],
            collate_fn=collate_scenes, num_workers=args.num_workers, pin_memory=True)
        rows, start = [], time.time()
        torch.cuda.reset_peak_memory_stats()
        per_scene = (out_dir / f"per_scene_{variant}.jsonl").open("w")
        for bi, raw in enumerate(loader):
            batch = base.move_batch(raw, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = evaluate_batch(model, normalizer, ta, batch, args.rollouts, args.solver_steps,
                                     12346 + bi * args.rollouts)
            for row in out:
                per_scene.write(json.dumps(row) + "\n")
            per_scene.flush()
            rows += out
            if bi % 10 == 0 or bi == len(batches) - 1:
                print(f"[{variant}] batch {bi+1}/{len(batches)} scenes {len(rows)} max_agents "
                      f"{max(r['num_agents'] for r in out)} ade_chain {np.mean([r['ade_chain'] for r in rows]):.3f} "
                      f"ade_gt {np.mean([r['ade_gt'] for r in rows]):.3f} | {time.time()-start:.0f}s "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
        per_scene.close()
        result["scene_cache"] = summarize(rows)
        result["seconds"] = round(time.time() - start, 1)
        result["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        report["variants"][variant] = result
        (out_dir / "summary.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({variant: result}, indent=1), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
