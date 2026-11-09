#!/usr/bin/env python3
"""MoN-8 H40/execute5 full-80 fine-tune with an optional violation-aware winner rule.

  --selection ade          winner = argmin motion distance d (pure MoN, the rule of train_..._mon80.py)
  --selection constrained  pool = {k: d_k <= max(guard_ratio * min d, min d + guard_slack)}; in the pool the fewest
                           collision agents, then offroad agents, then solid-line agents, then the smallest d
                           (mon_violations.py: per-agent NEW violations, an agent counts once per category).
The loss is unchanged: the winner's motion distance, backpropagated through its full 80-step rollout.
Defaults (guard_ratio 1.3, collision_depth 0.2 m, offroad_margin 0.3 m, offroad_lane_threshold 1.5 m) were chosen in
step 0 (wm_interaction_probe/results/mon_select/step0/final_selection.txt, 2026-11-06).

Differences from train_..._mon80.py (both rules share them, so ade vs constrained is a clean comparison):
  * the K candidates of all B scenes are generated in ONE batched call (B*K rows) from per-candidate seeded
    initial flow noise (generate_poses(noise=...)); --grad_mode all keeps the graph of all rows and backpropagates
    only the winners' distance (no selection/replay mismatch possible), --grad_mode replay reruns the winners.
  * validation also reports per-agent violation rates of the closed-loop validation rollouts.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'waymo/core'))
from waymo.training.world_model.action_kinematics import inverse_model_actions  # noqa: E402
from waymo.training.world_model import train_waymo_direct_action_flow_erd as erd  # noqa: E402
from waymo.training.world_model.direct_action_flow_mon_losses import motion_distances  # noqa: E402
from waymo.training.world_model.direct_action_flow import agents_to_bntf, gather_agent_window, rollout_receding_horizon  # noqa: E402
from waymo.training.world_model.mon_long_rollout import generate_poses  # noqa: E402
from waymo.training.world_model.mon_violations import SceneReference, candidate_violations, violation_counts, select_winners  # noqa: E402
from waymo.training.world_model.multisample_validation import flow_erd_cpd_metrics  # noqa: E402
base = erd.base
ROLLOUT, COMMIT = 80, 5
SDPA_BATCH_LIMIT = 32768


def _chunked_sdpa(sdpa, limit=SDPA_BATCH_LIMIT):
    """The map->agent cross attention runs with batch = rows x map polylines (e.g. 16 x 4887 = 78192); the SDPA
    backward fails above the 65535 CUDA grid limit. Split such calls along the (independent) batch dimension."""
    def f(q, k, v, attn_mask=None, **kw):
        n = q.shape[0]
        if n <= limit:
            return sdpa(q, k, v, attn_mask=attn_mask, **kw)
        part = lambda x, s: x[s:s + limit] if x is not None and x.shape[0] == n else x  # noqa: E731
        return torch.cat([sdpa(part(q, s), part(k, s), part(v, s), attn_mask=part(attn_mask, s), **kw) for s in range(0, n, limit)], 0)
    f._chunked = True
    return f


if not getattr(torch.nn.functional.scaled_dot_product_attention, '_chunked', False):
    torch.nn.functional.scaled_dot_product_attention = _chunked_sdpa(torch.nn.functional.scaled_dot_product_attention)


def repeat_batch(batch, K):
    B = batch['agents'].shape[0]
    return {k: (v.repeat_interleave(K, 0) if torch.is_tensor(v) and v.ndim and v.shape[0] == B else v) for k, v in batch.items()}


def candidate_noise(seeds, B, N, H, device):
    """(replans, B*K, N, H, 3) initial flow noise; row b*K+k is candidate k of scene b, drawn from seeds[k]."""
    R = ROLLOUT // COMMIT
    out = []
    for s in seeds:
        g = torch.Generator(device=device).manual_seed(int(s))
        out.append(torch.randn((R, B, N, H, 3), generator=g, device=device))
    return torch.stack(out, 2).reshape(R, B * len(seeds), N, H, 3)


def generate_rows(model, normalizer, prepared, batch, noise, rows, solver_steps, detach_every):
    """Rollouts of the given candidate rows (indices into B*K), rows grouped per scene."""
    K = noise.shape[1] // prepared.history.shape[0]
    scene = rows // K
    sub = {k: (v[scene] if torch.is_tensor(v) and v.ndim and v.shape[0] == prepared.history.shape[0] else v) for k, v in batch.items()}
    return generate_poses(model, normalizer, prepared.history[scene], sub, prepared.anchors[scene], rollout_steps=ROLLOUT,
                          commitment=COMMIT, detach_every=detach_every, solver_steps=solver_steps, noise=noise[:, rows])


def scene_distances(poses, targets, scene):
    d, _ = motion_distances(poses.float(), targets.future_pose[scene], targets.current_pose[scene], targets.valid[scene])
    return d


def backward_select(model, normalizer, prepared, batch, step, cfg):
    B, N = prepared.history.shape[:2]; K = cfg.num_candidates; dev = prepared.history.device
    seeds = [cfg.seed + 10000000 + step * K + k for k in range(K)]
    noise = candidate_noise(seeds, B, N, model.horizon, dev)
    rows = torch.arange(B * K, device=dev); scene = rows // K
    t = prepared.targets
    if cfg.grad_mode == 'all':
        with torch.autocast('cuda', dtype=torch.bfloat16):
            poses = generate_rows(model, normalizer, prepared, batch, noise, rows, cfg.solver_steps, cfg.detach_every)
    else:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            poses = generate_rows(model, normalizer, prepared, batch, noise, rows, cfg.solver_steps, cfg.detach_every)
    d = scene_distances(poses, t, scene).view(B, K)
    if not torch.isfinite(d).all(): raise FloatingPointError('Nonfinite selection loss')
    with torch.no_grad():
        ref = SceneReference(t, prepared.history, batch, batch.get('agent_lengths'))
        v = candidate_violations(poses.detach(), ref, K, batch['map_polylines'], batch['map_mask'], cfg.offroad_lane_threshold)
        counts = violation_counts(v, cfg.collision_depth, cfg.offroad_margin).view(B, K, 3)
        dd = d.detach(); plain = dd.argmin(1)
        if cfg.selection == 'ade':
            winners, pool = plain, dd <= dd.min(1, keepdim=True).values
        else:
            winners, pool = select_winners(dd, counts, cfg.guard_ratio, cfg.guard_slack)
    if cfg.grad_mode == 'all':
        loss = d.gather(1, winners[:, None]).mean(); replay_error = 0.
    else:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            wrows = torch.arange(B, device=dev) * K + winners
            pw = generate_rows(model, normalizer, prepared, batch, noise, wrows, cfg.solver_steps, cfg.detach_every)
            dw = scene_distances(pw, t, torch.arange(B, device=dev))
        replay_error = float((dw.detach() - dd.gather(1, winners[:, None])[:, 0]).abs().max())
        loss = dw.mean()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite MoN loss')
    loss.backward()
    ar = torch.arange(B, device=dev)
    cw, cp = counts[ar, winners].float(), counts[ar, plain].float()
    clean = (counts.sum(-1) == 0)
    return dict(mon_loss=float(dd[ar, winners].mean()), plain_min_loss=float(dd.min(1).values.mean()),
                candidate_gt_loss_mean=float(dd.mean()), winner_changed=float((winners != plain).float().mean()),
                pool_size=float(pool.float().sum(1).mean()), clean_in_pool=float((clean & pool).any(1).float().mean()),
                clean_any=float(clean.any(1).float().mean()),
                winner_collision=float(cw[:, 0].mean()), winner_offroad=float(cw[:, 1].mean()), winner_line=float(cw[:, 2].mean()),
                plain_collision=float(cp[:, 0].mean()), plain_offroad=float(cp[:, 1].mean()), plain_line=float(cp[:, 2].mean()),
                candidate_collision=float(counts[..., 0].float().mean()), candidate_offroad=float(counts[..., 1].float().mean()),
                candidate_line=float(counts[..., 2].float().mean()), replay_max_error=replay_error)


@torch.no_grad()
def validate(model, normalizer, loader, ta, cfg):
    """ERD closed-loop validation (same seeds/rollouts; ADE/minADE/CPD on the first 8 batches = old val_ade_m)
    plus per-agent violation rates of the same rollouts on all cfg.eval_batches batches."""
    device = torch.device('cuda'); model.eval()
    acc = dict(ade=0., minade=0., cpd=0., cpd_valid=0., scenes=0., ade_all=0., scenes_all=0.,
               collision=0., offroad=0., line=0., frozen=0., agents=0., road_agents=0., moving=0.,
               coll_scene=0., rollouts=0., speed_pred=0., speed_gt=0.)
    for bi, raw in enumerate(loader):
        if bi >= cfg.eval_batches: break
        b = base.move_batch(raw, device)
        agents = agents_to_bntf(b['agents'], b['agent_mask'])
        anchor = ta.history_length - 1
        anchors = torch.full((agents.shape[0],), anchor, device=device, dtype=torch.long)
        history, future = gather_agent_window(agents, anchors, history_length=ta.history_length, horizon=ROLLOUT)
        targets = inverse_model_actions(model, history, future, b['agent_mask'], agent_lengths=b.get('agent_lengths'),
                                        max_displacement_m=ta.physical_max_displacement_m, max_yaw_delta_rad=ta.physical_max_yaw_delta_rad)
        poses = []
        for k in range(cfg.eval_rollouts):
            gen = torch.Generator(device=device).manual_seed(12346 + bi * cfg.eval_rollouts + k)
            poses.append(rollout_receding_horizon(model, normalizer, initial_history=history, agent_lengths=b.get('agent_lengths'),
                agent_mask=b['agent_mask'], map_polylines=b['map_polylines'], map_mask=b['map_mask'],
                current_light_sequence=b['lights'][:, anchor:anchor + ROLLOUT], current_light_mask_sequence=b['light_mask'][:, anchor:anchor + ROLLOUT],
                **(dict(map_ids=b['map_ids'], map_is_lane=b['map_is_lane'], map_stop_sign=b['map_stop_sign'],
                        map_stop_point=b['map_stop_point'], light_id_sequence=b['light_ids'][:, anchor:anchor + ROLLOUT]) if 'map_adaptation' in b else {}),
                **(dict(lane_edges=b['lane_edges'], lane_edge_types=b['lane_edge_types']) if 'lane_edges' in b else {}),
                focus_action_sequence=None, focus_action_valid=None, rollout_steps=ROLLOUT, commitment=COMMIT,
                solver_steps=cfg.solver_steps, generator=gen))
        poses = torch.stack(poses, 1).float()                                    # B,R,N,T,3
        valid = targets.valid; R = cfg.eval_rollouts
        distance = (poses[..., :2] - targets.future_pose[:, None, ..., :2]).norm(dim=-1)
        ade = (distance * valid[:, None]).sum((2, 3)) / valid.sum((1, 2))[:, None].clamp_min(1)
        acc['ade_all'] += float(ade.mean(1).sum()); acc['scenes_all'] += ade.shape[0]
        if bi < 8:
            context = history[:, :, -1, :2][:, None, None].expand(-1, R, -1, -1, -1)
            xy = torch.cat((context, poses[..., :2].permute(0, 1, 3, 2, 4)), 2)
            metadata = torch.cat((history[:, :, -1:].permute(0, 2, 1, 3), future.permute(0, 2, 1, 3)), 1)
            _, comp = flow_erd_cpd_metrics(xy, metadata, future_start=1, type_scales=cfg.cpd_scales, exclude_focus=False)
            cv = comp['scene_valid']
            acc['ade'] += float(ade.mean(1).sum()); acc['minade'] += float(ade.min(1).values.sum())
            acc['cpd'] += float(comp['scene_cpd'][cv].sum()); acc['cpd_valid'] += int(cv.sum()); acc['scenes'] += ade.shape[0]
        ref = SceneReference(targets, history, b, b.get('agent_lengths'))
        v = candidate_violations(poses.flatten(0, 1), ref, R, b['map_polylines'], b['map_mask'], cfg.offroad_lane_threshold)
        c = violation_counts(v, cfg.collision_depth, cfg.offroad_margin).float()  # (B*R,3)
        alive = b['agent_mask'].repeat_interleave(R, 0) & valid.repeat_interleave(R, 0).any(-1)
        types = ref.types.repeat_interleave(R, 0)
        moving = alive & (types == 1) & (v['gt_path'] > 5)
        acc['collision'] += float(c[:, 0].sum()); acc['offroad'] += float(c[:, 1].sum()); acc['line'] += float(c[:, 2].sum())
        acc['coll_scene'] += float((c[:, 0] > 0).sum()); acc['rollouts'] += c.shape[0]
        acc['agents'] += float(alive.sum()); acc['road_agents'] += float((alive & ((types == 1) | (types == 3))).sum())
        acc['frozen'] += float((moving & (v['pred_path'] < 1)).sum()); acc['moving'] += float(moving.sum())
        acc['speed_pred'] += float(v['pred_path'][moving].sum()); acc['speed_gt'] += float(v['gt_path'][moving].sum())
    if not acc['scenes'] or not acc['cpd_valid']: raise RuntimeError('Empty validation')
    return dict(val_ade_m=acc['ade'] / acc['scenes'], val_minade_m=acc['minade'] / acc['scenes'], val_cpd=acc['cpd'] / acc['cpd_valid'],
                val_scenes=acc['scenes'], val_ade_m_all=acc['ade_all'] / acc['scenes_all'], val_scenes_all=acc['scenes_all'],
                val_collision_agent_rate=acc['collision'] / acc['agents'], val_offroad_agent_rate=acc['offroad'] / acc['road_agents'],
                val_line_agent_rate=acc['line'] / acc['road_agents'], val_collision_scene_rate=acc['coll_scene'] / acc['rollouts'],
                val_frozen_moving_vehicle_rate=acc['frozen'] / max(1, acc['moving']),
                val_path_ratio_moving_vehicles=acc['speed_pred'] / max(1e-6, acc['speed_gt']))


def train(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high'); base.seed_everything(cfg.seed)
    device = torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and (out / 'latest.pt').exists(): raise FileExistsError('Use --resume')
    initial = torch.load(cfg.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    ta = argparse.Namespace(**initial['args']); stats = initial['action_stats']
    if ta.condition_focus_actions or ta.horizon != 40 or ta.commitment != 5 or getattr(ta, 'balance_dynamic_loss', False): raise ValueError('Expected generate-all H40/B5')
    rollout_args = argparse.Namespace(**vars(ta)); rollout_args.horizon = ROLLOUT
    model = base.create_model(ta).to(device); model.load_state_dict(initial['ema_model']); model.eval()
    ema = base.ModelEMA(model, cfg.ema_decay); normalizer = base.build_normalizer(stats, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=.01)
    step = 0; best = float('inf'); validation = {}
    if cfg.resume:
        c = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        for key in ('detach_every', 'checkpoint', 'selection', 'grad_mode', 'guard_ratio', 'guard_slack', 'collision_depth', 'offroad_margin', 'offroad_lane_threshold',
                    'seed', 'lr', 'batch_size', 'num_candidates', 'solver_steps', 'ema_decay'):
            if c['mon_args'][key] != getattr(cfg, key): raise ValueError('Resume config mismatch: ' + key)
        model.load_state_dict(c['model']); ema.model.load_state_dict(c['ema_model']); opt.load_state_dict(c['optimizer'])
        step = c['step']; best = c['best_val']; validation = c.get('validation', {}); erd.restore_rng(c['rng']); del c
    del initial
    from waymo.lane_connectivity.dataset import ConnectivityDataset
    if not (getattr(ta, 'map_adaptation', False) and getattr(ta, 'lane_connectivity', False)): raise ValueError('Expected laneconn model')
    ta.map_cache_dir = cfg.map_cache_dir or ta.map_cache_dir; ta.lane_topology_dir = cfg.lane_topology_dir or ta.lane_topology_dir
    make_dataset = lambda root: ConnectivityDataset(root, ta.map_cache_dir, ta.lane_topology_dir)  # noqa: E731
    data = make_dataset(cfg.train_data); val = make_dataset(cfg.val_data)
    stop = min(cfg.max_steps, step + cfg.max_iterations) if cfg.max_iterations else cfg.max_steps
    loader = DataLoader(data, batch_sampler=erd.StepBatches(len(data), cfg.batch_size, step, stop, cfg.seed + 100000),
        num_workers=cfg.num_workers, collate_fn=base.collate_vector_batch, pin_memory=True,
        persistent_workers=cfg.num_workers > 0, generator=torch.Generator().manual_seed(cfg.seed))
    vl = base.make_loader(val, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers, device=device)
    halted = [False]
    def stop_signal(sig, frame): halted[0] = True
    signal.signal(signal.SIGTERM, stop_signal); signal.signal(signal.SIGINT, stop_signal)
    (out / 'config.json').write_text(json.dumps(vars(cfg), indent=2) + '\n')
    def save(name):
        erd.atomic_save(out / name, dict(model=model.state_dict(), ema_model=ema.model.state_dict(), optimizer=opt.state_dict(),
            args=vars(ta), action_stats=stats, step=step, epoch=0, best_val=best, validation=validation,
            mon_args=vars(cfg), rng=erd.rng_state(), stage='mon_select_' + cfg.selection))
    def evaluate():
        state = erd.rng_state()
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16): r = validate(ema.model, normalizer, vl, ta, cfg)
            if not all(np.isfinite(v) for v in r.values()): raise FloatingPointError('Nonfinite validation')
            return r
        finally: erd.restore_rng(state)
    if not validation and not cfg.skip_validation:
        validation = evaluate(); best = validation['val_ade_m']; save('best_ade.pt')
        entry = dict(kind='validation', step=step, **validation)
        print('validation ' + json.dumps(entry), flush=True)
        with (out / 'metrics.jsonl').open('a') as handle: handle.write(json.dumps(entry) + '\n')
    started = time.monotonic()
    print(f'start selection={cfg.selection} grad_mode={cfg.grad_mode} step={step} target={cfg.max_steps} K={cfg.num_candidates}', flush=True)
    with (out / 'metrics.jsonl').open('a') as log:
        for raw in loader:
            batch = base.move_batch(raw, device)
            prepared = base.prepare_batch(batch, normalizer, rollout_args, random_start=False)
            opt.zero_grad(set_to_none=True)
            row = backward_select(model, normalizer, prepared, batch, step, cfg)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            if not np.isfinite(norm) or norm == 0: raise FloatingPointError('Invalid gradient')
            opt.step(); ema.update(model); step += 1
            row.update(step=step, grad_norm=norm, elapsed_seconds=time.monotonic() - started,
                       cuda_peak_memory_mb=torch.cuda.max_memory_allocated() / 1024 ** 2)
            if step % cfg.log_every == 0 or step <= 2:
                print('train ' + json.dumps(row), flush=True)
            log.write(json.dumps(row) + '\n'); log.flush()
            if step % cfg.eval_every == 0 and not cfg.skip_validation:
                validation = evaluate(); entry = dict(kind='validation', step=step, **validation)
                print('validation ' + json.dumps(entry), flush=True); log.write(json.dumps(entry) + '\n'); log.flush()
                if validation['val_ade_m'] < best: best = validation['val_ade_m']; save('best_ade.pt')
            if step % cfg.save_every == 0: save('latest.pt')
            if step % cfg.snapshot_every == 0: save(f'step_{step:07d}.pt')
            if halted[0]: break
        save('latest.pt')
        if step >= cfg.max_steps: save('final.pt')
    print(f'finished step={step} interrupted={halted[0]}', flush=True)
    if halted[0]: raise SystemExit(130)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ('checkpoint', 'train_data', 'val_data', 'output_dir'): p.add_argument('--' + key, required=True)
    p.add_argument('--map_cache_dir'); p.add_argument('--lane_topology_dir')
    p.add_argument('--detach_every', type=int, choices=(0, 5, 20), default=0)
    p.add_argument('--selection', choices=('ade', 'constrained'), required=True)
    p.add_argument('--grad_mode', choices=('all', 'replay'), default='all')
    p.add_argument('--resume'); p.add_argument('--skip_validation', action='store_true')
    for key, value in dict(seed=20260911, batch_size=2, num_workers=4, num_candidates=8, solver_steps=8, max_steps=10000, max_iterations=0,
                           eval_every=500, eval_batches=16, eval_batch_size=4, eval_rollouts=4,
                           save_every=100, snapshot_every=500, log_every=10).items(): p.add_argument('--' + key, type=int, default=value)
    for key, value in dict(lr=1e-6, ema_decay=.999, guard_ratio=1.3, guard_slack=0., collision_depth=.2, offroad_margin=.3, offroad_lane_threshold=1.5).items():
        p.add_argument('--' + key, type=float, default=value)
    p.add_argument('--cpd_scales', nargs=3, type=float, default=[25.423524547767016, 4.925798640017075, 18.77286026475977])
    c = p.parse_args(argv)
    if min(c.batch_size, c.max_steps, c.solver_steps, c.num_candidates, c.eval_every, c.save_every, c.snapshot_every, c.log_every) < 1: p.error('Positive sizes required')
    if c.eval_batches < 8: p.error('eval_batches >= 8 (the first 8 give the old val_ade_m)')
    return c


if __name__ == '__main__': train(parse_args())
