#!/usr/bin/env python3
"""Fixed ten-scene flow-noise coverage experiment; 32 joint rollouts per scene."""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from waymo.evaluation import visualize_direct_action_flow_intersection_two_agent_rollouts as vis
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REFERENCE = ROOT / 'waymo/eval_results/world_model/typeaware_vxvy_finetune_step1000_intersection10_n10_plan40_b40_h80_20261001/run_config.json'


def diverse_parents(poses, slots, count=6):
    # Farthest-point selection in output trajectory space, both cars, every 1 s.
    features = poses[:, slots, 9::10, :2].flatten(1).float()
    first = int(((features - features.mean(0)) ** 2).mean(1).argmax())
    selected = [first]
    distances = ((features - features[first]) ** 2).mean(1)
    for _ in range(count - 1):
        nxt = int(distances.argmax())
        selected.append(nxt)
        distances = torch.minimum(distances, ((features - features[nxt]) ** 2).mean(1))
        distances[selected] = -1
    return selected


def visitation(curves, bounds, cell=1.0):
    xmin, xmax, ymin, ymax = bounds
    xe = np.arange(xmin, xmax + cell, cell)
    ye = np.arange(ymin, ymax + cell, cell)
    counts = np.zeros((len(xe) - 1, len(ye) - 1), dtype=np.float32)
    for curve in curves:
        points = []
        for a, b in zip(curve[:-1], curve[1:]):
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / (cell / 4))) + 1)
            points.append(np.linspace(a, b, n))
        points = np.concatenate(points)
        hist = np.histogram2d(points[:, 0], points[:, 1], bins=(xe, ye))[0]
        counts += hist > 0
    return counts.T / len(curves), xe, ye


def plot_scene(out, record, gt, poses, maps, masks, ids, mode):
    slots = [0, int(record['target_slot'])]
    curves = poses[:, slots, :, :2]
    current = gt[10, slots, :2]
    curves = np.concatenate((np.broadcast_to(current[None, :, None], (32, 2, 1, 2)), curves), axis=2)
    gt_curves = [gt[10:, slot, :2][gt[10:, slot, 5] > .5] for slot in slots]
    pts = np.concatenate([curves.reshape(-1, 2), *gt_curves])
    lo, hi = pts.min(0) - 10, pts.max(0) + 10
    span = max(hi - lo)
    middle = (hi + lo) / 2
    bounds = (middle[0] - span / 2, middle[0] + span / 2,
              middle[1] - span / 2, middle[1] + span / 2)
    fig, axes = plt.subplots(2, 3, figsize=(19, 12), dpi=130, constrained_layout=True)
    base_count = 32 if mode == 'random32' else 16
    heat_cmap = LinearSegmentedColormap.from_list(
        'light_to_dark_blue', plt.get_cmap('Blues')(np.linspace(.3, 1, 256)))
    stats = {}
    for row, slot in enumerate(slots):
        c = curves[:, row]
        heat, xe, ye = visitation(c[:base_count], bounds)
        all_heat, _, _ = visitation(c, bounds)
        for ax in axes[row]:
            ax.set_facecolor('#eeeeee')
            vis._draw_map(ax, maps, masks)
            ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:]); ax.set_aspect('equal')
            ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
            ax.plot(gt_curves[row][:, 0], gt_curves[row][:, 1], 'k--', lw=2, zorder=6, label='GT (logged valid)')
            ax.scatter(*current[row], c='black', marker='s', s=30, zorder=10)
        ax = axes[row, 0]
        for i, curve in enumerate(c):
            active = mode != 'random32' and i >= 16
            ax.plot(curve[:, 0], curve[:, 1], color='#e87521' if active else '#197bbd', alpha=.42, lw=1,
                    label=('active exploration' if active else 'base samples') if i == 0 or (active and i == 16) else None)
        ax.annotate(f"{'Ego' if row == 0 else 'Target'} | agent ID: {int(ids[slot])}",
                    xy=(0, .5), xycoords='axes fraction', xytext=(-65, 0),
                    textcoords='offset points', rotation=90, ha='center', va='center',
                    fontsize=16, fontweight='bold')
        if row == 0:
            ax.set_title('32 joint-rollout paths', fontsize=18, pad=14)
        ax.legend(fontsize=8)
        ax = axes[row, 1]
        hm = ax.pcolormesh(xe, ye, np.ma.masked_equal(heat, 0), cmap=heat_cmap, vmin=0, vmax=1, alpha=.9, zorder=3)
        if row == 0:
            ax.set_title(f'8 s visit probability\n{base_count} base samples, 1 m cells', fontsize=18, pad=14)
        ax.legend(fontsize=8, loc='upper right')
        fig.colorbar(hm, ax=ax, fraction=.035, label='Fraction of base rollouts visiting cell')
        if mode != 'random32':
            extra = (all_heat > 0) & (heat == 0)
            ax.pcolormesh(xe, ye, np.ma.masked_equal(extra.astype(float), 0), cmap='Oranges', vmin=0, vmax=1, alpha=.5, zorder=2)
            ax.text(.02, .02, 'Orange: extra explored coverage (NOT probability)', transform=ax.transAxes, fontsize=8)
        ax = axes[row, 2]
        for t, color in zip((20, 40, 60, 80), ('#277da8', '#43aa8b', '#f8961e', '#b5179e')):
            ax.scatter(c[:base_count, t, 0], c[:base_count, t, 1], c=color, s=15, alpha=.7, label=f'{t / 10:g} s base')
            if mode != 'random32':
                ax.scatter(c[16:, t, 0], c[16:, t, 1], edgecolors=color, facecolors='none', s=25, alpha=.7)
        if row == 0:
            title = 'Time-specific positions'
            if mode != 'random32':
                title += '\nHollow = active exploration'
            ax.set_title(title, fontsize=18, pad=14)
        ax.legend(fontsize=8)
        stats[str(slot)] = {'base_visited_area_m2': int((heat > 0).sum()),
                            'all_visited_area_m2': int((all_heat > 0).sum()),
                            'endpoint_max_pair_distance_m': float(np.linalg.norm(c[:, None, -1] - c[None, :, -1], axis=-1).max())}
    fig.suptitle(f"{record['scenario_id']} | {mode} | EMA step1000 | plan40 / commitment40 / 80 steps\n"
                 'Observed sample coverage, not a support bound; active samples are not density estimates', fontsize=13)
    path = out / f"scene_{int(record['sample_order']):02d}_{record['scenario_id']}.png"
    fig.savefig(path)
    plt.close(fig)
    return path, stats


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['random32', 'paired16_active16'], required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--max_scenes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    config = json.loads(REFERENCE.read_text())
    config.update(mode=args.mode, output_dir=str(out), device='cuda:0', num_rollouts=32,
                  cuda_visible_devices='0', batch_size=args.batch_size,
                  sampling_protocol='random32: iid Gaussian; paired16_active16: 8 Gaussian +/- pairs, then 12 mutations of 6 output-diverse parents + 4 fresh Gaussian samples',
                  active_mutation='0.8 * parent + 0.6 * independent Gaussian; parents chosen using both cars at 1-second intervals; both replans mutated',
                  plot_protocol='Base-only per-rollout grid visit frequency; active extra coverage separate; raw positions at 2/4/6/8 seconds',
                  metric_validity='raw logged GT for display; generated trajectories never truncated by inverse-action masks')
    (out / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (out / 'status.json').write_text(json.dumps({'state': 'loading', 'completed_scenes': 0}))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 0 unavailable')
    vis.seed_everything(config['seed'])
    model, normalizer, train_args, step = vis.load_model_and_normalizer(Path(config['checkpoint']), device, 'ema')
    assert not vis.resolve_focus_conditioning(train_args, 'generate_all')
    assert model.horizon == 40 and train_args.history_length == 11
    records = json.loads((ROOT / config['selected_manifest']).read_text())['samples']
    assert len(records) == 10
    records = records[:args.max_scenes]
    dataset = vis.WaymoVectorDataset(str(ROOT / config['val_data_dir']))
    loader = vis.make_loader(vis.Subset(dataset, [int(r['dataset_index']) for r in records]), batch_size=1,
                             shuffle=False, num_workers=0, device=device)
    images, summaries = [], []
    for scene_index, (record, raw) in enumerate(zip(records, loader)):
        print(f"START scene={scene_index} id={record['scenario_id']} mode={args.mode}", flush=True)
        batch = vis.move_batch(raw, device)
        agents = vis.agents_to_bntf(batch['agents'], batch['agent_mask'])
        history = agents[:, :, :11]
        ids = batch['agent_ids'][0].cpu().numpy()
        slot = int(record['target_slot'])
        assert int(ids[slot]) == int(record['target_track_id']) and record['target_type'] == 'vehicle'
        seed = int(config['seed']) + int(record['sample_order'])
        rng = torch.Generator(device=device).manual_seed(seed)
        shape = (2, agents.shape[1], 40, 3)
        def draw(n):
            return torch.randn((n, *shape), generator=rng, device=device)
        def rollout(noise):
            results = []
            for start in range(0, len(noise), args.batch_size):
                z = noise[start:start + args.batch_size]
                def rep(x):
                    return vis._repeat_batch(x, len(z))
                result = vis.rollout_receding_horizon(model, normalizer,
                    initial_history=rep(history), agent_lengths=rep(batch['agent_lengths']),
                    agent_mask=rep(batch['agent_mask']), map_polylines=rep(batch['map_polylines']),
                    map_mask=rep(batch['map_mask']), current_light_sequence=rep(batch['lights'][:, 10:90]),
                    current_light_mask_sequence=rep(batch['light_mask'][:, 10:90]),
                    focus_action_sequence=None, focus_action_valid=None,
                    rollout_steps=80, commitment=40, solver_steps=8, noise_sequence=z)
                results.append(result)
            return torch.cat(results)
        metadata = []
        if args.mode == 'random32':
            noise = draw(32)
            poses = rollout(noise)
            metadata = [{'source': 'iid_gaussian'} for _ in range(32)]
        else:
            z = draw(8)
            base = torch.stack((z, -z), dim=1).flatten(0, 1)
            base_poses = rollout(base)
            parents = diverse_parents(base_poses, [0, slot])
            mutated = torch.stack([.8 * base[p] + .6 * draw(1)[0] for p in parents for _ in range(2)])
            extra = torch.cat((mutated, draw(4)))
            extra_poses = rollout(extra)
            noise = torch.cat((base, extra))
            poses = torch.cat((base_poses, extra_poses))
            metadata = [{'source': 'paired_gaussian', 'pair': i // 2, 'sign': 1 if i % 2 == 0 else -1} for i in range(16)]
            metadata += [{'source': 'active_mutation', 'parent': p} for p in parents for _ in range(2)]
            metadata += [{'source': 'fresh_exploration'} for _ in range(4)]
            assert torch.equal(noise[:16:2], -noise[1:16:2])
        assert poses.shape[0] == 32 and torch.isfinite(poses).all()
        gt = agents[0].permute(1, 0, 2).cpu().numpy()
        pose_np = poses.cpu().numpy()
        stem = f"scene_{int(record['sample_order']):02d}_{record['scenario_id']}"
        np.savez_compressed(out / (stem + '.npz'), poses=pose_np, noise_sequence=noise.cpu().numpy(),
                            gt_tkf=gt, agent_ids=ids, selected_slots=np.array([0, slot]),
                            map_polylines=batch['map_polylines'][0].cpu().numpy(), map_mask=batch['map_mask'][0].cpu().numpy())
        (out / (stem + '_sampling.json')).write_text(json.dumps({'seed': seed, 'samples': metadata}, indent=2))
        path, stats = plot_scene(out, record, gt, pose_np, batch['map_polylines'][0].cpu().numpy(),
                                 batch['map_mask'][0].cpu().numpy().astype(bool), ids, args.mode)
        images.append(path)
        summaries.append({'record': record, 'coverage': stats, 'image': str(path)})
        (out / 'coverage_summary.json').write_text(json.dumps(summaries, indent=2))
        (out / 'status.json').write_text(json.dumps({'state': 'running', 'completed_scenes': len(images)}))
        print(f'DONE scene={scene_index} image={path}', flush=True)
    vis._make_contact_sheet(images, out / 'contact_sheet.png')
    (out / 'status.json').write_text(json.dumps({'state': 'complete', 'completed_scenes': len(images)}))
    print(f'COMPLETE {args.mode}: {len(images)} scenes', flush=True)


if __name__ == '__main__':
    main()
