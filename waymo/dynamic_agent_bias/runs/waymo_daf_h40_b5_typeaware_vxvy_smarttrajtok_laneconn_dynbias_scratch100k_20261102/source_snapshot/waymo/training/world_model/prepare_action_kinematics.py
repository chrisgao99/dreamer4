#!/usr/bin/env python3
"""CPU-only preparation utilities. Never starts training or submits GPU work."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import time
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from waymo.core.waymo_vector_dataset import WaymoVectorDataset
from waymo.training.world_model.action_kinematics import estimate_rho, inverse_actions
from waymo.training.world_model.direct_action_flow import agents_to_bntf, wrap_angle_rad


def training_paths(root, max_files):
    paths = sorted(Path(root).glob('*.npz'))
    if not paths:
        raise FileNotFoundError(root)
    if max_files > 0 and len(paths) > max_files:
        paths = [paths[i] for i in np.linspace(0, len(paths)-1, max_files, dtype=int)]
    return paths


def read_agent_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        keys = ('agents', 'agent_mask', 'agent_lengths', 'agent_ids', 'scenario_id')
        if 'agent_lengths' not in data:
            raise ValueError(f'{path}: missing measured agent_lengths')
        return path, {key: data[key] for key in keys if key in data}


def agent_records(paths, workers=4):
    # Bounded, ordered I/O prefetch; do not load map/light arrays for preparation.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for start in range(0, len(paths), 32):
            yield from pool.map(read_agent_arrays, paths[start:start+32])


def calibrate(args):
    # Deduplicate identical tracks in multiple OOI-centered copies of a scenario.
    seen = set()
    fields = [[] for _ in range(5)]
    paths = training_paths(args.train_data, args.max_files)
    started = time.monotonic()
    for file_index, (path, data) in enumerate(agent_records(paths, getattr(args, 'workers', 4))):
        if 'agent_lengths' not in data:
            raise ValueError(f'{path}: missing agent_lengths; regenerate NPZ with updated converter')
        mask = torch.from_numpy(data['agent_mask']).bool()[None]
        agents = agents_to_bntf(torch.from_numpy(data['agents']).float()[None], mask)[0]
        lengths = torch.from_numpy(data['agent_lengths']).float()
        poses = agents[..., [0, 1, 6]]
        types = agents[:, :-1, 7].round().long()
        valid = mask[0, :, None] & (agents[:, :-1, 5] > .5) & (agents[:, 1:, 5] > .5)
        scenario = str(data['scenario_id'])
        for i, track in enumerate(data['agent_ids']):
            key = (scenario, int(track))
            if key in seen:
                valid[i] = False
            elif mask[0, i]:
                seen.add(key)
        turn = wrap_angle_rad(poses[:, 1:, 2] - poses[:, :-1, 2]).abs()
        keep = valid & ((types == 1) | (types == 3)) & (turn >= args.min_abs_yaw)
        values = (poses[:, :-1][keep], poses[:, 1:][keep], types[keep],
                  lengths[:, None].expand_as(types)[keep], valid[keep])
        for dest, value in zip(fields, values):
            dest.append(value)
        if (file_index + 1) % 256 == 0:
            print(f'calibration files={file_index+1}/{len(paths)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    tensors = [torch.cat(f) for f in fields]
    result = estimate_rho(*tensors, min_abs_yaw=args.min_abs_yaw,
                         max_abs_yaw=args.max_abs_yaw, max_displacement_m=args.max_displacement_m,
                         min_samples=args.min_samples)
    result.update(version=1, estimator='signed_median_eq12', source_train_data=str(Path(args.train_data).resolve()),
                  num_files=len(paths), unique_tracks=len(seen), min_abs_yaw=args.min_abs_yaw,
                  max_abs_yaw=args.max_abs_yaw, max_displacement_m=args.max_displacement_m)
    return result


def compute_type_aware_statistics(args, config):
    """Moments of recursively executable H-step targets at all training anchors."""
    paths = training_paths(args.data_dir, args.stats_max_files)
    count = torch.zeros(args.num_agent_types, dtype=torch.float64)
    total = torch.zeros(args.num_agent_types, 3, dtype=torch.float64)
    square = torch.zeros_like(total)
    started = time.monotonic()
    for file_index, (path, item) in enumerate(agent_records(paths, getattr(args, 'num_workers', 4))):
        mask = torch.from_numpy(item['agent_mask']).bool()[None]
        agents = agents_to_bntf(torch.from_numpy(item['agents']).float()[None], mask)[0]
        lengths = torch.from_numpy(item['agent_lengths']).float()[None]
        anchors = torch.arange(args.history_length-1, agents.shape[1]-args.horizon)
        if not len(anchors):
            continue
        # Evaluate every anchor together; exactly the same recursive targets as
        # the former per-anchor loop, with a leading batch dimension for anchors.
        hi = anchors[:, None] + torch.arange(1-args.history_length, 1)[None]
        fi = anchors[:, None] + torch.arange(1, args.horizon+1)[None]
        history = agents[:, hi].permute(1,0,2,3).contiguous()
        future = agents[:, fi].permute(1,0,2,3).contiguous()
        target = inverse_actions(history, future, mask.expand(len(anchors), -1),
            config=config, agent_lengths=lengths.expand(len(anchors), -1),
            max_displacement_m=args.physical_max_displacement_m,
            max_yaw_delta_rad=args.physical_max_yaw_delta_rad)
        for typ in range(args.num_agent_types):
            keep = target.valid & (target.agent_type[..., None] == typ)
            values = target.actions[keep].double()
            count[typ] += len(values)
            total[typ] += values.sum(0)
            square[typ] += values.square().sum(0)
        if (file_index+1) % 128 == 0:
            print(f'type-aware action-stats files={file_index+1}/{len(paths)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    if count.sum() == 0:
        raise ValueError('No valid type-aware targets; inspect calibration and rejection thresholds')
    mean_global = total.sum(0) / count.sum()
    var_global = (square.sum(0)/count.sum() - mean_global.square()).clamp_min(0)
    mean = torch.where(count[:, None] > 0, total/count[:, None].clamp_min(1), mean_global)
    var = torch.where(count[:, None] > 0, square/count[:, None].clamp_min(1)-mean.square(), var_global)
    std = var.clamp_min(0).sqrt().clamp_min(torch.tensor([.02, .02, .002]))
    return dict(version=3, action_kinematics=config.to_dict(), mean=mean.tolist(), std=std.tolist(),
                count=count.tolist(), num_files=len(paths), history_length=args.history_length,
                horizon=args.horizon, source_data_dir=str(Path(args.data_dir).resolve()),
                physical_max_displacement_m=args.physical_max_displacement_m,
                physical_max_yaw_delta_rad=args.physical_max_yaw_delta_rad,
                action_order=['a_longitudinal_m', 'a_lateral_m', 'delta_yaw_rad'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_data', required=True, help='Enriched training NPZ directory, never validation/test')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_files', type=int, default=8192)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--min_abs_yaw', type=float, default=.02)
    parser.add_argument('--max_abs_yaw', type=float, default=.75)
    parser.add_argument('--max_displacement_m', type=float, default=5.)
    parser.add_argument('--min_samples', type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    result = calibrate(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
