#!/usr/bin/env python3
"""Recover a complete CSV prefix, evaluate its suffix, and merge scalar metrics.

Original CSVs are read-only. Detailed NPZ covers only the newly evaluated suffix.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import eval_waymo_direct_action_flow_multisample as evaluator


def read_rows(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def check_prefix(source, args):
    sf, scenes = read_rows(source / "scene_metrics.csv")
    rf, rollouts = read_rows(source / "rollout_ade.csv")
    batch_size, k = args.eval_batch_size, args.num_rollouts
    if not scenes or len(scenes) % batch_size or len(rollouts) != len(scenes) * k:
        raise ValueError("Source CSVs must contain the same complete batches; refusing partial or inconsistent files")
    if "nonfocus_scene_ade_m" not in rf:
        raise ValueError("This recovery helper requires conditioned/nonfocus CSVs")
    paths = evaluator.WaymoVectorDataset(str(Path(args.val_data_dir).resolve())).paths
    for i, scene in enumerate(scenes):
        batch, within = divmod(i, batch_size)
        for row in [scene] + rollouts[i*k:(i+1)*k]:
            if (int(row['dataset_index']), int(row['batch_index']), int(row['scene_in_batch'])) != (i, batch, within):
                raise ValueError(f"CSV index mismatch at scene {i}")
            if Path(row['npz_path']).name != Path(paths[i]).name:
                raise ValueError(f"Validation dataset ordering differs at scene {i}")
        for r, row in enumerate(rollouts[i*k:(i+1)*k]):
            if int(row['rollout_index']) != r or int(row['rollout_seed']) != args.seed + batch*k+r:
                raise ValueError(f"Rollout seed/order mismatch at scene {i}")
        values = np.array([float(r['nonfocus_scene_ade_m']) for r in rollouts[i*k:(i+1)*k]])
        if not np.isfinite(values).all() or not np.isclose(values.mean(), float(scene['mean_ade_over_rollouts_m']), rtol=1e-5, atol=1e-6) or not np.isclose(values.min(), float(scene['joint_minade_over_rollouts_m']), rtol=1e-5, atol=1e-6):
            raise ValueError(f"CSV ADE mismatch at scene {i}")
        # Legacy CSV omitted CPD validity. Strictly positive finite CPD proves
        # a nonempty roster; zero cannot distinguish invalid from deterministic.
        if not np.isfinite(float(scene['flow_erd_cpd'])) or float(scene['flow_erd_cpd']) <= 0:
            raise ValueError(f"Legacy CPD validity cannot be recovered at scene {i}")
    completed = len(scenes) // batch_size
    if not 0 < completed < args.eval_max_batches:
        raise ValueError("Expected a nonempty, unfinished evaluation")
    return completed, sf, scenes, rf, rollouts


def write_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--resume_from_dir', required=True)
    parser.add_argument('--check-only', action='store_true')
    own, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = evaluator.parse_args()
    source = Path(own.resume_from_dir).resolve()
    completed, sf, old_scenes, rf, old_rollouts = check_prefix(source, args)
    print(f'Resume checked: completed_batches={completed}; next_batch={completed+1}; remaining={args.eval_max_batches-completed}', flush=True)
    if own.check_only:
        return
    if args.focus_mode != 'conditioned':
        raise ValueError('Recovery requires --focus_mode conditioned')
    destinations = {key: Path(getattr(args, key)).resolve() for key in ('output_json', 'scene_metrics_csv', 'rollout_ade_csv', 'details_npz')}
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(f'Refusing overwrite: {path}')
    # Separate suffix outputs retain recoverable CSVs if this attempt stops too.
    suffix = destinations['output_json'].parent / 'suffix'
    args.output_json = str(suffix / 'result.json')
    args.scene_metrics_csv = str(suffix / 'scene_metrics.csv')
    args.rollout_ade_csv = str(suffix / 'rollout_ade.csv')
    args.details_npz = str(destinations['details_npz'])
    args.start_batch = completed
    result = evaluator.evaluate(args)
    _, new_scenes = read_rows(args.scene_metrics_csv)
    _, new_rollouts = read_rows(args.rollout_ade_csv)
    scenes, rollouts = old_scenes + new_scenes, old_rollouts + new_rollouts
    if len(scenes) != args.eval_max_batches * args.eval_batch_size:
        raise ValueError('Unexpected final scene count')
    ade = np.array([float(r['nonfocus_scene_ade_m']) for r in rollouts]).reshape(len(scenes), args.num_rollouts)
    old_n, new_n = len(old_scenes), len(new_scenes)
    old_cpd = np.array([float(r['flow_erd_cpd']) for r in old_scenes])
    old_unscaled = np.array([float(r['flow_erd_cpd_unscaled']) for r in old_scenes])
    new_valid = round(result['flow_erd_cpd_valid_scene_fraction'] * new_n)
    agents = np.array([int(r['valid_nonfocus_agents']) for r in scenes])
    agent_mins = np.array([float(r['per_agent_minade_m']) for r in scenes])
    result.update(
        mean_ade_m=float(ade.mean()), minade_m=float(ade.min(axis=1).mean()),
        first_rollout_ade_m=float(ade[:, 0].mean()), mean_ade_by_rollout_m=ade.mean(axis=0).tolist(),
        per_agent_minade_m=float((agents*agent_mins).sum()/agents.sum()),
        flow_erd_cpd=float((old_cpd.sum()+result['flow_erd_cpd']*new_valid)/(old_n+new_valid)),
        flow_erd_cpd_unscaled=float((old_unscaled.sum()+result['flow_erd_cpd_unscaled']*new_valid)/(old_n+new_valid)),
        flow_erd_cpd_valid_scene_fraction=(old_n+new_valid)/len(scenes),
        valid_nonfocus_agent_time_points_per_rollout=sum(int(r['valid_nonfocus_agent_time_points']) for r in scenes),
        scene_count=len(scenes), eval_batches=args.eval_max_batches, start_batch=0,
        resumed_from_batches=completed, resume_source_dir=str(source),
        details_complete=False, details_scope='new suffix scenes only; use NPZ dataset_index to identify scenes',
        details_scene_count=new_n, elapsed_seconds_scope='resume evaluation only',
        **{key: str(path) for key, path in destinations.items() if key != 'output_json'},
    )
    write_rows(destinations['scene_metrics_csv'], sf, scenes)
    write_rows(destinations['rollout_ade_csv'], rf, rollouts)
    evaluator.atomic_write_json(destinations['output_json'], result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
