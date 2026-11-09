#!/usr/bin/env python3
"""Matched CUDA0 baseline + lane graph only, fresh 100k stage-one experiment."""
import argparse
import json
from pathlib import Path
import re
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from waymo.training.world_model import train_waymo_direct_action_flow as base
from waymo.lane_connectivity.dataset import ConnectivityDataset


def training_args(topology_dir,run_name,data_root=None,map_cache_dir=None):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_name):raise ValueError('Invalid run name')
    saved=json.loads(Path(__file__).with_name('baseline_cuda0_args.json').read_text())
    source=Path(data_root).resolve() if data_root else ROOT/'data/waymo_vector_dataset_ooi_centered_50k_with_lengths'
    saved.update(data_dir=str(source/'train'),val_data_dir=str(source/'val'),
                 ckpt_dir=str(ROOT/'waymo/checkpoints'/run_name),wandb_run_name=run_name,
                 map_cache_dir=str(Path(map_cache_dir).resolve()) if map_cache_dir else str(ROOT/'data/waymo_map_smarttrajtok_5m_v2'),
                 lane_connectivity=True,lane_topology_dir=str(Path(topology_dir).resolve()))
    for key in ['action_stats_path','kinematics_calibration']:saved[key]=str(source/Path(saved[key]).name)
    if saved['resume'] is not None:raise ValueError('Expected baseline from scratch')
    return argparse.Namespace(**saved)


def preflight(args,report_path=None,paths=None,gpu=False):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    base.resolve_kinematics_args(args);stats=base.load_or_compute_action_statistics(args)
    device=torch.device('cuda' if gpu else 'cpu')
    normalizer=base.build_normalizer(stats,device);model=base.create_model(args).to(device)
    reports={}
    for split,root in [('train',args.data_dir),('val',args.val_data_dir)]:
        dataset=ConnectivityDataset(root,args.map_cache_dir,args.lane_topology_dir)
        indices=[dataset.paths.index(p) for p in paths if Path(p).parent.name==split] if paths else [0,len(dataset)//3,2*len(dataset)//3,len(dataset)-1]
        if not indices:continue
        take=args.batch_size if gpu else 2
        batch=base.move_batch(base.collate_vector_batch([dataset[i] for i in indices[:take]]),device)
        prepared=base.prepare_batch(batch,normalizer,args,random_start=False)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=gpu):
            scene=model.encode_scene(**base.scene_kwargs(batch,prepared))
            loss,_=base.flow_matching_loss(model,scene,prepared.normalized_actions,prepared.targets.valid,
                 condition_focus_actions=args.condition_focus_actions,loss_type=args.flow_loss_type,huber_beta=args.flow_huber_beta)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite topology preflight loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise FloatingPointError('Nonfinite topology gradients')
        graph_edges=(batch['lane_edge_types']>0).sum().item()
        report=dict(files=len(indices[:take]),flow_loss=float(loss.detach()),edges=graph_edges,
                    map_segments=batch['map_mask'].any(-1).sum(-1).tolist(),
                    gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20 if gpu else None)
        reports[split]=report;print(json.dumps(dict(event='connectivity_preflight',device=str(device),split=split,**report)),flush=True)
        model.zero_grad(set_to_none=True)
    if report_path:Path(report_path).write_text(json.dumps(reports,indent=2)+'\n')
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topology_dir',required=True);p.add_argument('--data_root',default=None);p.add_argument('--map_cache_dir',default=None)
    p.add_argument('--run_name',default='waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k_20261007')
    p.add_argument('--preflight_only',action='store_true');p.add_argument('--skip_preflight',action='store_true')
    options=p.parse_args();args=training_args(options.topology_dir,options.run_name,options.data_root,options.map_cache_dir)
    checkpoint=Path(args.ckpt_dir)
    if checkpoint.exists() and any(checkpoint.iterdir()):raise FileExistsError('Fresh run checkpoint already exists')
    summary=json.loads((Path(options.topology_dir)/'summary.json').read_text())
    if not summary['complete'] or summary['files']!=50000:raise ValueError('Incomplete topology cache')
    if summary['map_cache_dir']!=args.map_cache_dir:raise ValueError('Topology/baseline map root mismatch')
    for split,root in [('train',args.data_dir),('val',args.val_data_dir)]:
        expected={p.name for p in Path(root).glob('*.npz')}
        actual={p.name for p in (Path(options.topology_dir)/split).glob('*.npz')}
        if expected!=actual:raise ValueError(f'Incomplete topology split: {split}')
    run_dir=ROOT/'waymo/data_prep/runs'/options.run_name;run_dir.mkdir(parents=True,exist_ok=True)
    if not options.skip_preflight:
        preflight(args,run_dir/'cpu_preflight.json')
        preflight(args,run_dir/'cuda1_bf16_preflight.json',gpu=True)
    (run_dir/'resolved_train_args.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    if not options.preflight_only:
        print(json.dumps(dict(event='start_lane_connectivity_training',cuda_visible_device=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),max_steps=args.max_steps)),flush=True)
        base.train(args)

if __name__=='__main__':main()
