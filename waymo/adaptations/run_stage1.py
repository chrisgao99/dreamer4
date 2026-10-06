#!/usr/bin/env python3
"""Fresh stage-one training with frozen baseline argv + map adaptations only."""
import argparse
import json
from pathlib import Path
import sys
import re
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from waymo.training.world_model import train_waymo_direct_action_flow as base
from waymo.adaptations.map_dataset import MapAdaptationDataset


def training_args(cache_dir,run_name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_name):raise ValueError('Invalid run name')
    payload=json.loads(Path(__file__).with_name('baseline_launch_config.json').read_text())
    argv=payload['train_argv'][:]
    replacements={'--ckpt_dir':str(ROOT/'waymo/checkpoints'/run_name),'--wandb_run_name':run_name}
    for flag,value in replacements.items():argv[argv.index(flag)+1]=value
    argv += ['--map_adaptation','--map_cache_dir',str(Path(cache_dir).resolve()),
             '--map_neighbors','32','--agent_map_neighbors','64',
             '--map_radius_m','30','--agent_map_radius_m','100']
    return base.build_arg_parser().parse_args(argv)


def preflight(args,report_path=None,paths=None):
    torch.set_num_threads(1)
    base.resolve_kinematics_args(args)
    stats=base.load_or_compute_action_statistics(args)
    normalizer=base.build_normalizer(stats,torch.device('cpu'))
    torch.manual_seed(args.seed)
    model=base.create_model(args)
    reports={}
    for split,root in [('train',args.data_dir),('val',args.val_data_dir)]:
        dataset=MapAdaptationDataset(root,args.map_cache_dir)
        if paths:
            indices=[dataset.paths.index(p) for p in paths if Path(p).parent.name==split]
        else:
            indices=[0,len(dataset)//2,len(dataset)-1]
        if not indices:continue
        batch=base.collate_vector_batch([dataset[i] for i in indices[:2]])
        prepared=base.prepare_batch(batch,normalizer,args,random_start=False)
        scene=model.encode_scene(**base.scene_kwargs(batch,prepared))
        loss,_=base.flow_matching_loss(model,scene,prepared.normalized_actions,prepared.targets.valid,
                                       condition_focus_actions=False,loss_type=args.flow_loss_type,
                                       huber_beta=args.flow_huber_beta)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preflight loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite preflight gradient')
        model.zero_grad(set_to_none=True)
        reports[split]=dict(files=[dataset.paths[i] for i in indices[:2]],
            map_segments=batch['map_mask'].any(-1).sum(-1).tolist(),
            stop_sign_segments=batch['map_stop_sign'].sum(-1).tolist(),
            observed_signals=batch['light_mask'].sum((1,2)).tolist(),flow_loss=float(loss.detach()))
        print(json.dumps(dict(event='preflight',split=split,**reports[split])),flush=True)
    if report_path:Path(report_path).write_text(json.dumps(reports,indent=2)+'\n')
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map_cache_dir',required=True)
    p.add_argument('--run_name',default='waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_scratch100k_20261006')
    p.add_argument('--preflight_only',action='store_true');p.add_argument('--skip_preflight',action='store_true')
    args=p.parse_args();train_args=training_args(args.map_cache_dir,args.run_name)
    ckpt=Path(train_args.ckpt_dir)
    if ckpt.exists() and any(ckpt.iterdir()):raise FileExistsError(f'Fresh-run checkpoint directory exists: {ckpt}')
    summary=json.loads((Path(args.map_cache_dir)/'summary.json').read_text())
    if summary['data_root'] != str(Path(train_args.data_dir).parent.resolve()):
        raise ValueError('Map cache was generated for a different dataset')
    if summary['segment_length_m'] != 5. or summary['points_per_segment'] != 6:
        raise ValueError('Expected 5m/6-point map cache')
    if not summary['complete'] or summary['files']!=50000:
        raise ValueError('Expected complete 50k map cache; smoke subset cannot start training')
    for split,root in [('train',train_args.data_dir),('val',train_args.val_data_dir)]:
        sources={p.name for p in Path(root).glob('*.npz')}
        cached={p.name for p in (Path(args.map_cache_dir)/split).glob('*.npz')}
        if sources!=cached:raise ValueError(f'{split}: incomplete/mismatched map cache')
    run_dir=ROOT/'waymo/data_prep/runs'/args.run_name;run_dir.mkdir(parents=True,exist_ok=True)
    if not args.skip_preflight:preflight(train_args,run_dir/'preflight.json')
    base.resolve_kinematics_args(train_args)
    (run_dir/'resolved_train_args.json').write_text(json.dumps(vars(train_args),indent=2)+'\n')
    if not args.preflight_only:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable on training host')
        base.train(train_args)

if __name__=='__main__':main()
