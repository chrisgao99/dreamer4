#!/usr/bin/env python3
"""Sequential, fail-fast data preparation and fresh stage-one training launcher."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from waymo.data_prep.backfill_waymo_agent_lengths import atomic_json


def ensure_gpu_free(index,expected_uuid):
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={int(line.split(',')[0]):line.split(',')[1].strip() for line in output.strip().splitlines()}
    if mapping.get(index)!=expected_uuid:raise RuntimeError('Physical GPU index/UUID mapping changed')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(line.split(',')[0].strip()==expected_uuid for line in apps.splitlines()):
        raise RuntimeError(f'CUDA {index} became occupied during preparation; refusing to interfere')


def prepare_training(config):
    import torch
    from waymo.training.world_model import train_waymo_direct_action_flow as base
    torch.set_num_threads(1)
    args=base.build_arg_parser().parse_args(config['train_argv'])
    if args.resume is not None or args.horizon!=40 or args.commitment!=5 or args.action_execution!='type_aware':
        raise ValueError('Expected fresh type-aware H40/B5 training')
    base.resolve_kinematics_args(args)
    stats=base.load_or_compute_action_statistics(args)
    return args,stats


def preflight(args,stats,run_dir):
    import numpy as np
    import torch
    from waymo.training.world_model import train_waymo_direct_action_flow as base
    from waymo.training.world_model.action_kinematics import execute_model_actions
    model=base.create_model(args).eval()
    normalizer=base.build_normalizer(stats,torch.device('cpu'))
    report=dict(action_kinematics=model.kinematics.to_dict(),splits={})
    for split,root in [('train',args.data_dir),('val',args.val_data_dir)]:
        dataset=base.WaymoVectorDataset(root)
        selected=np.linspace(0,len(dataset)-1,32,dtype=int)
        valid_points=0;slots=0;max_error=0.
        for start in range(0,len(selected),4):
            batch=base.collate_vector_batch([dataset[int(i)] for i in selected[start:start+4]])
            prepared=base.prepare_batch(batch,normalizer,args,random_start=(split=='train'))
            target=prepared.targets
            if not torch.isfinite(prepared.normalized_actions).all():raise FloatingPointError('Nonfinite action targets')
            pose=execute_model_actions(model,target.current_pose,target.actions,target.valid,
                agent_type=target.agent_type,agent_lengths=batch['agent_lengths'])
            err=(pose[...,:2]-target.future_pose[...,:2]).norm(dim=-1)
            if target.valid.any():max_error=max(max_error,float(err[target.valid].max()))
            valid_points+=int(target.valid.sum());slots+=target.valid.numel()
            if start==0:
                scene=model.encode_scene(**base.scene_kwargs(batch,prepared))
                loss,_=base.flow_matching_loss(model,scene,prepared.normalized_actions,target.valid,
                    condition_focus_actions=False,loss_type=args.flow_loss_type,huber_beta=args.flow_huber_beta)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite preflight model loss')
                loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Nonfinite preflight model gradients')
                model.zero_grad(set_to_none=True)
        if valid_points==0:raise ValueError('No valid preflight targets')
        report['splits'][split]=dict(sampled_files=len(selected),valid_target_points=valid_points,
            padded_target_slots=slots,valid_fraction_of_padded_slots=valid_points/slots,
            max_reexecution_error_m=max_error)
    if any(v['max_reexecution_error_m']>args.max_reexecution_error_m+1e-3 for v in report['splits'].values()):
        raise ValueError('Re-execution error exceeded configured bound')
    atomic_json(run_dir/'preflight.json',report)
    print(json.dumps(dict(event='preflight',**report)),flush=True)


def run(config_path):
    config_path=Path(config_path).resolve();run_dir=config_path.parent
    config=json.loads(config_path.read_text());started=time.time()
    status=dict(status='running',stage='starting',pid=os.getpid(),started_unix=started,
                cuda_index=config['cuda_index'],cuda_uuid=config['cuda_uuid'])
    def stage(name):
        status.update(stage=name,elapsed_seconds=round(time.time()-started,2))
        atomic_json(run_dir/'status.json',status)
        print(json.dumps(dict(event='stage',**status)),flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    try:
        checkpoint=Path(config['checkpoint_dir'])
        if checkpoint.exists() and any(checkpoint.iterdir()):raise FileExistsError('Fresh checkpoint directory not empty')
        from waymo.training.world_model.train_waymo_direct_action_flow import build_arg_parser
        train_args=build_arg_parser().parse_args(config['train_argv'])
        stage('supplement_measured_lengths')
        subprocess.run([sys.executable,'waymo/data_prep/supplement_waymo_agent_lengths.py',
            '--data_root',config['data_root'],'--audit_path',config['length_audit'],
            '--workers','4','--history_length',str(train_args.history_length)],cwd=ROOT,env=env,check=True)
        stage('calibrate_no_slip_ratios')
        calibration=Path(config['calibration_path'])
        if not calibration.exists():
            subprocess.run([sys.executable,'waymo/training/world_model/prepare_action_kinematics.py',
                '--train_data',str(Path(config['data_root'])/'train'),'--output',str(calibration),
                '--max_files','8192','--workers','4'],cwd=ROOT,env=env,check=True)
        else:
            payload=json.loads(calibration.read_text())
            if payload['source_train_data']!=str(Path(config['data_root'])/'train') or payload['num_files']!=8192:
                raise ValueError('Existing calibration provenance mismatch')
        stage('compute_type_aware_action_statistics')
        args,stats=prepare_training(config)
        atomic_json(run_dir/'resolved_train_args.json',vars(args))
        stage('preflight')
        preflight(args,stats,run_dir)
        ensure_gpu_free(config['cuda_index'],config['cuda_uuid'])
        stage('training')
        env.update(CUDA_VISIBLE_DEVICES=config['cuda_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='offline',
                   WANDB_DIR=str(ROOT/'waymo/wandb'),PYTHONUNBUFFERED='1')
        (ROOT/'waymo/wandb').mkdir(exist_ok=True)
        subprocess.run([sys.executable,'waymo/training/world_model/train_waymo_direct_action_flow.py',
                        *config['train_argv']],cwd=ROOT,env=env,check=True)
        status['status']='complete';stage('complete')
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));stage(status['stage']);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
