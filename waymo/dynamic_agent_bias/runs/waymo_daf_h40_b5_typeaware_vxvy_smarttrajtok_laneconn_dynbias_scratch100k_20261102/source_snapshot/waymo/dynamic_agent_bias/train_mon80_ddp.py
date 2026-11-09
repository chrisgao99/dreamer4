#!/usr/bin/env python3
"""Pure MoN-8 H40/execute5 full80 finetune (train_waymo_direct_action_flow_mon80.py) on several GPUs.

Identical objective, data order, candidate seeds and update: every rank loads the same batch;
the K candidates are scored one at a time, as before, round-robin (candidate k on rank k % world).
Distances are summed across ranks; each winning candidate is replayed with gradients on the rank
that scored it (same batch-2 computation, replay must match to 1e-5 as in the original), without
activation checkpointing unless it runs out of memory (identical gradients). Gradients are
summed, so one update equals the single-GPU update. Rank 0 validates and saves.
Batching several candidates into one scoring rollout is NOT equivalent on a trained model: the
80-step closed loop amplifies bf16 rounding of different kernel shapes to ~2% score differences.
Launch with torchrun.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'waymo/core'))
from waymo.dynamic_agent_bias.common import broadcast_parameters,max_parameter_divergence,sum_gradients
from waymo.training.world_model import train_waymo_direct_action_flow_erd as erd
from waymo.training.world_model.direct_action_flow_mon_losses import motion_distances
from waymo.training.world_model.mon_long_rollout import generate_poses
from waymo.training.world_model.train_waymo_direct_action_flow_mon80 import candidate,validate
base=erd.base


def backward_mon(model,prepared,batch,step,cfg,normalizer,rank,world):
    seeds=[cfg.seed+10000000+step*cfg.num_candidates+k for k in range(cfg.num_candidates)]
    mine=[k for k in range(cfg.num_candidates) if k%world==rank]
    distances=torch.zeros((prepared.history.shape[0],cfg.num_candidates),device=prepared.history.device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for k in mine:
            poses=candidate(model,normalizer,prepared,batch,seeds[k],cfg.solver_steps,cfg.detach_every)
            d,_=motion_distances(poses,prepared.targets.future_pose,prepared.targets.current_pose,prepared.targets.valid)
            distances[:,k]=d.float()
    local=distances.clone()
    dist.all_reduce(distances)  # each column is written by exactly one rank
    if not torch.isfinite(distances).all(): raise FloatingPointError('Nonfinite selection loss')
    winners=distances.argmin(1)
    totals=dict(mon_loss=float(distances.min(1).values.mean()),candidate_gt_loss_mean=float(distances.mean()),
        collision=0.,offroad=0.,sideslip=0.,replay_max_error=0.,replay_checkpoint_fallbacks=0.)
    params=[p for p in model.parameters() if p.requires_grad]
    for k in mine:
        if not (winners==k).any(): continue
        # Replayed on the rank that scored it: the same batch-2 computation, as in the original.
        for checkpoint_grad in (False,True):
            try:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seeds[k])
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        poses=generate_poses(model,normalizer,prepared.history,batch,prepared.anchors,rollout_steps=80,
                            commitment=5,detach_every=cfg.detach_every,solver_steps=cfg.solver_steps,checkpoint_grad=checkpoint_grad)
                        d,_=motion_distances(poses,prepared.targets.future_pose,prepared.targets.current_pose,prepared.targets.valid)
                        totals['replay_max_error']=max(totals['replay_max_error'],float((d.detach().float()-local[:,k]).abs().max()))
                        if not torch.allclose(d.detach().float(),local[:,k],rtol=1e-5,atol=1e-5):
                            raise RuntimeError('MoN selection/replay differ')
                        loss=(d*(winners==k)).mean()
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite MoN loss')
                    grads=torch.autograd.grad(loss,params,allow_unused=True)
                break
            except torch.cuda.OutOfMemoryError:
                # Activation checkpointing gives identical gradients with far less memory.
                if checkpoint_grad: raise
                poses=d=loss=None;torch.cuda.empty_cache();totals['replay_checkpoint_fallbacks']+=1
        for p,g in zip(params,grads):
            if g is not None: p.grad=g if p.grad is None else p.grad+g
    stats=torch.tensor([totals['replay_max_error'],totals['replay_checkpoint_fallbacks']],device=distances.device)
    dist.all_reduce(stats,op=dist.ReduceOp.MAX)
    totals['replay_max_error'],totals['replay_checkpoint_fallbacks']=float(stats[0]),float(stats[1])
    totals['physical_warmup_scale']=0.
    return totals


def train(cfg):
    dist.init_process_group('nccl',timeout=timedelta(hours=3))
    rank,world=dist.get_rank(),dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high');base.seed_everything(cfg.seed)
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    log=(lambda *a:print(*a,flush=True)) if rank==0 else (lambda *a:None)
    out=Path(cfg.output_dir)
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    if not cfg.resume and (out/'latest.pt').exists(): raise FileExistsError('Use --resume')
    initial=torch.load(cfg.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    ta=argparse.Namespace(**initial['args']);stats=initial['action_stats']
    if ta.condition_focus_actions or ta.horizon!=40 or ta.commitment!=5 or getattr(ta,'balance_dynamic_loss',False): raise ValueError('Expected generate-all H40/B5')
    rollout_args=argparse.Namespace(**vars(ta));rollout_args.horizon=80
    model=base.create_model(ta).to(device);model.load_state_dict(initial['ema_model']);model.eval()
    normalizer=base.build_normalizer(stats,device);model.set_action_normalizer(normalizer)
    ema=base.ModelEMA(model,cfg.ema_decay)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.01)
    step=0;best=float('inf');validation={}
    if cfg.resume:
        c=torch.load(cfg.resume,map_location='cpu',weights_only=False)
        for key in ('detach_every','checkpoint','physical','seed','lr','batch_size','num_candidates','solver_steps','ema_decay'):
            if c['mon_args'][key]!=getattr(cfg,key): raise ValueError('Resume config mismatch: '+key)
        model.load_state_dict(c['model']);ema.model.load_state_dict(c['ema_model']);opt.load_state_dict(c['optimizer'])
        step=c['step'];best=c['best_val'];validation=c.get('validation',{})
        if rank==0:erd.restore_rng(c['rng'])
        del c
    del initial
    broadcast_parameters(model)
    from waymo.lane_connectivity.dataset import ConnectivityDataset
    if not getattr(ta,'lane_connectivity',False): raise ValueError('Expected the lane connectivity model')
    ta.map_cache_dir=cfg.map_cache_dir or ta.map_cache_dir
    ta.lane_topology_dir=cfg.lane_topology_dir or ta.lane_topology_dir
    make_dataset=lambda root:ConnectivityDataset(root,ta.map_cache_dir,ta.lane_topology_dir)
    data=make_dataset(cfg.train_data)
    stop=min(cfg.max_steps,step+cfg.max_iterations) if cfg.max_iterations else cfg.max_steps
    loader=DataLoader(data,batch_sampler=erd.StepBatches(len(data),cfg.batch_size,step,stop,cfg.seed+100000),
        num_workers=cfg.num_workers,collate_fn=base.collate_vector_batch,pin_memory=True,
        persistent_workers=cfg.num_workers>0,generator=torch.Generator().manual_seed(cfg.seed))
    vl=base.make_loader(make_dataset(cfg.val_data),batch_size=cfg.eval_batch_size,shuffle=False,
        num_workers=cfg.num_workers,device=device) if rank==0 else None
    halted=[False]
    def stop_signal(sig,frame): halted[0]=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    if rank==0:(out/'config.json').write_text(json.dumps({**vars(cfg),'ddp_world_size':world},indent=2)+'\n')
    def save(name):
        erd.atomic_save(out/name,dict(model=model.state_dict(),ema_model=ema.model.state_dict(),optimizer=opt.state_dict(),
            args=vars(ta),action_stats=stats,step=step,epoch=0,best_val=best,validation=validation,
            mon_args=vars(cfg),rng=erd.rng_state(),stage='mon_gt'))
    def evaluate():
        state=erd.rng_state()
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16): r=validate(ema.model,normalizer,vl,ta,cfg)
            if not all(np.isfinite(v) for v in r.values()): raise FloatingPointError('Nonfinite validation')
            return r
        finally: erd.restore_rng(state)
    def sync_best():
        nonlocal best
        value=torch.tensor([best],device=device,dtype=torch.float64);dist.broadcast(value,0);best=float(value)
        dist.barrier()
    if not validation and not cfg.skip_validation:
        if rank==0:
            validation=evaluate();best=validation['val_ade_m'];save('best_ade.pt')
            entry=dict(kind='validation',step=step,**validation)
            log('validation '+json.dumps(entry))
            with (out/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(entry)+'\n')
        sync_best()
    started=time.monotonic()
    log(f'start method=mon_gt step={step} target={cfg.max_steps} K={cfg.num_candidates} world={world}')
    metrics=(out/'metrics.jsonl').open('a') if rank==0 else None
    for raw in loader:
        batch=base.move_batch(raw,device)
        prepared=base.prepare_batch(batch,normalizer,rollout_args,random_start=False)
        opt.zero_grad(set_to_none=True)
        row=backward_mon(model,prepared,batch,step,cfg,normalizer,rank,world)
        sum_gradients(model.parameters())
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
        if not np.isfinite(norm) or norm==0: raise FloatingPointError('Invalid gradient')
        opt.step();ema.update(model);step+=1
        row.update(step=step,grad_norm=norm,elapsed_seconds=time.monotonic()-started,
            cuda_peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2)
        if rank==0 and (step%cfg.log_every==0 or step<=2):
            log('train '+json.dumps(row));metrics.write(json.dumps(row)+'\n');metrics.flush()
        if step%100==0:
            divergence=max_parameter_divergence(model)
            if divergence>0:
                log(f'parameter divergence {divergence:.3g} at step {step}; resynchronizing from rank 0')
                broadcast_parameters(model)
        boundary=(step%cfg.eval_every==0 and not cfg.skip_validation) or step%cfg.save_every==0 or step%cfg.snapshot_every==0
        if boundary:
            if rank==0:
                if step%cfg.eval_every==0 and not cfg.skip_validation:
                    validation=evaluate();entry=dict(kind='validation',step=step,**validation)
                    log('validation '+json.dumps(entry));metrics.write(json.dumps(entry)+'\n');metrics.flush()
                    if validation['val_ade_m']<best: best=validation['val_ade_m'];save('best_ade.pt')
                if step%cfg.save_every==0:save('latest.pt')
                if step%cfg.snapshot_every==0:save(f'step_{step:07d}.pt')
            sync_best()
        stop_now=torch.tensor([float(halted[0])],device=device);dist.all_reduce(stop_now,op=dist.ReduceOp.MAX)
        if stop_now.item(): halted[0]=True;break
    if rank==0:
        save('latest.pt')
        if step>=cfg.max_steps: save('final.pt')
        metrics.close()
    log(f'finished step={step} interrupted={halted[0]}')
    dist.barrier();dist.destroy_process_group()
    if halted[0]:raise SystemExit(130)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','train_data','val_data','output_dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--map_cache_dir');p.add_argument('--lane_topology_dir')
    p.add_argument('--detach_every',type=int,choices=(0,5,20),required=True)
    p.add_argument('--resume');p.add_argument('--skip_validation',action='store_true')
    for key,value in dict(seed=20260911,batch_size=2,num_workers=4,num_candidates=8,solver_steps=8,max_steps=10000,max_iterations=0,
        eval_every=500,eval_batches=8,physical_eval_batches=0,eval_batch_size=4,eval_rollouts=4,
        save_every=100,snapshot_every=1000,log_every=10,physical_warmup_steps=500).items():p.add_argument('--'+key,type=int,default=value)
    for key,value in dict(lr=1e-6,ema_decay=.999,collision_weight=.1,offroad_weight=.1,sideslip_weight=.1).items():p.add_argument('--'+key,type=float,default=value)
    p.add_argument('--cpd_scales',nargs=3,type=float,default=[25.423524547767016,4.925798640017075,18.77286026475977])
    c=p.parse_args();c.physical=False
    if min(c.batch_size,c.max_steps,c.solver_steps,c.num_candidates,c.eval_every,c.save_every,c.snapshot_every,c.log_every)<1:p.error('Positive sizes required')
    if c.num_candidates!=8:p.error('This experiment is MoN-8')
    return c

if __name__=='__main__':train(parse_args())
