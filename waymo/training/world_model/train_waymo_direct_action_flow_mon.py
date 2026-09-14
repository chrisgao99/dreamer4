#!/usr/bin/env python3
"""MoN-8 GT / MoN-8 GT + physical stage-two ablation, generate-all 75k EMA."""
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

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'waymo/core'))
from waymo.training.world_model import train_waymo_direct_action_flow_erd as erd
from waymo.training.world_model.direct_action_flow_mon_losses import motion_distances,physical_losses,reliable_road_mask
from waymo.training.world_model.direct_action_flow import execute_holonomic_actions,agents_to_bntf,gather_agent_window,inverse_holonomic_actions,rollout_receding_horizon
base=erd.base


def candidate(model,normalizer,prepared,batch,seed,solver_steps):
    # Local RNG makes no-grad selection and gradient replay identical, and keeps
    # training anchors/data identical between the two loss ablations.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        actions,mask=erd.generate_committed(model,normalizer,prepared.history,batch,prepared.anchors,solver_steps,5)
        metric=normalizer.denormalize(actions,prepared.targets.agent_type)
        return execute_holonomic_actions(prepared.targets.current_pose,metric,mask)


def backward_mon(model,normalizer,prepared,batch,step,cfg):
    road_support=None
    if cfg.physical:
        road_support=reliable_road_mask(prepared.targets.future_pose,prepared.targets.current_pose,
            prepared.targets.valid,prepared.targets.agent_type,batch['map_polylines'],batch['map_mask'])
    distances=[];seeds=[cfg.seed+10000000+step*cfg.num_candidates+k for k in range(cfg.num_candidates)]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for seed in seeds:
            poses=candidate(model,normalizer,prepared,batch,seed,cfg.solver_steps)
            d,_=motion_distances(poses,prepared.targets.future_pose,prepared.targets.current_pose,prepared.targets.valid)
            distances.append(d)
    distances=torch.stack(distances,1)
    if not torch.isfinite(distances).all(): raise FloatingPointError('Nonfinite selection loss')
    winners=distances.argmin(1)
    totals=dict(mon_loss=float(distances.min(1).values.mean()),candidate_gt_loss_mean=float(distances.mean()),
        collision=0.,offroad=0.,sideslip=0.,replay_max_error=0.)
    scale=min(1.,(step+1)/cfg.physical_warmup_steps) if cfg.physical_warmup_steps>0 else 1.
    for k,seed in enumerate(seeds):
        if not cfg.physical and not (winners==k).any(): continue
        with torch.autocast('cuda',dtype=torch.bfloat16):
            poses=candidate(model,normalizer,prepared,batch,seed,cfg.solver_steps)
            d,_=motion_distances(poses,prepared.targets.future_pose,prepared.targets.current_pose,prepared.targets.valid)
            difference=float((d.detach()-distances[:,k]).abs().max())
            totals['replay_max_error']=max(totals['replay_max_error'],difference)
            if not torch.allclose(d.detach(),distances[:,k],rtol=1e-5,atol=1e-5):
                raise RuntimeError('MoN selection/replay differ')
            loss=(d*(winners==k)).mean()
            if cfg.physical:
                losses,_=physical_losses(poses,prepared.targets.current_pose,prepared.targets.valid,
                    prepared.targets.agent_type,batch['map_polylines'],batch['map_mask'],road_support=road_support)
                for name in ('collision','offroad','sideslip'):
                    value=losses[name].mean()/cfg.num_candidates
                    loss=loss+scale*getattr(cfg,name+'_weight')*value
                    totals[name]+=float(value.detach())
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite MoN/physical loss')
        loss.backward()
    if road_support is not None:
        totals['road_reliable_valid_fraction']=float(road_support.sum()/prepared.targets.valid.sum().clamp_min(1))
    totals['physical_warmup_scale']=scale if cfg.physical else 0.
    return totals


@torch.no_grad()
def validate(model,normalizer,loader,ta,cfg):
    """Same ADE/minADE/CPD evaluator as ERD plus physical diagnostics."""
    device=torch.device('cuda')
    results=erd.validate(model,normalizer,loader,ta,device,cfg)
    acc={};count=0
    for bi,raw in enumerate(loader):
        if bi>=cfg.physical_eval_batches: break
        b=base.move_batch(raw,device);agents=agents_to_bntf(b['agents'],b['agent_mask'])
        anchors=torch.full((agents.shape[0],),10,device=device,dtype=torch.long)
        h,f=gather_agent_window(agents,anchors,history_length=11,horizon=80)
        t=inverse_holonomic_actions(h,f,b['agent_mask'],max_displacement_m=ta.physical_max_displacement_m,max_yaw_delta_rad=ta.physical_max_yaw_delta_rad)
        support=reliable_road_mask(t.future_pose,t.current_pose,t.valid,t.agent_type,b['map_polylines'],b['map_mask'])
        for k in range(cfg.eval_rollouts):
            poses=rollout_receding_horizon(model,normalizer,initial_history=h,agent_mask=b['agent_mask'],
                map_polylines=b['map_polylines'],map_mask=b['map_mask'],current_light_sequence=b['lights'][:,10:90],
                current_light_mask_sequence=b['light_mask'][:,10:90],focus_action_sequence=None,focus_action_valid=None,
                rollout_steps=80,commitment=5,solver_steps=cfg.solver_steps,
                generator=torch.Generator(device=device).manual_seed(12346+bi*cfg.eval_rollouts+k))
            _,m=physical_losses(poses,t.current_pose,t.valid,t.agent_type,b['map_polylines'],b['map_mask'],road_support=support)
            for key,v in m.items(): acc[key]=acc.get(key,0.)+float(v.sum())
            count+=poses.shape[0]
    results.update({'val_'+k:v/max(1,count) for k,v in acc.items()})
    results['physical_eval_scene_rollouts']=count
    return results


def train(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high');base.seed_everything(cfg.seed)
    device=torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    if not cfg.resume and (out/'latest.pt').exists(): raise FileExistsError('Use --resume')
    initial=torch.load(cfg.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    ta=argparse.Namespace(**initial['args']);stats=initial['action_stats']
    if initial['step']!=75000 or ta.condition_focus_actions or ta.horizon!=15 or ta.commitment!=5: raise ValueError('Expected generate-all 75k H15/B5')
    model=base.create_model(ta).to(device);model.load_state_dict(initial['ema_model']);model.eval()
    ema=base.ModelEMA(model,cfg.ema_decay);normalizer=base.build_normalizer(stats,device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.01)
    step=0;best=float('inf');validation={}
    if cfg.resume:
        c=torch.load(cfg.resume,map_location='cpu',weights_only=False)
        for key in ('checkpoint','physical','seed','lr','batch_size','num_candidates','solver_steps','collision_weight','offroad_weight','sideslip_weight','physical_warmup_steps','ema_decay'):
            if c['mon_args'][key]!=getattr(cfg,key): raise ValueError('Resume config mismatch: '+key)
        model.load_state_dict(c['model']);ema.model.load_state_dict(c['ema_model']);opt.load_state_dict(c['optimizer'])
        step=c['step'];best=c['best_val'];validation=c.get('validation',{});erd.restore_rng(c['rng']);del c
    del initial
    data=base.WaymoVectorDataset(cfg.train_data);val=base.WaymoVectorDataset(cfg.val_data)
    stop=min(cfg.max_steps,step+cfg.max_iterations) if cfg.max_iterations else cfg.max_steps
    loader=DataLoader(data,batch_sampler=erd.StepBatches(len(data),cfg.batch_size,step,stop,cfg.seed+100000),
        num_workers=cfg.num_workers,collate_fn=base.collate_vector_batch,pin_memory=True,
        persistent_workers=cfg.num_workers>0,generator=torch.Generator().manual_seed(cfg.seed))
    vl=base.make_loader(val,batch_size=cfg.eval_batch_size,shuffle=False,num_workers=cfg.num_workers,device=device)
    halted=[False]
    def stop_signal(sig,frame): halted[0]=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    (out/'config.json').write_text(json.dumps(vars(cfg),indent=2)+'\n')
    def save(name):
        erd.atomic_save(out/name,dict(model=model.state_dict(),ema_model=ema.model.state_dict(),optimizer=opt.state_dict(),
            args=vars(ta),action_stats=stats,step=step,epoch=0,best_val=best,validation=validation,
            mon_args=vars(cfg),rng=erd.rng_state(),stage='mon_physical' if cfg.physical else 'mon_gt'))
    def evaluate():
        state=erd.rng_state()
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16): r=validate(ema.model,normalizer,vl,ta,cfg)
            if not all(np.isfinite(v) for v in r.values()): raise FloatingPointError('Nonfinite validation')
            return r
        finally: erd.restore_rng(state)
    if not cfg.resume and not cfg.skip_validation:
        validation=evaluate();best=validation['val_ade_m'];save('best_ade.pt')
        print('validation '+json.dumps(dict(step=step,**validation)),flush=True)
    started=time.monotonic()
    print(f'start method={"mon_physical" if cfg.physical else "mon_gt"} step={step} target={cfg.max_steps} K={cfg.num_candidates}',flush=True)
    with (out/'metrics.jsonl').open('a') as log:
        for raw in loader:
            batch=base.move_batch(raw,device)
            prepared=base.prepare_batch(batch,normalizer,ta,random_start=True)
            opt.zero_grad(set_to_none=True)
            row=backward_mon(model,normalizer,prepared,batch,step,cfg)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            if not np.isfinite(norm) or norm==0: raise FloatingPointError('Invalid gradient')
            opt.step();ema.update(model);step+=1
            row.update(step=step,grad_norm=norm,elapsed_seconds=time.monotonic()-started,
                cuda_peak_memory_mb=torch.cuda.max_memory_allocated()/1024**2)
            if step%cfg.log_every==0 or step<=2:
                print('train '+json.dumps(row),flush=True);log.write(json.dumps(row)+'\n');log.flush()
            if step%cfg.eval_every==0 and not cfg.skip_validation:
                validation=evaluate();entry=dict(kind='validation',step=step,**validation)
                print('validation '+json.dumps(entry),flush=True);log.write(json.dumps(entry)+'\n');log.flush()
                if validation['val_ade_m']<best: best=validation['val_ade_m'];save('best_ade.pt')
            if step%cfg.save_every==0:save('latest.pt')
            if step%cfg.snapshot_every==0:save(f'step_{step:07d}.pt')
            if halted[0]: break
        save('latest.pt')
        if step>=cfg.max_steps: save('final.pt')
    print(f'finished step={step} interrupted={halted[0]}',flush=True)
    if halted[0]:raise SystemExit(130)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','train_data','val_data','output_dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--resume');p.add_argument('--physical',action='store_true');p.add_argument('--skip_validation',action='store_true')
    for key,value in dict(seed=20260911,batch_size=2,num_workers=4,num_candidates=8,solver_steps=8,max_steps=10000,max_iterations=0,
        eval_every=250,eval_batches=8,physical_eval_batches=2,eval_batch_size=4,eval_rollouts=4,
        save_every=100,snapshot_every=1000,log_every=10,physical_warmup_steps=500).items():p.add_argument('--'+key,type=int,default=value)
    for key,value in dict(lr=1e-6,ema_decay=.999,collision_weight=.1,offroad_weight=.1,sideslip_weight=.1).items():p.add_argument('--'+key,type=float,default=value)
    p.add_argument('--cpd_scales',nargs=3,type=float,default=[25.423524547767016,4.925798640017075,18.77286026475977])
    c=p.parse_args()
    if min(c.batch_size,c.max_steps,c.solver_steps,c.num_candidates,c.eval_every,c.save_every,c.snapshot_every,c.log_every)<1:p.error('Positive sizes required')
    if c.num_candidates!=8:p.error('This experiment is MoN-8')
    return c

if __name__=='__main__':train(parse_args())
