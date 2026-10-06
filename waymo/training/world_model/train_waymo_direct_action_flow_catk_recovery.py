#!/usr/bin/env python3
"""Sampled closest-of-K closed-loop contexts + recovery flow matching."""
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
from waymo.training.world_model.catk_flow_recovery import backward_recovery
base=erd.base

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
    if initial['step']!=95000 or ta.condition_focus_actions or ta.horizon!=40 or ta.commitment!=5 or getattr(ta,'balance_dynamic_loss',False): raise ValueError('Expected original generate-all 95k H40/B5')
    rollout_args=argparse.Namespace(**vars(ta));rollout_args.horizon=80
    model=base.create_model(ta).to(device);model.load_state_dict(initial['ema_model']);model.eval()
    ema=base.ModelEMA(model,cfg.ema_decay);normalizer=base.build_normalizer(stats,device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=.01)
    step=0;best=float('inf');validation={}
    if cfg.resume:
        c=torch.load(cfg.resume,map_location='cpu',weights_only=False)
        for key in ('checkpoint','seed','lr','batch_size','num_candidates','solver_steps','ema_decay'):
            if c['recovery_args'][key]!=getattr(cfg,key): raise ValueError('Resume config mismatch: '+key)
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
            recovery_args=vars(cfg),rng=erd.rng_state(),stage='catk_inspired_recovery_flow'))
    def evaluate():
        state=erd.rng_state()
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16): r=erd.validate(ema.model,normalizer,vl,ta,device,cfg)
            if not all(np.isfinite(v) for v in r.values()): raise FloatingPointError('Nonfinite validation')
            return r
        finally: erd.restore_rng(state)
    if not validation and not cfg.skip_validation:
        validation=evaluate();best=validation['val_ade_m'];save('best_ade.pt')
        entry=dict(kind='validation',step=step,**validation)
        print('validation '+json.dumps(entry),flush=True)
        with (out/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(entry)+'\n')
    started=time.monotonic()
    print(f'start method=catk_inspired_recovery_flow step={step} target={cfg.max_steps} K={cfg.num_candidates}',flush=True)
    with (out/'metrics.jsonl').open('a') as log:
        for raw in loader:
            batch=base.move_batch(raw,device)
            prepared=base.prepare_batch(batch,normalizer,rollout_args,random_start=False)
            opt.zero_grad(set_to_none=True)
            row=backward_recovery(model,normalizer,prepared,batch,step,cfg,ta)
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
    p.add_argument('--resume');p.add_argument('--skip_validation',action='store_true')
    for key,value in dict(seed=20260911,batch_size=2,num_workers=4,num_candidates=8,solver_steps=8,max_steps=10000,max_iterations=0,
        eval_every=500,eval_batches=8,eval_batch_size=4,eval_rollouts=4,
        save_every=100,snapshot_every=1000,log_every=10).items():p.add_argument('--'+key,type=int,default=value)
    for key,value in dict(lr=1e-6,ema_decay=.999).items():p.add_argument('--'+key,type=float,default=value)
    p.add_argument('--cpd_scales',nargs=3,type=float,default=[25.423524547767016,4.925798640017075,18.77286026475977])
    c=p.parse_args()
    if min(c.batch_size,c.max_steps,c.solver_steps,c.num_candidates,c.eval_every,c.save_every,c.snapshot_every,c.log_every)<1:p.error('Positive sizes required')
    if c.num_candidates!=8:p.error('This experiment uses 8 sampled candidates')
    return c

if __name__=='__main__':train(parse_args())
