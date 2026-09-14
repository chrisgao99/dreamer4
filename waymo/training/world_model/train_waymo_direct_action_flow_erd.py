#!/usr/bin/env python3
"""Flow-ERD stage-two adaptation for the existing holonomic DirectActionFlow.

See ERD_STAGE2.md for the agreed deviations/underspecified paper settings.
Three committed B=5 chunks form a 15-step sample scored at its logged anchor.
No GT future actions condition the generator or score networks.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.checkpoint import checkpoint as activation_checkpoint

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'waymo/core'))
from waymo.training.world_model import train_waymo_direct_action_flow as base
from waymo.training.world_model.direct_action_flow import (
    agents_to_bntf, gather_agent_window, execute_holonomic_actions,
    inverse_holonomic_actions, rollout_receding_horizon,
)
from waymo.training.world_model.multisample_validation import flow_erd_cpd_metrics


def velocity_to_score(x, velocity, lam):
    """For x_l=(1-l)z+l*x_1, z~N(0,I): score=(l*v-x_l)/(1-l)."""
    return (lam * velocity.float() - x.float()) / (1 - lam)


def masked_mean(x, mask):
    return (x * mask[..., None]).sum() / (mask.sum().clamp_min(1) * x.shape[-1])


def dmd_surrogate(clean, score_difference, mask):
    # A first-order surrogate only: no derivatives through the score estimator.
    return masked_mean(clean.float() * score_difference.detach(), mask)


def generate_committed(model, normalizer, history, batch, anchors, solver_steps=8,
                       commitment=5, checkpoint_grad=True):
    """Differentiable full-H closed loop, preserving gradients across replans."""
    types = history[:, :, -1, 7].round().long().clamp_min(0)
    pieces, masks = [], []
    indices = torch.arange(history.shape[0], device=history.device)
    history_length = history.shape[2]
    # One checkpoint per entire plan: recompute encoder + all flow steps on backward.
    def plan(h, light, light_mask):
        scene = model.encode_scene(history=h, agent_mask=batch['agent_mask'],
            map_polylines=batch['map_polylines'], map_mask=batch['map_mask'],
            current_lights=light, current_light_mask=light_mask)
        mask = scene.agent_mask[:, :, None].expand(-1, -1, model.horizon)
        x = torch.randn((*mask.shape, 3), device=h.device, dtype=torch.float32)
        x = x * mask[..., None]
        for step in range(solver_steps):
            t = torch.full((h.shape[0],), step / solver_steps, device=h.device)
            v = model.decode_velocity(x, t, scene, mask)
            x = (x + v.float() / solver_steps) * mask[..., None]
        return x, mask
    for elapsed in range(0, model.horizon, commitment):
        light = batch['lights'][indices, anchors + elapsed]
        light_mask = batch['light_mask'][indices, anchors + elapsed]
        if torch.is_grad_enabled() and checkpoint_grad:
            actions, mask = activation_checkpoint(plan, history, light, light_mask,
                use_reentrant=False, preserve_rng_state=True)
        else:
            actions, mask = plan(history, light, light_mask)
        take = min(commitment, model.horizon - elapsed)
        actions, mask = actions[:, :, :take], mask[:, :, :take]
        pieces.append(actions)
        masks.append(mask)
        metric = normalizer.denormalize(actions, types)
        pose = torch.cat((history[:, :, -1, :2], history[:, :, -1, 6:7]), dim=-1)
        poses = execute_holonomic_actions(pose, metric, mask)
        # Match the existing rollout's generated-frame feature layout exactly.
        zeros = history.new_zeros((*poses.shape[:-1], 3))
        frames = torch.cat((poses[..., :2], zeros, mask[..., None].to(history.dtype),
            poses[..., 2:3], types[:, :, None, None].expand(-1,-1,take,1).to(history.dtype)), dim=-1)
        if history.shape[-1] > 8:
            frames = torch.cat((frames, history.new_zeros((*frames.shape[:-1], history.shape[-1]-8))), -1)
        history = torch.cat((history, frames), dim=2)[:, :, -history_length:]
    return torch.cat(pieces, 2), torch.cat(masks, 2)


class StepBatches(Sampler):
    """Stateless deterministic sampling; prefetch does not affect resume indices."""
    def __init__(self, size, batch_size, start, stop, seed):
        self.size, self.batch_size, self.start, self.stop, self.seed = size, batch_size, start, stop, seed
    def __iter__(self):
        for step in range(self.start, self.stop):
            g = torch.Generator().manual_seed(self.seed + step)
            yield torch.randint(self.size, (self.batch_size,), generator=g).tolist()
    def __len__(self):
        return self.stop-self.start


def phase_at(iteration, length, ncritic):
    q = iteration % (2 * length)
    return ('fake_only', False) if q < length else ('both', (q-length+1) % ncritic == 0)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model, normalizer, loader, ta, device, cfg):
    """Fixed H80/B5 closed-loop validation, not the training surrogate loss."""
    model.eval()
    totals = dict(ade=0., minade=0., cpd=0., cpd_valid=0., scenes=0.,
                  invalid_motion=0., motion_points=0.)
    for bi, raw in enumerate(loader):
        if bi >= cfg.eval_batches: break
        b = base.move_batch(raw, device)
        agents = agents_to_bntf(b['agents'], b['agent_mask'])
        anchor = ta.history_length-1
        anchors = torch.full((agents.shape[0],), anchor, device=device, dtype=torch.long)
        history, future = gather_agent_window(agents, anchors, history_length=ta.history_length, horizon=80)
        targets = inverse_holonomic_actions(history, future, b['agent_mask'],
            max_displacement_m=ta.physical_max_displacement_m,
            max_yaw_delta_rad=ta.physical_max_yaw_delta_rad)
        poses = []
        for k in range(cfg.eval_rollouts):
            gen = torch.Generator(device=device).manual_seed(12346 + bi*cfg.eval_rollouts+k)
            poses.append(rollout_receding_horizon(model, normalizer, initial_history=history,
                agent_mask=b['agent_mask'], map_polylines=b['map_polylines'], map_mask=b['map_mask'],
                current_light_sequence=b['lights'][:, anchor:anchor+80],
                current_light_mask_sequence=b['light_mask'][:, anchor:anchor+80],
                focus_action_sequence=None, focus_action_valid=None, rollout_steps=80,
                commitment=5, solver_steps=cfg.solver_steps, generator=gen))
        poses = torch.stack(poses, 1)
        valid = targets.valid
        distance = (poses[..., :2]-targets.future_pose[:, None, ..., :2]).norm(dim=-1)
        ade = (distance*valid[:, None]).sum((2,3))/valid.sum((1,2))[:,None].clamp_min(1)
        context = history[:, :, -1, :2][:,None,None].expand(-1,cfg.eval_rollouts,-1,-1,-1)
        xy = torch.cat((context, poses[..., :2].permute(0,1,3,2,4)),2)
        metadata = torch.cat((history[:,:,-1:].permute(0,2,1,3),future.permute(0,2,1,3)),1)
        _, components = flow_erd_cpd_metrics(xy, metadata, future_start=1,
            type_scales=cfg.cpd_scales, exclude_focus=False)
        cv = components['scene_valid']
        totals['ade'] += float(ade.mean(1).sum()); totals['minade'] += float(ade.min(1).values.sum())
        totals['cpd'] += float(components['scene_cpd'][cv].sum()); totals['cpd_valid'] += int(cv.sum())
        totals['scenes'] += ade.shape[0]
        displacement = (xy[:,:,1:]-xy[:,:,:-1]).norm(dim=-1).permute(0,1,3,2)
        totals['invalid_motion'] += float(((displacement>5)*valid[:,None]).sum())
        totals['motion_points'] += int(valid.sum())*cfg.eval_rollouts
    if not totals['scenes'] or not totals['cpd_valid']: raise RuntimeError('Empty validation')
    return dict(val_ade_m=totals['ade']/totals['scenes'], val_minade_m=totals['minade']/totals['scenes'],
        val_cpd=totals['cpd']/totals['cpd_valid'], val_displacement_gt5_fraction=totals['invalid_motion']/max(1,totals['motion_points']),
        val_scenes=totals['scenes'])


def train(cfg):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    base.seed_everything(cfg.seed)
    output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and (output/'latest.pt').exists(): raise FileExistsError('Use --resume for existing run')
    initial = torch.load(cfg.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    ta = argparse.Namespace(**initial['args'])
    if initial['step'] != 75000 or ta.condition_focus_actions or ta.horizon != 15 or ta.commitment != 5:
        raise ValueError('Expected generate-all 75k H15/B5 checkpoint')
    stats = initial['action_stats']
    teacher = base.create_model(ta).to(device)
    teacher.load_state_dict(initial['ema_model']); teacher.requires_grad_(False).eval()
    generator, fake = copy.deepcopy(teacher), copy.deepcopy(teacher)
    generator.requires_grad_(True); fake.requires_grad_(True)
    # Dropout disabled for all distributions; gradients remain enabled on trainable weights.
    generator.eval(); fake.eval()
    normalizer = base.build_normalizer(stats, device)
    ema = base.ModelEMA(generator, cfg.ema_decay)
    go = torch.optim.AdamW(generator.parameters(), lr=cfg.generator_lr, weight_decay=.01)
    fo = torch.optim.AdamW(fake.parameters(), lr=cfg.fake_lr, weight_decay=.01)
    iteration, updates, best, validation = 0, 0, float('inf'), {}
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        for key in ('beta','seed','phase_length','ncritic','batch_size','solver_steps','time_min','time_max','generator_lr','fake_lr','ema_decay'):
            if saved['erd_args'][key] != getattr(cfg,key): raise ValueError(f'Resume setting mismatch: {key}')
        if saved['erd_args']['checkpoint'] != cfg.checkpoint: raise ValueError('Teacher checkpoint mismatch')
        generator.load_state_dict(saved['model']); fake.load_state_dict(saved['fake_model'])
        teacher.load_state_dict(saved['teacher_model']); ema.model.load_state_dict(saved['ema_model'])
        go.load_state_dict(saved['generator_optimizer']); fo.load_state_dict(saved['fake_optimizer'])
        iteration, updates, best = saved['iteration'], saved['step'], saved['best_val']
        validation = saved.get('validation', {})
        restore_rng(saved['rng']); del saved
    del initial
    dataset = base.WaymoVectorDataset(cfg.train_data)
    validation_set = base.WaymoVectorDataset(cfg.val_data)
    # Stop by generator updates; bound sampler by full phase cycles.
    stop = math.ceil(cfg.max_generator_steps/(cfg.phase_length//cfg.ncritic))*2*cfg.phase_length
    if cfg.max_iterations: stop = min(stop, iteration+cfg.max_iterations)
    # Separate DataLoader RNG prevents iterator construction changing training RNG on resume.
    loader = DataLoader(dataset, batch_sampler=StepBatches(len(dataset),cfg.batch_size,iteration,stop,cfg.seed+100000),
        num_workers=cfg.num_workers, collate_fn=base.collate_vector_batch, pin_memory=True,
        generator=torch.Generator().manual_seed(cfg.seed), persistent_workers=cfg.num_workers>0)
    vl = base.make_loader(validation_set,batch_size=cfg.eval_batch_size,shuffle=False,num_workers=cfg.num_workers,device=device)
    interrupted = [False]
    def interrupt(signum, frame): interrupted[0]=True
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    (output/'config.json').write_text(json.dumps(vars(cfg),indent=2)+'\n')
    started=time.monotonic()
    def save(name):
        atomic_save(output/name, dict(model=generator.state_dict(),ema_model=ema.model.state_dict(),
            fake_model=fake.state_dict(),teacher_model=teacher.state_dict(),generator_optimizer=go.state_dict(),
            fake_optimizer=fo.state_dict(),args=vars(ta),action_stats=stats,step=updates,iteration=iteration,
            epoch=0,best_val=best,erd_args=vars(cfg),rng=rng_state(),validation=validation,stage='erd'))
    def do_validation():
        state=rng_state()
        try:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=validate(ema.model,normalizer,vl,ta,device,cfg)
            if not all(np.isfinite(v) for v in result.values()): raise FloatingPointError('Nonfinite validation')
            return result
        finally: restore_rng(state)
    if not cfg.resume and not cfg.skip_validation:
        validation=do_validation(); best=validation['val_ade_m']; save('best_ade.pt')
        print('validation '+json.dumps(dict(step=updates,**validation)),flush=True)
    print(f'start beta={cfg.beta} iteration={iteration} generator_step={updates} target={cfg.max_generator_steps} batch={cfg.batch_size}',flush=True)
    with (output/'metrics.jsonl').open('a') as metrics:
        for raw in loader:
            phase, update_generator = phase_at(iteration,cfg.phase_length,cfg.ncritic)
            b=base.move_batch(raw,device)
            prepared=base.prepare_batch(b,normalizer,ta,random_start=True)
            kwargs=base.scene_kwargs(b,prepared)
            go.zero_grad(set_to_none=True); fo.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(update_generator), torch.autocast('cuda',dtype=torch.bfloat16):
                clean, mask=generate_committed(generator,normalizer,prepared.history,b,prepared.anchors,
                    cfg.solver_steps,ta.commitment)
            if not torch.isfinite(clean).all(): raise FloatingPointError('Nonfinite generated actions')
            noise=torch.randn_like(clean)
            time_t=torch.rand(clean.shape[0],device=device)*(cfg.time_max-cfg.time_min)+cfg.time_min
            lam=time_t[:,None,None,None]
            noisy=((1-lam)*noise+lam*clean.detach())*mask[...,None]
            target=(clean.detach()-noise)*mask[...,None]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                fscene=fake.encode_scene(**kwargs)
                fv=fake.decode_velocity(noisy,time_t,fscene,mask)
                floss=masked_mean((fv.float()-target.float()).square(),mask)
            if not torch.isfinite(floss): raise FloatingPointError('Nonfinite fake loss')
            floss.backward()
            fnorm=torch.nn.utils.clip_grad_norm_(fake.parameters(),1.,error_if_nonfinite=True)
            fo.step(); fo.zero_grad(set_to_none=True)
            gnorm=0.; gl=0.; snorm=0.
            if update_generator:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    real_v=teacher.decode_velocity(noisy,time_t,teacher.encode_scene(**kwargs),mask)
                    fake_v=fake.decode_velocity(noisy,time_t,fake.encode_scene(**kwargs),mask)
                    difference=velocity_to_score(noisy,fake_v,lam)-cfg.beta*velocity_to_score(noisy,real_v,lam)
                if not torch.isfinite(difference).all(): raise FloatingPointError('Nonfinite DMD score')
                gloss=dmd_surrogate(clean,difference,mask)
                gloss.backward()
                gnorm=float(torch.nn.utils.clip_grad_norm_(generator.parameters(),1.,error_if_nonfinite=True))
                if not np.isfinite(gnorm) or gnorm==0: raise FloatingPointError('Invalid generator gradient')
                go.step(); ema.update(generator); updates+=1
                gl=float(gloss.detach()); snorm=float(masked_mean(difference.square(),mask).sqrt())
            iteration+=1
            row=dict(iteration=iteration,generator_step=updates,beta=cfg.beta,phase=phase,
                fake_loss=float(floss.detach()),fake_grad_norm=float(fnorm),generator_grad_norm=gnorm,
                generator_surrogate=gl,score_difference_rms=snorm,action_rms=float(masked_mean(clean.detach().square(),mask).sqrt()),
                elapsed_seconds=time.monotonic()-started)
            if iteration%cfg.log_every==0 or update_generator:
                metrics.write(json.dumps(row)+'\n'); metrics.flush()
                print('train '+json.dumps(row),flush=True)
            if update_generator and updates%cfg.eval_every==0 and not cfg.skip_validation:
                validation=do_validation(); row=dict(kind='validation',step=updates,**validation)
                metrics.write(json.dumps(row)+'\n'); metrics.flush(); print('validation '+json.dumps(row),flush=True)
                if validation['val_ade_m'] < best:
                    best=validation['val_ade_m']; save('best_ade.pt')
            if iteration%cfg.save_every_iterations==0 or (update_generator and updates%cfg.save_every==0):
                save('latest.pt')
            if update_generator and updates%cfg.snapshot_every==0:
                save(f'generator_step_{updates:07d}.pt')
            if interrupted[0] or updates>=cfg.max_generator_steps: break
        save('latest.pt')
        if updates>=cfg.max_generator_steps: save('final.pt')
    print(f'finished generator_step={updates} iteration={iteration} interrupted={interrupted[0]}',flush=True)
    if interrupted[0]: raise SystemExit(130)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','train_data','val_data','output_dir'): p.add_argument('--'+key,required=True)
    p.add_argument('--resume')
    p.add_argument('--beta',type=float,required=True)
    for key,default in dict(seed=20260911,batch_size=2,num_workers=4,solver_steps=8,phase_length=100,ncritic=5,
        max_generator_steps=10000,max_iterations=0,eval_every=250,eval_batches=8,eval_batch_size=4,
        eval_rollouts=4,save_every=100,save_every_iterations=500,snapshot_every=1000,log_every=10).items():
        p.add_argument('--'+key,type=int,default=default)
    for key,default in dict(generator_lr=1e-6,fake_lr=1e-5,time_min=.02,time_max=.90,ema_decay=.999).items():
        p.add_argument('--'+key,type=float,default=default)
    p.add_argument('--cpd_scales',type=float,nargs=3,default=[25.423524547767016,4.925798640017075,18.77286026475977])
    p.add_argument('--skip_validation',action='store_true')
    a=p.parse_args()
    if not 0<a.beta<=1 or not 0<a.time_min<a.time_max<1: p.error('Invalid beta/time range')
    if a.ncritic<1 or a.phase_length<a.ncritic or a.phase_length%a.ncritic: p.error('phase_length must be divisible by ncritic')
    return a

if __name__=='__main__': train(parse_args())
