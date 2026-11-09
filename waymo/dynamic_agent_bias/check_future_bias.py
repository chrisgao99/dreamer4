"""Correctness probes for FutureAgentBias on the dynbias stage-one checkpoint (first replan only).

A. Kinematics: planned_chunk_poses (holonomic) vs execute_model_actions (type_aware) on the same
   actions -- the final sample and the logged GT actions -- at chunk ends.
B. Plan quality: self-condition estimate at solver step k vs the final sample (executed poses).
C. Magnitude: spread over keys of the future bias vs the current-pose relative bias.
D. Ablation: same noise with the future bias forced off -> change in executed plan.
"""
import sys, argparse, json
from pathlib import Path
import numpy as np, torch
ROOT = Path('/p/yufeng/tri30/dreamer4'); sys.path.insert(0, str(ROOT))
from waymo.evaluation.eval_waymo_direct_action_flow_multisample import load_model_and_normalizer
from waymo.training.world_model.direct_action_flow import agents_to_bntf, gather_agent_window
from waymo.training.world_model.train_waymo_direct_action_flow import make_loader, move_batch
from waymo.training.world_model.action_kinematics import inverse_model_actions, execute_model_actions
from waymo.lane_connectivity.dataset import ConnectivityDataset

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', default=str(ROOT / 'waymo/checkpoints/waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_dynbias_scratch100k_20261102/step_000090000.pt'))
p.add_argument('--batches', type=int, default=16)
p.add_argument('--samples', type=int, default=4)
p.add_argument('--out', required=True)
a = p.parse_args()
dev = torch.device('cuda')
model, norm, targs, step = load_model_and_normalizer(Path(a.checkpoint), dev, 'ema')
ds = ConnectivityDataset(str(ROOT / 'data/waymo_vector_dataset_ooi_centered_50k_with_lengths/val'),
                         str(ROOT / 'data/waymo_map_smarttrajtok_5m_v2'), str(ROOT / 'data/waymo_lane_connectivity_5m_v1'))
loader = make_loader(ds, batch_size=4, shuffle=False, num_workers=2, device=dev)
H, C, cs = model.horizon, model.num_chunks, model.chunk_size
ends = torch.arange(cs - 1, H, cs, device=dev)

capture = {}
orig_decode = model.decode_velocity
force_off = {'on': False}
def decode_hook(x, t, scene, mask, **kw):
    if force_off['on']:
        kw.pop('self_condition', None)
    sc = kw.get('self_condition')
    capture.setdefault('est', []).append(None if sc is None else sc.detach().clone())
    out = orig_decode(x, t, scene, mask, **kw)
    return out
model.decode_velocity = decode_hook
orig_fab = model.future_agent_bias.forward
def fab_hook(chunk_pose, agent_type, flow_time, available):
    b = orig_fab(chunk_pose, agent_type, flow_time, available)
    capture['last_bias'] = b.detach().float(); capture['last_avail'] = available.detach().clone()
    return b
model.future_agent_bias.forward = fab_hook

def disp(x, y, m):  # mean xy error over mask
    e = (x[..., :2] - y[..., :2]).norm(dim=-1)
    return e, m

acc = {k: [] for k in ['kin_sample', 'kin_gt', 'gt_exec', 'est_vs_final', 'bias_spread', 'rel_spread', 'abl_first5', 'abl_h40']}
type_tag = []
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for bi, raw in enumerate(loader):
        if bi >= a.batches: break
        batch = move_batch(raw, dev)
        agents = agents_to_bntf(batch['agents'], batch['agent_mask'])
        B = agents.shape[0]; anchor = int(targs.history_length) - 1
        hist, fut = gather_agent_window(agents, torch.full((B,), anchor, dtype=torch.long, device=dev),
                                        history_length=int(targs.history_length), horizon=H)
        kw = dict(history=hist, agent_mask=batch['agent_mask'], map_polylines=batch['map_polylines'], map_mask=batch['map_mask'],
                  current_lights=batch['lights'][:, anchor], current_light_mask=batch['light_mask'][:, anchor],
                  map_ids=batch['map_ids'], map_is_lane=batch['map_is_lane'], map_stop_sign=batch['map_stop_sign'],
                  map_stop_point=batch['map_stop_point'], current_light_ids=batch['light_ids'][:, anchor],
                  lane_edges=batch['lane_edges'], lane_edge_types=batch['lane_edge_types'])
        scene = model.encode_scene(**kw)
        types = hist[:, :, -1, 7].round().long(); lengths = batch.get('agent_lengths')
        amask = scene.agent_mask[:, :, None].expand(-1, -1, H).clone()
        pose0 = scene.agent_pose.float()
        # ---- GT actions: holonomic integration vs type-aware execution vs logged poses
        tg = inverse_model_actions(model, hist, fut, batch['agent_mask'], agent_lengths=lengths,
                                   max_displacement_m=float(targs.physical_max_displacement_m),
                                   max_yaw_delta_rad=float(targs.physical_max_yaw_delta_rad))
        full_valid = tg.valid.all(-1) & scene.agent_mask.bool()  # agents valid over whole H
        gt_norm = norm.normalize(tg.actions.float(), types)
        hol = model.planned_chunk_poses(gt_norm, scene, tg.valid)
        exe = execute_model_actions(model, pose0, tg.actions.float(), tg.valid, agent_type=types, agent_lengths=lengths)[:, :, ends]
        logged = tg.future_pose.float()[:, :, ends]
        acc['kin_gt'].append(((hol[..., :2] - exe[..., :2]).norm(dim=-1)[full_valid]).cpu())
        if logged is not None:
            acc['gt_exec'].append(((exe[..., :2] - logged[..., :2]).norm(dim=-1)[full_valid]).cpu())
        type_tag.append(types[full_valid].cpu())
        # relative-bias spread over keys (masked)
        km = scene.agent_mask.bool()
        def spread(bias):  # bias (..., heads, N, N); std over valid keys per query, mean over valid queries/heads
            m = km[:, None, None, :].expand_as(bias) if bias.dim() == 4 else km[:, None, None, None, :].expand_as(bias)
            x = torch.where(m, bias, torch.nan)
            return np.nanmean(torch.nanmean((x - torch.nanmean(x, -1, keepdim=True)) ** 2, -1).sqrt().cpu().numpy())
        acc['rel_spread'].append(spread(scene.relative_bias.float()))
        for k in range(a.samples):
            g = torch.Generator(device=dev).manual_seed(1000 * bi + k)
            noise = torch.randn((*amask.shape, 3), device=dev, generator=g)
            capture.clear(); force_off['on'] = False
            x = model.sample_normalized_actions(scene, amask, None, solver_steps=8, initial_noise=noise).float()
            ests = capture['est']; acc['bias_spread'].append(spread(capture['last_bias']))
            metric = norm.denormalize(x, types)
            exe_s = execute_model_actions(model, pose0, metric, amask, agent_type=types, agent_lengths=lengths)
            hol_s = model.planned_chunk_poses(x, scene, amask)
            m = scene.agent_mask.bool()
            acc['kin_sample'].append(((hol_s[..., :2] - exe_s[:, :, ends, :2]).norm(dim=-1)[m]).cpu())
            row = []
            for e in ests[1:]:
                pe = model.planned_chunk_poses(e, scene, amask)
                row.append((pe[..., :2] - exe_s[:, :, ends, :2]).norm(dim=-1)[m].mean().item())
            acc['est_vs_final'].append(row)
            capture.clear(); force_off['on'] = True
            x0 = model.sample_normalized_actions(scene, amask, None, solver_steps=8, initial_noise=noise).float()
            exe_0 = execute_model_actions(model, pose0, norm.denormalize(x0, types), amask, agent_type=types, agent_lengths=lengths)
            d = (exe_0[..., :2] - exe_s[..., :2]).norm(dim=-1)
            acc['abl_first5'].append(d[:, :, :5][m].mean().item()); acc['abl_h40'].append(d[:, :, -1][m].mean().item())
            moving = (exe_s[:, :, -1, :2] - pose0[:, :, :2]).norm(dim=-1) > 2.0
            cp = exe_s[:, :, ends, :2]
            dist = (cp[:, :, None] - cp[:, None]).norm(dim=-1).amin(-1)  # (B,N,N) min over chunks
            eye = torch.eye(dist.shape[1], dtype=torch.bool, device=dev)[None]
            vm = m[:, :, None] & m[:, None, :] & ~eye
            near = torch.where(vm, dist, torch.inf).amin(-1) < 5.0
            acc.setdefault('abl5_moving', []).append(d[:, :, :5].mean(-1)[m & moving].cpu())
            acc.setdefault('abl40_moving', []).append(d[:, :, -1][m & moving].cpu())
            acc.setdefault('abl5_near', []).append(d[:, :, :5].mean(-1)[m & moving & near].cpu())
            acc.setdefault('abl40_near', []).append(d[:, :, -1][m & moving & near].cpu())
        print('batch', bi, flush=True)

types = torch.cat(type_tag).numpy()
kg = torch.cat(acc['kin_gt']).numpy()  # (n_agents, C)
res = {'checkpoint_step': step, 'scenes': a.batches * 4, 'samples_per_scene': a.samples}
names = {1: 'vehicle', 2: 'pedestrian', 3: 'cyclist'}
res['A_gt_holonomic_vs_typeaware_m_by_chunk'] = {names[t]: np.round(kg[types == t].mean(0), 3).tolist() for t in (1, 2, 3) if (types == t).any()}
res['A_gt_holonomic_vs_typeaware_m_chunk8_p90'] = {names[t]: float(np.percentile(kg[types == t][:, -1], 90)) for t in (1, 2, 3) if (types == t).any()}
if acc['gt_exec']:
    ge = torch.cat(acc['gt_exec']).numpy(); res['A_gt_typeaware_exec_vs_logged_m_by_chunk'] = np.round(ge.mean(0), 3).tolist()
ks = torch.cat(acc['kin_sample']).numpy(); res['A_sample_holonomic_vs_typeaware_m_by_chunk'] = np.round(ks.mean(0), 3).tolist()
res['B_estimate_step_k_vs_final_exec_m'] = np.round(np.mean(acc['est_vs_final'], 0), 3).tolist()
res['C_future_bias_std_over_keys'] = float(np.mean(acc['bias_spread'])); res['C_relative_bias_std_over_keys'] = float(np.mean(acc['rel_spread']))
res['D_ablation_exec_diff_first5_m'] = float(np.mean(acc['abl_first5'])); res['D_ablation_exec_diff_step40_m'] = float(np.mean(acc['abl_h40']))
for k in ['abl5_moving', 'abl40_moving', 'abl5_near', 'abl40_near']:
    v = torch.cat(acc[k]).numpy()
    res['D_' + k] = {'n': int(v.size), 'mean': float(v.mean()), 'p90': float(np.percentile(v, 90)),
                     'p99': float(np.percentile(v, 99)), 'max': float(v.max())}
print(json.dumps({k: v for k, v in res.items() if k.startswith('D_')}, indent=1)); Path(a.out).write_text(json.dumps(res, indent=1))
