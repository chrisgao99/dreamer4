"""CAT-K-inspired sampled joint selection and detached-context flow supervision.

Samples are not likelihood top-K. Selection constructs training states only;
recovery flow targets, not the selected sample, provide the training labels.
"""
import torch
from waymo.training.world_model.action_kinematics import (
    KinematicsConfig, inverse_actions, execute_model_actions, model_config,
)
from waymo.training.world_model.direct_action_flow import (
    inverse_holonomic_actions, execute_holonomic_actions, flow_matching_loss,
)
from waymo.training.world_model.direct_action_flow_mon_losses import motion_distances


def recovery_targets(history, full_gt, full_valid, agent_mask, normalizer,
                     elapsed, horizon=40, max_displacement=5., max_yaw=.75,
                     normalized_limit=5., agent_lengths=None, config=KinematicsConfig()):
    """First action corrects from generated pose; subsequent actions follow GT.

    Implausible corrections and out-of-range normalized targets are excluded,
    never clipped. Masks are contiguous prefixes and cannot resurrect GT tracks.
    """
    history = history.detach()
    available = min(horizon, full_gt.shape[2] - elapsed)
    if available <= 0:
        raise ValueError('No future GT available')
    future = history.new_zeros((*history.shape[:2], horizon, history.shape[-1]))
    future[:, :, :available, :2] = full_gt[:, :, elapsed:elapsed+available, :2]
    future[:, :, :available, 6] = full_gt[:, :, elapsed:elapsed+available, 2]
    future[:, :, :available, 5] = full_valid[:, :, elapsed:elapsed+available]
    future[..., 7] = history[:, :, -1:, 7]
    target = inverse_actions(history, future, agent_mask, config=config,
        agent_lengths=agent_lengths,
        max_displacement_m=max_displacement, max_yaw_delta_rad=max_yaw)
    normalized = normalizer.normalize(target.actions, target.agent_type)
    in_range = torch.isfinite(normalized).all(-1)
    if normalized_limit > 0:
        in_range &= normalized.abs().amax(-1) <= normalized_limit
    valid = (target.valid & in_range).long().cumprod(-1).bool()
    normalized = torch.where(valid[..., None], normalized, torch.zeros_like(normalized))
    eligible = full_valid[:, :, elapsed] & agent_mask.bool() & (history[:, :, -1, 5] > .5)
    metrics = dict(
        eligible_first=float(eligible.sum()),
        rejected_physical_first=float((eligible & ~target.valid[:, :, 0]).sum()),
        rejected_normalized_first=float((eligible & target.valid[:, :, 0] & ~valid[:, :, 0]).sum()),
        target_points=float(valid.sum()),
        available_gt_points=float(full_valid[:, :, elapsed:elapsed+available].sum()),
        first_recovery_displacement_sum=float((target.actions[:, :, 0, :2].norm(dim=-1)*valid[:, :, 0]).sum()),
        first_recovery_count=float(valid[:, :, 0].sum()),
    )
    return normalized.detach(), valid.detach(), metrics


def update_history(history, poses, valid):
    take = poses.shape[2]
    types = history[:, :, -1, 7]
    frames = history.new_zeros((*history.shape[:2], take, history.shape[-1]))
    frames[..., :2] = poses[..., :2]
    frames[..., 5] = valid
    frames[..., 6] = poses[..., 2]
    frames[..., 7] = types[:, :, None]
    return torch.cat((history, frames), 2)[:, :, -history.shape[2]:].detach()


@torch.no_grad()
def select_segment(model, normalizer, scene, history, target, elapsed, cfg, seed, agent_lengths=None):
    """One candidate for the whole scene; score only the next executed five steps."""
    take = min(5, target.future_pose.shape[2] - elapsed)
    mask = scene.agent_mask[:, :, None].expand(-1, -1, model.horizon)
    initial = torch.cat((history[:, :, -1, :2], history[:, :, -1, 6:7]), -1)
    gt = target.future_pose[:, :, elapsed:elapsed+take]
    valid = target.valid[:, :, elapsed:elapsed+take]
    candidates, scores = [], []
    for k in range(cfg.num_candidates):
        generator = torch.Generator(device=history.device).manual_seed(seed + k)
        actions = model.sample_normalized_actions(scene, mask,
            solver_steps=cfg.solver_steps, generator=generator)
        metric = normalizer.denormalize(actions, target.agent_type)
        poses = execute_model_actions(model, initial, metric[:, :, :take], mask[:, :, :take],
            agent_type=target.agent_type, agent_lengths=agent_lengths)
        distance, _ = motion_distances(poses, gt, initial, valid)
        candidates.append(poses)
        scores.append(distance)
    scores = torch.stack(scores, 1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Nonfinite candidate scores')
    winners = scores.argmin(1)
    poses = torch.stack(candidates, 1)[torch.arange(history.shape[0], device=history.device), winners]
    return poses.detach(), mask[:, :, :take], dict(
        selected_segment_distance=float(scores.min(1).values.mean()),
        candidate_segment_distance=float(scores.mean()),
    )


def backward_recovery(model, normalizer, prepared, batch, step, cfg, ta):
    history = prepared.history.detach()
    totals = {}
    bidx = torch.arange(history.shape[0], device=history.device)
    target = prepared.targets
    supervised_segments = 0
    for elapsed in range(0, 80, 5):
        kwargs = dict(history=history, agent_mask=batch['agent_mask'],
            map_polylines=batch['map_polylines'], map_mask=batch['map_mask'],
            current_lights=batch['lights'][bidx, prepared.anchors+elapsed],
            current_light_mask=batch['light_mask'][bidx, prepared.anchors+elapsed])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            sampling_scene = model.encode_scene(**kwargs)
            poses, executed_valid, diagnostics = select_segment(model, normalizer,
                sampling_scene, history, target, elapsed, cfg,
                cfg.seed+10000000+step*16*cfg.num_candidates+(elapsed//5)*cfg.num_candidates,
                agent_lengths=batch.get("agent_lengths"))
            next_history = update_history(history, poses, executed_valid)
        with torch.no_grad():
            normalized, valid, recovery = recovery_targets(history, target.future_pose,
                target.valid, batch['agent_mask'], normalizer, elapsed,
                horizon=model.horizon, max_displacement=ta.physical_max_displacement_m,
                max_yaw=ta.physical_max_yaw_delta_rad,
                normalized_limit=ta.train_normalized_action_clip,
                config=model_config(model), agent_lengths=batch.get("agent_lengths"))
        for key, value in recovery.items():totals[key]=totals.get(key, 0.)+value
        for key, value in diagnostics.items():totals[key]=totals.get(key, 0.)+value/16
        if valid.any():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scene = model.encode_scene(**kwargs)
                loss, metrics = flow_matching_loss(model, scene, normalized, valid,
                    condition_focus_actions=False, flow_time_max=ta.train_flow_time_max,
                    loss_type=ta.flow_loss_type, huber_beta=ta.flow_huber_beta)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite recovery loss')
            (loss/16).backward()
            totals['recovery_flow_loss']=totals.get('recovery_flow_loss', 0.)+float(loss.detach())/16
            supervised_segments += 1
        history = next_history
    totals['supervised_segments'] = supervised_segments
    eligible=max(1.,totals['eligible_first'])
    totals['rejected_physical_first_fraction']=totals['rejected_physical_first']/eligible
    totals['rejected_normalized_first_fraction']=totals['rejected_normalized_first']/eligible
    totals['recovery_target_coverage']=totals['target_points']/max(1.,totals['available_gt_points'])
    totals['mean_first_recovery_displacement_m']=totals['first_recovery_displacement_sum']/max(1.,totals['first_recovery_count'])
    return totals
