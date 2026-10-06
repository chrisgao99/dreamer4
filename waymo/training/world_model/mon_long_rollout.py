"""Differentiable pose rollout with explicit context-gradient boundaries."""
import torch
from waymo.training.world_model.action_kinematics import execute_model_actions, inverse_model_actions
from torch.utils.checkpoint import checkpoint
from waymo.training.world_model.direct_action_flow import execute_holonomic_actions


def generate_poses(model, normalizer, history, batch, anchors, *,
                   rollout_steps=80, commitment=5, detach_every=0,
                   solver_steps=8, checkpoint_grad=True):
    if detach_every < 0 or (detach_every and detach_every % commitment):
        raise ValueError('detach_every must be zero or a multiple of commitment')
    if commitment < 1 or commitment > model.horizon:
        raise ValueError('Invalid commitment')
    types = history[:, :, -1, 7].round().long().clamp_min(0)
    indices = torch.arange(history.shape[0], device=history.device)
    history_length = history.shape[2]
    pieces = []

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

    for elapsed in range(0, rollout_steps, commitment):
        light = batch['lights'][indices, anchors + elapsed]
        light_mask = batch['light_mask'][indices, anchors + elapsed]
        if torch.is_grad_enabled() and checkpoint_grad:
            actions, mask = checkpoint(plan, history, light, light_mask,
                use_reentrant=False, preserve_rng_state=True)
        else:
            actions, mask = plan(history, light, light_mask)
        take = min(commitment, rollout_steps - elapsed)
        actions, mask = actions[:, :, :take], mask[:, :, :take]
        metric = normalizer.denormalize(actions, types)
        pose = torch.cat((history[:, :, -1, :2], history[:, :, -1, 6:7]), dim=-1)
        poses = execute_model_actions(model, pose, metric, mask,
            agent_type=types, agent_lengths=batch.get("agent_lengths"))
        # Keep executed poses, NOT concatenated actions re-integrated from t=0:
        # re-integrating all actions would bypass the intended detach boundaries.
        pieces.append(poses)
        motion = history.new_zeros((*poses.shape[:-1], 3))
        if getattr(model, 'include_agent_velocity', False):
            previous_xy = torch.cat((pose[:, :, None, :2], poses[:, :, :-1, :2]), dim=2)
            velocity = (poses[..., :2] - previous_xy) / 0.1
            velocity = torch.where(mask[..., None], velocity, torch.zeros_like(velocity))
            motion = torch.cat((torch.linalg.vector_norm(velocity, dim=-1, keepdim=True), velocity), dim=-1)
        frames = torch.cat((poses[..., :2], motion,
            mask[..., None].to(history.dtype), poses[..., 2:3],
            types[:, :, None, None].expand(-1, -1, take, 1).to(history.dtype)), dim=-1)
        if history.shape[-1] > 8:
            frames = torch.cat((frames, history.new_zeros((*frames.shape[:-1], history.shape[-1]-8))), -1)
        history = torch.cat((history, frames), dim=2)[:, :, -history_length:]
        if detach_every and (elapsed + take) % detach_every == 0:
            history = history.detach()
    return torch.cat(pieces, dim=2)
