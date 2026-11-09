"""Opt-in Flow-ERD Eq. (2), (10)--(14) execution; no learned parameters.

Waymo types: 1 vehicle, 2 pedestrian, 3 cyclist. Other types retain holonomic
execution. The paper does not specify rejection thresholds or how the unused
lateral training channel is populated; we explicitly use zero for that channel.
"""
from dataclasses import asdict, dataclass
import math
import torch


@dataclass(frozen=True)
class KinematicsConfig:
    mode: str = 'holonomic'
    vehicle_rho: float | None = None
    cyclist_rho: float | None = None
    max_reexecution_error_m: float = 0.25

    def __post_init__(self):
        if self.mode not in ('holonomic', 'type_aware'):
            raise ValueError('Unknown action execution mode: ' + self.mode)
        if not math.isfinite(self.max_reexecution_error_m) or self.max_reexecution_error_m <= 0:
            raise ValueError('max_reexecution_error_m must be finite and positive')
        if self.mode == 'type_aware':
            for name in ('vehicle_rho', 'cyclist_rho'):
                value = getattr(self, name)
                if value is None or not math.isfinite(value):
                    raise ValueError(name + ' must be explicitly calibrated from turning data')

    @classmethod
    def from_args(cls, args):
        return cls(**getattr(args, 'action_kinematics', {}))

    def to_dict(self):
        return asdict(self)


def geometry(types, lengths, valid, config):
    wheeled = (types == 1) | (types == 3)
    if lengths is None:
        raise ValueError('type_aware execution requires agent_lengths (measured metres); regenerate NPZ data')
    if lengths.shape != types.shape:
        raise ValueError('agent_lengths must have the same (B,N) shape as agent types')
    if (valid & wheeled & (~torch.isfinite(lengths) | (lengths <= 0))).any():
        raise ValueError('Active vehicle/cyclist has missing or invalid measured length')
    lengths = torch.where(valid & wheeled, lengths, torch.zeros_like(lengths))
    rho = torch.where(types == 1, float(config.vehicle_rho), float(config.cyclist_rho))
    return wheeled, lengths * rho


def execute_step(pose, action, types, lengths, config, valid=None):
    from .direct_action_flow import execute_holonomic_step, wrap_angle_rad
    if config.mode == 'holonomic':
        return execute_holonomic_step(pose, action)
    if valid is None:
        valid = torch.ones_like(types, dtype=torch.bool)
    wheeled, offset = geometry(types, lengths, valid, config)
    # torch.sinc is normalized by pi, and is differentiable at a_yaw == 0.
    half = action[..., 2] / 2
    mid = pose[..., 2] + half
    forward = action[..., 0] * torch.sinc(half / math.pi)
    swing = 2 * offset * torch.sin(half)
    dx = forward * torch.cos(mid) - swing * torch.sin(mid)
    dy = forward * torch.sin(mid) + swing * torch.cos(mid)
    nh = torch.stack((pose[..., 0] + dx, pose[..., 1] + dy,
                      wrap_angle_rad(pose[..., 2] + action[..., 2])), -1)
    hol = execute_holonomic_step(pose, action)
    return torch.where(wheeled[..., None], nh, hol)


def execute_actions(pose, actions, valid=None, *, agent_type=None, agent_lengths=None,
                    config=KinematicsConfig()):
    from .direct_action_flow import execute_holonomic_actions
    if config.mode == 'holonomic':
        return execute_holonomic_actions(pose, actions, valid)
    if agent_type is None:
        raise ValueError('type_aware execution requires agent_type')
    if valid is None:
        valid = torch.ones(actions.shape[:-1], dtype=torch.bool, device=actions.device)
    output = []
    for t in range(actions.shape[-2]):
        mask = valid[..., t].bool()
        action = torch.where(mask[..., None], actions[..., t, :], 0.)
        nxt = execute_step(pose, action, agent_type, agent_lengths, config, mask)
        pose = torch.where(mask[..., None], nxt, pose)
        output.append(pose)
    return torch.stack(output, -2)


def inverse_actions(history, future, agent_mask, *, agent_lengths=None,
                    config=KinematicsConfig(), max_displacement_m=0., max_yaw_delta_rad=0.):
    from .direct_action_flow import (ActionTargets, inverse_holonomic_actions,
                                    physical_transition_valid, wrap_angle_rad)
    if config.mode == 'holonomic':
        return inverse_holonomic_actions(history, future, agent_mask,
            max_displacement_m=max_displacement_m, max_yaw_delta_rad=max_yaw_delta_rad)
    current = history[..., -1, :]
    initial = torch.stack((current[..., 0], current[..., 1], current[..., 6]), -1)
    types = current[..., 7].round().long()
    alive = agent_mask.bool() & (current[..., 5] > .5) & torch.isfinite(initial).all(-1)
    wheeled, _ = geometry(types, agent_lengths, alive, config)
    # Sanitise inactive tracks before nonlinear arithmetic (also for gradients).
    pose = torch.where(alive[..., None], initial, 0.)
    actions, masks, targets = [], [], []
    for t in range(future.shape[-2]):
        frame = future[..., t, :]
        gt = torch.stack((frame[..., 0], frame[..., 1], frame[..., 6]), -1)
        finite = torch.isfinite(gt).all(-1)
        safe_gt = torch.where(finite[..., None], gt, pose)
        delta = safe_gt[..., :2] - pose[..., :2]
        yaw = wrap_angle_rad(safe_gt[..., 2] - pose[..., 2])
        mid = pose[..., 2] + yaw / 2
        sinc = torch.sinc(yaw / (2 * math.pi))
        along = (delta[..., 0] * mid.cos() + delta[..., 1] * mid.sin()) / sinc
        c, s = pose[..., 2].cos(), pose[..., 2].sin()
        hol_long = c * delta[..., 0] + s * delta[..., 1]
        hol_lat = -s * delta[..., 0] + c * delta[..., 1]
        action = torch.stack((torch.where(wheeled, along, hol_long),
                              torch.where(wheeled, 0., hol_lat), yaw), -1)
        valid = alive & finite & (frame[..., 5] > .5) & physical_transition_valid(
            delta, yaw, max_displacement_m=max_displacement_m,
            max_yaw_delta_rad=max_yaw_delta_rad)
        action = torch.where(valid[..., None], action, 0.)
        executed = execute_step(pose, action, types, agent_lengths, config, valid)
        error = (executed[..., :2] - safe_gt[..., :2]).norm(dim=-1)
        valid = valid & (error <= config.max_reexecution_error_m)
        actions.append(torch.where(valid[..., None], action, 0.))
        masks.append(valid)
        targets.append(safe_gt)
        # Eq. (13)--(14): next inverse starts from executed pose, never GT reset.
        pose = torch.where(valid[..., None], executed, pose)
        alive = valid
    return ActionTargets(torch.stack(actions, -2), torch.stack(masks, -1),
                         torch.stack(targets, -2), initial, types)


def model_config(model):
    return getattr(model, 'kinematics', KinematicsConfig())


def execute_model_actions(model, pose, actions, valid=None, *, agent_type=None, agent_lengths=None):
    return execute_actions(pose, actions, valid, agent_type=agent_type,
                           agent_lengths=agent_lengths, config=model_config(model))


def inverse_model_actions(model, history, future, agent_mask, *, agent_lengths=None, **kwargs):
    return inverse_actions(history, future, agent_mask, agent_lengths=agent_lengths,
                           config=model_config(model), **kwargs)


def estimate_rho(pose, next_pose, types, lengths, valid, *, min_abs_yaw=.02,
                 max_abs_yaw=.75, max_displacement_m=5., min_samples=100):
    """Eq. (12), signed per-type median on usable turning intervals only.

    Thresholds and median are explicit reproduction choices, not claimed paper
    hyperparameters. Caller must use training data, not validation/test data.
    """
    from .direct_action_flow import wrap_angle_rad
    if not 0 < min_abs_yaw < max_abs_yaw < math.pi or min_samples < 1:
        raise ValueError('Invalid calibration thresholds')
    yaw = wrap_angle_rad(next_pose[..., 2] - pose[..., 2])
    mid = pose[..., 2] + yaw / 2
    delta = next_pose[..., :2] - pose[..., :2]
    usable = (valid & torch.isfinite(pose).all(-1) & torch.isfinite(next_pose).all(-1)
              & torch.isfinite(lengths) & (lengths > 0)
              & (yaw.abs() >= min_abs_yaw) & (yaw.abs() <= max_abs_yaw)
              & (delta.norm(dim=-1) <= max_displacement_m))
    lateral = -delta[..., 0] * mid.sin() + delta[..., 1] * mid.cos()
    denominator = 2 * lengths * torch.sin(yaw / 2)
    estimates = lateral / torch.where(usable, denominator, torch.ones_like(denominator))
    result = {}
    for typ, name in ((1, 'vehicle'), (3, 'cyclist')):
        values = estimates[usable & (types == typ)]
        if values.numel() < min_samples:
            raise ValueError(f'{name}: need {min_samples} turning intervals, got {values.numel()}')
        result[name + '_rho'] = float(values.median())
        result[name + '_count'] = values.numel()
    return result
