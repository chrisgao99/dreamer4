"""Differentiable physical proxy losses for decoded joint Waymo rollouts.

The first implementation deliberately uses one fixed sedan footprint for every
agent.  Collision uses a heading-aware projected-radius clearance rather than
an exact oriented-box intersection.  Offroad uses every retained Waymo road
edge (types 15/16), not the current lane boundary.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


ROAD_EDGE_TYPES = (15, 16)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(device=value.device, dtype=value.dtype)
    while weight.dim() < value.dim():
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _smooth_relu(value: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = max(float(temperature), 1e-4)
    return F.softplus(value / temperature) * temperature


def _smooth_l1_zero(value: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(value, torch.zeros_like(value), reduction="none", beta=1.0)


def _decoded_state(
    agent_continuous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return xy, speed, vxvy, yaw from (..., 7) decoded state."""
    if agent_continuous.shape[-1] != 7:
        raise ValueError(
            "Expected decoded agent state (...,7) containing "
            "x,y,speed,vx,vy,sin(yaw),cos(yaw); got "
            f"{tuple(agent_continuous.shape)}"
        )
    xy = agent_continuous[..., 0:2].float()
    speed = agent_continuous[..., 2].float()
    vxvy = agent_continuous[..., 3:5].float()
    yaw = torch.atan2(agent_continuous[..., 5].float(), agent_continuous[..., 6].float())
    return xy, speed, vxvy, yaw


def decoded_motion_ground_truth_loss(
    agent_continuous: torch.Tensor,
    agents_btkf: torch.Tensor,
    *,
    future_start: int,
    agent_loss_weight_multiplier: torch.Tensor | None = None,
    xy_weight: float = 1.0,
    velocity_weight: float = 0.5,
    yaw_weight: float = 0.5,
    focus_agent_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Supervise every decoded future state directly from raw motion targets.

    The frozen tokenizer decoder remains in the autograd graph, so this loss
    updates only the world model while providing gradients at every valid
    future timestep. ``agent_continuous`` is a single selected rollout with
    shape ``(B,T,K,7)``; Minimum-over-N winner selection happens before this
    function is called.
    """
    if agent_continuous.dim() != 4 or agent_continuous.shape[-1] != 7:
        raise ValueError(
            "Expected agent_continuous=(B,T,K,7), got "
            f"{tuple(agent_continuous.shape)}"
        )
    if agents_btkf.dim() != 4 or agents_btkf.shape[-1] < 7:
        raise ValueError(f"Expected agents_btkf=(B,T,K,F>=7), got {tuple(agents_btkf.shape)}")
    if agent_continuous.shape[:3] != agents_btkf.shape[:3]:
        raise ValueError(
            "Decoded and target motion shapes must agree through (B,T,K); got "
            f"{tuple(agent_continuous.shape[:3])} vs {tuple(agents_btkf.shape[:3])}"
        )

    start = int(future_start)
    if start < 0 or start >= int(agent_continuous.shape[1]):
        raise ValueError(
            f"future_start must be in [0,{int(agent_continuous.shape[1])}), got {future_start}"
        )

    pred = agent_continuous[:, start:].float()
    target = agents_btkf[:, start:].float()
    valid = target[..., 5] > 0.5
    state_weight = valid.to(dtype=pred.dtype)
    if agent_loss_weight_multiplier is not None:
        multiplier = agent_loss_weight_multiplier[:, start:].to(device=pred.device, dtype=pred.dtype)
        if multiplier.shape != state_weight.shape:
            raise ValueError(
                "agent_loss_weight_multiplier must match (B,T,K); got "
                f"{tuple(multiplier.shape)} vs {tuple(state_weight.shape)}"
            )
        state_weight = state_weight * multiplier
    if float(focus_agent_weight) != 1.0 and int(state_weight.shape[-1]) > 0:
        state_weight = state_weight.clone()
        state_weight[..., 0] *= float(focus_agent_weight)

    target_yaw = target[..., 6]
    target_yaw_vector = torch.stack((torch.sin(target_yaw), torch.cos(target_yaw)), dim=-1)
    xy_raw = F.smooth_l1_loss(pred[..., 0:2], target[..., 0:2], reduction="none")
    velocity_raw = F.smooth_l1_loss(pred[..., 2:5], target[..., 2:5], reduction="none")
    yaw_raw = F.smooth_l1_loss(pred[..., 5:7], target_yaw_vector, reduction="none")
    xy_loss = _masked_mean(xy_raw, state_weight)
    velocity_loss = _masked_mean(velocity_raw, state_weight)
    yaw_loss = _masked_mean(yaw_raw, state_weight)
    total = (
        float(xy_weight) * xy_loss
        + float(velocity_weight) * velocity_loss
        + float(yaw_weight) * yaw_loss
    )

    pred_yaw = torch.atan2(pred[..., 5], pred[..., 6])
    yaw_error = torch.atan2(torch.sin(pred_yaw - target_yaw), torch.cos(pred_yaw - target_yaw)).abs()
    metrics = {
        "loss_motion_gt": total.detach(),
        "loss_motion_gt_xy": xy_loss.detach(),
        "loss_motion_gt_velocity": velocity_loss.detach(),
        "loss_motion_gt_yaw": yaw_loss.detach(),
        "motion_gt_xy_mae_m": _masked_mean((pred[..., 0:2] - target[..., 0:2]).norm(dim=-1), state_weight).detach(),
        "motion_gt_speed_mae_mps": _masked_mean((pred[..., 2] - target[..., 2]).abs(), state_weight).detach(),
        "motion_gt_vxvy_mae_mps": _masked_mean((pred[..., 3:5] - target[..., 3:5]).norm(dim=-1), state_weight).detach(),
        "motion_gt_yaw_mae_deg": (_masked_mean(yaw_error, state_weight) * (180.0 / torch.pi)).detach(),
        "motion_gt_valid_states": valid.sum().detach().to(torch.float32),
    }
    return total, metrics


def fixed_sedan_collision_loss(
    agent_continuous: torch.Tensor,
    valid: torch.Tensor,
    *,
    future_start: int,
    vehicle_length_m: float = 4.8,
    vehicle_width_m: float = 2.0,
    warning_clearance_m: float = 1.0,
    temperature_m: float = 0.2,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize close agent pairs using a fixed heading-aware sedan proxy.

    Args:
        agent_continuous: (B,N,T,K,7), where N is the number of MoN samples.
        valid: (B,T,K) ground-truth validity.  Predicted validity is not used,
            preventing the model from hiding unsafe agents by marking them
            invalid.
    """
    xy, _, _, yaw = _decoded_state(agent_continuous)
    xy = xy[:, :, int(future_start) :]
    yaw = yaw[:, :, int(future_start) :]
    valid_future = valid[:, int(future_start) :].to(torch.bool)
    num_agents = int(xy.shape[-2])

    forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    lateral = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
    delta = xy.unsqueeze(-3) - xy.unsqueeze(-2)  # j - i: (B,N,H,K,K,2)
    raw_center_distance = delta.norm(dim=-1)
    fallback_direction = forward.unsqueeze(-2).expand_as(delta)
    direction = torch.where(
        (raw_center_distance > 1e-4).unsqueeze(-1),
        delta / raw_center_distance.clamp_min(1e-4).unsqueeze(-1),
        fallback_direction,
    )
    center_distance = raw_center_distance
    half_length = 0.5 * float(vehicle_length_m)
    half_width = 0.5 * float(vehicle_width_m)

    # Project each fixed sedan onto the line joining the two centers.  This is
    # much cheaper than SAT while preserving the crucial distinction between
    # longitudinal following and normal side-by-side lane traffic.
    radius_i = (
        half_length * (forward.unsqueeze(-2) * direction).sum(dim=-1).abs()
        + half_width * (lateral.unsqueeze(-2) * direction).sum(dim=-1).abs()
    )
    radius_j = (
        half_length * (forward.unsqueeze(-3) * direction).sum(dim=-1).abs()
        + half_width * (lateral.unsqueeze(-3) * direction).sum(dim=-1).abs()
    )
    clearance = center_distance - radius_i - radius_j

    pair_valid = valid_future[:, None, :, :, None] & valid_future[:, None, :, None, :]
    upper = torch.triu(
        torch.ones((num_agents, num_agents), device=xy.device, dtype=torch.bool), diagonal=1
    )
    pair_valid = pair_valid & upper.view(1, 1, 1, num_agents, num_agents)

    violation = _smooth_relu(float(warning_clearance_m) - clearance, temperature_m)
    loss = _masked_mean(_smooth_l1_zero(violation), pair_valid)
    overlap_rate = _masked_mean((clearance < 0.0).to(clearance.dtype), pair_valid)
    warning_rate = _masked_mean(
        (clearance < float(warning_clearance_m)).to(clearance.dtype), pair_valid
    )
    mean_violation = _masked_mean(violation, pair_valid)
    return loss, {
        "loss_collision": loss.detach(),
        "collision_overlap_rate_proxy": overlap_rate.detach(),
        "collision_warning_rate_proxy": warning_rate.detach(),
        "collision_mean_violation_m": mean_violation.detach(),
        "collision_valid_pairs": pair_valid.sum().detach().to(torch.float32),
    }


def decoded_kinematic_consistency_loss(
    agent_continuous: torch.Tensor,
    valid: torch.Tensor,
    *,
    future_start: int,
    dt: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Match decoded position deltas to decoded vx/vy and speed/yaw."""
    if int(future_start) < 1:
        raise ValueError(f"future_start must be >= 1, got {future_start}")
    xy, speed, vxvy, yaw = _decoded_state(agent_continuous)
    start = int(future_start)
    delta_xy = xy[:, :, start:] - xy[:, :, start - 1 : -1]
    vxvy_delta = vxvy[:, :, start - 1 : -1] * float(dt)
    speed_yaw_delta = torch.stack(
        (
            speed[:, :, start - 1 : -1] * torch.cos(yaw[:, :, start - 1 : -1]) * float(dt),
            speed[:, :, start - 1 : -1] * torch.sin(yaw[:, :, start - 1 : -1]) * float(dt),
        ),
        dim=-1,
    )
    consecutive = (
        valid[:, start - 1 : -1].to(torch.bool) & valid[:, start:].to(torch.bool)
    )[:, None]
    velocity_raw = F.smooth_l1_loss(delta_xy, vxvy_delta, reduction="none")
    speed_yaw_raw = F.smooth_l1_loss(delta_xy, speed_yaw_delta, reduction="none")
    velocity_loss = _masked_mean(velocity_raw, consecutive)
    speed_yaw_loss = _masked_mean(speed_yaw_raw, consecutive)
    velocity_mae = _masked_mean((delta_xy - vxvy_delta).norm(dim=-1), consecutive)
    speed_yaw_mae = _masked_mean((delta_xy - speed_yaw_delta).norm(dim=-1), consecutive)
    return velocity_loss, speed_yaw_loss, {
        "loss_kinematic_xy": velocity_loss.detach(),
        "loss_speed_yaw_kinematic": speed_yaw_loss.detach(),
        "kinematic_xy_mae_m": velocity_mae.detach(),
        "speed_yaw_kinematic_mae_m": speed_yaw_mae.detach(),
    }


def _nearest_oriented_edge_signed_distance(
    query_xy: torch.Tensor,
    edge_xy: torch.Tensor,
    edge_direction: torch.Tensor,
    *,
    query_chunk_size: int,
) -> torch.Tensor:
    """Approximate signed road-edge distance from nearest sampled edge points.

    Waymo road edges are oriented with their port/left side on-road.  For an
    east-facing edge, a point north (left) therefore receives a negative sign.
    Nearest-point selection is discrete and detached; the selected signed
    perpendicular distance remains differentiable with respect to the query.
    """
    if query_xy.dim() != 2 or query_xy.shape[-1] != 2:
        raise ValueError(f"Expected query_xy=(Q,2), got {tuple(query_xy.shape)}")
    if edge_xy.numel() == 0:
        return query_xy.sum(dim=-1) * 0.0

    edge_xy_f = edge_xy.float()
    edge_direction_f = F.normalize(edge_direction.float(), dim=-1, eps=1e-6)
    signed_parts = []
    chunk_size = max(1, int(query_chunk_size))
    for start in range(0, int(query_xy.shape[0]), chunk_size):
        query = query_xy[start : start + chunk_size].float()
        with torch.no_grad():
            nearest = torch.cdist(query.detach(), edge_xy_f).argmin(dim=-1)
        nearest_xy = edge_xy_f.index_select(0, nearest)
        nearest_dir = edge_direction_f.index_select(0, nearest)
        offset = query - nearest_xy
        # cross(offset, tangent): negative on the port/left/on-road side.
        signed_parts.append(offset[:, 0] * nearest_dir[:, 1] - offset[:, 1] * nearest_dir[:, 0])
    return torch.cat(signed_parts, dim=0)


def fixed_footprint_offroad_loss(
    agent_continuous: torch.Tensor,
    valid: torch.Tensor,
    map_polylines: torch.Tensor,
    map_mask: torch.Tensor,
    *,
    future_start: int,
    vehicle_length_m: float = 4.8,
    vehicle_width_m: float = 2.0,
    boundary_margin_m: float = 0.3,
    temperature_m: float = 0.2,
    query_chunk_size: int = 1024,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize the most off-road corner of a fixed sedan footprint.

    All retained road-edge map elements are queried.  Lane markings and the
    agent's current lane are intentionally irrelevant, so legal lane changes
    are not penalized.
    """
    xy, _, _, yaw = _decoded_state(agent_continuous)
    start = int(future_start)
    xy = xy[:, :, start:]
    yaw = yaw[:, :, start:]
    valid_future = valid[:, start:].to(torch.bool)
    bsz, num_candidates, horizon, num_agents = xy.shape[:4]

    forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    lateral = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
    half_length = 0.5 * float(vehicle_length_m)
    half_width = 0.5 * float(vehicle_width_m)
    signs = xy.new_tensor(((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)))
    corners = (
        xy.unsqueeze(-2)
        + signs[:, 0].view(1, 1, 1, 1, 4, 1) * half_length * forward.unsqueeze(-2)
        + signs[:, 1].view(1, 1, 1, 1, 4, 1) * half_width * lateral.unsqueeze(-2)
    )

    signed_box_parts = []
    scene_has_edges = []
    for batch_index in range(bsz):
        point_type = map_polylines[batch_index, ..., 4].round().to(torch.long)
        edge_valid = map_mask[batch_index].to(torch.bool)
        edge_valid = edge_valid & ((point_type == ROAD_EDGE_TYPES[0]) | (point_type == ROAD_EDGE_TYPES[1]))
        edge_direction = map_polylines[batch_index, ..., 2:4]
        edge_valid = edge_valid & (edge_direction.float().norm(dim=-1) > 1e-6)
        edge_xy = map_polylines[batch_index, ..., 0:2][edge_valid]
        edge_dir = edge_direction[edge_valid]
        scene_has_edge = bool(edge_xy.shape[0] > 0)
        scene_has_edges.append(scene_has_edge)

        scene_corners = corners[batch_index]
        signed_corners = _nearest_oriented_edge_signed_distance(
            scene_corners.reshape(-1, 2),
            edge_xy,
            edge_dir,
            query_chunk_size=query_chunk_size,
        ).reshape(num_candidates, horizon, num_agents, 4)
        signed_box_parts.append(signed_corners.max(dim=-1).values)

    signed_box_distance = torch.stack(signed_box_parts, dim=0)
    edge_coverage = torch.tensor(scene_has_edges, device=xy.device, dtype=torch.bool)
    scored_valid = valid_future[:, None] & edge_coverage[:, None, None, None]
    violation = _smooth_relu(signed_box_distance + float(boundary_margin_m), temperature_m)
    loss = _masked_mean(_smooth_l1_zero(violation), scored_valid)
    offroad_rate = _masked_mean((signed_box_distance > 0.0).to(xy.dtype), scored_valid)
    mean_violation = _masked_mean(violation, scored_valid)
    mean_signed_distance = _masked_mean(signed_box_distance, scored_valid)
    return loss, {
        "loss_offroad": loss.detach(),
        "offroad_rate_proxy": offroad_rate.detach(),
        "offroad_mean_violation_m": mean_violation.detach(),
        "road_edge_signed_distance_m": mean_signed_distance.detach(),
        "road_edge_scene_coverage": edge_coverage.float().mean().detach(),
    }


def decoded_rollout_physical_losses(
    agent_continuous: torch.Tensor,
    agents_btkf: torch.Tensor,
    map_polylines: torch.Tensor,
    map_mask: torch.Tensor,
    *,
    future_start: int,
    vehicle_length_m: float,
    vehicle_width_m: float,
    collision_warning_clearance_m: float,
    collision_temperature_m: float,
    offroad_boundary_margin_m: float,
    offroad_temperature_m: float,
    road_edge_query_chunk_size: int,
    kinematic_dt: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute all-candidate decoded physical losses and diagnostics."""
    valid = agents_btkf[..., 5] > 0.5
    collision, collision_metrics = fixed_sedan_collision_loss(
        agent_continuous,
        valid,
        future_start=future_start,
        vehicle_length_m=vehicle_length_m,
        vehicle_width_m=vehicle_width_m,
        warning_clearance_m=collision_warning_clearance_m,
        temperature_m=collision_temperature_m,
    )
    offroad, offroad_metrics = fixed_footprint_offroad_loss(
        agent_continuous,
        valid,
        map_polylines,
        map_mask,
        future_start=future_start,
        vehicle_length_m=vehicle_length_m,
        vehicle_width_m=vehicle_width_m,
        boundary_margin_m=offroad_boundary_margin_m,
        temperature_m=offroad_temperature_m,
        query_chunk_size=road_edge_query_chunk_size,
    )
    kinematic_xy, speed_yaw, kinematic_metrics = decoded_kinematic_consistency_loss(
        agent_continuous,
        valid,
        future_start=future_start,
        dt=kinematic_dt,
    )
    losses = {
        "collision": collision,
        "offroad": offroad,
        "kinematic_xy": kinematic_xy,
        "speed_yaw_kinematic": speed_yaw,
    }
    metrics = {**collision_metrics, **offroad_metrics, **kinematic_metrics}
    return losses, metrics
