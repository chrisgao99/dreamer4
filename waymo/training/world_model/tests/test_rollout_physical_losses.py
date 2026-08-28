from __future__ import annotations

import math

import torch

from waymo.training.world_model.rollout_physical_losses import (
    decoded_kinematic_consistency_loss,
    fixed_footprint_offroad_loss,
    fixed_sedan_collision_loss,
)


def _states(xy: torch.Tensor, *, yaw: float = 0.0, speed: float = 0.0) -> torch.Tensor:
    """Build (B=1,N=1,T,K,7) decoded states."""
    out = torch.zeros((1, 1, *xy.shape[:-1], 7), dtype=torch.float32)
    out[..., 0:2] = xy
    out[..., 2] = speed
    out[..., 5] = math.sin(yaw)
    out[..., 6] = math.cos(yaw)
    return out


def test_fixed_sedan_collision_penalty_increases_as_clearance_closes() -> None:
    # Two east-facing sedans. Adjacent-lane spacing is safe, while a rear-end
    # gap approaching the fixed 4.8 m body length receives increasing penalty.
    valid = torch.ones((1, 2, 2), dtype=torch.bool)

    adjacent_xy = torch.tensor([[[0.0, 0.0], [0.0, 3.5]], [[0.0, 0.0], [0.0, 3.5]]])
    touching_xy = torch.tensor([[[0.0, 0.0], [4.8, 0.0]], [[0.0, 0.0], [4.8, 0.0]]])
    overlapping_xy = torch.tensor([[[0.0, 0.0], [4.0, 0.0]], [[0.0, 0.0], [4.0, 0.0]]])

    adjacent, _ = fixed_sedan_collision_loss(_states(adjacent_xy), valid, future_start=1)
    touching, _ = fixed_sedan_collision_loss(_states(touching_xy), valid, future_start=1)
    overlapping_state = _states(overlapping_xy).requires_grad_(True)
    overlapping, metrics = fixed_sedan_collision_loss(overlapping_state, valid, future_start=1)

    assert adjacent < touching < overlapping
    assert metrics["collision_overlap_rate_proxy"].item() == 1.0
    overlapping.backward()
    assert overlapping_state.grad is not None
    assert torch.isfinite(overlapping_state.grad).all()


def test_offroad_uses_all_road_edges_and_penalizes_offroad_side() -> None:
    # East-facing edge: north/left is on-road (negative), south/right off-road.
    map_polylines = torch.zeros((1, 1, 3, 6), dtype=torch.float32)
    map_polylines[0, 0, :, 0] = torch.tensor([-5.0, 0.0, 5.0])
    map_polylines[0, 0, :, 2] = 1.0
    map_polylines[0, 0, :, 4] = 15.0
    map_polylines[0, 0, :, 5] = 1.0
    map_mask = torch.ones((1, 1, 3), dtype=torch.bool)
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    onroad_xy = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]]])
    offroad_xy = torch.tensor([[[0.0, -2.0]], [[0.0, -2.0]]])
    onroad, onroad_metrics = fixed_footprint_offroad_loss(
        _states(onroad_xy), valid, map_polylines, map_mask, future_start=1
    )
    offroad_state = _states(offroad_xy).requires_grad_(True)
    offroad, offroad_metrics = fixed_footprint_offroad_loss(
        offroad_state, valid, map_polylines, map_mask, future_start=1
    )

    assert onroad < offroad
    assert onroad_metrics["offroad_rate_proxy"].item() == 0.0
    assert offroad_metrics["offroad_rate_proxy"].item() == 1.0
    offroad.backward()
    assert offroad_state.grad is not None
    assert torch.isfinite(offroad_state.grad).all()


def test_kinematic_consistency_matches_position_velocity_and_speed_yaw() -> None:
    valid = torch.ones((1, 3, 1), dtype=torch.bool)
    xy = torch.tensor([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]])
    state = _states(xy, speed=10.0)
    state[..., 3] = 10.0
    state.requires_grad_(True)

    velocity_loss, speed_yaw_loss, _ = decoded_kinematic_consistency_loss(
        state, valid, future_start=1, dt=0.1
    )
    assert velocity_loss.item() < 1e-8
    assert speed_yaw_loss.item() < 1e-8

    inconsistent = state.detach().clone()
    inconsistent[..., 3] = 0.0
    inconsistent.requires_grad_(True)
    velocity_bad, _, _ = decoded_kinematic_consistency_loss(
        inconsistent, valid, future_start=1, dt=0.1
    )
    assert velocity_bad.item() > velocity_loss.item()
    velocity_bad.backward()
    assert torch.isfinite(inconsistent.grad).all()
