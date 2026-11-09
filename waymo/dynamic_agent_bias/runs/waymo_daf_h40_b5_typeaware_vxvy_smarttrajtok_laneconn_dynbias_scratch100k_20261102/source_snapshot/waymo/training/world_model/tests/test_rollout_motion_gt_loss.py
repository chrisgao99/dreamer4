from __future__ import annotations

import math

import torch

from waymo.training.world_model.rollout_physical_losses import decoded_motion_ground_truth_loss


def _raw_agents(time_steps: int = 5, num_agents: int = 2) -> torch.Tensor:
    agents = torch.zeros((1, time_steps, num_agents, 7), dtype=torch.float32)
    agents[..., 0] = torch.arange(time_steps, dtype=torch.float32).view(1, -1, 1)
    agents[..., 2] = 10.0
    agents[..., 3] = 10.0
    agents[..., 5] = 1.0
    agents[..., 6] = math.pi / 4.0
    return agents


def _decoded_from_raw(agents: torch.Tensor) -> torch.Tensor:
    yaw = agents[..., 6]
    return torch.cat(
        (
            agents[..., 0:5],
            torch.sin(yaw).unsqueeze(-1),
            torch.cos(yaw).unsqueeze(-1),
        ),
        dim=-1,
    )


def test_motion_gt_supervises_every_future_timestep_but_not_context() -> None:
    agents = _raw_agents()
    decoded = _decoded_from_raw(agents)
    decoded[:, 0, :, 0] += 100.0
    decoded[:, 1:, :, 0] += 1.0
    decoded.requires_grad_(True)

    loss, metrics = decoded_motion_ground_truth_loss(decoded, agents, future_start=1)
    assert loss.item() > 0.0
    assert metrics["motion_gt_valid_states"].item() == 8.0
    loss.backward()

    assert decoded.grad is not None
    assert decoded.grad[:, 0].abs().sum().item() == 0.0
    for timestep in range(1, 5):
        assert decoded.grad[:, timestep, :, 0].abs().sum().item() > 0.0


def test_motion_gt_respects_valid_mask_and_agent_weights() -> None:
    agents = _raw_agents(time_steps=3, num_agents=2)
    agents[:, 2, 1, 5] = 0.0
    decoded = _decoded_from_raw(agents)
    decoded[:, 1:, :, 0] += 1.0
    multiplier = torch.ones((1, 3, 2), dtype=torch.float32)
    multiplier[..., 1] = 0.0

    loss, _ = decoded_motion_ground_truth_loss(
        decoded,
        agents,
        future_start=1,
        agent_loss_weight_multiplier=multiplier,
    )
    reference, _ = decoded_motion_ground_truth_loss(
        decoded[:, :, :1],
        agents[:, :, :1],
        future_start=1,
    )
    torch.testing.assert_close(loss, reference)
