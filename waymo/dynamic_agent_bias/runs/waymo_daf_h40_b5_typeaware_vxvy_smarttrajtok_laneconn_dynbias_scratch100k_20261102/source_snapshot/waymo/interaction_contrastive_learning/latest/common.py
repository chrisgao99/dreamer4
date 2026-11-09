"""Small array helpers shared by the current full-pair pipeline."""

from __future__ import annotations

import numpy as np


def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians to the half-open interval [-pi, pi)."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def ensure_agent_time_layout(agents: np.ndarray, agent_mask: np.ndarray) -> np.ndarray:
    """Return an agent tensor with shape (agents, time, features)."""
    if agents.ndim != 3:
        raise ValueError(f"Expected rank-3 agents, got {agents.shape}")
    num_agents = int(agent_mask.shape[0])
    if agents.shape[0] == num_agents:
        return agents
    if agents.shape[1] == num_agents:
        return np.transpose(agents, (1, 0, 2))
    raise ValueError(
        f"Cannot infer agent/time layout from {agents.shape} and mask {agent_mask.shape}"
    )
