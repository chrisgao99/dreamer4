from __future__ import annotations

import torch

from waymo.training.world_model.direct_action_flow import (
    ActionNormalizer,
    DirectActionFlowModel,
    _modulate,
    execute_holonomic_actions,
    flow_matching_loss,
    inverse_holonomic_actions,
    rollout_receding_horizon,
    select_window_anchors,
)


def _synthetic_batch(
    *, batch_size: int = 2, num_agents: int = 3, history: int = 4, horizon: int = 6
) -> dict[str, torch.Tensor]:
    total = history + horizon
    agents = torch.zeros(batch_size, num_agents, total, 8)
    agent_mask = torch.ones(batch_size, num_agents, dtype=torch.bool)
    time = torch.arange(total).float()
    for agent in range(num_agents):
        yaw = 0.1 * agent + 0.01 * time
        agents[:, agent, :, 0] = agent * 4.0 + 0.5 * time
        agents[:, agent, :, 1] = agent * 1.5 + 0.05 * time.square()
        agents[:, agent, :, 5] = 1.0
        agents[:, agent, :, 6] = yaw
        agents[:, agent, :, 7] = 1 + (agent % 3)
    map_polylines = torch.zeros(batch_size, 4, 3, 6)
    map_mask = torch.ones(batch_size, 4, 3, dtype=torch.bool)
    map_polylines[..., 0] = torch.arange(3).float()
    map_polylines[..., 2] = 1.0
    map_polylines[..., 4] = 1.0
    map_polylines[..., 5] = 1.0
    lights = torch.zeros(batch_size, total, 2, 4)
    light_mask = torch.ones(batch_size, total, 2, dtype=torch.bool)
    lights[..., 0] = 5.0
    lights[..., 2] = 3.0
    lights[..., 3] = 1.0
    return {
        "agents": agents,
        "agent_mask": agent_mask,
        "map_polylines": map_polylines,
        "map_mask": map_mask,
        "lights": lights,
        "light_mask": light_mask,
    }


def _small_model(history: int = 4, horizon: int = 6) -> DirectActionFlowModel:
    return DirectActionFlowModel(
        d_model=32,
        n_heads=4,
        history_length=history,
        horizon=horizon,
        chunk_size=2,
        history_depth=1,
        map_depth=1,
        scene_depth=1,
        action_depth=1,
        step_refiner_depth=1,
        hidden_dim=16,
        dropout=0.0,
    )


def test_inverse_actions_round_trip_logged_poses() -> None:
    batch = _synthetic_batch()
    history = batch["agents"][:, :, :4]
    future = batch["agents"][:, :, 4:]
    targets = inverse_holonomic_actions(history, future, batch["agent_mask"])
    reconstructed = execute_holonomic_actions(
        targets.current_pose, targets.actions, targets.valid
    )
    torch.testing.assert_close(
        reconstructed[..., 0:2], targets.future_pose[..., 0:2], atol=2e-6, rtol=0
    )
    angle_error = torch.atan2(
        torch.sin(reconstructed[..., 2] - targets.future_pose[..., 2]),
        torch.cos(reconstructed[..., 2] - targets.future_pose[..., 2]),
    )
    assert angle_error.abs().max().item() < 2e-6


def test_physical_filter_invalidates_jump_and_remaining_suffix() -> None:
    batch = _synthetic_batch(batch_size=1)
    history = batch["agents"][:, :, :4]
    future = batch["agents"][:, :, 4:].clone()
    future[:, 1, 2, 0] += 20.0
    targets = inverse_holonomic_actions(
        history,
        future,
        batch["agent_mask"],
        max_displacement_m=5.0,
        max_yaw_delta_rad=0.75,
    )
    assert targets.valid[0, 1, :2].all()
    assert not targets.valid[0, 1, 2:].any()
    assert torch.equal(
        targets.actions[0, 1, 2:], torch.zeros_like(targets.actions[0, 1, 2:])
    )


def test_random_anchor_selection_accepts_physical_integer_scores() -> None:
    batch = _synthetic_batch(batch_size=2, history=4, horizon=6)
    anchors = select_window_anchors(
        batch["agents"],
        batch["agent_mask"],
        history_length=4,
        horizon=6,
        random_start=True,
        max_displacement_m=5.0,
        max_yaw_delta_rad=0.75,
    )
    assert anchors.dtype == torch.long
    assert anchors.shape == (2,)


def test_bounded_modulation_limits_scale_and_shift() -> None:
    x = torch.ones(2, 3, 4)
    params = torch.cat(
        (
            torch.full((2, 4), 1_000.0),
            torch.full((2, 4), 1_000.0),
            torch.zeros(2, 4),
        ),
        dim=-1,
    )
    modulated, gate = _modulate(x, params, scale_limit=2.0, shift_limit=5.0)
    assert modulated.abs().max().item() <= 8.0
    torch.testing.assert_close(gate, torch.full_like(gate, 0.5))


def test_joint_flow_has_explicit_agent_and_action_tokens() -> None:
    batch = _synthetic_batch()
    model = _small_model()
    history = batch["agents"][:, :, :4]
    future = batch["agents"][:, :, 4:]
    targets = inverse_holonomic_actions(history, future, batch["agent_mask"])
    scene = model.encode_scene(
        history=history,
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        current_lights=batch["lights"][:, 3],
        current_light_mask=batch["light_mask"][:, 3],
    )
    assert scene.agent_tokens.shape == (2, 3, 32)
    normalizer = ActionNormalizer(torch.zeros(16, 3), torch.ones(16, 3))
    normalized = normalizer.normalize(targets.actions, targets.agent_type)
    loss, metrics = flow_matching_loss(
        model, scene, normalized, targets.valid, focus_index=0
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["valid_action_fraction"].item() > 0
    assert model.velocity_head[-1].weight.grad is not None


def test_huber_flow_loss_bounds_extreme_target_penalty() -> None:
    torch.manual_seed(7)
    batch = _synthetic_batch()
    model = _small_model()
    history = batch["agents"][:, :, :4]
    future = batch["agents"][:, :, 4:]
    targets = inverse_holonomic_actions(history, future, batch["agent_mask"])
    scene = model.encode_scene(
        history=history,
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        current_lights=batch["lights"][:, 3],
        current_light_mask=batch["light_mask"][:, 3],
    )
    normalized = targets.actions.clone()
    normalized[:, 1, 0, 0] = 1_000.0
    torch.manual_seed(11)
    mse, _ = flow_matching_loss(
        model, scene, normalized, targets.valid, loss_type="mse"
    )
    torch.manual_seed(11)
    huber, _ = flow_matching_loss(
        model,
        scene,
        normalized,
        targets.valid,
        loss_type="huber",
        huber_beta=1.0,
    )
    assert torch.isfinite(huber)
    assert huber < mse


def test_sampling_keeps_known_focus_plan_exact() -> None:
    batch = _synthetic_batch(batch_size=1)
    model = _small_model()
    history = batch["agents"][:, :, :4]
    future = batch["agents"][:, :, 4:]
    targets = inverse_holonomic_actions(history, future, batch["agent_mask"])
    scene = model.encode_scene(
        history=history,
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        current_lights=batch["lights"][:, 3],
        current_light_mask=batch["light_mask"][:, 3],
    )
    model_mask = scene.agent_mask[:, :, None].expand(-1, -1, 6)
    sampled = model.sample_normalized_actions(
        scene, model_mask, targets.actions[:, 0], solver_steps=2
    )
    torch.testing.assert_close(sampled[:, 0], targets.actions[:, 0])


def test_sampling_without_focus_actions_generates_focus() -> None:
    batch = _synthetic_batch(batch_size=1)
    model = _small_model()
    history = batch["agents"][:, :, :4]
    scene = model.encode_scene(
        history=history,
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        current_lights=batch["lights"][:, 3],
        current_light_mask=batch["light_mask"][:, 3],
    )
    model_mask = scene.agent_mask[:, :, None].expand(-1, -1, 6)
    first = model.sample_normalized_actions(
        scene,
        model_mask,
        focus_actions=None,
        solver_steps=1,
        generator=torch.Generator().manual_seed(1),
    )
    second = model.sample_normalized_actions(
        scene,
        model_mask,
        focus_actions=None,
        solver_steps=1,
        generator=torch.Generator().manual_seed(2),
    )
    assert not torch.equal(first[:, 0], second[:, 0])


def test_unconditioned_flow_loss_supervises_focus() -> None:
    batch = _synthetic_batch(batch_size=1)
    model = _small_model()
    history = batch["agents"][:, :, :4]
    future = batch["agents"][:, :, 4:]
    targets = inverse_holonomic_actions(history, future, batch["agent_mask"])
    scene = model.encode_scene(
        history=history,
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        current_lights=batch["lights"][:, 3],
        current_light_mask=batch["light_mask"][:, 3],
    )
    focus_only_valid = torch.zeros_like(targets.valid)
    focus_only_valid[:, 0] = targets.valid[:, 0]
    conditioned_loss, _ = flow_matching_loss(
        model,
        scene,
        targets.actions,
        focus_only_valid,
        condition_focus_actions=True,
    )
    generated_loss, metrics = flow_matching_loss(
        model,
        scene,
        targets.actions,
        focus_only_valid,
        condition_focus_actions=False,
    )
    torch.testing.assert_close(conditioned_loss, torch.tensor(0.0))
    assert generated_loss.item() > 0.0
    assert metrics["valid_action_fraction"].item() > 0.0


def test_receding_rollout_generate_h_execute_b_shape() -> None:
    batch = _synthetic_batch(batch_size=1, history=4, horizon=8)
    model = _small_model(history=4, horizon=6)
    normalizer = ActionNormalizer(torch.zeros(16, 3), torch.ones(16, 3))
    initial_history = batch["agents"][:, :, :4]
    future = batch["agents"][:, :, 4:]
    targets = inverse_holonomic_actions(
        initial_history, future, batch["agent_mask"]
    )
    poses = rollout_receding_horizon(
        model,
        normalizer,
        initial_history=initial_history,
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        current_light_sequence=batch["lights"][:, 3 : 3 + 8],
        current_light_mask_sequence=batch["light_mask"][:, 3 : 3 + 8],
        focus_action_sequence=targets.actions[:, 0],
        focus_action_valid=targets.valid[:, 0],
        rollout_steps=8,
        commitment=2,
        solver_steps=1,
    )
    assert poses.shape == (1, 3, 8, 3)
    assert torch.isfinite(poses).all()


def test_receding_rollout_can_generate_all_agents() -> None:
    batch = _synthetic_batch(batch_size=1, history=4, horizon=8)
    model = _small_model(history=4, horizon=6)
    normalizer = ActionNormalizer(torch.zeros(16, 3), torch.ones(16, 3))
    poses = rollout_receding_horizon(
        model,
        normalizer,
        initial_history=batch["agents"][:, :, :4],
        agent_mask=batch["agent_mask"],
        map_polylines=batch["map_polylines"],
        map_mask=batch["map_mask"],
        current_light_sequence=batch["lights"][:, 3 : 3 + 8],
        current_light_mask_sequence=batch["light_mask"][:, 3 : 3 + 8],
        focus_action_sequence=None,
        focus_action_valid=None,
        rollout_steps=8,
        commitment=2,
        solver_steps=1,
    )
    assert poses.shape == (1, 3, 8, 3)
    assert torch.isfinite(poses).all()
