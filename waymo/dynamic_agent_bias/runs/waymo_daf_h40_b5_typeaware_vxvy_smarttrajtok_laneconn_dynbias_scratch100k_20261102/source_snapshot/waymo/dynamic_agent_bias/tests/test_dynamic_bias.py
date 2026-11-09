"""Dynamic (per-chunk, self-conditioned) agent bias: equivalence and gradient checks.

Run: python waymo/dynamic_agent_bias/tests/test_dynamic_bias.py  (one GPU)
"""
import copy
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from waymo.dynamic_agent_bias.common import stage1_args  # noqa: E402
from waymo.lane_connectivity.dataset import ConnectivityDataset  # noqa: E402
from waymo.training.world_model import train_waymo_direct_action_flow as base  # noqa: E402
from waymo.training.world_model.direct_action_flow import execute_holonomic_actions, flow_matching_loss  # noqa: E402
from waymo.training.world_model.mon_long_rollout import generate_poses  # noqa: E402


def main():
    torch.manual_seed(0)
    device = torch.device('cuda')
    args = stage1_args('test_dynamic_bias')
    base.resolve_kinematics_args(args)
    stats = base.load_or_compute_action_statistics(args)
    normalizer = base.build_normalizer(stats, device)
    dyn = base.create_model(args).to(device).eval()
    dyn.set_action_normalizer(normalizer)
    plain_args = copy.copy(args); plain_args.dynamic_agent_bias = False
    plain = base.create_model(plain_args).to(device).eval()
    missing, unexpected = plain.load_state_dict(dyn.state_dict(), strict=False)
    assert not missing and all(k.startswith('future_agent_bias.') for k in unexpected), (missing, unexpected)

    data = ConnectivityDataset(args.data_dir, args.map_cache_dir, args.lane_topology_dir)
    batch = base.move_batch(base.collate_vector_batch([data[i] for i in (0, 7)]), device)
    prepared = base.prepare_batch(batch, normalizer, args, random_start=False)
    kwargs = base.scene_kwargs(batch, prepared)
    with torch.no_grad():
        scene_d, scene_p = dyn.encode_scene(**kwargs), plain.encode_scene(**kwargs)
        mask = scene_d.agent_mask[:, :, None].expand(-1, -1, dyn.horizon)
        x = torch.randn((*mask.shape, 3), device=device) * mask[..., None]
        t = torch.full((2,), 0.375, device=device)
        estimate = prepared.normalized_actions
        ref = plain.decode_velocity(x, t, scene_p, mask)
        # 1) zero-initialized output layer: identical with or without an estimate.
        for sc in (None, estimate):
            out = dyn.decode_velocity(x, t, scene_d, mask, self_condition=sc)
            assert torch.allclose(out, ref, atol=1e-5), float((out - ref).abs().max())
        # 2) trained-like weights: no estimate (or masked off) leaves outputs unchanged.
        torch.nn.init.normal_(dyn.future_agent_bias.mlp[-1].weight, std=0.5)
        torch.nn.init.normal_(dyn.future_agent_bias.mlp[-1].bias, std=0.5)
        off = dyn.decode_velocity(x, t, scene_d, mask)
        masked = dyn.decode_velocity(x, t, scene_d, mask, self_condition=estimate,
                                     self_condition_mask=torch.zeros(2, dtype=torch.bool, device=device))
        on = dyn.decode_velocity(x, t, scene_d, mask, self_condition=estimate)
        partial = dyn.decode_velocity(x, t, scene_d, mask, self_condition=estimate,
                                      self_condition_mask=torch.tensor([True, False], device=device))
    for name, value in (('off', off), ('masked', masked)):
        assert torch.allclose(value, ref, atol=1e-4), (name, float((value - ref).abs().max()))
    assert (on - ref).abs().max() > 1e-3, 'planned-pose bias has no effect'
    assert torch.allclose(partial[0], on[0], atol=1e-5) and torch.allclose(partial[1], ref[1], atol=1e-4)
    # 3) planned poses: integrating the GT actions reproduces the GT future at chunk ends.
    poses = dyn.planned_chunk_poses(prepared.normalized_actions, scene_d, prepared.targets.valid)
    # Vectorized integration == the step loop of execute_holonomic_actions.
    metric = normalizer.denormalize(prepared.normalized_actions.float().clamp(-5, 5), scene_d.agent_type)
    loop = execute_holonomic_actions(scene_d.agent_pose.float(), metric, prepared.targets.valid)
    loop = loop[:, :, dyn.chunk_size - 1::dyn.chunk_size]
    assert torch.allclose(poses[..., :2], loop[..., :2], atol=1e-3), float((poses - loop)[..., :2].abs().max())
    assert torch.allclose(torch.cos(poses[..., 2] - loop[..., 2]), torch.ones_like(poses[..., 2]), atol=1e-5)
    gt = prepared.targets.future_pose[:, :, dyn.chunk_size - 1::dyn.chunk_size]
    ok = prepared.targets.valid[:, :, dyn.chunk_size - 1::dyn.chunk_size]
    err = (poses[..., :2] - gt[..., :2]).norm(dim=-1)[ok]
    print('planned-pose error vs GT at chunk ends (holonomic, m): median %.3f p99 %.3f' % (
        err.median(), err.quantile(.99)))
    assert err.median() < 0.5
    # 4) training loss trains the new module; sampling and the MON rollout run.
    dyn.train()
    # The 50% self-conditioning draw can switch both examples off (zero gradient by
    # softmax invariance); use the batch with at least one example switched on.
    for seed in range(20):
        dyn.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        draw = torch.rand(10, device=device)  # flow_matching_loss: x0 uses randn, then rand(B) twice
        torch.manual_seed(seed)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, metrics = flow_matching_loss(dyn, dyn.encode_scene(**kwargs), prepared.normalized_actions,
                                               prepared.targets.valid, condition_focus_actions=False,
                                               flow_time_max=0.9, loss_type='huber')
        loss.backward()
        g = dyn.future_agent_bias.mlp[0].weight.grad
        if g is not None and g.abs().sum() > 0:
            break
    del draw
    grad = dyn.future_agent_bias.mlp[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0, 'no gradient into future_agent_bias'
    # Regression: the no-grad self-conditioning pass must not poison the autocast cache.
    missing = [n for n, q in dyn.named_parameters() if q.grad is None]
    assert not missing, f'{len(missing)} parameters without gradient, e.g. {missing[:3]}'
    dyn.zero_grad(set_to_none=True); dyn.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        sample = dyn.sample_normalized_actions(dyn.encode_scene(**kwargs), mask, solver_steps=8)
    assert torch.isfinite(sample).all()
    mon_args = copy.copy(args); mon_args.horizon = 80
    mprep = base.prepare_batch(batch, normalizer, mon_args, random_start=False)
    runs = []
    for _ in range(2):
        torch.manual_seed(5)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            poses = generate_poses(dyn, normalizer, mprep.history, batch, mprep.anchors, rollout_steps=10)
        runs.append(poses)
    runs[1].float().sum().backward()
    assert torch.equal(runs[0], runs[1]), 'MON rollout not reproducible under checkpointing'
    assert dyn.future_agent_bias.mlp[0].weight.grad.abs().sum() > 0
    missing = [n for n, q in dyn.named_parameters() if q.grad is None]
    assert not missing, f'MON rollout: {len(missing)} parameters without gradient'
    print('all dynamic agent bias tests passed', {k: round(float(v), 4) for k, v in metrics.items()})


if __name__ == '__main__':
    main()
