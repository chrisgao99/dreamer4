import argparse
import json
import math
from types import SimpleNamespace
import pytest
import torch
from waymo.training.world_model.action_kinematics import (
    KinematicsConfig, execute_step, execute_actions, inverse_actions, estimate_rho,
)
from waymo.training.world_model.direct_action_flow import execute_holonomic_actions, rollout_receding_horizon
from waymo.training.world_model.train_waymo_direct_action_flow import (
    build_arg_parser, create_model, resolve_kinematics_args, load_or_compute_action_statistics,
)
from waymo.training.world_model.mon_long_rollout import generate_poses

CFG = KinematicsConfig('type_aware', .3, .2, .25)


def test_turn_matches_paper_and_lateral_is_ignored_only_for_wheeled():
    pose = torch.zeros(1, 3, 3, dtype=torch.float64)
    types = torch.tensor([[1, 2, 3]])
    lengths = torch.tensor([[4., .8, 2.]], dtype=torch.float64)
    action = torch.tensor([[[2., 7., .4]]]*3, dtype=torch.float64).transpose(0, 1)
    actual = execute_step(pose, action, types, lengths, CFG)
    # Independent closed-form integration for no-slip reference point + offset swing.
    theta = .4
    expected_vehicle = torch.tensor([2*torch.sin(torch.tensor(theta))/theta + 1.2*(torch.cos(torch.tensor(theta))-1),
                                     2*(1-torch.cos(torch.tensor(theta)))/theta + 1.2*torch.sin(torch.tensor(theta)), theta],dtype=torch.float64)
    torch.testing.assert_close(actual[0, 0], expected_vehicle, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[0, 1], action[0, 1])
    modified = action.clone();modified[..., 1] = -99
    changed = execute_step(pose, modified, types, lengths, CFG)
    torch.testing.assert_close(actual[:, [0, 2]], changed[:, [0, 2]])
    assert not torch.allclose(actual[:, 1], changed[:, 1])


def test_straight_limit_gradients_and_rotation_equivariance():
    pose = torch.tensor([[[0., 0., .7]]], dtype=torch.float64)
    types = torch.tensor([[1]]);lengths = torch.tensor([[4.]],dtype=torch.float64)
    actions = torch.tensor([[[2., 9., 0.]]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a: execute_step(pose,a,types,lengths,CFG), (actions,))
    out = execute_step(pose, actions, types, lengths, CFG)
    torch.testing.assert_close(out[0,0,:2], torch.tensor([2*math.cos(.7), 2*math.sin(.7)], dtype=torch.float64))
    out.sum().backward();assert actions.grad[...,1].item() == 0
    assert torch.isfinite(actions.grad).all()


def make_history_future(actions, types, lengths):
    start = torch.zeros(*types.shape,3,dtype=actions.dtype)
    poses = execute_actions(start, actions, agent_type=types,agent_lengths=lengths,config=CFG)
    history = torch.zeros(*types.shape, 2, 8, dtype=actions.dtype)
    history[...,5] = 1;history[...,7] = types[...,None]
    future = torch.zeros(*types.shape,actions.shape[-2],8,dtype=actions.dtype)
    future[...,:2] = poses[...,:2];future[...,6] = poses[...,2]
    future[...,5] = 1;future[...,7] = types[...,None]
    return history,future


def test_recursive_inverse_roundtrip_and_unused_channel():
    types=torch.tensor([[1,2,3]]);lengths=torch.tensor([[4.,.8,2.]],dtype=torch.float64)
    actions=torch.randn(1,3,8,3,dtype=torch.float64)*.1
    actions[...,0] += .5
    history,future=make_history_future(actions,types,lengths)
    target=inverse_actions(history,future,torch.ones_like(types,dtype=torch.bool),agent_lengths=lengths,config=CFG)
    assert target.valid.all()
    out=execute_actions(target.current_pose,target.actions,target.valid,agent_type=types,agent_lengths=lengths,config=CFG)
    torch.testing.assert_close(out,target.future_pose)
    assert (target.actions[:,[0,2],:,1] == 0).all()


def test_inverse_reuses_executed_state_and_rejects_suffix():
    types=torch.tensor([[1]]);lengths=torch.tensor([[4.]])
    actions=torch.zeros(1,1,3,3);actions[...,0]=1
    history,future=make_history_future(actions,types,lengths)
    future[...,1] = torch.tensor([.1,.2,.3])
    t=inverse_actions(history,future,torch.ones_like(types,dtype=torch.bool),agent_lengths=lengths,config=CFG)
    assert t.valid.tolist() == [[[True,True,False]]]
    # If reset to GT instead of executed state, each residual would be .1 and pass.
    assert t.actions[0,0,2].eq(0).all()


def test_missing_geometry_fails_and_masked_nan_does_not_contaminate():
    types=torch.tensor([[1,3]]);lengths=torch.tensor([[4.,float('nan')]])
    pose=torch.zeros(1,2,3);actions=torch.ones(1,2,2,3)*.1
    valid=torch.tensor([[[True,True],[False,False]]])
    with pytest.raises(ValueError,match='requires agent_lengths'):
        execute_actions(pose,actions,valid,agent_type=types,config=CFG)
    actions[:,1] = float('nan')
    out=execute_actions(pose,actions,valid,agent_type=types,agent_lengths=lengths,config=CFG)
    assert torch.isfinite(out).all()
    assert out[:,1].eq(0).all()


def test_holonomic_default_is_exactly_backward_compatible():
    pose=torch.randn(2,3,3);actions=torch.randn(2,3,5,3)
    torch.testing.assert_close(execute_actions(pose,actions), execute_holonomic_actions(pose,actions),rtol=0,atol=0)


def test_calibration_recovers_signed_offsets_and_ignores_straight_data():
    types=torch.tensor([1]*10+[3]*10)
    lengths=torch.where(types==1,4.,2.).double()
    pose=torch.zeros(20,3,dtype=torch.float64)
    action=torch.zeros_like(pose);action[:,0]=.5;action[:,2]=torch.tensor([-.1,.1]*10)
    nxt=execute_step(pose,action,types,lengths,CFG)
    result=estimate_rho(pose,nxt,types,lengths,torch.ones(20,dtype=torch.bool),min_samples=5)
    assert result['vehicle_rho'] == pytest.approx(.3)
    assert result['cyclist_rho'] == pytest.approx(.2)
    with pytest.raises(ValueError,match='turning intervals'):
        estimate_rho(pose,pose,types,lengths,torch.ones(20,dtype=torch.bool),min_samples=1)


def test_config_and_stale_statistics_guard(tmp_path):
    with pytest.raises(ValueError,match='calibrated'):
        KinematicsConfig('type_aware')
    path=tmp_path/'old_stats.json';path.write_text(json.dumps({'mean':[], 'std':[]}))
    args=SimpleNamespace(action_kinematics=CFG.to_dict(),action_stats_path=str(path))
    with pytest.raises(ValueError,match='different kinematics'):
        load_or_compute_action_statistics(args)
    cal=tmp_path/'cal.json';cal.write_text(json.dumps({'vehicle_rho':.3,'cyclist_rho':.2}))
    args=build_arg_parser().parse_args(['--data_dir','unused','--val_data_dir','unused',
        '--ckpt_dir','unused','--action_stats_path',str(path),'--action_execution','type_aware',
        '--kinematics_calibration',str(cal),'--d_model','16','--n_heads','2',
        '--history_depth','1','--map_depth','1','--scene_depth','1','--action_depth','1',
        '--step_refiner_depth','1'])
    resolve_kinematics_args(args)
    assert create_model(args).kinematics == CFG
    restored=argparse.Namespace(**json.loads(json.dumps(vars(args))))
    cal.unlink()
    assert create_model(restored).kinematics == CFG


class Toy(torch.nn.Module):
    horizon=5
    def __init__(self):
        super().__init__();self.kinematics=CFG
        self.action=torch.nn.Parameter(torch.tensor([.5,9.,.1]))
    def encode_scene(self,**kw):return SimpleNamespace(agent_mask=kw['agent_mask'])
    def decode_velocity(self,x,t,scene,mask):return self.action.expand_as(x)-x
    def sample_normalized_actions(self,scene,mask,focus=None,**kw):return self.action.expand(*mask.shape,3)


class Identity:
    def denormalize(self,a,t):return a


def test_closed_loop_and_differentiable_rollout_use_same_executor():
    model=Toy();h=torch.zeros(1,2,11,8);h[...,5]=1;h[...,7]=torch.tensor([1,2])[None,:,None]
    batch=dict(agent_mask=torch.ones(1,2,dtype=torch.bool),agent_lengths=torch.tensor([[4.,.8]]),
        map_polylines=torch.zeros(1,1,1,6),map_mask=torch.ones(1,1,1,dtype=torch.bool),
        lights=torch.zeros(1,21,1,4),light_mask=torch.ones(1,21,1,dtype=torch.bool))
    diff=generate_poses(model,Identity(),h,batch,torch.tensor([10]),rollout_steps=10,
                        commitment=5,solver_steps=1,checkpoint_grad=True)
    actual=rollout_receding_horizon(model,Identity(),initial_history=h,agent_mask=batch['agent_mask'],
        agent_lengths=batch['agent_lengths'],map_polylines=batch['map_polylines'],map_mask=batch['map_mask'],
        current_light_sequence=batch['lights'][:,10:],current_light_mask_sequence=batch['light_mask'][:,10:],
        focus_action_sequence=None,focus_action_valid=None,rollout_steps=10,commitment=5,solver_steps=1)
    torch.testing.assert_close(diff,actual,atol=1e-5,rtol=1e-5)
    diff.sum().backward();assert torch.isfinite(model.action.grad).all()
    assert model.action.grad[2].abs()>0


def test_measured_lengths_preserve_agent_selection_and_padding():
    import numpy as np
    from waymo.core.waymo_vector_filter import _selected_agent_lengths, N_AGENTS_WAYMO
    raw={'state/current/length':np.arange(N_AGENTS_WAYMO,dtype=np.float32)+1}
    actual=_selected_agent_lengths(raw,np.array([3,1,-1]),np.array([True,True,False]))
    np.testing.assert_array_equal(actual,[4,2,0])


def test_preparation_pipeline_on_synthetic_npz(tmp_path):
    import numpy as np
    from waymo.core.waymo_vector_dataset import WaymoVectorDataset
    from waymo.training.world_model.prepare_action_kinematics import calibrate,compute_type_aware_statistics
    from waymo.training.world_model.train_waymo_direct_action_flow import prepare_batch, build_normalizer
    types=torch.tensor([[1,2,3]]);lengths=torch.tensor([[4.,.8,2.]])
    actions=torch.zeros(1,3,8,3);actions[...,0]=.5;actions[...,2]=.1
    h,f=make_history_future(actions,types,lengths)
    # h has two zero states. Remaining states follow physically exact turns.
    agents=torch.cat((h,f),dim=2)[0].numpy()
    np.savez(tmp_path/'sample.npz',agents=agents,agent_mask=np.ones(3,bool),agent_ids=np.arange(3),
        agent_lengths=lengths[0].numpy(),scenario_id='synthetic',
        map_polylines=np.zeros((1,2,6),np.float32),map_mask=np.ones((1,2),bool),map_ids=np.array([0]),
        lights=np.zeros((10,1,4),np.float32),light_mask=np.ones((10,1),bool),light_ids=np.array([0]),
        ego_origin_xy=np.zeros(2,np.float32),ego_heading=np.array(0,np.float32))
    config=SimpleNamespace(train_data=str(tmp_path),max_files=1,min_abs_yaw=.02,max_abs_yaw=.75,
        max_displacement_m=5.,min_samples=2,data_dir=str(tmp_path),stats_max_files=1,
        num_agent_types=16,history_length=2,horizon=4,physical_max_displacement_m=5.,
        physical_max_yaw_delta_rad=.75,action_kinematics=CFG.to_dict())
    result=calibrate(config)
    assert result['vehicle_rho'] == pytest.approx(.3,abs=1e-5)
    assert result['cyclist_rho'] == pytest.approx(.2,abs=1e-5)
    stats=compute_type_aware_statistics(config,CFG)
    assert stats['count'][1] > 0 and stats['action_kinematics'] == CFG.to_dict()
    assert stats['mean'][1][1] == 0 and stats['mean'][3][1] == 0
    item=WaymoVectorDataset(str(tmp_path))[0]
    batch={k:v[None] for k,v in item.items() if torch.is_tensor(v)}
    norm=build_normalizer(stats,torch.device('cpu'))
    prepared=prepare_batch(batch,norm,config,random_start=False)
    assert prepared.targets.valid.all() and torch.isfinite(prepared.normalized_actions).all()


def test_type_aware_recovery_target_and_real_network_gradient():
    from waymo.training.world_model.catk_flow_recovery import recovery_targets
    from waymo.training.world_model.direct_action_flow import ActionNormalizer, flow_matching_loss
    from waymo.training.world_model.tests.test_direct_action_flow import _small_model, _synthetic_batch
    model=_small_model();model.kinematics=CFG
    batch=_synthetic_batch();lengths=torch.tensor([[4.,.8,2.]]).expand(2,-1)
    history=batch['agents'][:,:,:4].clone();history[...,0:2]=0;history[...,6]=0
    actions=torch.zeros(2,3,6,3);actions[...,0]=.5;actions[...,2]=.1
    types=history[:,:,-1,7].long()
    gt=execute_actions(torch.zeros(2,3,3),actions,agent_type=types,agent_lengths=lengths,config=CFG)
    norm=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3))
    target,valid,_=recovery_targets(history,gt,torch.ones(2,3,6,dtype=torch.bool),batch['agent_mask'],
        norm,0,horizon=6,agent_lengths=lengths,config=CFG)
    assert valid.all()
    scene=model.encode_scene(history=history,agent_mask=batch['agent_mask'],map_polylines=batch['map_polylines'],
        map_mask=batch['map_mask'],current_lights=batch['lights'][:,3],current_light_mask=batch['light_mask'][:,3])
    loss,_=flow_matching_loss(model,scene,target,valid,condition_focus_actions=False)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_resume_cannot_reinterpret_old_checkpoint(tmp_path):
    from waymo.training.world_model.train_waymo_direct_action_flow import load_checkpoint
    path=tmp_path/'old.pt';torch.save({'args':{}},path)
    with pytest.raises(ValueError,match='different action kinematics'):
        load_checkpoint(path,model=SimpleNamespace(kinematics=CFG),ema=None,optimizer=None,scaler=None)
