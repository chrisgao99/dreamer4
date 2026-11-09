import math
import numpy as np
import torch
from waymo.lane_losses.losses import map_trajectory_losses
from waymo.lane_losses.prepare_geometry import lane_profile
from waymo.adaptations.scenario_reader import scenario_class


def straight_batch():
    maps=torch.zeros(1,4,6,6)
    for i,(start,y) in enumerate([(0,0),(5,0),(0,3.5),(5,3.5)]):
        maps[0,i,:,0]=torch.linspace(start,start+5,6);maps[0,i,:,1]=y;maps[0,i,:,2]=1;maps[0,i,:,4:]=1
    return dict(map_polylines=maps,map_mask=torch.ones(1,4,6,dtype=torch.bool),map_is_lane=torch.ones(1,4,dtype=torch.bool),
                map_ids=torch.tensor([[10,10,20,20]]),lane_half_widths=torch.ones(1,4,2)*1.75,lane_width_mask=torch.ones(1,4,dtype=torch.bool),
                agent_lengths=torch.tensor([[4.]]),agent_widths=torch.tensor([[2.]]),agent_width_mask=torch.ones(1,1,dtype=torch.bool),
                lane_edges=torch.tensor([[[0,1],[1,-1],[2,3],[3,-1]]]),lane_edge_types=torch.tensor([[[1,1],[1,0],[1,1],[1,0]]]))


def test_legal_onroad_aligned_trajectory_has_zero_losses():
    b=straight_batch();pose=torch.tensor([[[[3.,0.,0.],[6.,0.,0.]]]],requires_grad=True)
    loss=map_trajectory_losses(pose,torch.tensor([[[2.,0.,0.]]]),torch.ones(1,1,2,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),b)
    for key in ['offroad_loss','lane_direction_loss','illegal_lane_transition_loss']:assert loss[key]==0,key


def test_footprint_penalizes_corners_even_when_center_is_onroad():
    b=straight_batch();b['agent_widths'][:]=5.
    pose=torch.tensor([[[[3.,0.,0.]]]],requires_grad=True)
    loss=map_trajectory_losses(pose,torch.tensor([[[2.,0.,0.]]]),torch.ones(1,1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),b)
    assert loss['offroad_loss']>0
    loss['offroad_loss'].backward();assert torch.isfinite(pose.grad).all()
    pose2=torch.tensor([[[[3.,-3.,0.]]]],requires_grad=True)
    losses=map_trajectory_losses(pose2,torch.tensor([[[2.,0.,0.]]]),torch.ones(1,1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),b)
    losses['offroad_loss'].backward();assert pose2.grad[...,1].abs().sum()>0


def test_wrong_way_direction_penalty_has_yaw_gradient():
    b=straight_batch();pose=torch.tensor([[[[3.,0.,math.pi-.01]]]],requires_grad=True)
    terms=map_trajectory_losses(pose,torch.tensor([[[2.,0.,0.]]]),torch.ones(1,1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),b)
    assert terms['lane_direction_loss']>.9
    terms['lane_direction_loss'].backward();assert pose.grad[...,2].abs().sum()>0


def test_illegal_transition_penalty_has_gradient_and_respects_legal_edge():
    b=straight_batch();pose=torch.tensor([[[[3.,3.5,0.]]]],requires_grad=True)
    args=(torch.tensor([[[2.,0.,0.]]]),torch.ones(1,1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),b)
    terms=map_trajectory_losses(pose,*args)
    assert terms['illegal_transition_fraction']==1 and terms['illegal_lane_transition_loss']>0
    terms['illegal_lane_transition_loss'].backward();assert pose.grad[...,1].abs().sum()>0
    b['lane_edges'][0,0,1]=2;b['lane_edge_types'][0,0,1]=3
    terms=map_trajectory_losses(pose,*args)
    assert terms['illegal_transition_fraction']==0 and terms['illegal_lane_transition_loss']==0


def test_unknown_road_regions_and_invalid_agents_are_masked():
    b=straight_batch();b['lane_width_mask'][:]=False
    pose=torch.tensor([[[[3.,-4.,2.]]]],requires_grad=True)
    terms=map_trajectory_losses(pose,torch.tensor([[[2.,0.,0.]]]),torch.ones(1,1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),b)
    assert terms['offroad_loss']==0 and terms['offroad_valid_pose_count']==0
    terms=map_trajectory_losses(pose,torch.tensor([[[2.,0.,0.]]]),torch.zeros(1,1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),b)
    sum(terms[k] for k in ['offroad_loss','lane_direction_loss','illegal_lane_transition_loss']).backward()
    assert torch.isfinite(pose.grad).all() and pose.grad.abs().sum()==0


def test_boundary_widths_are_measured_and_missing_boundaries_stay_unknown():
    scene=scenario_class()();lane=scene.map_features.add();lane.id=10;lane.lane.type=2
    for x in [0.,5.,10.]:p=lane.lane.polyline.add();p.x=x;p.y=0
    for fid,y,name in [(100,1.8,'left_boundaries'),(200,-1.7,'right_boundaries')]:
        feature=scene.map_features.add();feature.id=fid;feature.road_line.type=2
        for x in [0.,10.]:p=feature.road_line.polyline.add();p.x=x;p.y=y
        boundary=getattr(lane.lane,name).add();boundary.boundary_feature_id=fid;boundary.lane_start_index=0;boundary.lane_end_index=2
    features={int(f.id):f for f in scene.map_features}
    widths,valid=lane_profile(lane.lane,features,2)
    assert valid.all();np.testing.assert_allclose(widths,[[1.8,1.7],[1.8,1.7]],atol=1e-5)
    del lane.lane.right_boundaries[:]
    widths,valid=lane_profile(lane.lane,features,2)
    assert not valid.any() and not widths.any()


def test_invalid_nan_observations_do_not_poison_gradients():
    b=straight_batch();b['agent_lengths'][:]=float('nan');b['agent_widths'][:]=float('nan');b['agent_width_mask'][:]=False
    poses=torch.full((1,1,1,3),float('nan'),requires_grad=True)
    terms=map_trajectory_losses(poses,torch.zeros(1,1,3),torch.zeros(1,1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long),b)
    loss=sum(terms[k] for k in ['offroad_loss','lane_direction_loss','illegal_lane_transition_loss'])
    assert loss==0
    loss.backward();assert torch.isfinite(poses.grad).all() and poses.grad.abs().sum()==0


def test_full_objective_adds_losses_without_extra_model_forward():
    from unittest.mock import patch
    from waymo.adaptations.tests.test_map_adaptation import adapted_batch,small_args
    from waymo.training.world_model import train_waymo_direct_action_flow as base
    from waymo.training.world_model.direct_action_flow import ActionNormalizer
    b=adapted_batch();m=b['map_ids'].shape[1];n=b['agent_mask'].shape[1]
    b['lane_edges']=torch.arange(m)[None,:,None];b['lane_edge_types']=torch.ones(1,m,1,dtype=torch.long)
    b['lane_half_widths']=torch.ones(1,m,2)*1.75;b['lane_width_mask']=torch.ones(1,m,dtype=torch.bool)
    b['agent_lengths']=torch.ones(1,n)*4;b['agent_widths']=torch.ones(1,n)*2;b['agent_width_mask']=torch.ones(1,n,dtype=torch.bool)
    args=small_args();args.lane_connectivity=True;args.map_aux_losses=True;args.offroad_weight=.1
    args.lane_direction_weight=.05;args.lane_transition_weight=.1;args.aux_loss_warmup_steps=5000;args.condition_focus_actions=False
    model=base.create_model(args);norm=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3))
    prepared=base.prepare_batch(b,norm,args,random_start=False);scene=model.encode_scene(**base.scene_kwargs(b,prepared))
    with patch.object(model,'decode_velocity',wraps=model.decode_velocity) as decoder:
        torch.manual_seed(5)
        loss,metrics=base.flow_matching_loss(model,scene,prepared.normalized_actions,prepared.targets.valid,
           condition_focus_actions=False,**base.map_aux_kwargs(model,norm,prepared,b,args,5000))
        assert decoder.call_count==1
    torch.testing.assert_close(loss,metrics['flow_loss']+metrics['map_aux_loss'])
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_loss_run_matches_cuda1_except_new_objective_and_outputs():
    import json
    from pathlib import Path
    from waymo.lane_losses.run_stage1 import training_args
    saved=json.loads(Path('waymo/lane_losses/baseline_cuda1_args.json').read_text())
    args=training_args('/tmp/geometry','loss_compare')
    differences={k for k,v in saved.items() if vars(args)[k]!=v}
    assert differences=={'ckpt_dir','wandb_run_name'}
    assert args.map_aux_losses and args.resume is None and args.lane_connectivity
