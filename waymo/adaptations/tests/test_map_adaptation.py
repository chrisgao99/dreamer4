import argparse
import math
import numpy as np
import torch
from waymo.adaptations.local_map_encoder import RelativeMapBias, nearest_neighbors, LaneControlEncoder
from waymo.adaptations.prepare_map_cache import metric_segments
from waymo.adaptations.run_stage1 import training_args
from waymo.training.world_model import train_waymo_direct_action_flow as base
from waymo.training.world_model.direct_action_flow import ActionNormalizer, rollout_receding_horizon
from waymo.training.world_model.tests.test_direct_action_flow import _synthetic_batch


def test_segments_preserve_length_and_tail():
    segments=metric_segments([[0,0],[12,0]],2)
    assert len(segments)==3
    np.testing.assert_allclose([x[-1,0]-x[0,0] for x in segments],[5,5,2])
    for a,b in zip(segments,segments[1:]):np.testing.assert_array_equal(a[-1,:2],b[0,:2])
    np.testing.assert_allclose(segments[-1][-1,:2],[12,0])
    assert len(metric_segments([[3,4],[3,4]],2))==1


def test_relative_geometry_is_rigid_transform_invariant():
    torch.manual_seed(1)
    bias=RelativeMapBias(4,16,100.)
    q=torch.tensor([[[1.,2.,.4]]]);k=torch.tensor([[[[4.,6.,1.2],[1.,2.,-.2]]]])
    original=bias(q,k)
    angle=.8;c=math.cos(angle);s=math.sin(angle)
    rotation=torch.tensor([[c,-s],[s,c]])
    qq=q.clone();kk=k.clone()
    qq[...,:2]=q[...,:2]@rotation.T+torch.tensor([14.,-20.])
    kk[...,:2]=k[...,:2]@rotation.T+torch.tensor([14.,-20.])
    qq[...,2]+=angle;kk[...,2]+=angle
    torch.testing.assert_close(bias(qq,kk),original,atol=1e-6,rtol=1e-5)
    assert not torch.allclose(bias(q,k+torch.tensor([3.,0.,.5])),original)


def test_neighbors_use_all_maps_and_handle_empty_radius():
    keys=torch.zeros(1,600,3);keys[0,:,0]=torch.arange(600.)
    mask=torch.ones(1,600,dtype=torch.bool)
    indices,valid=nearest_neighbors(torch.tensor([[[598.,0.,0.],[800.,0.,0.]]]),keys,mask,4,3.)
    assert indices[0,0,0]==598 and valid[0,0].all()
    assert not valid[0,1].any()
    mask[:]=False
    _,valid=nearest_neighbors(keys[:,:1],keys,mask,4,100.)
    assert not valid.any()


def test_controls_match_lane_id_and_ignore_masked_signals():
    torch.manual_seed(0)
    enc=LaneControlEncoder(32,16,100.).eval()
    pose=torch.zeros(1,2,3);ids=torch.tensor([[10,20]]);lane=torch.ones(1,2,dtype=torch.bool)
    lights=torch.tensor([[[4.,0.,4.,1.],[8.,0.,6.,1.]]]);lid=torch.tensor([[20,10]])
    mask=torch.ones(1,2,dtype=torch.bool);stop=torch.tensor([[True,False]]);point=torch.zeros(1,2,2)
    out=enc(pose,ids,lane,lights,mask,lid,stop,point)
    perm=torch.tensor([1,0])
    torch.testing.assert_close(out,enc(pose,ids,lane,lights[:,perm],mask[:,perm],lid[:,perm],stop,point))
    changed=lights.clone();changed[0,0,2]=1
    output=enc(pose,ids,lane,changed,mask,lid,stop,point)
    torch.testing.assert_close(out[:,0],output[:,0])
    assert not torch.allclose(out[:,1],output[:,1])
    mask[:]=False
    torch.testing.assert_close(enc(pose,ids,lane,lights,mask,lid,stop,point),
                               enc(pose,ids,lane,changed,mask,lid,stop,point))


def adapted_batch():
    b=_synthetic_batch(batch_size=1,history=4,horizon=8)
    m=b['map_polylines'].shape[1]
    b['map_ids']=torch.arange(m)[None]+10
    b['map_is_lane']=torch.ones(1,m,dtype=torch.bool)
    b['map_stop_sign']=torch.zeros(1,m,dtype=torch.bool)
    b['map_stop_point']=torch.zeros(1,m,2)
    b['light_ids']=torch.full(b['light_mask'].shape,10,dtype=torch.long)
    b['map_adaptation']=torch.tensor([True])
    return b


def small_args():
    args=base.build_arg_parser().parse_args(['--data_dir','a','--val_data_dir','b','--ckpt_dir','c',
        '--action_stats_path','d','--d_model','32','--n_heads','4','--hidden_dim','16',
        '--history_length','4','--horizon','6','--commitment','2','--history_depth','1',
        '--map_depth','1','--scene_depth','1','--action_depth','1','--step_refiner_depth','1',
        '--dropout','0','--include_agent_velocity','--map_adaptation'])
    return args


def test_adapted_flow_backward_checkpoint_and_empty_map():
    b=adapted_batch();args=small_args();model=base.create_model(args)
    norm=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3))
    prepared=base.prepare_batch(b,norm,args,random_start=False)
    for empty in (False,True):
        if empty:b['map_mask'][:]=False
        scene=model.encode_scene(**base.scene_kwargs(b,prepared))
        loss,_=base.flow_matching_loss(model,scene,prepared.normalized_actions,prepared.targets.valid,
                                      condition_focus_actions=False)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.zero_grad(set_to_none=True)
    restored=base.create_model(argparse.Namespace(**vars(args)))
    restored.load_state_dict(model.state_dict(),strict=True)
    assert restored.map_adaptation


def test_rollout_recomputes_geometry_from_executed_poses():
    b=adapted_batch();model=base.create_model(small_args()).eval()
    queries=[]
    hook=model.scene_encoder.agent_map_bias.register_forward_pre_hook(
        lambda module,args:queries.append(args[0].detach().clone()))
    normalizer=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3))
    poses=rollout_receding_horizon(model,normalizer,initial_history=b['agents'][:,:,:4],
        agent_mask=b['agent_mask'],map_polylines=b['map_polylines'],map_mask=b['map_mask'],
        map_ids=b['map_ids'],map_is_lane=b['map_is_lane'],map_stop_sign=b['map_stop_sign'],
        map_stop_point=b['map_stop_point'],light_id_sequence=b['light_ids'][:,3:7],
        current_light_sequence=b['lights'][:,3:7],current_light_mask_sequence=b['light_mask'][:,3:7],
        focus_action_sequence=None,focus_action_valid=None,rollout_steps=4,commitment=2,solver_steps=1)
    hook.remove()
    assert len(queries)==2
    torch.testing.assert_close(queries[1],poses[:,:,1])


def test_launch_matches_baseline_except_adaptation_and_outputs():
    import json
    from pathlib import Path
    payload=json.loads(Path('waymo/adaptations/baseline_launch_config.json').read_text())
    original=base.build_arg_parser().parse_args(payload['train_argv'])
    adapted=training_args('/tmp/cache','new_run')
    changed={k for k in vars(original) if getattr(original,k)!=getattr(adapted,k)}
    assert changed=={'ckpt_dir','wandb_run_name','map_adaptation','map_cache_dir'}
    assert adapted.resume is None and not adapted.balance_dynamic_loss


def test_future_light_states_do_not_enter_current_condition():
    b=adapted_batch();args=small_args();model=base.create_model(args).eval()
    normalizer=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3))
    prepared=base.prepare_batch(b,normalizer,args,random_start=False)
    original=model.encode_scene(**base.scene_kwargs(b,prepared)).agent_tokens
    anchor=int(prepared.anchors[0])
    b['lights'][:,anchor+1:,:,2]=7
    b['light_mask'][:,anchor+1:]=True
    changed=base.prepare_batch(b,normalizer,args,random_start=False)
    torch.testing.assert_close(original,model.encode_scene(**base.scene_kwargs(b,changed)).agent_tokens,
                               rtol=0,atol=0)


def test_padded_map_tokens_do_not_change_local_context():
    b=adapted_batch();args=small_args();model=base.create_model(args).eval()
    normalizer=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3))
    prepared=base.prepare_batch(b,normalizer,args,random_start=False)
    original=model.encode_scene(**base.scene_kwargs(b,prepared)).agent_tokens
    for key in ['map_polylines','map_mask','map_ids','map_is_lane','map_stop_sign','map_stop_point']:
        value=b[key];shape=list(value.shape);shape[1]=3
        fill=-1 if key=='map_ids' else 0
        b[key]=torch.cat((value,value.new_full(shape,fill)),1)
    torch.testing.assert_close(original,model.encode_scene(**base.scene_kwargs(b,prepared)).agent_tokens,
                               rtol=1e-5,atol=1e-6)
