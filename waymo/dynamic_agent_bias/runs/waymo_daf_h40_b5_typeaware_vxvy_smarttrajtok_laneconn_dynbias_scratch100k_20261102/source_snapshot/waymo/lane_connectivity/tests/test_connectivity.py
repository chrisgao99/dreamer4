import argparse
import torch
import numpy as np
from waymo.adaptations.scenario_reader import scenario_class
from waymo.lane_connectivity.prepare_topology import build_graph
from waymo.lane_connectivity.model import legal_path_neighbors
from waymo.training.world_model import train_waymo_direct_action_flow as base
from waymo.adaptations.tests.test_map_adaptation import adapted_batch,small_args
from waymo.training.world_model.direct_action_flow import ActionNormalizer,rollout_receding_horizon


def make_scene(marking=1):
    scene=scenario_class()();scene.scenario_id='lanes'
    for fid,y,xs in [(10,0.,[0.,5.,10.]),(20,3.,[0.,5.,10.]),(30,0.,[10.,15.])]:
        feature=scene.map_features.add();feature.id=fid;lane=feature.lane;lane.type=2
        for x in xs:point=lane.polyline.add();point.x=x;point.y=y
    lane=scene.map_features[0].lane;lane.exit_lanes.append(30)
    neighbor=lane.left_neighbors.add();neighbor.feature_id=20;neighbor.self_start_index=0;neighbor.self_end_index=2
    neighbor.neighbor_start_index=0;neighbor.neighbor_end_index=2
    boundary=neighbor.boundaries.add();boundary.lane_start_index=0;boundary.lane_end_index=2;boundary.boundary_type=marking
    return scene


def test_directed_successors_and_permitted_changes():
    ids=np.array([10,10,20,20,30]);is_lane=np.ones(5,bool)
    edge,types,stats=build_graph(make_scene(),ids,is_lane)
    links={i:dict(zip(edge[i][edge[i]>=0],types[i][edge[i]>=0])) for i in range(5)}
    assert links[0][1]==1 and links[1][4]==2
    assert links[0][2]==3 and links[1][3]==3
    assert 0 not in links[1] and 1 not in links[4] # no invented reverse edges
    assert stats['left_change']==2


def test_solid_unknown_and_partial_spans_disallow_changes():
    ids=np.array([10,10,20,20,30]);lane=np.ones(5,bool)
    for marking in (0,2,3,6,7,8,15):
        _,types,_=build_graph(make_scene(marking),ids,lane)
        assert not np.isin(types,[3,4]).any()
    scene=make_scene(1);scene.map_features[0].lane.left_neighbors[0].self_end_index=1
    _,types,_=build_graph(scene,ids,lane)
    assert np.count_nonzero(types==3)==1


def test_paths_include_successors_exclude_reverse_and_unconnected_lanes():
    pose=torch.tensor([[[0.,0.,0.],[0.,0.,3.14159265]]])
    maps=torch.tensor([[[0.,0.,0.],[5.,0.,0.],[10.,0.,0.],[0.,30.,0.]]])
    lane=torch.ones(1,4,dtype=torch.bool)
    edges=torch.tensor([[[0,1],[1,2],[2,-1],[3,-1]]]);types=torch.tensor([[[1,1],[1,2],[1,0],[1,0]]])
    index,mask,hops=legal_path_neighbors(pose,maps,lane,edges,types,candidates=1,path_tokens=4,depth=3)
    assert set(index[0,0][mask[0,0]].tolist())=={0,1,2}
    assert not mask[0,1].any()
    assert hops[0,0,mask[0,0]].max()==2


def test_connectivity_flow_gradients_checkpoint_and_rollout():
    b=adapted_batch();m=b['map_ids'].shape[1]
    edges=torch.arange(m)[None,:,None].repeat(1,1,2)
    edges[:,:,1]=torch.arange(m)[None]+1;edges[:,-1,1]=-1
    b['lane_edges']=edges;b['lane_edge_types']=(edges>=0).long()
    args=small_args();args.lane_connectivity=True;args.lane_topology_dir='dummy'
    model=base.create_model(args);norm=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3))
    prepared=base.prepare_batch(b,norm,args,random_start=False)
    scene=model.encode_scene(**base.scene_kwargs(b,prepared))
    loss,_=base.flow_matching_loss(model,scene,prepared.normalized_actions,prepared.targets.valid,condition_focus_actions=False)
    loss.backward();assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    restored=base.create_model(argparse.Namespace(**vars(args)));restored.load_state_dict(model.state_dict(),strict=True)
    model.eval()
    poses=rollout_receding_horizon(model,norm,initial_history=b['agents'][:,:,:4],agent_mask=b['agent_mask'],
        map_polylines=b['map_polylines'],map_mask=b['map_mask'],map_ids=b['map_ids'],map_is_lane=b['map_is_lane'],
        map_stop_sign=b['map_stop_sign'],map_stop_point=b['map_stop_point'],lane_edges=b['lane_edges'],lane_edge_types=b['lane_edge_types'],
        light_id_sequence=b['light_ids'][:,3:7],current_light_sequence=b['lights'][:,3:7],
        current_light_mask_sequence=b['light_mask'][:,3:7],focus_action_sequence=None,focus_action_valid=None,
        rollout_steps=4,commitment=2,solver_steps=1)
    assert poses.shape==(1,3,4,3) and torch.isfinite(poses).all()


def test_matched_cuda0_configuration():
    import json
    from pathlib import Path
    from waymo.lane_connectivity.run_stage1 import training_args
    saved=json.loads(Path('waymo/lane_connectivity/baseline_cuda0_args.json').read_text())
    args=training_args('/tmp/topology','new_lane_run')
    changed={key for key,val in saved.items() if vars(args)[key]!=val}
    assert changed=={'ckpt_dir','wandb_run_name'}
    assert args.lane_connectivity and args.resume is None and not args.balance_dynamic_loss
