"""Physical diagnostics on the SAME joint rollouts used for ADE/CPD.

Ratios retain numerators/denominators, avoiding zero-denominator scene bias.
Geometry uses type-dependent proxy footprints, not official WOSAC scoring.
"""
import numpy as np
import torch
from waymo.training.world_model.direct_action_flow_mon_losses import collision_gap,road_geometry,reliable_road_mask,wrap

METRIC_NAMES=(
    'collision_pair_time_rate_proxy','collision_agent_time_rate_proxy','collision_scene_rollout_rate_proxy',
    'collision_warning_pair_time_rate_proxy','offroad_agent_time_rate_proxy_raw',
    'offroad_agent_time_rate_proxy_reliable','offroad_scene_rollout_rate_proxy_reliable',
    'road_reliable_coverage','road_edge_scene_coverage','lateral_speed_mean_abs_mps',
    'sideslip_angle_mean_abs_deg_moving','sideslip_gt10deg_rate_moving','reverse_motion_rate_moving',
    'speed_mean_mps','acceleration_mean_norm_mps2','jerk_mean_norm_mps3','yaw_rate_mean_abs_radps',
    'displacement_gt5m_rate','yaw_change_gt075rad_rate',
)
DEFINITIONS={
    'aggregation':'sum of numerators / sum of denominators across all scenes and all rollouts; null if no valid denominator',
    'geometry':'oriented-rectangle SAT using proxy sizes: vehicle 4.8x2m, pedestrian .8x.8m, cyclist 2x.8m, unknown 1x1m',
    'collision':'all valid agents; overlap when all four SAT axes overlap; warning when maximum separating-axis gap <1m',
    'road':'vehicle/cyclist corners against nearest retained oriented road-edge samples (types15/16); raw plus GT/map-reliable subset',
    'road_reliability':'logged footprint inside with .3m margin at initial anchor and corresponding future frame; mask fixed across models and samples',
    'motion':'finite differences dt=.1s; sideslip uses midpoint heading for vehicles/cyclists, moving angle/reverse metrics require speed>1m/s',
    'validity':'same physically-filtered logged transition mask as ADE; acceleration/jerk need 2/3 consecutive valid transitions',
    'scope':'all K rollouts, no best-candidate selection for physical metrics',
    'status':'physical proxy diagnostics, not official WOSAC metrics',
}


def _stats(value,mask):
    dims=tuple(range(1,value.ndim))
    weight=mask.expand_as(value)
    return torch.stack(((value.double()*weight).sum(dims),weight.double().sum(dims)),-1)


@torch.no_grad()
def one_rollout(poses,initial,valid,types,maps,map_mask,support,dt=.1):
    b,n,t,_=poses.shape
    gap=collision_gap(poses,types)
    vm=valid.transpose(1,2)
    pairs=vm.unsqueeze(-1)&vm.unsqueeze(-2)
    upper=torch.triu(torch.ones(n,n,device=poses.device,dtype=torch.bool),diagonal=1)
    pair_mask=pairs&upper
    overlap=(gap<0)&pair_mask
    agent_hit=(overlap|overlap.transpose(-1,-2)).any(-1)
    scene_has_pair=pair_mask.flatten(1).any(1)
    scene_hit=overlap.flatten(1).any(1)
    signed,covered=road_geometry(poses,types,maps,map_mask)
    road_agents=((types==1)|(types==3))[:,:,None]
    road_valid=valid&road_agents&covered[:,None,None]
    reliable=road_valid&support
    outside=signed>0
    previous=torch.cat((initial[:,:,None],poses[:,:,:-1]),2)
    displacement=poses[...,:2]-previous[...,:2]
    velocity=displacement/dt;speed=velocity.norm(dim=-1)
    dyaw=wrap(poses[...,2]-previous[...,2])
    heading=previous[...,2]+.5*dyaw
    lateral=-heading.sin()*velocity[...,0]+heading.cos()*velocity[...,1]
    longitudinal=heading.cos()*velocity[...,0]+heading.sin()*velocity[...,1]
    moving=valid&road_agents&(speed>1.)
    angle=torch.atan2(lateral.abs(),longitudinal.abs()).rad2deg()
    acceleration=(velocity[:,:,1:]-velocity[:,:,:-1])/dt
    av=valid[:,:,1:]&valid[:,:,:-1]
    jerk=(acceleration[:,:,1:]-acceleration[:,:,:-1])/dt
    jv=av[:,:,1:]&av[:,:,:-1]
    entries=[
        _stats(overlap,pair_mask),_stats(agent_hit,vm),
        _stats(scene_hit[:,None],scene_has_pair[:,None]),
        _stats(gap<1.,pair_mask),_stats(outside,road_valid),_stats(outside,reliable),
        _stats((outside&reliable).flatten(1).any(1)[:,None],reliable.flatten(1).any(1)[:,None]),
        _stats(reliable,road_valid),_stats(covered[:,None],torch.ones(b,1,device=poses.device,dtype=torch.bool)),
        _stats(lateral.abs(),valid&road_agents),_stats(angle,moving),_stats(angle>10.,moving),
        _stats(longitudinal<0,moving),_stats(speed,valid),_stats(acceleration.norm(dim=-1),av),
        _stats(jerk.norm(dim=-1),jv),_stats(dyaw.abs()/dt,valid),
        _stats(displacement.norm(dim=-1)>5,valid),_stats(dyaw.abs()>.75,valid),
    ]
    return torch.stack(entries,1)  # B,M,2


@torch.no_grad()
def evaluate_batch(poses,initial,gt,valid,types,maps,map_mask):
    support=reliable_road_mask(gt,initial,valid,types,maps,map_mask)
    generated=torch.stack([one_rollout(poses[:,k],initial,valid,types,maps,map_mask,support) for k in range(poses.shape[1])],1)
    reference=one_rollout(gt,initial,valid,types,maps,map_mask,support)[:,None]
    return generated.cpu().numpy(),reference.cpu().numpy()


def summarize(values):
    total=values.sum(axis=(0,1))
    return {name:float(total[i,0]/total[i,1]) if total[i,1]>0 else None for i,name in enumerate(METRIC_NAMES)}
