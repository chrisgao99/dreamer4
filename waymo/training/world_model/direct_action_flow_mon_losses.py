"""Scene-joint MoN motion losses and type-aware physical proxies for action flows."""
import torch
import torch.nn.functional as F
from waymo.training.world_model.rollout_physical_losses import _nearest_oriented_edge_signed_distance


def scene_mean(value, valid):
    while valid.ndim < value.ndim: valid=valid.unsqueeze(-1)
    weight=valid.expand_as(value).to(value.dtype)
    dims=tuple(range(1,value.ndim))
    return (value*weight).sum(dims)/weight.sum(dims).clamp_min(1)


def wrap(x): return torch.atan2(torch.sin(x),torch.cos(x))


def motion_distances(poses, gt, initial_pose, valid, dt=.1):
    """(B,N,T,3) -> per-scene distance, same joint winner for xy/v/yaw."""
    previous=torch.cat((initial_pose[:,:,None],poses[:,:,:-1]),2)
    gt_previous=torch.cat((initial_pose[:,:,None],gt[:,:,:-1]),2)
    v=(poses[...,:2]-previous[...,:2])/dt
    vgt=(gt[...,:2]-gt_previous[...,:2])/dt
    p_heading=torch.stack((poses[...,2].sin(),poses[...,2].cos()),-1)
    g_heading=torch.stack((gt[...,2].sin(),gt[...,2].cos()),-1)
    xy=scene_mean(F.smooth_l1_loss(poses[...,:2],gt[...,:2],reduction='none'),valid)
    velocity=scene_mean(F.smooth_l1_loss(v,vgt,reduction='none'),valid)
    yaw=scene_mean(F.smooth_l1_loss(p_heading,g_heading,reduction='none'),valid)
    return xy+.5*velocity+.5*yaw,dict(xy=xy,velocity=velocity,yaw=yaw)


def footprint(types):
    # NPZ carries no measured agent dimensions. Explicit type-specific proxies.
    length=torch.ones_like(types,dtype=torch.float32)
    width=torch.ones_like(length)
    for typ,l,w in ((1,4.8,2.),(2,.8,.8),(3,2.,.8)):
        length=torch.where(types==typ,l,length);width=torch.where(types==typ,w,width)
    return length,width


def collision_gap(poses,types):
    """Exact separating-axis gap for proxy oriented rectangles, four box axes."""
    xy=poses[...,:2].transpose(1,2)  # B,T,N,2
    yaw=poses[...,2].transpose(1,2)
    forward=torch.stack((yaw.cos(),yaw.sin()),-1)
    lateral=torch.stack((-yaw.sin(),yaw.cos()),-1)
    length,width=footprint(types)
    hl=length[:,None]/2;hw=width[:,None]/2
    delta=xy.unsqueeze(-3)-xy.unsqueeze(-2)
    gaps=[]
    for axes in (forward.unsqueeze(-2),lateral.unsqueeze(-2),forward.unsqueeze(-3),lateral.unsqueeze(-3)):
        projection=(delta*axes).sum(-1)
        # Choose a valid nonzero subgradient at perfect alignment so overlapping
        # collinear boxes still receive a separating force.
        center=torch.where(projection>=0,projection,-projection)
        ri=hl.unsqueeze(-1)*(forward.unsqueeze(-2)*axes).sum(-1).abs()+hw.unsqueeze(-1)*(lateral.unsqueeze(-2)*axes).sum(-1).abs()
        rj=hl.unsqueeze(-2)*(forward.unsqueeze(-3)*axes).sum(-1).abs()+hw.unsqueeze(-2)*(lateral.unsqueeze(-3)*axes).sum(-1).abs()
        gaps.append(center-ri-rj)
    return torch.stack(gaps,-1).max(-1).values


def road_geometry(poses,types,map_polylines,map_mask):
    n=poses.shape[1]
    length,width=footprint(types)
    yaw=poses[...,2]
    f=torch.stack((yaw.cos(),yaw.sin()),-1);l=torch.stack((-yaw.sin(),yaw.cos()),-1)
    signs=poses.new_tensor([[1,1],[1,-1],[-1,1],[-1,-1]])
    corners=poses[...,:2].unsqueeze(-2)
    corners=corners+signs[:,0,None]*length[:,:,None,None,None]/2*f.unsqueeze(-2)+signs[:,1,None]*width[:,:,None,None,None]/2*l.unsqueeze(-2)
    road_values=[];coverage=[]
    for bi in range(poses.shape[0]):
        p=map_polylines[bi];m=map_mask[bi].bool()
        typ=p[...,4].round().long();edge=m&((typ==15)|(typ==16))&(p[...,2:4].norm(dim=-1)>1e-6)
        coverage.append(bool(edge.any()))
        signed=_nearest_oriented_edge_signed_distance(corners[bi].reshape(-1,2),p[...,:2][edge],p[...,2:4][edge],query_chunk_size=1024)
        road_values.append(signed.reshape(n,poses.shape[2],4).max(-1).values)
    signed=torch.stack(road_values)
    covered=torch.tensor(coverage,device=poses.device,dtype=torch.bool)
    return signed,covered


@torch.no_grad()
def reliable_road_mask(gt,initial_pose,valid,types,map_polylines,map_mask,margin=.3):
    """Exclude GT/map-inconsistent footprint locations, with coverage reported.

    Cropped/sampled road edges cannot reliably sign every point. The reference
    footprint must be inside with margin at both the logged anchor and target.
    This mask depends only on data, never on predicted validity or positions.
    """
    signed,covered=road_geometry(gt,types,map_polylines,map_mask)
    anchor,_=road_geometry(initial_pose[:,:,None],types,map_polylines,map_mask)
    return valid&(signed<=-margin)&(anchor<=-margin)&covered[:,None,None]&(((types==1)|(types==3))[:,:,None])


def physical_losses(poses,initial_pose,valid,types,map_polylines,map_mask,
                    dt=.1,warning=1.,margin=.3,temperature=.2,road_support=None):
    gap=collision_gap(poses,types)
    vm=valid.transpose(1,2)
    n=valid.shape[1]
    pairs=vm.unsqueeze(-1)&vm.unsqueeze(-2)&torch.triu(torch.ones(n,n,device=poses.device,dtype=torch.bool),diagonal=1)
    collision=scene_mean(F.smooth_l1_loss(F.softplus((warning-gap)/temperature)*temperature,torch.zeros_like(gap),reduction='none'),pairs)
    signed,covered=road_geometry(poses,types,map_polylines,map_mask)
    road_agents=(types==1)|(types==3)
    road_mask=valid&road_agents[:,:,None]&covered[:,None,None]
    raw_road_mask=road_mask
    if road_support is not None: road_mask=road_mask&road_support
    penalty=F.softplus((signed+margin)/temperature)*temperature
    offroad=scene_mean(F.smooth_l1_loss(penalty,torch.zeros_like(penalty),reduction='none'),road_mask)
    previous=torch.cat((initial_pose[:,:,None],poses[:,:,:-1]),2)
    midpoint=previous[...,2]+.5*wrap(poses[...,2]-previous[...,2])
    displacement=(poses[...,:2]-previous[...,:2])/dt
    lateral_velocity=-midpoint.sin()*displacement[...,0]+midpoint.cos()*displacement[...,1]
    vehicle_mask=valid&road_agents[:,:,None]
    sideslip=scene_mean(F.smooth_l1_loss(lateral_velocity,torch.zeros_like(lateral_velocity),reduction='none'),vehicle_mask)
    return dict(collision=collision,offroad=offroad,sideslip=sideslip),dict(
        collision_overlap_rate_proxy=scene_mean((gap<0).float(),pairs),
        offroad_rate_proxy=scene_mean((signed>0).float(),road_mask),
        offroad_rate_proxy_unfiltered=scene_mean((signed>0).float(),raw_road_mask),
        road_reliable_fraction=scene_mean(road_mask.float(),raw_road_mask),
        road_edge_scene_coverage=covered.float(),
        lateral_speed_mae_mps=scene_mean(lateral_velocity.abs(),vehicle_mask))
