"""Differentiable map losses on a single flow endpoint's H-step trajectory.

No generated state is fed back into the model. Discrete associations/permission
checks are detached; metric distance/yaw penalties keep gradients to actions.
"""
import math
import torch
import torch.nn.functional as F
from waymo.adaptations.local_map_encoder import gather_neighbors,nearest_neighbors
from waymo.training.world_model.action_kinematics import execute_model_actions
from waymo.training.world_model.direct_action_flow import wrap_angle_rad


def masked_mean(values,mask):
    return torch.where(mask,values,torch.zeros_like(values)).sum()/mask.sum().clamp_min(1)


def map_poses(batch):
    poly=batch['map_polylines'].float();weights=batch['map_mask'].float()
    denominator=weights.sum(-1).clamp_min(1)[...,None]
    center=(poly[...,:2]*weights[...,None]).sum(2)/denominator
    direction=(poly[...,2:4]*weights[...,None]).sum(2)/denominator
    return torch.cat((center,torch.atan2(direction[...,1],direction[...,0])[...,None]),-1)


def point_line_distances(points,polylines):
    # points B,Q,R,2; polylines B,Q,K,P,2 -> distances B,Q,R,K,P-1.
    start=polylines[:,:,:,:-1];delta=polylines[:,:,:,1:]-start
    offset=points[:,:,:,None,None]-start[:,:,None]
    square=delta.square().sum(-1).clamp_min(1e-8)
    fraction=(offset*delta[:,:,None]).sum(-1)/square[:,:,None]
    projected=start[:,:,None]+fraction.clamp(0,1)[...,None]*delta[:,:,None]
    residual=points[:,:,:,None,None]-projected
    distance=torch.linalg.vector_norm(residual,dim=-1)
    signed=delta[:,:,None,...,0]*residual[...,1]-delta[:,:,None,...,1]*residual[...,0]
    heading=torch.atan2(delta[...,1],delta[...,0])
    return distance,signed,heading


def assign_lanes(pose,batch,map_pose,lane_mask):
    """Spatial closest centerline, independent of predicted heading."""
    with torch.no_grad():
        idx,mask=nearest_neighbors(pose.detach(),map_pose,lane_mask,8,float('inf'))
        lines=gather_neighbors(batch['map_polylines'][...,:2].float(),idx)
        distances,_,heading=point_line_distances(pose.detach()[...,:2].unsqueeze(2),lines)
        distances=distances.squeeze(2).masked_fill(~mask[...,None],1e6)
        values,flat=distances.flatten(2).min(-1)
        piece_count=lines.shape[3]-1
        lane_idx=idx.gather(2,(flat//piece_count)[...,None]).squeeze(-1)
        theta=heading.flatten(2).gather(2,flat[...,None]).squeeze(-1)
        valid=mask.any(-1)&(values<=8.)
    return lane_idx,theta,values,valid


def footprint_loss(poses,batch,map_pose,valid,types):
    b,n,h,_=poses.shape;q=n*h;flat=poses.reshape(b,q,3)
    road=(batch['map_is_lane']&batch['map_mask'].any(-1)&
          ((batch['map_polylines'][:,:,0,4]>=1)&(batch['map_polylines'][:,:,0,4]<=2)))
    idx,mask=nearest_neighbors(flat.detach(),map_pose,road,8,float('inf'))
    lines=gather_neighbors(batch['map_polylines'][...,:2].float(),idx)
    widths=gather_neighbors(batch['lane_half_widths'].float(),idx)
    supported=gather_neighbors(batch['lane_width_mask'][...,None],idx).squeeze(-1)&mask
    lengths=batch['agent_lengths'][:,:,None].expand(-1,-1,h).reshape(b,q)
    body_width=batch['agent_widths'][:,:,None].expand(-1,-1,h).reshape(b,q)
    lengths=torch.where(torch.isfinite(lengths)&(lengths>0),lengths,torch.zeros_like(lengths))
    body_width=torch.where(torch.isfinite(body_width)&(body_width>0),body_width,torch.zeros_like(body_width))
    # Center, four corners, and four edge midpoints.
    offsets=poses.new_tensor([[0,0],[.5,.5],[.5,-.5],[-.5,.5],[-.5,-.5],
                             [.5,0],[-.5,0],[0,.5],[0,-.5]])
    local=offsets[None,None]*torch.stack((lengths,body_width),-1)[:,:,None]
    c=flat[...,2].cos()[:,:,None];s=flat[...,2].sin()[:,:,None]
    points=flat[:,:,:2][:,:,None]+torch.stack((c*local[...,0]-s*local[...,1],s*local[...,0]+c*local[...,1]),-1)
    distance,signed,_=point_line_distances(points,lines)
    half_width=torch.where(signed>=0,widths[:,:,None,:,None,0],widths[:,:,None,:,None,1])
    outside=(distance-half_width).clamp_min(0)
    outside=outside.masked_fill(~supported[:,:,None,:,None],1e4).flatten(3).min(-1).values
    # Skip an unannotated road region, even if an annotated lane is nearby.
    center_dist,_,_=point_line_distances(flat[:,:,:2].unsqueeze(2),lines)
    closest=center_dist.squeeze(2).masked_fill(~mask[...,None],1e6).flatten(2).argmin(-1)//(lines.shape[3]-1)
    support_center=supported.gather(2,closest[...,None]).squeeze(-1)
    active=(valid&(types[:,:,None]==1)&batch['agent_width_mask'][:,:,None]).reshape(b,q)&support_center
    cost=F.smooth_l1_loss(outside,torch.zeros_like(outside),reduction='none',beta=1.)
    loss=masked_mean(cost,active[:,:,None].expand_as(cost))
    return loss,active.sum()


def reachable_neighbors(source,next_pose,batch,map_pose,source_valid,cap=16,depth=3):
    b,q=source.shape;m=map_pose.shape[1];cap=min(cap,m)
    current=source[...,None];mask=source_valid[...,None]
    batch_index=torch.arange(b,device=source.device)[:,None,None]
    with torch.no_grad():
        for _ in range(depth):
            links=batch['lane_edges'][batch_index,current]
            kinds=batch['lane_edge_types'][batch_index,current]
            good=(links>=0)&(kinds>0)&mask[...,None]
            # Deduplicate before pruning: repeated hold edges must not crowd
            # out real successors or permitted lane changes.
            pool=torch.cat((current,links.clamp_min(0).flatten(2)),2)
            ok=torch.cat((mask,good.flatten(2)),2)
            sentinel=pool.masked_fill(~ok,m)
            ordered=sentinel.sort(-1).values
            unique=(ordered<m)&torch.cat((torch.ones_like(ordered[:,:,:1],dtype=torch.bool),ordered[:,:,1:]!=ordered[:,:,:-1]),-1)
            safe=ordered.clamp_max(m-1)
            neighbor_pose=gather_neighbors(map_pose,safe)
            scores=torch.linalg.vector_norm(neighbor_pose[...,:2]-next_pose[:,:,None,:2],dim=-1).masked_fill(~unique,float('inf'))
            values,selected=scores.topk(min(cap,scores.shape[-1]),dim=-1,largest=False)
            current=safe.gather(2,selected);mask=torch.isfinite(values)
    return current,mask


def map_trajectory_losses(poses,current_pose,valid,types,batch):
    valid=valid&torch.isfinite(poses).all(-1)
    poses=torch.where(valid[...,None],poses,torch.zeros_like(poses))
    current_pose=torch.where(torch.isfinite(current_pose).all(-1,keepdim=True),current_pose,torch.zeros_like(current_pose))
    b,n,h,_=poses.shape;q=n*h;map_pose=map_poses(batch)
    lane_mask=batch['map_is_lane']&batch['map_mask'].any(-1)
    sequence=torch.cat((current_pose[:,:,None],poses),2).reshape(b,n*(h+1),3)
    assignments,heading,distance,associated=assign_lanes(sequence,batch,map_pose,lane_mask)
    assignments=assignments.reshape(b,n,h+1);heading=heading.reshape(b,n,h+1)
    associated=associated.reshape(b,n,h+1)
    wheeled=(types==1)|(types==3)
    directional_mask=valid&wheeled[:,:,None]&associated[:,:,1:]
    angle=wrap_angle_rad(poses[...,2]-heading[:,:,1:])
    direction=masked_mean((angle/math.pi).square(),directional_mask)
    offroad,offroad_count=footprint_loss(poses,batch,map_pose,valid,types)
    source=assignments[:,:,:-1].reshape(b,q);dest=assignments[:,:,1:].reshape(b,q)
    transition_valid=(directional_mask&associated[:,:,:-1]).reshape(b,q)
    pool,pool_mask=reachable_neighbors(source,poses.detach().reshape(b,q,3),batch,map_pose,transition_valid)
    ids=batch['map_ids'];source_id=ids.gather(1,source);dest_id=ids.gather(1,dest)
    reachable_ids=gather_neighbors(ids[...,None],pool).squeeze(-1)
    legal=(source_id==dest_id)|((reachable_ids==dest_id[...,None])&pool_mask).any(-1)
    illegal=transition_valid&~legal
    lines=gather_neighbors(batch['map_polylines'][...,:2].float(),pool)
    costs,_,legal_heading=point_line_distances(poses.reshape(b,q,3)[...,:2].unsqueeze(2),lines)
    costs=costs.squeeze(2)
    yaw_error=wrap_angle_rad(poses.reshape(b,q,3)[...,2,None,None]-legal_heading)
    distance_cost=F.smooth_l1_loss((costs-.5).clamp_min(0),torch.zeros_like(costs),reduction='none',beta=1.)
    joint=distance_cost+(yaw_error/math.pi).square()
    joint=joint.masked_fill(~pool_mask[...,None],1e4).flatten(2).min(-1).values
    transition=masked_mean(torch.where(illegal,joint,torch.zeros_like(joint)),transition_valid)
    return dict(offroad_loss=offroad,lane_direction_loss=direction,illegal_lane_transition_loss=transition,
                illegal_transition_fraction=illegal.sum()/transition_valid.sum().clamp_min(1),
                offroad_valid_pose_count=offroad_count,direction_valid_pose_count=directional_mask.sum())


def endpoint_regularizer(model,normalizer,prepared,batch,args,step):
    """One-shot endpoint rollout from real history; no model re-encoding."""
    def regularize(endpoint):
        with torch.autocast(device_type=endpoint.device.type,enabled=False):
            actions=normalizer.denormalize(endpoint.float(),prepared.targets.agent_type)
            valid=prepared.targets.valid.clone()
            if args.condition_focus_actions:valid[:,0]=False
            current=prepared.targets.current_pose.float()
            current=torch.where(torch.isfinite(current).all(-1,keepdim=True),current,torch.zeros_like(current))
            poses=execute_model_actions(model,current,actions,
                    valid,agent_type=prepared.targets.agent_type,agent_lengths=batch['agent_lengths'])
            terms=map_trajectory_losses(poses,current,valid,
                                       prepared.targets.agent_type,batch)
            scale=min(1.,float(step+1)/max(1,args.aux_loss_warmup_steps))
            auxiliary=scale*(args.offroad_weight*terms['offroad_loss']+
                       args.lane_direction_weight*terms['lane_direction_loss']+
                       args.lane_transition_weight*terms['illegal_lane_transition_loss'])
        return auxiliary,{k:v.detach() for k,v in terms.items()}|{'map_aux_loss':auxiliary.detach(),'aux_loss_scale':endpoint.new_tensor(scale)}
    return regularize
