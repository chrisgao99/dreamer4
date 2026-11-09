"""Lane graph conditioning, preserving the baseline flow objective/executor."""
import torch
from torch import nn
from waymo.adaptations.local_map_encoder import (
    AdaptiveSceneEncoder, LocalCross, gather_neighbors, nearest_neighbors,
)
from waymo.training.world_model.direct_action_flow import DirectActionFlowModel, SceneEncoding


def legal_path_neighbors(pose,map_pose,is_lane,edges,types,candidates=4,path_tokens=64,depth=16):
    """Associate current poses with candidate lanes and traverse directed paths.

    Search is scene-local and uses no ground truth future trajectory. Keep all
    discovered reachable nodes; bound the frontier width to path_tokens at each
    hop to avoid exponential branching. UNKNOWN/non-lane neighborhoods fall back
    to the unchanged geometric map context, rather than forced lane assignment.
    """
    b,n,_=pose.shape;m=map_pose.shape[1]
    k=min(candidates,m);cap=min(path_tokens,m)
    with torch.no_grad():
        distance=torch.cdist(pose[...,:2].float(),map_pose[...,:2].float())
        aligned=(pose[:,:,None,2]-map_pose[:,None,:,2]).cos()>0.
        eligible=is_lane[:,None]&aligned&(distance<=8.)
        score=(distance+4.*(1.-(pose[:,:,None,2]-map_pose[:,None,:,2]).cos())).masked_fill(~eligible,float('inf'))
        values,seeds=score.topk(k,dim=-1,largest=False)
        seed_valid=torch.isfinite(values)
        hops=pose.new_full((b,n,m),float('inf'))
        hops.scatter_reduce_(2,seeds,torch.where(seed_valid,torch.zeros_like(values),torch.full_like(values,float('inf'))),reduce='amin',include_self=True)
        frontier=seeds;frontier_valid=seed_valid
        batch=torch.arange(b,device=pose.device)[:,None,None]
        for step in range(1,depth+1):
            next_index=edges[batch,frontier]
            next_types=types[batch,frontier]
            good=(next_index>=0)&(next_types>0)&frontier_valid[...,None]
            safe=next_index.clamp_min(0)
            propagated=torch.full_like(safe,float(step),dtype=torch.float32)
            propagated=propagated.masked_fill(~good,float('inf'))
            hops.scatter_reduce_(2,safe.flatten(2),propagated.flatten(2),reduce='amin',include_self=True)
            new=(hops==step)
            scores=distance.masked_fill(~new,float('inf'))
            values,frontier=scores.topk(cap,dim=-1,largest=False)
            frontier_valid=torch.isfinite(values)
        reachable=torch.isfinite(hops)
        # Encode forward continuations even when a nearer parallel lane exists.
        ranked=(hops*.2+distance).masked_fill(~reachable,float('inf'))
        values,index=ranked.topk(cap,dim=-1,largest=False)
        mask=torch.isfinite(values)
        selected_hops=hops.gather(2,index).clamp_max(depth+1)
        selected_hops=torch.where(mask,selected_hops,torch.zeros_like(selected_hops))
    return index,mask,selected_hops


class ConnectivitySceneEncoder(AdaptiveSceneEncoder):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        d=kwargs['d_model'];heads=kwargs['n_heads']
        self.edge_bias=nn.Embedding(5,heads,padding_idx=0)
        self.path_hop=nn.Embedding(18,d)
        self.path_cross=LocalCross(d,heads,kwargs['dropout'])

    def forward(self,*,history,agent_mask,map_polylines,map_mask,current_lights,current_light_mask,
                map_ids,map_is_lane,map_stop_sign,map_stop_point,current_light_ids,
                lane_edges,lane_edge_types):
        tokens,alive,pose,agent_types=self.agent_encoder(history,agent_mask)
        maps,valid=self.map_encoder(map_polylines,map_mask)
        weights=map_mask.float();denom=weights.sum(-1).clamp_min(1)[...,None]
        center=(map_polylines[...,:2]*weights[...,None]).sum(2)/denom
        direction=(map_polylines[...,2:4]*weights[...,None]).sum(2)/denom
        map_pose=torch.cat((center,torch.atan2(direction[...,1],direction[...,0])[...,None]),-1).float()
        maps=maps+self.controls(map_pose,map_ids,map_is_lane,current_lights,current_light_mask,
                                current_light_ids,map_stop_sign,map_stop_point)
        maps=maps*valid[...,None].to(maps.dtype)
        idx,neighbor_mask=nearest_neighbors(map_pose,map_pose,valid,self.map_neighbors,self.map_radius_m)
        bias=self.map_bias(map_pose,gather_neighbors(map_pose,idx))
        # Typed directed topology relations inside the same sparse neighborhoods.
        matches=(idx[...,None]==lane_edges[:,:,None,:])&(lane_edges[:,:,None,:]>=0)
        edge_type=(matches.long()*lane_edge_types[:,:,None,:]).amax(-1)
        typed_bias=self.edge_bias(edge_type).permute(0,1,3,2).unsqueeze(-2)
        bias=bias+typed_bias.to(bias.dtype)
        for layer in self.map_layers:maps=layer(maps,valid,idx,neighbor_mask,bias)
        idx,local_mask=nearest_neighbors(pose,map_pose,valid,self.agent_map_neighbors,self.agent_map_radius_m)
        local_maps=gather_neighbors(maps,idx)
        local_bias=self.agent_map_bias(pose,gather_neighbors(map_pose,idx))
        path_idx,path_mask,hops=legal_path_neighbors(pose,map_pose,map_is_lane&valid,lane_edges,lane_edge_types)
        # Do not assign pedestrian tokens to road lane paths.
        path_mask=path_mask&alive[...,None]&((agent_types==1)|(agent_types==3))[...,None]
        path_maps=gather_neighbors(maps,path_idx)+self.path_hop(hops.long())
        path_bias=self.agent_map_bias(pose,gather_neighbors(map_pose,path_idx))
        tokens=self.path_cross(tokens,alive,path_maps,path_mask,path_bias)
        pair_bias=self.relative_bias(pose,agent_types)
        for cross,layer in zip(self.scene_cross,self.scene_self):
            tokens=cross(tokens,alive,local_maps,local_mask,local_bias)
            tokens=layer(tokens,alive,bias=pair_bias)
        tokens=self.final_norm(tokens)*alive[...,None].to(tokens.dtype)
        b,n,d=tokens.shape
        context=tokens[:,None].expand(-1,n,-1,-1);context_mask=alive[:,None].expand(-1,n,-1)
        memory=torch.cat((context,local_maps,path_maps),2)
        memory_mask=torch.cat((context_mask,local_mask,path_mask),2)
        cross_bias=torch.cat((pair_bias.permute(0,2,1,3).unsqueeze(-2),local_bias,path_bias),-1)
        empty=tokens[:,:0];empty_mask=alive[:,:0]
        return SceneEncoding(agent_tokens=tokens,agent_mask=alive,agent_pose=pose,agent_type=agent_types,
            map_tokens=maps,map_mask=valid,light_tokens=empty,light_mask=empty_mask,
            memory=tokens,memory_mask=alive,relative_bias=pair_bias,
            agent_memory=memory,agent_memory_mask=memory_mask,agent_memory_bias=cross_bias)


class ConnectivityFlowModel(DirectActionFlowModel):
    def __init__(self,**kwargs):
        if not kwargs.get('map_adaptation'):raise ValueError('Connectivity requires map adaptation')
        super().__init__(**kwargs)
        keys=('d_model','n_heads','history_depth','map_depth','scene_depth','history_length',
              'hidden_dim','dropout','mlp_ratio','position_scale_m','include_agent_velocity',
              'velocity_scale_mps','map_neighbors','agent_map_neighbors','map_radius_m','agent_map_radius_m')
        self.scene_encoder=ConnectivitySceneEncoder(**{key:kwargs[key] for key in keys})
        self.lane_connectivity=True
