"""Sparse geometry-aware map conditioning for the stage-one action flow."""
import math
import torch
from torch import nn
from waymo.training.world_model.direct_action_flow import (
    AgentHistoryEncoder, PolylineEncoder, RelativeAgentBias, SceneEncoding,
    MultiheadAttention, FeedForward, MaskedTransformerBlock,
)


def gather_neighbors(values, indices):
    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    return values[batch, indices]


def nearest_neighbors(query_pose, key_pose, key_mask, count, radius):
    """Chunk distance search to avoid allocating a B x M x M matrix."""
    count = min(int(count), key_pose.shape[1])
    indices=[]; masks=[]
    with torch.no_grad():
        for start in range(0, query_pose.shape[1], 128):
            distance = torch.cdist(query_pose[:,start:start+128,:2].float(),key_pose[...,:2].float())
            distance = distance.masked_fill(~key_mask[:,None],float('inf'))
            dist, idx = distance.topk(count,dim=-1,largest=False,sorted=True)
            indices.append(idx); masks.append(torch.isfinite(dist) & (dist<=radius))
    return torch.cat(indices,1), torch.cat(masks,1)


class RelativeMapBias(nn.Module):
    """Distance, sin/cos bearing and sin/cos heading difference, directed i->j."""
    def __init__(self,heads,hidden,scale):
        super().__init__()
        self.scale=scale
        self.mlp=nn.Sequential(nn.Linear(5,hidden),nn.SiLU(),nn.Linear(hidden,heads))

    def forward(self,query_pose,neighbor_pose):
        delta=neighbor_pose[...,:2]-query_pose[:,:,None,:2]
        c=query_pose[...,2].cos()[:,:,None];s=query_pose[...,2].sin()[:,:,None]
        x=c*delta[...,0]+s*delta[...,1];y=-s*delta[...,0]+c*delta[...,1]
        distance=torch.linalg.vector_norm(delta,dim=-1)
        # Safe unit bearing at zero distance. All geometry is computed in FP32.
        denom=distance.clamp_min(1e-6)
        dyaw=neighbor_pose[...,2]-query_pose[:,:,None,2]
        features=torch.stack((distance/self.scale,y/denom,x/denom,dyaw.sin(),dyaw.cos()),-1)
        return self.mlp(features).permute(0,1,3,2).unsqueeze(-2) # B,Q,heads,1,K


class LocalCross(nn.Module):
    def __init__(self,dim,heads,dropout):
        super().__init__()
        self.q_norm=nn.LayerNorm(dim);self.kv_norm=nn.LayerNorm(dim)
        self.attn=MultiheadAttention(dim,heads,dropout,cross=True)

    def forward(self,query,query_mask,memory,mask,bias):
        b,q,k,d=memory.shape
        output=self.attn(self.q_norm(query).reshape(b*q,1,d),
            key_value=self.kv_norm(memory).reshape(b*q,k,d),
            query_mask=query_mask.reshape(b*q,1),key_mask=mask.reshape(b*q,k),
            bias=bias.reshape(b*q,bias.shape[2],1,k))
        # Attention's safe empty key is a numerical fallback only; an empty
        # neighborhood must contribute exactly zero, including projection bias.
        output=output.reshape(b,q,d)*mask.any(-1)[...,None].to(output.dtype)
        return (query+output)*query_mask[...,None].to(query.dtype)


class LocalMapBlock(nn.Module):
    def __init__(self,dim,heads,dropout,ratio):
        super().__init__()
        self.cross=LocalCross(dim,heads,dropout)
        self.norm=nn.LayerNorm(dim);self.ffn=FeedForward(dim,ratio,dropout)

    def forward(self,x,mask,idx,neighbor_mask,bias):
        x=self.cross(x,mask,gather_neighbors(x,idx),neighbor_mask,bias)
        return (x+self.ffn(self.norm(x)))*mask[...,None].to(x.dtype)


class LaneControlEncoder(nn.Module):
    def __init__(self,dim,hidden,scale):
        super().__init__();self.scale=scale
        self.state=nn.Embedding(16,hidden)
        # current signal presence, stop-sign flag, relative signal stop XY,
        # relative stop-sign XY (all in segment frame).
        self.mlp=nn.Sequential(nn.Linear(hidden+6,hidden),nn.SiLU(),nn.Linear(hidden,dim))

    def forward(self,pose,map_ids,is_lane,lights,light_mask,light_ids,stop_sign,stop_point):
        match=(map_ids[:,:,None]==light_ids[:,None,:]) & light_mask[:,None] & is_lane[:,:,None]
        present=match.any(-1)
        idx=match.long().argmax(-1)
        signal=lights.gather(1,idx[...,None].expand(-1,-1,4))
        signal=torch.where(present[...,None],signal,torch.zeros_like(signal))
        def relative(xy,valid):
            delta=xy-pose[...,:2];c=pose[...,2].cos();s=pose[...,2].sin()
            local=torch.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1]),-1)
            return torch.where(valid[...,None],local/self.scale,torch.zeros_like(local))
        control=torch.cat((present[...,None],stop_sign[...,None],
                           relative(signal[...,:2],present),relative(stop_point,stop_sign)),-1).float()
        state=signal[...,2].round().long().clamp(0,15)
        return self.mlp(torch.cat((self.state(state),control),-1))


class AdaptiveSceneEncoder(nn.Module):
    def __init__(self,*,d_model,n_heads,history_depth,map_depth,scene_depth,history_length,
                 hidden_dim,dropout,mlp_ratio,position_scale_m,include_agent_velocity,
                 velocity_scale_mps,map_neighbors=32,agent_map_neighbors=64,
                 map_radius_m=30.,agent_map_radius_m=100.):
        super().__init__()
        if min(map_neighbors,agent_map_neighbors)<1 or min(map_radius_m,agent_map_radius_m)<=0:
            raise ValueError('Map neighborhood settings must be positive')
        self.map_neighbors=map_neighbors;self.agent_map_neighbors=agent_map_neighbors
        self.map_radius_m=map_radius_m;self.agent_map_radius_m=agent_map_radius_m
        self.agent_encoder=AgentHistoryEncoder(d_model,n_heads,history_depth,history_length,dropout,
            mlp_ratio,position_scale_m,16,include_agent_velocity,velocity_scale_mps)
        self.map_encoder=PolylineEncoder(d_model,hidden_dim,64,position_scale_m)
        self.controls=LaneControlEncoder(d_model,hidden_dim,position_scale_m)
        self.map_bias=RelativeMapBias(n_heads,hidden_dim,position_scale_m)
        self.agent_map_bias=RelativeMapBias(n_heads,hidden_dim,position_scale_m)
        self.map_layers=nn.ModuleList(LocalMapBlock(d_model,n_heads,dropout,mlp_ratio) for _ in range(map_depth))
        self.scene_cross=nn.ModuleList(LocalCross(d_model,n_heads,dropout) for _ in range(scene_depth))
        self.scene_self=nn.ModuleList(MaskedTransformerBlock(d_model,n_heads,dropout,mlp_ratio) for _ in range(scene_depth))
        self.relative_bias=RelativeAgentBias(n_heads,hidden_dim,position_scale_m)
        self.final_norm=nn.LayerNorm(d_model)

    def forward(self,*,history,agent_mask,map_polylines,map_mask,current_lights,current_light_mask,
                map_ids,map_is_lane,map_stop_sign,map_stop_point,current_light_ids):
        tokens,alive,pose,types=self.agent_encoder(history,agent_mask)
        maps,valid=self.map_encoder(map_polylines,map_mask)
        weights=map_mask.float();denom=weights.sum(-1).clamp_min(1)[...,None]
        center=(map_polylines[...,:2]*weights[...,None]).sum(2)/denom
        direction=(map_polylines[...,2:4]*weights[...,None]).sum(2)/denom
        heading=torch.atan2(direction[...,1],direction[...,0])
        map_pose=torch.cat((center,heading[...,None]),-1).float()
        maps=maps+self.controls(map_pose,map_ids,map_is_lane,current_lights,current_light_mask,
                                current_light_ids,map_stop_sign,map_stop_point)
        maps=maps*valid[...,None].to(maps.dtype)
        idx,neighbor_mask=nearest_neighbors(map_pose,map_pose,valid,self.map_neighbors,self.map_radius_m)
        bias=self.map_bias(map_pose,gather_neighbors(map_pose,idx))
        for layer in self.map_layers: maps=layer(maps,valid,idx,neighbor_mask,bias)
        # Recomputed from current history every replan; no static map->agent cache.
        idx,local_mask=nearest_neighbors(pose,map_pose,valid,self.agent_map_neighbors,self.agent_map_radius_m)
        local_maps=gather_neighbors(maps,idx)
        local_bias=self.agent_map_bias(pose,gather_neighbors(map_pose,idx))
        pair_bias=self.relative_bias(pose,types)
        for cross,layer in zip(self.scene_cross,self.scene_self):
            tokens=cross(tokens,alive,local_maps,local_mask,local_bias)
            tokens=layer(tokens,alive,bias=pair_bias)
        tokens=self.final_norm(tokens)*alive[...,None].to(tokens.dtype)
        b,n,d=tokens.shape
        context=tokens[:,None].expand(-1,n,-1,-1)
        context_mask=alive[:,None].expand(-1,n,-1)
        # Decoder queries all agents and only its own local map neighborhood.
        memory=torch.cat((context,local_maps),2)
        memory_mask=torch.cat((context_mask,local_mask),2)
        cross_bias=torch.cat((pair_bias.permute(0,2,1,3).unsqueeze(-2),local_bias),-1)
        empty=tokens[:,:0];empty_mask=alive[:,:0]
        return SceneEncoding(agent_tokens=tokens,agent_mask=alive,agent_pose=pose,agent_type=types,
            map_tokens=maps,map_mask=valid,light_tokens=empty,light_mask=empty_mask,
            memory=tokens,memory_mask=alive,relative_bias=pair_bias,
            agent_memory=memory,agent_memory_mask=memory_mask,agent_memory_bias=cross_bias)
