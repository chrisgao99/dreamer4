import torch
from waymo.training.world_model.direct_action_flow_mon_losses import motion_distances,physical_losses,collision_gap,reliable_road_mask


def test_gt_identity_and_joint_winner():
    initial=torch.zeros(1,2,3)
    gt=torch.zeros(1,2,3,3);gt[...,0]=torch.arange(1,4)*.1
    mask=torch.ones(1,2,3,dtype=torch.bool)
    d,_=motion_distances(gt,gt,initial,mask);assert d.item()==0
    # Different agents prefer different candidates: choose a single joint sample.
    c1=gt.clone();c1[:,1,:,0]+=1;c1.requires_grad_()
    c2=gt.clone();c2[:,0,:,0]+=2;c2.requires_grad_()
    scores=torch.stack([motion_distances(c,gt,initial,mask)[0] for c in (c1,c2)],1)
    winner=scores.detach().argmin(1)
    assert winner.item()==0
    scores.gather(1,winner[:,None]).mean().backward()
    assert c1.grad.abs().sum()>0 and c2.grad.abs().sum()==0


def scene(x=0.,y=0.):
    poses=torch.zeros(1,2,3,3);poses[:,0,:,0]=x;poses[:,1,:,0]=x+20;poses[...,1]=y
    init=poses[:,:,0].clone();valid=torch.ones(1,2,3,dtype=torch.bool);types=torch.ones(1,2,dtype=torch.long)
    # East oriented lower boundary: north is the drivable side.
    road=torch.tensor([[[[-10.,0.,1.,0.,15.],[0.,0.,1.,0.,15.],[20.,0.,1.,0.,15.]]]])
    rm=torch.ones(1,1,3,dtype=torch.bool)
    return poses,init,valid,types,road,rm


def test_collision_sat_clear_and_overlap():
    p,i,v,t,road,rm=scene(y=5)
    assert collision_gap(p,t)[0,0,0,1]>0
    p[:,1,:,0]=1
    assert collision_gap(p,t)[0,0,0,1]<0
    p.requires_grad_();loss,_=physical_losses(p,i,v,t,road,rm)
    loss['collision'].sum().backward();assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_road_direction_and_pedestrian_exemption():
    inside=scene(y=5);outside=scene(y=-5)
    good,gm=physical_losses(*inside);bad,bm=physical_losses(*outside)
    assert bad['offroad']>good['offroad'] and bm['offroad_rate_proxy'].item()==1
    p,i,v,t,road,rm=outside;t[:]=2
    loss,_=physical_losses(p,i,v,t,road,rm)
    assert loss['offroad'].item()==0 and loss['sideslip'].item()==0
    loss,_=physical_losses(p,i,v,t,road,torch.zeros_like(rm))
    assert loss['offroad'].item()==0


def test_sideslip_forward_zero_lateral_nonzero():
    p,i,v,t,road,rm=scene(y=5)
    p[:,:,1:,0]+=torch.tensor([1.,2.])
    forward,_=physical_losses(p,i,v,t,road,rm)
    assert forward['sideslip'].item()==0
    p[:,:,1:,1]+=torch.tensor([1.,2.])
    lateral,_=physical_losses(p,i,v,t,road,rm)
    assert lateral['sideslip'].item()>0

def test_data_only_road_reliability():
    gt,initial,valid,types,road,rm=scene(y=5)
    support=reliable_road_mask(gt,initial,valid,types,road,rm)
    assert support.all()
    predicted=gt.clone();predicted[...,1]=-5
    loss,metrics=physical_losses(predicted,initial,valid,types,road,rm,road_support=support)
    assert loss['offroad'].item()>0 and metrics['offroad_rate_proxy'].item()==1
    bad_gt=predicted.clone()
    bad_support=reliable_road_mask(bad_gt,initial,valid,types,road,rm)
    assert not bad_support.any()
    loss,metrics=physical_losses(predicted,initial,valid,types,road,rm,road_support=bad_support)
    assert loss['offroad'].item()==0 and metrics['road_reliable_fraction'].item()==0
    assert metrics['offroad_rate_proxy_unfiltered'].item()==1

if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_'):fn();print('PASS',name)
