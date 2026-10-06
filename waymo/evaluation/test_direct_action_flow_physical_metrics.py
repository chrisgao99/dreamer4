import numpy as np
import torch
from waymo.evaluation.direct_action_flow_physical_metrics import evaluate_batch,summarize,METRIC_NAMES


def test_straight_motion_and_collision():
    initial=torch.tensor([[[0.,5.,0.],[20.,5.,0.]]])
    gt=initial[:,:,None,:].expand(-1,-1,5,-1).clone();gt[...,0]+=torch.arange(1,6)*.1
    valid=torch.ones(1,2,5,dtype=torch.bool);types=torch.ones(1,2,dtype=torch.long)
    maps=torch.tensor([[[[-10.,0.,1.,0.,15.],[0.,0.,1.,0.,15.],[20.,0.,1.,0.,15.]]]])
    mm=torch.ones(1,1,3,dtype=torch.bool)
    values,reference=evaluate_batch(gt[:,None].expand(-1,2,-1,-1,-1),initial,gt,valid,types,maps,mm)
    m=summarize(values)
    assert m['collision_pair_time_rate_proxy']==0 and m['offroad_agent_time_rate_proxy_raw']==0
    assert m['lateral_speed_mean_abs_mps']==0 and abs(m['speed_mean_mps']-1)<1e-4
    assert m['acceleration_mean_norm_mps2']<1e-3
    np.testing.assert_allclose(values[:,0],reference[:,0])
    overlap=gt.clone();overlap[:,1,:,0]=overlap[:,0,:,0]
    values,_=evaluate_batch(overlap[:,None],initial,gt,valid,types,maps,mm)
    assert summarize(values)['collision_pair_time_rate_proxy']==1
    invalid=torch.zeros_like(valid)
    values,_=evaluate_batch(gt[:,None],initial,gt,invalid,types,maps,mm)
    assert summarize(values)['collision_pair_time_rate_proxy'] is None


def test_weighted_aggregation():
    x=np.zeros((2,1,len(METRIC_NAMES),2));x[0,0,:,0]=1;x[0,0,:,1]=2;x[1,0,:,1]=8
    assert summarize(x)[METRIC_NAMES[0]]==.1

if __name__=='__main__':
    test_straight_motion_and_collision();test_weighted_aggregation();print('PASS physical metrics and weighted aggregation')
