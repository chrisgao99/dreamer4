import unittest
import torch
from waymo.training.world_model.direct_action_flow import balanced_agent_flow_loss,dynamic_agent_mask,select_window_anchors
from waymo.training.world_model.analyze_dynamic_windows import window_stats

class DynamicBalanceTest(unittest.TestCase):
    def test_agent_average_and_group_weights(self):
        loss=torch.tensor([[[[2.],[2.]],[[6.],[999.]],[[10.],[10.]]]],requires_grad=True)
        valid=torch.tensor([[[1,1],[1,0],[1,1]]],dtype=torch.bool)
        moving=torch.tensor([[True,True,False]])
        out,_=balanced_agent_flow_loss(loss,valid,moving)
        self.assertAlmostEqual(out.item(),.7*4+.3*10,places=5)
        out.backward();self.assertEqual(loss.grad[0,1,1].item(),0)
        self.assertAlmostEqual(loss.grad[0,:2].sum().item(),.7,places=5)
    def test_empty_group(self):
        x=torch.ones(1,2,3,3,requires_grad=True)*4
        v=torch.ones(1,2,3,dtype=torch.bool)
        for dynamic in (torch.ones(1,2,dtype=torch.bool),torch.zeros(1,2,dtype=torch.bool)):
            out,_=balanced_agent_flow_loss(x,v,dynamic);self.assertAlmostEqual(out.item(),4)
        out,_=balanced_agent_flow_loss(x,~v,dynamic);self.assertEqual(out.item(),0)
    def test_speed_and_invalid_steps(self):
        a=torch.zeros(1,3,15,3);a[0,0,:,0]=.1;a[0,1,:,0]=.01;a[0,2,:,0]=100
        v=torch.ones(1,3,15,dtype=torch.bool);v[0,2]=False
        self.assertEqual(dynamic_agent_mask(a,v).tolist(),[[True,False,False]])
    def test_windows_and_fallback(self):
        a=torch.zeros(2,3,91,8);a[...,5]=1;a[...,7]=1
        a[:,0,:,0]=torch.arange(91)*.1
        a[1,0,20:,5]=0
        mask=torch.ones(2,3,dtype=torch.bool)
        sel,fb,n,r,*_=window_stats(a,mask)
        self.assertEqual(sel[0].sum(),66);self.assertTrue(fb[1]);self.assertEqual(sel[1].sum(),1)
        self.assertAlmostEqual(float(r[0,0]),1/3,places=6)
        anchors=select_window_anchors(a,mask,history_length=11,horizon=15,random_start=False,max_displacement_m=5,max_yaw_delta_rad=.75)
        self.assertEqual(int(sel[1].argmax())+10,anchors[1].item())

if __name__=='__main__':unittest.main()
