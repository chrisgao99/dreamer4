import unittest
from types import SimpleNamespace
import torch
from waymo.training.world_model.direct_action_flow import ActionNormalizer, execute_holonomic_actions
from waymo.training.world_model.catk_flow_recovery import recovery_targets, update_history, select_segment


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.history=torch.zeros(1,2,11,8)
        self.history[...,5]=1;self.history[...,7]=1
        self.gt=torch.zeros(1,2,80,3)
        self.gt[...,0]=torch.arange(1,81)*.1
        self.valid=torch.ones(1,2,80,dtype=torch.bool)
        self.mask=torch.ones(1,2,dtype=torch.bool)
        self.norm=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3))

    def test_recovery_from_generated_pose_and_padding(self):
        self.history[...,0]=7.4
        self.history.requires_grad_()
        a,v,m=recovery_targets(self.history,self.gt,self.valid,self.mask,self.norm,75)
        self.assertFalse(a.requires_grad)
        self.assertEqual(int(v.sum()),10)
        self.assertFalse(v[:,:,5:].any())
        self.assertAlmostEqual(float(a[0,0,0,0]),.2,places=5)
        initial=torch.zeros(1,2,3);initial[...,0]=7.4
        poses=execute_holonomic_actions(initial,a[:,:,:5],v[:,:,:5])
        torch.testing.assert_close(poses,self.gt[:,:,75:])

    def test_physical_rejection_and_no_resurrection(self):
        self.history[:,0,:,0]=-10
        self.valid[:,1,:]=False
        a,v,m=recovery_targets(self.history,self.gt,self.valid,self.mask,self.norm,0)
        self.assertFalse(v.any());self.assertEqual(m['rejected_physical_first'],1)
        self.assertEqual(m['eligible_first'],1)
        self.assertTrue(torch.isfinite(a).all())

    def test_normalized_rejection_not_clipping(self):
        self.history[...,0]=-1
        norm=ActionNormalizer(torch.zeros(16,3),torch.ones(16,3)*.1)
        a,v,m=recovery_targets(self.history,self.gt,self.valid,self.mask,norm,0)
        self.assertFalse(v.any());self.assertEqual(m['rejected_normalized_first'],2)
        self.assertEqual(float(a.abs().sum()),0)

    def test_feedback_detached(self):
        poses=torch.ones(1,2,5,3,requires_grad=True)
        h=update_history(self.history,poses,torch.ones(1,2,5,dtype=torch.bool))
        self.assertFalse(h.requires_grad);self.assertEqual(h.shape,self.history.shape)
        torch.testing.assert_close(h[:,:,-5:,:2],poses.detach()[...,:2])

    def test_joint_winner_not_per_agent_mosaic(self):
        class Model:
            horizon=40
            count=0
            def sample_normalized_actions(model,scene,mask,**kw):
                actions=torch.zeros(1,2,40,3,requires_grad=True)
                # Candidate0 is exact for agent0, candidate1 exact for agent1.
                with torch.no_grad():
                    actions[...,0]=.1
                    actions[:,1 if model.count==0 else 0,:,0]+=.2 if model.count==0 else .4
                model.count+=1
                return actions
        target=SimpleNamespace(future_pose=self.gt,valid=self.valid,agent_type=torch.ones(1,2,dtype=torch.long))
        poses,_,_=select_segment(Model(),self.norm,SimpleNamespace(agent_mask=self.mask),
            self.history,target,0,SimpleNamespace(num_candidates=2,solver_steps=1),0)
        self.assertFalse(poses.requires_grad)
        torch.testing.assert_close(poses[:,0],self.gt[:,0,:5])
        self.assertGreater(float((poses[:,1]-self.gt[:,1,:5]).abs().sum()),0)


if __name__=='__main__':unittest.main()
