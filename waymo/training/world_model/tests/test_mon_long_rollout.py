import unittest
from types import SimpleNamespace
import torch
from waymo.training.world_model.mon_long_rollout import generate_poses


class ToyModel(torch.nn.Module):
    horizon = 40
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(16) * .1)
    def encode_scene(self, **kw):
        return SimpleNamespace(agent_mask=kw['agent_mask'],
            index=int(kw['current_lights'][0, 0, 0]))
    def decode_velocity(self, x, t, scene, mask):
        target = torch.zeros_like(x)
        target[..., 0] = self.weights[scene.index]
        return target - x


class IdentityNormalizer:
    def denormalize(self, actions, types):
        return actions


def run(detach, checkpoint):
    model = ToyModel()
    history = torch.zeros(1, 1, 11, 8)
    history[..., 5] = 1
    batch = dict(agent_mask=torch.ones(1, 1, dtype=torch.bool),
        map_polylines=torch.zeros(1, 1, 1, 6), map_mask=torch.ones(1, 1, 1,dtype=torch.bool),
        lights=torch.zeros(1, 91, 1, 4), light_mask=torch.ones(1, 91, 1,dtype=torch.bool))
    for i in range(16): batch['lights'][:, 10+i*5, :, 0] = i
    torch.manual_seed(0)
    poses=generate_poses(model,IdentityNormalizer(),history,batch,torch.tensor([10]),
        detach_every=detach,solver_steps=1,checkpoint_grad=checkpoint)
    return model, poses


class GradientBoundariesTest(unittest.TestCase):
    def test_equal_forward_different_gradient_boundaries(self):
        reference=None
        for checkpoint in (False,True):
            for detach,start in ((0,0),(5,15),(20,12)):
                model,poses=run(detach,checkpoint)
                if reference is None:reference=poses.detach()
                torch.testing.assert_close(poses,reference)
                poses[:,:,-1,0].sum().backward()
                expected=torch.zeros(16);expected[start:]=5
                torch.testing.assert_close(model.weights.grad,expected)

    def test_earlier_segments_keep_their_own_supervision(self):
        for detach in (5,20):
            model,poses=run(detach,True)
            poses[...,0].sum().backward()
            self.assertTrue((model.weights.grad>0).all())


if __name__=='__main__':unittest.main()


class VelocityFeedbackModel(ToyModel):
    include_agent_velocity = True
    def encode_scene(self, **kw):
        scene=super().encode_scene(**kw)
        scene.vx=kw['history'][:,:,-1,3]
        return scene
    def decode_velocity(self,x,t,scene,mask):
        target=torch.zeros_like(x)
        target[...,0]=self.weights[scene.index]+.01*scene.vx[...,None]
        return target-x


def test_typeaware_velocity_feedback_full80_gradient_and_checkpoint_replay():
    from waymo.training.world_model.action_kinematics import KinematicsConfig
    outputs=[];gradients=[]
    for use_checkpoint in (False,True):
        model=VelocityFeedbackModel()
        model.kinematics=KinematicsConfig(mode='type_aware',vehicle_rho=.22,cyclist_rho=.03)
        h=torch.zeros(1,1,11,8);h[...,5]=1;h[...,7]=1
        b=dict(agent_mask=torch.ones(1,1,dtype=torch.bool),agent_lengths=torch.full((1,1),4.),
            map_polylines=torch.zeros(1,1,1,6),map_mask=torch.ones(1,1,1,dtype=torch.bool),
            lights=torch.zeros(1,91,1,4),light_mask=torch.ones(1,91,1,dtype=torch.bool))
        for i in range(16):b['lights'][:,10+i*5,:,0]=i
        torch.manual_seed(0)
        p=generate_poses(model,IdentityNormalizer(),h,b,torch.tensor([10]),detach_every=0,
            solver_steps=1,checkpoint_grad=use_checkpoint)
        # First segment moves .1 m/step -> 1 m/s. Next uses that velocity: .11 m/step.
        torch.testing.assert_close(p[0,0,5,0],torch.tensor(.61))
        p[:,:,-1,0].sum().backward()
        assert torch.isfinite(model.weights.grad).all() and (model.weights.grad>0).all()
        assert model.weights.grad[0]>5  # Includes the history-velocity feedback path.
        outputs.append(p.detach());gradients.append(model.weights.grad.clone())
    torch.testing.assert_close(outputs[0],outputs[1])
    torch.testing.assert_close(gradients[0],gradients[1])
