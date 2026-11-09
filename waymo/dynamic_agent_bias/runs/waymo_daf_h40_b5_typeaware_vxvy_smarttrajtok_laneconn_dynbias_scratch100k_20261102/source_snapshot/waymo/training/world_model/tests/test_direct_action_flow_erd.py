import copy
from types import SimpleNamespace
import torch
from waymo.training.world_model.train_waymo_direct_action_flow_erd import (
    velocity_to_score, dmd_surrogate, generate_committed, phase_at, StepBatches,
)


def test_score_gaussian_and_tempering():
    x=torch.tensor([-.8,.2,1.4],dtype=torch.float64)
    for t in (.02,.3,.9):
        variance=t*t+(1-t)**2
        velocity=(2*t-1)/variance*x
        score=velocity_to_score(x,velocity,t)
        torch.testing.assert_close(score,-x.float()/variance)
    # beta != 1 requires the x term too; just scaling velocity is incorrect.
    difference=velocity_to_score(x,velocity,.9)-.99*velocity_to_score(x,velocity,.9)
    torch.testing.assert_close(difference,.01*velocity_to_score(x,velocity,.9),atol=1e-6,rtol=1e-4)


def test_surrogate_gradient_and_detach():
    clean=torch.randn(1,2,3,3,requires_grad=True)
    score=torch.randn_like(clean,requires_grad=True)
    mask=torch.tensor([[[True,True,True],[False,False,False]]])
    dmd_surrogate(clean,score,mask).backward()
    torch.testing.assert_close(clean.grad,score.detach()*mask[...,None]/9)
    assert score.grad is None
    clean.grad=None
    dmd_surrogate(clean,torch.zeros_like(clean),mask).backward()
    assert clean.grad.abs().sum()==0


class TinyModel(torch.nn.Module):
    horizon=15
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(.02))
    def encode_scene(self,history,agent_mask,**kw):
        return SimpleNamespace(agent_mask=agent_mask, value=history[:,:,-1,0:1])
    def decode_velocity(self,x,t,scene,mask):
        return self.weight*x+scene.value[:,:,None,:]*.02

class IdentityNormalizer:
    def denormalize(self,x,types): return x


def inputs():
    h=torch.zeros(1,2,11,8); h[...,5]=1; h[...,7]=1; h.requires_grad_()
    b=dict(agent_mask=torch.ones(1,2,dtype=torch.bool),map_polylines=torch.zeros(1),map_mask=torch.ones(1,dtype=torch.bool),
        lights=torch.zeros(1,40,1,1),light_mask=torch.ones(1,40,1,dtype=torch.bool))
    return h,b,torch.tensor([10])


def test_checkpoint_preserves_full_replan_gradient():
    model=TinyModel(); h,b,a=inputs()
    torch.manual_seed(3)
    x,m=generate_committed(model,IdentityNormalizer(),h,b,a,solver_steps=2,checkpoint_grad=False)
    # Only final chunk loss: earlier generated history must affect its value.
    x[:,:,10:].sum().backward()
    expected=x.detach(); grad=model.weight.grad.clone(); hgrad=h.grad.clone()
    model.weight.grad=None; h.grad=None
    torch.manual_seed(3)
    x2,m2=generate_committed(model,IdentityNormalizer(),h,b,a,solver_steps=2,checkpoint_grad=True)
    x2[:,:,10:].sum().backward()
    torch.testing.assert_close(x2,expected)
    torch.testing.assert_close(model.weight.grad,grad)
    torch.testing.assert_close(h.grad,hgrad)
    assert h.grad.abs().sum()>0 and model.weight.grad.abs()>0


def test_schedule_and_resume_sampling():
    updates=[i for i in range(200) if phase_at(i,100,5)[1]]
    assert len(updates)==20 and updates[0]==104 and updates[-1]==199
    original=list(StepBatches(100,2,0,10,1000))
    assert list(StepBatches(100,2,4,10,1000))==original[4:]
