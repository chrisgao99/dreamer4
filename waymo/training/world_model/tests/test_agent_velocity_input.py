import argparse
import io
import torch

from waymo.training.world_model.direct_action_flow import (
    AgentHistoryEncoder, ActionNormalizer, DirectActionFlowModel, rollout_receding_horizon,
)
from waymo.training.world_model.tests.test_direct_action_flow import _synthetic_batch
from waymo.training.world_model.train_waymo_direct_action_flow import build_arg_parser, create_model


def test_velocity_affects_only_opted_in_encoder_and_has_gradients():
    history=_synthetic_batch(batch_size=1)['agents'][:,:,:4].clone()
    mask=torch.ones(1,3,dtype=torch.bool)
    changed=history.clone();changed[...,3]=12.;changed[...,4]=-3.
    for enabled in (False,True):
        torch.manual_seed(0)
        enc=AgentHistoryEncoder(32,4,1,4,0.,4.,100.,16,include_agent_velocity=enabled)
        enc.eval()
        original=enc(history,mask)[0]
        modified=enc(changed,mask)[0]
        assert enc.state_mlp[0].in_features==(6 if enabled else 4)
        if enabled:
            assert not torch.allclose(original,modified)
            changed.requires_grad_();enc(changed,mask)[0].square().sum().backward()
            assert torch.isfinite(changed.grad).all()
            assert changed.grad[...,3:5].abs().sum()>0
        else:
            torch.testing.assert_close(original,modified,rtol=0,atol=0)


def test_invalid_velocity_is_masked_before_projection():
    h=_synthetic_batch(batch_size=1)['agents'][:,:,:4].clone()
    h[:,:,0,5]=0
    enc=AgentHistoryEncoder(32,4,1,4,0.,4.,100.,16,include_agent_velocity=True).eval()
    mask=torch.ones(1,3,dtype=torch.bool)
    expected=enc(h,mask)[0]
    h[:,:,0,3:5]=float('nan')
    torch.testing.assert_close(enc(h,mask)[0],expected)


def test_checkpoint_args_restore_velocity_architecture_and_legacy_defaults():
    args=build_arg_parser().parse_args(['--data_dir','a','--val_data_dir','b','--ckpt_dir','c',
        '--action_stats_path','d','--d_model','32','--n_heads','4','--hidden_dim','16',
        '--history_depth','1','--map_depth','1','--scene_depth','1','--action_depth','1',
        '--include_agent_velocity'])
    model=create_model(args)
    buffer=io.BytesIO();torch.save({'args':vars(args),'model':model.state_dict()},buffer);buffer.seek(0)
    checkpoint=torch.load(buffer,weights_only=False)
    restored=create_model(argparse.Namespace(**checkpoint['args']))
    restored.load_state_dict(checkpoint['model'],strict=True)
    assert restored.scene_encoder.agent_encoder.state_mlp[0].in_features==6
    del args.include_agent_velocity;del args.velocity_scale_mps
    legacy=create_model(args)
    assert legacy.scene_encoder.agent_encoder.state_mlp[0].in_features==4


def test_receding_history_velocity_comes_from_executed_displacement():
    b=_synthetic_batch(batch_size=1,history=4,horizon=8)
    model=DirectActionFlowModel(d_model=32,n_heads=4,history_length=4,horizon=6,chunk_size=2,
        history_depth=1,map_depth=1,scene_depth=1,action_depth=1,step_refiner_depth=1,
        hidden_dim=16,dropout=0.,include_agent_velocity=True).eval()
    histories=[]
    hook=model.scene_encoder.agent_encoder.register_forward_pre_hook(
        lambda module,args: histories.append(args[0].detach().clone()))
    poses=rollout_receding_horizon(model,ActionNormalizer(torch.zeros(16,3),torch.ones(16,3)),
        initial_history=b['agents'][:,:,:4],agent_mask=b['agent_mask'],
        map_polylines=b['map_polylines'],map_mask=b['map_mask'],
        current_light_sequence=b['lights'][:,3:11],current_light_mask_sequence=b['light_mask'][:,3:11],
        focus_action_sequence=None,focus_action_valid=None,rollout_steps=4,commitment=2,solver_steps=1)
    hook.remove()
    prev=torch.cat((b['agents'][:,:,3:4,:2],poses[:,:,:1,:2]),dim=2)
    expected=(poses[:,:,:2,:2]-prev)/.1
    torch.testing.assert_close(histories[1][:,:,-2:,3:5],expected)
    torch.testing.assert_close(histories[1][:,:,-2:,2],expected.norm(dim=-1))
