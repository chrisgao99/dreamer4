"""GT-history reset diagnostic; run using the dreamer4 Python environment."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-gtreset95000')
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch
ROOT = Path('/p/yufeng/tri30/dreamer4')
sys.path.insert(0, str(ROOT))
from waymo.evaluation.visualize_direct_action_flow_intersection_two_agent_rollouts import (
    load_model_and_normalizer, _repeat_batch, _plot_generate_all_scene, _make_contact_sheet)
from waymo.training.world_model.action_kinematics import execute_model_actions

OUT = Path(__file__).resolve().parent
CKPT = ROOT / 'waymo/checkpoints/waymo_daf_h40_b5_typeaware_vxvy_scratch100k_20260925/best.pt'
MANIFEST = ROOT / 'waymo/eval_results/world_model/h90_intersection10_two_agent_n10_ctx11_h80_20260907/selected_intersection10_manifest.json'
DATA = ROOT / 'data/waymo_vector_dataset_ooi_centered_50k_with_lengths/val'

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    torch.set_float32_matmul_precision('high')
    model, normalizer, train_args, step = load_model_and_normalizer(CKPT, torch.device('cpu'), 'ema')
    assert step == 95000 and model.horizon == 40 and train_args.history_length == 11
    records = json.loads(MANIFEST.read_text())['samples']
    args = argparse.Namespace(eval_ctx=11, horizon=80, margin_m=12., dpi=150,
        model_label='Type-aware + vx/vy H40 step95k (EMA)\nGT history reset every 5 steps (0.5s); joined short predictions')
    rows, images = [], []
    start = time.time()
    for index, record in enumerate(records):
        path = DATA / record['filename']
        with np.load(path, allow_pickle=False) as data:
            batch = {k: torch.from_numpy(data[k].copy()).unsqueeze(0) for k in
                ('agents','agent_mask','map_polylines','map_mask','lights','light_mask','agent_lengths','agent_ids')}
        agents = batch['agents'].float()
        assert agents.shape[2:] == (91,8)
        assert int(batch['agent_ids'][0,record['target_slot']]) == record['target_track_id']
        seed = 20260907 + int(record['sample_order'])
        generator = torch.Generator(device='cpu').manual_seed(seed)
        parts, masks = [], []
        for elapsed in range(0,80,5):
            anchor = 10 + elapsed
            # Read every observation from GT. No prediction is fed back.
            history = _repeat_batch(agents[:,:,anchor-10:anchor+1],10)
            scene = model.encode_scene(history=history,
                agent_mask=_repeat_batch(batch['agent_mask'].bool(),10),
                map_polylines=_repeat_batch(batch['map_polylines'].float(),10),
                map_mask=_repeat_batch(batch['map_mask'].bool(),10),
                current_lights=_repeat_batch(batch['lights'][:,anchor].float(),10),
                current_light_mask=_repeat_batch(batch['light_mask'][:,anchor].bool(),10))
            mask = scene.agent_mask[:,:,None].expand(-1,-1,40)
            normalized = model.sample_normalized_actions(scene, mask, solver_steps=8, generator=generator)
            actions = normalizer.denormalize(normalized,scene.agent_type)
            poses = execute_model_actions(model,scene.agent_pose,actions[:,:,:5],mask[:,:,:5],
                agent_type=scene.agent_type,agent_lengths=_repeat_batch(batch['agent_lengths'].float(),10))
            assert torch.isfinite(poses).all()
            parts.append(poses)
            masks.append(mask[:,:,:5])
        poses = torch.cat(parts,dim=2)
        valid = torch.cat(masks,dim=2)
        gt = agents[0].permute(1,0,2).numpy()
        pred = np.concatenate((np.repeat(gt[None,:11,:,:2],10,axis=0),poses[...,:2].permute(0,2,1,3).numpy()),axis=1)
        assert pred.shape == (10,91,32,2)
        image = OUT / f'gtreset_b5_scene_{index:02d}_{record["scenario_id"]}.png'
        metrics = _plot_generate_all_scene(args=args,record=record,gt_tkf=gt,pred_xy=pred,
            map_polylines=batch['map_polylines'][0].numpy(),map_mask=batch['map_mask'][0].numpy().astype(bool),
            agent_ids=batch['agent_ids'][0].numpy(),output_path=image)
        np.savez_compressed(OUT / f'scene_{index:02d}_trajectories.npz',
            predicted_poses=poses.numpy(),prediction_valid=valid.numpy(),gt_agents=agents[0].numpy(),
            reset_anchors=np.arange(10,90,5),seed=seed)
        rows.append(dict(scenario_id=record['scenario_id'],filename=record['filename'],seed=seed,**metrics))
        images.append(image)
        (OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(f'generated {index+1}/10 elapsed={time.time()-start:.1f}s {image.name}',flush=True)
    _make_contact_sheet(images,OUT/'gtreset_b5_contact_sheet.png')
    config=dict(checkpoint=str(CKPT),checkpoint_step=step,weights='ema',manifest=str(MANIFEST),
        data_dir=str(DATA),mode='all-agent GT history and pose reset every 5 steps',
        plan_horizon=40,history_length=11,commitment_steps=5,rollout_steps=80,
        num_rollouts=10,solver_steps=8,seed=20260907,device='cpu',focus_mode='generate_all',
        note='Connected plot lines may bridge reset discontinuities. ADE measures short GT-conditioned predictions, not closed-loop performance.')
    (OUT/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('COMPLETE',flush=True)

if __name__ == '__main__':
    main()
