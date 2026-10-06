import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path('/p/yufeng/tri30/dreamer4')
sys.path.insert(0, str(ROOT))
from waymo.evaluation import visualize_direct_action_flow_intersection_two_agent_rollouts as vis
OUT = Path(__file__).resolve().parent
reference = ROOT / 'waymo/eval_results/world_model/typeaware_vxvy_step95000_intersection10_n10_plan40_b40_h80_20260929/run_config.json'
config = json.loads(reference.read_text())
config.update(checkpoint=str(ROOT / 'waymo/checkpoints/waymo_daf_h40_typeaware_vxvy_best95k_mon8_gt_unroll80_b5_detach0_10k_20260929/step_0001000.pt'), checkpoint_step=1000, output_dir=str(OUT))
(OUT / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
sys.argv = [__file__, '--checkpoint', config['checkpoint'], '--val_data_dir', config['val_data_dir'], '--selected_manifest', config['selected_manifest'], '--reference_h90_dir', str(Path(config['selected_manifest']).parent), '--output_dir', str(OUT), '--device', 'cpu', '--weights', 'ema', '--focus_mode', 'generate_all', '--num_workers', '0', '--num_scenes', '10', '--num_rollouts', '10', '--rollout_steps', '80', '--commitment_steps', '40', '--solver_steps', '8', '--seed', str(config['seed']), '--model_label', 'Type-aware vxvy finetune step1000 (EMA)', '--images_only', '--filename_prefix', 'typeaware_vxvy_ft1000_']
original_plot = vis._plot_generate_all_scene
rows, images = [], []
def save_plot(**kwargs):
    metrics = original_plot(**kwargs)
    path = kwargs['output_path']
    np.savez_compressed(path.with_suffix('.npz'), gt_tkf=kwargs['gt_tkf'], pred_xy=kwargs['pred_xy'], agent_ids=kwargs['agent_ids'])
    rows.append(dict(kwargs['record'], **metrics, image=str(path)))
    images.append(path)
    return metrics
vis._plot_generate_all_scene = save_plot
vis.visualize(vis.parse_args())
vis._make_contact_sheet(images, OUT / 'contact_sheet.png')
(OUT / 'rollout_metrics.json').write_text(json.dumps(dict(config=config, samples=rows), indent=2) + '\n')
print('Complete: 10 scenes, trajectories, metrics and contact_sheet.png', flush=True)
