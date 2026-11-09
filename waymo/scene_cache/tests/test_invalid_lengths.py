"""The two WOMD training tracks with a negative current length no longer crash prepare_batch.

Found by /p/yufeng/waymo_scene_prep/code/scan_invalid_lengths.py; the run
waymo_daf_scene_full_knn48r80_sdcfact_loss48_30ep_20261022 died on one at step ~410k.
"""
import argparse, json, sys
import numpy as np, torch
sys.path.insert(0, '/p/yufeng/tri30/dreamer4')
from waymo.scene_cache.dataset import SceneCacheDataset, collate_scenes
from waymo.training.world_model import train_waymo_direct_action_flow as base
from waymo.training.world_model.direct_action_flow import agents_to_bntf, gather_agent_window
from waymo.training.world_model.action_kinematics import KinematicsConfig, inverse_actions

bad = [json.loads(l) for l in open('/p/yufeng/waymo_scene_prep/data/invalid_lengths_training.jsonl')]
ck = torch.load('/p/yufeng/tri30/dreamer4/waymo/checkpoints/waymo_daf_laneconn90k_mon8_gt_unroll80_b5_detach0_10k_20261009/best_ade.pt',
                map_location='cpu', weights_only=False, mmap=True)
ta = argparse.Namespace(**ck['args']); config = KinematicsConfig.from_args(ta)
for row in bad:
    d = np.load(row['file'])
    centers = [int(d['sdc_track_index'])] + [int(i) for i in d['tracks_to_predict']] + [row['track_index']]
    for center in centers:
        ds = SceneCacheDataset([row['file']], agent_radius_m=150, max_agents=128)
        ds._choose_center = lambda d, current, c=center: c
        item = ds[0]
        j = int(np.flatnonzero(item['agent_ids'].numpy() == row['track_id'])[0]) if row['track_id'] in item['agent_ids'].tolist() else None
        if j is not None:
            assert abs(float(item['agent_lengths'][j]) - row['median_positive_length']) < 1e-4
        batch = collate_scenes([item])
        agents = agents_to_bntf(batch['agents'], batch['agent_mask'])
        for anchor in range(10, 91 - ta.horizon):  # every window start prepare_batch can pick
            h, f = gather_agent_window(agents, torch.tensor([anchor]), history_length=ta.history_length, horizon=ta.horizon)
            inverse_actions(h, f, batch['agent_mask'], agent_lengths=batch['agent_lengths'], config=config,
                            max_displacement_m=ta.physical_max_displacement_m, max_yaw_delta_rad=ta.physical_max_yaw_delta_rad)
    print(row['scenario_id'], 'track', row['track_id'], f"length {row['length']:.3f} -> {row['median_positive_length']:.3f}",
          f'ok for {len(centers)} centers x {91 - ta.horizon - 10} anchors')
print('ALL OK')
