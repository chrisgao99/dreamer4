"""SDC factorization isolates the SDC's denoising; training loader fields are sane."""
import argparse, sys, random
import torch
sys.path.insert(0, '/p/yufeng/tri30/dreamer4')
from waymo.scene_cache.dataset import SceneCacheDataset, collate_scenes, read_manifest
from waymo.scene_cache.sparse_agent_attention import patch_model, set_sdc_mask
from waymo.training.world_model import train_waymo_direct_action_flow as base

ck = torch.load('/p/yufeng/tri30/dreamer4/waymo/checkpoints/waymo_daf_laneconn90k_mon8_gt_unroll80_b5_detach0_10k_20261009/best_ade.pt', map_location='cpu', weights_only=False)
ta = argparse.Namespace(**ck['args']); dev = torch.device('cuda')
norm = base.build_normalizer(ck['action_stats'], dev)
rows = random.Random(1).sample(read_manifest('/p/yufeng/waymo_scene_prep/data/manifest_training.jsonl', min_current_valid=60), 4)
ds = SceneCacheDataset(rows, center='mixed', agent_radius_m=150, max_agents=128, map_crop_m=120, loss_agents=48)
items = [ds[i] for i in range(4)]
for it in items:
    n = len(it['agent_mask']); sdc = int(it['agent_is_sdc'].sum())
    assert it['agent_loss_mask'].sum() >= min(48, n) and it['agent_loss_mask'][0]
    assert it['agent_eval_mask'].sum() <= it['agent_loss_mask'].sum() and bool((it['agent_eval_mask'] <= it['agent_loss_mask']).all())
    print(f"agents {n} loss {int(it['agent_loss_mask'].sum())} eval {int(it['agent_eval_mask'].sum())} sdc_slots {sdc} "
          f"center_is_sdc {bool(it['agent_is_sdc'][0])} map {len(it['map_ids'])}")
batch = base.move_batch(collate_scenes(items), dev)
for factorized in (False, True):
    model = base.create_model(ta).to(dev); model.load_state_dict(ck['ema_model']); model.eval()
    patch_model(model, knn=48, radius_m=80., sdc_factorized=factorized)
    set_sdc_mask(model, batch['agent_is_sdc'])
    with torch.no_grad():
        p = base.prepare_batch(batch, norm, ta, random_start=False)
        scene = model.encode_scene(**base.scene_kwargs(batch, p))
        mask = scene.agent_mask[:, :, None].expand(-1, -1, model.horizon)
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn((*mask.shape, 3), device=dev, generator=g)
        x2 = x.clone(); sdc = batch['agent_is_sdc']
        x2[~sdc] = torch.randn_like(x2[~sdc])          # change every non-SDC agent's noise
        t = torch.full((x.shape[0],), 0.5, device=dev)
        v1 = model.decode_velocity(x, t, scene, mask); v2 = model.decode_velocity(x2, t, scene, mask)
        diff = (v1[sdc] - v2[sdc]).abs().max().item()
    print(f'sdc_factorized={factorized}: max change of SDC velocity = {diff:.3e}')
    assert (diff < 1e-5) if factorized else (diff > 1e-4)
print('ALL OK')
