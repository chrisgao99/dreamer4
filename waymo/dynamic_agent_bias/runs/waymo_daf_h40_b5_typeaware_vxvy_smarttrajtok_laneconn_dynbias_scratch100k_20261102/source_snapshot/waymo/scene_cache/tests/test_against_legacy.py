"""Cross-check SceneCacheDataset against the legacy focus-centered NPZ + sidecars.

Both versions are mapped back to global coordinates; agents are matched by track id
and map segments by (map id, segment center). Also checks that the dense patched
RelativeAgentBias reproduces the original exactly and that kNN with K>=N and an
infinite radius equals dense.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/p/yufeng/tri30/dreamer4')
from waymo.scene_cache.dataset import SceneCacheDataset, collate_scenes
from waymo.lane_connectivity.dataset import ConnectivityDataset

D = '/p/yufeng/tri30/dreamer4/data'
legacy = ConnectivityDataset(f'{D}/waymo_vector_dataset_ooi_centered_50k_with_lengths/val',
                             f'{D}/waymo_map_smarttrajtok_5m_v2', f'{D}/waymo_lane_connectivity_5m_v1')
rows = {}
for line in open('/p/yufeng/waymo_scene_prep/data/manifest_training.jsonl'):
    r = json.loads(line); rows[r['scenario_id']] = r

def to_global(xy, origin, heading):
    c, s = np.cos(heading), np.sin(heading)
    return np.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], -1) + origin

worst = dict(xy=0., vel=0., yaw=0., speed=0., map=0.)
for i in range(0, len(legacy), len(legacy) // 12)[:12]:
    old = legacy[i]
    new = SceneCacheDataset([rows[old['scenario_id']]])[0]
    for item in (old, new):
        item['g_xy'] = to_global(item['agents'][..., 0:2].numpy(), item['ego_origin_xy'].numpy(), float(item['ego_heading']))
        item['g_vel'] = to_global(item['agents'][..., 3:5].numpy(), 0., float(item['ego_heading']))
        item['g_yaw'] = item['agents'][..., 6].numpy() + float(item['ego_heading'])
    new_slot = {int(t): k for k, t in enumerate(new['agent_ids'])}
    matched = 0
    for k, (tid, m) in enumerate(zip(old['agent_ids'].tolist(), old['agent_mask'].tolist())):
        if not m or old['agents'][k, 10, 5] < .5: continue
        j = new_slot[tid]; matched += 1
        v = (old['agents'][k, :, 5] > .5).numpy()
        assert np.array_equal(v, new['agents'][j, :, 5].numpy() > .5), 'validity mismatch'
        worst['xy'] = max(worst['xy'], np.abs(old['g_xy'][k, v] - new['g_xy'][j, v]).max())
        worst['vel'] = max(worst['vel'], np.abs(old['g_vel'][k, v] - new['g_vel'][j, v]).max())
        dyaw = np.angle(np.exp(1j * (old['g_yaw'][k, v] - new['g_yaw'][j, v])))
        worst['yaw'] = max(worst['yaw'], np.abs(dyaw).max())
        worst['speed'] = max(worst['speed'], np.abs(old['agents'][k, v, 2] - new['agents'][j, v, 2]).max().item())
        assert int(old['agents'][k, 0, 7]) == int(new['agents'][j, 0, 7]), 'type mismatch'
        assert abs(float(old['agent_lengths'][k]) - float(new['agent_lengths'][j])) < 1e-3, 'length mismatch'
    # Every legacy map segment must exist in the whole-scene map with the same id.
    def segs(item):
        g = to_global(item['map_polylines'][..., 0:2].numpy(), item['ego_origin_xy'].numpy(), float(item['ego_heading']))
        return g.mean(1), item['map_ids'].numpy()
    oc, oid = segs(old); nc, nid = segs(new)
    for c, mid in zip(oc, oid):
        same = nc[nid == mid]
        worst['map'] = max(worst['map'], np.linalg.norm(same - c, axis=-1).min())
    print(old['scenario_id'], 'agents matched', matched, 'legacy/new segments', len(oid), len(nid))
print('max abs differences:', {k: round(float(v), 5) for k, v in worst.items()})
assert worst['xy'] < 0.01 and worst['vel'] < 0.01 and worst['yaw'] < 1e-3 and worst['map'] < 0.01

# Patched relative bias: dense == original; kNN(K>=N, R=inf) == dense outside masked pairs.
from waymo.training.world_model.direct_action_flow import RelativeAgentBias
from waymo.scene_cache.sparse_agent_attention import patch_agent_attention
import types as T
torch.manual_seed(0)
ref = RelativeAgentBias(8, 128, 100.)
pose = torch.randn(2, 150, 3) * torch.tensor([60., 60., 3.]); typ = torch.randint(1, 4, (2, 150))
expected = ref(pose, typ)
class Holder(torch.nn.Module): pass
for knn, radius in ((0, 60.), (150, float('inf'))):
    m = Holder(); m.scene_encoder = Holder(); m.scene_encoder.relative_bias = ref; m.scene_encoder.agent_encoder = Holder()
    m.scene_encoder.agent_encoder.forward = lambda *a: None
    from waymo.scene_cache import sparse_agent_attention as S
    ref.knn, ref.radius_m = knn, radius
    got = S._forward(ref, pose, typ)
    print(f'knn={knn}: max |patched - original| = {(got - expected).abs().max():.2e}')
    assert torch.allclose(got, expected, atol=1e-5)
print('ALL OK')
