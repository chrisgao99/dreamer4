import csv
import fcntl
import json
import sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
r = json.loads((out / 'result.json').read_text())
for key, expected in {'scene_count':512, 'num_rollouts':32, 'history_context_steps':11, 'model_plan_horizon':40, 'rollout_commitment_steps':5, 'rollout_steps':80}.items():
    assert r[key] == expected, (key, r[key])
parts = {h: [] for h in (20,40,60,80)}
for path in sorted((out / 'rollout_batches').glob('batch_*.npz')):
    with np.load(path) as d:
        dist = np.linalg.norm(d['poses'][..., :2] - d['gt_pose'][:, None, ..., :2], axis=-1)
        valid = d['valid']
        for h in parts:
            weights = valid[..., :h]
            ade = (dist[..., :h] * weights[:, None]).sum(axis=(2,3)) / np.maximum(weights.sum(axis=(1,2))[:,None],1)
            parts[h].append(ade)
prefix = {}
for h, chunks in parts.items():
    values = np.concatenate(chunks)
    assert values.shape == (512,32), values.shape
    prefix[str(h)] = {'mean_ade_m':float(values.mean()), 'minade_m':float(values.min(axis=1).mean())}
assert np.isclose(prefix['80']['mean_ade_m'], r['mean_ade_m'], rtol=1e-5)
assert np.isclose(prefix['80']['minade_m'], r['minade_m'], rtol=1e-5)
(out / 'prefix_ade.json').write_text(json.dumps({'definition':'Cumulative future steps 1..H; valid agent-time weighted within scene, equal scene and rollout weights; minADE selects best joint rollout per scene.', 'horizons':prefix},indent=2)+'\n')
row = {'experiment':out.name, 'checkpoint_step':r['checkpoint_step'], 'scenes':r['scene_count'], 'mean_ade_m':r['mean_ade_m'], 'minade_at_32_m':r['minade_m'], 'cpd':r['flow_erd_cpd']}
for h, metrics in prefix.items():
    row.update({f'first_{h}_{k}':v for k,v in metrics.items()})
(out / 'summary.json').write_text(json.dumps(row,indent=2)+'\n')
with (out.parent / '.summary.lock').open('w') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    rows = [json.loads(p.read_text()) for p in sorted(out.parent.glob('*/summary.json'))]
    temp = out.parent / 'summary.csv.tmp'
    with temp.open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(rows)
    temp.replace(out.parent / 'summary.csv')
print(json.dumps(row),flush=True)
