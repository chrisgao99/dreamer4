import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
sys.path.insert(0,str(ROOT))
from waymo.data_prep.backfill_waymo_agent_lengths import atomic_json
from waymo.training.world_model.run_typeaware_stage1 import ensure_gpu_free

UUID='GPU-6cc98813-604a-2fe9-3f99-ff2c3c890bed'
jobs=json.loads((OUT/'jobs.json').read_text())
status=dict(status='running',started_unix=time.time(),pid=os.getpid(),cuda_index=0,
            jobs={j['label']:'queued' for j in jobs})
def update():
    atomic_json(OUT/'queue_status.json',status)
    print(json.dumps(status),flush=True)

def evaluate(job,smoke):
    dest=OUT/('smoke_'+job['label'] if smoke else job['label'])
    dest.mkdir(exist_ok=False)
    args=[sys.executable,'waymo/evaluation/eval_waymo_direct_action_flow_multisample.py',
        '--checkpoint',job['checkpoint'],'--val_data_dir',str(ROOT/'data/waymo_vector_dataset_ooi_centered_50k_with_lengths/val'),
        '--output_json',str(dest/'result.json'),'--rollout_ade_csv',str(dest/'rollout_ade.csv'),
        '--scene_metrics_csv',str(dest/'scene_metrics.csv'),'--details_npz',str(dest/'details.npz'),
        '--device','cuda','--weights','ema','--focus_mode','generate_all','--eval_batch_size','4',
        '--eval_max_batches','1' if smoke else '128','--num_workers','4',
        '--num_rollouts','2' if smoke else '32','--rollout_steps','80','--commitment_steps','5',
        '--solver_steps','8','--seed','12346','--log_every','1','--physical_metrics',
        '--metric_validity','holonomic','--cpd_type_scales','25.423524547767016','4.925798640017075','18.77286026475977']
    atomic_json(dest/'command.json',args)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',
             MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with (dest/'run.log').open('w') as log:
        process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status.update(active_job=job['label'],stage='smoke' if smoke else 'evaluation',child_pid=process.pid)
        if not smoke:status['jobs'][job['label']]='running'
        update()
        code=process.wait()
    (dest/'exit_code').write_text(str(code)+'\n')
    if code:raise RuntimeError(f'{job["label"]} {"smoke" if smoke else "evaluation"} failed: exit {code}')
    result=json.loads((dest/'result.json').read_text())
    if result['scene_count']!=(4 if smoke else 512):raise ValueError('Unexpected scene count')
    if not smoke:
        status['jobs'][job['label']]='complete';update()
    return result

try:
    ensure_gpu_free(0,UUID)
    update()
    for job in jobs:evaluate(job,True)
    summary={}
    for job in jobs:
        result=evaluate(job,False)
        summary[job['label']]={k:result[k] for k in ['checkpoint','checkpoint_step','scene_count','num_rollouts',
            'rollout_steps','rollout_commitment_steps','metric_validity','minade_m','mean_ade_m','flow_erd_cpd','first40_mean_ade_m']}
        atomic_json(OUT/'summary.json',summary)
    status.update(status='complete',stage='complete',elapsed_seconds=time.time()-status['started_unix']);update()
except BaseException as exc:
    status.update(status='failed',error=repr(exc));update();raise
