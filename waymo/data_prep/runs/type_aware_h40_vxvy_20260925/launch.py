"""Launch the velocity-input ablation using immutable, already-prepared inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[3]
sys.path.insert(0,str(ROOT))
from waymo.data_prep.backfill_waymo_agent_lengths import atomic_json
from waymo.training.world_model.run_typeaware_stage1 import prepare_training,preflight,ensure_gpu_free


def main():
    config=json.loads((RUN/'launch_config.json').read_text())
    started=time.time()
    status=dict(status='running',pid=os.getpid(),started_unix=started,cuda_index=config['cuda_index'],cuda_uuid=config['cuda_uuid'])
    def stage(value):
        status.update(stage=value,elapsed_seconds=round(time.time()-started,2))
        atomic_json(RUN/'status.json',status)
        print(json.dumps(status),flush=True)
    try:
        stage('verify_prepared_inputs')
        checkpoint=Path(config['checkpoint_dir'])
        if checkpoint.exists() and any(checkpoint.iterdir()):raise FileExistsError(checkpoint)
        summary=json.loads((Path(config['data_root'])/'length_supplement_summary.json').read_text())
        if summary['status']!='complete' or summary['first_required_frame']!=10:raise ValueError('Lengths not ready for L11')
        baseline=json.loads((ROOT/'waymo/data_prep/runs/type_aware_h40_20260924/launch_config.json').read_text())
        argv=config['train_argv'];old=baseline['train_argv']
        expected=list(old)
        expected[expected.index('--ckpt_dir')+1]=config['checkpoint_dir']
        expected[expected.index('--wandb_run_name')+1]=config['run_name']
        expected+=['--include_agent_velocity','--velocity_scale_mps','10.0']
        if argv!=expected:raise ValueError('Unexpected baseline configuration changes')
        stats_path=Path(argv[argv.index('--action_stats_path')+1])
        inputs=[stats_path,Path(config['calibration_path']),Path(config['data_root'])/'length_supplement_summary.json']
        fingerprints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
        atomic_json(RUN/'config_comparison.json',dict(baseline=baseline['run_name'],same_training_arguments=True,
            exceptions=['ckpt_dir','wandb_run_name','include_agent_velocity=true','velocity_scale_mps=10'],
            reused_inputs_sha256=fingerprints))
        stage('preflight')
        args,stats=prepare_training(config)
        if not args.include_agent_velocity:raise ValueError('Velocity input disabled')
        atomic_json(RUN/'resolved_train_args.json',vars(args))
        preflight(args,stats,RUN)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=sha for p,sha in fingerprints.items()):
            raise ValueError('Shared preparation inputs changed')
        source_files=['waymo/training/world_model/direct_action_flow.py',
            'waymo/training/world_model/train_waymo_direct_action_flow.py',
            'waymo/training/world_model/action_kinematics.py']
        atomic_json(RUN/'source_sha256.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files})
        ensure_gpu_free(config['cuda_index'],config['cuda_uuid'])
        stage('training')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=config['cuda_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline',
            WANDB_DIR=str(ROOT/'waymo/wandb'),PYTHONUNBUFFERED='1')
        subprocess.run([sys.executable,'waymo/training/world_model/train_waymo_direct_action_flow.py',*argv],
            cwd=ROOT,env=env,check=True)
        status['status']='complete';stage('complete')
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));stage(status['stage']);raise

if __name__=='__main__':main()
