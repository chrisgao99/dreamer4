import json,os,subprocess,sys,time,hashlib
from pathlib import Path
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[3]
sys.path.insert(0,str(ROOT))
from waymo.data_prep.backfill_waymo_agent_lengths import atomic_json
from waymo.training.world_model.run_typeaware_stage1 import ensure_gpu_free
config=json.loads((RUN/'launch_config.json').read_text())
status=dict(status='running',stage='starting',pid=os.getpid(),started_unix=time.time(),cuda_index=0)
def update(stage):
    status.update(stage=stage,elapsed_seconds=time.time()-status['started_unix'])
    atomic_json(RUN/'status.json',status);print(json.dumps(status),flush=True)
def execute(args,stage,logname):
    update(stage)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=config['cuda_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (RUN/logname).open('x') as log:
        proc=subprocess.Popen([sys.executable,'waymo/training/world_model/train_waymo_direct_action_flow_mon80.py',*args],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status['child_pid']=proc.pid;update(stage)
        code=proc.wait()
    if code:raise RuntimeError(f'{stage} exited {code}: {RUN/logname}')
try:
    ensure_gpu_free(0,config['cuda_uuid'])
    args=config['train_argv'];output=Path(args[args.index('--output_dir')+1])
    if output.exists():raise FileExistsError(output)
    sources=['mon_long_rollout.py','train_waymo_direct_action_flow_mon80.py','direct_action_flow.py','action_kinematics.py']
    atomic_json(RUN/'source_sha256.json',{n:hashlib.sha256((ROOT/'waymo/training/world_model'/n).read_bytes()).hexdigest() for n in sources})
    smoke=list(args);smoke[smoke.index('--output_dir')+1]=str(RUN/'smoke_checkpoint')
    smoke+=['--max_iterations','1','--skip_validation']
    execute(smoke,'smoke_full80_backward','smoke.log')
    # Fresh fine-tuning starts from best.pt, not from the smoke-test optimizer update.
    execute(args,'training','train.log')
    status['status']='complete';update('complete');(RUN/'exit_code').write_text('0\n')
except BaseException as exc:
    status.update(status='failed',error=repr(exc));update(status['stage']);(RUN/'exit_code').write_text('1\n');raise
