#!/usr/bin/env bash
set -u
cd /p/yufeng/tri30/dreamer4
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
/p/yufeng/.conda/envs/dreamer4/bin/python -u \
  waymo/training/world_model/run_typeaware_stage1.py \
  --config waymo/data_prep/runs/type_aware_h40_20260924/launch_config.json \
  >> waymo/data_prep/runs/type_aware_h40_20260924/run.log 2>&1
task_status=$?
printf '%s\n' "$task_status" > waymo/data_prep/runs/type_aware_h40_20260924/exit_code
exit "$task_status"
