#!/usr/bin/env bash
set -u
cd /p/yufeng/tri30/dreamer4
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
/p/yufeng/.conda/envs/dreamer4/bin/python -u waymo/data_prep/runs/type_aware_h40_vxvy_20260925/launch.py >> waymo/data_prep/runs/type_aware_h40_vxvy_20260925/run.log 2>&1
task_status=$?
printf '%s\n' "$task_status" > waymo/data_prep/runs/type_aware_h40_vxvy_20260925/exit_code
exit "$task_status"
