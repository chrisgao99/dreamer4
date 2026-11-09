#!/usr/bin/env bash
set -u
cd /p/yufeng/tri30/dreamer4
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nice -n 10 /p/yufeng/.conda/envs/dreamer4/bin/python -u \
  waymo/data_prep/backfill_waymo_agent_lengths.py \
  --source_dir /p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_ooi_centered_50k \
  --output_dir /p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_ooi_centered_50k_with_lengths \
  --workers 4 \
  --resume \
  >> /p/yufeng/tri30/dreamer4/waymo/data_prep/runs/agent_lengths_20260924/run.log 2>&1
task_status=$?
printf '%s\n' "$task_status" > /p/yufeng/tri30/dreamer4/waymo/data_prep/runs/agent_lengths_20260924/exit_code
exit "$task_status"
