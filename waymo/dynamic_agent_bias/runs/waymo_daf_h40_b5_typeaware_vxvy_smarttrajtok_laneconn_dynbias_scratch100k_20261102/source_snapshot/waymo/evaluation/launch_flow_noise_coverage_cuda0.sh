#!/usr/bin/env bash
set -uo pipefail
cd /p/yufeng/tri30/dreamer4
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mpl-flow-noise-coverage
PYTHON=/p/yufeng/.conda/envs/dreamer4/bin/python
BASE=waymo/eval_results/world_model
for mode in random32 paired16_active16; do
    out="$BASE/typeaware_vxvy_ft1000_intersection10_${mode}_plan40_b40_h80_20261002"
    mkdir -p "$out"
    "$PYTHON" waymo/evaluation/visualize_flow_noise_coverage.py --mode "$mode" --output_dir "$out" > "$out/run.log" 2>&1
    code=$?
    echo "$code" > "$out/exit_code.txt"
    if [ "$code" -ne 0 ]; then
        echo '{"state":"failed","see":"run.log"}' > "$out/status.json"
    fi
done
