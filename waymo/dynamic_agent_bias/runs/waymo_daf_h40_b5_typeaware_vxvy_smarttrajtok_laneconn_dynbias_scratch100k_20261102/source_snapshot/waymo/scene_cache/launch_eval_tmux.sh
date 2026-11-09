#!/usr/bin/env bash
# Detached tmux run of eval_scene_ade.py (survives logout). Usage:
#   CUDA_DEVICE=1 bash waymo/scene_cache/launch_eval_tmux.sh
set -euo pipefail

REPO=/p/yufeng/tri30/dreamer4
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
CKPT="${CKPT:-$REPO/waymo/checkpoints/waymo_daf_laneconn90k_mon8_gt_unroll80_b5_detach0_10k_20261009/best_ade.pt}"
RUN_NAME="${RUN_NAME:-scene_ade_laneconn90k_mon8_best_val256r512l_$(date +%Y%m%d)}"
OUT="$REPO/waymo/eval_results/scene_cache/$RUN_NAME"
SESSION="${SESSION:-scene_ade_cuda${CUDA_DEVICE}}"
EXTRA_ARGS="${EXTRA_ARGS:---n_random 256 --n_large 512 --large_min 64 --variants dense knn32_r60}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION exists: tmux attach -t $SESSION" >&2; exit 1
fi
if [ -e "$OUT/summary.json" ]; then
  echo "$OUT already has results; set RUN_NAME" >&2; exit 1
fi
mkdir -p "$OUT"
tmux new-session -d -s "$SESSION" bash -c "
  cd $REPO
  export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=4
  { echo \"start \$(date) host \$(hostname) gpu $CUDA_DEVICE\"
    $PYTHON waymo/scene_cache/eval_scene_ade.py --checkpoint $CKPT --output_dir $OUT $EXTRA_ARGS
    echo \"exit \$? at \$(date)\"; } > $OUT/run.log 2>&1
"
echo "Started tmux session $SESSION on GPU $CUDA_DEVICE"
echo "Log: $OUT/run.log"
echo "Results: $OUT/summary.json, $OUT/per_scene_<variant>.jsonl"
