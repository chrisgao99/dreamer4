#!/usr/bin/env bash
set -euo pipefail
ROOT=/p/yufeng/tri30/dreamer4
SESSION=stats_dynamic_windows_h15_cuda1
if [[ ${RUN_INSIDE_TMUX:-0} != 1 ]]; then
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session exists"; exit 1; }
  printf -v command '%q ' env RUN_INSIDE_TMUX=1 bash "$(readlink -f "${BASH_SOURCE[0]}")"
  tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "Started $SESSION"
  exit 0
fi
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/training/world_model/analyze_dynamic_windows.py \
 --data_dir "$ROOT/data/waymo_vector_dataset_ooi_centered_50k" \
 --output_dir "$ROOT/waymo/eval_results/dynamic_window_distribution_h15_v05_20260916" \
 2>&1 | tee "$ROOT/waymo/logs/wm/dynamic_window_distribution_h15_v05_20260916.log"
