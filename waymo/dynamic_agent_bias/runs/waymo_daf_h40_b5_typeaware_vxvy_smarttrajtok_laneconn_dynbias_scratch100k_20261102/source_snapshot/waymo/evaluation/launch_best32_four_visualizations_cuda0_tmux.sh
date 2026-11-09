#!/usr/bin/env bash
set -euo pipefail
ROOT=/p/yufeng/tri30/dreamer4
SESSION=vis_best32_four_intersection_cuda0
OUT="$ROOT/waymo/eval_results/world_model/best32_four_models_intersection10_n10_h80_b5_20260916"
LOG="$ROOT/waymo/logs/evaluation/best32_four_visualizations_20260916.log"
if [[ ${RUN_INSIDE_TMUX:-0} != 1 ]]; then
  tmux has-session -t "$SESSION" 2>/dev/null && { echo "Session exists: $SESSION"; exit 1; }
  printf -v command '%q ' env RUN_INSIDE_TMUX=1 bash "$(readlink -f "${BASH_SOURCE[0]}")"
  tmux new-session -d -s "$SESSION" -c "$ROOT" "$command"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "Started $SESSION; output=$OUT; log=$LOG"
  exit 0
fi
mkdir -p "$OUT" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/matplotlib-best32-four
REF="$ROOT/waymo/eval_results/world_model/h90_intersection10_two_agent_n10_ctx11_h80_20260907"
for MODEL in mon_gt_step5000 mon_physical_step2500 erd_beta100_step4750 erd_beta099_step5750; do
  echo "Starting $MODEL at $(date -Is)"
  /p/yufeng/.conda/envs/dreamer4/bin/python waymo/evaluation/visualize_direct_action_flow_intersection_two_agent_rollouts.py \
    --checkpoint "$ROOT/waymo/checkpoints/selected_best32_meanade_20260915/$MODEL.pt" \
    --val_data_dir "$ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
    --selected_manifest "$REF/selected_intersection10_manifest.json" \
    --reference_h90_dir "$REF" --output_dir "$OUT" \
    --device cuda --weights ema --focus_mode generate_all --num_workers 4 \
    --num_scenes 10 --num_rollouts 10 --rollout_steps 80 \
    --commitment_steps 5 --solver_steps 8 --seed 20260907 \
    --model_label "$MODEL (EMA)" --images_only --filename_prefix "${MODEL}_"
  echo "Finished $MODEL at $(date -Is)"
done
echo "All four visualizations complete at $(date -Is)"
