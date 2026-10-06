#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
SESSION_NAME="${SESSION_NAME:-eval_best32_mon_b5_phys_cuda1}"
OUT_ROOT="$REPO_ROOT/waymo/eval_results/world_model/best32_four_models_h80_b5_k32_val128_physical_20260915"
CKPT_ROOT="$REPO_ROOT/waymo/checkpoints/selected_best32_meanade_20260915"
DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
EVAL="$REPO_ROOT/waymo/evaluation/eval_waymo_direct_action_flow_multisample.py"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
items=(mon_gt_step5000 mon_physical_step2500)
for item in "${items[@]}"; do [[ -f "$CKPT_ROOT/$item.pt" ]] || { echo "Missing $item checkpoint";exit 1;};done
[[ -d "$DATA" && -f "$EVAL" && -x "$PYTHON" ]] || { echo 'Missing evaluator, data or Python';exit 1;}
if [[ "${RUN_INSIDE_TMUX:-0}" != 1 ]]; then
  if tmux has-session -t "=$SESSION_NAME" 2>/dev/null;then echo "Session exists: $SESSION_NAME";exit 1;fi
  printf -v command '%q ' env RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" CUDA_DEVICE="$CUDA_DEVICE" SESSION_NAME="$SESSION_NAME" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
  tmux set-window-option -t "$SESSION_NAME:0" remain-on-exit on
  tmux respawn-pane -k -t "$SESSION_NAME:0.0" "$command"
  echo "Started $SESSION_NAME on CUDA $CUDA_DEVICE: ${items[*]}"
  echo "Results: $OUT_ROOT"
  exit 0
fi
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG_DIR/best32_mon_b5_physical_queue_20260915.log") 2>&1
for item in "${items[@]}";do
  out="$OUT_ROOT/$item"
  if [[ -f "$out/result.json" ]];then echo "Already complete: $item";continue;fi
  mkdir -p "$out"
  echo "===== $(date -Is) $item CUDA=$CUDA_DEVICE B5 K32 scenes512 ====="
  "$PYTHON" "$EVAL" --checkpoint "$CKPT_ROOT/$item.pt" --val_data_dir "$DATA" \
    --output_json "$out/result.json" --rollout_ade_csv "$out/rollout_ade.csv" \
    --scene_metrics_csv "$out/scene_metrics.csv" --details_npz "$out/details.npz" \
    --device cuda --weights ema --focus_mode generate_all --eval_batch_size 4 --eval_max_batches 128 \
    --num_workers 4 --num_rollouts 32 --rollout_steps 80 --commitment_steps 5 --solver_steps 8 \
    --seed 12346 --log_every 1 --physical_metrics \
    --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977 \
    2>&1 | tee -a "$LOG_DIR/best32_${item}_b5_physical_20260915.log"
  "$PYTHON" "$REPO_ROOT/waymo/evaluation/summarize_best32_physical.py" "$OUT_ROOT"
done
echo "===== $(date -Is) mon queue complete ====="
