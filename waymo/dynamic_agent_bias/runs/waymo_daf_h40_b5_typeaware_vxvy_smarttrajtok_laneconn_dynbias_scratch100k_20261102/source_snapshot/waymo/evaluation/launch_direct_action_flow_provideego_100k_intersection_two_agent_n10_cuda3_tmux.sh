#!/usr/bin/env bash
# Run the fixed ten-scene paired ego/target visualization with provideego 100k checkpoint.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
SESSION_NAME="${SESSION_NAME:-daf100k_provideego_intersection_n10_gpu3}"
SCRIPT="$REPO_ROOT/waymo/evaluation/visualize_direct_action_flow_intersection_two_agent_rollouts.py"
CHECKPOINT="$REPO_ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_h15_b5_provideego_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5/final_step_000100000.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
H90_DIR="$REPO_ROOT/waymo/eval_results/world_model/h90_intersection10_two_agent_n10_ctx11_h80_20260907"
SELECTED_MANIFEST="$H90_DIR/selected_intersection10_manifest.json"
OUT_DIR="$REPO_ROOT/waymo/eval_results/world_model/direct_action_flow_provideego_100k_intersection10_two_agent_n10_ctx11_h80_20260911"
LOG="$REPO_ROOT/waymo/logs/wm/wm_direct_action_flow_provideego_100k_intersection10_two_agent_n10_cuda3.log"

for path in "$PYTHON" "$SCRIPT" "$CHECKPOINT" "$SELECTED_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
[[ ! -e "$OUT_DIR/rollout_metrics.json" ]] || { echo "Results already exist: $OUT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    CUDA_DEVICE="$CUDA_DEVICE" SESSION_NAME="$SESSION_NAME" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
  tmux set-window-option -t "$SESSION_NAME:0" remain-on-exit on
  tmux respawn-pane -k -t "$SESSION_NAME:0.0" "$tmux_command"
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $LOG"
  echo "Output: $OUT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-daf-provideego-intersection}"

{
  echo "===== $(date) DirectActionFlow provideego intersection paired rollout visualization start ====="
  echo "cuda=$CUDA_DEVICE checkpoint=$CHECKPOINT weights=ema checkpoint_step=100000"
  echo "protocol=same_intersection10 ctx11 native_h15 commitment5 receding_h80 solver8 paired_ego_target rollouts10"

  "$PYTHON" "$SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --val_data_dir "$VAL_DATA" \
    --selected_manifest "$SELECTED_MANIFEST" \
    --reference_h90_dir "$H90_DIR" \
    --output_dir "$OUT_DIR" \
    --device cuda --weights ema --focus_mode conditioned --num_workers 4 \
    --num_scenes 10 --num_rollouts 10 --rollout_steps 80 \
    --commitment_steps 5 --solver_steps 8 --seed 20260907 \
    --model_label "DirectActionFlow provideego 100k (EMA)"

  echo "===== $(date) DirectActionFlow provideego intersection paired rollout visualization complete ====="
} 2>&1 | tee -a "$LOG"
