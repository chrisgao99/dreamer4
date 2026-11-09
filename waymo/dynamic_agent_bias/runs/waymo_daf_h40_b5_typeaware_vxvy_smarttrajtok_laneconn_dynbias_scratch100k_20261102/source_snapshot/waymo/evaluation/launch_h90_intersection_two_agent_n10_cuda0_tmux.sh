#!/usr/bin/env bash
# Render ten intersection-rich scenes: one recorded ego plan, one target GT,
# and ten stochastic target-agent rollouts per figure.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SESSION_NAME="${SESSION_NAME:-wm_h90_intersection_two_agent_n10_cuda0}"
SCRIPT="$REPO_ROOT/waymo/evaluation/visualize_h90_intersection_two_agent_rollouts.py"
TOKENIZER="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
CHECKPOINT="$REPO_ROOT/waymo/checkpoints/n8_motion_h30_21k_h90_27k/h90_step27k/step_00027000.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
MANIFEST="$REPO_ROOT/waymo/evaluation/val_random128_seed0_manifest.json"
OUT_DIR="$REPO_ROOT/waymo/eval_results/world_model/h90_intersection10_two_agent_n10_ctx11_h80_20260907"
LOG="$REPO_ROOT/waymo/logs/wm/wm_h90_intersection10_two_agent_n10_cuda0.log"

for path in "$PYTHON" "$SCRIPT" "$TOKENIZER" "$CHECKPOINT" "$MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
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
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $LOG"
  echo "Output: $OUT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-h90-intersection}"

{
  echo "===== $(date) h90 intersection two-agent N=10 visualization start ====="
  echo "cuda=$CUDA_DEVICE checkpoint=$CHECKPOINT"
  echo "protocol=source_val128 select_intersection10 ctx11 future80 D1 rollouts10"

  "$PYTHON" "$SCRIPT" \
    --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
    --subset_manifest "$MANIFEST" --source_subset_size 128 --num_scenes 10 \
    --tokenizer_ckpt "$TOKENIZER" --eval_ckpt "$CHECKPOINT" \
    --output_dir "$OUT_DIR" --device cuda --num_workers 4 \
    --eval_seq_len 91 --eval_ctx 11 --horizon 80 --min_future_valid_steps 80 \
    --eval_num_rollouts 10 --eval_multisample_seed 20260907 \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
    --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0 \
    --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
    --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
    --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4 \
    --use_ego_actions --ego_action_source focus --ego_action_normalization raw \
    --no-ego_action_clamp --agent_far_weight 1.0 --agent_near_radius_m 50 \
    --agent_distance_source focus

  echo "===== $(date) h90 intersection two-agent N=10 visualization complete ====="
} 2>&1 | tee -a "$LOG"
