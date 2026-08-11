#!/usr/bin/env bash
# Paired action-sensitivity stress test for the stage-3 step-40k checkpoint.
# The control and treatment share initial latent state, validation scene, and
# rollout noise.  Only future focus actions differ.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_extreme_random_focus_actions.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
WORLD_MODEL_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage/waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k/step_00040000.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
SUBSET_MANIFEST="$REPO_ROOT/waymo/evaluation/val_random128_seed0_manifest.json"

CUDA_DEVICE="${CUDA_DEVICE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
RANDOM_ACTION_SEED="${RANDOM_ACTION_SEED:-20260807}"
SESSION_NAME="${SESSION_NAME:-wm_step40k_extreme_random_actions_cuda1}"
RUN_ID="${RUN_ID:-world_model_step40k_extreme_random_focus_actions_val128_seed${RANDOM_ACTION_SEED}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model/$RUN_ID}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/paired_extreme_random_focus_actions_h10_30_50_80_90.json}"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

# Deliberately far outside normal road motion.  Each field and timestep is
# sampled independently; negative speed and inconsistent speed/vx/vy are kept.
RANDOM_DELTA_XY_MAX_M="${RANDOM_DELTA_XY_MAX_M:-50}"
RANDOM_DELTA_YAW_MAX_RAD="${RANDOM_DELTA_YAW_MAX_RAD:-3.141592653589793}"
RANDOM_SPEED_ABS_MAX_MPS="${RANDOM_SPEED_ABS_MAX_MPS:-200}"
RANDOM_VELOCITY_ABS_MAX_MPS="${RANDOM_VELOCITY_ABS_MAX_MPS:-200}"

for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$WORLD_MODEL_CKPT" "$SUBSET_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux is not available" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" \
    EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES" RANDOM_ACTION_SEED="$RANDOM_ACTION_SEED" \
    SESSION_NAME="$SESSION_NAME" RUN_ID="$RUN_ID" OUT_DIR="$OUT_DIR" \
    OUTPUT_JSON="$OUTPUT_JSON" LOG_FILE="$LOG_FILE" \
    RANDOM_DELTA_XY_MAX_M="$RANDOM_DELTA_XY_MAX_M" \
    RANDOM_DELTA_YAW_MAX_RAD="$RANDOM_DELTA_YAW_MAX_RAD" \
    RANDOM_SPEED_ABS_MAX_MPS="$RANDOM_SPEED_ABS_MAX_MPS" \
    RANDOM_VELOCITY_ABS_MAX_MPS="$RANDOM_VELOCITY_ABS_MAX_MPS" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started detached tmux session: $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Results: $OUTPUT_JSON"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) paired extreme-random focus-action evaluation start ====="
  echo "physical_cuda=$CUDA_DEVICE checkpoint=$WORLD_MODEL_CKPT"
  echo "protocol=fixed_val128 ctx1 shared_H90 shortcut4 paired_noise horizons=10,30,50,80,90"
  echo "random_action_seed=$RANDOM_ACTION_SEED delta_xy_each=+/-${RANDOM_DELTA_XY_MAX_M}m delta_yaw=+/-${RANDOM_DELTA_YAW_MAX_RAD}rad speed=+/-${RANDOM_SPEED_ABS_MAX_MPS}mps vxvy_each=+/-${RANDOM_VELOCITY_ABS_MAX_MPS}mps"

  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$WORLD_MODEL_CKPT" \
    --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
    --eval_batch_size 1 --eval_max_batches "$EVAL_MAX_BATCHES" \
    --subset_manifest "$SUBSET_MANIFEST" --subset_size 128 --subset_seed 0 \
    --random_action_seed "$RANDOM_ACTION_SEED" \
    --random_delta_xy_max_m "$RANDOM_DELTA_XY_MAX_M" \
    --random_delta_yaw_max_rad "$RANDOM_DELTA_YAW_MAX_RAD" \
    --random_speed_abs_max_mps "$RANDOM_SPEED_ABS_MAX_MPS" \
    --random_velocity_abs_max_mps "$RANDOM_VELOCITY_ABS_MAX_MPS" \
    --eval_seq_len 91 --eval_ctx 1 --horizons "10 30 50 80 90" \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
    --max_rollout_window 11 --eval_schedule shortcut --eval_d 0.25 \
    --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
    --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
    --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
    --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
    --progress_every 8 --output_json "$OUTPUT_JSON"

  echo "===== $(date) paired extreme-random focus-action evaluation complete ====="
} 2>&1 | tee -a "$LOG_FILE"
