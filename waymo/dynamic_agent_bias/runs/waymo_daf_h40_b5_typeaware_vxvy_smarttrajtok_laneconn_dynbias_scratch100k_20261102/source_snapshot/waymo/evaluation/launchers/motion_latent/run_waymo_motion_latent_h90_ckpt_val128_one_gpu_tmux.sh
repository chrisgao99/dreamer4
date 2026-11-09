#!/usr/bin/env bash
# Evaluate one MotionLatent H90 checkpoint on the fixed val128 subset. Launch
# this script once per checkpoint/GPU to run a checkpoint sweep in parallel.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_motion_latent_shared_rollout_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
CKPT_DIR="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_v1_matched_time1_ctx10_best600k_h30_50k_exact_h90_b1_50k"

CKPT_STEP="${CKPT_STEP:?Set CKPT_STEP, for example 35000}"
CUDA_DEVICE="${CUDA_DEVICE:?Set CUDA_DEVICE, for example 0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HORIZONS="${HORIZONS:-10 30 50 80 90}"

printf -v STEP8 '%08d' "$CKPT_STEP"
STEP_K="$((CKPT_STEP / 1000))k"
EVAL_CKPT="$CKPT_DIR/step_${STEP8}.pt"

BASE_RESULT_DIR="$REPO_ROOT/waymo/eval_results/world_model/motion_latent_three_stages_shared_h90_d1_val128_seed0"
SUBSET_MANIFEST="$BASE_RESULT_DIR/val_random128_seed0_manifest.json"
RUN_GROUP="motion_latent_h90_ckpt_sweep_chunk32s30_shared_h90_d1_val128_seed0"
OUT_DIR="$REPO_ROOT/waymo/eval_results/world_model/$RUN_GROUP"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation/$RUN_GROUP"
LOG_FILE="$LOG_DIR/step_${STEP8}_cuda${CUDA_DEVICE}.log"
OUTPUT_JSON="$OUT_DIR/step_${STEP8}_ctx1_chunk32s30_shared_h90_d1_h10_30_50_80_90_val128.json"

SESSION_NAME="${SESSION_NAME:-wm_ml_h90_${STEP_K}_val128_cuda${CUDA_DEVICE}}"

for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$EVAL_CKPT" "$SUBSET_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    CKPT_STEP="$CKPT_STEP" CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" \
    HORIZONS="$HORIZONS" SESSION_NAME="$SESSION_NAME" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Checkpoint: $EVAL_CKPT"
  echo "Log: $LOG_FILE"
  echo "Results: $OUTPUT_JSON"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) MotionLatent H90 $STEP_K val128 evaluation start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES checkpoint=$EVAL_CKPT"
  echo "protocol=same_recorded_val_samples=128 batch_size=1 ctx=1 shared_rollout_horizon=90 direct_d1=1"
  echo "tokenizer_chunk_window=32 tokenizer_chunk_stride=30 ranges_for_T91=[0,32),[30,62),[59,91)"
  echo "metric_horizons=$HORIZONS subset_manifest=$SUBSET_MANIFEST"

  if [[ -f "$OUTPUT_JSON" ]]; then
    echo "Result already exists; refusing to overwrite: $OUTPUT_JSON" >&2
    exit 1
  fi

  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$EVAL_CKPT" \
    --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
    --eval_batch_size 1 --eval_max_batches 0 \
    --subset_manifest "$SUBSET_MANIFEST" --subset_size 128 --subset_seed 0 \
    --eval_seq_len 91 --eval_ctx 1 --horizons "$HORIZONS" \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
    --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0 \
    --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
    --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
    --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
    --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
    --output_json "$OUTPUT_JSON"

  echo "===== $(date) MotionLatent H90 $STEP_K val128 evaluation complete ====="
} 2>&1 | tee -a "$LOG_FILE"
