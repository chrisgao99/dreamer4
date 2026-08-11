#!/usr/bin/env bash
# Restart MotionLatent H90 exact-rollout fine-tuning from the H30 final
# checkpoint, with tokenizer encoding matched to chunk32/stride30 evaluation.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_motion_latent_v1.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
READER_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_semantic_reader_agent32_zonly_d256_depth2_b8_20k_v1/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
INIT_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_v1_matched_time1_ctx10_best600k_exact_h30_b1_50k/final_step_00050000.pt"

RUN_NAME="waymo_motion_latent_v1_h30final_exact_h90_chunk32s30_b1_30k"
CKPT_DIR="$REPO_ROOT/waymo/checkpoints/$RUN_NAME"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"
LOG_FILE="$LOG_DIR/$RUN_NAME.log"
PIPELINE_LOG="$LOG_DIR/${RUN_NAME}_pipeline.log"

SESSION_NAME="${SESSION_NAME:-wm_motion_latent_h30final_h90_chunk32s30_30k_cuda1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

for path in "$PYTHON" "$TRAIN_SCRIPT" "$TOKENIZER_CKPT" "$READER_CKPT" "$INIT_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$DATA_ROOT/train" ]] || { echo "Missing training directory: $DATA_ROOT/train" >&2; exit 1; }
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$REPO_ROOT/waymo/wandb"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    SESSION_NAME="$SESSION_NAME" CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Training log: $LOG_FILE"
  echo "Pipeline log: $PIPELINE_LOG"
  echo "Checkpoints: $CKPT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

{
  echo "===== $(date) MotionLatent H90 chunk32s30 30k training start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "run=$RUN_NAME init_ckpt=$INIT_CKPT"
  echo "objective=exact_rollout ctx=1 horizon=90 batch=1 steps=30000 lr=5e-6"
  echo "tokenizer_chunk_window=32 tokenizer_chunk_stride=30 ranges_for_T91=[0,32),[30,62),[59,91)"
} | tee -a "$PIPELINE_LOG"

start=(--init_ckpt "$INIT_CKPT")
if [[ -f "$CKPT_DIR/latest.pt" ]]; then
  start=(--resume "$CKPT_DIR/latest.pt")
fi

"$PYTHON" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_ROOT/train" \
  --tokenizer_ckpt "$TOKENIZER_CKPT" --semantic_reader_ckpt "$READER_CKPT" \
  --ckpt_dir "$CKPT_DIR" "${start[@]}" \
  --device cuda --seed 0 --batch_size 1 --num_workers "$NUM_WORKERS" \
  --max_steps 30000 --train_mode exact_rollout --rollout_end 90 --max_context 10 \
  --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
  --d_model 512 --depth 8 --n_heads 8 --time_every 1 --map_cross_every 1 \
  --packing_factor 2 --n_register 8 --k_max 64 --kinematic_dt 0.1 \
  --motion_weight 1 --motion_validity_weight 0.2 --consistency_weight 0.1 \
  --lr 5e-6 --weight_decay 0 --grad_clip 1 --amp_dtype bf16 \
  --log_every 20 --save_every 5000 \
  --ego_action_source focus --ego_action_normalization raw \
  --wandb --wandb_project waymo-world-model --wandb_run_name "$RUN_NAME" \
  2>&1 | tee -a "$LOG_FILE"

final_ckpt="$CKPT_DIR/final_step_00030000.pt"
[[ -f "$final_ckpt" ]] || { echo "Training ended without $final_ckpt" >&2; exit 1; }
echo "===== $(date) MotionLatent H90 chunk32s30 30k training complete =====" | tee -a "$PIPELINE_LOG"
