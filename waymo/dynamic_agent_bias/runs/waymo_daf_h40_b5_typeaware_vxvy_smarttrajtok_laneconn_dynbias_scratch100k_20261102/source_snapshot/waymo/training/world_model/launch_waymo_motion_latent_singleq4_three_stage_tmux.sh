#!/usr/bin/env bash
# Three-stage single-q/four-step-latent pipeline for one experiment variant.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_motion_latent_singleq4.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
READER_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_semantic_reader_agent32_zonly_d256_depth2_b8_20k_v1/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"

VARIANT="${VARIANT:?Set VARIANT=noqgt or VARIANT=qgt_detach}"
case "$VARIANT" in
  noqgt)
    Q_GT_WEIGHT=0
    DETACH_FLAG=--no-detach_q_condition
    DEFAULT_CUDA=2
    ;;
  qgt_detach)
    Q_GT_WEIGHT=1
    DETACH_FLAG=--detach_q_condition
    DEFAULT_CUDA=3
    ;;
  *)
    echo "Unsupported VARIANT=$VARIANT; expected noqgt or qgt_detach" >&2
    exit 1
    ;;
esac

CUDA_DEVICE="${CUDA_DEVICE:-$DEFAULT_CUDA}"
SESSION_NAME="${SESSION_NAME:-wm_motion_latent_singleq4_${VARIANT}_three_stage_cuda${CUDA_DEVICE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STAGE1_BATCH="${STAGE1_BATCH:-8}"
STAGE1_STEPS="${STAGE1_STEPS:-600000}"
H30_STEPS="${H30_STEPS:-30000}"
H90_STEPS="${H90_STEPS:-30000}"

PREFIX="waymo_motion_latent_singleq4_${VARIANT}_chunk32s30"
STAGE1_RUN="${PREFIX}_stage1_b${STAGE1_BATCH}_${STAGE1_STEPS}"
H30_RUN="${PREFIX}_stage2_h30_b1_${H30_STEPS}"
H90_RUN="${PREFIX}_stage3_h90_b1_${H90_STEPS}"
STAGE1_DIR="$REPO_ROOT/waymo/checkpoints/$STAGE1_RUN"
H30_DIR="$REPO_ROOT/waymo/checkpoints/$H30_RUN"
H90_DIR="$REPO_ROOT/waymo/checkpoints/$H90_RUN"
printf -v STAGE1_STEP8 '%08d' "$STAGE1_STEPS"
printf -v H30_STEP8 '%08d' "$H30_STEPS"
printf -v H90_STEP8 '%08d' "$H90_STEPS"
STAGE1_FINAL="$STAGE1_DIR/final_step_${STAGE1_STEP8}.pt"
H30_FINAL="$H30_DIR/final_step_${H30_STEP8}.pt"
H90_FINAL="$H90_DIR/final_step_${H90_STEP8}.pt"

LOG_DIR="$REPO_ROOT/waymo/logs/wm"
PIPELINE_LOG="$LOG_DIR/${PREFIX}_three_stage_pipeline.log"
STAGE1_LOG="$LOG_DIR/$STAGE1_RUN.log"
H30_LOG="$LOG_DIR/$H30_RUN.log"
H90_LOG="$LOG_DIR/$H90_RUN.log"

for path in "$PYTHON" "$TRAIN_SCRIPT" "$TOKENIZER_CKPT" "$READER_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$DATA_ROOT/train" ]] || { echo "Missing training directory: $DATA_ROOT/train" >&2; exit 1; }
mkdir -p "$STAGE1_DIR" "$H30_DIR" "$H90_DIR" "$LOG_DIR" "$REPO_ROOT/waymo/wandb"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    VARIANT="$VARIANT" CUDA_DEVICE="$CUDA_DEVICE" SESSION_NAME="$SESSION_NAME" \
    NUM_WORKERS="$NUM_WORKERS" STAGE1_BATCH="$STAGE1_BATCH" \
    STAGE1_STEPS="$STAGE1_STEPS" H30_STEPS="$H30_STEPS" H90_STEPS="$H90_STEPS" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Pipeline log: $PIPELINE_LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

common_args=(
  --data_dir "$DATA_ROOT/train"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --semantic_reader_ckpt "$READER_CKPT"
  --device cuda --seed 0 --num_workers "$NUM_WORKERS"
  --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30
  --shortcut_steps 4
  --d_model 512 --depth 8 --n_heads 8 --time_every 1 --map_cross_every 1
  --packing_factor 2 --n_register 8 --k_max 64 --kinematic_dt 0.1
  --motion_weight 1 --motion_validity_weight 0.2 --consistency_weight 0.1
  --q_gt_weight "$Q_GT_WEIGHT" --q_gt_validity_weight 0.2 "$DETACH_FLAG"
  --grad_clip 1 --amp_dtype bf16
  --ego_action_source focus --ego_action_normalization raw
  --wandb --wandb_project waymo-world-model
)

{
  echo "===== $(date) single-q/four-step pipeline start ====="
  echo "variant=$VARIANT session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "semantics=one deterministic q prediction + one hard integration + four latent shortcut steps per physical frame"
  echo "q_gt_weight=$Q_GT_WEIGHT detach_q_condition=$([[ "$DETACH_FLAG" == --detach_q_condition ]] && echo true || echo false)"
  echo "tokenizer_chunk_window=32 tokenizer_chunk_stride=30"
  echo "stage1=$STAGE1_STEPS stage2_h30=$H30_STEPS stage3_h90=$H90_STEPS"
} | tee -a "$PIPELINE_LOG"

if [[ ! -f "$STAGE1_FINAL" ]]; then
  start=()
  [[ -f "$STAGE1_DIR/latest.pt" ]] && start=(--resume "$STAGE1_DIR/latest.pt")
  "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
    --ckpt_dir "$STAGE1_DIR" "${start[@]}" \
    --batch_size "$STAGE1_BATCH" --max_steps "$STAGE1_STEPS" \
    --train_mode online_step --rollout_end 90 --max_context 10 \
    --lr 1e-4 --weight_decay 1e-2 --log_every 100 --save_every 50000 \
    --wandb_run_name "$STAGE1_RUN" \
    2>&1 | tee -a "$STAGE1_LOG"
fi
[[ -f "$STAGE1_FINAL" ]] || { echo "Stage 1 ended without $STAGE1_FINAL" >&2; exit 1; }
echo "===== $(date) Stage 1 complete; starting H30 =====" | tee -a "$PIPELINE_LOG"

if [[ ! -f "$H30_FINAL" ]]; then
  start=(--init_ckpt "$STAGE1_FINAL")
  [[ -f "$H30_DIR/latest.pt" ]] && start=(--resume "$H30_DIR/latest.pt")
  "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
    --ckpt_dir "$H30_DIR" "${start[@]}" \
    --batch_size 1 --max_steps "$H30_STEPS" \
    --train_mode rollout_stream --rollout_end 30 --max_context 10 \
    --lr 1e-5 --weight_decay 0 --log_every 20 --save_every 5000 \
    --wandb_run_name "$H30_RUN" \
    2>&1 | tee -a "$H30_LOG"
fi
[[ -f "$H30_FINAL" ]] || { echo "H30 ended without $H30_FINAL" >&2; exit 1; }
echo "===== $(date) H30 complete; starting H90 =====" | tee -a "$PIPELINE_LOG"

if [[ ! -f "$H90_FINAL" ]]; then
  start=(--init_ckpt "$H30_FINAL")
  [[ -f "$H90_DIR/latest.pt" ]] && start=(--resume "$H90_DIR/latest.pt")
  "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
    --ckpt_dir "$H90_DIR" "${start[@]}" \
    --batch_size 1 --max_steps "$H90_STEPS" \
    --train_mode rollout_stream --rollout_end 90 --max_context 10 \
    --lr 5e-6 --weight_decay 0 --log_every 20 --save_every 5000 \
    --wandb_run_name "$H90_RUN" \
    2>&1 | tee -a "$H90_LOG"
fi
[[ -f "$H90_FINAL" ]] || { echo "H90 ended without $H90_FINAL" >&2; exit 1; }
echo "===== $(date) pipeline complete =====" | tee -a "$PIPELINE_LOG"

