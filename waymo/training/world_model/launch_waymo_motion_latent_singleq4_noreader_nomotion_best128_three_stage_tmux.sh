#!/usr/bin/env bash
# Three-stage single-q experiment with no reader/motion loss and validation-gated best.pt.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_motion_latent_singleq4.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
VAL_MANIFEST="$REPO_ROOT/waymo/evaluation/val_random128_seed0_manifest.json"

VARIANT="${VARIANT:?Set VARIANT=noqgt or VARIANT=qgt_after_integrate}"
case "$VARIANT" in
  noqgt)
    Q_GT_WEIGHT=0
    DETACH_FLAG=--no-detach_q_condition
    DEFAULT_CUDA=2
    ;;
  qgt_after_integrate)
    Q_GT_WEIGHT=1
    DETACH_FLAG=--detach_q_condition
    DEFAULT_CUDA=3
    ;;
  *)
    echo "Unsupported VARIANT=$VARIANT; expected noqgt or qgt_after_integrate" >&2
    exit 1
    ;;
esac

CUDA_DEVICE="${CUDA_DEVICE:-$DEFAULT_CUDA}"
SESSION_NAME="${SESSION_NAME:-wm_singleq4_noreader_nomotion_${VARIANT}_cuda${CUDA_DEVICE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
STAGE1_STEPS="${STAGE1_STEPS:-300000}"
H30_STEPS="${H30_STEPS:-4000}"
H90_STEPS="${H90_STEPS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

PREFIX="waymo_motion_latent_singleq4_noreader_nomotion_${VARIANT}_gtvalid_best128_chunk32s30"
STAGE1_RUN="${PREFIX}_stage1_b${BATCH_SIZE}_${STAGE1_STEPS}"
H30_RUN="${PREFIX}_stage2_h30_b${BATCH_SIZE}_${H30_STEPS}"
H90_RUN="${PREFIX}_stage3_h90_b${BATCH_SIZE}_${H90_STEPS}"
STAGE1_DIR="$REPO_ROOT/waymo/checkpoints/$STAGE1_RUN"
H30_DIR="$REPO_ROOT/waymo/checkpoints/$H30_RUN"
H90_DIR="$REPO_ROOT/waymo/checkpoints/$H90_RUN"
STAGE1_DONE="$STAGE1_DIR/STAGE_COMPLETE"
H30_DONE="$H30_DIR/STAGE_COMPLETE"
H90_DONE="$H90_DIR/STAGE_COMPLETE"

LOG_DIR="$REPO_ROOT/waymo/logs/wm"
PIPELINE_LOG="$LOG_DIR/${PREFIX}_three_stage_pipeline.log"
STAGE1_LOG="$LOG_DIR/$STAGE1_RUN.log"
H30_LOG="$LOG_DIR/$H30_RUN.log"
H90_LOG="$LOG_DIR/$H90_RUN.log"

for path in "$PYTHON" "$TRAIN_SCRIPT" "$TOKENIZER_CKPT" "$VAL_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$DATA_ROOT/train" ]] || { echo "Missing training directory: $DATA_ROOT/train" >&2; exit 1; }
[[ -d "$DATA_ROOT/val" ]] || { echo "Missing validation directory: $DATA_ROOT/val" >&2; exit 1; }
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
    NUM_WORKERS="$NUM_WORKERS" VAL_NUM_WORKERS="$VAL_NUM_WORKERS" \
    STAGE1_STEPS="$STAGE1_STEPS" H30_STEPS="$H30_STEPS" H90_STEPS="$H90_STEPS" \
    BATCH_SIZE="$BATCH_SIZE" bash "$SCRIPT_PATH"
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
  --device cuda --seed 0 --num_workers "$NUM_WORKERS"
  --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30
  --shortcut_steps 4
  --d_model 512 --depth 8 --n_heads 8 --time_every 1 --map_cross_every 1
  --packing_factor 2 --n_register 8 --k_max 64 --kinematic_dt 0.1
  --motion_weight 0 --consistency_weight 0
  --q_gt_weight "$Q_GT_WEIGHT" --q_gt_validity_weight 0.2 "$DETACH_FLAG"
  --grad_clip 1 --amp_dtype bf16
  --ego_action_source focus --ego_action_normalization raw
  --val_data_dir "$DATA_ROOT/val"
  --val_manifest "$VAL_MANIFEST"
  --val_subset_size 128 --val_batch_size 8 --val_num_workers "$VAL_NUM_WORKERS"
  --val_seed 0 --best_metric decoder_xy_mae_m --best_only
  --wandb --wandb_project waymo-world-model
)

{
  echo "===== $(date) no-reader/no-motion three-stage pipeline start ====="
  echo "variant=$VARIANT session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "motion_weight=0 consistency_weight=0 reader=disabled"
  echo "q_gt_weight=$Q_GT_WEIGHT detach_q_condition=$([[ "$DETACH_FLAG" == --detach_q_condition ]] && echo true || echo false)"
  echo "GT-invalid continuous features are masked; validity classification is retained"
  echo "validation=context1 fixed_manifest=$VAL_MANIFEST samples=128 includes_focus=true continuous_mask=GT-valid"
  echo "best_metric=decoder_xy_mae_m; stage1_h90=$STAGE1_STEPS stage2_h30=$H30_STEPS stage3_h90=$H90_STEPS batch=$BATCH_SIZE"
} | tee -a "$PIPELINE_LOG"

if [[ ! -f "$STAGE1_DONE" ]]; then
  start=()
  [[ -f "$STAGE1_DIR/best.pt" ]] && start=(--resume "$STAGE1_DIR/best.pt")
  "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
    --ckpt_dir "$STAGE1_DIR" "${start[@]}" \
    --batch_size "$BATCH_SIZE" --max_steps "$STAGE1_STEPS" \
    --train_mode online_step --rollout_end 90 --max_context 10 \
    --best_eval_horizon 90 \
    --lr 1e-4 --weight_decay 1e-2 --log_every 100 --save_every 50000 \
    --wandb_run_name "$STAGE1_RUN" \
    2>&1 | tee -a "$STAGE1_LOG"
  [[ -f "$STAGE1_DIR/best.pt" ]] || { echo "Stage 1 ended without best.pt" >&2; exit 1; }
  touch "$STAGE1_DONE"
fi
echo "===== $(date) Stage 1 complete; starting H30 from Stage 1 best.pt =====" | tee -a "$PIPELINE_LOG"

if [[ ! -f "$H30_DONE" ]]; then
  start=(--init_ckpt "$STAGE1_DIR/best.pt")
  [[ -f "$H30_DIR/best.pt" ]] && start=(--resume "$H30_DIR/best.pt")
  "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
    --ckpt_dir "$H30_DIR" "${start[@]}" \
    --batch_size "$BATCH_SIZE" --max_steps "$H30_STEPS" \
    --train_mode rollout_stream --rollout_end 30 --max_context 10 \
    --best_eval_horizon 30 \
    --lr 1e-5 --weight_decay 0 --log_every 20 --save_every 1000 \
    --wandb_run_name "$H30_RUN" \
    2>&1 | tee -a "$H30_LOG"
  [[ -f "$H30_DIR/best.pt" ]] || { echo "H30 ended without best.pt" >&2; exit 1; }
  touch "$H30_DONE"
fi
echo "===== $(date) H30 complete; starting H90 from H30 best.pt =====" | tee -a "$PIPELINE_LOG"

if [[ ! -f "$H90_DONE" ]]; then
  start=(--init_ckpt "$H30_DIR/best.pt")
  [[ -f "$H90_DIR/best.pt" ]] && start=(--resume "$H90_DIR/best.pt")
  "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
    --ckpt_dir "$H90_DIR" "${start[@]}" \
    --batch_size "$BATCH_SIZE" --max_steps "$H90_STEPS" \
    --train_mode rollout_stream --rollout_end 90 --max_context 10 \
    --best_eval_horizon 90 \
    --lr 5e-6 --weight_decay 0 --log_every 20 --save_every 1000 \
    --wandb_run_name "$H90_RUN" \
    2>&1 | tee -a "$H90_LOG"
  [[ -f "$H90_DIR/best.pt" ]] || { echo "H90 ended without best.pt" >&2; exit 1; }
  touch "$H90_DONE"
fi
echo "===== $(date) pipeline complete =====" | tee -a "$PIPELINE_LOG"
