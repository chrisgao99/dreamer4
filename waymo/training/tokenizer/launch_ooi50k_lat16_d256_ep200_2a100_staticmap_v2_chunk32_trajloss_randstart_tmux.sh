#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
RUN_NAME="${RUN_NAME:-ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_trajloss_randstart_noamp}"
SESSION_NAME="${SESSION_NAME:-ooi50k_staticmap_v2_chunk32_trajloss_randstart_noamp}"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs}"
LOG="${LOG:-$LOG_DIR/${RUN_NAME}.log}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/waymo/wandb}"
RESUME="${RESUME:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
WANDB_MODE="${WANDB_MODE:-online}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-200}"
D_MODEL="${D_MODEL:-256}"
DEPTH="${DEPTH:-4}"
DECODER_DEPTH="${DECODER_DEPTH:-4}"
N_LATENTS="${N_LATENTS:-16}"
D_BOTTLENECK="${D_BOTTLENECK:-32}"
TIME_WINDOW="${TIME_WINDOW:-32}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
BOTTLENECK_OUTPUT="${BOTTLENECK_OUTPUT:-tanh}"
DECODER_USE_AGENT_TOKENS="${DECODER_USE_AGENT_TOKENS:-0}"
DECODER_AGENT_TOKEN_MODE="${DECODER_AGENT_TOKEN_MODE:-none}"
DECODER_ATTEND_MAP="${DECODER_ATTEND_MAP:-0}"
DECODER_MAP_CROSS_EVERY="${DECODER_MAP_CROSS_EVERY:-1}"
DECODER_MAP_QUERY_TOKENS="${DECODER_MAP_QUERY_TOKENS:-all}"
OVERWRITE_RUN="${OVERWRITE_RUN:-0}"

ENCODER_VARIANT="${ENCODER_VARIANT:-static_map_query}"
MAP_DEPTH="${MAP_DEPTH:-2}"
MAP_CROSS_EVERY="${MAP_CROSS_EVERY:-1}"
MAP_QUERY_TOKENS="${MAP_QUERY_TOKENS:-latent_agent}"
AGENT_XY_LOSS="${AGENT_XY_LOSS:-smooth_l1}"
AGENT_XY_PARAMETERIZATION="${AGENT_XY_PARAMETERIZATION:-absolute}"
AGENT_DELTA_XY_WEIGHT="${AGENT_DELTA_XY_WEIGHT:-5}"
AGENT_FDE_XY_WEIGHT="${AGENT_FDE_XY_WEIGHT:-2}"
AGENT_KINEMATIC_XY_WEIGHT="${AGENT_KINEMATIC_XY_WEIGHT:-0}"
AGENT_SPEED_YAW_KINEMATIC_WEIGHT="${AGENT_SPEED_YAW_KINEMATIC_WEIGHT:-0}"
KINEMATIC_DT="${KINEMATIC_DT:-0.1}"
FOCUS_AGENT_WEIGHT="${FOCUS_AGENT_WEIGHT:-4}"
INTERACTION_AUX_WEIGHT="${INTERACTION_AUX_WEIGHT:-0}"
INTERACTION_RELEVANCE_WEIGHT="${INTERACTION_RELEVANCE_WEIGHT:-1}"
INTERACTION_TYPE_WEIGHT="${INTERACTION_TYPE_WEIGHT:-1}"
INTERACTION_RESPONSE_BIN_WEIGHT="${INTERACTION_RESPONSE_BIN_WEIGHT:-1}"
INTERACTION_RESPONSE_REG_WEIGHT="${INTERACTION_RESPONSE_REG_WEIGHT:-0.2}"
INTERACTION_QUERY_STEP="${INTERACTION_QUERY_STEP:--1}"
INTERACTION_FUTURE_STEPS="${INTERACTION_FUTURE_STEPS:-50}"
INTERACTION_FOCUS_INDEX="${INTERACTION_FOCUS_INDEX:-0}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NO_AMP="${NO_AMP:-1}"

check_required_paths() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found or not executable: $PYTHON" >&2
    exit 1
  fi
  if [[ ! -f "$REPO_ROOT/waymo/training/train_waymo_vector_tokenizer.py" ]]; then
    echo "Training script not found under REPO_ROOT: $REPO_ROOT" >&2
    exit 1
  fi
  if [[ ! -d "$DATA_DIR/train" || ! -d "$DATA_DIR/val" ]]; then
    echo "Expected dataset directories are missing:" >&2
    echo "  $DATA_DIR/train" >&2
    echo "  $DATA_DIR/val" >&2
    echo "Set DATA_DIR to the directory containing train/ and val/ before launching." >&2
    exit 1
  fi
  if ! compgen -G "$DATA_DIR/train/*.npz" >/dev/null; then
    echo "No training .npz files found in: $DATA_DIR/train" >&2
    exit 1
  fi
  if ! compgen -G "$DATA_DIR/val/*.npz" >/dev/null; then
    echo "No validation .npz files found in: $DATA_DIR/val" >&2
    exit 1
  fi
}

check_required_paths

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" ]]
}

prepare_overwrite_run() {
  if ! is_truthy "$OVERWRITE_RUN"; then
    mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"
    return
  fi
  if [[ -z "$RUN_NAME" ]]; then
    echo "Refusing OVERWRITE_RUN=1 with empty RUN_NAME" >&2
    exit 1
  fi
  case "$CKPT_DIR" in
    "$REPO_ROOT/waymo/checkpoints/"*) ;;
    *)
      echo "Refusing to remove checkpoint dir outside $REPO_ROOT/waymo/checkpoints: $CKPT_DIR" >&2
      exit 1
      ;;
  esac
  case "$LOG" in
    "$REPO_ROOT/waymo/logs/"*) ;;
    *)
      echo "Refusing to truncate log outside $REPO_ROOT/waymo/logs: $LOG" >&2
      exit 1
      ;;
  esac
  rm -rf "$CKPT_DIR"
  mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"
  : > "$LOG"
}

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"
  if ! command -v tmux >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
      module load tmux >/dev/null 2>&1 || true
    elif [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck source=/dev/null
      source /etc/profile.d/modules.sh
      module load tmux >/dev/null 2>&1 || true
    fi
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found after trying: module load tmux" >&2
    echo "running in the current shell instead."
    RUN_INSIDE_TMUX=1 exec bash "$0"
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    echo "Log: $LOG"
    exit 0
  fi
  tmux new-session -d -s "$SESSION_NAME" \
    "RUN_INSIDE_TMUX=1 bash '$0'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  echo "Detach with: Ctrl-b then d"
  echo "Log: $LOG"
  exit 0
fi

cd "$REPO_ROOT"
prepare_overwrite_run

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export WANDB_MODE
export WANDB_DIR

train_args=(
  waymo/training/train_waymo_vector_tokenizer.py
  --data_dir "$DATA_DIR/train"
  --val_data_dir "$DATA_DIR/val"
  --ckpt_dir "$CKPT_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --time_window "$TIME_WINDOW"
  --random_time_window_start
  --epochs "$EPOCHS"
  --d_model "$D_MODEL"
  --depth "$DEPTH"
  --decoder_depth "$DECODER_DEPTH"
  --n_latents "$N_LATENTS"
  --d_bottleneck "$D_BOTTLENECK"
  --encoder_variant "$ENCODER_VARIANT"
  --map_depth "$MAP_DEPTH"
  --map_cross_every "$MAP_CROSS_EVERY"
  --map_query_tokens "$MAP_QUERY_TOKENS"
  --bottleneck_output "$BOTTLENECK_OUTPUT"
  --agent_xy_loss "$AGENT_XY_LOSS"
  --agent_xy_parameterization "$AGENT_XY_PARAMETERIZATION"
  --agent_delta_xy_weight "$AGENT_DELTA_XY_WEIGHT"
  --agent_fde_xy_weight "$AGENT_FDE_XY_WEIGHT"
  --agent_kinematic_xy_weight "$AGENT_KINEMATIC_XY_WEIGHT"
  --agent_speed_yaw_kinematic_weight "$AGENT_SPEED_YAW_KINEMATIC_WEIGHT"
  --kinematic_dt "$KINEMATIC_DT"
  --focus_agent_weight "$FOCUS_AGENT_WEIGHT"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --log_every "$LOG_EVERY"
  --eval_every "$EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --wandb
  --wandb_project waymo-vector-tokenizer
  --wandb_run_name "$RUN_NAME"
)

if [[ "$NO_AMP" == "1" || "$NO_AMP" == "true" || "$NO_AMP" == "TRUE" ]]; then
  train_args+=(--no_amp)
fi
if [[ "$DECODER_USE_AGENT_TOKENS" == "1" || "$DECODER_USE_AGENT_TOKENS" == "true" || "$DECODER_USE_AGENT_TOKENS" == "TRUE" ]]; then
  train_args+=(--decoder_use_agent_tokens)
fi
if [[ "$DECODER_AGENT_TOKEN_MODE" != "none" ]]; then
  train_args+=(--decoder_agent_token_mode "$DECODER_AGENT_TOKEN_MODE")
fi
if [[ "$DECODER_ATTEND_MAP" == "1" || "$DECODER_ATTEND_MAP" == "true" || "$DECODER_ATTEND_MAP" == "TRUE" ]]; then
  train_args+=(
    --decoder_attend_map
    --decoder_map_cross_every "$DECODER_MAP_CROSS_EVERY"
    --decoder_map_query_tokens "$DECODER_MAP_QUERY_TOKENS"
  )
fi

if [[ "$INTERACTION_AUX_WEIGHT" != "0" && "$INTERACTION_AUX_WEIGHT" != "0.0" ]]; then
  train_args+=(
    --interaction_aux_weight "$INTERACTION_AUX_WEIGHT"
    --interaction_relevance_weight "$INTERACTION_RELEVANCE_WEIGHT"
    --interaction_type_weight "$INTERACTION_TYPE_WEIGHT"
    --interaction_response_bin_weight "$INTERACTION_RESPONSE_BIN_WEIGHT"
    --interaction_response_reg_weight "$INTERACTION_RESPONSE_REG_WEIGHT"
    --interaction_query_step "$INTERACTION_QUERY_STEP"
    --interaction_future_steps "$INTERACTION_FUTURE_STEPS"
    --interaction_focus_index "$INTERACTION_FOCUS_INDEX"
  )
fi

if [[ -n "$RESUME" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "Resume checkpoint not found: $RESUME" >&2
    exit 1
  fi
  train_args+=(--resume "$RESUME")
elif [[ -f "$CKPT_DIR/latest.pt" ]]; then
  train_args+=(--resume "$CKPT_DIR/latest.pt")
fi

{
  echo
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME"
  echo "session=$SESSION_NAME"
  echo "cuda=$CUDA_VISIBLE_DEVICES"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "wandb_mode=$WANDB_MODE"
  echo "overwrite_run=$OVERWRITE_RUN"
  echo "data_dir=$DATA_DIR"
  echo "ckpt_dir=$CKPT_DIR"
  echo "resume=${RESUME:-auto_latest_if_present}"
  echo "log=$LOG"
  echo "batch_size=$BATCH_SIZE epochs=$EPOCHS d_model=$D_MODEL depth=$DEPTH decoder_depth=$DECODER_DEPTH n_latents=$N_LATENTS d_bottleneck=$D_BOTTLENECK time_window=$TIME_WINDOW random_time_window_start=1"
  echo "encoder_variant=$ENCODER_VARIANT map_depth=$MAP_DEPTH map_cross_every=$MAP_CROSS_EVERY map_query_tokens=$MAP_QUERY_TOKENS"
  echo "bottleneck_output=$BOTTLENECK_OUTPUT decoder_use_agent_tokens=$DECODER_USE_AGENT_TOKENS decoder_agent_token_mode=$DECODER_AGENT_TOKEN_MODE decoder_attend_map=$DECODER_ATTEND_MAP decoder_map_cross_every=$DECODER_MAP_CROSS_EVERY decoder_map_query_tokens=$DECODER_MAP_QUERY_TOKENS"
  echo "agent_xy_loss=$AGENT_XY_LOSS agent_xy_parameterization=$AGENT_XY_PARAMETERIZATION agent_delta_xy_weight=$AGENT_DELTA_XY_WEIGHT agent_fde_xy_weight=$AGENT_FDE_XY_WEIGHT agent_kinematic_xy_weight=$AGENT_KINEMATIC_XY_WEIGHT agent_speed_yaw_kinematic_weight=$AGENT_SPEED_YAW_KINEMATIC_WEIGHT kinematic_dt=$KINEMATIC_DT focus_agent_weight=$FOCUS_AGENT_WEIGHT"
  echo "interaction_aux_weight=$INTERACTION_AUX_WEIGHT interaction_relevance_weight=$INTERACTION_RELEVANCE_WEIGHT interaction_type_weight=$INTERACTION_TYPE_WEIGHT interaction_response_bin_weight=$INTERACTION_RESPONSE_BIN_WEIGHT interaction_response_reg_weight=$INTERACTION_RESPONSE_REG_WEIGHT interaction_query_step=$INTERACTION_QUERY_STEP interaction_future_steps=$INTERACTION_FUTURE_STEPS interaction_focus_index=$INTERACTION_FOCUS_INDEX"
  echo "lr=$LR weight_decay=$WEIGHT_DECAY grad_clip=$GRAD_CLIP no_amp=$NO_AMP"
  echo "loss_fix=normalized_agent_targets_no_double_transpose"
  echo "========================"
} | tee -a "$LOG"

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "${train_args[@]}" \
  2>&1 | tee -a "$LOG"
