#!/usr/bin/env bash
# Run the map-conditioned Waymo world-model comparison on local CUDA 2 and 3.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

TOKENIZER_RUN="ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp"
if [[ -z "${TOKENIZER_CKPT:-}" ]]; then
  tokenizer_candidates=(
    "$REPO_ROOT/waymo/checkpoints/$TOKENIZER_RUN/best.pt"
    "$REPO_ROOT/waymo/checkpoints/$TOKENIZER_RUN/latest.pt"
    "/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/$TOKENIZER_RUN/best.pt"
    "/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/$TOKENIZER_RUN/latest.pt"
  )
  TOKENIZER_CKPT="${tokenizer_candidates[0]}"
  for candidate in "${tokenizer_candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      TOKENIZER_CKPT="$candidate"
      break
    fi
  done
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
  data_candidates=(
    "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
    "$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k"
    "/scratch/baz7dy/tri30/dreamer4/waymo/data/waymo_vector_dataset_ooi_centered_50k"
  )
  DATA_ROOT="${data_candidates[0]}"
  for candidate in "${data_candidates[@]}"; do
    if [[ -d "$candidate/train" && -d "$candidate/val" ]]; then
      DATA_ROOT="$candidate"
      break
    fi
  done
fi

RUN_NAME="${RUN_NAME:-waymo_wm_v1_egoact_focus_raw_noclamp_win11_ctx1_h10_randstart_b2_norecon_mapx1_1m}"
SESSION_NAME="${SESSION_NAME:-wm_raw11_mapx1_cuda23}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/wm}"
LOG="${LOG:-$LOG_DIR/$RUN_NAME.log}"
RESUME="${RESUME:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
WANDB_MODE="${WANDB_MODE:-online}"
USE_WANDB="${USE_WANDB:-1}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-1000000}"
LOG_EVERY="${LOG_EVERY:-100}"
EVAL_EVERY="${EVAL_EVERY:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
SAVE_EVERY="${SAVE_EVERY:-50000}"

SEQ_LEN="${SEQ_LEN:-11}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-11}"
EVAL_CTX="${EVAL_CTX:-1}"
EVAL_HORIZON="${EVAL_HORIZON:-10}"
MAX_ROLLOUT_WINDOW="${MAX_ROLLOUT_WINDOW:-11}"

D_MODEL_DYN="${D_MODEL_DYN:-512}"
DYN_DEPTH="${DYN_DEPTH:-8}"
N_HEADS="${N_HEADS:-8}"
TIME_EVERY="${TIME_EVERY:-4}"
MAP_CROSS_EVERY="${MAP_CROSS_EVERY:-1}"
PACKING_FACTOR="${PACKING_FACTOR:-2}"
N_REGISTER="${N_REGISTER:-8}"
K_MAX="${K_MAX:-64}"
SELF_FRACTION="${SELF_FRACTION:-0.857142857}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    return 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    return 1
  fi
}

require_file "$PYTHON"
require_file "$TRAIN_SCRIPT"
if ! require_file "$TOKENIZER_CKPT"; then
  echo "Set TOKENIZER_CKPT to the original lat64/b64 tokenizer checkpoint." >&2
  exit 1
fi
require_dir "$DATA_ROOT/train"
require_dir "$DATA_ROOT/val"
mkdir -p "$CKPT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is not available." >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
  fi

  launch_env=(
    "RUN_INSIDE_TMUX=1"
    "REPO_ROOT=$REPO_ROOT"
    "PYTHON=$PYTHON"
    "TOKENIZER_CKPT=$TOKENIZER_CKPT"
    "DATA_ROOT=$DATA_ROOT"
    "RUN_NAME=$RUN_NAME"
    "SESSION_NAME=$SESSION_NAME"
    "CKPT_DIR=$CKPT_DIR"
    "LOG_DIR=$LOG_DIR"
    "LOG=$LOG"
    "RESUME=$RESUME"
    "AUTO_RESUME=$AUTO_RESUME"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "NPROC_PER_NODE=$NPROC_PER_NODE"
    "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    "WANDB_MODE=$WANDB_MODE"
    "USE_WANDB=$USE_WANDB"
    "BATCH_SIZE=$BATCH_SIZE"
    "EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE"
    "NUM_WORKERS=$NUM_WORKERS"
    "MAX_STEPS=$MAX_STEPS"
    "LOG_EVERY=$LOG_EVERY"
    "EVAL_EVERY=$EVAL_EVERY"
    "EVAL_MAX_BATCHES=$EVAL_MAX_BATCHES"
    "SAVE_EVERY=$SAVE_EVERY"
    "SEQ_LEN=$SEQ_LEN"
    "EVAL_SEQ_LEN=$EVAL_SEQ_LEN"
    "EVAL_CTX=$EVAL_CTX"
    "EVAL_HORIZON=$EVAL_HORIZON"
    "MAX_ROLLOUT_WINDOW=$MAX_ROLLOUT_WINDOW"
    "D_MODEL_DYN=$D_MODEL_DYN"
    "DYN_DEPTH=$DYN_DEPTH"
    "N_HEADS=$N_HEADS"
    "TIME_EVERY=$TIME_EVERY"
    "MAP_CROSS_EVERY=$MAP_CROSS_EVERY"
    "PACKING_FACTOR=$PACKING_FACTOR"
    "N_REGISTER=$N_REGISTER"
    "K_MAX=$K_MAX"
    "SELF_FRACTION=$SELF_FRACTION"
    "LR=$LR"
    "WEIGHT_DECAY=$WEIGHT_DECAY"
    "GRAD_CLIP=$GRAD_CLIP"
    "AMP_DTYPE=$AMP_DTYPE"
  )
  launch_cmd=(env "${launch_env[@]}" bash "$SCRIPT_PATH")
  printf -v tmux_command '%q ' "${launch_cmd[@]}"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"

  echo "Started tmux session: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Detach: Ctrl-b then d"
  echo "Log: $LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES OMP_NUM_THREADS WANDB_MODE PYTHONUNBUFFERED=1

train_args=(
  "$TRAIN_SCRIPT"
  --data_dir "$DATA_ROOT/train"
  --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --ckpt_dir "$CKPT_DIR"
  --seed 0
  --seq_len "$SEQ_LEN"
  --random_time_window_start
  --eval_seq_len "$EVAL_SEQ_LEN"
  --eval_ctx "$EVAL_CTX"
  --eval_horizon "$EVAL_HORIZON"
  --max_rollout_window "$MAX_ROLLOUT_WINDOW"
  --eval_schedule shortcut
  --eval_d 0.25
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --max_steps "$MAX_STEPS"
  --log_every "$LOG_EVERY"
  --eval_every "$EVAL_EVERY"
  --eval_max_batches "$EVAL_MAX_BATCHES"
  --save_every "$SAVE_EVERY"
  --d_model_dyn "$D_MODEL_DYN"
  --dyn_depth "$DYN_DEPTH"
  --n_heads "$N_HEADS"
  --time_every "$TIME_EVERY"
  --dynamics_attend_map
  --map_cross_every "$MAP_CROSS_EVERY"
  --packing_factor "$PACKING_FACTOR"
  --n_register "$N_REGISTER"
  --k_max "$K_MAX"
  --self_fraction "$SELF_FRACTION"
  --bootstrap_start 0
  --train_objective shortcut
  --tf_context 10
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --amp_dtype "$AMP_DTYPE"
  --agent_xy_loss smooth_l1
  --agent_xy_parameterization absolute
  --focus_agent_weight 4
  --agent_kinematic_xy_weight 5
  --agent_speed_yaw_kinematic_weight 2
  --use_ego_actions
  --ego_action_source focus
  --ego_action_normalization raw
  --no-ego_action_clamp
  --agent_far_weight 0.25
  --agent_near_radius_m 50.0
  --agent_distance_source focus
  --train_decoded_loss_weight 0.0
  --wandb_project waymo-world-model
  --wandb_run_name "$RUN_NAME"
)

if [[ -n "$RESUME" ]]; then
  require_file "$RESUME"
  train_args+=(--resume "$RESUME")
elif is_truthy "$AUTO_RESUME" && [[ -f "$CKPT_DIR/latest.pt" ]]; then
  RESUME="$CKPT_DIR/latest.pt"
  train_args+=(--resume "$RESUME")
fi
if is_truthy "$USE_WANDB"; then
  train_args+=(--wandb)
fi

{
  echo
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME"
  echo "session=$SESSION_NAME"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES nproc_per_node=$NPROC_PER_NODE"
  echo "tokenizer_ckpt=$TOKENIZER_CKPT"
  echo "train_data=$DATA_ROOT/train"
  echo "val_data=$DATA_ROOT/val"
  echo "ckpt_dir=$CKPT_DIR"
  echo "resume=${RESUME:-none}"
  echo "batch_size_per_process=$BATCH_SIZE seq_len=$SEQ_LEN random_time_window_start=1 max_steps=$MAX_STEPS"
  echo "d_model_dyn=$D_MODEL_DYN depth=$DYN_DEPTH heads=$N_HEADS time_every=$TIME_EVERY"
  echo "dynamics_attend_map=1 map_cross_every=$MAP_CROSS_EVERY map_query_tokens=spatial_only"
  echo "objective=shortcut ego_actions=focus_raw_noclamp decoded_loss_weight=0"
  echo "eval_every=$EVAL_EVERY save_every=$SAVE_EVERY wandb=$USE_WANDB wandb_mode=$WANDB_MODE"
  echo "========================"
} | tee -a "$LOG"

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "${train_args[@]}" \
  2>&1 | tee -a "$LOG"
