#!/usr/bin/env bash
# Train DirectActionFlow for 100k optimizer steps from random initialization.
# All agents, including slot-zero focus, are generated and supervised; no
# future action sequence is supplied to the model. This launches detached tmux.

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
DEFAULT_REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../..")"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
OWNER_ROOT="$(dirname "$(dirname "$REPO_ROOT")")"
if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$OWNER_ROOT/.conda/envs/dreamer4/bin/python"
  OWNER_NAME="$(basename "$OWNER_ROOT")"
  if [[ ! -x "$PYTHON" && -x "/home/$OWNER_NAME/.conda/envs/dreamer4/bin/python" ]]; then
    PYTHON="/home/$OWNER_NAME/.conda/envs/dreamer4/bin/python"
  fi
  if [[ ! -x "$PYTHON" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  fi
  if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python || true)"
  fi
fi
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_direct_action_flow.py"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_NAME="${RUN_NAME:-waymo_direct_action_flow_v1_h15_b5_generateall_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5}"
SESSION_NAME="${SESSION_NAME:-wm_daf_h15_generateall_scratch100k_cuda${CUDA_DEVICE}}"
MAX_STEPS="${MAX_STEPS:-100000}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-500000}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TRAIN_FLOW_TIME_MAX="${TRAIN_FLOW_TIME_MAX:-0.90}"
TRAIN_ACTION_CLIP="${TRAIN_ACTION_CLIP:-5}"
PHYSICAL_MAX_DISPLACEMENT_M="${PHYSICAL_MAX_DISPLACEMENT_M:-5.0}"
PHYSICAL_MAX_YAW_DELTA_RAD="${PHYSICAL_MAX_YAW_DELTA_RAD:-0.75}"
FLOW_HUBER_BETA="${FLOW_HUBER_BETA:-1.0}"
MODULATION_SCALE_LIMIT="${MODULATION_SCALE_LIMIT:-2.0}"
MODULATION_SHIFT_LIMIT="${MODULATION_SHIFT_LIMIT:-5.0}"

ACTION_STATS="${ACTION_STATS:-$DATA_ROOT/direct_action_stats_l11_v2_phys5_yaw075_8192.json}"
CKPT_DIR="$REPO_ROOT/waymo/checkpoints/$RUN_NAME"
TRAIN_LOG="$REPO_ROOT/waymo/logs/wm/$RUN_NAME.log"

for required_file in "$PYTHON" "$TRAIN_SCRIPT" "$ACTION_STATS"; do
  [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 1; }
done
[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 1; }
for required_dir in "$DATA_ROOT/train" "$DATA_ROOT/val"; do
  [[ -d "$required_dir" ]] || { echo "Missing required directory: $required_dir" >&2; exit 1; }
done

# This experiment is intentionally from scratch. Refuse both cross-run resume
# and accidental reuse of a partial directory from an earlier launch.
if [[ -e "$CKPT_DIR" || -e "$TRAIN_LOG" ]]; then
  echo "Fresh-run output already exists; refusing to resume or overwrite." >&2
  echo "checkpoint_dir=$CKPT_DIR" >&2
  echo "log=$TRAIN_LOG" >&2
  echo "Set a new RUN_NAME to start another independent run." >&2
  exit 1
fi

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 \
    REPO_ROOT="$REPO_ROOT" \
    PYTHON="$PYTHON" \
    DATA_ROOT="$DATA_ROOT" \
    CUDA_DEVICE="$CUDA_DEVICE" \
    RUN_NAME="$RUN_NAME" \
    SESSION_NAME="$SESSION_NAME" \
    MAX_STEPS="$MAX_STEPS" \
    LR_DECAY_STEPS="$LR_DECAY_STEPS" \
    LR="$LR" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    NUM_WORKERS="$NUM_WORKERS" \
    TRAIN_FLOW_TIME_MAX="$TRAIN_FLOW_TIME_MAX" \
    TRAIN_ACTION_CLIP="$TRAIN_ACTION_CLIP" \
    PHYSICAL_MAX_DISPLACEMENT_M="$PHYSICAL_MAX_DISPLACEMENT_M" \
    PHYSICAL_MAX_YAW_DELTA_RAD="$PHYSICAL_MAX_YAW_DELTA_RAD" \
    FLOW_HUBER_BETA="$FLOW_HUBER_BETA" \
    MODULATION_SCALE_LIMIT="$MODULATION_SCALE_LIMIT" \
    MODULATION_SHIFT_LIMIT="$MODULATION_SHIFT_LIMIT" \
    ACTION_STATS="$ACTION_STATS" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started fresh all-agent training in tmux: $SESSION_NAME"
  echo "GPU: $CUDA_DEVICE"
  echo "Log: $TRAIN_LOG"
  echo "Checkpoints: $CKPT_DIR"
  echo "Attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

mkdir -p "$CKPT_DIR" "$(dirname "$TRAIN_LOG")" "$REPO_ROOT/waymo/wandb"

wandb_args=()
if [[ "${WANDB_MODE:-online}" != "disabled" ]]; then
  wandb_args=(--wandb)
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

{
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "initialization=random resume=none max_steps=$MAX_STEPS"
  echo "history=11 horizon=15 commitment=5"
  echo "focus=no_future_action generated=all_agents supervised=all_valid_agents"
  echo "physical_filter=displacement<=${PHYSICAL_MAX_DISPLACEMENT_M}m yaw_delta<=${PHYSICAL_MAX_YAW_DELTA_RAD}rad"
  echo "flow_loss=huber beta=$FLOW_HUBER_BETA"
  echo "modulation_scale_limit=$MODULATION_SCALE_LIMIT modulation_shift_limit=$MODULATION_SHIFT_LIMIT"
  echo "train_flow_time_max=$TRAIN_FLOW_TIME_MAX train_normalized_action_clip=$TRAIN_ACTION_CLIP"
  echo "lr=$LR lr_decay_steps=$LR_DECAY_STEPS"
  echo "batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS amp=bf16 grad_clip=1 fail_on_nonfinite=1"
  echo "action_stats=$ACTION_STATS"
  echo "========================"
} | tee -a "$TRAIN_LOG"

"$PYTHON" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_ROOT/train" \
  --val_data_dir "$DATA_ROOT/val" \
  --ckpt_dir "$CKPT_DIR" \
  --action_stats_path "$ACTION_STATS" \
  --device cuda \
  --seed 0 \
  --history_length 11 \
  --horizon 15 \
  --commitment 5 \
  --no-condition_focus_actions \
  --position_scale_m 100 \
  --num_agent_types 16 \
  --d_model 256 \
  --n_heads 8 \
  --hidden_dim 128 \
  --history_depth 2 \
  --map_depth 2 \
  --scene_depth 4 \
  --action_depth 8 \
  --step_refiner_depth 2 \
  --dropout 0.05 \
  --mlp_ratio 4 \
  --modulation_scale_limit "$MODULATION_SCALE_LIMIT" \
  --modulation_shift_limit "$MODULATION_SHIFT_LIMIT" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size 4 \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --num_workers "$NUM_WORKERS" \
  --stats_batch_size 64 \
  --stats_max_files 8192 \
  --max_steps "$MAX_STEPS" \
  --lr "$LR" \
  --min_lr_ratio 0.1 \
  --warmup_steps 5000 \
  --lr_decay_steps "$LR_DECAY_STEPS" \
  --weight_decay 0.01 \
  --grad_clip 1 \
  --fail_on_nonfinite \
  --ema_decay 0.9999 \
  --amp_dtype bf16 \
  --train_flow_time_max "$TRAIN_FLOW_TIME_MAX" \
  --train_normalized_action_clip "$TRAIN_ACTION_CLIP" \
  --physical_max_displacement_m "$PHYSICAL_MAX_DISPLACEMENT_M" \
  --physical_max_yaw_delta_rad "$PHYSICAL_MAX_YAW_DELTA_RAD" \
  --flow_loss_type huber \
  --flow_huber_beta "$FLOW_HUBER_BETA" \
  --log_every 20 \
  --eval_every 2500 \
  --save_every 2500 \
  --eval_batches 32 \
  --sample_eval_every 10000 \
  --sample_eval_batches 4 \
  --eval_num_rollouts 4 \
  --eval_solver_steps 8 \
  --eval_seed 12345 \
  --receding_eval_every 50000 \
  --receding_eval_batches 1 \
  --receding_eval_horizon 80 \
  --wandb_project waymo-world-model \
  --wandb_run_name "$RUN_NAME" \
  "${wandb_args[@]}" \
  2>&1 | tee -a "$TRAIN_LOG"

echo "Finished at $(date)" | tee -a "$TRAIN_LOG"
