#!/usr/bin/env bash
# Causal stability ablation from the same safe H15 step-45k checkpoint:
#   tmax095: full actions, but sample training flow time only from [0, 0.95).
#   clip10:  full [0, 1) flow time, but clip training actions to +/-10 sigma.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_direct_action_flow.py"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
ACTION_STATS="$DATA_ROOT/direct_action_stats_l11_v1_8192.json"
SOURCE_H15_STEP45K="$REPO_ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_explicitagent_h15_b5_d256_lr2e4_control_60k/step_000045000.pt"

EXPERIMENT="${EXPERIMENT:-both}"
CUDA_TMAX="${CUDA_TMAX:-0}"
CUDA_CLIP="${CUDA_CLIP:-1}"
MAX_STEPS="${MAX_STEPS:-70000}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-500000}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SESSION_PREFIX="${SESSION_PREFIX:-wm_daf_v1_causal}"

TMAX_RUN_NAME="${TMAX_RUN_NAME:-waymo_direct_action_flow_v1_h15_resume45k_tmax095}"
CLIP_RUN_NAME="${CLIP_RUN_NAME:-waymo_direct_action_flow_v1_h15_resume45k_actionclip10}"

for required_file in "$PYTHON" "$TRAIN_SCRIPT" "$ACTION_STATS" "$SOURCE_H15_STEP45K"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done
for required_dir in "$DATA_ROOT/train" "$DATA_ROOT/val"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Missing required directory: $required_dir" >&2
    exit 1
  fi
done

launch_detached() {
  local experiment="$1"
  local cuda_device="$2"
  local session_name="$3"
  local script_path
  local tmux_command

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session already exists: $session_name" >&2
    return 1
  fi
  script_path="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 \
    EXPERIMENT="$experiment" \
    CUDA_DEVICE="$cuda_device" \
    REPO_ROOT="$REPO_ROOT" \
    PYTHON="$PYTHON" \
    MAX_STEPS="$MAX_STEPS" \
    LR_DECAY_STEPS="$LR_DECAY_STEPS" \
    LR="$LR" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    NUM_WORKERS="$NUM_WORKERS" \
    TMAX_RUN_NAME="$TMAX_RUN_NAME" \
    CLIP_RUN_NAME="$CLIP_RUN_NAME" \
    SESSION_PREFIX="$SESSION_PREFIX" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    bash "$script_path"
  tmux new-session -d -s "$session_name" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$session_name" remain-on-exit on
  echo "Started $experiment in tmux session $session_name on GPU $cuda_device"
}

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  case "$EXPERIMENT" in
    both)
      if [[ "$CUDA_TMAX" == "$CUDA_CLIP" ]]; then
        echo "CUDA_TMAX and CUDA_CLIP must differ when EXPERIMENT=both" >&2
        exit 2
      fi
      launch_detached tmax095 "$CUDA_TMAX" "${SESSION_PREFIX}_tmax095"
      launch_detached clip10 "$CUDA_CLIP" "${SESSION_PREFIX}_clip10"
      ;;
    tmax095)
      launch_detached tmax095 "${CUDA_DEVICE:-$CUDA_TMAX}" "${SESSION_PREFIX}_tmax095"
      ;;
    clip10)
      launch_detached clip10 "${CUDA_DEVICE:-$CUDA_CLIP}" "${SESSION_PREFIX}_clip10"
      ;;
    *)
      echo "EXPERIMENT must be both, tmax095, or clip10; got: $EXPERIMENT" >&2
      exit 2
      ;;
  esac
  echo "List sessions: tmux ls"
  exit 0
fi

case "$EXPERIMENT" in
  tmax095)
    RUN_NAME="$TMAX_RUN_NAME"
    TRAIN_FLOW_TIME_MAX=0.95
    TRAIN_ACTION_CLIP=0
    ;;
  clip10)
    RUN_NAME="$CLIP_RUN_NAME"
    TRAIN_FLOW_TIME_MAX=1.0
    TRAIN_ACTION_CLIP=10
    ;;
  *)
    echo "Internal launch requires tmax095 or clip10; got: $EXPERIMENT" >&2
    exit 2
    ;;
esac

CUDA_DEVICE="${CUDA_DEVICE:?CUDA_DEVICE is required inside tmux}"
CKPT_DIR="$REPO_ROOT/waymo/checkpoints/$RUN_NAME"
TRAIN_LOG="$REPO_ROOT/waymo/logs/wm/$RUN_NAME.log"
mkdir -p "$CKPT_DIR" "$(dirname "$TRAIN_LOG")" "$REPO_ROOT/waymo/wandb"

resume_args=()
if [[ -f "$CKPT_DIR/latest.pt" ]]; then
  resume_args=(--resume "$CKPT_DIR/latest.pt")
else
  resume_args=(--resume "$SOURCE_H15_STEP45K")
fi

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
  echo "experiment=$EXPERIMENT run_name=$RUN_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "source_checkpoint=$SOURCE_H15_STEP45K"
  echo "history=11 horizon=15 commitment=5"
  echo "train_flow_time_max=$TRAIN_FLOW_TIME_MAX train_normalized_action_clip=$TRAIN_ACTION_CLIP"
  echo "lr=$LR lr_decay_steps=$LR_DECAY_STEPS max_steps=$MAX_STEPS"
  echo "batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS amp=bf16 grad_clip=1 fail_on_nonfinite=1"
  echo "resume=${resume_args[1]}"
  echo "========================"
} | tee -a "$TRAIN_LOG"

"$PYTHON" "$TRAIN_SCRIPT" \
  --data_dir "$DATA_ROOT/train" \
  --val_data_dir "$DATA_ROOT/val" \
  --ckpt_dir "$CKPT_DIR" \
  --action_stats_path "$ACTION_STATS" \
  "${resume_args[@]}" \
  --device cuda \
  --seed 0 \
  --history_length 11 \
  --horizon 15 \
  --commitment 5 \
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
