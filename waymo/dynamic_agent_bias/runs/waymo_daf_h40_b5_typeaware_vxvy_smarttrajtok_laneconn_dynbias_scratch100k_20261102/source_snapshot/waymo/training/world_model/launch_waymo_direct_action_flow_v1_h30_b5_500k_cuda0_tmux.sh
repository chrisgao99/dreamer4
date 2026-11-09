#!/usr/bin/env bash
# Long tokenizer-free V1 training: explicit agents, joint H30 action flow, B5 commitment.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_direct_action_flow.py"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
ACTION_STATS="$DATA_ROOT/direct_action_stats_l11_v1_8192.json"

RUN_NAME="${RUN_NAME:-waymo_direct_action_flow_v1_explicitagent_h30_b5_d256_500k}"
SESSION_NAME="${SESSION_NAME:-wm_direct_action_v1_h30_b5_cuda0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_STEPS="${MAX_STEPS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CKPT_DIR="$REPO_ROOT/waymo/checkpoints/$RUN_NAME"
TRAIN_LOG="$REPO_ROOT/waymo/logs/wm/$RUN_NAME.log"

for required_file in "$PYTHON" "$TRAIN_SCRIPT"; do
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
mkdir -p "$CKPT_DIR" "$(dirname "$TRAIN_LOG")" "$REPO_ROOT/waymo/wandb"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 \
    REPO_ROOT="$REPO_ROOT" \
    PYTHON="$PYTHON" \
    RUN_NAME="$RUN_NAME" \
    SESSION_NAME="$SESSION_NAME" \
    CUDA_DEVICE="$CUDA_DEVICE" \
    MAX_STEPS="$MAX_STEPS" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    NUM_WORKERS="$NUM_WORKERS" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started detached tmux session: $SESSION_NAME"
  echo "GPU: $CUDA_DEVICE"
  echo "Log: $TRAIN_LOG"
  echo "Checkpoints: $CKPT_DIR"
  echo "Attach: tmux attach -t $SESSION_NAME"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

resume_args=()
if [[ -f "$CKPT_DIR/latest.pt" ]]; then
  resume_args=(--resume "$CKPT_DIR/latest.pt")
fi
wandb_args=()
if [[ "$WANDB_MODE" != "disabled" ]]; then
  wandb_args=(--wandb)
fi

{
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME session=$SESSION_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "data=$DATA_ROOT action_stats=$ACTION_STATS"
  echo "model=direct_action_flow_v1 tokenizer=none latent=none"
  echo "agent_state=x,y,yaw valid_as_mask=1 agent_type_condition=1"
  echo "light_condition=current_only future_light=0 predict_light=0"
  echo "focus_action=known_h30 generated_agents=nonfocus"
  echo "history=11 horizon=30 commitment=5 action=local_long_lat_dyaw"
  echo "d_model=256 heads=8 depths=history2,map2,scene4,action8,refiner2"
  echo "batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS max_steps=$MAX_STEPS"
  if [[ ${#resume_args[@]} -gt 0 ]]; then
    echo "resume=${resume_args[1]}"
  else
    echo "resume=none"
  fi
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
  --horizon 30 \
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
  --lr 2e-4 \
  --min_lr_ratio 0.1 \
  --warmup_steps 5000 \
  --weight_decay 0.01 \
  --grad_clip 1 \
  --ema_decay 0.9999 \
  --amp_dtype bf16 \
  --log_every 20 \
  --eval_every 5000 \
  --save_every 25000 \
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
