#!/bin/bash
# Long tokenizer-free V1 training: explicit agents, joint H30 action flow, B5 commitment.
# Slurm equivalent of launch_waymo_direct_action_flow_v1_h30_b5_500k_cuda0_tmux.sh.
# Submit with:
#   sbatch waymo/training/world_model/submit_waymo_direct_action_flow_v1_h30_b5_500k.slurm

#SBATCH --job-name=wm_direct_h30_b5
#SBATCH --account=lia-lab-members
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|a100|a40|a6000"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --open-mode=truncate
#SBATCH --output=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.out
#SBATCH --error=/scratch/baz7dy/tri30/dreamer4/waymo/logs/wm/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_direct_action_flow.py"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k}"
ACTION_STATS="${ACTION_STATS:-$DATA_ROOT/direct_action_stats_l11_v1_8192.json}"

RUN_NAME="${RUN_NAME:-waymo_direct_action_flow_v1_explicitagent_h30_b5_d256_500k}"
MAX_STEPS="${MAX_STEPS:-500000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/wm}"
TRAIN_LOG="${TRAIN_LOG:-$LOG_DIR/$RUN_NAME.log}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/waymo/wandb}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
WANDB_MODE="${WANDB_MODE:-online}"

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory not found: $1" >&2
    exit 1
  fi
}

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found or not executable: $PYTHON" >&2
  exit 1
fi
require_file "$TRAIN_SCRIPT"
require_dir "$DATA_ROOT/train"
require_dir "$DATA_ROOT/val"

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"

cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS
export WANDB_MODE
export WANDB_DIR

resume_args=()
if [[ -f "$CKPT_DIR/latest.pt" ]]; then
  resume_args=(--resume "$CKPT_DIR/latest.pt")
fi

wandb_args=()
if [[ "$WANDB_MODE" != "disabled" && "$WANDB_MODE" != "offline-disabled" ]]; then
  wandb_args=(--wandb)
fi

{
  echo "===== Slurm job start: $(date) ====="
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
  echo "run_name=$RUN_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "data=$DATA_ROOT action_stats=$ACTION_STATS"
  echo "ckpt_dir=$CKPT_DIR"
  echo "model=direct_action_flow_v1 tokenizer=none latent=none"
  echo "agent_state=x,y,yaw valid_as_mask=1 agent_type_condition=1"
  echo "light_condition=current_only future_light=0 predict_light=0"
  echo "focus_action=known_h30 generated_agents=nonfocus"
  echo "history=11 horizon=30 commitment=5 action=local_long_lat_dyaw"
  echo "d_model=256 heads=8 depths=history2,map2,scene4,action8,refiner2"
  echo "batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS max_steps=$MAX_STEPS"
  echo "num_workers=$NUM_WORKERS eval_batch=$EVAL_BATCH_SIZE"
  if [[ ${#resume_args[@]} -gt 0 ]]; then
    echo "resume=${resume_args[1]}"
  else
    echo "resume=none"
  fi
  echo "========================"
} | tee -a "$TRAIN_LOG"

train_args=(
  "$TRAIN_SCRIPT"
  --data_dir "$DATA_ROOT/train"
  --val_data_dir "$DATA_ROOT/val"
  --ckpt_dir "$CKPT_DIR"
  --action_stats_path "$ACTION_STATS"
  "${resume_args[@]}"
  --device cuda
  --seed 0
  --history_length 11
  --horizon 30
  --commitment 5
  --position_scale_m 100
  --num_agent_types 16
  --d_model 256
  --n_heads 8
  --hidden_dim 128
  --history_depth 2
  --map_depth 2
  --scene_depth 4
  --action_depth 8
  --step_refiner_depth 2
  --dropout 0.05
  --mlp_ratio 4
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --num_workers "$NUM_WORKERS"
  --stats_batch_size 64
  --stats_max_files 8192
  --max_steps "$MAX_STEPS"
  --lr 2e-4
  --min_lr_ratio 0.1
  --warmup_steps 5000
  --weight_decay 0.01
  --grad_clip 1
  --ema_decay 0.9999
  --amp_dtype bf16
  --log_every 20
  --eval_every 5000
  --save_every 25000
  --eval_batches 32
  --sample_eval_every 10000
  --sample_eval_batches 4
  --eval_num_rollouts 4
  --eval_solver_steps 8
  --eval_seed 12345
  --receding_eval_every 50000
  --receding_eval_batches 1
  --receding_eval_horizon 80
  --wandb_project waymo-world-model
  --wandb_run_name "$RUN_NAME"
  "${wandb_args[@]}"
)

if is_truthy "${PRINT_TRAIN_COMMAND:-0}"; then
  printf '%q ' "$PYTHON" "${train_args[@]}"
  printf '\n'
fi

"$PYTHON" "${train_args[@]}" 2>&1 | tee -a "$TRAIN_LOG"

echo "===== Slurm job end: $(date) =====" | tee -a "$TRAIN_LOG"
