#!/usr/bin/env bash
# Train the time-every-layer + map model, then run exact ctx1/H30 rollout fine-tuning.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"

BASE_RUN="waymo_wm_v1_egoact_focus_raw_noclamp_win11_randstart_b8_self05_norecon_time1_mapx1_1m"
ROLLOUT_RUN="${BASE_RUN}_rollout_fixed_ctx1_h30_50k"
BASE_CKPT_DIR="$REPO_ROOT/waymo/checkpoints/$BASE_RUN"
ROLLOUT_CKPT_DIR="$REPO_ROOT/waymo/checkpoints/$ROLLOUT_RUN"
BASE_FINAL="$BASE_CKPT_DIR/final_step_01000000.pt"
ROLLOUT_FINAL="$ROLLOUT_CKPT_DIR/final_step_00050000.pt"

SESSION_NAME="${SESSION_NAME:-wm_time1_mapx1_base1m_then_rollout_h30_cuda0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"
PIPELINE_LOG="$LOG_DIR/${BASE_RUN}_then_rollout_h30_pipeline.log"
BASE_LOG="$LOG_DIR/$BASE_RUN.log"
ROLLOUT_LOG="$LOG_DIR/$ROLLOUT_RUN.log"

for required_file in "$PYTHON" "$TRAIN_SCRIPT" "$TOKENIZER_CKPT"; do
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
mkdir -p "$BASE_CKPT_DIR" "$ROLLOUT_CKPT_DIR" "$LOG_DIR" "$REPO_ROOT/waymo/wandb"

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
    SESSION_NAME="$SESSION_NAME" \
    CUDA_DEVICE="$CUDA_DEVICE" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Pipeline log: $PIPELINE_LOG"
  echo "Base log: $BASE_LOG"
  echo "H30 rollout log: $ROLLOUT_LOG"
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
  --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --device cuda
  --seed 0
  --max_rollout_window 11
  --eval_schedule shortcut
  --eval_d 0.25
  --num_workers 4
  --d_model_dyn 512
  --dyn_depth 8
  --n_heads 8
  --time_every 1
  --dynamics_attend_map
  --map_cross_every 1
  --packing_factor 2
  --n_register 8
  --k_max 64
  --grad_clip 1.0
  --amp_dtype bf16
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
  --train_decoded_loss_weight 0
  --wandb
  --wandb_project waymo-world-model
)

{
  echo "===== $(date) pipeline start ====="
  echo "session=$SESSION_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "stage1=$BASE_RUN objective=shortcut batch=8 self_fraction=0.5 steps=1000000 lr=1e-4"
  echo "stage2=$ROLLOUT_RUN objective=rollout fixed_start=1 ctx=1 horizon=30 batch=1 steps=50000 lr=1e-5"
  echo "architecture=time_every_1 map_cross_every_1"
} | tee -a "$PIPELINE_LOG"

if [[ ! -f "$BASE_FINAL" ]]; then
  base_resume=()
  if [[ -f "$BASE_CKPT_DIR/latest.pt" ]]; then
    base_resume=(--resume "$BASE_CKPT_DIR/latest.pt")
  fi
  "$PYTHON" "$TRAIN_SCRIPT" \
    "${common_args[@]}" \
    --ckpt_dir "$BASE_CKPT_DIR" \
    --seq_len 11 \
    --random_time_window_start \
    --eval_seq_len 11 \
    --eval_ctx 1 \
    --eval_horizon 10 \
    --batch_size 8 \
    --eval_batch_size 4 \
    --max_steps 1000000 \
    --log_every 100 \
    --eval_every 0 \
    --eval_max_batches 0 \
    --save_every 50000 \
    --self_fraction 0.5 \
    --bootstrap_start 0 \
    --train_objective shortcut \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --wandb_run_name "$BASE_RUN" \
    "${base_resume[@]}" \
    2>&1 | tee -a "$BASE_LOG"
else
  echo "$(date): base final already exists; skipping stage 1: $BASE_FINAL" | tee -a "$PIPELINE_LOG"
fi

if [[ ! -f "$BASE_FINAL" ]]; then
  echo "Base stage ended without expected final checkpoint: $BASE_FINAL" | tee -a "$PIPELINE_LOG" >&2
  exit 1
fi

echo "===== $(date) base complete; starting exact H30 rollout =====" | tee -a "$PIPELINE_LOG"

if [[ ! -f "$ROLLOUT_FINAL" ]]; then
  rollout_start=(--init_ckpt "$BASE_FINAL")
  if [[ -f "$ROLLOUT_CKPT_DIR/latest.pt" ]]; then
    rollout_start=(--resume "$ROLLOUT_CKPT_DIR/latest.pt")
  fi
  "$PYTHON" "$TRAIN_SCRIPT" \
    "${common_args[@]}" \
    --ckpt_dir "$ROLLOUT_CKPT_DIR" \
    --seq_len 31 \
    --eval_seq_len 31 \
    --eval_ctx 1 \
    --eval_horizon 30 \
    --batch_size 1 \
    --eval_batch_size 4 \
    --max_steps 50000 \
    --log_every 20 \
    --eval_every 5000 \
    --eval_max_batches 128 \
    --save_every 5000 \
    --no-save_latest_each_epoch \
    --self_fraction 0.5 \
    --train_objective rollout \
    --lr 1e-5 \
    --weight_decay 0 \
    --wandb_run_name "$ROLLOUT_RUN" \
    "${rollout_start[@]}" \
    2>&1 | tee -a "$ROLLOUT_LOG"
else
  echo "$(date): H30 final already exists; skipping stage 2: $ROLLOUT_FINAL" | tee -a "$PIPELINE_LOG"
fi

echo "===== $(date) pipeline complete =====" | tee -a "$PIPELINE_LOG"
