#!/usr/bin/env bash
# Exact fixed-parameter ctx1/H30 rollout fine-tuning for MotionLatent V1.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_motion_latent_v1.py"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
READER_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_semantic_reader_agent32_zonly_d256_depth2_b8_20k_v1/best.pt"
INIT_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_v1_qkin_preader_ctx1ctx11_b8_mapx1_1m/step_00650000.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"

RUN_NAME="waymo_motion_latent_v1_qkin_preader_exact_fixed_ctx1_h30_b4_init650k_50k"
SESSION_NAME="${SESSION_NAME:-wm_motion_latent_v1_exact_h30_init650k_cuda1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
CKPT_DIR="$REPO_ROOT/waymo/checkpoints/$RUN_NAME"
FINAL_CKPT="$CKPT_DIR/final_step_00050000.pt"
TRAIN_LOG="$REPO_ROOT/waymo/logs/wm/$RUN_NAME.log"
EVAL_LOG="$REPO_ROOT/waymo/logs/evaluation/${RUN_NAME}_eval_h30.log"
OUTPUT_JSON="$REPO_ROOT/waymo/eval_results/world_model/$RUN_NAME/final_ctx01_h30_batches128.json"

for required_file in "$PYTHON" "$TRAIN_SCRIPT" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$READER_CKPT" "$INIT_CKPT"; do
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
mkdir -p "$CKPT_DIR" "$(dirname "$TRAIN_LOG")" "$(dirname "$EVAL_LOG")" "$(dirname "$OUTPUT_JSON")" "$REPO_ROOT/waymo/wandb"

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
  echo "Train log: $TRAIN_LOG"
  echo "Eval log: $EVAL_LOG"
  echo "Results: $OUTPUT_JSON"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

if [[ ! -f "$FINAL_CKPT" ]]; then
  start_args=(--init_ckpt "$INIT_CKPT")
  if [[ -f "$CKPT_DIR/latest.pt" ]]; then
    start_args=(--resume "$CKPT_DIR/latest.pt")
  fi
  {
    echo "===== $(date) ====="
    echo "run_name=$RUN_NAME session=$SESSION_NAME cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
    echo "init_ckpt=$INIT_CKPT"
    echo "task=exact_fixed_parameter_rollout fixed_start=1 ctx=1 horizon=30 direct_d1=1"
    echo "batch_size=4 max_steps=50000 lr=1e-5 max_context=11"
    echo "checkpoint_dir=$CKPT_DIR"
    echo "========================"
  } | tee -a "$TRAIN_LOG"

  "$PYTHON" "$TRAIN_SCRIPT" \
    --data_dir "$DATA_ROOT/train" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" \
    --semantic_reader_ckpt "$READER_CKPT" \
    --ckpt_dir "$CKPT_DIR" \
    "${start_args[@]}" \
    --device cuda \
    --seed 0 \
    --batch_size 4 \
    --num_workers 4 \
    --max_steps 50000 \
    --train_mode exact_rollout \
    --rollout_end 30 \
    --max_context 11 \
    --d_model 512 \
    --depth 8 \
    --n_heads 8 \
    --time_every 4 \
    --map_cross_every 1 \
    --packing_factor 2 \
    --n_register 8 \
    --k_max 64 \
    --kinematic_dt 0.1 \
    --motion_weight 1 \
    --motion_validity_weight 0.2 \
    --consistency_weight 0.1 \
    --lr 1e-5 \
    --weight_decay 0 \
    --grad_clip 1 \
    --amp_dtype bf16 \
    --log_every 20 \
    --save_every 5000 \
    --ego_action_source focus \
    --ego_action_normalization raw \
    --wandb \
    --wandb_project waymo-world-model \
    --wandb_run_name "$RUN_NAME" \
    2>&1 | tee -a "$TRAIN_LOG"
fi

if [[ ! -f "$FINAL_CKPT" ]]; then
  echo "Training ended without expected final checkpoint: $FINAL_CKPT" >&2
  exit 1
fi

{
  echo "===== $(date) final H30 evaluation ====="
  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$DATA_ROOT/train" \
    --val_data_dir "$DATA_ROOT/val" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" \
    --eval_ckpt "$FINAL_CKPT" \
    --device cuda \
    --seed 0 \
    --eval_batch_size 4 \
    --eval_max_batches 128 \
    --num_workers 4 \
    --eval_seq_len 31 \
    --eval_ctx 1 \
    --horizons 30 \
    --max_rollout_window 11 \
    --packing_factor 2 \
    --k_max 64 \
    --use_ego_actions \
    --ego_action_source focus \
    --ego_action_normalization raw \
    --no-ego_action_clamp \
    --agent_far_weight 0.25 \
    --agent_near_radius_m 50 \
    --agent_distance_source focus \
    --output_json "$OUTPUT_JSON"
  echo "Finished at $(date)"
} 2>&1 | tee -a "$EVAL_LOG"
