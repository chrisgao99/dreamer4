#!/usr/bin/env bash
# Evaluate the Stage-3 H90 step-10k SingleQ4 checkpoints on the recorded val128 protocol.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_motion_latent_singleq4_shared_rollout_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
SEMANTIC_READER_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_semantic_reader_agent32_zonly_d256_depth2_b8_20k_v1/best.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
SUBSET_MANIFEST="$REPO_ROOT/waymo/evaluation/val_random128_seed0_manifest.json"

RUN_GROUP="${RUN_GROUP:-motion_latent_singleq4_stage3_step10k_ctx1_h80_val128_seed0_gtvalidq}"
BASE_OUT_DIR="$REPO_ROOT/waymo/eval_results/world_model/$RUN_GROUP"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
NUM_WORKERS="${NUM_WORKERS:-4}"
HORIZONS="${HORIZONS:-10 20 30 50 80}"
SESSION_TAG="${SESSION_TAG:-_stage3_step10k_gtvalidq_run2}"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  for spec in "noqgt:0" "qgt_detach:1"; do
    variant="${spec%%:*}"
    cuda_device="${spec##*:}"
    session_name="wm_singleq4_eval_${variant}${SESSION_TAG}_cuda${cuda_device}"
    if tmux has-session -t "$session_name" 2>/dev/null; then
      echo "tmux session already exists: $session_name" >&2
      exit 1
    fi
    printf -v tmux_command '%q ' env \
      RUN_INSIDE_TMUX=1 VARIANT="$variant" CUDA_DEVICE="$cuda_device" \
      REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" NUM_WORKERS="$NUM_WORKERS" \
      HORIZONS="$HORIZONS" RUN_GROUP="$RUN_GROUP" SESSION_TAG="$SESSION_TAG" \
      SESSION_NAME="$session_name" bash "$SCRIPT_PATH"
    tmux new-session -d -s "$session_name" -c "$REPO_ROOT" "$tmux_command"
    tmux set-option -t "$session_name" remain-on-exit on
    echo "Started $session_name: variant=$variant physical_cuda=$cuda_device"
  done
  echo "Results: $BASE_OUT_DIR"
  echo "Logs: $LOG_DIR/${RUN_GROUP}_{noqgt,qgt_detach}.log"
  exit 0
fi

VARIANT="${VARIANT:?VARIANT is required inside tmux}"
CUDA_DEVICE="${CUDA_DEVICE:?CUDA_DEVICE is required inside tmux}"
SESSION_NAME="${SESSION_NAME:-wm_singleq4_eval_${VARIANT}_stage3_step10k_cuda${CUDA_DEVICE}}"
case "$VARIANT" in
  noqgt)
    EVAL_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_singleq4_noqgt_chunk32s30_stage3_h90_b1_30000/step_00010000.pt"
    ;;
  qgt_detach)
    EVAL_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_singleq4_qgt_detach_chunk32s30_stage3_h90_b1_30000/step_00010000.pt"
    ;;
  *)
    echo "Unsupported VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

OUT_DIR="$BASE_OUT_DIR/$VARIANT"
LOG_FILE="$LOG_DIR/${RUN_GROUP}_${VARIANT}.log"
OUTPUT_JSON="$OUT_DIR/stage3_h90_step10k_ctx1_shared_h80_h10_20_30_50_80_val128.json"
for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$SEMANTIC_READER_CKPT" "$EVAL_CKPT" "$SUBSET_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) SingleQ4 Stage-3 step-10k validation start ====="
  echo "session=$SESSION_NAME variant=$VARIANT physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "checkpoint=$EVAL_CKPT output=$OUTPUT_JSON"
  echo "protocol=recorded_val128 batch_size=1 ctx=1 shared_rollout_horizon=80"
  echo "q_metric_mask=all_agents_and_gt_valid_only"
  echo "metric_horizons=$HORIZONS subset_manifest=$SUBSET_MANIFEST"
  if [[ -f "$OUTPUT_JSON" ]]; then
    echo "Skipping completed result: $OUTPUT_JSON"
  else
    "$PYTHON" "$EVAL_SCRIPT" \
      --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
      --tokenizer_ckpt "$TOKENIZER_CKPT" --semantic_reader_ckpt "$SEMANTIC_READER_CKPT" \
      --eval_ckpt "$EVAL_CKPT" \
      --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
      --eval_batch_size 1 --eval_max_batches 0 \
      --subset_manifest "$SUBSET_MANIFEST" --subset_size 128 --subset_seed 0 \
      --eval_seq_len 81 --eval_ctx 1 --horizons "$HORIZONS" \
      --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
      --max_rollout_window 10 --packing_factor 2 --n_register 8 --k_max 64 \
      --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
      --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
      --output_json "$OUTPUT_JSON"
  fi
  echo "===== $(date) SingleQ4 Stage-3 step-10k validation complete: $VARIANT ====="
} 2>&1 | tee -a "$LOG_FILE"
