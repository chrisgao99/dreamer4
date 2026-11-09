#!/usr/bin/env bash
# Evaluate all three SingleQ4 stages for both variants with one shared H90 rollout.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_motion_latent_singleq4_shared_rollout_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
SUBSET_MANIFEST="$REPO_ROOT/waymo/evaluation/val_random128_seed0_manifest.json"

RUN_GROUP="${RUN_GROUP:-motion_latent_singleq4_all_stages_ctx1_h90_val128_seed0_gtvalidq}"
BASE_OUT_DIR="$REPO_ROOT/waymo/eval_results/world_model/$RUN_GROUP"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
NUM_WORKERS="${NUM_WORKERS:-4}"
HORIZONS="${HORIZONS:-10 20 30 50 80 90}"
SESSION_TAG="${SESSION_TAG:-_allstages_h90_gtvalidq_run2}"

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
SESSION_NAME="${SESSION_NAME:-wm_singleq4_eval_${VARIANT}_allstages_h90_cuda${CUDA_DEVICE}}"
case "$VARIANT" in
  noqgt)
    STAGE1_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_singleq4_noqgt_chunk32s30_stage1_b8_600000/final_step_00600000.pt"
    STAGE2_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_singleq4_noqgt_chunk32s30_stage2_h30_b1_30000/step_00015000.pt"
    STAGE3_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_singleq4_noqgt_chunk32s30_stage3_h90_b1_30000/step_00010000.pt"
    ;;
  qgt_detach)
    STAGE1_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_singleq4_qgt_detach_chunk32s30_stage1_b8_600000/final_step_00600000.pt"
    STAGE2_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_singleq4_qgt_detach_chunk32s30_stage2_h30_b1_30000/step_00015000.pt"
    STAGE3_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_motion_latent_singleq4_qgt_detach_chunk32s30_stage3_h90_b1_30000/step_00010000.pt"
    ;;
  *)
    echo "Unsupported VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

OUT_DIR="$BASE_OUT_DIR/$VARIANT"
LOG_FILE="$LOG_DIR/${RUN_GROUP}_${VARIANT}.log"
for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$STAGE1_CKPT" "$STAGE2_CKPT" "$STAGE3_CKPT" "$SUBSET_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) SingleQ4 all-stage shared-H90 validation start ====="
  echo "session=$SESSION_NAME variant=$VARIANT physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "protocol=recorded_val128 batch_size=1 ctx=1 shared_rollout_horizon=90"
  echo "q_metric_mask=all_agents_and_gt_valid_only"
  echo "metric_horizons=$HORIZONS subset_manifest=$SUBSET_MANIFEST"

  labels=(stage1_600k stage2_15k stage3_h90_10k)
  checkpoints=("$STAGE1_CKPT" "$STAGE2_CKPT" "$STAGE3_CKPT")
  for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    checkpoint="${checkpoints[$index]}"
    output_json="$OUT_DIR/${label}_ctx1_shared_h90_h10_20_30_50_80_90_val128.json"
    echo "===== $(date) evaluating $VARIANT/$label ====="
    echo "checkpoint=$checkpoint output=$output_json"
    if [[ -f "$output_json" ]]; then
      echo "Skipping completed result: $output_json"
      continue
    fi
    "$PYTHON" "$EVAL_SCRIPT" \
      --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
      --tokenizer_ckpt "$TOKENIZER_CKPT" \
      --eval_ckpt "$checkpoint" \
      --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
      --eval_batch_size 1 --eval_max_batches 0 \
      --subset_manifest "$SUBSET_MANIFEST" --subset_size 128 --subset_seed 0 \
      --eval_seq_len 91 --eval_ctx 1 --horizons "$HORIZONS" \
      --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
      --max_rollout_window 10 --packing_factor 2 --n_register 8 --k_max 64 \
      --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
      --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
      --output_json "$output_json"
    echo "===== $(date) completed $VARIANT/$label ====="
  done
  echo "===== $(date) SingleQ4 all-stage shared-H90 validation complete: $VARIANT ====="
} 2>&1 | tee -a "$LOG_FILE"
