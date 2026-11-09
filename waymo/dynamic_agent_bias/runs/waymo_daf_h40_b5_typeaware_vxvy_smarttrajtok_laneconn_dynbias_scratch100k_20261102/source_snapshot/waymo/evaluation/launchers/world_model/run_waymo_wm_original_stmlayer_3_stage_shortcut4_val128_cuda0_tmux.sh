#!/usr/bin/env bash
# Evaluate the best checkpoint from each original STM-layer stage serially.
# Each sample gets one ctx=1 -> H90 rollout with four shortcut solver steps
# per predicted frame; H10/H30/H50/H80/H90 are prefixes of that same rollout.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_shared_rollout_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"

BUNDLE_ROOT="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage"
STAGE1_CKPT="$BUNDLE_ROOT/waymo_wm_v1_egoact_focus_raw_noclamp_win11_randstart_b8_self05_norecon_time1_mapx1_1m/step_00500000.pt"
STAGE2_CKPT="$BUNDLE_ROOT/waymo_wm_time1_mapx1_best_upto600k_exact_ctx1_h30_b1_50k/step_00010000.pt"
STAGE3_CKPT="$BUNDLE_ROOT/waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k/step_00050000.pt"

BASE_RESULT_DIR="$REPO_ROOT/waymo/eval_results/world_model/motion_latent_three_stages_shared_h90_d1_val128_seed0"
SUBSET_MANIFEST="$BASE_RESULT_DIR/val_random128_seed0_manifest.json"
RUN_GROUP="waymo_wm_original_stmlayer_3_stage_chunk32s30_shared_h90_shortcut4_val128_seed0"
OUT_DIR="$REPO_ROOT/waymo/eval_results/world_model/$RUN_GROUP"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
LOG_FILE="$LOG_DIR/$RUN_GROUP.log"

SESSION_NAME="${SESSION_NAME:-wm_original_stmlayer_3stage_shortcut4_val128_cuda0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HORIZONS="${HORIZONS:-10 30 50 80 90}"

for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" \
  "$STAGE1_CKPT" "$STAGE2_CKPT" "$STAGE3_CKPT" "$SUBSET_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    SESSION_NAME="$SESSION_NAME" CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" \
    HORIZONS="$HORIZONS" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Results: $OUT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) original STM-layer three-stage shortcut-4 evaluation start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "protocol=same_recorded_val_samples=128 batch_size=1 ctx=1 shared_rollout_horizon=90 shortcut_steps=4 eval_d=0.25"
  echo "tokenizer_chunk_window=32 tokenizer_chunk_stride=30 ranges_for_T91=[0,32),[30,62),[59,91)"
  echo "metric_horizons=$HORIZONS subset_manifest=$SUBSET_MANIFEST"

  labels=(stage1_best_step500k stage2_best_step10k stage3_best_step50k)
  checkpoints=("$STAGE1_CKPT" "$STAGE2_CKPT" "$STAGE3_CKPT")
  for index in "${!labels[@]}"; do
    label="${labels[$index]}"
    checkpoint="${checkpoints[$index]}"
    output_json="$OUT_DIR/${label}_ctx1_chunk32s30_shared_h90_shortcut4_h10_30_50_80_90_val128.json"
    echo "===== $(date) evaluating $label ====="
    echo "checkpoint=$checkpoint output=$output_json"
    if [[ -f "$output_json" ]]; then
      echo "Skipping completed result: $output_json"
      continue
    fi
    "$PYTHON" "$EVAL_SCRIPT" \
      --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
      --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$checkpoint" \
      --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
      --eval_batch_size 1 --eval_max_batches 0 \
      --subset_manifest "$SUBSET_MANIFEST" --subset_size 128 --subset_seed 0 \
      --eval_seq_len 91 --eval_ctx 1 --horizons "$HORIZONS" \
      --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
      --max_rollout_window 11 --eval_schedule shortcut --eval_d 0.25 \
      --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
      --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
      --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
      --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
      --output_json "$output_json"
    echo "===== $(date) completed $label ====="
  done
  echo "===== $(date) original STM-layer three-stage shortcut-4 evaluation complete ====="
} 2>&1 | tee -a "$LOG_FILE"
