#!/usr/bin/env bash
# Paired fixed-val128 ablation of tokenizer overlap stitching for the stage-3
# 40k world-model checkpoint. All tokenizer calls remain <=32 timesteps.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_shared_rollout_horizons.py"
SUMMARY_SCRIPT="$REPO_ROOT/waymo/evaluation/summarize_tokenizer_center_ablation.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
WORLD_MODEL_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage/waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k/step_00040000.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
SUBSET_MANIFEST="$REPO_ROOT/waymo/evaluation/val_random128_seed0_manifest.json"

CUDA_DEVICE=1
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
SESSION_NAME="${SESSION_NAME:-wm_step40k_tokenizer_center_val128_cuda1}"
RUN_ID="${RUN_ID:-world_model_step40k_tokenizer_center_ablation_val128_seed0}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model/$RUN_ID}"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

BASELINE_JSON="$OUT_DIR/baseline_keepfirst_s30.json"
DECODER_CENTER_JSON="$OUT_DIR/decoder_center_s16.json"
ENCDEC_CENTER_JSON="$OUT_DIR/encoder_decoder_center_s16.json"

for path in "$PYTHON" "$EVAL_SCRIPT" "$SUMMARY_SCRIPT" "$TOKENIZER_CKPT" "$WORLD_MODEL_CKPT" "$SUBSET_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux is not available" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" NUM_WORKERS="$NUM_WORKERS" \
    EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES" SESSION_NAME="$SESSION_NAME" RUN_ID="$RUN_ID" \
    OUT_DIR="$OUT_DIR" LOG_FILE="$LOG_FILE" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started detached tmux session: $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Results: $OUT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

run_variant() {
  local label="$1"
  local output_json="$2"
  local encode_stride="$3"
  local encode_stitch="$4"
  local decode_stride="$5"
  local decode_stitch="$6"
  echo "===== $(date) variant=$label start ====="
  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$WORLD_MODEL_CKPT" \
    --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
    --eval_batch_size 1 --eval_max_batches "$EVAL_MAX_BATCHES" \
    --subset_manifest "$SUBSET_MANIFEST" --subset_size 128 --subset_seed 0 \
    --eval_seq_len 91 --eval_ctx 1 --horizons "10 30 50 80 90" \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
    --tokenizer_encode_chunk_stride "$encode_stride" --tokenizer_encode_stitch_mode "$encode_stitch" \
    --tokenizer_decode_chunk_stride "$decode_stride" --tokenizer_decode_stitch_mode "$decode_stitch" \
    --max_rollout_window 11 --eval_schedule shortcut --eval_d 0.25 \
    --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
    --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
    --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
    --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
    --output_json "$output_json"
  echo "===== $(date) variant=$label complete ====="
}

{
  echo "===== $(date) step40k tokenizer center-select paired val128 ablation start ====="
  echo "physical_cuda=$CUDA_DEVICE checkpoint=$WORLD_MODEL_CKPT"
  echo "protocol=fixed_val128 ctx1 shared_H90 shortcut4 seed0 max_tokenizer_chunk32"
  run_variant baseline_keepfirst_s30 "$BASELINE_JSON" 30 keep_first 30 keep_first
  run_variant decoder_center_s16 "$DECODER_CENTER_JSON" 30 keep_first 16 center_select
  run_variant encoder_decoder_center_s16 "$ENCDEC_CENTER_JSON" 16 center_select 16 center_select
  if [[ "$EVAL_MAX_BATCHES" -eq 0 ]]; then
    "$PYTHON" "$SUMMARY_SCRIPT" \
      --baseline "$BASELINE_JSON" \
      --decoder_center "$DECODER_CENTER_JSON" \
      --encoder_decoder_center "$ENCDEC_CENTER_JSON" \
      --output_dir "$OUT_DIR"
  else
    echo "smoke_only=true; skipping paired summary"
  fi
  echo "===== $(date) step40k tokenizer center-select paired val128 ablation complete ====="
} 2>&1 | tee -a "$LOG_FILE"
