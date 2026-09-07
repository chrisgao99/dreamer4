#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
WAYMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$WAYMO_ROOT/.." && pwd)"

PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$WAYMO_ROOT/eval_results/interaction_contrastive_global_retrieval_10}"
LOG_DIR="${LOG_DIR:-$WAYMO_ROOT/logs/interaction_contrastive_learning}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/global_contrastive_retrieval_10_cuda${CUDA_DEVICE}.log}"
SESSION_NAME="${SESSION_NAME:-interaction_global_retrieval_10_cuda${CUDA_DEVICE}}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:-$WAYMO_ROOT/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt}"
READER_CHECKPOINT="${READER_CHECKPOINT:-$WAYMO_ROOT/checkpoints/interaction_contrastive_hard_relneg_dupfiltered_v1/best_stage_a.pt}"
HARD_CHECKPOINT="${HARD_CHECKPOINT:-$WAYMO_ROOT/checkpoints/interaction_contrastive_hard_relneg_dupfiltered_v1/best.pt}"
HYBRID_CHECKPOINT="${HYBRID_CHECKPOINT:-$WAYMO_ROOT/checkpoints/interaction_contrastive_hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3/best.pt}"
CACHE="${CACHE:-$WAYMO_ROOT/cache/interaction_full_pairs_50k_v2_contrastive_v1/val_contrastive_training.npz}"
RMS_FEATURES="${RMS_FEATURES:-$WAYMO_ROOT/cache/interaction_full_pairs_50k_v2_no_topk_rms_v0/val_rms_features.npz}"
VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-$WAYMO_ROOT/checkpoints/interaction_contrastive_hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3/validation_manifest.npz}"

worker() {
  mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
  cd "$REPO_ROOT"

  export PYTHONPATH="$WAYMO_ROOT"
  export PYTHONNOUSERSITE=1
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-interaction-global-retrieval-${USER:-local}}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

  for required in \
    "$SCRIPT_DIR/visualize_global_contrastive_retrieval.py" \
    "$BASE_CHECKPOINT" \
    "$READER_CHECKPOINT" \
    "$HARD_CHECKPOINT" \
    "$HYBRID_CHECKPOINT" \
    "$CACHE" \
    "$RMS_FEATURES" \
    "$VALIDATION_MANIFEST"; do
    [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
  done

  {
    echo "===== global contrastive retrieval start: $(date) ====="
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "STAGES=raw_z,reader_z,hard,hybrid"
    echo "READER_CHECKPOINT=$READER_CHECKPOINT"

    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u \
      "$SCRIPT_DIR/visualize_global_contrastive_retrieval.py" \
      --base_checkpoint "$BASE_CHECKPOINT" \
      --reader_checkpoint "$READER_CHECKPOINT" \
      --hard_checkpoint "$HARD_CHECKPOINT" \
      --hybrid_checkpoint "$HYBRID_CHECKPOINT" \
      --cache "$CACHE" \
      --rms_features "$RMS_FEATURES" \
      --validation_manifest "$VALIDATION_MANIFEST" \
      --output_dir "$OUTPUT_DIR" \
      --num_anchors "${NUM_ANCHORS:-10}" \
      --top_k "${TOP_K:-10}" \
      --candidate_scope "${CANDIDATE_SCOPE:-history}" \
      --history_steps "${HISTORY_STEPS:-32}" \
      --batch_size "${BATCH_SIZE:-8}" \
      --device cuda

    echo "===== global contrastive retrieval end: $(date) ====="
    echo "report: $OUTPUT_DIR/index.html"
  } 2>&1 | tee -a "$LOG_FILE"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker
  exit 0
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }
mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  echo "Attach with: tmux attach -t $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "env PYTHON='$PYTHON' CUDA_DEVICE='$CUDA_DEVICE' OUTPUT_DIR='$OUTPUT_DIR' LOG_DIR='$LOG_DIR' LOG_FILE='$LOG_FILE' SESSION_NAME='$SESSION_NAME' BASE_CHECKPOINT='$BASE_CHECKPOINT' READER_CHECKPOINT='$READER_CHECKPOINT' HARD_CHECKPOINT='$HARD_CHECKPOINT' HYBRID_CHECKPOINT='$HYBRID_CHECKPOINT' CACHE='$CACHE' RMS_FEATURES='$RMS_FEATURES' VALIDATION_MANIFEST='$VALIDATION_MANIFEST' NUM_ANCHORS='${NUM_ANCHORS:-10}' TOP_K='${TOP_K:-10}' CANDIDATE_SCOPE='${CANDIDATE_SCOPE:-history}' HISTORY_STEPS='${HISTORY_STEPS:-32}' BATCH_SIZE='${BATCH_SIZE:-8}' OMP_NUM_THREADS='${OMP_NUM_THREADS:-8}' bash '$SCRIPT_PATH' --worker"

echo "Started detached global retrieval on CUDA $CUDA_DEVICE"
echo "tmux:   $SESSION_NAME"
echo "attach: tmux attach -t $SESSION_NAME"
echo "log:    tail -f $LOG_FILE"
echo "output: $OUTPUT_DIR/index.html"
