#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
WAYMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$WAYMO_ROOT/.." && pwd)"
RMS_PYTHON="${RMS_PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
RMS_DATASET_DIR="${RMS_DATASET_DIR:-$WAYMO_ROOT/cache/interaction_full_pairs_50k_v2_no_topk}"
RMS_OUTPUT_DIR="${RMS_OUTPUT_DIR:-$WAYMO_ROOT/cache/interaction_full_pairs_50k_v2_no_topk_rms_v0}"
RMS_LOG_DIR="${RMS_LOG_DIR:-$WAYMO_ROOT/logs/interaction_contrastive_learning}"
RMS_LOG_FILE="${RMS_LOG_FILE:-$RMS_LOG_DIR/full_pairs_50k_v2_no_topk_rms_v0.log}"
RMS_SESSION_NAME="${RMS_SESSION_NAME:-interaction_full_pairs_50k_v2_no_topk_rms_v0}"

worker() {
  cd "$REPO_ROOT"
  mkdir -p "$RMS_OUTPUT_DIR" "$RMS_LOG_DIR"
  export PYTHONPATH="$WAYMO_ROOT"
  export PYTHONNOUSERSITE=1

  echo "===== event-aligned RMS pipeline start: $(date) ====="
  echo "RMS_DATASET_DIR=$RMS_DATASET_DIR"
  echo "RMS_OUTPUT_DIR=$RMS_OUTPUT_DIR"
  echo "DEVICE=CPU"

  wait_count=0
  while [[ ! -f "$RMS_DATASET_DIR/summary.json" ]]; do
    if (( wait_count % 10 == 0 )); then
      echo "waiting for completed source dataset: $RMS_DATASET_DIR/summary.json ($(date))"
    fi
    wait_count=$((wait_count + 1))
    sleep 60
  done

  "$RMS_PYTHON" "$SCRIPT_DIR/build_full_pair_rms_neighbors.py" \
    --dataset_dir "$RMS_DATASET_DIR" \
    --output_dir "$RMS_OUTPUT_DIR" \
    --retrieval_candidates "${RETRIEVAL_CANDIDATES:-1024}" \
    --num_neighbours "${NUM_NEIGHBOURS:-32}" \
    --validation_anchors "${VALIDATION_ANCHORS:-96}"

  "$RMS_PYTHON" "$SCRIPT_DIR/visualize_full_pair_rms_neighbors.py" \
    --rms_dir "$RMS_OUTPUT_DIR" \
    --output_dir "$RMS_OUTPUT_DIR/visual_audit" \
    --split train \
    --num_samples "${AUDIT_SAMPLES:-24}" \
    --rank 1

  echo "===== event-aligned RMS pipeline end: $(date) ====="
}

if [[ "${1:-}" == "--worker" ]]; then
  worker 2>&1 | tee -a "$RMS_LOG_FILE"
  exit 0
fi

mkdir -p "$RMS_LOG_DIR"
if tmux has-session -t "$RMS_SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $RMS_SESSION_NAME"
  echo "Attach with: tmux attach -t $RMS_SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$RMS_SESSION_NAME" "bash '$SCRIPT_PATH' --worker"
echo "Started tmux session: $RMS_SESSION_NAME"
echo "Attach: tmux attach -t $RMS_SESSION_NAME"
echo "Log: tail -f $RMS_LOG_FILE"
echo "Output: $RMS_OUTPUT_DIR"
