#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
WAYMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$WAYMO_ROOT/.." && pwd)"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"
OUTPUT_DIR="${OUTPUT_DIR:-$WAYMO_ROOT/cache/interaction_soft_pairs_50k_v1}"
LOG_DIR="${LOG_DIR:-$WAYMO_ROOT/logs/interaction_contrastive_learning}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/soft_pairs_50k_v1.log}"
SESSION_NAME="${SESSION_NAME:-interaction_soft_pairs_50k_v1}"

worker() {
  cd "$REPO_ROOT"
  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
  export PYTHONPATH="$WAYMO_ROOT"
  export PYTHONNOUSERSITE=1

  echo "===== shared-time-axis soft pairs 50k v1 start: $(date) ====="
  echo "PYTHON=$PYTHON"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "DEVICE=CPU"

  "$PYTHON" "$SCRIPT_DIR/build_soft_pair_dataset.py" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --splits train,val \
    --max_focus_samples 0 \
    --max_pet_steps 30 \
    --max_spatial_distance_m 6.0 \
    --history_steps 20 \
    --post_first_steps 40 \
    --num_neighbours 32 \
    --retrieval_candidates 256 \
    --dct_coefficients 3 \
    --local_scale_k 20 \
    --query_chunk_size 2048 \
    --log_every 250

  echo "===== shared-time-axis soft pairs 50k v1 end: $(date) ====="
}

if [[ "${1:-}" == "--worker" ]]; then
  worker 2>&1 | tee -a "$LOG_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "bash '$SCRIPT_PATH' --worker"
echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Log: tail -f $LOG_FILE"
echo "Output: $OUTPUT_DIR"
