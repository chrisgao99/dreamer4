#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
WAYMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$WAYMO_ROOT/.." && pwd)"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"
OUTPUT_DIR="${OUTPUT_DIR:-$WAYMO_ROOT/cache/interaction_full_pairs_50k_v2}"
LOG_DIR="${LOG_DIR:-$WAYMO_ROOT/logs/interaction_contrastive_learning}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/full_pairs_50k_v2.log}"
SESSION_NAME="${SESSION_NAME:-interaction_full_pairs_50k_v2}"

worker() {
  cd "$REPO_ROOT"
  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
  export PYTHONPATH="$WAYMO_ROOT"
  export PYTHONNOUSERSITE=1
  export MPLCONFIGDIR="/tmp/matplotlib-interaction-full-pairs-v2"

  echo "===== full 91-step physical-contact pairs v2 start: $(date) ====="
  echo "PYTHON=$PYTHON"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "DEVICE=CPU"

  "$PYTHON" "$SCRIPT_DIR/build_full_pair_dataset.py" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --shard_size "${SHARD_SIZE:-5000}" \
    --contact_buffer_m "${CONTACT_BUFFER_M:-1.0}" \
    --pet_soft_scale_s "${PET_SOFT_SCALE_S:-3.0}" \
    --non_ooi_top_k_per_focus "${NON_OOI_TOP_K_PER_FOCUS:-0}" \
    ${FULL_PAIR_RESUME:+--resume}

  echo "===== full 91-step physical-contact pairs v2 end: $(date) ====="
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

tmux new-session -d -s "$SESSION_NAME" \
  "env PYTHON='$PYTHON' DATA_ROOT='$DATA_ROOT' OUTPUT_DIR='$OUTPUT_DIR' LOG_DIR='$LOG_DIR' \
LOG_FILE='$LOG_FILE' SESSION_NAME='$SESSION_NAME' FULL_PAIR_RESUME='${FULL_PAIR_RESUME:-}' \
bash '$SCRIPT_PATH' --worker"
echo "Started tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Log: tail -f $LOG_FILE"
echo "Output: $OUTPUT_DIR"
