#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
WAYMO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$WAYMO_ROOT/.." && pwd)}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
RAW_DIR="${RAW_DIR:-/p/liverobotics/waymo_open_dataset_motion/tf_example/training}"

RUN_NAME="${RUN_NAME:-waymo_ooi_training_all}"
SESSION_NAME="${SESSION_NAME:-prep_${RUN_NAME}}"
STATS_DIR="${STATS_DIR:-$WAYMO_ROOT/evaluation/reports/ooi_raw_stats_train_all}"
OUT_DIR="${OUT_DIR:-$WAYMO_ROOT/data/waymo_vector_dataset_ooi_centered_training_all}"
LOG_DIR="${LOG_DIR:-$WAYMO_ROOT/logs/data_prep}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}.log}"

MAX_FILES="${MAX_FILES:-0}"
MAX_RECORDS_PER_FILE="${MAX_RECORDS_PER_FILE:-}"
NUM_FOCUS_SAMPLES="${NUM_FOCUS_SAMPLES:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-1000}"

NUM_AGENTS="${NUM_AGENTS:-32}"
AGENT_DISTANCE_THRESHOLD="${AGENT_DISTANCE_THRESHOLD:-80}"
MAP_DISTANCE_THRESHOLD="${MAP_DISTANCE_THRESHOLD:-100}"
MAX_MAP_POLYLINES="${MAX_MAP_POLYLINES:-256}"
MAX_POINTS_PER_POLYLINE="${MAX_POINTS_PER_POLYLINE:-20}"

# Optional: set this to skip samples already listed in an existing manifest.
# Example:
#   EXCLUDE_MANIFEST=$WAYMO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/manifest.csv
EXCLUDE_MANIFEST="${EXCLUDE_MANIFEST:-}"
EXCLUDE_LEVEL="${EXCLUDE_LEVEL:-sample}"

worker() {
  cd "$REPO_ROOT"
  mkdir -p "$STATS_DIR" "$OUT_DIR" "$LOG_DIR"

  if [[ ! -x "$PYTHON" ]]; then
    echo "Python not executable: $PYTHON" >&2
    exit 1
  fi
  if [[ ! -d "$RAW_DIR" ]]; then
    echo "Raw Waymo training dir not found: $RAW_DIR" >&2
    exit 1
  fi
  if [[ -n "$EXCLUDE_MANIFEST" && ! -f "$EXCLUDE_MANIFEST" ]]; then
    echo "Exclude manifest not found: $EXCLUDE_MANIFEST" >&2
    exit 1
  fi

  echo "===== OOI training data prep start: $(date) ====="
  echo "REPO_ROOT=$REPO_ROOT"
  echo "PYTHON=$PYTHON"
  echo "RAW_DIR=$RAW_DIR"
  echo "STATS_DIR=$STATS_DIR"
  echo "OUT_DIR=$OUT_DIR"
  echo "NUM_FOCUS_SAMPLES=$NUM_FOCUS_SAMPLES"
  echo "EXCLUDE_MANIFEST=${EXCLUDE_MANIFEST:-<none>}"

  analyze_args=(
    waymo/data_prep/analyze_waymo_ooi_raw.py
    --raw_dir "$RAW_DIR"
    --output_dir "$STATS_DIR"
    --max_files "$MAX_FILES"
    --log_every "$LOG_EVERY"
  )
  if [[ -n "$MAX_RECORDS_PER_FILE" ]]; then
    analyze_args+=(--max_records_per_file "$MAX_RECORDS_PER_FILE")
  fi

  echo "===== Stage 1/2: analyze raw OOI stats: $(date) ====="
  "$PYTHON" "${analyze_args[@]}"

  stats_csv="$STATS_DIR/waymo_ooi_scenario_stats.csv"
  if [[ ! -f "$stats_csv" ]]; then
    echo "Expected stats CSV missing after analyze stage: $stats_csv" >&2
    exit 1
  fi

  prepare_args=(
    waymo/data_prep/prepare_waymo_vector_ooi_centered.py
    --stats_csv "$stats_csv"
    --output_dir "$OUT_DIR"
    --num_focus_samples "$NUM_FOCUS_SAMPLES"
    --val_fraction "$VAL_FRACTION"
    --seed "$SEED"
    --num_agents "$NUM_AGENTS"
    --agent_distance_threshold "$AGENT_DISTANCE_THRESHOLD"
    --map_distance_threshold "$MAP_DISTANCE_THRESHOLD"
    --max_map_polylines "$MAX_MAP_POLYLINES"
    --max_points_per_polyline "$MAX_POINTS_PER_POLYLINE"
    --log_every "$LOG_EVERY"
  )
  if [[ -n "$EXCLUDE_MANIFEST" ]]; then
    prepare_args+=(--exclude_manifest "$EXCLUDE_MANIFEST" --exclude_level "$EXCLUDE_LEVEL")
  fi

  echo "===== Stage 2/2: write OOI-centered NPZ: $(date) ====="
  "$PYTHON" "${prepare_args[@]}"

  train_count=$(find "$OUT_DIR/train" -name '*.npz' | wc -l)
  val_count=$(find "$OUT_DIR/val" -name '*.npz' | wc -l)
  echo "Prepared train NPZ: $train_count"
  echo "Prepared val NPZ: $val_count"
  echo "Stats CSV: $stats_csv"
  echo "Manifest: $OUT_DIR/manifest.csv"
  echo "Summary: $OUT_DIR/prepare_summary.json"
  echo "===== OOI training data prep end: $(date) ====="
}

if [[ "${1:-}" == "--worker" ]]; then
  worker 2>&1 | tee -a "$LOG_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR"
{
  echo "===== tmux launch requested: $(date) ====="
  echo "SCRIPT_PATH=$SCRIPT_PATH"
  echo "REPO_ROOT=$REPO_ROOT"
  echo "RAW_DIR=$RAW_DIR"
  echo "STATS_DIR=$STATS_DIR"
  echo "OUT_DIR=$OUT_DIR"
} >> "$LOG_FILE"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "bash '$SCRIPT_PATH' --worker"
echo "Started detached tmux session: $SESSION_NAME"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Tail log: tail -f $LOG_FILE"
echo "Output dir: $OUT_DIR"
