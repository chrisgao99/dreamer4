#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/sfs/weka/scratch/baz7dy/tri30/dreamer4}"
PYTHON="${PYTHON:-$HOME/.conda/envs/dreamer4/bin/python}"

STATS_CSV="${STATS_CSV:-$REPO_ROOT/waymo/evaluation/reports/ooi_raw_stats_train/waymo_ooi_scenario_stats.csv}"
BASE_DATA_DIR="${BASE_DATA_DIR:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k}"
OUT="${OUT:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_extra_50k_seed1}"
EXCLUDE_MANIFEST="${EXCLUDE_MANIFEST:-$BASE_DATA_DIR/manifest.csv}"
LOG="${LOG:-$REPO_ROOT/waymo/prepare_waymo_vector_ooi_centered_incremental.log}"
PIDFILE="${PIDFILE:-$REPO_ROOT/waymo/prepare_waymo_vector_ooi_centered_incremental.pid}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$LOG")" "$OUT"

if [[ ! -f "$STATS_CSV" ]]; then
  echo "Stats CSV not found: $STATS_CSV" >&2
  echo "Run waymo/data_prep/analyze_waymo_ooi_raw.py first, or pass STATS_CSV=/path/to/waymo_ooi_scenario_stats.csv" >&2
  exit 1
fi
if [[ ! -f "$EXCLUDE_MANIFEST" ]]; then
  echo "Exclude manifest not found: $EXCLUDE_MANIFEST" >&2
  exit 1
fi

nohup "$PYTHON" waymo/data_prep/prepare_waymo_vector_ooi_centered.py \
  --stats_csv "$STATS_CSV" \
  --output_dir "$OUT" \
  --exclude_manifest "$EXCLUDE_MANIFEST" \
  --exclude_level "${EXCLUDE_LEVEL:-sample}" \
  --num_focus_samples "${NUM_FOCUS_SAMPLES:-50000}" \
  --val_fraction "${VAL_FRACTION:-0.1}" \
  --seed "${SEED:-1}" \
  --num_agents "${NUM_AGENTS:-32}" \
  --agent_distance_threshold "${AGENT_DISTANCE_THRESHOLD:-80}" \
  --map_distance_threshold "${MAP_DISTANCE_THRESHOLD:-100}" \
  --max_map_polylines "${MAX_MAP_POLYLINES:-256}" \
  --max_points_per_polyline "${MAX_POINTS_PER_POLYLINE:-20}" \
  --log_every "${LOG_EVERY:-1000}" \
  > "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PIDFILE"
echo "Started incremental OOI-centered Waymo vector preparation."
echo "PID: $pid"
echo "Base data: $BASE_DATA_DIR"
echo "Exclude manifest: $EXCLUDE_MANIFEST"
echo "Output: $OUT"
echo "Log: $LOG"
echo "PID file: $PIDFILE"
