#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
OUT="${OUT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"
STATS_CSV="${STATS_CSV:-$REPO_ROOT/waymo/evaluation/reports/ooi_raw_stats_train/waymo_ooi_scenario_stats.csv}"
LOG="${LOG:-$REPO_ROOT/waymo/prepare_waymo_vector_ooi_centered_50k.log}"
PIDFILE="${PIDFILE:-$REPO_ROOT/waymo/prepare_waymo_vector_ooi_centered_50k.pid}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$LOG")" "$OUT"

nohup "$PYTHON" waymo/data_prep/prepare_waymo_vector_ooi_centered.py \
  --stats_csv "$STATS_CSV" \
  --output_dir "$OUT" \
  --num_focus_samples "${NUM_FOCUS_SAMPLES:-50000}" \
  --val_fraction "${VAL_FRACTION:-0.1}" \
  --seed "${SEED:-0}" \
  --num_agents "${NUM_AGENTS:-32}" \
  --agent_distance_threshold "${AGENT_DISTANCE_THRESHOLD:-80}" \
  --map_distance_threshold "${MAP_DISTANCE_THRESHOLD:-100}" \
  --max_map_polylines "${MAX_MAP_POLYLINES:-256}" \
  --max_points_per_polyline "${MAX_POINTS_PER_POLYLINE:-20}" \
  --log_every "${LOG_EVERY:-1000}" \
  > "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PIDFILE"
echo "Started OOI-centered Waymo vector preparation."
echo "PID: $pid"
echo "Output: $OUT"
echo "Log: $LOG"
echo "PID file: $PIDFILE"
