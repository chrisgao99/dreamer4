#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
LOG="${LOG:-$REPO_ROOT/waymo/prepare_waymo_vector_5k.log}"
PIDFILE="${PIDFILE:-$REPO_ROOT/waymo/prepare_waymo_vector_5k.pid}"

cd "$REPO_ROOT"
nohup bash waymo/data_prep/prepare_waymo_vector_dataset.sh > "$LOG" 2>&1 &
pid=$!
echo "$pid" > "$PIDFILE"

echo "Started Waymo vector filtering in background."
echo "PID: $pid"
echo "Log: $LOG"
echo "PID file: $PIDFILE"
