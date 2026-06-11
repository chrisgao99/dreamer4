#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
RAW_DIR="${RAW_DIR:-/p/liverobotics/waymo_open_dataset_motion/tf_example/training}"
OUT="${OUT:-/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_5k}"
N_FILES="${N_FILES:-50}"
MAX_RECORDS="${MAX_RECORDS:-100}"
NUM_AGENTS="${NUM_AGENTS:-16}"
AGENT_DISTANCE_THRESHOLD="${AGENT_DISTANCE_THRESHOLD:-80}"
MAP_DISTANCE_THRESHOLD="${MAP_DISTANCE_THRESHOLD:-100}"
MAX_MAP_POLYLINES="${MAX_MAP_POLYLINES:-256}"
MAX_POINTS_PER_POLYLINE="${MAX_POINTS_PER_POLYLINE:-20}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"

cd "$REPO_ROOT"
mkdir -p "$OUT"

echo "Writing filtered Waymo vector NPZ files to: $OUT"
echo "Using first $N_FILES TFRecords, max $MAX_RECORDS scenarios per file."

count=0
while IFS= read -r tfrecord; do
  count=$((count + 1))
  echo "[$count/$N_FILES] $tfrecord"
  "$PYTHON" waymo/waymo_vector_filter.py "$tfrecord" \
    --output_dir "$OUT" \
    --max_records "$MAX_RECORDS" \
    --num_agents "$NUM_AGENTS" \
    --agent_distance_threshold "$AGENT_DISTANCE_THRESHOLD" \
    --map_distance_threshold "$MAP_DISTANCE_THRESHOLD" \
    --max_map_polylines "$MAX_MAP_POLYLINES" \
    --max_points_per_polyline "$MAX_POINTS_PER_POLYLINE" \
    --history_only_selection
done < <(find "$RAW_DIR" -maxdepth 1 -type f | sort | head -n "$N_FILES")

echo "Done."
echo -n "NPZ count: "
find "$OUT" -name '*.npz' | wc -l
