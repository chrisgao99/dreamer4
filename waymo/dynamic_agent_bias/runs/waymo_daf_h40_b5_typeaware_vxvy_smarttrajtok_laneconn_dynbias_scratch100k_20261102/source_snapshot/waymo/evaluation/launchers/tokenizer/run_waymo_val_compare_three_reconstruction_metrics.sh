#!/usr/bin/env bash
set -euo pipefail

cd /scratch/baz7dy/tri30/dreamer4

PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
VAL_DIR="${VAL_DIR:-waymo/data/waymo_vector_dataset_ooi_centered_50k/val}"
OUT_DIR="${OUT_DIR:-waymo/evaluation/reports/val_reconstruction_compare}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHUNK_WINDOW="${CHUNK_WINDOW:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DEVICE="${DEVICE:-}"
STRATIFY_FOCUS_MOTION="${STRATIFY_FOCUS_MOTION:-0}"
STRAIGHT_HEADING_DEG="${STRAIGHT_HEADING_DEG:-5}"
CURVE_HEADING_DEG="${CURVE_HEADING_DEG:-15}"

cmd=(
  "$PYTHON" waymo/evaluation/compare_vector_tokenizer_val_metrics.py
  --data_dir "$VAL_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --chunk_window "$CHUNK_WINDOW"
  --max_samples "$MAX_SAMPLES"
  --summary_csv "$OUT_DIR/summary.csv"
  --summary_json "$OUT_DIR/summary.json"
  --model
  fulltraj_trajloss_full91
  waymo/checkpoints/ooi50k_lat16_d256_ep200_2a100_staticmap_v2_fulltraj_trajloss/latest.pt
  full
  --model
  staticmap_v2_chunk32
  waymo/checkpoints/ooi50k_lat16_d256_ep200_2a100_staticmap_v2_lossfix/latest.pt
  chunked
  --model
  repeatmap_chunk32
  waymo/checkpoints/ooi50k_lat16_d256_ep200_2gpu_lossfix/latest.pt
  chunked
)

if [[ -n "$DEVICE" ]]; then
  cmd+=(--device "$DEVICE")
fi

if [[ "$STRATIFY_FOCUS_MOTION" == "1" ]]; then
  cmd+=(
    --stratify_focus_motion
    --straight_heading_deg "$STRAIGHT_HEADING_DEG"
    --curve_heading_deg "$CURVE_HEADING_DEG"
  )
fi

echo "Validation data: /scratch/baz7dy/tri30/dreamer4/$VAL_DIR"
echo "Output: /scratch/baz7dy/tri30/dreamer4/$OUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Chunk window: $CHUNK_WINDOW"
echo "Max samples: $MAX_SAMPLES"
echo "Stratify focus motion: $STRATIFY_FOCUS_MOTION"
if [[ "$STRATIFY_FOCUS_MOTION" == "1" ]]; then
  echo "Straight heading deg: $STRAIGHT_HEADING_DEG"
  echo "Curve heading deg: $CURVE_HEADING_DEG"
fi
echo

"${cmd[@]}"
