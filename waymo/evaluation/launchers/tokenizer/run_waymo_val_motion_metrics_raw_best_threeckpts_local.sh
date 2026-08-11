#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WAYMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$WAYMO_ROOT/.." && pwd)"

cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
VAL_DIR="${VAL_DIR:-waymo/data/waymo_vector_dataset_ooi_centered_50k/val}"
OUT_ROOT="${OUT_ROOT:-waymo/evaluation/reports/val_motion_metrics_raw_best_threeckpts}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHUNK_WINDOW="${CHUNK_WINDOW:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
DEVICE="${DEVICE:-cuda}"
STRAIGHT_HEADING_DEG="${STRAIGHT_HEADING_DEG:-5}"
CURVE_HEADING_DEG="${CURVE_HEADING_DEG:-15}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

LABELS=(
  "raw_gmm_kin_fde_s130500"
  "raw_kin_fde_s108500"
  "raw_kin_nofde_s70500"
)

CKPTS=(
  "/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_gmm_kinematic_fdeloss_focus_randstart_noamp/step_00130500.pt"
  "/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_fdeloss_focus_randstart_noamp/step_00108500.pt"
  "/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/step_00070500.pt"
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
    exit 1
  fi
}

require_file "$PYTHON"
require_dir "$VAL_DIR"
if ! find "$VAL_DIR" -maxdepth 1 -name '*.npz' -print -quit | grep -q .; then
  echo "No .npz files found in: $VAL_DIR" >&2
  exit 1
fi
for ckpt in "${CKPTS[@]}"; do
  require_file "$ckpt"
done

mkdir -p "$OUT_ROOT"

{
  echo "Selected checkpoints, run order:"
  for i in "${!LABELS[@]}"; do
    echo "$((i + 1)). ${LABELS[$i]}"
    echo "   ${CKPTS[$i]}"
  done
  echo
  echo "Run config:"
  echo "  val_dir: $REPO_ROOT/$VAL_DIR"
  echo "  out_root: $REPO_ROOT/$OUT_ROOT"
  echo "  batch_size: $BATCH_SIZE"
  echo "  num_workers: $NUM_WORKERS"
  echo "  chunk_window: $CHUNK_WINDOW"
  echo "  max_samples: $MAX_SAMPLES"
  echo "  device: $DEVICE"
  echo "  straight_heading_deg: $STRAIGHT_HEADING_DEG"
  echo "  curve_heading_deg: $CURVE_HEADING_DEG"
} | tee "$OUT_ROOT/run_config.txt"

for i in "${!LABELS[@]}"; do
  label="${LABELS[$i]}"
  ckpt="${CKPTS[$i]}"
  out_dir="$OUT_ROOT/$label"
  mkdir -p "$out_dir"

  if [[ "$SKIP_COMPLETED" == "1" && -s "$out_dir/summary.csv" && -s "$out_dir/summary.json" ]]; then
    echo
    echo "============================================================"
    echo "[$((i + 1))/${#LABELS[@]}] Skipping $label (summary already exists)"
    echo "output:     $REPO_ROOT/$out_dir"
    echo "============================================================"
    continue
  fi

  echo
  echo "============================================================"
  echo "[$((i + 1))/${#LABELS[@]}] Evaluating $label"
  echo "checkpoint: $ckpt"
  echo "output:     $REPO_ROOT/$out_dir"
  echo "============================================================"

  cmd=(
    "$PYTHON" waymo/evaluation/compare_vector_tokenizer_val_metrics.py
    --data_dir "$VAL_DIR"
    --model "$label" "$ckpt" chunked
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --chunk_window "$CHUNK_WINDOW"
    --max_samples "$MAX_SAMPLES"
    --device "$DEVICE"
    --progress_every "$PROGRESS_EVERY"
    --stratify_focus_motion
    --straight_heading_deg "$STRAIGHT_HEADING_DEG"
    --curve_heading_deg "$CURVE_HEADING_DEG"
    --summary_csv "$out_dir/summary.csv"
    --summary_json "$out_dir/summary.json"
  )

  "${cmd[@]}" 2>&1 | tee "$out_dir/run.log"
done

echo
echo "Done."
echo "Results:"
for label in "${LABELS[@]}"; do
  echo "  $REPO_ROOT/$OUT_ROOT/$label/summary.csv"
done
