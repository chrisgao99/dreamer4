#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WAYMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$WAYMO_ROOT/.." && pwd)"

cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
TIME_WINDOW="${TIME_WINDOW:-32}"
DATA_DIR="${DATA_DIR:-waymo/data/waymo_vector_dataset_ooi_centered_50k/val}"
LIST_FILE="${LIST_FILE:-waymo/evaluation/waymo_fulltraj_random10_threeckpts.txt}"
OUT_DIR="${OUT_DIR:-waymo/evaluation/reports/reconstruction_fulltraj_random10_staticmap_v2_chunk${TIME_WINDOW}_threeckpts}"
PANEL_SIZE="${PANEL_SIZE:-900}"

CKPT_FDELOSS="waymo/checkpoints/ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_fdeloss_focus_randstart_noamp/latest.pt"
CKPT_KINEMATIC="waymo/checkpoints/ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_kinematic_focus_randstart_noamp/latest.pt"
CKPT_TRAJLOSS="waymo/checkpoints/ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_trajloss_randstart_noamp/latest.pt"

mkdir -p "$(dirname "$LIST_FILE")" "$OUT_DIR"

find "$DATA_DIR" -name '*.npz' | sort | shuf -n 10 > "$LIST_FILE"

echo "Repo:       $REPO_ROOT"
echo "Data dir:   $REPO_ROOT/$DATA_DIR"
echo "List file:  $REPO_ROOT/$LIST_FILE"
echo "Output dir: $REPO_ROOT/$OUT_DIR"
echo "Mode:       chunked_full_trajectory, chunk_window=$TIME_WINDOW"
echo
echo "Selected inputs:"
cat "$LIST_FILE"
echo
echo "Checkpoints:"
echo "  fdeloss_focus:   $REPO_ROOT/$CKPT_FDELOSS"
echo "  kinematic_focus: $REPO_ROOT/$CKPT_KINEMATIC"
echo "  trajloss:        $REPO_ROOT/$CKPT_TRAJLOSS"
echo

while IFS= read -r npz; do
  [[ -z "$npz" ]] && continue
  "$PYTHON" waymo/evaluation/visualize_vector_tokenizer_reconstruction.py \
    --npz "$npz" \
    --checkpoint \
      "$CKPT_FDELOSS" \
      "$CKPT_KINEMATIC" \
      "$CKPT_TRAJLOSS" \
    --label fdeloss_focus kinematic_focus trajloss \
    --chunked_full_trajectory \
    --time_window "$TIME_WINDOW" \
    --split_panels \
    --output_dir "$OUT_DIR" \
    --panel_size "$PANEL_SIZE"
done < "$LIST_FILE"

echo
echo "Done."
echo "Input list: $REPO_ROOT/$LIST_FILE"
echo "Output:     $REPO_ROOT/$OUT_DIR"
