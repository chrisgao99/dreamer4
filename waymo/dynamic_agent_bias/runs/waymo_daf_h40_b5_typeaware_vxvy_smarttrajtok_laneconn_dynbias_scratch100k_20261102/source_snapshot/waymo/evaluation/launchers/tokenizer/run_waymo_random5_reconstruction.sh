#!/usr/bin/env bash
set -euo pipefail

cd /scratch/baz7dy/tri30/dreamer4

DATA_DIR="waymo/data/waymo_vector_dataset_ooi_centered_50k/val"
LIST_FILE="/tmp/waymo_random5.txt"
OUT_91="waymo/evaluation/reports/reconstruction_compare_random5_chunked91"
OUT_32="waymo/evaluation/reports/reconstruction_compare_random5_decode32"

CKPT_REPEAT_MAP="waymo/checkpoints/ooi50k_lat16_d256_ep200_2gpu_lossfix/latest.pt"
CKPT_STATIC_MAP="waymo/checkpoints/ooi50k_lat16_d256_ep200_2a100_staticmap_v2_lossfix/latest.pt"

find "$DATA_DIR" -name '*.npz' | shuf -n 5 > "$LIST_FILE"

echo "Selected inputs:"
cat "$LIST_FILE"
echo

echo "Rendering 91-step reconstructions to $OUT_91"
while IFS= read -r npz; do
  python waymo/evaluation/visualize_vector_tokenizer_reconstruction.py \
    --npz "$npz" \
    --checkpoint "$CKPT_REPEAT_MAP" "$CKPT_STATIC_MAP" \
    --label repeat_map staticmap_v2 \
    --chunked_full_trajectory \
    --split_panels \
    --output_dir "$OUT_91" \
    --panel_size 900
done < "$LIST_FILE"

echo
echo "Rendering 32-step reconstructions to $OUT_32"
while IFS= read -r npz; do
  python waymo/evaluation/visualize_vector_tokenizer_reconstruction.py \
    --npz "$npz" \
    --checkpoint "$CKPT_REPEAT_MAP" "$CKPT_STATIC_MAP" \
    --label repeat_map staticmap_v2 \
    --time_window 32 \
    --split_panels \
    --output_dir "$OUT_32" \
    --panel_size 900
done < "$LIST_FILE"

echo
echo "Done."
echo "91-step output: /scratch/baz7dy/tri30/dreamer4/$OUT_91"
echo "32-step output: /scratch/baz7dy/tri30/dreamer4/$OUT_32"
