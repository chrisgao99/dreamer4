#!/usr/bin/env bash
set -euo pipefail

cd /scratch/baz7dy/tri30/dreamer4

TIME_WINDOW="${TIME_WINDOW:-32}"
LIST_FILE="waymo/evaluation/waymo_fulltraj_random10.txt"
OUT_DIR="${OUT_DIR:-waymo/evaluation/reports/reconstruction_fulltraj_random10_repeatmap_chunk${TIME_WINDOW}}"
CKPT="waymo/checkpoints/ooi50k_lat16_d256_ep200_2gpu_lossfix/latest.pt"

echo "Checkpoint: /scratch/baz7dy/tri30/dreamer4/$CKPT"
echo "Input list: /scratch/baz7dy/tri30/dreamer4/$LIST_FILE"
echo "Output: /scratch/baz7dy/tri30/dreamer4/$OUT_DIR"
echo "Mode: chunked_full_trajectory, chunk_window=$TIME_WINDOW, final short chunk decoded without padding"
echo

while IFS= read -r npz; do
  [[ -z "$npz" ]] && continue
  python waymo/evaluation/visualize_vector_tokenizer_reconstruction.py \
    --npz "$npz" \
    --checkpoint "$CKPT" \
    --label "repeatmap_chunk${TIME_WINDOW}" \
    --chunked_full_trajectory \
    --time_window "$TIME_WINDOW" \
    --split_panels \
    --output_dir "$OUT_DIR" \
    --panel_size 900
done < "$LIST_FILE"

echo
echo "Done."
echo "Output: /scratch/baz7dy/tri30/dreamer4/$OUT_DIR"
