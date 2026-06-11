#!/usr/bin/env bash
set -euo pipefail

cd /scratch/baz7dy/tri30/dreamer4

DATA_DIR="waymo/data/waymo_vector_dataset_ooi_centered_50k/val"
LIST_FILE="/tmp/waymo_fulltraj_random10.txt"
OUT_DIR="waymo/evaluation/reports/reconstruction_fulltraj_random10"

CKPT="waymo/checkpoints/ooi50k_lat16_d256_ep200_2a100_staticmap_v2_fulltraj_trajloss/latest.pt"

find "$DATA_DIR" -name '*.npz' | shuf -n 10 > "$LIST_FILE"

echo "Checkpoint: /scratch/baz7dy/tri30/dreamer4/$CKPT"
echo "Selected inputs:"
cat "$LIST_FILE"
echo

echo "Rendering 10 full-trajectory reconstructions to /scratch/baz7dy/tri30/dreamer4/$OUT_DIR"
while IFS= read -r npz; do
  python waymo/evaluation/visualize_vector_tokenizer_reconstruction.py \
    --npz "$npz" \
    --checkpoint "$CKPT" \
    --label fulltraj_recon \
    --time_window -1 \
    --split_panels \
    --output_dir "$OUT_DIR" \
    --panel_size 900
done < "$LIST_FILE"

echo
echo "Done."
echo "Input list: $LIST_FILE"
echo "Output: /scratch/baz7dy/tri30/dreamer4/$OUT_DIR"
