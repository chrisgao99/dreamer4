#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
STATS_CSV="${STATS_CSV:-$REPO_ROOT/waymo/reports/ooi_raw_stats_train/waymo_ooi_scenario_stats.csv}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/vector_tokenizer_ooi_centered_50k}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

cd "$REPO_ROOT"

echo "=== Stage 1: prepare/resume OOI-centered data ==="
"$PYTHON" waymo/prepare_waymo_vector_ooi_centered.py \
  --stats_csv "$STATS_CSV" \
  --output_dir "$DATA_DIR" \
  --num_focus_samples "${NUM_FOCUS_SAMPLES:-50000}" \
  --val_fraction "${VAL_FRACTION:-0.1}" \
  --seed "${SEED:-0}" \
  --num_agents "${NUM_AGENTS:-32}" \
  --agent_distance_threshold "${AGENT_DISTANCE_THRESHOLD:-80}" \
  --map_distance_threshold "${MAP_DISTANCE_THRESHOLD:-100}" \
  --max_map_polylines "${MAX_MAP_POLYLINES:-256}" \
  --max_points_per_polyline "${MAX_POINTS_PER_POLYLINE:-20}" \
  --log_every "${PREP_LOG_EVERY:-1000}"

train_count=$(find "$DATA_DIR/train" -name '*.npz' | wc -l)
val_count=$(find "$DATA_DIR/val" -name '*.npz' | wc -l)
echo "Prepared train NPZ: $train_count"
echo "Prepared val NPZ: $val_count"
if [[ "$train_count" -eq 0 || "$val_count" -eq 0 ]]; then
  echo "ERROR: train/val data missing after preparation." >&2
  exit 1
fi

echo "=== Stage 2: train tokenizer ==="
train_args=(
  waymo/train_waymo_vector_tokenizer.py
  --data_dir "$DATA_DIR/train"
  --val_data_dir "$DATA_DIR/val"
  --ckpt_dir "$CKPT_DIR"
  --batch_size "${BATCH_SIZE:-8}"
  --num_workers "${NUM_WORKERS:-4}"
  --time_window "${TIME_WINDOW:-32}"
  --epochs "${EPOCHS:-50}"
  --d_model "${D_MODEL:-128}"
  --depth "${DEPTH:-3}"
  --decoder_depth "${DECODER_DEPTH:-3}"
  --n_latents "${N_LATENTS:-8}"
  --d_bottleneck "${D_BOTTLENECK:-32}"
  --log_every "${TRAIN_LOG_EVERY:-20}"
  --eval_every "${EVAL_EVERY:-500}"
  --save_every "${SAVE_EVERY:-500}"
)

if [[ -f "$CKPT_DIR/latest.pt" && "${RESUME_IF_EXISTS:-1}" == "1" ]]; then
  echo "Resuming from $CKPT_DIR/latest.pt"
  train_args+=(--resume "$CKPT_DIR/latest.pt")
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" "${train_args[@]}"
else
  "$PYTHON" "${train_args[@]}"
fi
