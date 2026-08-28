#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
WAYMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$WAYMO_ROOT/.." && pwd)"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
HARD_BEST="${HARD_BEST:-$WAYMO_ROOT/checkpoints/interaction_contrastive_hard_relneg_dupfiltered_v1/best.pt}"
CACHE_DIR="${CACHE_DIR:-$WAYMO_ROOT/cache/interaction_full_pairs_50k_v2_contrastive_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$WAYMO_ROOT/checkpoints/interaction_contrastive_hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3}"
LOG_DIR="${LOG_DIR:-$WAYMO_ROOT/logs/interaction_contrastive_learning}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3.log}"
SESSION_NAME="${SESSION_NAME:-interaction_hybrid_soft_v2_cuda3}"

worker() {
  cd "$REPO_ROOT"
  mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
  export PYTHONPATH="$WAYMO_ROOT"
  export PYTHONNOUSERSITE=1
  export MPLCONFIGDIR="/tmp/matplotlib-interaction-hybrid-soft-v2"

  [[ -f "$HARD_BEST" ]] || { echo "Missing hard checkpoint: $HARD_BEST"; exit 1; }
  [[ -f "$CACHE_DIR/summary.json" ]] || { echo "Missing cache: $CACHE_DIR"; exit 1; }

  {
    echo "===== hybrid soft v2 start: $(date) ====="
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    echo "HARD_BEST=$HARD_BEST"
    echo "CACHE_DIR=$CACHE_DIR"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u \
      "$SCRIPT_DIR/train_interaction_contrastive.py" \
      --mode hybrid \
      --tokenizer_ckpt "$HARD_BEST" \
      --head_init_ckpt "$HARD_BEST" \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$OUTPUT_DIR" \
      --device cuda \
      --seed "${SEED:-17}" \
      --batch_size "${BATCH_SIZE:-2}" \
      --num_workers "${NUM_WORKERS:-4}" \
      --stage_a_steps "${STAGE_A_STEPS:-5000}" \
      --stage_b_steps "${STAGE_B_STEPS:-40000}" \
      --encoder_unfreeze_blocks "${ENCODER_UNFREEZE_BLOCKS:-1}" \
      --head_lr "${HEAD_LR:-5e-5}" \
      --encoder_lr "${ENCODER_LR:-5e-6}" \
      --hybrid_separation_weight "${HYBRID_SEPARATION_WEIGHT:-0.005}" \
      --hybrid_rank_weight "${HYBRID_RANK_WEIGHT:-0.001}" \
      --contrastive_ramp_steps "${CONTRASTIVE_RAMP_STEPS:-4000}" \
      --temperature "${TEMPERATURE:-0.07}" \
      --rank_temperature "${RANK_TEMPERATURE:-0.10}" \
      --soft_sigma_floor "${SOFT_SIGMA_FLOOR:-0.02}" \
      --gradient_probe_every "${GRADIENT_PROBE_EVERY:-500}" \
      --amp_dtype "${AMP_DTYPE:-bfloat16}" \
      --val_anchors "${VAL_ANCHORS:-512}" \
      --eval_batches 0 \
      --eval_at_start \
      --eval_every "${EVAL_EVERY:-2000}" \
      --save_every "${SAVE_EVERY:-2000}" \
      --log_every "${LOG_EVERY:-20}"
    "$PYTHON" "$SCRIPT_DIR/summarize_hybrid_soft_metrics.py" \
      --metrics "$OUTPUT_DIR/metrics.jsonl" \
      --output_json "$OUTPUT_DIR/training_summary.json" \
      --output_html "$OUTPUT_DIR/training_report.html"
    echo "===== hybrid soft v2 end: $(date) ====="
  } 2>&1 | tee -a "$LOG_FILE"
}

if [[ "${1:-}" == "--worker" ]]; then
  worker
  exit 0
fi

mkdir -p "$LOG_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" \
  "env PYTHON='$PYTHON' CUDA_DEVICE='$CUDA_DEVICE' HARD_BEST='$HARD_BEST' CACHE_DIR='$CACHE_DIR' OUTPUT_DIR='$OUTPUT_DIR' LOG_DIR='$LOG_DIR' LOG_FILE='$LOG_FILE' SESSION_NAME='$SESSION_NAME' bash '$SCRIPT_PATH' --worker"

echo "Submitted hybrid-soft-v2 to CUDA $CUDA_DEVICE"
echo "tmux:  $SESSION_NAME"
echo "attach: tmux attach -t $SESSION_NAME"
echo "log:    tail -f $LOG_FILE"
echo "output: $OUTPUT_DIR"
