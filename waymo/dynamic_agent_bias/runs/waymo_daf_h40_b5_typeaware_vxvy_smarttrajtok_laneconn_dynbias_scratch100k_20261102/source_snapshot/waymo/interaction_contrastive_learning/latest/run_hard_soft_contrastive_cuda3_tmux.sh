#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
WAYMO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$WAYMO_ROOT/.." && pwd)"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$WAYMO_ROOT/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt}"
CACHE_DIR="${CACHE_DIR:-$WAYMO_ROOT/cache/interaction_full_pairs_50k_v2_contrastive_v1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$WAYMO_ROOT/checkpoints}"
HARD_OUTPUT_DIR="${HARD_OUTPUT_DIR:-$CHECKPOINT_ROOT/interaction_contrastive_hard_relneg_dupfiltered_v1}"
SOFT_OUTPUT_DIR="${SOFT_OUTPUT_DIR:-$CHECKPOINT_ROOT/interaction_contrastive_soft_relneg_dupfiltered_v1}"
LOG_DIR="${LOG_DIR:-$WAYMO_ROOT/logs/interaction_contrastive_learning}"
HARD_LOG="${HARD_LOG:-$LOG_DIR/hard_relneg_dupfiltered_v1_cuda3.log}"
SOFT_LOG="${SOFT_LOG:-$LOG_DIR/soft_relneg_dupfiltered_v1_cuda3.log}"
SESSION_NAME="${SESSION_NAME:-interaction_contrastive_hard_soft_cuda3}"

run_one() {
  local mode="$1"
  local output_dir="$2"
  local log_file="$3"
  mkdir -p "$output_dir" "$LOG_DIR"
  {
    echo "===== ${mode} contrastive start: $(date) ====="
    echo "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
    echo "TOKENIZER_CKPT=$TOKENIZER_CKPT"
    echo "CACHE_DIR=$CACHE_DIR"
    echo "OUTPUT_DIR=$output_dir"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" -u "$SCRIPT_DIR/train_interaction_contrastive.py" \
      --mode "$mode" \
      --tokenizer_ckpt "$TOKENIZER_CKPT" \
      --cache_dir "$CACHE_DIR" \
      --output_dir "$output_dir" \
      --device cuda \
      --seed "${SEED:-0}" \
      --batch_size "${BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-4}" \
      --stage_a_steps "${STAGE_A_STEPS:-5000}" \
      --stage_b_steps "${STAGE_B_STEPS:-20000}" \
      --encoder_unfreeze_blocks "${ENCODER_UNFREEZE_BLOCKS:-1}" \
      --head_lr "${HEAD_LR:-1e-4}" \
      --encoder_lr "${ENCODER_LR:-1e-5}" \
      --contrastive_weight "${CONTRASTIVE_WEIGHT:-0.1}" \
      --contrastive_ramp_steps "${CONTRASTIVE_RAMP_STEPS:-2000}" \
      --temperature "${TEMPERATURE:-0.07}" \
      --amp_dtype "${AMP_DTYPE:-bfloat16}" \
      --eval_every "${EVAL_EVERY:-500}" \
      --eval_batches "${EVAL_BATCHES:-64}" \
      --save_every "${SAVE_EVERY:-1000}" \
      --log_every "${LOG_EVERY:-20}"
    echo "===== ${mode} contrastive end: $(date) ====="
  } 2>&1 | tee -a "$log_file"
}

worker() {
  cd "$REPO_ROOT"
  export PYTHONPATH="$WAYMO_ROOT"
  export PYTHONNOUSERSITE=1
  export MPLCONFIGDIR="/tmp/matplotlib-interaction-contrastive"

  [[ -f "$TOKENIZER_CKPT" ]] || { echo "Missing checkpoint: $TOKENIZER_CKPT"; exit 1; }
  [[ -f "$CACHE_DIR/summary.json" ]] || { echo "Missing cache summary: $CACHE_DIR/summary.json"; exit 1; }
  [[ -f "$CACHE_DIR/train_contrastive_training.npz" ]] || { echo "Missing train cache"; exit 1; }
  [[ -f "$CACHE_DIR/val_contrastive_training.npz" ]] || { echo "Missing val cache"; exit 1; }

  # A single A100 cannot run both large tokenizers efficiently at once.  Both
  # jobs are submitted now; soft starts automatically when hard completes.
  run_one hard "$HARD_OUTPUT_DIR" "$HARD_LOG"
  run_one soft "$SOFT_OUTPUT_DIR" "$SOFT_LOG"
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
  "env PYTHON='$PYTHON' CUDA_DEVICE='$CUDA_DEVICE' TOKENIZER_CKPT='$TOKENIZER_CKPT' CACHE_DIR='$CACHE_DIR' CHECKPOINT_ROOT='$CHECKPOINT_ROOT' HARD_OUTPUT_DIR='$HARD_OUTPUT_DIR' SOFT_OUTPUT_DIR='$SOFT_OUTPUT_DIR' LOG_DIR='$LOG_DIR' HARD_LOG='$HARD_LOG' SOFT_LOG='$SOFT_LOG' SESSION_NAME='$SESSION_NAME' bash '$SCRIPT_PATH' --worker"

echo "Submitted hard then soft contrastive experiments to CUDA $CUDA_DEVICE"
echo "tmux:  $SESSION_NAME"
echo "attach: tmux attach -t $SESSION_NAME"
echo "hard:   tail -f $HARD_LOG"
echo "soft:   tail -f $SOFT_LOG"
