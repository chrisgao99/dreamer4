#!/usr/bin/env bash
# 4-GPU BC pretraining on the whole-scene cache in a detached tmux session.
# Re-running with the same RUN_NAME resumes from latest.pt.
#   bash waymo/scene_cache/launch_train_tmux.sh
set -euo pipefail

REPO=/p/yufeng/tri30/dreamer4
PYTHON_BIN="${PYTHON_BIN:-/p/yufeng/.conda/envs/dreamer4/bin}"
GPUS="${GPUS:-0,1,2,3}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
RUN_NAME="${RUN_NAME:-waymo_daf_scene_full_knn48r80_sdcfact_loss48_30ep_20261022}"
OUT="$REPO/waymo/checkpoints/$RUN_NAME"
SESSION="${SESSION:-scene_train}"
EXTRA_ARGS="${EXTRA_ARGS:---epochs 30 --wandb}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session $SESSION exists: tmux attach -t $SESSION" >&2; exit 1
fi
RESUME=""
if [ -f "$OUT/latest.pt" ]; then RESUME="--resume $OUT/latest.pt"; echo "Resuming from $OUT/latest.pt"; fi
mkdir -p "$OUT"
LOG="$OUT/train_$(date +%Y%m%d_%H%M%S).log"
tmux new-session -d -s "$SESSION" bash -c "
  cd $REPO
  export CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  { echo \"start \$(date) host \$(hostname) gpus $GPUS\"
    $PYTHON_BIN/torchrun --standalone --nproc_per_node $NPROC waymo/scene_cache/train_scene_daf.py \
      --output_dir $OUT $RESUME $EXTRA_ARGS
    echo \"exit \$? at \$(date)\"; } > $LOG 2>&1
"
echo "Started tmux session $SESSION on GPUs $GPUS"
echo "Log: $LOG"
echo "Checkpoints: $OUT (latest.pt every 5k steps, best.pt by val minade_eval_agent)"
