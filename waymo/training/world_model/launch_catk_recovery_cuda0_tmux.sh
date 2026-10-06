#!/usr/bin/env bash
set -euo pipefail
SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT")/../../.." && pwd)"
SESSION=catk_recovery_h40_95k_k8_b5_h80_cuda0_20260922
RUN=waymo_daf_h40_95k_catk_sample8_recovery_flow_unroll80_b5_10k_20260922
OUT="$ROOT/waymo/checkpoints/$RUN"
LOG="$ROOT/waymo/logs/wm/$RUN.log"
CKPT="$ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_h40_b5_generateall_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5/step_000095000.pt"
PYTHON=/p/yufeng/.conda/envs/dreamer4/bin/python
[[ -f "$CKPT" && -x "$PYTHON" ]] || exit 1
if [[ ${RUN_INSIDE_TMUX:-0} != 1 ]]; then
  [[ ! -e "$OUT" && ! -e "$LOG" ]] || { echo 'Output exists; refusing overwrite'; exit 1; }
  if tmux has-session -t "=$SESSION" 2>/dev/null; then echo 'Session exists'; exit 1; fi
  printf -v command '%q ' env RUN_INSIDE_TMUX=1 bash "$SCRIPT"
  tmux new-session -d -s "$SESSION" -c "$ROOT"
  tmux set-window-option -t "$SESSION:0" remain-on-exit on
  tmux respawn-pane -k -t "$SESSION:0.0" "$command"
  echo "Started $SESSION; log=$LOG; checkpoints=$OUT"
  exit 0
fi
cd "$ROOT"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo "Started $(date -Is): CUDA=0 sampled-K8 joint segment selection; detached context; recovery flow; plan40 execute5 rollout80"
"$PYTHON" waymo/training/world_model/train_waymo_direct_action_flow_catk_recovery.py \
  --checkpoint "$CKPT" --train_data "$ROOT/data/waymo_vector_dataset_ooi_centered_50k/train" \
  --val_data "$ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" --output_dir "$OUT" \
  --seed 20260911 --batch_size 2 --num_workers 4 --num_candidates 8 --solver_steps 8 \
  --max_steps 10000 --lr 1e-6 --ema_decay .999 --eval_every 500 \
  --eval_batches 8 --eval_batch_size 4 --eval_rollouts 4 \
  --save_every 100 --snapshot_every 1000 --log_every 10
echo "Training finished $(date -Is)"
