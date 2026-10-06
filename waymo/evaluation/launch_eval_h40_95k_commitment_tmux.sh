#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
PYTHON=/p/yufeng/.conda/envs/dreamer4/bin/python
COMMITMENT="${COMMITMENT:?Set COMMITMENT to 40, 15 or 5}"
CUDA_DEVICE="${CUDA_DEVICE:?Set CUDA_DEVICE}"
case "$COMMITMENT:$CUDA_DEVICE" in 40:0|15:1|5:2) ;; *) echo 'Expected 40:0, 15:1 or 5:2'; exit 1;; esac
SESSION="eval_h40_95k_b${COMMITMENT}_cuda${CUDA_DEVICE}_20260917"
OUT_ROOT="$REPO_ROOT/waymo/eval_results/world_model/h40_step95k_ctx11_h80_k32_scenes512_physical_20260917"
OUT="$OUT_ROOT/b$COMMITMENT"
CKPT="$REPO_ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_h40_b5_generateall_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5/step_000095000.pt"
LOG="$REPO_ROOT/waymo/logs/evaluation/h40_95k_b${COMMITMENT}_physical_20260917.log"
[[ -f "$CKPT" && -x "$PYTHON" ]] || exit 1
if [[ ${RUN_INSIDE_TMUX:-0} != 1 ]]; then
  [[ ! -e "$OUT" && ! -e "$LOG" ]] || { echo 'Output already exists'; exit 1; }
  if tmux has-session -t "=$SESSION" 2>/dev/null; then echo 'Session exists'; exit 1; fi
  printf -v command '%q ' env RUN_INSIDE_TMUX=1 COMMITMENT="$COMMITMENT" CUDA_DEVICE="$CUDA_DEVICE" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION" -c "$REPO_ROOT"
  tmux set-window-option -t "$SESSION:0" remain-on-exit on
  tmux respawn-pane -k -t "$SESSION:0.0" "$command"
  echo "Started $SESSION; output=$OUT; log=$LOG"
  exit 0
fi
cd "$REPO_ROOT"
mkdir -p "$OUT" "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG") 2>&1
echo "Started $(date -Is): CUDA=$CUDA_DEVICE context=11 plan=40 execute=$COMMITMENT total=80 scenes=512 K=32"
"$PYTHON" waymo/evaluation/eval_waymo_direct_action_flow_multisample.py \
  --checkpoint "$CKPT" --val_data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
  --output_json "$OUT/result.json" --rollout_ade_csv "$OUT/rollout_ade.csv" \
  --scene_metrics_csv "$OUT/scene_metrics.csv" --details_npz "$OUT/details.npz" \
  --device cuda --weights ema --focus_mode generate_all --eval_batch_size 4 --eval_max_batches 128 \
  --num_workers 4 --num_rollouts 32 --rollout_steps 80 --commitment_steps "$COMMITMENT" --solver_steps 8 \
  --seed 12346 --log_every 1 --physical_metrics \
  --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977
flock "$OUT_ROOT/.summary.lock" "$PYTHON" waymo/evaluation/summarize_h40_commitment.py "$OUT_ROOT"
echo "Finished $(date -Is)"
