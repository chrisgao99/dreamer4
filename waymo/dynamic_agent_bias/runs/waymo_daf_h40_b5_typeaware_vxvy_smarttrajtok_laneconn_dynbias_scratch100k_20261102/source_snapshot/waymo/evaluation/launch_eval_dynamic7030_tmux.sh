#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
PYTHON=/p/yufeng/.conda/envs/dreamer4/bin/python
CUDA_DEVICE="${CUDA_DEVICE:?Set CUDA_DEVICE 0..3}"
case "$CUDA_DEVICE" in
  0) ITEMS=(h40_b40 h15_b15);;
  1) ITEMS=(h40_b15);;
  2) ITEMS=(h40_b5);;
  3) ITEMS=(h15_b5);;
  *) exit 1;;
esac
SESSION="eval_dyn7030_cuda${CUDA_DEVICE}_20260918"
OUT_ROOT="$ROOT/waymo/eval_results/world_model/dynamic7030_h15_90k_h40_95k_ctx11_h80_k32_scenes512_20260918"
if [[ ${RUN_INSIDE_TMUX:-0} != 1 ]]; then
  if tmux has-session -t "=$SESSION" 2>/dev/null; then echo 'Session exists'; exit 1; fi
  for ITEM in "${ITEMS[@]}"; do [[ ! -e "$OUT_ROOT/$ITEM" ]] || { echo "Output exists: $ITEM"; exit 1; }; done
  printf -v command '%q ' env RUN_INSIDE_TMUX=1 CUDA_DEVICE="$CUDA_DEVICE" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION" -c "$ROOT"
  tmux set-window-option -t "$SESSION:0" remain-on-exit on
  tmux respawn-pane -k -t "$SESSION:0.0" "$command"
  echo "Started $SESSION: ${ITEMS[*]}"
  exit 0
fi
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
for ITEM in "${ITEMS[@]}"; do
  HORIZON="${ITEM%%_b*}"; HORIZON="${HORIZON#h}"
  COMMITMENT="${ITEM##*_b}"
  RUN="waymo_direct_action_flow_v1_h${HORIZON}_b5_generateall_dyn70static30_v05_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5"
  if [[ "$HORIZON" == 15 ]]; then RUN="${RUN}_rerun_cuda3_20260917"; STEP=000090000; else STEP=000095000; fi
  CKPT="$ROOT/waymo/checkpoints/$RUN/step_$STEP.pt"
  [[ -f "$CKPT" ]] || { echo "Missing $CKPT"; exit 1; }
  OUT="$OUT_ROOT/$ITEM"
  LOG="$ROOT/waymo/logs/evaluation/dyn7030_${ITEM}_20260918.log"
  mkdir -p "$OUT" "$(dirname "$LOG")"
  (
    echo "Started $(date -Is) CUDA=$CUDA_DEVICE plan=$HORIZON execute=$COMMITMENT checkpoint=$CKPT"
    "$PYTHON" waymo/evaluation/eval_waymo_direct_action_flow_multisample.py \
      --checkpoint "$CKPT" --val_data_dir "$ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
      --output_json "$OUT/result.json" --rollout_ade_csv "$OUT/rollout_ade.csv" \
      --scene_metrics_csv "$OUT/scene_metrics.csv" --details_npz "$OUT/details.npz" \
      --device cuda --weights ema --focus_mode generate_all --eval_batch_size 4 --eval_max_batches 128 \
      --num_workers 4 --num_rollouts 32 --rollout_steps 80 --commitment_steps "$COMMITMENT" --solver_steps 8 \
      --seed 12346 --log_every 1 --physical_metrics \
      --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977
    "$PYTHON" waymo/evaluation/summarize_ade_by_motion.py "$OUT"
    flock "$OUT_ROOT/.summary.lock" "$PYTHON" waymo/evaluation/summarize_dynamic7030_eval.py "$OUT_ROOT"
    echo "Finished $(date -Is)"
  ) 2>&1 | tee "$LOG"
done
