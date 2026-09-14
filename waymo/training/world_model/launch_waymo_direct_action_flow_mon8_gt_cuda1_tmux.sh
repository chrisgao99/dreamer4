#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd)}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
SESSION_NAME="${SESSION_NAME:-wm_daf75k_mon8_gt_cuda1}"
RUN_NAME="${RUN_NAME:-waymo_direct_action_flow_generateall75k_mon8_gt_h15_b5_10k_20260911}"
MAX_STEPS="${MAX_STEPS:-10000}"
CHECKPOINT="$REPO_ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_h15_b5_generateall_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
OUT="$REPO_ROOT/waymo/checkpoints/$RUN_NAME"
LOG="$REPO_ROOT/waymo/logs/wm/$RUN_NAME.log"
TRAIN="$REPO_ROOT/waymo/training/world_model/train_waymo_direct_action_flow_mon.py"
for f in "$CHECKPOINT" "$TRAIN" "$PYTHON"; do [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }; done
[[ -d "$DATA_ROOT/train" && -d "$DATA_ROOT/val" ]] || { echo 'Missing data'; exit 1; }
if [[ "${RUN_INSIDE_TMUX:-0}" != 1 ]]; then
    if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then echo "Session exists: $SESSION_NAME"; exit 1; fi
    printf -v command '%q ' env RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" CUDA_DEVICE="$CUDA_DEVICE" SESSION_NAME="$SESSION_NAME" RUN_NAME="$RUN_NAME" MAX_STEPS="$MAX_STEPS" bash "$SCRIPT_PATH"
    tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
    tmux set-window-option -t "$SESSION_NAME:0" remain-on-exit on
    tmux respawn-pane -k -t "$SESSION_NAME:0.0" "$command"
    echo "Started $SESSION_NAME on CUDA $CUDA_DEVICE"
    echo "Log: $LOG"
    exit 0
fi
cd "$REPO_ROOT"
mkdir -p "$(dirname "$LOG")"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG") 2>&1
resume=()
if [[ -f "$OUT/latest.pt" ]]; then resume=(--resume "$OUT/latest.pt"); fi
echo "===== $(date -Is) mon8_gt training CUDA=$CUDA_DEVICE target=$MAX_STEPS ====="
"$PYTHON" "$TRAIN" --checkpoint "$CHECKPOINT" --train_data "$DATA_ROOT/train" --val_data "$DATA_ROOT/val" --output_dir "$OUT" --max_steps "$MAX_STEPS"  "${resume[@]}"
for b in 1 5 15; do
    result="$REPO_ROOT/waymo/eval_results/world_model/${RUN_NAME}_final_h80_b${b}_k32_val128_cpd"
    [[ -f "$result/result.json" ]] && continue
    mkdir -p "$result"
    "$PYTHON" "$REPO_ROOT/waymo/evaluation/eval_waymo_direct_action_flow_multisample.py" \
      --checkpoint "$OUT/final.pt" --val_data_dir "$DATA_ROOT/val" \
      --output_json "$result/result.json" --rollout_ade_csv "$result/rollout_ade.csv" \
      --scene_metrics_csv "$result/scene_metrics.csv" --details_npz "$result/details.npz" \
      --device cuda --weights ema --focus_mode generate_all --eval_batch_size 4 --eval_max_batches 128 \
      --num_workers 4 --num_rollouts 32 --rollout_steps 80 --commitment_steps "$b" --solver_steps 8 \
      --seed 12346 --log_every 1 --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977 \
      2>&1 | tee -a "${LOG%.log}_eval_b${b}.log"
done
echo "===== $(date -Is) mon8_gt training/evaluation complete ====="
