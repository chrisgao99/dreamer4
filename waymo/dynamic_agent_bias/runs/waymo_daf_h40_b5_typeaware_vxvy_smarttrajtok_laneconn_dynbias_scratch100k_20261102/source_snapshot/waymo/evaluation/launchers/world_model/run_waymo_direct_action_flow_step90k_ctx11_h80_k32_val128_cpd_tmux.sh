#!/usr/bin/env bash
# Evaluate the DirectActionFlow 90k EMA checkpoint on 128 batches.  Each scene
# gets 32 complete 80-step receding-horizon rollouts.  The job is launched in a
# detached tmux session and records every scene/rollout ADE plus Flow-ERD CPD.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_direct_action_flow_multisample.py"
VAL_DATA="${VAL_DATA:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val}"
EVAL_CKPT="${EVAL_CKPT:-$REPO_ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_h15_phys5_yaw075_huber_bounded_tmax090_lr5e5_from62500_v2/step_000090000.pt}"

RUN_NAME="${RUN_NAME:-waymo_direct_action_flow_step90k_ctx11_h80_k32_val128_cpd}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/evaluation}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUT_DIR/result.json}"
ROLLOUT_ADE_CSV="${ROLLOUT_ADE_CSV:-$OUT_DIR/rollout_ade.csv}"
SCENE_METRICS_CSV="${SCENE_METRICS_CSV:-$OUT_DIR/scene_metrics.csv}"
DETAILS_NPZ="${DETAILS_NPZ:-$OUT_DIR/details.npz}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_NAME.log}"

SESSION_NAME="${SESSION_NAME:-wm_daf_90k_h80_k32_cpd}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-128}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SOLVER_STEPS="${SOLVER_STEPS:-8}"
EVAL_SEED="${EVAL_SEED:-12346}"
COMMITMENT_STEPS="${COMMITMENT_STEPS:-5}"

# sigma_c = sqrt(E_train[||p_t-p_0||^2]), estimated on the matching OOI50k
# training data.  Order follows Waymo IDs: vehicle, pedestrian, cyclist.
CPD_VEHICLE_SCALE="${CPD_VEHICLE_SCALE:-25.423524547767016}"
CPD_PEDESTRIAN_SCALE="${CPD_PEDESTRIAN_SCALE:-4.925798640017075}"
CPD_CYCLIST_SCALE="${CPD_CYCLIST_SCALE:-18.77286026475977}"

for required_file in "$PYTHON" "$EVAL_SCRIPT" "$EVAL_CKPT"; do
  [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 \
    REPO_ROOT="$REPO_ROOT" \
    PYTHON="$PYTHON" \
    VAL_DATA="$VAL_DATA" \
    EVAL_CKPT="$EVAL_CKPT" \
    RUN_NAME="$RUN_NAME" \
    OUT_DIR="$OUT_DIR" \
    LOG_DIR="$LOG_DIR" \
    OUTPUT_JSON="$OUTPUT_JSON" \
    ROLLOUT_ADE_CSV="$ROLLOUT_ADE_CSV" \
    SCENE_METRICS_CSV="$SCENE_METRICS_CSV" \
    DETAILS_NPZ="$DETAILS_NPZ" \
    LOG_FILE="$LOG_FILE" \
    SESSION_NAME="$SESSION_NAME" \
    CUDA_DEVICE="$CUDA_DEVICE" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    EVAL_MAX_BATCHES="$EVAL_MAX_BATCHES" \
    NUM_ROLLOUTS="$NUM_ROLLOUTS" \
    NUM_WORKERS="$NUM_WORKERS" \
    SOLVER_STEPS="$SOLVER_STEPS" \
    EVAL_SEED="$EVAL_SEED" \
    COMMITMENT_STEPS="$COMMITMENT_STEPS" \
    CPD_VEHICLE_SCALE="$CPD_VEHICLE_SCALE" \
    CPD_PEDESTRIAN_SCALE="$CPD_PEDESTRIAN_SCALE" \
    CPD_CYCLIST_SCALE="$CPD_CYCLIST_SCALE" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "GPU: $CUDA_DEVICE"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Summary: $OUTPUT_JSON"
  echo "Every rollout ADE: $ROLLOUT_ADE_CSV"
  exit 0
fi

mkdir -p "$OUT_DIR" "$LOG_DIR"
for output_file in "$OUTPUT_JSON" "$ROLLOUT_ADE_CSV" "$SCENE_METRICS_CSV" "$DETAILS_NPZ"; do
  [[ ! -e "$output_file" ]] || { echo "Output exists; refusing to overwrite: $output_file" >&2; exit 1; }
done

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) DirectActionFlow K32 ADE/minADE/CPD evaluation start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "checkpoint=$EVAL_CKPT weights=ema"
  echo "protocol=context11 native_horizon15 trained_commitment5 rollout_commitment${COMMITMENT_STEPS} rollout80 solver_steps=$SOLVER_STEPS"
  echo "evaluation=batches${EVAL_MAX_BATCHES}_batch_size${EVAL_BATCH_SIZE}_rollouts${NUM_ROLLOUTS}_max_scenes$((EVAL_MAX_BATCHES * EVAL_BATCH_SIZE))"
  echo "scope=generated_nonfocus_agents focus_future=conditioned_and_excluded"
  echo "cpd_scales_vehicle_pedestrian_cyclist=$CPD_VEHICLE_SCALE,$CPD_PEDESTRIAN_SCALE,$CPD_CYCLIST_SCALE"

  "$PYTHON" "$EVAL_SCRIPT" \
    --checkpoint "$EVAL_CKPT" \
    --val_data_dir "$VAL_DATA" \
    --output_json "$OUTPUT_JSON" \
    --rollout_ade_csv "$ROLLOUT_ADE_CSV" \
    --scene_metrics_csv "$SCENE_METRICS_CSV" \
    --details_npz "$DETAILS_NPZ" \
    --device cuda \
    --weights ema \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --eval_max_batches "$EVAL_MAX_BATCHES" \
    --num_workers "$NUM_WORKERS" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --rollout_steps 80 \
    --commitment_steps "$COMMITMENT_STEPS" \
    --solver_steps "$SOLVER_STEPS" \
    --seed "$EVAL_SEED" \
    --log_every 1 \
    --cpd_type_scales "$CPD_VEHICLE_SCALE" "$CPD_PEDESTRIAN_SCALE" "$CPD_CYCLIST_SCALE"

  echo "===== $(date) DirectActionFlow K32 ADE/minADE/CPD evaluation complete ====="
} 2>&1 | tee -a "$LOG_FILE"
