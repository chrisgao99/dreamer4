#!/usr/bin/env bash
# Resume from complete saved CSV batches. Provide-ego step100k: H15 plans, execute B=5 per replan, H80/K32/512 scenes.
# Standalone tmux launcher. Run from any directory; --dry-run checks configuration.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
COMMITMENT_STEPS=5
RUN_NAME="${RUN_NAME:-waymo_direct_action_flow_provideego_step100k_ctx11_h80_b5_k32_val128_cpd_resume}"
SESSION_NAME="${SESSION_NAME:-wm_daf_provideego_step100k_h80_b5_k32_cpd_resume}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
VAL_DATA="${VAL_DATA:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val}"
EVAL_CKPT="${EVAL_CKPT:-$REPO_ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_h15_b5_provideego_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5/final_step_000100000.pt}"
RESUME_FROM_DIR="${RESUME_FROM_DIR:-$REPO_ROOT/waymo/eval_results/world_model/waymo_direct_action_flow_provideego_step100k_ctx11_h80_b5_k32_val128_cpd}"
RESUME_FROM_DIR="$(realpath -m "$RESUME_FROM_DIR")"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/resume_waymo_direct_action_flow_multisample.py"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/world_model/$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$REPO_ROOT/waymo/logs/evaluation/$RUN_NAME.log}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Resolve overrides before tmux changes the working directory.
VAL_DATA="$(realpath -m "$VAL_DATA")"
EVAL_CKPT="$(realpath -m "$EVAL_CKPT")"
OUT_DIR="$(realpath -m "$OUT_DIR")"
LOG_FILE="$(realpath -m "$LOG_FILE")"

if [[ -z "${PYTHON:-}" ]]; then
  OWNER_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
  for candidate in \
    "${CONDA_PREFIX:-/nonexistent}/bin/python" \
    "$OWNER_ROOT/.conda/envs/dreamer4/bin/python" \
    "$HOME/.conda/envs/dreamer4/bin/python" \
    "$HOME/miniconda3/envs/dreamer4/bin/python" \
    "$HOME/anaconda3/envs/dreamer4/bin/python"; do
    if [[ -x "$candidate" ]]; then PYTHON="$candidate"; break; fi
  done
  PYTHON="${PYTHON:-$(command -v python3 || true)}"
else
  PYTHON="$(command -v "$PYTHON")"
fi
[[ -x "$PYTHON" ]] || { echo "Set PYTHON to the dreamer4 environment's Python." >&2; exit 1; }
PYTHON="$(realpath "$PYTHON")"
for required in "$EVAL_SCRIPT" "$EVAL_CKPT"; do
  [[ -f "$required" ]] || { echo "Missing file: $required" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation data: $VAL_DATA" >&2; exit 1; }
for output in "$OUT_DIR/result.json" "$OUT_DIR/rollout_ade.csv" \
              "$OUT_DIR/scene_metrics.csv" "$OUT_DIR/details_suffix.npz" "$LOG_FILE"; do
  [[ ! -e "$output" ]] || { echo "Output already exists: $output. Set RUN_NAME for a new run." >&2; exit 1; }
done

command=("$PYTHON" "$EVAL_SCRIPT" --resume_from_dir "$RESUME_FROM_DIR"
  --checkpoint "$EVAL_CKPT" --val_data_dir "$VAL_DATA"
  --output_json "$OUT_DIR/result.json" --rollout_ade_csv "$OUT_DIR/rollout_ade.csv"
  --scene_metrics_csv "$OUT_DIR/scene_metrics.csv" --details_npz "$OUT_DIR/details_suffix.npz"
  --device cuda --weights ema --focus_mode conditioned
  --eval_batch_size 4 --eval_max_batches 128 --num_workers 4 --num_rollouts 32
  --rollout_steps 80 --commitment_steps "$COMMITMENT_STEPS" --solver_steps 8
  --seed 12346 --log_every 1
  --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977)

"${command[@]}" --check-only

if [[ "${1:-}" == "--dry-run" && "$#" == 1 ]]; then
  echo "repo=$REPO_ROOT"
  echo "python=$PYTHON"
  echo "session=$SESSION_NAME gpu=$CUDA_DEVICE commitment=$COMMITMENT_STEPS"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
elif [[ "$#" != 0 ]]; then
  echo "Usage: bash $SCRIPT_PATH [--dry-run]" >&2
  exit 2
fi

if [[ "${RUN_INSIDE_TMUX:-0}" != 1 ]]; then
  command -v tmux >/dev/null || { echo "tmux is required." >&2; exit 1; }
  if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  printf -v tmux_command '%q ' env RUN_INSIDE_TMUX=1 \
    REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" CUDA_DEVICE="$CUDA_DEVICE" \
    RESUME_FROM_DIR="$RESUME_FROM_DIR" VAL_DATA="$VAL_DATA" EVAL_CKPT="$EVAL_CKPT" RUN_NAME="$RUN_NAME" \
    SESSION_NAME="$SESSION_NAME" OUT_DIR="$OUT_DIR" LOG_FILE="$LOG_FILE" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" bash "$SCRIPT_PATH"
  # Configure the new window before starting work so even an early failure is retained.
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
  tmux set-window-option -t "$SESSION_NAME:0" remain-on-exit on
  tmux respawn-pane -k -t "$SESSION_NAME:0.0" "$tmux_command"
  echo "Started: $SESSION_NAME (GPU $CUDA_DEVICE)"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Results: $OUT_DIR"
  exit 0
fi

mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 OMP_NUM_THREADS
{
  echo "===== $(date -Is) Provide-ego ADE/minADE/CPD evaluation start ====="
  echo "host=$(hostname) session=$SESSION_NAME physical_cuda=$CUDA_DEVICE"
  echo "checkpoint=$EVAL_CKPT weights=ema focus_future=recorded_condition generated=nonfocus_agents"
  echo "protocol=context11 native_horizon15 commitment${COMMITMENT_STEPS} rollout80 solver_steps8 seed12346"
  echo "evaluation=batches128 batch_size4 rollouts32 scenes512; ADE/CPD exclude conditioned focus"
  "${command[@]}"
  echo "===== $(date -Is) Provide-ego ADE/minADE/CPD evaluation complete ====="
} 2>&1 | tee "$LOG_FILE"
