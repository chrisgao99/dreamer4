#!/usr/bin/env bash
# Detached 50-view throughput and correctness test for the local oracle-focus
# WOSAC protocol.  Run this file directly; it creates and returns from a tmux
# session, while the evaluation continues after the terminal disconnects.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
WOSAC_PYTHON="${WOSAC_PYTHON:-/p/liverobotics/.conda/envs/scene_edit_py310/bin/python}"
GENERATE_SCRIPT="$REPO_ROOT/waymo/evaluation/generate_wosac_oracle_focus_rollouts.py"
SCORE_SCRIPT="$REPO_ROOT/waymo/evaluation/score_wosac_oracle_focus_rollouts.py"

TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
WORLD_MODEL_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage/waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k/step_00050000.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
SCENARIO_MANIFEST="$REPO_ROOT/waymo/cache/wosac_internal_val_scenarios/eligible_ooi_pair_manifest.csv"

NUM_VIEWS="${NUM_VIEWS:-50}"
TOTAL_VIEWS="${TOTAL_VIEWS:-5000}"
NUM_ROLLOUTS=32
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WOSAC_WORKERS="${WOSAC_WORKERS:-0}"
WOSAC_DEVICE="${WOSAC_DEVICE:-auto}"
EVAL_D="${EVAL_D:-0.25}"
SESSION_NAME="${SESSION_NAME:-wosac_oracle_focus_val50_a6000}"
RUN_ID="${RUN_ID:-wosac_oracle_focus_val50_k4_rb${ROLLOUT_BATCH_SIZE}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/wosac/$RUN_ID}"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"
GENERATION_JSON="$OUT_DIR/generation_summary.json"
WOSAC_JSON="$OUT_DIR/wosac_summary.json"
ETA_JSON="$OUT_DIR/benchmark_eta.json"

for path in "$PYTHON" "$WOSAC_PYTHON" "$GENERATE_SCRIPT" "$SCORE_SCRIPT" \
  "$TOKENIZER_CKPT" "$WORLD_MODEL_CKPT" "$SCENARIO_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux is not available" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" WOSAC_PYTHON="$WOSAC_PYTHON" \
    NUM_VIEWS="$NUM_VIEWS" TOTAL_VIEWS="$TOTAL_VIEWS" ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" \
    CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" WOSAC_WORKERS="$WOSAC_WORKERS" \
    WOSAC_DEVICE="$WOSAC_DEVICE" EVAL_D="$EVAL_D" \
    SESSION_NAME="$SESSION_NAME" RUN_ID="$RUN_ID" OUT_DIR="$OUT_DIR" LOG_FILE="$LOG_FILE" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started detached tmux session: $SESSION_NAME"
  echo "Attach:  tmux attach -t $SESSION_NAME"
  echo "Log:     tail -f $LOG_FILE"
  echo "Results: $OUT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-wosac-${USER:-user}}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

{
  RUN_START_EPOCH="$(date +%s)"
  echo "===== $(date) local WOSAC oracle-focus val${NUM_VIEWS} benchmark start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_DEVICE"
  echo "views=$NUM_VIEWS total_views_for_eta=$TOTAL_VIEWS rollouts_per_view=$NUM_ROLLOUTS rollout_batch_size=$ROLLOUT_BATCH_SIZE wosac_device=$WOSAC_DEVICE wosac_workers=$WOSAC_WORKERS"
  echo "protocol=original_context_indices_1_to_10 original_future_indices_11_to_90 H80 oracle_focus selected_current_valid_only"
  echo "schedule=shortcut eval_d=$EVAL_D expected_solver_steps_per_frame=$(awk -v d="$EVAL_D" 'BEGIN {printf "%.0f", 1/d}')"
  echo "checkpoint=$WORLD_MODEL_CKPT"
  echo "manifest=$SCENARIO_MANIFEST"
  echo "output_dir=$OUT_DIR"

  GENERATION_START_EPOCH="$(date +%s)"
  "$PYTHON" "$GENERATE_SCRIPT" \
    --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
    --manifest "$SCENARIO_MANIFEST" --max_views "$NUM_VIEWS" \
    --num_rollouts "$NUM_ROLLOUTS" --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
    --total_views_for_eta "$TOTAL_VIEWS" --output_dir "$OUT_DIR" --output_json "$GENERATION_JSON" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$WORLD_MODEL_CKPT" \
    --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
    --eval_seq_len 91 --eval_ctx 10 --eval_horizon 80 \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
    --max_rollout_window 11 --eval_schedule shortcut --eval_d "$EVAL_D" \
    --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
    --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
    --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
    --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute
  GENERATION_END_EPOCH="$(date +%s)"
  echo "generation_wall_seconds=$((GENERATION_END_EPOCH - GENERATION_START_EPOCH))"

  SCORE_START_EPOCH="$(date +%s)"
  "$WOSAC_PYTHON" "$SCORE_SCRIPT" \
    --rollout_manifest "$OUT_DIR/rollout_manifest.jsonl" \
    --output_dir "$OUT_DIR" --output_json "$WOSAC_JSON" \
    --max_views "$NUM_VIEWS" --total_views_for_eta "$TOTAL_VIEWS" \
    --num_workers "$WOSAC_WORKERS" --device "$WOSAC_DEVICE"
  SCORE_END_EPOCH="$(date +%s)"
  echo "wosac_scoring_wall_seconds=$((SCORE_END_EPOCH - SCORE_START_EPOCH))"

  "$PYTHON" -c 'import json, pathlib, sys; g=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[2])); total=float(g["projected_total_seconds"])+float(s["projected_total_seconds"]); out={"benchmark_views":int(g["selected_views"]),"target_views":int(g["total_views_for_eta"]),"generation_projected_hours":float(g["projected_total_hours"]),"wosac_scoring_projected_hours":float(s["projected_total_hours"]),"end_to_end_projected_hours":total/3600.0,"generation_seconds_per_view":float(g["seconds_per_view"]),"wosac_seconds_per_view":float(s["seconds_per_view"]),"rollout_batch_size":int(g["rollout_batch_size"]),"solver_steps_per_frame":int(g["solver_steps_per_frame"])}; pathlib.Path(sys.argv[3]).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print("benchmark_eta="+json.dumps(out,sort_keys=True))' "$GENERATION_JSON" "$WOSAC_JSON" "$ETA_JSON"

  RUN_END_EPOCH="$(date +%s)"
  echo "actual_val${NUM_VIEWS}_end_to_end_seconds=$((RUN_END_EPOCH - RUN_START_EPOCH))"
  echo "===== $(date) local WOSAC oracle-focus val${NUM_VIEWS} benchmark complete ====="
  echo "ETA summary: $ETA_JSON"
} 2>&1 | tee -a "$LOG_FILE"
