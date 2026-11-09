#!/usr/bin/env bash
# Run the first 1,000 eligible validation focus-views with the stage-3 40k
# checkpoint. Both tokenizer encode and decode are restricted to <=32 frames.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
WOSAC_PYTHON="${WOSAC_PYTHON:-/p/liverobotics/.conda/envs/scene_edit_py310/bin/python}"
CUDNN_COMPAT_LIB="${CUDNN_COMPAT_LIB:-/p/liverobotics/.conda/envs/hptr/lib}"
GENERATE_SCRIPT="$REPO_ROOT/waymo/evaluation/generate_wosac_oracle_focus_rollouts.py"
SCORE_SCRIPT="$REPO_ROOT/waymo/evaluation/score_wosac_oracle_focus_rollouts.py"

TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
WORLD_MODEL_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage/waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k/step_00040000.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
SCENARIO_MANIFEST="$REPO_ROOT/waymo/cache/wosac_internal_val_scenarios/eligible_ooi_pair_manifest.csv"

NUM_VIEWS=1000
TOTAL_VIEWS=1000
NUM_ROLLOUTS=32
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
CUDA_DEVICE=0
NUM_WORKERS="${NUM_WORKERS:-4}"
WOSAC_WORKERS=1
WOSAC_DEVICE=gpu
RESUME_SCORING_ONLY="${RESUME_SCORING_ONLY:-0}"
WOSAC_SCORE_CHUNK_VIEWS="${WOSAC_SCORE_CHUNK_VIEWS:-25}"
WOSAC_GPU_STAGNANT_LIMIT="${WOSAC_GPU_STAGNANT_LIMIT:-2}"
EVAL_D="${EVAL_D:-0.25}"
TOKENIZER_CHUNK_WINDOW=32
TOKENIZER_CHUNK_STRIDE=30
SESSION_NAME="${SESSION_NAME:-wosac_oracle_focus_step40k_val1000_a100_cuda0}"
RUN_ID="${RUN_ID:-wosac_oracle_focus_step40k_val1000_k4_rb${ROLLOUT_BATCH_SIZE}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/waymo/eval_results/wosac/$RUN_ID}"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"
GENERATION_JSON="$OUT_DIR/generation_summary.json"
WOSAC_JSON="$OUT_DIR/wosac_summary.json"
RUN_STATUS_JSON="$OUT_DIR/run_status.json"

for path in "$PYTHON" "$WOSAC_PYTHON" "$GENERATE_SCRIPT" "$SCORE_SCRIPT" \
  "$TOKENIZER_CKPT" "$WORLD_MODEL_CKPT" "$SCENARIO_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
[[ -f "$CUDNN_COMPAT_LIB/libcudnn.so.8" ]] || { echo "Missing cuDNN 8 compatibility library: $CUDNN_COMPAT_LIB/libcudnn.so.8" >&2; exit 1; }
[[ "$TOKENIZER_CHUNK_WINDOW" -le 32 ]] || { echo "Tokenizer chunk exceeds 32 timesteps" >&2; exit 1; }
[[ "$(wc -l < "$SCENARIO_MANIFEST")" -ge 1001 ]] || { echo "Manifest has fewer than 1000 data rows" >&2; exit 1; }
command -v tmux >/dev/null || { echo "tmux is not available" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  if [[ "$RESUME_SCORING_ONLY" != "1" ]] && \
     [[ -e "$GENERATION_JSON" || -e "$WOSAC_JSON" || -e "$OUT_DIR/rollout_manifest.jsonl" ]]; then
    echo "Output directory already contains run artifacts: $OUT_DIR" >&2
    echo "Set a new RUN_ID instead of overwriting them." >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" WOSAC_PYTHON="$WOSAC_PYTHON" \
    CUDNN_COMPAT_LIB="$CUDNN_COMPAT_LIB" \
    ROLLOUT_BATCH_SIZE="$ROLLOUT_BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" EVAL_D="$EVAL_D" \
    RESUME_SCORING_ONLY="$RESUME_SCORING_ONLY" WOSAC_SCORE_CHUNK_VIEWS="$WOSAC_SCORE_CHUNK_VIEWS" \
    WOSAC_GPU_STAGNANT_LIMIT="$WOSAC_GPU_STAGNANT_LIMIT" \
    SESSION_NAME="$SESSION_NAME" RUN_ID="$RUN_ID" OUT_DIR="$OUT_DIR" LOG_FILE="$LOG_FILE" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started detached tmux session: $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Results: $OUT_DIR"
  exit 0
fi

cd "$REPO_ROOT"
source /etc/profile.d/modules.sh
module purge
module load cuda/11.8.0
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export LD_LIBRARY_PATH="$CUDNN_COMPAT_LIB:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-wosac-${USER:-user}}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-4}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

write_status() {
  "$PYTHON" -c 'import datetime,json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps({"status":sys.argv[2],"updated_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"session":sys.argv[3],"output_dir":sys.argv[4]},indent=2,sort_keys=True)+"\n")' "$RUN_STATUS_JSON" "$1" "$SESSION_NAME" "$OUT_DIR"
}

write_status running
trap 'rc=$?; if [[ $rc -ne 0 ]]; then write_status failed || true; fi' EXIT

{
  RUN_START_EPOCH="$(date +%s)"
  echo "===== $(date) local WOSAC oracle-focus step40k val1000 start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_DEVICE"
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(); p=torch.cuda.get_device_properties(0); print(f"pytorch_gpu={p.name} memory_gib={p.total_memory/1024**3:.1f}")'
  "$WOSAC_PYTHON" -c 'import tensorflow as tf; g=tf.config.list_physical_devices("GPU"); assert g, "TensorFlow cannot see CUDA 0"; print("tensorflow_gpus="+str(g))'
  echo "views=$NUM_VIEWS manifest_selection=first_1000_eligible_rows rollouts_per_view=$NUM_ROLLOUTS rollout_batch_size=$ROLLOUT_BATCH_SIZE"
  echo "protocol=original_context_indices_1_to_10 original_future_indices_11_to_90 H80 oracle_focus selected_current_valid_only"
  echo "tokenizer_encode_decode_chunk_window=$TOKENIZER_CHUNK_WINDOW stride=$TOKENIZER_CHUNK_STRIDE ranges=[0,32),[30,62),[59,91)"
  echo "schedule=shortcut eval_d=$EVAL_D expected_solver_steps_per_frame=$(awk -v d="$EVAL_D" 'BEGIN {printf "%.0f", 1/d}')"
  echo "checkpoint=$WORLD_MODEL_CKPT"
  echo "manifest=$SCENARIO_MANIFEST"
  echo "output_dir=$OUT_DIR"

  if [[ "$RESUME_SCORING_ONLY" == "1" ]]; then
    [[ -f "$GENERATION_JSON" ]] || { echo "Missing completed generation summary: $GENERATION_JSON" >&2; exit 1; }
    [[ -f "$OUT_DIR/rollout_manifest.jsonl" ]] || { echo "Missing rollout manifest" >&2; exit 1; }
    [[ "$(wc -l < "$OUT_DIR/rollout_manifest.jsonl")" -eq "$NUM_VIEWS" ]] || {
      echo "Rollout manifest does not contain exactly $NUM_VIEWS rows" >&2
      exit 1
    }
    echo "resume_scoring_only=true; reusing all $NUM_VIEWS generated rollout bundles"
  else
    "$PYTHON" "$GENERATE_SCRIPT" \
      --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
      --manifest "$SCENARIO_MANIFEST" --max_views "$NUM_VIEWS" \
      --num_rollouts "$NUM_ROLLOUTS" --rollout_batch_size "$ROLLOUT_BATCH_SIZE" \
      --total_views_for_eta "$TOTAL_VIEWS" --output_dir "$OUT_DIR" --output_json "$GENERATION_JSON" \
      --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$WORLD_MODEL_CKPT" \
      --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
      --eval_seq_len 91 --eval_ctx 10 --eval_horizon 80 \
      --tokenizer_chunk_window "$TOKENIZER_CHUNK_WINDOW" --tokenizer_chunk_stride "$TOKENIZER_CHUNK_STRIDE" \
      --max_rollout_window 11 --eval_schedule shortcut --eval_d "$EVAL_D" \
      --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
      --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
      --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
      --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute
  fi

  echo "===== $(date) rollout generation complete; starting WOSAC scoring ====="
  completed_views=0
  if [[ -f "$OUT_DIR/per_view_wosac_metrics.jsonl" ]]; then
    completed_views="$(wc -l < "$OUT_DIR/per_view_wosac_metrics.jsonl")"
  fi
  stagnant_gpu_failures=0
  echo "wosac_resume_start=$completed_views/$NUM_VIEWS gpu_chunk_views=$WOSAC_SCORE_CHUNK_VIEWS"
  while [[ "$completed_views" -lt "$NUM_VIEWS" ]]; do
    before_views="$completed_views"
    echo "===== $(date) WOSAC GPU scoring chunk start completed=$before_views/$NUM_VIEWS ====="
    if "$WOSAC_PYTHON" "$SCORE_SCRIPT" \
      --rollout_manifest "$OUT_DIR/rollout_manifest.jsonl" \
      --output_dir "$OUT_DIR" --output_json "$WOSAC_JSON" \
      --max_views "$NUM_VIEWS" --total_views_for_eta "$TOTAL_VIEWS" \
      --num_workers "$WOSAC_WORKERS" --device "$WOSAC_DEVICE" \
      --resume --max_new_views "$WOSAC_SCORE_CHUNK_VIEWS" --fsync_every 1; then
      chunk_rc=0
    else
      chunk_rc=$?
      echo "WOSAC GPU chunk exited nonzero: rc=$chunk_rc"
    fi
    completed_views="$(wc -l < "$OUT_DIR/per_view_wosac_metrics.jsonl")"
    if [[ "$completed_views" -gt "$before_views" ]]; then
      stagnant_gpu_failures=0
    else
      stagnant_gpu_failures=$((stagnant_gpu_failures + 1))
    fi
    echo "wosac_resume_after_gpu_chunk=$completed_views/$NUM_VIEWS stagnant_failures=$stagnant_gpu_failures"
    if [[ "$completed_views" -ge "$NUM_VIEWS" ]]; then
      break
    fi
    if [[ "$stagnant_gpu_failures" -ge "$WOSAC_GPU_STAGNANT_LIMIT" ]]; then
      echo "===== $(date) repeated GPU failure at view $completed_views; CPU fallback for one view ====="
      CUDA_VISIBLE_DEVICES="" "$WOSAC_PYTHON" "$SCORE_SCRIPT" \
        --rollout_manifest "$OUT_DIR/rollout_manifest.jsonl" \
        --output_dir "$OUT_DIR" --output_json "$WOSAC_JSON" \
        --max_views "$NUM_VIEWS" --total_views_for_eta "$TOTAL_VIEWS" \
        --num_workers 1 --device cpu --resume --max_new_views 1 --fsync_every 1
      after_cpu_views="$(wc -l < "$OUT_DIR/per_view_wosac_metrics.jsonl")"
      if [[ "$after_cpu_views" -le "$completed_views" ]]; then
        echo "CPU fallback did not complete a new view" >&2
        exit 1
      fi
      completed_views="$after_cpu_views"
      stagnant_gpu_failures=0
      echo "wosac_resume_after_cpu_fallback=$completed_views/$NUM_VIEWS"
    fi
  done
  [[ -f "$WOSAC_JSON" ]] || { echo "Missing final WOSAC summary after $completed_views views" >&2; exit 1; }

  RUN_END_EPOCH="$(date +%s)"
  echo "actual_val1000_end_to_end_seconds=$((RUN_END_EPOCH - RUN_START_EPOCH))"
  echo "===== $(date) local WOSAC oracle-focus step40k val1000 complete ====="
  echo "final_wosac_summary=$WOSAC_JSON"
} 2>&1 | tee -a "$LOG_FILE"

write_status completed
trap - EXIT
