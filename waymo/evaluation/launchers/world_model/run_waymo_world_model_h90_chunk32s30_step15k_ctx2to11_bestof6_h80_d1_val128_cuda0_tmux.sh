#!/usr/bin/env bash
# Per scene: use original frames 2..11 as context, sample frames 12..91 six
# times, select the rollout with minimum all-agent FDE at H80, then report
# H10/H30/H50/H80 metrics from prefixes of the selected rollout.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_best_of_n_rollout_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
VAL_DATA="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val"
EVAL_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k/step_00015000.pt"

BASE_RESULT_DIR="$REPO_ROOT/waymo/eval_results/world_model/motion_latent_three_stages_shared_h90_d1_val128_seed0"
SUBSET_MANIFEST="$BASE_RESULT_DIR/val_random128_seed0_manifest.json"
RUN_GROUP="world_model_h90_chunk32s30_step15k_ctx2to11_bestof6_h80_d1_val128_seed0"
OUT_DIR="$REPO_ROOT/waymo/eval_results/world_model/$RUN_GROUP"
LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
LOG_FILE="$LOG_DIR/$RUN_GROUP.log"
OUTPUT_JSON="$OUT_DIR/step15k_ctx2to11_bestof6_agentfde_h80_d1_h10_30_50_80_val128.json"

SESSION_NAME="${SESSION_NAME:-wm_step15k_ctx2to11_bestof6_h80_cuda0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$EVAL_CKPT" "$SUBSET_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -d "$VAL_DATA" ]] || { echo "Missing validation directory: $VAL_DATA" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    SESSION_NAME="$SESSION_NAME" CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $LOG_FILE"
  echo "Results: $OUTPUT_JSON"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) World Model step15k ctx2..11 best-of-6 H80 evaluation start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "checkpoint=$EVAL_CKPT"
  echo "protocol=same_recorded_val_samples=128 batch_size=1 original_context_frames=2..11 original_rollout_frames=12..91"
  echo "rollouts_per_scene=6 selection=minimum_agent_fde_mae_m_at_h80 shared_rollout_horizon=80 shortcut_d1=1"
  echo "tokenizer_chunk_window=32 tokenizer_chunk_stride=30 ranges_for_T90=[0,32),[30,62),[58,90)"
  echo "metric_horizons=10 30 50 80 subset_manifest=$SUBSET_MANIFEST"

  if [[ -f "$OUTPUT_JSON" ]]; then
    echo "Result already exists; refusing to overwrite: $OUTPUT_JSON" >&2
    exit 1
  fi

  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$VAL_DATA" --val_data_dir "$VAL_DATA" \
    --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$EVAL_CKPT" \
    --device cuda --seed 0 --num_workers "$NUM_WORKERS" \
    --eval_batch_size 1 --eval_max_batches 0 \
    --subset_manifest "$SUBSET_MANIFEST" --subset_size 128 --subset_seed 0 \
    --sequence_start 1 --eval_seq_len 90 --eval_ctx 10 --horizons "10 30 50 80" \
    --num_rollouts 6 --selection_horizon 80 \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
    --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0 \
    --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
    --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
    --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
    --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
    --output_json "$OUTPUT_JSON"

  echo "===== $(date) World Model step15k ctx2..11 best-of-6 H80 evaluation complete ====="
} 2>&1 | tee -a "$LOG_FILE"
