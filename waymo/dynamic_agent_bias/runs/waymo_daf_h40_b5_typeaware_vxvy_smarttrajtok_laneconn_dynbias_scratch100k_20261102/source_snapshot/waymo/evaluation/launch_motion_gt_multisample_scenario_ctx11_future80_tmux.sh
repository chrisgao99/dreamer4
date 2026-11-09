#!/usr/bin/env bash
# Render eight stochastic rollouts using 11 displayed GT context frames and
# 80 predicted frames. With max_rollout_window=11, the first prediction token
# attends to context frames 2..11; context frame 1 is display-only.

set -euo pipefail

STAGE="${STAGE:?Set STAGE to h30 or h90}"
CUDA_DEVICE="${CUDA_DEVICE:?Set CUDA_DEVICE}"
REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
SCRIPT="$REPO_ROOT/waymo/evaluation/visualize_waymo_world_model_multisample_scenario.py"
TOKENIZER="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
SCENARIO="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val/77046993fcd9f0bc_focus_278_src3.npz"
BASE="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage"
OUT_ROOT="$REPO_ROOT/waymo/eval_results/world_model/motion_gt_h30_h90_scenario_77046993fcd9f0bc_n8_20260824"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"

case "$STAGE" in
  h30)
    CKPT="$BASE/waymo_wm_stage1best_mon8_fullmotion_physproxy_ctx1_h30_d1_chunk32s30_30k_from3k/step_00021000.pt"
    OUT="$OUT_ROOT/h30_step21000_ctx11_future80"
    MODEL_LABEL="N=8 + Motion GT - H30 step21k"
    ;;
  h90)
    CKPT="$BASE/waymo_wm_stage1best_mon8_fullmotion_physproxy_h30best30k_ctx1_h90_d1_chunk32s30_30k/step_00027000.pt"
    OUT="$OUT_ROOT/h90_step27000_ctx11_future80"
    MODEL_LABEL="N=8 + Motion GT - H90 step27k"
    ;;
  *)
    echo "Unknown STAGE=$STAGE" >&2
    exit 2
    ;;
esac

SESSION_NAME="${SESSION_NAME:-wm_vis_motion_gt_${STAGE}_ctx11_f80_cuda${CUDA_DEVICE}}"
PIPELINE_LOG="$LOG_DIR/${SESSION_NAME}.log"
MPL_DIR="/tmp/matplotlib-waymo-world-model-${STAGE}-ctx11-f80"
mkdir -p "$OUT" "$LOG_DIR" "$MPL_DIR"

for path in "$PYTHON" "$SCRIPT" "$TOKENIZER" "$SCENARIO" "$CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 STAGE="$STAGE" CUDA_DEVICE="$CUDA_DEVICE" \
    SESSION_NAME="$SESSION_NAME" REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $PIPELINE_LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$MPL_DIR"

{
  echo "===== $(date) $MODEL_LABEL ctx11 future80 visualization start ====="
  "$PYTHON" "$SCRIPT" \
    --data_dir "$SCENARIO" --tokenizer_ckpt "$TOKENIZER" --device cuda \
    --eval_ckpt "$CKPT" --scenario_npz "$SCENARIO" --output_dir "$OUT" \
    --model_label "$MODEL_LABEL" \
    --eval_seq_len 91 --eval_ctx 11 --eval_horizon 80 \
    --eval_num_rollouts 8 --eval_multisample_seed 20260824 \
    --eval_schedule shortcut --eval_d 1.0 \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 --max_rollout_window 11 \
    --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
    --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
    --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4 \
    --use_ego_actions --ego_action_source focus --ego_action_normalization raw \
    --no-ego_action_clamp --agent_far_weight 0.25 --agent_near_radius_m 50 \
    --agent_distance_source focus --num_workers 0
  echo "===== $(date) $MODEL_LABEL ctx11 future80 visualization complete ====="
} 2>&1 | tee -a "$PIPELINE_LOG"
