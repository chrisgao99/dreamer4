#!/usr/bin/env bash
# Evaluate the H30-MinADE and H90-MinADE candidates from one experiment under
# one common protocol: context=1, future=90, N=8, 128 batches x batch size 4.

set -euo pipefail

VARIANT="${VARIANT:?Set VARIANT to mon8_phys, mon8_motion_phys, n1_motion, or stage1}"
CUDA_DEVICE="${CUDA_DEVICE:?Set CUDA_DEVICE}"
REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
BASE="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage"
OUT="$REPO_ROOT/waymo/eval_results/world_model/minade_h30_h90_ctx1_h90_n8_128batches_20260824"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"

case "$VARIANT" in
  mon8_phys)
    LABELS=(mon8_phys_h30_step3000 mon8_phys_h90_step0_from_h30step18000)
    CKPTS=(
      "$BASE/waymo_wm_stage1best_mon8_physproxy_ctx1_h30_d1_chunk32s30_3k/final_step_00003000.pt"
      "$BASE/waymo_wm_stage1best_mon8_physproxy_ctx1_h30_d1_chunk32s30_30k_from3k/step_00018000.pt"
    )
    ;;
  mon8_motion_phys)
    LABELS=(mon8_motion_phys_h30_step21000 mon8_motion_phys_h90_step27000)
    CKPTS=(
      "$BASE/waymo_wm_stage1best_mon8_fullmotion_physproxy_ctx1_h30_d1_chunk32s30_30k_from3k/step_00021000.pt"
      "$BASE/waymo_wm_stage1best_mon8_fullmotion_physproxy_h30best30k_ctx1_h90_d1_chunk32s30_30k/step_00027000.pt"
    )
    ;;
  n1_motion)
    LABELS=(n1_motion_h30_step15000 n1_motion_h90_step30000)
    CKPTS=(
      "$BASE/waymo_wm_stage1best_n1_fullmotion_only_ctx1_h30_d1_chunk32s30_30k_from3k/step_00015000.pt"
      "$BASE/waymo_wm_stage1best_n1_fullmotion_only_h30best30k_ctx1_h90_d1_chunk32s30_30k/step_00030000.pt"
    )
    ;;
  stage1)
    LABELS=(stage1_reference)
    CKPTS=(
      "$BASE/waymo_wm_v1_egoact_focus_raw_noclamp_win11_randstart_b8_self05_norecon_time1_mapx1_1m/best.pt"
    )
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

SESSION_NAME="${SESSION_NAME:-wm_eval_${VARIANT}_ctx1_h90_n8_128b_cuda${CUDA_DEVICE}}"
PIPELINE_LOG="$LOG_DIR/${SESSION_NAME}.log"
mkdir -p "$OUT" "$LOG_DIR"

for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER" "${CKPTS[@]}"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 VARIANT="$VARIANT" CUDA_DEVICE="$CUDA_DEVICE" \
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) $VARIANT unified H90 evaluation start ====="
  echo "cuda=$CUDA_VISIBLE_DEVICES protocol=ctx1_future90_N8_batches128_batchsize4_scenes512"
  for index in "${!CKPTS[@]}"; do
    label="${LABELS[$index]}"
    ckpt="${CKPTS[$index]}"
    output="$OUT/${label}_ctx1_h90_n8_128batches.json"
    echo "===== $(date) evaluate $label ====="
    echo "checkpoint=$ckpt"
    if [[ -f "$output" ]]; then
      echo "Already evaluated: $output"
      continue
    fi
    "$PYTHON" "$EVAL_SCRIPT" \
      --data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
      --val_data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
      --tokenizer_ckpt "$TOKENIZER" --eval_ckpt "$ckpt" --output_json "$output" \
      --device cuda --eval_seq_len 91 --eval_ctx 1 --horizons 90 \
      --eval_batch_size 4 --eval_max_batches 128 --eval_subset_size 512 \
      --eval_subset_seed 20260824 \
      --num_workers 4 --eval_num_rollouts 8 --eval_multisample_seed 20260824 \
      --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
      --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0 \
      --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
      --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
      --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4 \
      --use_ego_actions --ego_action_source focus --ego_action_normalization raw \
      --no-ego_action_clamp --agent_far_weight 0.25 --agent_near_radius_m 50 \
      --agent_distance_source focus
  done
  echo "===== $(date) $VARIANT unified H90 evaluation complete ====="
} 2>&1 | tee -a "$PIPELINE_LOG"
