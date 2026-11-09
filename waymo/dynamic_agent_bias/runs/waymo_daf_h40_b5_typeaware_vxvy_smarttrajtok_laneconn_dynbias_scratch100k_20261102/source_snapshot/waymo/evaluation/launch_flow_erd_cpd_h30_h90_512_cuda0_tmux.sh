#!/usr/bin/env bash
# Run paper-formula Flow-ERD CPD for the selected h30/h90 checkpoints.
# The two jobs run sequentially on one GPU and share an identical scene/noise set.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SESSION_NAME="${SESSION_NAME:-wm_flow_erd_cpd_h30_h90_cuda0}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
CKPT_ROOT="$REPO_ROOT/waymo/checkpoints/n8_motion_h30_21k_h90_27k"
OUT="$REPO_ROOT/waymo/eval_results/world_model/flow_erd_cpd_h30_h90_ctx11_h80_n8_512scenes_20260903"
LOG="$REPO_ROOT/waymo/logs/wm/wm_flow_erd_cpd_h30_h90_cuda0.log"
SCALE_JSON="$REPO_ROOT/waymo/eval_results/world_model/flow_erd_cpd_scale_stats_ooi50k_train_ctx11_h80.json"

# sigma_c = sqrt(E_train[||p_t - p_0||^2]), in Waymo type-id order
# 1=vehicle, 2=pedestrian, 3=cyclist.
CPD_VEHICLE_SCALE="25.423524547767016"
CPD_PEDESTRIAN_SCALE="4.925798640017075"
CPD_CYCLIST_SCALE="18.77286026475977"

mkdir -p "$OUT" "$(dirname "$LOG")"

for path in "$PYTHON" "$EVAL_SCRIPT" "$TOKENIZER" "$SCALE_JSON" \
  "$CKPT_ROOT/h30_step21k/step_00021000.pt" \
  "$CKPT_ROOT/h90_step27k/step_00027000.pt"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    CUDA_DEVICE="$CUDA_DEVICE" SESSION_NAME="$SESSION_NAME" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Log: $LOG"
  echo "Output: $OUT"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

{
  echo "===== $(date) Flow-ERD CPD pipeline start ====="
  echo "cuda=$CUDA_VISIBLE_DEVICES protocol=ctx11_future80_K8_batches128_batchsize4_scenes512"
  echo "cpd_scales_vehicle_pedestrian_cyclist=$CPD_VEHICLE_SCALE,$CPD_PEDESTRIAN_SCALE,$CPD_CYCLIST_SCALE"
  echo "scale_definition=sigma_c=sqrt(E_train[||p_t-p_0||_2^2])"
  echo "scale_stats=$SCALE_JSON"

  for stage in h30 h90; do
    case "$stage" in
      h30)
        label="motion_gt_h30_step21000"
        checkpoint="$CKPT_ROOT/h30_step21k/step_00021000.pt"
        ;;
      h90)
        label="motion_gt_h90_step27000"
        checkpoint="$CKPT_ROOT/h90_step27k/step_00027000.pt"
        ;;
    esac

    output_json="$OUT/${label}_ctx11_h80_n8_512scenes_flow_erd_cpd.json"
    component_npz="$OUT/${label}_ctx11_h80_n8_512scenes_flow_erd_cpd_components.npz"
    echo "===== $(date) $label start ====="
    echo "checkpoint=$checkpoint"
    if [[ -f "$output_json" && -f "$component_npz" ]]; then
      echo "Already evaluated: $output_json"
    else
      "$PYTHON" "$EVAL_SCRIPT" \
        --data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
        --val_data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
        --tokenizer_ckpt "$TOKENIZER" --eval_ckpt "$checkpoint" --output_json "$output_json" \
        --device cuda --eval_seq_len 91 --eval_ctx 11 --horizons 80 \
        --eval_batch_size 4 --eval_max_batches 128 --eval_subset_size 512 \
        --eval_subset_seed 20260824 \
        --num_workers 4 --eval_num_rollouts 8 --eval_multisample_seed 20260824 \
        --eval_flow_erd_cpd \
        --eval_cpd_type_scales "$CPD_VEHICLE_SCALE" "$CPD_PEDESTRIAN_SCALE" "$CPD_CYCLIST_SCALE" \
        --eval_cpd_exclude_focus --eval_cpd_components_output "$component_npz" \
        --no-eval_multisample_physical \
        --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
        --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0 \
        --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
        --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
        --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4 \
        --use_ego_actions --ego_action_source focus --ego_action_normalization raw \
        --no-ego_action_clamp --agent_far_weight 1.0 --agent_near_radius_m 50 \
        --agent_distance_source focus
    fi
    echo "===== $(date) $label complete ====="
  done
  echo "===== $(date) Flow-ERD CPD pipeline complete ====="
} 2>&1 | tee -a "$LOG"
