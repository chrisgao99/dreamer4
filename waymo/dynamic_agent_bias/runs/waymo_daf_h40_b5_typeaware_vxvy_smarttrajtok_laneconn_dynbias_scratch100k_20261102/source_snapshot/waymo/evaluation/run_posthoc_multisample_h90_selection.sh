#!/usr/bin/env bash
# Scan one completed H90 run with fixed val32 / N=8 joint rollouts.

set -euo pipefail

VARIANT="${1:?usage: $0 mon8|n1 CUDA_DEVICE}"
CUDA_DEVICE="${2:?usage: $0 mon8|n1 CUDA_DEVICE}"
REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
BASE="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage"
OUT="$REPO_ROOT/waymo/eval_results/world_model/multisample_checkpoint_selection_20260813"

case "$VARIANT" in
  mon8)
    RUN="waymo_wm_stage1best_mon8_fullmotion_physproxy_h30step3k_ctx1_h90_d1_chunk32s30_3k"
    PREFIX="mon8_fullmotion_phys"
    ;;
  n1)
    RUN="waymo_wm_stage1best_n1_fullmotion_only_h30step3k_ctx1_h90_d1_chunk32s30_3k"
    PREFIX="n1_fullmotion"
    ;;
  *)
    echo "Unknown variant: $VARIANT (expected mon8 or n1)" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1

for step in 500 1000 1500 2000 2500; do
  step8="$(printf '%08d' "$step")"
  ckpt="$BASE/$RUN/step_${step8}.pt"
  output="$OUT/${PREFIX}_step${step}_h90_n8_val32.json"
  [[ -f "$ckpt" ]] || { echo "Missing checkpoint: $ckpt" >&2; exit 1; }
  [[ -f "$output" ]] && { echo "Already evaluated: $output"; continue; }
  "$PYTHON" "$EVAL_SCRIPT" \
    --data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
    --val_data_dir "$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
    --tokenizer_ckpt "$TOKENIZER" --eval_ckpt "$ckpt" --output_json "$output" \
    --device cuda --eval_seq_len 91 --eval_ctx 1 --horizons 90 \
    --eval_batch_size 4 --eval_max_batches 8 --num_workers 4 \
    --eval_num_rollouts 8 --eval_multisample_seed 20260813 \
    --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30 \
    --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0 \
    --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
    --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64 \
    --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4 \
    --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
    --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus
done
