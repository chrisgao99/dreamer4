#!/usr/bin/env bash
# Stage-1 best -> H30 3k -> H90 3k. Scene-level latent MoN N=8 and
# full-future raw motion supervision train the winner; collision, offroad, and
# kinematic self-consistency losses train every candidate.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
STAGE1_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage/waymo_wm_v1_egoact_focus_raw_noclamp_win11_randstart_b8_self05_norecon_time1_mapx1_1m/best.pt"

H30_RUN="waymo_wm_stage1best_mon8_fullmotion_physproxy_ctx1_h30_d1_chunk32s30_3k"
H90_RUN="waymo_wm_stage1best_mon8_fullmotion_physproxy_h30step3k_ctx1_h90_d1_chunk32s30_3k"
H30_DIR="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage/$H30_RUN"
H90_DIR="$REPO_ROOT/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage/$H90_RUN"
H30_FINAL="$H30_DIR/final_step_00003000.pt"
H90_FINAL="$H90_DIR/final_step_00003000.pt"

SESSION_NAME="${SESSION_NAME:-wm_stage1best_mon8_fullmotion_physproxy_h30_h90_3k_cuda1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"
H30_LOG="$LOG_DIR/$H30_RUN.log"
H90_LOG="$LOG_DIR/$H90_RUN.log"
PIPELINE_LOG="$LOG_DIR/${H90_RUN}_pipeline.log"

for path in "$PYTHON" "$TRAIN_SCRIPT" "$TOKENIZER_CKPT" "$STAGE1_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
for path in "$DATA_ROOT/train" "$DATA_ROOT/val"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 1; }
done
mkdir -p "$H30_DIR" "$H90_DIR" "$LOG_DIR" "$REPO_ROOT/waymo/wandb"

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
  echo "Pipeline log: $PIPELINE_LOG"
  echo "H30 log: $H30_LOG"
  echo "H90 log: $H90_LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

{
  echo "===== $(date) MoN8 full-motion + physical-proxy H30->H90 pipeline start ====="
  echo "session=$SESSION_NAME physical_cuda=$CUDA_VISIBLE_DEVICES"
  echo "stage1_ckpt=$STAGE1_CKPT"
  echo "h30_run=$H30_RUN steps=3000 lr=1e-5"
  echo "h90_run=$H90_RUN steps=3000 lr=5e-6 init=h30_final"
  echo "objective=rollout_mon N=8 ctx=1 sampler=D1 batch=1"
  echo "motion_gt=winner full_future xy:1 velocity:0.5 yaw:0.5 weight=1"
  echo "decoder=chunk32 stride30 keep_first; H30=one_31_frame_chunk H90=three_overlapping_chunks"
  echo "collision=fixed_4.8x2.0m projected_clearance warning=1.0m weight=0.1 all_candidates"
  echo "offroad=fixed_4.8x2.0m corners all_road_edges_15_16 margin=0.3m weight=0.1 all_candidates"
  echo "kinematic_weights=vxy:5,speed_yaw:2 all_candidates shortcut_retention_weight=0.1 warmup=500"
} | tee -a "$PIPELINE_LOG"

common_args=(
  --data_dir "$DATA_ROOT/train" --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT" --device cuda --seed 0 --num_workers "$NUM_WORKERS"
  --tokenizer_chunk_window 32 --tokenizer_chunk_stride 30
  --max_rollout_window 11 --eval_schedule shortcut --eval_d 1.0
  --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1
  --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64
  --grad_clip 1 --amp_dtype bf16
  --agent_xy_weight 1 --agent_vel_weight 0.5 --agent_yaw_weight 0.5
  --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4
  --agent_kinematic_xy_weight 5 --agent_speed_yaw_kinematic_weight 2 --kinematic_dt 0.1
  --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp
  --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus
  --train_decoded_loss_weight 0 --motion_gt_loss_weight 1 --train_objective rollout_mon
  --mon_num_samples 8 --mon_loss_weight 1 --mon_checkpoint_dynamics --mon_checkpoint_decoder
  --motion_checkpoint_dynamics --motion_checkpoint_decoder
  --rollout_shortcut_weight 0.1
  --physical_vehicle_length_m 4.8 --physical_vehicle_width_m 2.0
  --collision_loss_weight 0.1 --collision_warning_clearance_m 1.0 --collision_temperature_m 0.2
  --offroad_loss_weight 0.1 --offroad_boundary_margin_m 0.3 --offroad_temperature_m 0.2
  --road_edge_query_chunk_size 1024 --physical_warmup_steps 500
  --batch_size 1 --eval_batch_size 4 --max_steps 3000
  --log_every 10 --eval_every 500 --eval_subset_size 128 --eval_subset_seed 20260813
  --eval_max_batches 0 --save_every 500 --no-save_latest_each_epoch
  --weight_decay 0 --wandb --wandb_project waymo-world-model
)

if [[ ! -f "$H30_FINAL" ]]; then
  h30_start=(--init_ckpt "$STAGE1_CKPT")
  [[ -f "$H30_DIR/latest.pt" ]] && h30_start=(--resume "$H30_DIR/latest.pt")
  {
    echo "===== $(date) H30 start ====="
    "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
      --ckpt_dir "$H30_DIR" --seq_len 31 --eval_seq_len 31 --eval_ctx 1 --eval_horizon 30 \
      --eval_num_rollouts 8 --eval_multisample_seed 20260813 --eval_diversity_floor_ratio 0.5 \
      --eval_multisample_reference_json "$H30_DIR/stage1_multisample_reference.json" \
      --eval_multisample_reference_ckpt "$STAGE1_CKPT" \
      --lr 1e-5 --wandb_run_name "$H30_RUN" "${h30_start[@]}"
  } 2>&1 | tee -a "$H30_LOG"
fi
[[ -f "$H30_FINAL" ]] || { echo "H30 ended without $H30_FINAL" >&2; exit 1; }

echo "===== $(date) H30 complete; H90 start =====" | tee -a "$PIPELINE_LOG"
if [[ ! -f "$H90_FINAL" ]]; then
  h90_start=(--init_ckpt "$H30_FINAL")
  [[ -f "$H90_DIR/latest.pt" ]] && h90_start=(--resume "$H90_DIR/latest.pt")
  {
    echo "===== $(date) H90 start ====="
    "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
      --ckpt_dir "$H90_DIR" --seq_len 91 --eval_seq_len 91 --eval_ctx 1 --eval_horizon 90 \
      --eval_num_rollouts 8 --eval_multisample_seed 20260813 --eval_diversity_floor_ratio 0.5 \
      --eval_multisample_reference_json "$H90_DIR/stage1_multisample_reference.json" \
      --eval_multisample_reference_ckpt "$STAGE1_CKPT" \
      --lr 5e-6 --wandb_run_name "$H90_RUN" "${h90_start[@]}"
  } 2>&1 | tee -a "$H90_LOG"
fi
[[ -f "$H90_FINAL" ]] || { echo "H90 ended without $H90_FINAL" >&2; exit 1; }

echo "===== $(date) MoN8 full-motion + physical-proxy pipeline complete =====" | tee -a "$PIPELINE_LOG"
