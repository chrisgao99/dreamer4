#!/usr/bin/env bash
# Local single-GPU tmux launcher for the q-free hybrid-tokenizer comparison:
#   Stage 1: shortcut 300k + H30 selection over checkpoints every 50k
#   Stage 2: H30 rollout MoN N=8 + full motion + physical proxies, 30k
#   Stage 3: H90 with the same objective from H30 selected best, 30k

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HOME/.conda/envs/dreamer4/bin/python" ]]; then
    PYTHON="$HOME/.conda/envs/dreamer4/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi

TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/interaction_contrastive_hybrid_soft_v2_from_hard_relneg_dupfiltered_cuda3/best.pt}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
STAGE1_STEPS=300000
H30_STEPS=30000
H90_STEPS=30000
STAGE1_SELECTION_BATCHES=32

PREFIX="waymo_wm_hybridtok_n8_fullmotion_physproxy"
STAGE1_RUN="${PREFIX}_stage1_shortcut_b8_${STAGE1_STEPS}"
H30_RUN="${PREFIX}_stage1best_ctx1_h30_d1_chunk32s30_b1_${H30_STEPS}"
H90_RUN="${PREFIX}_h30best_ctx1_h90_d1_chunk32s30_b1_${H90_STEPS}"
SESSION_NAME="${SESSION_NAME:-wm_hybridtok_n8_motion_three_stage_gpu${CUDA_DEVICE}}"

STAGE1_DIR="$REPO_ROOT/waymo/checkpoints/$STAGE1_RUN"
H30_DIR="$REPO_ROOT/waymo/checkpoints/$H30_RUN"
H90_DIR="$REPO_ROOT/waymo/checkpoints/$H90_RUN"
STAGE1_FINAL="$STAGE1_DIR/final_step_00300000.pt"
STAGE1_BEST="$STAGE1_DIR/best.pt"
H30_FINAL="$H30_DIR/final_step_00030000.pt"
H30_BEST="$H30_DIR/best_multisample_finetuned.pt"
H90_FINAL="$H90_DIR/final_step_00030000.pt"
H90_BEST="$H90_DIR/best_multisample_finetuned.pt"

SELECTION_DIR="$REPO_ROOT/waymo/eval_results/world_model/$STAGE1_RUN/select_upto300k_h30_batches${STAGE1_SELECTION_BATCHES}"
SELECTION_TSV="$SELECTION_DIR/selection_summary.tsv"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"
EVAL_LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
PIPELINE_LOG="$LOG_DIR/${PREFIX}_three_stage_pipeline.log"
TMUX_LOG="$LOG_DIR/${PREFIX}_tmux_console.log"
STAGE1_LOG="$LOG_DIR/$STAGE1_RUN.log"
SELECTION_LOG="$EVAL_LOG_DIR/${STAGE1_RUN}_select_upto300k_h30.log"
H30_LOG="$LOG_DIR/$H30_RUN.log"
H90_LOG="$LOG_DIR/$H90_RUN.log"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing required directory: $1" >&2; exit 1; }
}

[[ -n "$PYTHON" && -x "$PYTHON" ]] || {
  echo "Could not find an executable Python. Activate dreamer4 or set PYTHON=/path/to/python." >&2
  exit 1
}
require_file "$TRAIN_SCRIPT"
require_file "$EVAL_SCRIPT"
require_file "$TOKENIZER_CKPT"
require_dir "$DATA_ROOT/train"
require_dir "$DATA_ROOT/val"

mkdir -p \
  "$STAGE1_DIR" "$H30_DIR" "$H90_DIR" "$SELECTION_DIR" \
  "$LOG_DIR" "$EVAL_LOG_DIR" "$REPO_ROOT/waymo/wandb"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed or not on PATH" >&2; exit 1; }
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
  fi

  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    TOKENIZER_CKPT="$TOKENIZER_CKPT" DATA_ROOT="$DATA_ROOT" \
    CUDA_DEVICE="$CUDA_DEVICE" NUM_WORKERS="$NUM_WORKERS" \
    WANDB_MODE="$WANDB_MODE" OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    SESSION_NAME="$SESSION_NAME" bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on

  echo "Started tmux session: $SESSION_NAME"
  echo "GPU: $CUDA_DEVICE"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Console log: $TMUX_LOG"
  echo "Pipeline log: $PIPELINE_LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OMP_NUM_THREADS
export PYTHONUNBUFFERED=1
export WANDB_MODE
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

# Capture everything visible in the tmux pane in addition to per-stage logs.
exec > >(tee -a "$TMUX_LOG") 2>&1

run_logged() {
  local log_path="$1"
  shift
  set +e
  "$@" 2>&1 | tee -a "$log_path" &
  local pipeline_pid=$!
  wait "$pipeline_pid"
  local status=$?
  set -e
  return "$status"
}

# Abort before expensive training if this is not the intended hybrid checkpoint.
"$PYTHON" - "$TOKENIZER_CKPT" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
assert checkpoint.get("format") == "waymo_interaction_contrastive_tokenizer_v1", checkpoint.get("format")
assert checkpoint.get("step") == 40000, checkpoint.get("step")
args = checkpoint.get("args", {})
if not isinstance(args, dict):
    args = vars(args)
assert args.get("contrastive_refined") is True, args.get("contrastive_refined")
assert args.get("contrastive_mode") == "hybrid", args.get("contrastive_mode")
print(f"Validated hybrid tokenizer: {path} (step={checkpoint['step']})", flush=True)
PY

{
  echo "===== $(date) local hybrid-tokenizer q-free N8-motion pipeline start/resume ====="
  echo "session=$SESSION_NAME host=$(hostname) physical_cuda=$CUDA_DEVICE visible_cuda=$CUDA_VISIBLE_DEVICES"
  echo "repo=$REPO_ROOT"
  echo "python=$PYTHON"
  echo "tokenizer=$TOKENIZER_CKPT"
  echo "data=$DATA_ROOT"
  echo "stage1=shortcut/300000; selection=H30 focus-FDE over 50k checkpoints"
  echo "stage2=rollout_mon/N8/full-motion/physical/H30/30000"
  echo "stage3=rollout_mon/N8/full-motion/physical/H90/30000"
  echo "model=standard latent dynamics; no q tokens, q state, q head, or q loss"
} | tee -a "$PIPELINE_LOG"

stage1_common=(
  --data_dir "$DATA_ROOT/train" --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT" --device cuda --seed 0 --num_workers "$NUM_WORKERS"
  --dynamics_variant standard
  --max_rollout_window 11 --eval_schedule shortcut --eval_d 0.25
  --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1
  --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64
  --grad_clip 1 --amp_dtype bf16
  --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4
  --agent_kinematic_xy_weight 5 --agent_speed_yaw_kinematic_weight 2
  --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp
  --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus
  --train_decoded_loss_weight 0
  --wandb --wandb_project waymo-world-model
)

if [[ ! -f "$STAGE1_FINAL" ]]; then
  stage1_start=()
  [[ -f "$STAGE1_DIR/latest.pt" ]] && stage1_start=(--resume "$STAGE1_DIR/latest.pt")
  run_logged "$STAGE1_LOG" \
    "$PYTHON" "$TRAIN_SCRIPT" "${stage1_common[@]}" \
    --ckpt_dir "$STAGE1_DIR" "${stage1_start[@]}" \
    --seq_len 11 --random_time_window_start \
    --eval_seq_len 11 --eval_ctx 1 --eval_horizon 10 \
    --batch_size 8 --eval_batch_size 4 --max_steps "$STAGE1_STEPS" \
    --log_every 100 --eval_every 0 --eval_max_batches 0 --save_every 50000 \
    --self_fraction 0.5 --bootstrap_start 0 --train_objective shortcut \
    --lr 1e-4 --weight_decay 1e-2 --wandb_run_name "$STAGE1_RUN"
fi
require_file "$STAGE1_FINAL"
echo "===== $(date) Stage 1 complete; selecting its best <=300k checkpoint =====" | tee -a "$PIPELINE_LOG"

if [[ ! -f "$STAGE1_BEST" ]]; then
  for step in 50000 100000 150000 200000 250000 300000; do
    printf -v step8 '%08d' "$step"
    candidate="$STAGE1_DIR/step_${step8}.pt"
    result="$SELECTION_DIR/step_${step8}_h30_batches${STAGE1_SELECTION_BATCHES}.json"
    require_file "$candidate"
    if [[ ! -f "$result" ]]; then
      echo "===== $(date) Stage-1 H30 selection step=$step =====" | tee -a "$SELECTION_LOG"
      run_logged "$SELECTION_LOG" \
        "$PYTHON" "$EVAL_SCRIPT" \
        --data_dir "$DATA_ROOT/train" --val_data_dir "$DATA_ROOT/val" \
        --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$candidate" --device cuda --seed 0 \
        --eval_batch_size 4 --eval_max_batches "$STAGE1_SELECTION_BATCHES" --num_workers "$NUM_WORKERS" \
        --eval_seq_len 31 --eval_ctx 1 --horizons 30 --max_rollout_window 11 \
        --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
        --packing_factor 2 --n_register 8 --k_max 64 \
        --dynamics_attend_map --map_cross_every 1 \
        --eval_schedule shortcut --eval_d 0.25 \
        --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
        --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
        --output_json "$result"
    fi
  done

  "$PYTHON" - "$SELECTION_DIR" "$SELECTION_TSV" "$STAGE1_BEST" <<'PY'
import json
import os
import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2])
destination = pathlib.Path(sys.argv[3])
rows = []
for path in sorted(root.glob("step_*_h30_batches*.json")):
    data = json.loads(path.read_text())
    metrics = data["metrics"]["h30"]
    rows.append(
        (
            float(metrics["focus_agent_fde_m"]),
            float(metrics["focus_agent_xy_mae_m"]),
            float(metrics["latent_mse_future"]),
            int(data["ckpt_step"]),
            str(data["eval_ckpt"]),
        )
    )
if len(rows) != 6:
    raise SystemExit(f"Expected six Stage-1 selection results, found {len(rows)}")
rows.sort()
summary.write_text(
    "step\tfocus_fde_m\tfocus_ade_m\tlatent_mse\tcheckpoint\n"
    + "".join(
        f"{step}\t{fde:.8f}\t{ade:.8f}\t{latent:.8f}\t{checkpoint}\n"
        for fde, ade, latent, step, checkpoint in rows
    )
)
source = pathlib.Path(rows[0][4])
temporary = destination.with_suffix(".tmp")
shutil.copy2(source, temporary)
os.replace(temporary, destination)
print(f"Selected Stage-1 step={rows[0][3]} focus_fde_m={rows[0][0]:.8f}: {source}", flush=True)
print(f"Copied selected checkpoint to: {destination}", flush=True)
PY
fi
require_file "$STAGE1_BEST"

rollout_common=(
  --data_dir "$DATA_ROOT/train" --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT" --device cuda --seed 0 --num_workers "$NUM_WORKERS"
  --dynamics_variant standard
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
  --motion_checkpoint_dynamics --motion_checkpoint_decoder --rollout_shortcut_weight 0.1
  --physical_vehicle_length_m 4.8 --physical_vehicle_width_m 2.0
  --collision_loss_weight 0.1 --collision_warning_clearance_m 1.0 --collision_temperature_m 0.2
  --offroad_loss_weight 0.1 --offroad_boundary_margin_m 0.3 --offroad_temperature_m 0.2
  --road_edge_query_chunk_size 1024 --physical_warmup_steps 500
  --batch_size 1 --eval_batch_size 4 --max_steps 30000
  --log_every 10 --eval_every 3000 --eval_subset_size 128 --eval_subset_seed 20260813
  --eval_max_batches 0 --eval_num_rollouts 8 --eval_multisample_seed 20260813
  --eval_diversity_floor_ratio 0.5 --eval_multisample_physical
  --save_every 3000 --no-save_latest_each_epoch
  --weight_decay 0 --wandb --wandb_project waymo-world-model
)

echo "===== $(date) Stage 1 selected; starting N8 full-motion physical H30 =====" | tee -a "$PIPELINE_LOG"
if [[ ! -f "$H30_FINAL" ]]; then
  h30_start=(--init_ckpt "$STAGE1_BEST")
  [[ -f "$H30_DIR/latest.pt" ]] && h30_start=(--resume "$H30_DIR/latest.pt")
  run_logged "$H30_LOG" \
    "$PYTHON" "$TRAIN_SCRIPT" "${rollout_common[@]}" \
    --ckpt_dir "$H30_DIR" --seq_len 31 --eval_seq_len 31 --eval_ctx 1 --eval_horizon 30 \
    --eval_multisample_reference_json "$H30_DIR/stage1_multisample_reference.json" \
    --eval_multisample_reference_ckpt "$STAGE1_BEST" \
    --lr 1e-5 --wandb_run_name "$H30_RUN" "${h30_start[@]}"
fi
require_file "$H30_FINAL"
require_file "$H30_BEST"

echo "===== $(date) H30 complete; starting H90 from H30 selected best =====" | tee -a "$PIPELINE_LOG"
if [[ ! -f "$H90_FINAL" ]]; then
  h90_start=(--init_ckpt "$H30_BEST")
  [[ -f "$H90_DIR/latest.pt" ]] && h90_start=(--resume "$H90_DIR/latest.pt")
  run_logged "$H90_LOG" \
    "$PYTHON" "$TRAIN_SCRIPT" "${rollout_common[@]}" \
    --ckpt_dir "$H90_DIR" --seq_len 91 --eval_seq_len 91 --eval_ctx 1 --eval_horizon 90 \
    --eval_multisample_reference_json "$H90_DIR/stage1_multisample_reference.json" \
    --eval_multisample_reference_ckpt "$STAGE1_BEST" \
    --lr 5e-6 --wandb_run_name "$H90_RUN" "${h90_start[@]}"
fi
require_file "$H90_FINAL"
require_file "$H90_BEST"

{
  echo "===== $(date) all three stages complete ====="
  echo "stage1_selected=$STAGE1_BEST"
  echo "h30_selected=$H30_BEST"
  echo "h90_selected=$H90_BEST"
  echo "selection_summary=$SELECTION_TSV"
} | tee -a "$PIPELINE_LOG"
