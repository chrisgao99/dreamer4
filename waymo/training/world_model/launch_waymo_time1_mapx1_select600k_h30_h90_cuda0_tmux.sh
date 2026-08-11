#!/usr/bin/env bash
# Select the best <=600k shortcut checkpoint on H30 validation, then run exact H30 and H90 fine-tuning.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"

BASE_RUN="waymo_wm_v1_egoact_focus_raw_noclamp_win11_randstart_b8_self05_norecon_time1_mapx1_1m"
BASE_DIR="$REPO_ROOT/waymo/checkpoints/$BASE_RUN"
H30_RUN="waymo_wm_time1_mapx1_best_upto600k_exact_ctx1_h30_b1_50k"
H90_RUN="waymo_wm_time1_mapx1_best600k_h30_50k_exact_ctx1_h90_b1_50k"
H30_DIR="$REPO_ROOT/waymo/checkpoints/$H30_RUN"
H90_DIR="$REPO_ROOT/waymo/checkpoints/$H90_RUN"
H30_FINAL="$H30_DIR/final_step_00050000.pt"
H90_FINAL="$H90_DIR/final_step_00050000.pt"

SESSION_NAME="${SESSION_NAME:-wm_time1_mapx1_select600k_h30_h90_cuda0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SELECTION_BATCHES="${SELECTION_BATCHES:-32}"
SELECTION_DIR="$REPO_ROOT/waymo/eval_results/world_model/$BASE_RUN/select_upto600k_h30_batches${SELECTION_BATCHES}"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"
EVAL_LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
PIPELINE_LOG="$LOG_DIR/${H90_RUN}_pipeline.log"
SELECTION_LOG="$EVAL_LOG_DIR/${BASE_RUN}_select_upto600k_h30_cuda0.log"
H30_LOG="$LOG_DIR/$H30_RUN.log"
H90_LOG="$LOG_DIR/$H90_RUN.log"

for path in "$PYTHON" "$TRAIN_SCRIPT" "$EVAL_SCRIPT" "$TOKENIZER_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
for path in "$DATA_ROOT/train" "$DATA_ROOT/val" "$BASE_DIR"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 1; }
done
mkdir -p "$H30_DIR" "$H90_DIR" "$SELECTION_DIR" "$LOG_DIR" "$EVAL_LOG_DIR" "$REPO_ROOT/waymo/wandb"

if [[ "${RUN_INSIDE_TMUX:-0}" != "1" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    exit 1
  fi
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  printf -v tmux_command '%q ' env \
    RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" \
    SESSION_NAME="$SESSION_NAME" CUDA_DEVICE="$CUDA_DEVICE" \
    NUM_WORKERS="$NUM_WORKERS" SELECTION_BATCHES="$SELECTION_BATCHES" \
    bash "$SCRIPT_PATH"
  tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$tmux_command"
  tmux set-option -t "$SESSION_NAME" remain-on-exit on
  echo "Started tmux session: $SESSION_NAME"
  echo "Pipeline log: $PIPELINE_LOG"
  echo "Selection log: $SELECTION_LOG"
  exit 0
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

{
  echo "===== $(date) pipeline start ====="
  echo "session=$SESSION_NAME cuda=$CUDA_VISIBLE_DEVICES"
  echo "selection=candidates_50k_to_600k metric=focus_agent_fde_m horizon=30 batches=$SELECTION_BATCHES"
  echo "h30=50000_steps batch1 lr1e-5 h90=50000_steps batch1 lr5e-6"
} | tee -a "$PIPELINE_LOG"

# Evaluate the twelve saved Stage-1 checkpoints with an identical deterministic validation subset.
for step in $(seq 50000 50000 600000); do
  printf -v step8 '%08d' "$step"
  ckpt="$BASE_DIR/step_${step8}.pt"
  result="$SELECTION_DIR/step_${step8}_h30_batches${SELECTION_BATCHES}.json"
  [[ -f "$ckpt" ]] || { echo "Missing selection candidate: $ckpt" >&2; exit 1; }
  if [[ ! -f "$result" ]]; then
    {
      echo "===== $(date) selecting step=$step ====="
      "$PYTHON" "$EVAL_SCRIPT" \
        --data_dir "$DATA_ROOT/train" --val_data_dir "$DATA_ROOT/val" \
        --tokenizer_ckpt "$TOKENIZER_CKPT" --eval_ckpt "$ckpt" --device cuda --seed 0 \
        --eval_batch_size 4 --eval_max_batches "$SELECTION_BATCHES" --num_workers "$NUM_WORKERS" \
        --eval_seq_len 31 --eval_ctx 1 --horizons 30 --max_rollout_window 11 \
        --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1 \
        --packing_factor 2 --n_register 8 --k_max 64 \
        --dynamics_attend_map --map_cross_every 1 \
        --eval_schedule shortcut --eval_d 0.25 \
        --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp \
        --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus \
        --output_json "$result"
    } 2>&1 | tee -a "$SELECTION_LOG"
  fi
done

SELECTION_TSV="$SELECTION_DIR/selection_summary.tsv"
BEST_CKPT="$($PYTHON -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
rows = []
for path in sorted(root.glob("step_*_h30_batches*.json")):
    data = json.loads(path.read_text())
    metrics = data["metrics"]["h30"]
    rows.append((float(metrics["focus_agent_fde_m"]), float(metrics["focus_agent_xy_mae_m"]), float(metrics["latent_mse_future"]), int(data["ckpt_step"]), data["eval_ckpt"]))
if not rows:
    raise SystemExit("No completed checkpoint-selection JSON files")
rows.sort()
out.write_text("step\tfocus_fde_m\tfocus_ade_m\tlatent_mse\tcheckpoint\n" + "".join(f"{step}\t{fde:.8f}\t{ade:.8f}\t{latent:.8f}\t{ckpt}\n" for fde, ade, latent, step, ckpt in rows))
print(rows[0][4])
' "$SELECTION_DIR" "$SELECTION_TSV")"

{
  echo "===== $(date) selection complete ====="
  echo "best_ckpt=$BEST_CKPT"
  echo "selection_summary=$SELECTION_TSV"
} | tee -a "$PIPELINE_LOG"

common_args=(
  --data_dir "$DATA_ROOT/train" --val_data_dir "$DATA_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT" --device cuda --seed 0 --num_workers "$NUM_WORKERS"
  --max_rollout_window 11 --eval_schedule shortcut --eval_d 0.25
  --d_model_dyn 512 --dyn_depth 8 --n_heads 8 --time_every 1
  --dynamics_attend_map --map_cross_every 1 --packing_factor 2 --n_register 8 --k_max 64
  --grad_clip 1 --amp_dtype bf16
  --agent_xy_loss smooth_l1 --agent_xy_parameterization absolute --focus_agent_weight 4
  --agent_kinematic_xy_weight 5 --agent_speed_yaw_kinematic_weight 2
  --use_ego_actions --ego_action_source focus --ego_action_normalization raw --no-ego_action_clamp
  --agent_far_weight 0.25 --agent_near_radius_m 50 --agent_distance_source focus
  --train_decoded_loss_weight 0 --train_objective rollout
  --batch_size 1 --eval_batch_size 4 --max_steps 50000
  --log_every 20 --eval_every 5000 --eval_max_batches 32 --save_every 5000 --no-save_latest_each_epoch
  --weight_decay 0 --wandb --wandb_project waymo-world-model
)

if [[ ! -f "$H30_FINAL" ]]; then
  start=(--init_ckpt "$BEST_CKPT")
  [[ -f "$H30_DIR/latest.pt" ]] && start=(--resume "$H30_DIR/latest.pt")
  "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
    --ckpt_dir "$H30_DIR" --seq_len 31 --eval_seq_len 31 --eval_ctx 1 --eval_horizon 30 \
    --lr 1e-5 --wandb_run_name "$H30_RUN" "${start[@]}" 2>&1 | tee -a "$H30_LOG"
fi
[[ -f "$H30_FINAL" ]] || { echo "H30 ended without $H30_FINAL" >&2; exit 1; }

echo "===== $(date) H30 complete; starting H90 =====" | tee -a "$PIPELINE_LOG"
if [[ ! -f "$H90_FINAL" ]]; then
  start=(--init_ckpt "$H30_FINAL")
  [[ -f "$H90_DIR/latest.pt" ]] && start=(--resume "$H90_DIR/latest.pt")
  "$PYTHON" "$TRAIN_SCRIPT" "${common_args[@]}" \
    --ckpt_dir "$H90_DIR" --seq_len 91 --eval_seq_len 91 --eval_ctx 1 --eval_horizon 90 \
    --lr 5e-6 --wandb_run_name "$H90_RUN" "${start[@]}" 2>&1 | tee -a "$H90_LOG"
fi
[[ -f "$H90_FINAL" ]] || { echo "H90 ended without $H90_FINAL" >&2; exit 1; }

echo "===== $(date) pipeline complete =====" | tee -a "$PIPELINE_LOG"

