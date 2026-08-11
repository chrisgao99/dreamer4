#!/usr/bin/env bash
# Matched MotionLatent V1: random-init time_every=1/context=10 base, then exact H30 and H90.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_motion_latent_v1.py"
EVAL_SCRIPT="$REPO_ROOT/waymo/evaluation/eval_waymo_world_model_horizons.py"
TOKENIZER_CKPT="$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt"
READER_CKPT="$REPO_ROOT/waymo/checkpoints/waymo_semantic_reader_agent32_zonly_d256_depth2_b8_20k_v1/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"

BASE_RUN="waymo_motion_latent_v1_matched_time1_ctx10_b8_stage1_600k"
H30_RUN="waymo_motion_latent_v1_matched_time1_ctx10_best600k_exact_h30_b1_50k"
H90_RUN="waymo_motion_latent_v1_matched_time1_ctx10_best600k_h30_50k_exact_h90_b1_50k"
BASE_DIR="$REPO_ROOT/waymo/checkpoints/$BASE_RUN"
H30_DIR="$REPO_ROOT/waymo/checkpoints/$H30_RUN"
H90_DIR="$REPO_ROOT/waymo/checkpoints/$H90_RUN"
BASE_FINAL="$BASE_DIR/final_step_00600000.pt"
H30_FINAL="$H30_DIR/final_step_00050000.pt"
H90_FINAL="$H90_DIR/final_step_00050000.pt"

SESSION_NAME="${SESSION_NAME:-wm_motion_latent_v1_matched600k_h30_h90_cuda1}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SELECTION_BATCHES="${SELECTION_BATCHES:-32}"
SELECTION_DIR="$REPO_ROOT/waymo/eval_results/world_model/$BASE_RUN/select_upto600k_h30_batches${SELECTION_BATCHES}"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"
EVAL_LOG_DIR="$REPO_ROOT/waymo/logs/evaluation"
PIPELINE_LOG="$LOG_DIR/${H90_RUN}_pipeline.log"
BASE_LOG="$LOG_DIR/$BASE_RUN.log"
SELECTION_LOG="$EVAL_LOG_DIR/${BASE_RUN}_select_upto600k_h30_cuda1.log"
H30_LOG="$LOG_DIR/$H30_RUN.log"
H90_LOG="$LOG_DIR/$H90_RUN.log"

for path in "$PYTHON" "$TRAIN_SCRIPT" "$EVAL_SCRIPT" "$TOKENIZER_CKPT" "$READER_CKPT"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
for path in "$DATA_ROOT/train" "$DATA_ROOT/val"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 1; }
done
mkdir -p "$BASE_DIR" "$H30_DIR" "$H90_DIR" "$SELECTION_DIR" "$LOG_DIR" "$EVAL_LOG_DIR" "$REPO_ROOT/waymo/wandb"

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
  echo "matched=time_every1 max_context10 base_batch8 base_steps600k exact_batch1 exact_steps50k+50k"
  echo "selection=H30 focus_agent_fde_m batches=$SELECTION_BATCHES"
} | tee -a "$PIPELINE_LOG"

if [[ ! -f "$BASE_FINAL" ]]; then
  start=()
  [[ -f "$BASE_DIR/latest.pt" ]] && start=(--resume "$BASE_DIR/latest.pt")
  "$PYTHON" "$TRAIN_SCRIPT" \
    --data_dir "$DATA_ROOT/train" --tokenizer_ckpt "$TOKENIZER_CKPT" \
    --semantic_reader_ckpt "$READER_CKPT" --ckpt_dir "$BASE_DIR" "${start[@]}" \
    --device cuda --seed 0 --batch_size 8 --num_workers "$NUM_WORKERS" \
    --max_steps 600000 --train_mode online_step --rollout_end 90 --max_context 10 \
    --d_model 512 --depth 8 --n_heads 8 --time_every 1 --map_cross_every 1 \
    --packing_factor 2 --n_register 8 --k_max 64 --kinematic_dt 0.1 \
    --motion_weight 1 --motion_validity_weight 0.2 --consistency_weight 0.1 \
    --bootstrap_start 20000 --bootstrap_ramp_end 60000 --bootstrap_weight 0.1 \
    --lr 1e-4 --weight_decay 1e-2 --grad_clip 1 --amp_dtype bf16 \
    --log_every 100 --save_every 50000 \
    --ego_action_source focus --ego_action_normalization raw \
    --wandb --wandb_project waymo-world-model --wandb_run_name "$BASE_RUN" \
    2>&1 | tee -a "$BASE_LOG"
fi
[[ -f "$BASE_FINAL" ]] || { echo "Base training ended without $BASE_FINAL" >&2; exit 1; }

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
  echo "===== $(date) Stage 1 selection complete ====="
  echo "best_ckpt=$BEST_CKPT"
  echo "selection_summary=$SELECTION_TSV"
} | tee -a "$PIPELINE_LOG"

run_exact_stage() {
  local run_name="$1" ckpt_dir="$2" init_ckpt="$3" horizon="$4" lr="$5" log="$6"
  local final_ckpt="$ckpt_dir/final_step_00050000.pt"
  if [[ ! -f "$final_ckpt" ]]; then
    local -a start=(--init_ckpt "$init_ckpt")
    [[ -f "$ckpt_dir/latest.pt" ]] && start=(--resume "$ckpt_dir/latest.pt")
    "$PYTHON" "$TRAIN_SCRIPT" \
      --data_dir "$DATA_ROOT/train" --tokenizer_ckpt "$TOKENIZER_CKPT" \
      --semantic_reader_ckpt "$READER_CKPT" --ckpt_dir "$ckpt_dir" "${start[@]}" \
      --device cuda --seed 0 --batch_size 1 --num_workers "$NUM_WORKERS" \
      --max_steps 50000 --train_mode exact_rollout --rollout_end "$horizon" --max_context 10 \
      --d_model 512 --depth 8 --n_heads 8 --time_every 1 --map_cross_every 1 \
      --packing_factor 2 --n_register 8 --k_max 64 --kinematic_dt 0.1 \
      --motion_weight 1 --motion_validity_weight 0.2 --consistency_weight 0.1 \
      --lr "$lr" --weight_decay 0 --grad_clip 1 --amp_dtype bf16 \
      --log_every 20 --save_every 5000 \
      --ego_action_source focus --ego_action_normalization raw \
      --wandb --wandb_project waymo-world-model --wandb_run_name "$run_name" \
      2>&1 | tee -a "$log"
  fi
  [[ -f "$final_ckpt" ]] || { echo "$run_name ended without $final_ckpt" >&2; exit 1; }
}

run_exact_stage "$H30_RUN" "$H30_DIR" "$BEST_CKPT" 30 1e-5 "$H30_LOG"
echo "===== $(date) H30 complete; starting H90 =====" | tee -a "$PIPELINE_LOG"
run_exact_stage "$H90_RUN" "$H90_DIR" "$H30_FINAL" 90 5e-6 "$H90_LOG"

echo "===== $(date) pipeline complete =====" | tee -a "$PIPELINE_LOG"

