#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-}"
if [[ -z "$EXPERIMENT" ]]; then
  echo "Usage: $0 {fixed_empirical|fixed_bootstrap|eight_random}" >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_world_model.py"
MANIFEST="${MANIFEST:-$REPO_ROOT/waymo/training/world_model/manifests/overfit_world_model_scenes8.txt}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-$REPO_ROOT/waymo/checkpoints/ooi50k_lat64_b32_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_decmap_noamp/best.pt}"

CUDA_DEVICE="${CUDA_DEVICE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WANDB_MODE="${WANDB_MODE:-online}"
USE_WANDB="${USE_WANDB:-1}"
DRY_RUN="${DRY_RUN:-0}"

case "$EXPERIMENT" in
  fixed_empirical)
    RUN_NAME="${RUN_NAME:-wm_overfit_fixed1_empirical_ctx1_h10_20260716}"
    NUM_SCENES=1
    TRAIN_REPEATS=1
    BATCH_SIZE=1
    EVAL_BATCH_SIZE=1
    MAX_STEPS=10000
    RANDOM_TIME_WINDOW_START=0
    SELF_FRACTION=0
    ;;
  fixed_bootstrap)
    RUN_NAME="${RUN_NAME:-wm_overfit_fixed1_shortcut_ctx1_h10_20260716}"
    NUM_SCENES=1
    TRAIN_REPEATS=8
    BATCH_SIZE=8
    EVAL_BATCH_SIZE=1
    MAX_STEPS=20000
    RANDOM_TIME_WINDOW_START=0
    SELF_FRACTION=0.857142857
    ;;
  eight_random)
    RUN_NAME="${RUN_NAME:-wm_overfit_scenes8_randwin_ctx1_h10_20260716}"
    NUM_SCENES=8
    TRAIN_REPEATS=1
    BATCH_SIZE=8
    EVAL_BATCH_SIZE=8
    MAX_STEPS=50000
    RANDOM_TIME_WINDOW_START=1
    SELF_FRACTION=0.857142857
    ;;
  *)
    echo "Unknown experiment: $EXPERIMENT" >&2
    echo "Expected one of: fixed_empirical fixed_bootstrap eight_random" >&2
    exit 2
    ;;
esac

SUBSET_ROOT="${SUBSET_ROOT:-$REPO_ROOT/waymo/data/world_model_overfit_subsets/$RUN_NAME}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/world_model_overfit/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/world_model_overfit}"
LOG="${LOG:-$LOG_DIR/$RUN_NAME.log}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file not found: $1" >&2
    exit 1
  fi
}

is_truthy() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" || "$1" == "YES" ]]
}

require_file "$PYTHON"
require_file "$TRAIN_SCRIPT"
require_file "$MANIFEST"
require_file "$TOKENIZER_CKPT"

mapfile -t ALL_SCENES < <(awk 'NF && $1 !~ /^#/' "$MANIFEST")
if [[ "${#ALL_SCENES[@]}" -lt "$NUM_SCENES" ]]; then
  echo "Manifest has ${#ALL_SCENES[@]} scenes, need $NUM_SCENES: $MANIFEST" >&2
  exit 1
fi

mkdir -p "$SUBSET_ROOT/train" "$SUBSET_ROOT/val" "$CKPT_DIR" "$LOG_DIR" "$REPO_ROOT/waymo/wandb"

expected_train=$((NUM_SCENES * TRAIN_REPEATS))
for ((scene_idx = 0; scene_idx < NUM_SCENES; scene_idx++)); do
  src="${ALL_SCENES[$scene_idx]}"
  require_file "$src"
  base="$(basename "$src")"

  val_link="$SUBSET_ROOT/val/$(printf '%02d__%s' "$scene_idx" "$base")"
  if [[ ! -L "$val_link" ]]; then
    ln -s "$src" "$val_link"
  elif [[ "$(readlink -f "$val_link")" != "$(readlink -f "$src")" ]]; then
    echo "Existing validation link points elsewhere: $val_link" >&2
    exit 1
  fi

  for ((repeat_idx = 0; repeat_idx < TRAIN_REPEATS; repeat_idx++)); do
    train_link="$SUBSET_ROOT/train/$(printf '%02d_%02d__%s' "$scene_idx" "$repeat_idx" "$base")"
    if [[ ! -L "$train_link" ]]; then
      ln -s "$src" "$train_link"
    elif [[ "$(readlink -f "$train_link")" != "$(readlink -f "$src")" ]]; then
      echo "Existing training link points elsewhere: $train_link" >&2
      exit 1
    fi
  done
done

actual_train="$(find "$SUBSET_ROOT/train" -maxdepth 1 -type l | wc -l)"
actual_val="$(find "$SUBSET_ROOT/val" -maxdepth 1 -type l | wc -l)"
if [[ "$actual_train" -ne "$expected_train" || "$actual_val" -ne "$NUM_SCENES" ]]; then
  echo "Unexpected subset contents under $SUBSET_ROOT: train=$actual_train/$expected_train val=$actual_val/$NUM_SCENES" >&2
  exit 1
fi

if [[ -f "$CKPT_DIR/latest.pt" ]]; then
  echo "Checkpoint already exists; refusing to overwrite or implicitly resume: $CKPT_DIR/latest.pt" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export WANDB_MODE
export WANDB_DIR="$REPO_ROOT/waymo/wandb"

train_args=(
  "$TRAIN_SCRIPT"
  --data_dir "$SUBSET_ROOT/train"
  --val_data_dir "$SUBSET_ROOT/val"
  --tokenizer_ckpt "$TOKENIZER_CKPT"
  --ckpt_dir "$CKPT_DIR"
  --seed 0
  --seq_len 11
  --eval_seq_len 11
  --eval_ctx 1
  --eval_horizon 10
  --max_rollout_window 11
  --eval_schedule shortcut
  --eval_d 0.25
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --max_steps "$MAX_STEPS"
  --log_every 20
  --eval_every 200
  --eval_max_batches 0
  --save_every 1000
  --no-save_latest_each_epoch
  --d_model_dyn 512
  --dyn_depth 8
  --n_heads 8
  --time_every 4
  --packing_factor 2
  --n_register 8
  --k_max 64
  --self_fraction "$SELF_FRACTION"
  --bootstrap_start 0
  --train_objective shortcut
  --tf_context 10
  --lr 1e-4
  --weight_decay 0
  --grad_clip 1.0
  --amp_dtype bf16
  --agent_xy_loss smooth_l1
  --agent_xy_parameterization absolute
  --focus_agent_weight 4
  --agent_kinematic_xy_weight 5
  --agent_speed_yaw_kinematic_weight 2
  --use_ego_actions
  --ego_action_source focus
  --ego_action_normalization raw
  --no-ego_action_clamp
  --agent_far_weight 0.25
  --agent_near_radius_m 50.0
  --agent_distance_source focus
  --train_decoded_loss_weight 0.0
  --wandb_project waymo-world-model-overfit
  --wandb_run_name "$RUN_NAME"
)

if is_truthy "$RANDOM_TIME_WINDOW_START"; then
  train_args+=(--random_time_window_start)
fi
if is_truthy "$USE_WANDB"; then
  train_args+=(--wandb)
fi

{
  echo "===== $(date) ====="
  echo "experiment=$EXPERIMENT run_name=$RUN_NAME"
  echo "cuda_physical=$CUDA_DEVICE tokenizer_ckpt=$TOKENIZER_CKPT"
  echo "subset_root=$SUBSET_ROOT train_samples=$actual_train val_samples=$actual_val"
  echo "num_unique_scenarios=$NUM_SCENES train_repeats=$TRAIN_REPEATS batch_size=$BATCH_SIZE"
  echo "fixed_window_start=$((1 - RANDOM_TIME_WINDOW_START)) random_time_window_start=$RANDOM_TIME_WINDOW_START"
  echo "self_fraction=$SELF_FRACTION max_steps=$MAX_STEPS weight_decay=0"
  echo "checkpoint_dir=$CKPT_DIR log=$LOG"
  echo "selected_scenes:"
  printf '  %s\n' "${ALL_SCENES[@]:0:$NUM_SCENES}"
  echo "========================"
} | tee "$LOG"

if is_truthy "$DRY_RUN"; then
  printf 'Dry-run command:\n  '
  printf '%q ' "$PYTHON" "${train_args[@]}"
  printf '\n'
  exit 0
fi

cd "$REPO_ROOT"
"$PYTHON" "${train_args[@]}" 2>&1 | tee -a "$LOG"
