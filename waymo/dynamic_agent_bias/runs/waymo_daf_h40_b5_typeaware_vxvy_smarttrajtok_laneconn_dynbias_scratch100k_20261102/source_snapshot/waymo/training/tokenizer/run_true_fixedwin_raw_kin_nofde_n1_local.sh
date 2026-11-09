#!/usr/bin/env bash
set -euo pipefail

# True fixed-window overfit sanity check:
# - one scene only
# - train and val point to the same NPZ
# - fixed time window start = 0, no random window sampling
# - dropout = 0, weight decay = 0
# - train from scratch, no resume/init checkpoint

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_NAME="${RUN_NAME:-true_fixedwin_raw_kin_nofde_n1_start0_dropout0_wd0}"
DATA_FILE="${DATA_FILE:-$REPO_ROOT/waymo/data/waymo_vector_dataset_ooi_centered_50k/val/9b21bf1678ba3688_focus_814_src3.npz}"
SUBSET_ROOT="${SUBSET_ROOT:-$REPO_ROOT/waymo/data/overfit_subsets/${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/overfit/true_fixedwin/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/overfit/true_fixedwin}"
LOG="${LOG:-$LOG_DIR/${RUN_NAME}.log}"

BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-20000}"
MAX_STEPS="${MAX_STEPS:-20000}"
TIME_WINDOW="${TIME_WINDOW:-32}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_EVERY="${EVAL_EVERY:-100}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
SEED="${SEED:-0}"

D_MODEL="${D_MODEL:-256}"
N_HEADS="${N_HEADS:-4}"
DEPTH="${DEPTH:-4}"
DECODER_DEPTH="${DECODER_DEPTH:-4}"
N_LATENTS="${N_LATENTS:-16}"
D_BOTTLENECK="${D_BOTTLENECK:-32}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
MLP_RATIO="${MLP_RATIO:-4.0}"
TIME_EVERY="${TIME_EVERY:-1}"

ENCODER_VARIANT="${ENCODER_VARIANT:-static_map_query}"
MAP_DEPTH="${MAP_DEPTH:-2}"
MAP_CROSS_EVERY="${MAP_CROSS_EVERY:-1}"
MAP_QUERY_TOKENS="${MAP_QUERY_TOKENS:-latent_agent}"
BOTTLENECK_OUTPUT="${BOTTLENECK_OUTPUT:-tanh}"

LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
DROPOUT="${DROPOUT:-0.0}"

AGENT_XY_LOSS="${AGENT_XY_LOSS:-smooth_l1}"
AGENT_XY_PARAMETERIZATION="${AGENT_XY_PARAMETERIZATION:-absolute}"
AGENT_DELTA_XY_WEIGHT="${AGENT_DELTA_XY_WEIGHT:-0}"
AGENT_FDE_XY_WEIGHT="${AGENT_FDE_XY_WEIGHT:-0}"
AGENT_KINEMATIC_XY_WEIGHT="${AGENT_KINEMATIC_XY_WEIGHT:-5}"
AGENT_SPEED_YAW_KINEMATIC_WEIGHT="${AGENT_SPEED_YAW_KINEMATIC_WEIGHT:-2}"
KINEMATIC_DT="${KINEMATIC_DT:-0.1}"
FOCUS_AGENT_WEIGHT="${FOCUS_AGENT_WEIGHT:-4}"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export WANDB_MODE="${WANDB_MODE:-offline}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_file "$PYTHON"
require_file "$DATA_FILE"
require_file "$REPO_ROOT/waymo/training/train_waymo_vector_tokenizer.py"

mkdir -p "$SUBSET_ROOT/train" "$SUBSET_ROOT/val" "$CKPT_DIR" "$LOG_DIR"
base="$(basename "$DATA_FILE")"
for split in train val; do
  link="$SUBSET_ROOT/$split/$base"
  if [[ -e "$link" || -L "$link" ]]; then
    current="$(readlink "$link" || true)"
    if [[ "$current" != "$DATA_FILE" ]]; then
      echo "Existing subset link points elsewhere: $link -> $current" >&2
      echo "Expected: $DATA_FILE" >&2
      exit 1
    fi
  else
    ln -s "$DATA_FILE" "$link"
  fi
done
printf '%s\n' "$DATA_FILE" > "$SUBSET_ROOT/manifest.txt"

cd "$REPO_ROOT"

{
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME"
  echo "mode=true_fixed_window_overfit"
  echo "source_best_grid_ckpt_params=raw_kin_nofde_n10"
  echo "train_from_scratch=1 resume=0 init_from=none"
  echo "data_file=$DATA_FILE"
  echo "subset_root=$SUBSET_ROOT"
  echo "ckpt_dir=$CKPT_DIR"
  echo "log=$LOG"
  echo "cuda=$CUDA_VISIBLE_DEVICES"
  echo "batch_size=$BATCH_SIZE epochs=$EPOCHS max_steps=$MAX_STEPS time_window=$TIME_WINDOW time_window_start=0 random_time_window_start=0 eval_random_time_window_start=0"
  echo "d_model=$D_MODEL n_heads=$N_HEADS depth=$DEPTH decoder_depth=$DECODER_DEPTH n_latents=$N_LATENTS d_bottleneck=$D_BOTTLENECK hidden_dim=$HIDDEN_DIM"
  echo "encoder_variant=$ENCODER_VARIANT map_depth=$MAP_DEPTH map_cross_every=$MAP_CROSS_EVERY map_query_tokens=$MAP_QUERY_TOKENS bottleneck_output=$BOTTLENECK_OUTPUT"
  echo "dropout=$DROPOUT weight_decay=$WEIGHT_DECAY lr=$LR grad_clip=$GRAD_CLIP no_amp=1"
  echo "agent_xy_loss=$AGENT_XY_LOSS agent_xy_parameterization=$AGENT_XY_PARAMETERIZATION agent_delta_xy_weight=$AGENT_DELTA_XY_WEIGHT agent_fde_xy_weight=$AGENT_FDE_XY_WEIGHT agent_kinematic_xy_weight=$AGENT_KINEMATIC_XY_WEIGHT agent_speed_yaw_kinematic_weight=$AGENT_SPEED_YAW_KINEMATIC_WEIGHT kinematic_dt=$KINEMATIC_DT focus_agent_weight=$FOCUS_AGENT_WEIGHT"
  echo "selected_scenes:"
  sed 's/^/  /' "$SUBSET_ROOT/manifest.txt"
  echo "========================"
} | tee "$LOG"

"$PYTHON" waymo/training/train_waymo_vector_tokenizer.py \
  --data_dir "$SUBSET_ROOT/train" \
  --val_data_dir "$SUBSET_ROOT/val" \
  --ckpt_dir "$CKPT_DIR" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --time_window "$TIME_WINDOW" \
  --epochs "$EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --d_model "$D_MODEL" \
  --n_heads "$N_HEADS" \
  --depth "$DEPTH" \
  --decoder_depth "$DECODER_DEPTH" \
  --n_latents "$N_LATENTS" \
  --d_bottleneck "$D_BOTTLENECK" \
  --hidden_dim "$HIDDEN_DIM" \
  --mlp_ratio "$MLP_RATIO" \
  --time_every "$TIME_EVERY" \
  --encoder_variant "$ENCODER_VARIANT" \
  --map_depth "$MAP_DEPTH" \
  --map_cross_every "$MAP_CROSS_EVERY" \
  --map_query_tokens "$MAP_QUERY_TOKENS" \
  --bottleneck_output "$BOTTLENECK_OUTPUT" \
  --dropout "$DROPOUT" \
  --agent_xy_loss "$AGENT_XY_LOSS" \
  --agent_xy_parameterization "$AGENT_XY_PARAMETERIZATION" \
  --agent_delta_xy_weight "$AGENT_DELTA_XY_WEIGHT" \
  --agent_fde_xy_weight "$AGENT_FDE_XY_WEIGHT" \
  --agent_kinematic_xy_weight "$AGENT_KINEMATIC_XY_WEIGHT" \
  --agent_speed_yaw_kinematic_weight "$AGENT_SPEED_YAW_KINEMATIC_WEIGHT" \
  --kinematic_dt "$KINEMATIC_DT" \
  --focus_agent_weight "$FOCUS_AGENT_WEIGHT" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --grad_clip "$GRAD_CLIP" \
  --log_every "$LOG_EVERY" \
  --eval_every "$EVAL_EVERY" \
  --save_every "$SAVE_EVERY" \
  --no_amp \
  2>&1 | tee -a "$LOG"
