#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/baz7dy/tri30/dreamer4}"
PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
LIST_FILE="${LIST_FILE:-waymo/evaluation/waymo_fulltraj_random10.txt}"
VARIANT="${VARIANT:-raw_gmm_kin_fde}"
OVERFIT_N="${OVERFIT_N:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
WANDB_MODE="${WANDB_MODE:-offline}"
USE_WANDB="${USE_WANDB:-0}"
DRY_RUN="${DRY_RUN:-0}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-20000}"
MAX_STEPS="${MAX_STEPS:-20000}"
D_MODEL="${D_MODEL:-256}"
DEPTH="${DEPTH:-4}"
DECODER_DEPTH="${DECODER_DEPTH:-4}"
N_LATENTS="${N_LATENTS:-16}"
D_BOTTLENECK="${D_BOTTLENECK:-32}"
TIME_WINDOW="${TIME_WINDOW:-32}"
LOG_EVERY="${LOG_EVERY:-20}"
EVAL_EVERY="${EVAL_EVERY:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
BOTTLENECK_OUTPUT="${BOTTLENECK_OUTPUT:-tanh}"
DECODER_USE_AGENT_TOKENS="${DECODER_USE_AGENT_TOKENS:-0}"

ENCODER_VARIANT="${ENCODER_VARIANT:-static_map_query}"
MAP_DEPTH="${MAP_DEPTH:-2}"
MAP_CROSS_EVERY="${MAP_CROSS_EVERY:-1}"
MAP_QUERY_TOKENS="${MAP_QUERY_TOKENS:-latent_agent}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
NO_AMP="${NO_AMP:-1}"
SEED="${SEED:-0}"
RESUME_IF_EXISTS="${RESUME_IF_EXISTS:-0}"
AGENT_XY_PARAMETERIZATION="${AGENT_XY_PARAMETERIZATION:-absolute}"

case "$VARIANT" in
  raw_gmm_kin_fde)
    AGENT_XY_LOSS="${AGENT_XY_LOSS:-gmm}"
    AGENT_DELTA_XY_WEIGHT="${AGENT_DELTA_XY_WEIGHT:-0}"
    AGENT_FDE_XY_WEIGHT="${AGENT_FDE_XY_WEIGHT:-2}"
    AGENT_KINEMATIC_XY_WEIGHT="${AGENT_KINEMATIC_XY_WEIGHT:-5}"
    AGENT_SPEED_YAW_KINEMATIC_WEIGHT="${AGENT_SPEED_YAW_KINEMATIC_WEIGHT:-2}"
    ;;
  raw_kin_fde)
    AGENT_XY_LOSS="${AGENT_XY_LOSS:-smooth_l1}"
    AGENT_DELTA_XY_WEIGHT="${AGENT_DELTA_XY_WEIGHT:-0}"
    AGENT_FDE_XY_WEIGHT="${AGENT_FDE_XY_WEIGHT:-2}"
    AGENT_KINEMATIC_XY_WEIGHT="${AGENT_KINEMATIC_XY_WEIGHT:-5}"
    AGENT_SPEED_YAW_KINEMATIC_WEIGHT="${AGENT_SPEED_YAW_KINEMATIC_WEIGHT:-2}"
    ;;
  raw_kin_nofde)
    AGENT_XY_LOSS="${AGENT_XY_LOSS:-smooth_l1}"
    AGENT_DELTA_XY_WEIGHT="${AGENT_DELTA_XY_WEIGHT:-0}"
    AGENT_FDE_XY_WEIGHT="${AGENT_FDE_XY_WEIGHT:-0}"
    AGENT_KINEMATIC_XY_WEIGHT="${AGENT_KINEMATIC_XY_WEIGHT:-5}"
    AGENT_SPEED_YAW_KINEMATIC_WEIGHT="${AGENT_SPEED_YAW_KINEMATIC_WEIGHT:-2}"
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    echo "Expected one of: raw_gmm_kin_fde raw_kin_fde raw_kin_nofde" >&2
    exit 1
    ;;
esac

KINEMATIC_DT="${KINEMATIC_DT:-0.1}"
FOCUS_AGENT_WEIGHT="${FOCUS_AGENT_WEIGHT:-4}"

if [[ "$LIST_FILE" != /* ]]; then
  LIST_FILE="$REPO_ROOT/$LIST_FILE"
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_file "$PYTHON"
require_file "$LIST_FILE"
require_file "$REPO_ROOT/waymo/training/train_waymo_vector_tokenizer.py"

if [[ "$OVERFIT_N" -le 0 ]]; then
  echo "OVERFIT_N must be positive, got $OVERFIT_N" >&2
  exit 1
fi

LIST_STEM="$(basename "$LIST_FILE" .txt)"
SUBSET_HASH="$(
  awk -v n="$OVERFIT_N" 'NF { print; c++; if (c >= n) exit }' "$LIST_FILE" | sha1sum | cut -c1-8
)"
SUBSET_ROOT="${SUBSET_ROOT:-$REPO_ROOT/waymo/data/overfit_subsets/${LIST_STEM}_n${OVERFIT_N}_${SUBSET_HASH}}"
mkdir -p "$SUBSET_ROOT/train" "$SUBSET_ROOT/val"

MANIFEST="$SUBSET_ROOT/manifest.txt"
: > "$MANIFEST"
count=0
while IFS= read -r npz; do
  [[ -z "$npz" ]] && continue
  count=$((count + 1))
  if [[ "$count" -gt "$OVERFIT_N" ]]; then
    break
  fi
  src="$npz"
  if [[ "$src" != /* ]]; then
    src="$REPO_ROOT/$src"
  fi
  require_file "$src"
  base="$(basename "$src")"
  for split in train val; do
    link="$SUBSET_ROOT/$split/$base"
    if [[ -e "$link" || -L "$link" ]]; then
      current="$(readlink "$link" || true)"
      if [[ "$current" != "$src" ]]; then
        echo "Existing subset link points elsewhere: $link -> $current" >&2
        echo "Expected: $src" >&2
        exit 1
      fi
    else
      ln -s "$src" "$link"
    fi
  done
  echo "$src" >> "$MANIFEST"
done < "$LIST_FILE"

if [[ "$count" -lt "$OVERFIT_N" ]]; then
  echo "List only contains $count non-empty entries, but OVERFIT_N=$OVERFIT_N" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-waymo_overfit_${VARIANT}_n${OVERFIT_N}_chunk${TIME_WINDOW}_raw_${SUBSET_HASH}}"
SESSION_NAME="${SESSION_NAME:-${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/waymo/checkpoints/overfit/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/overfit}"
LOG="${LOG:-$LOG_DIR/${RUN_NAME}.log}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/waymo/wandb}"

cd "$REPO_ROOT"
mkdir -p "$CKPT_DIR" "$LOG_DIR" "$WANDB_DIR"

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export WANDB_MODE
export WANDB_DIR

train_args=(
  waymo/training/train_waymo_vector_tokenizer.py
  --data_dir "$SUBSET_ROOT/train"
  --val_data_dir "$SUBSET_ROOT/val"
  --ckpt_dir "$CKPT_DIR"
  --seed "$SEED"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --time_window "$TIME_WINDOW"
  --random_time_window_start
  --epochs "$EPOCHS"
  --max_steps "$MAX_STEPS"
  --d_model "$D_MODEL"
  --depth "$DEPTH"
  --decoder_depth "$DECODER_DEPTH"
  --n_latents "$N_LATENTS"
  --d_bottleneck "$D_BOTTLENECK"
  --encoder_variant "$ENCODER_VARIANT"
  --map_depth "$MAP_DEPTH"
  --map_cross_every "$MAP_CROSS_EVERY"
  --map_query_tokens "$MAP_QUERY_TOKENS"
  --bottleneck_output "$BOTTLENECK_OUTPUT"
  --agent_xy_loss "$AGENT_XY_LOSS"
  --agent_xy_parameterization "$AGENT_XY_PARAMETERIZATION"
  --agent_delta_xy_weight "$AGENT_DELTA_XY_WEIGHT"
  --agent_fde_xy_weight "$AGENT_FDE_XY_WEIGHT"
  --agent_kinematic_xy_weight "$AGENT_KINEMATIC_XY_WEIGHT"
  --agent_speed_yaw_kinematic_weight "$AGENT_SPEED_YAW_KINEMATIC_WEIGHT"
  --kinematic_dt "$KINEMATIC_DT"
  --focus_agent_weight "$FOCUS_AGENT_WEIGHT"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --log_every "$LOG_EVERY"
  --eval_every "$EVAL_EVERY"
  --save_every "$SAVE_EVERY"
  --wandb_project waymo-vector-tokenizer
  --wandb_run_name "$RUN_NAME"
)

if [[ "$NO_AMP" == "1" || "$NO_AMP" == "true" || "$NO_AMP" == "TRUE" ]]; then
  train_args+=(--no_amp)
fi
if [[ "$USE_WANDB" == "1" || "$USE_WANDB" == "true" || "$USE_WANDB" == "TRUE" ]]; then
  train_args+=(--wandb)
fi
if [[ "$DECODER_USE_AGENT_TOKENS" == "1" || "$DECODER_USE_AGENT_TOKENS" == "true" || "$DECODER_USE_AGENT_TOKENS" == "TRUE" ]]; then
  train_args+=(--decoder_use_agent_tokens)
fi
if [[ "$RESUME_IF_EXISTS" == "1" && -f "$CKPT_DIR/latest.pt" ]]; then
  train_args+=(--resume "$CKPT_DIR/latest.pt")
fi

{
  echo
  echo "===== $(date) ====="
  echo "run_name=$RUN_NAME"
  echo "variant=$VARIANT overfit_n=$OVERFIT_N subset_hash=$SUBSET_HASH"
  echo "subset_root=$SUBSET_ROOT"
  echo "manifest=$MANIFEST"
  echo "ckpt_dir=$CKPT_DIR"
  echo "log=$LOG"
  echo "cuda=$CUDA_VISIBLE_DEVICES nproc_per_node=$NPROC_PER_NODE wandb_mode=$WANDB_MODE use_wandb=$USE_WANDB dry_run=$DRY_RUN"
  echo "batch_size=$BATCH_SIZE epochs=$EPOCHS max_steps=$MAX_STEPS d_model=$D_MODEL depth=$DEPTH decoder_depth=$DECODER_DEPTH n_latents=$N_LATENTS d_bottleneck=$D_BOTTLENECK time_window=$TIME_WINDOW random_time_window_start=1"
  echo "encoder_variant=$ENCODER_VARIANT map_depth=$MAP_DEPTH map_cross_every=$MAP_CROSS_EVERY map_query_tokens=$MAP_QUERY_TOKENS"
  echo "bottleneck_output=$BOTTLENECK_OUTPUT decoder_use_agent_tokens=$DECODER_USE_AGENT_TOKENS"
  echo "agent_xy_loss=$AGENT_XY_LOSS agent_xy_parameterization=$AGENT_XY_PARAMETERIZATION agent_delta_xy_weight=$AGENT_DELTA_XY_WEIGHT agent_fde_xy_weight=$AGENT_FDE_XY_WEIGHT agent_kinematic_xy_weight=$AGENT_KINEMATIC_XY_WEIGHT agent_speed_yaw_kinematic_weight=$AGENT_SPEED_YAW_KINEMATIC_WEIGHT kinematic_dt=$KINEMATIC_DT focus_agent_weight=$FOCUS_AGENT_WEIGHT"
  echo "lr=$LR weight_decay=$WEIGHT_DECAY grad_clip=$GRAD_CLIP no_amp=$NO_AMP"
  echo "selected_scenes:"
  sed 's/^/  /' "$MANIFEST"
  echo "========================"
} | tee -a "$LOG"

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "TRUE" ]]; then
  printf "Dry-run command:\n  "
  printf "%q " "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "${train_args[@]}"
  printf "\n"
  exit 0
fi

"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "${train_args[@]}" \
  2>&1 | tee -a "$LOG"
