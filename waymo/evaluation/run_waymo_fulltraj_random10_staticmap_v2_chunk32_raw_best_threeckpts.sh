#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WAYMO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$WAYMO_ROOT/.." && pwd)"

cd "$REPO_ROOT"

PYTHON="${PYTHON:-/home/baz7dy/.conda/envs/dreamer4/bin/python}"
TIME_WINDOW="${TIME_WINDOW:-32}"
LIST_FILE="${LIST_FILE:-waymo/evaluation/waymo_fulltraj_random10.txt}"
OUT_DIR="${OUT_DIR:-waymo/evaluation/reports/reconstruction_fulltraj_random10_staticmap_v2_chunk${TIME_WINDOW}_raw_best_threeckpts}"
PANEL_SIZE="${PANEL_SIZE:-900}"
DEVICE="${DEVICE:-}"

RUN_RAW_GMM_KIN_FDE="ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_gmm_kinematic_fdeloss_focus_randstart_noamp"
RUN_RAW_KIN_FDE="ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_fdeloss_focus_randstart_noamp"
RUN_RAW_KIN_NOFDE="ooi50k_lat16_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp"

LOG_RAW_GMM_KIN_FDE="waymo/logs/${RUN_RAW_GMM_KIN_FDE}.log"
LOG_RAW_KIN_FDE="waymo/logs/${RUN_RAW_KIN_FDE}.log"
LOG_RAW_KIN_NOFDE="waymo/logs/${RUN_RAW_KIN_NOFDE}.log"

CKPT_DIR_RAW_GMM_KIN_FDE="waymo/checkpoints/${RUN_RAW_GMM_KIN_FDE}"
CKPT_DIR_RAW_KIN_FDE="waymo/checkpoints/${RUN_RAW_KIN_FDE}"
CKPT_DIR_RAW_KIN_NOFDE="waymo/checkpoints/${RUN_RAW_KIN_NOFDE}"

best_eval_line() {
  local log_path="$1"
  awk '
    BEGIN { best = 1e99 }
    /^eval step=/ {
      step = loss = xy = fde = fxy = ffde = ""
      if (match($0, /step=[0-9]+/)) step = substr($0, RSTART + 5, RLENGTH - 5) + 0
      if (match($0, /loss_total=[-0-9.]+/)) loss = substr($0, RSTART + 11, RLENGTH - 11) + 0
      if (match($0, /agent_xy_mae_m=[-0-9.]+/)) xy = substr($0, RSTART + 15, RLENGTH - 15) + 0
      if (match($0, /agent_fde_mae_m=[-0-9.]+/)) fde = substr($0, RSTART + 16, RLENGTH - 16) + 0
      if (match($0, /focus_agent_xy_mae_m=[-0-9.]+/)) fxy = substr($0, RSTART + 21, RLENGTH - 21) + 0
      if (match($0, /focus_agent_fde_m=[-0-9.]+/)) ffde = substr($0, RSTART + 18, RLENGTH - 18) + 0
      if (loss < best) {
        best = loss
        bestline = sprintf("step=%d loss_total=%.4f agent_xy_mae_m=%.4f agent_fde_mae_m=%.4f focus_agent_xy_mae_m=%.4f focus_agent_fde_m=%.4f", step, loss, xy, fde, fxy, ffde)
      }
    }
    END {
      if (bestline == "") {
        exit 1
      }
      print bestline
    }
  ' "$log_path"
}

step_from_best_line() {
  local line="$1"
  local step="${line#step=}"
  step="${step%% *}"
  printf "%08d" "$step"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_file "$PYTHON"
require_file "$LIST_FILE"
require_file "$LOG_RAW_GMM_KIN_FDE"
require_file "$LOG_RAW_KIN_FDE"
require_file "$LOG_RAW_KIN_NOFDE"

BEST_RAW_GMM_KIN_FDE="$(best_eval_line "$LOG_RAW_GMM_KIN_FDE")"
BEST_RAW_KIN_FDE="$(best_eval_line "$LOG_RAW_KIN_FDE")"
BEST_RAW_KIN_NOFDE="$(best_eval_line "$LOG_RAW_KIN_NOFDE")"

STEP_RAW_GMM_KIN_FDE="$(step_from_best_line "$BEST_RAW_GMM_KIN_FDE")"
STEP_RAW_KIN_FDE="$(step_from_best_line "$BEST_RAW_KIN_FDE")"
STEP_RAW_KIN_NOFDE="$(step_from_best_line "$BEST_RAW_KIN_NOFDE")"

CKPT_RAW_GMM_KIN_FDE="${CKPT_DIR_RAW_GMM_KIN_FDE}/step_${STEP_RAW_GMM_KIN_FDE}.pt"
CKPT_RAW_KIN_FDE="${CKPT_DIR_RAW_KIN_FDE}/step_${STEP_RAW_KIN_FDE}.pt"
CKPT_RAW_KIN_NOFDE="${CKPT_DIR_RAW_KIN_NOFDE}/step_${STEP_RAW_KIN_NOFDE}.pt"

require_file "$CKPT_RAW_GMM_KIN_FDE"
require_file "$CKPT_RAW_KIN_FDE"
require_file "$CKPT_RAW_KIN_NOFDE"

LABEL_RAW_GMM_KIN_FDE="raw_gmm_kin_fde_s$((10#$STEP_RAW_GMM_KIN_FDE))"
LABEL_RAW_KIN_FDE="raw_kin_fde_s$((10#$STEP_RAW_KIN_FDE))"
LABEL_RAW_KIN_NOFDE="raw_kin_nofde_s$((10#$STEP_RAW_KIN_NOFDE))"

mkdir -p "$OUT_DIR"

{
  echo "Selection rule: minimum eval loss_total in each log."
  echo "Selected at: $(date)"
  echo
  echo "$RUN_RAW_GMM_KIN_FDE"
  echo "  log:        $REPO_ROOT/$LOG_RAW_GMM_KIN_FDE"
  echo "  best_eval:  $BEST_RAW_GMM_KIN_FDE"
  echo "  checkpoint: $REPO_ROOT/$CKPT_RAW_GMM_KIN_FDE"
  echo
  echo "$RUN_RAW_KIN_FDE"
  echo "  log:        $REPO_ROOT/$LOG_RAW_KIN_FDE"
  echo "  best_eval:  $BEST_RAW_KIN_FDE"
  echo "  checkpoint: $REPO_ROOT/$CKPT_RAW_KIN_FDE"
  echo
  echo "$RUN_RAW_KIN_NOFDE"
  echo "  log:        $REPO_ROOT/$LOG_RAW_KIN_NOFDE"
  echo "  best_eval:  $BEST_RAW_KIN_NOFDE"
  echo "  checkpoint: $REPO_ROOT/$CKPT_RAW_KIN_NOFDE"
} > "$OUT_DIR/best_checkpoints.txt"

echo "Repo:       $REPO_ROOT"
echo "List file:  $REPO_ROOT/$LIST_FILE"
echo "Output dir: $REPO_ROOT/$OUT_DIR"
echo "Mode:       chunked_full_trajectory, chunk_window=$TIME_WINDOW"
echo
echo "Selected checkpoints:"
cat "$OUT_DIR/best_checkpoints.txt"
echo
echo "Selected inputs:"
cat "$LIST_FILE"
echo

while IFS= read -r npz; do
  [[ -z "$npz" ]] && continue
  args=(
    waymo/evaluation/visualize_vector_tokenizer_reconstruction.py
    --npz "$npz"
    --checkpoint
      "$CKPT_RAW_GMM_KIN_FDE"
      "$CKPT_RAW_KIN_FDE"
      "$CKPT_RAW_KIN_NOFDE"
    --label
      "$LABEL_RAW_GMM_KIN_FDE"
      "$LABEL_RAW_KIN_FDE"
      "$LABEL_RAW_KIN_NOFDE"
    --chunked_full_trajectory
    --time_window "$TIME_WINDOW"
    --split_panels
    --output_dir "$OUT_DIR"
    --panel_size "$PANEL_SIZE"
  )
  if [[ -n "$DEVICE" ]]; then
    args+=(--device "$DEVICE")
  fi
  "$PYTHON" "${args[@]}"
done < "$LIST_FILE"

echo
echo "Done."
echo "Input list: $REPO_ROOT/$LIST_FILE"
echo "Output:     $REPO_ROOT/$OUT_DIR"
