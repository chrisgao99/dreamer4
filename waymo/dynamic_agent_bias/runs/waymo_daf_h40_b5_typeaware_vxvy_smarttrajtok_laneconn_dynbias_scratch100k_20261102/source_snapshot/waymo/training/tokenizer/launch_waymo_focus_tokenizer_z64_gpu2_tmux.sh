#!/usr/bin/env bash
# Launch the focus-only 1x64 latent tokenizer on physical CUDA device 2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WORKER="$SCRIPT_DIR/run_waymo_focus_tokenizer_single.sh"

SESSION="${SESSION:-focus_tok_c_z64}"
GPU="${GPU:-2}"
RUN_NAME="${RUN_NAME:-focus_tokenizer_c_z1x64_raw_map_lr1e4}"
LR="${LR:-1e-4}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/focus_tokenizer}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/${SESSION}.log}"

if [[ ! -x "$WORKER" ]]; then
  echo "Worker is missing or not executable: $WORKER" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

tmux new-session -d -s "$SESSION" \
  "set -o pipefail; cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES='$GPU' REPO_ROOT='$REPO_ROOT' RUN_NAME='$RUN_NAME' D_LATENT=64 LR='$LR' bash '$WORKER' latent_z64 2>&1 | tee '$LOG_PATH'"

echo "started session=$SESSION gpu=$GPU representation=latent_z64 d_latent=64 lr=$LR"
echo "run_name=$RUN_NAME"
echo "log=$LOG_PATH"
echo "attach: tmux attach -t $SESSION"
echo "tail:   tail -f '$LOG_PATH'"
