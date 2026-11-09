#!/usr/bin/env bash
# Launch experiments A and B concurrently in detached tmux sessions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
WORKER="$SCRIPT_DIR/run_waymo_focus_tokenizer_single.sh"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/waymo/logs/focus_tokenizer}"
SESSION_A="${SESSION_A:-focus_tok_a_agent}"
SESSION_B="${SESSION_B:-focus_tok_b_z16}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

mkdir -p "$LOG_DIR"

launch_one() {
  local session="$1"
  local gpu="$2"
  local representation="$3"
  local log="$4"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    return 1
  fi

  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES='$gpu' REPO_ROOT='$REPO_ROOT' bash '$WORKER' '$representation' 2>&1 | tee '$log'"
  echo "started session=$session gpu=$gpu representation=$representation log=$log"
}

launch_one "$SESSION_A" "$GPU_A" agent_token "$LOG_DIR/${SESSION_A}.log"
launch_one "$SESSION_B" "$GPU_B" latent_z16 "$LOG_DIR/${SESSION_B}.log"

echo
echo "Both tokenizers are running detached."
echo "Attach A: tmux attach -t $SESSION_A"
echo "Attach B: tmux attach -t $SESSION_B"
echo "List:     tmux ls"
echo "Logs:     tail -f '$LOG_DIR/${SESSION_A}.log'"
echo "          tail -f '$LOG_DIR/${SESSION_B}.log'"
