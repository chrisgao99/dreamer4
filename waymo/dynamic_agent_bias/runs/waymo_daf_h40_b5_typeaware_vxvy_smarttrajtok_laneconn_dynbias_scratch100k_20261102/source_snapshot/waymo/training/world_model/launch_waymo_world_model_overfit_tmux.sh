#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/p/yufeng/tri30/dreamer4}"
RUN_SCRIPT="$REPO_ROOT/waymo/training/world_model/run_waymo_world_model_overfit_one.sh"

experiments=(fixed_empirical fixed_bootstrap eight_random)
sessions=(wm_of_fixed_emp wm_of_fixed_boot wm_of_eight_rand)

for session in "${sessions[@]}"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

for idx in "${!experiments[@]}"; do
  experiment="${experiments[$idx]}"
  session="${sessions[$idx]}"
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && exec env CUDA_DEVICE=1 bash '$RUN_SCRIPT' '$experiment'"
  echo "started $session -> $experiment"
done

echo
echo "Active sessions:"
tmux list-sessions -F '#{session_name} #{session_created_string}' | grep '^wm_of_' || true
echo
echo "Attach with: tmux attach -t wm_of_fixed_emp"
