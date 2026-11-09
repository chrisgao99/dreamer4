#!/usr/bin/env bash
# B=1 commitment ablation for the 90k DirectActionFlow checkpoint.

set -euo pipefail

LAUNCHER="/p/yufeng/tri30/dreamer4/waymo/evaluation/launchers/world_model/run_waymo_direct_action_flow_step90k_ctx11_h80_k32_val128_cpd_tmux.sh"
[[ -f "$LAUNCHER" ]] || { echo "Missing shared launcher: $LAUNCHER" >&2; exit 1; }

export COMMITMENT_STEPS=1
export RUN_NAME="${RUN_NAME:-waymo_direct_action_flow_step90k_ctx11_h80_b1_k32_val128_cpd}"
export SESSION_NAME="${SESSION_NAME:-wm_daf_90k_h80_b1_k32_cpd}"

exec bash "$LAUNCHER"
