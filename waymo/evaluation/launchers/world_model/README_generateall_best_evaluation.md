# Generate-all best.pt: independent B1 / B5 / B15 evaluations

These three standalone Bash launchers each create a detached tmux session:

- `run_waymo_direct_action_flow_generateall_best_ctx11_h80_b1_k32_val128_cpd_tmux.sh`
- `run_waymo_direct_action_flow_generateall_best_ctx11_h80_b5_k32_val128_cpd_tmux.sh`
- `run_waymo_direct_action_flow_generateall_best_ctx11_h80_b15_k32_val128_cpd_tmux.sh`

All use the EMA weights in
`waymo/checkpoints/waymo_direct_action_flow_v1_h15_b5_generateall_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5/best.pt`.
The checked local file is step 75000, epoch 27, with `condition_focus_actions=False`.

Each model call generates a 15-step action plan. B1 / B5 / B15 executes the
first 1 / 5 / 15 steps before replanning, matching the previous ablation.
All runs use context 11, total rollout 80, solver steps 8, 128 batches of 4
validation scenes (512 total), 32 joint rollouts per scene, and seed 12346.
The validation files and their sorted order must match on both servers.
CPD uses the same vehicle/pedestrian/cyclist scales as the previous evaluation.

## Copying to the other server

Preserve paths relative to the `dreamer4` directory. Copy the three launchers
**and the updated `waymo/evaluation/eval_waymo_direct_action_flow_multisample.py`**.
Copy this README as well, or extract the provided `direct_action_flow_generateall_best_launchers.tar.gz`
from within the destination `dreamer4` directory. The archive contains those five
files, without data or checkpoint weights. The destination already needs the
generate-all training/model code, dataset, checkpoint, tmux, and dreamer4 Python environment.

The updated evaluator is necessary: the old version always supplied logged
focus actions and excluded focus from scoring. The new launchers explicitly
assert `--focus_mode generate_all` and reject a conditioned checkpoint.

No `/p/yufeng` or `/sfs/weka` prefix is hardcoded. The repository root is inferred
from each launcher's location. Python is selected from the active Conda environment,
the repository owner's `.conda/envs/dreamer4`, or the user's Conda installations.
Set `PYTHON=/absolute/path/to/dreamer4/bin/python` to override selection.

## Launch independently on either server

From the destination `dreamer4` directory, choose any of:

```bash
CUDA_DEVICE=0 bash waymo/evaluation/launchers/world_model/run_waymo_direct_action_flow_generateall_best_ctx11_h80_b1_k32_val128_cpd_tmux.sh
CUDA_DEVICE=0 bash waymo/evaluation/launchers/world_model/run_waymo_direct_action_flow_generateall_best_ctx11_h80_b5_k32_val128_cpd_tmux.sh
CUDA_DEVICE=0 bash waymo/evaluation/launchers/world_model/run_waymo_direct_action_flow_generateall_best_ctx11_h80_b15_k32_val128_cpd_tmux.sh
```

Assign each command to your chosen server/GPU. Use different `CUDA_DEVICE`
values when running concurrently on different GPUs on one server.
Each command returns immediately after starting its tmux session.
Scripts also work when invoked by their absolute path from another directory.

Append `--dry-run` to validate paths, Python imports, and the evaluation command
without creating a tmux session or running inference. For example:

```bash
bash waymo/evaluation/launchers/world_model/run_waymo_direct_action_flow_generateall_best_ctx11_h80_b1_k32_val128_cpd_tmux.sh --dry-run
tmux attach -t wm_daf_generateall_best_h80_b1_k32_cpd
```

Session names substitute `b5` or `b15` for the other jobs. Completed and failed
panes remain available for inspection. Existing sessions and outputs are rejected.
For an intentional repeat, set both a fresh `RUN_NAME` and `SESSION_NAME`.
Other supported overrides: `REPO_ROOT`, `VAL_DATA`, `EVAL_CKPT`, `OUT_DIR`,
`LOG_FILE`, and `OMP_NUM_THREADS`.

## Outputs and metric scope

For B=1, outputs go to
`waymo/eval_results/world_model/waymo_direct_action_flow_generateall_best_ctx11_h80_b1_k32_val128_cpd/`;
the log is
`waymo/logs/evaluation/waymo_direct_action_flow_generateall_best_ctx11_h80_b1_k32_val128_cpd.log`.
B5 and B15 use their respective names.

- `result.json`: mean ADE, scene-level joint minADE@32, per-agent minADE,
  CPD, checkpoint step, mode, protocol, and elapsed time.
- `ade_by_agent_scope` in `result.json`: separate all/nonfocus/focus mean ADE
  and joint minADE, with valid scene/point counts. Empty scopes return null.
- `rollout_ade.csv`: every scene/rollout ADE.
- `scene_metrics.csv`: per-scene ADE/minADE/CPD.
- `details.npz`: candidate scene/agent ADE, valid step counts, identifiers,
  CPD components, and focus-conditioning metadata.

Top-level ADE and CPD include focus; no future focus actions condition the model.
Use `ade_by_agent_scope.nonfocus` for ADE comparison with the previous conditioned
model. CPD here includes focus, whereas the previous experiment's CPD excluded it.
Scope-specific minADE chooses the best rollout separately for that scope.

## Verification

All three launchers passed Bash syntax checks and real-environment dry runs.
Relocated-launcher checks exercised paths containing spaces, tmux command construction,
duplicate-output rejection, and failure propagation with a simulated tmux executable.
19 model/metric/evaluator tests passed. Each commitment also completed a CPU smoke
evaluation using the actual 75k best.pt: one scene, two 16-step rollouts, eight solver
steps. The smoke checks verified that both future-focus arguments were None and
focus predictions were included in metrics. Full CUDA evaluations were not launched.
