# Type-aware H40 comparison, 512 scenes / 32 rollouts / 80 steps

Both models use step 95000 EMA, L11 context, H40 plans, B5 execution,
solver 8, seed 12346, 128 batches x 4 scenes, generate_all.
Scene filenames/order are verified identical to the 20260917 H40 evaluation.
Input is the completed with_lengths validation dataset.
Execution uses checkpoint type-aware kinematics; vx/vy model updates generated
history velocities from scene-frame displacement / 0.1 s.
Scoring uses the old holonomic physical GT validity mask (5 m, 0.75 rad limits),
not the extra type-aware reexecution rejection. This matches the previous metrics.
CPD scales are unchanged: 25.423524547767016, 4.925798640017075, 18.77286026475977.

Each model saves result.json (minade_m, mean_ade_m, flow_erd_cpd,
first40_mean_ade_m), CSV details, details.npz and full rollout_batches.
First40 ADE is the first 40 steps of the same B5 rolling 80-step trajectories,
with the same valid-point weighting within scenes and equal scene/rollout averaging.
It is not a separately generated open-loop H40 evaluation.
Physical diagnostics are also retained, matching the previous evaluation.

Both jobs run sequentially on physical CUDA 0 in tmux; each gets a 4-scene,
2-rollout, full-80-step smoke test before the full jobs begin. Smoke outputs are
separate. See queue_status.json, queue.log, per-model run.log and summary.json.
