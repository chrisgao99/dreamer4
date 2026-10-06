# H40 step95000 commitment comparison

All three runs use the EMA weights in step_000095000.pt from the H40 scratch100k experiment.
Initial context: 11 ground-truth frames. Each replan generates 40 frames and
updates context with the last 11 frames of the executed generated trajectory.
Total rollout: 80 frames (8 seconds). Commitments: 40 (CUDA0), 15 (CUDA1), 5 (CUDA2).
The final B15 segment executes only the 5 remaining frames.

Exactly the first 512 validation scenes in sorted dataset order, as in the old
H15 val128 evaluation (128 batches x 4 scenes). See scene_manifest.json.
K=32 joint rollouts per scene, solver=8, seed=12346; same batch-index seed formula
and CPD type scales as the previous evaluation. All agents including focus are
generated; no future focus actions are supplied. Maps and logged traffic-light
inputs follow the existing evaluator.

Each b*/result.json contains mean ADE, joint minADE@32, CPD, physical metrics
and ground-truth physical references. Physical metrics aggregate all 32 samples,
not only the best candidate. Metrics include collision, offroad (raw/reliable),
road coverage, lateral speed, sideslip, reverse motion, speed, acceleration,
jerk, yaw rate and displacement/yaw threshold violations. Geometry uses proxy
footprints and road edges, not official WOSAC scoring.

Per-scene CSV, per-rollout CSV, details.npz, physical_metrics.npz and full
rollout_batches are saved. Each completed run updates summary.csv under a lock.

Validation before launch: checkpoint metadata verified (step95000/L11/H40),
launcher shell syntax and summary Python compile passed, physical metrics
standalone regression passed. The broader pytest suite was unavailable because
pytest is not installed in the dreamer4 environment.
