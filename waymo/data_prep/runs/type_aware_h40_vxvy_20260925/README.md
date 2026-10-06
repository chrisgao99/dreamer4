# Stage-one H40 type-aware model with vx/vy history inputs

CUDA 0; fresh random initialization. The baseline is
`waymo_daf_h40_b5_typeaware_scratch100k_20260924` on CUDA 1.
Only the history state encoder input changes from [x/100,y/100,sin(yaw),cos(yaw)]
to [x/100,y/100,sin(yaw),cos(yaw),vx/10,vy/10]. Velocities are in the
same fixed scene coordinate frame as the existing NPZ positions, in m/s.
The original data already contain vx/vy in feature columns 3 and 4.
All baseline training arguments remain identical except names/output paths and
new velocity-input options. Uses the same completed enriched dataset, calibration,
and action statistics without repeating or modifying data preparation.
At receding evaluation, new history velocities use executed displacement / 0.1 s.
The baseline encoder remains the default for old configurations/checkpoints.

See launch_config.json, resolved_train_args.json, config_comparison.json,
preflight.json, status.json and run.log. W&B is offline as for the baseline.
