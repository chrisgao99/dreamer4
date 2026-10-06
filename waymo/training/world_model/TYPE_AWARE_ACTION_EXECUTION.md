# Flow-ERD type-aware action execution

The original implementation was prepared without launching experiments.
A fresh H40 first-stage run is now configured in
`waymo/data_prep/runs/type_aware_h40_20260924/launch_config.json`. Its background
pipeline supplements missing measured dimensions, calibrates rho on training
data, computes fresh action statistics, runs a preflight, then trains on CUDA 1.

## Implemented equations

`action_kinematics.py` implements Flow-ERD §III-B / §IV-A, Eq. (2), (10)--(14).
Waymo type 1 (vehicle) and type 3 (cyclist) use non-holonomic execution. Type 2
(pedestrian), and other types, retain the existing holonomic execution.
For wheeled agents, with `r = rho[type] * measured_length`, `q = a_yaw/2`, and
`mid = yaw + q`:

```
forward = a_long * sinc(q)       # unnormalized sinc; torch.sinc(q/pi)
swing   = 2 * r * sin(q)
delta_x = forward * cos(mid) - swing * sin(mid)
delta_y = forward * sin(mid) + swing * cos(mid)
yaw_new = wrap(yaw + a_yaw)
```

The lateral action channel is ignored during wheeled execution. Straight motion
has a stable, differentiable zero-turn limit. No encoder/DiT/parameter shapes
change. This implements the action model, not the paper's full ERD method.

The inverse projects displacement onto the midpoint forward direction and divides
by `sinc(q)`. Subsequent inverse targets start at the **executed** pose, rather
than resetting to GT. Arbitrary recorded motion need not lie on this kinematic
manifold: reject targets whose position re-execution error exceeds the configured
threshold, and invalidate the rest of that track's target window. Preserve GT
poses for supervision/metrics. Invalid targets are masked, not clipped into
apparently valid transitions.

## Explicit choices where the paper is underspecified

- Unused wheeled lateral target is zero, with the existing three-channel flow
  loss retained. The paper does not prescribe that channel's target/loss.
- Re-execution position error threshold defaults to 0.25 m; configurable via
  `--max_reexecution_error_m`. This is not a claimed paper hyperparameter.
- Estimate signed per-type rho with the median of Eq. (12) over training turning
  intervals. Defaults: |delta yaw| in [0.02, 0.75] rad, displacement <=5 m,
  at least 100 intervals per type. These are configurable choices. Calibration
  deduplicates scene/track copies from multiple OOI-centered samples.
- Use measured box length as a fixed track attribute. Keep positive original-
  current-frame lengths. For tracks missing that measurement, the authorized
  supplementary pass reads the first valid positive length in the original
  track (past/current/future); it does not synthesize a size. This supports later
  random training anchors when previously absent tracks become active.
  Any still-missing active vehicle/cyclist lengths fail explicitly.
- Target validity can differ from holonomic runs. Existing target-masked metrics
  therefore are not automatically an apples-to-apples comparison; compare on a
  common evaluation mask and report coverage for a future experiment.

## Data and calibration preparation (commands not executed)

The existing converter `waymo/core/waymo_vector_filter.py` now includes
`agent_lengths` in selected-agent order, read from raw `state/current/length`.
The dataset loader exposes it as a separate tensor without changing the 8-D
agent features. Old NPZ files still load for holonomic runs.

The sampled current training NPZ lacks lengths. Regenerate the same selected
train/validation split into **new directories** with the updated data-prep
pipeline. Keep manifests/splits identical; do not mix enriched and old NPZ files
in one dataset. The training split alone supplies calibration and action moments.

Example calibration after enriched data are available:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python \
  waymo/training/world_model/prepare_action_kinematics.py \
  --train_data /path/to/enriched/train \
  --output /path/to/new/kinematics_calibration.json
```

This tool is CPU-only and refuses to overwrite its output. It does not train a
model. The rho values/counts and preparation thresholds are saved in JSON.

## Enabling a future run

Keep the existing network settings and add these arguments to the stage-one
trainer, with fresh checkpoint and action-statistics paths:

```
--action_execution type_aware
--kinematics_calibration /path/to/new/kinematics_calibration.json
--max_reexecution_error_m 0.25
--action_stats_path /path/to/new/type_aware_action_stats.json
```

The new `run_typeaware_stage1.py` runs the preparation/training pipeline from
an explicit JSON configuration. Trainer defaults remain holonomic. Type-aware
action statistics use recursive H-step targets over training anchors, and carry
execution, horizon, history, and filtering metadata. They must be recomputed;
existing holonomic statistics are rejected. Calibration values are embedded in
checkpoint `args.action_kinematics`, so evaluation does not require the original
calibration JSON. Resuming with changed execution semantics is rejected even
though neural weight shapes happen to match.

Stage-one preparation/validation, receding-horizon rollout, multisample and
receding evaluators, the two-agent visualization, differentiable MoN/ERD rollout
helpers, and recovery targets/sampling all dispatch through the same executor.
Current stage-two scripts inherit the new mode from a **new type-aware**
checkpoint and retain their pre-existing 95k H40/B5 experiment restrictions;
they do not switch the existing holonomic 95k weights to new semantics.

## Verification

CPU tests cover the closed-form turn equation, independent lateral control for
pedestrians only, the zero-turn gradient, inverse/re-execution consistency,
recursive error rejection, missing lengths, masked NaNs, signed rho recovery,
old-mode numerical compatibility, checkpoint/statistics mismatch rejection,
metadata persistence, measured-length indexing, synthetic NPZ calibration and
statistics, real-network flow-loss backward, recovery targets, and agreement
between ordinary and differentiable closed-loop rollout.

Validation performed: 47 CPU tests passed (new kinematics tests plus existing
flow, MoN, recovery, ERD, multisample-evaluation and dynamic-balance tests).
Seven training/preparation/evaluation/visualization CLI `--help` entrypoints
passed; changed Python modules parsed successfully and `git diff --check` passed.
Tests used the existing dreamer4 Python/Torch environment and the existing
pytest package from the dm3env site-packages, with plugin autoload disabled;
no packages were installed or training environments modified.
