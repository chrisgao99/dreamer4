# H15 dynamic/static loss comparison (2026-09-16)

From-scratch 100k training on CUDA 0. Matched original H15 generate-all settings:
context11, H15, B5, Huber beta1, tmax .9, normalized target clip5, LR5e-5,
cosine endpoint500k, warmup5k, batch8 x accumulation2, seed0. Unchanged shuffled
file sampling and uniform valid focus-window anchor sampling. No resampling.

Only optional loss aggregation changes: compute mean flow loss per valid agent
across valid time and 3 action coordinates; average agents within each motion
group across the microbatch; mix .7 dynamic + .3 static/low-speed. Empty groups
are omitted and remaining weights renormalized. Gradient accumulation averages
these microbatch objectives. Every supervised agent with >=1 valid transition
participates; no extra focus weighting. Dynamic is mean metric GT displacement
norm / .1s > .5m/s, using contiguous physically valid prefixes BEFORE normalized
clipping. Metric actions use local coordinates, whose norm equals world displacement.

Validation uses the same balanced loss; best.pt is selected by that loss, not
ADE. Original unbalanced loss and both group losses/counts are also logged.
The optional switch defaults off, preserving existing experiments.

Independent CUDA1 audit uses only raw agent data, all train/val files and every
focus-complete H15 anchor (10..75 for 91 frames), with the same longest-prefix
fallback as training. Reports dynamic ratio for >=1 valid transition and robust
>=10 transitions. Per-window gzip CSV and aggregate summary JSON distinguish
window-weighted histograms from scene-uniform means and within-scene variation.
Audit does not change sampling or stop the H40 job already on CUDA1.

## CUDA 3 restart (2026-09-17)

The original run exited with status 1 while appending metrics at step 2280:
its checkpoint directory was no longer present. There is no directory deletion
in the trainer, and the available logs do not establish why it disappeared.
It had not reached the first scheduled checkpoint at step 2500.

Training and validation metric writes now recreate a missing parent directory
and emit a warning; removed historical metrics/checkpoints are not restored.
The directory-removal regression and all four dynamic balance tests passed.

Restarted from scratch with unchanged training settings on physical CUDA 3:
- Run suffix: `_rerun_cuda3_20260917`
- tmux: `wm_daf_h15_dyn70static30_scratch100k_cuda3_20260917`
- W&B: https://wandb.ai/yufenggao/waymo-world-model/runs/0y80ihtp
- Launcher: `CUDA_DEVICE=3`, new `RUN_NAME` and `SESSION_NAME` overrides of
  `launch_waymo_direct_action_flow_h15_dynamic7030_cuda0_tmux.sh`.

Confirmed live GPU process and successful metrics writes after restart.

## H40 dynamic/static comparison (2026-09-17)

Launched on CUDA 2 with
`launch_waymo_direct_action_flow_h40_dynamic7030_cuda2_tmux.sh`.
Matched the original H40 scratch100k launcher: L11/H40/B5, batch4 x
accumulation4, LR5e-5, warmup5k, cosine endpoint500k, seed0, unchanged sampling.
Only the loss aggregation changes to 0.7 dynamic + 0.3 static/low-speed,
using the same >0.5m/s criterion as the H15 balanced experiment.
Fresh initialization, 100k optimizer steps; no checkpoint resume.
Session: `wm_daf_h40_dyn70static30_scratch100k_cuda2`.
Run name: `waymo_direct_action_flow_v1_h40_b5_generateall_dyn70static30_v05_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5`.
Validation/checkpoint interval remains 2500; best is chosen by balanced flow
loss. Follow-up comparison should use the same 512 scenes and report ADE by GT
net-displacement group, static drift, physical metrics and CPD. That evaluation
is not automatically scheduled by this training launcher.
