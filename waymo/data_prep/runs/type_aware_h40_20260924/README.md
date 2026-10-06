# Fresh H40 type-aware stage-one run, CUDA 1

Requested from random initialization, not resumed from the old holonomic model.
Run name: `waymo_daf_h40_b5_typeaware_scratch100k_20260924`.

`run.sh` starts `run_typeaware_stage1.py` from `launch_config.json` in a detached
tmux session. `status.json` records the current stage and failures. `run.log`
contains all preparation and training output. `exit_code` appears when the whole
pipeline exits. Offline W&B logging avoids network/authentication blocking.

Stages:
1. Supplement missing static lengths in the new enriched NPZs. Keep positive
   current-frame lengths; otherwise use the track's first valid positive raw
   measurement. This was explicitly selected by the user. Do not drop tracks.
   Original source NPZs remain read-only; only the new `agent_lengths` field is
   replaced, preserving original compressed feature contents.
2. Calibrate signed no-slip ratios using 8,192 deterministically selected
   training files and unique scenario/track turning intervals. No validation
   data are used for rho or normalization statistics.
3. Compute recursive H40 target moments at all training anchors from the same
   sized training subset. Anchors are batched to accelerate this without changing
   target semantics. `resolved_train_args.json` stores the calibrated settings.
4. CPU preflight: inverse/re-execution checks on 32 train + 32 val samples, and
   real H40-network forward/backward finite checks. Saves `preflight.json`.
5. Verify physical CUDA 1 is still free, bind its UUID, and start fresh training.
   If another job has occupied it, fail instead of interfering.

Matched baseline: L11/H40/B5, d256/8 heads, encoder depths 2/2/4, DiT depth 8,
step refiner depth 2, all-agent generation, batch 4 x accumulation 4, 100k updates,
LR 5e-5, warmup 5k, decay schedule 500k, BF16, Huber flow loss, flow time <=0.9,
normalized action clip 5, displacement <=5 m and yaw increment <=0.75 rad filters.
No dynamic/static 70/30 weighting. Re-execution error bound is 0.25 m (a declared
implementation choice, not a published paper hyperparameter). Vehicles/cyclists
use the paper's non-holonomic displacement formula; pedestrians remain holonomic.

Checkpoints: `waymo/checkpoints/waymo_daf_h40_b5_typeaware_scratch100k_20260924/`.
Calibration and action-statistics JSONs live in the enriched dataset root with
new names; old holonomic statistics are not reused.

Verification before submission: 20 CPU tests passed covering enrichment,
supplementation, type-aware execution and preparation. Real-data supplementation
smoke-test details are in `supplement_smoke.json`.

Length supplementation fix (2026-09-25): the launcher passes history_length=11.
Tracks with no positive measured length are retained and audited only if they
are inactive at every frame from index 10 onward. Later active tracks still
fail strictly. No absolute-value/default dimensions or track deletion is used.
This conservative rule also covers later rollout frames beyond stage-one anchors.
`length_fix_smoke.json` records verification on a copy of the failing real sample.
The journal records `unresolved_inactive_tracks` and the summary counts them.
