# minADE checkpoint visualizations

Same 10 validation intersection scenes and plotting configuration as best32_four_models_intersection10_n10_h80_b5_20260916. These scenes come from the validation split.

EMA; seed=20260907; 10 rollouts/scene; horizon=80; commitment=5; solver_steps=8; generate_all.

| Experiment | Step | Validation minADE (m) | Selection |
|---|---:|---:|---|
| cuda0_catk_step1000 | 1000 | 1.668216 | Best retained checkpoint; historical step 500 (1.658257 m) was overwritten |
| cuda1_mon_detach0_step2500 | 2500 | 1.613926 | Historical minimum |
| cuda2_mon_detach5_step3000 | 3000 | 1.515572 | Historical minimum |
| cuda3_mon_detach20_step2500 | 2500 | 1.500505 | Historical minimum |

Per-model *_selection.json records original checkpoint paths and validation metrics. The .pt files are frozen EMA inference checkpoints.

Plotting fix: images_only skips unused reference-bound calculations after old reference trajectory NPZ files were removed. Rendering and rollout settings are unchanged.
