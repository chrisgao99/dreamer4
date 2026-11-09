# Adapted map MON finetuning

Launch: `VARIANT=mapadapt bash waymo/training/world_model/launch_adapted_mon80_tmux.sh` (CUDA 3) or `VARIANT=laneconn bash waymo/training/world_model/launch_adapted_mon80_tmux.sh` (CUDA 2).

Sources: map adaptation final 100k and connectivity step 90k. Uses source EMA weights, fresh AdamW and EMA finetune state. Matches saved typeaware vxvy baseline MON argv in baseline_mon_argv.json: pure MON-8, 80-step differentiable closed loop, H40/B5, detach_every=0, solver8, batch2, seed20260911, lr1e-6, weight_decay0.01, EMA0.999, 10k updates. Evaluate every500, snapshot every1000, save every100. No extra physical or map losses.

Full map semantic and lane graph overlays are loaded; controls and dynamic agent/lane associations update each replan. Checkpoint recomputation receives current light IDs explicitly, preventing elapsed-time closure changes from altering backward replay. Original raw agent datasets remain unchanged. Existing adapted datasets and topology caches are reused.

Both jobs begin with the same initial validation as the baseline, then train automatically in detached tmux. External comparison uses holonomic 512-scene scoring as documented in waymo/evaluation/EVALUATION_DEFAULTS.md.
