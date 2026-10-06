# Dynamic/static 70/30 evaluation, 2026-09-18

Models: H40 step95000 EMA and H15 step90000 EMA, selected by balanced validation flow loss.
Common settings: first512 validation scenes (verified equal to baseline manifest),
batch4 x128, K32, initial context11, total rollout80, solver8, seed12346,
generate_all, identical CPD scales and physical metrics as the original evaluations.
Generated history is fed back after each executed segment.

GPU0: h40_b40 then h15_b15. GPU1: h40_b15. GPU2: h40_b5. GPU3: h15_b5.
Launcher: waymo/evaluation/launch_eval_dynamic7030_tmux.sh.
Each run saves result.json, per-scene/per-rollout CSV, full rollout batches,
physical metrics and ade_by_gt_motion.json. Motion grouping uses GT net
movement <2m, 2-20m, >20m; primary motion summary requires all80 valid steps
and weights each agent equally, unlike the scene-equal overall ADE.
summary.csv is regenerated under a lock after each completed evaluation and
motion analysis. Physical metrics are proxy diagnostics, not official WOSAC.
