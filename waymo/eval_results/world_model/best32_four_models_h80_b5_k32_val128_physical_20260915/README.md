# Best32 four-model evaluation, 2026-09-15

Fixed best32 EMA snapshots in `waymo/checkpoints/selected_best32_meanade_20260915`.
CUDA1 queue: MoN GT step5000, MoN Physical step2500.
CUDA3 queue: ERD beta1 step4750, ERD beta.99 step5750.

All: first512 validation scenes, batch4 x128, K32, context11/H15/B5, rollout80,
Euler solver8, seed12346, generate_all. Sample seeds exactly follow the existing
multisample evaluator. Best checkpoints were selected on the small32-scene subset;
this is validation comparison, not an independent held-out test claim.

Each result.json contains mean_ade_m, minade_m, flow_erd_cpd, physical_metrics,
ground_truth_physical_metrics and detailed physical_metric_definitions.
Physical rates use total numerator / total valid denominator across all samples;
null means no valid observations, never an artificial zero. No physical metric
selects the best candidate. Proxy footprints and sampled road-edge geometry are
not official WOSAC metrics. Report raw road rates, GT/map-reliable rates and
coverage together. All methods use the same data-derived reliable-road masks.

Each rollout_batches/batch_NNNNN.npz atomically preserves sampled trajectories,
GT poses, initial poses, validity, types, maps, seeds, ADE/CPD and physical stats.
physical_metrics.npz preserves per-scene/per-rollout numerators and denominators.
summary.csv is updated after each completed model; remaining models run serially
within their GPU queue. Existing partial outputs are never silently overwritten.
