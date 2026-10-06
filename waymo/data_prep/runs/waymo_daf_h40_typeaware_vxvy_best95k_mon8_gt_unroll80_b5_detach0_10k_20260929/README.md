# Type-aware + vx/vy, MoN-GT full80 fine-tuning

Initialize model from EMA in the user-specified best.pt (stage-one step 95000).
Use the same enriched with_lengths dataset and checkpoint action normalization.
Pure MoN-8 GT loss: smooth-L1 xy + 0.5 velocity + 0.5 sin/cos yaw, joint winner
per scene. L11 / plan H40 / execute B5 repeated 16 times, 80 simulated steps.
No context/pose/velocity detach. Differentiable generated vx/vy are displacement
in the scene frame divided by 0.1 s. No GT state resets within the rollout.
Selection is no-grad; winning candidates are exactly replayed with gradients.
Activation checkpointing preserves RNG and the complete gradient chain.
Dropout disabled via model.eval(), parameters still optimized (as prior MoN-GT).
Type-aware GT validity/reexecution filtering is retained from the checkpoint.
LR 1e-6; batch 2; K8; solver8; 10k updates; EMA .999; BF16; clip grad norm 1.
Optimizer starts fresh. No physical penalty or auxiliary flow loss.
Initial and every-500-update validation: 32 scenes x 4 rollouts. Latest saved
at 100 updates, snapshots at 1000, best_ade.pt by validation mean ADE.
A separate one-update full80 smoke test must pass before fresh main training.
Tests: 20 CPU regression tests passed, including type-aware vx/vy full80
end-to-end gradient and checkpoint replay equivalence.
