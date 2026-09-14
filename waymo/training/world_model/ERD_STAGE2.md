# ERD stage two on generate-all 75k

This is the approved adaptation of Flow-ERD IV-B / Algorithm 1 / Eq.19 to
our existing holonomic DirectActionFlow checkpoint, not a full AFM reproduction.
The paper does not release code or specify all training hyperparameters.

## Fixed algorithm

- Teacher, generator, fake-score start from the same best75k EMA weights.
- Teacher is frozen; generator and fake-score have independent AdamW optimizers.
- Jointly generate all agents, without conditioning on future GT actions.
- Each generated training sample concatenates the actually committed B=5 actions
  of three H=15 plans. All three replans remain differentiable, including the
  Euler flow solves (8 steps each). Activation checkpointing recomputes complete
  plans; no truncation or detached history is used.
- Both scores condition on the same logged 11-frame window anchor. Within the
  generated sample, the generator conditions on its own executed history.
- With x_l=(1-l)z+l*x_1, score=(l*v-x_l)/(1-l). For beta != 1, subtracting scaled
  velocities is NOT equivalent to subtracting scores.
- Fake-score trains with masked flow MSE on detached on-policy samples.
- Generator uses stopgrad(fake_score - beta*real_score) dotted with generated
  normalized actions, averaged over valid action coordinates. This surrogate
  value is not a KL estimate and need not decrease during successful training.
- Alternate 100 FAKE ONLY iterations and 100 BOTH iterations. In BOTH, update
  generator every fifth fake update. Fresh on-policy samples each iteration.
- Frozen parameters and score evaluation have no gradients; only generated
  actions/history/flow trajectory receive the DMD gradient.

## Explicit choices (not author-provided defaults)

- Existing all-holonomic execution and initial Huber-trained checkpoint retained.
- EMA checkpoint initializes all networks; model dropout disabled for all three
  distributions, gradients enabled independently. No pretrained optimizer reuse.
- Lambda uniform [0.02,0.90], normalized-action space, no added score normalization,
  BC loss, road penalty, action clipping or inference-noise inflation.
- Generator LR 1e-6; fake LR 1e-5; AdamW weight decay .01; gradient clip 1.
- bf16 autocast, float32 action integration/score conversion, batch size 2.
- Deterministic PyTorch algorithms and CUBLAS workspace configuration enabled
  to make same-device checkpoint-resume tests reproducible.
- Generator EMA .999. Constant LRs for long-run fine-tuning.
- Stateless with-replacement training-batch sampling (seed + iteration), original
  data split and action statistics. Logged anchor selection follows stage one.
- Two matched seeds, beta=1 and .99. Target 10,000 generator / 100,000 fake updates
  each. GPU3 queue alternates 500-generator-update blocks between the runs so both
  progress; complete network/optimizer/RNG state persists between blocks.

## Validation and recovery

- Initial and every 250 generator updates: fixed 32 scenes, 4 rollouts, H80/B5,
  ADE, minADE, CPD, and fraction of valid steps with displacement >5m. The latter
  is a motion diagnostic, not an official off-road metric.
- Save latest every 500 fake iterations or 100 generator updates, and a retained
  snapshot every 1,000 generator updates. best_ade.pt uses validation ADE only;
  do not interpret it as the best realism/diversity checkpoint.
- Every checkpoint stores teacher, generator, fake, generator EMA, optimizers,
  next iteration, generator update count, RNG states, original model args and
  action stats. DataLoader has a separate RNG so resumed iterator creation does
  not change training randomness. SIGTERM/INT save after the current iteration.
- Queue final evaluation: 512 scenes, K32, H80, B1/B5/B15 using final EMA; full
  details and independent logs. Check validation trends for convergence; reaching
  10k updates does not itself establish convergence.
- Additional road-adherence and maneuver-intent analyses require their own
  geometry/label validation; they are not replaced by CPD or the >5m diagnostic.

Tests cover Gaussian velocity-to-score conversion, beta scaling, gradient masking
and stop-gradient, checkpointed vs uncheckpointed multi-replan gradients, phase
schedule, deterministic resumed batches, and real-checkpoint CUDA smoke/resume.
