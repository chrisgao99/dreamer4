# Waymo evaluation

Evaluation implementations stay in this directory. Shell entry points are
grouped under `launchers/` by the model they evaluate:

- `launchers/tokenizer/`: reconstruction visualizations and tokenizer metrics.
- `launchers/motion_latent/`: MotionLatent checkpoint and rollout evaluation.
- `launchers/world_model/`: world-model context, rollout, oracle, and overfit evaluation.

Prefer a parameterized launcher when evaluating another checkpoint with the
same protocol. Keep a separate script only when the model architecture,
evaluation protocol, context, rollout policy, or solver settings differ.

The current chunked-tokenizer protocol uses a window of 32 and stride of 30;
the superseded unchunked stage-comparison launchers have been removed.
