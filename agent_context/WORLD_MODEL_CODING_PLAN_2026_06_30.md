# Waymo World Model Coding Plan

Date: 2026-06-30

Project path: `/scratch/baz7dy/tri30/dreamer4`

Main code paths:

- Dreamer-style dynamics repo: `/scratch/baz7dy/tri30/dreamer4/dreamer4`
- Waymo vector tokenizer code: `/scratch/baz7dy/tri30/dreamer4/waymo`
- DreamerV4 paper: `/scratch/baz7dy/tri30/dreamer4/dreamer4/dreamerv4.pdf`

## Current Decision

The next implementation step should be a Waymo-specific world model that trains
on frozen vector-tokenizer latents, not on raw vector states directly.

The right first target is:

```text
raw Waymo vector scene
  -> frozen vector tokenizer encoder
  -> continuous latent Z
  -> DreamerV4-style shortcut/flow dynamics
  -> predicted future latent Z
  -> frozen vector tokenizer decoder for evaluation
```

This repo is not the official DreamerV4 repo. Its dynamics implementation is a
compact non-official implementation of the paper's interactive dynamics idea:
continuous x-prediction in tokenizer latent space with shortcut forcing. It is
not a Dreamer3/RSSM-style recurrent state-space model.

## Tokenizer Choice

Use this tokenizer as the main world-model tokenizer:

```text
/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/
  ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/latest.pt
```

Reason:

- `lat64_b64` is the best of the three inspected candidates for all-agent
  reconstruction while keeping strong focus-agent reconstruction.
- Final logged eval from the inspected run:
  - all-agent XY MAE: about `0.6428m`
  - all-agent FDE: about `0.5876m`
  - focus-agent XY MAE: about `0.1459m`
  - focus-agent FDE: about `0.2276m`
  - yaw MAE: about `1.7332deg`

Secondary/debug option:

```text
ooi50k_lat64_b32_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp
```

This has slightly worse all-agent reconstruction but lower bottleneck dimension.

Avoid using `lat32_b16` as the first main world-model tokenizer. It is useful
as a compression stress test, but its all-agent reconstruction is much worse:
final inspected all-agent XY MAE was about `1.2886m`.

## Important Existing Code Facts

Waymo tokenizer:

- `waymo/core/vector_tokenizer_encoder.py`
  - `VectorStaticMapQueryEncoder` is the current preferred encoder variant.
  - It encodes dynamic agent/light tokens and static map memory.
  - It outputs `z` with shape `(B, T, N_latents, D_bottleneck)`.
- `waymo/core/vector_tokenizer_decoder.py`
  - `VectorBlockCausalTokenizerDecoder` decodes only from latent `z` plus learned
    agent/light query tokens by default.
  - `vector_tokenizer_reconstruction_loss()` already provides useful metrics:
    agent XY, FDE, speed, velocity, yaw, validity, and traffic-light metrics.
- `waymo/training/tokenizer/train_waymo_vector_tokenizer.py`
  - Rebuilds and trains the current tokenizer family.
  - Its `build_model()` and checkpoint loading patterns should be reused for
    the frozen tokenizer loader.

Dreamer dynamics:

- `dreamer4/model.py`
  - `Dynamics` consumes packed tokenizer latents, action token, shortcut signal
    token, shortcut step-size token, register tokens, and optional agent tokens.
  - It predicts clean packed latent tokens with `flow_x_head`.
  - The `wm_agent_isolated` mode prevents world-model tokens from attending to
    agent/policy tokens, matching the later policy-finetuning idea.
- `dreamer4/train_dynamics.py`
  - Currently image-tokenizer-specific.
  - It contains reusable pieces:
    - `dynamics_pretrain_loss()`
    - `make_tau_schedule()`
    - `sample_one_timestep_packed()`
    - `sample_autoregressive_packed_sequence()`
  - It must not be copied blindly because it assumes image patchify/unpatchify.

## DreamerV4 Paper Alignment

The relevant paper design is Section 3.2 Interactive Dynamics:

- Freeze a tokenizer and train dynamics on tokenizer representations.
- Corrupt clean latent `z1` by mixing Gaussian noise and data:

```text
z_tilde = (1 - tau) * z0 + tau * z1
z0 ~ N(0, I)
```

- Feed dynamics with corrupted latents, signal level, step size, and optional
  actions.
- Predict clean latents `z1_hat`, not velocity.
- Use x-space flow loss at the finest step size.
- Use shortcut bootstrap loss at coarser step sizes.
- Use ramp weighting:

```text
w(tau) = 0.9 * tau + 0.1
```

- At inference, generate autoregressively and use `K = 4` shortcut steps as a
  practical fast setting.
- The paper also slightly corrupts past context to signal level `tau_ctx = 0.1`
  during inference for robustness.

## Known Repo vs Paper Differences

The current `dreamer4/train_dynamics.py` step-size sampling is not exactly the
paper objective.

Current repo:

- Splits each batch into empirical rows and self/bootstrap rows.
- Empirical rows use only finest step `d_min`.
- Self rows sample only coarser powers-of-two step sizes.
- This is controlled by `self_fraction`.

Paper target:

- Sample `d` directly from powers of two:

```text
d in {1, 1/2, 1/4, ..., d_min}
```

- Apply flow loss where `d == d_min`.
- Apply bootstrap loss otherwise.

Recommended staged handling:

1. V0: reuse the current loss to get a working Waymo world model.
2. V1: implement exact per-sample/per-timestep step-size sampling.
3. V1 or V2: implement inference-time context corruption with `tau_ctx`.

Approximate V0 setting for `k_max=64`:

```text
self_fraction ~= 6 / 7 = 0.857142857
bootstrap_start = 0
batch_size divisible by 7 if possible
```

## Coding Plan

### V0: Action-Free Waymo Latent Dynamics

Add:

```text
waymo/training/world_model/train_waymo_world_model.py
```

Purpose:

- Train dynamics on frozen Waymo tokenizer latents.
- No action conditioning yet.
- No policy/agent tokens yet.
- Evaluate by decoding predicted latents back into Waymo vector states.

Core training loop:

1. Load `WaymoVectorDataset`.
2. Slice contiguous windows, initially `seq_len=32`.
3. Move batch to device.
4. With no grad, encode:

```python
out = tokenizer.encoder(
    agents=batch["agents"],
    agent_mask=batch["agent_mask"],
    map_polylines=batch["map_polylines"],
    map_mask=batch["map_mask"],
    lights=batch["lights"],
    light_mask=batch["light_mask"],
)
z_btLd = out.z
```

5. Pack latents for `Dynamics`:

```text
lat64_b64 with packing_factor=2:
  (B, T, 64, 64) -> (B, T, 32, 128)
```

6. Call `dynamics_pretrain_loss()` with:

```text
actions = None
act_mask = None
agent_tokens = None
n_agent = 0
```

7. Backprop only through dynamics.
8. Save dynamics checkpoints.
9. Log latent loss and decoded rollout metrics.

### Frozen Tokenizer Loader

Implement a helper in the new training script:

```python
load_frozen_waymo_vector_tokenizer(ckpt_path, device)
```

It should:

- Load `torch.load(ckpt_path, map_location="cpu")`.
- Read `ckpt["args"]`.
- Rebuild the exact encoder and decoder using the same logic as
  `waymo/training/tokenizer/train_waymo_vector_tokenizer.py`.
- Load `ckpt["model"]` strictly.
- Move to device, set eval mode, and freeze all parameters.

### Eval Path

For each eval batch:

1. Encode ground-truth sequence to `z_gt`.
2. Use first `ctx_length` latent frames as context.
3. Autoregressively generate `horizon` future latent frames.
4. Unpack predicted packed latents back to `(B,T,N_latents,D_bottleneck)`.
5. Decode with frozen vector decoder:

```python
pred = tokenizer.decoder(
    z_pred,
    agent_mask=batch["agent_mask"],
    light_mask=batch["light_mask"][:, :T_eval],
)
```

6. Score with `vector_tokenizer_reconstruction_loss()` against the same batch.
7. Compare against baselines:
   - repeat last decoded frame
   - teacher-forced tokenizer reconstruction upper bound
   - optionally linear extrapolation from raw kinematics

Primary eval metrics:

- focus-agent ADE/FDE
- all-agent ADE/FDE
- yaw MAE
- speed/velocity MAE
- light state/valid accuracy
- decoded rollout metrics over horizon only, not context

### First Training Config

Small smoke/overfit:

```text
seq_len = 100
packing_factor = 2
d_model_dyn = 256
dyn_depth = 4
n_heads = 4
time_every = 4
n_register = 4
n_agent = 0
k_max = 64
bootstrap_start = 0
self_fraction = 0.857142857
eval_ctx = 11
eval_horizon = 16
max_rollout_window = 100
eval_d = 0.25
amp_dtype = bf16 if supported, otherwise fp16 or none
```

Main first run:

```text
seq_len = 100
packing_factor = 2
d_model_dyn = 512
dyn_depth = 8
n_heads = 8
time_every = 4
n_register = 8
n_agent = 0
k_max = 64
bootstrap_start = 0
self_fraction = 0.857142857
eval_ctx = 11
eval_horizon = 80
max_rollout_window = 100
batch_size = tune to memory; full-scene windows are much heavier than 32-step chunks
```

## Acceptance Criteria For V0

Before adding action or policy machinery, V0 should satisfy:

1. Overfit a tiny subset without NaNs.
2. Latent flow loss decreases steadily.
3. Decoded rollout beats repeat-last baseline on focus-agent horizon ADE/FDE.
4. Autoregressive rollout remains finite and does not explode over 16 future
   steps.
5. Teacher-forced decoded tokenizer reconstruction remains close to the tokenizer
   eval logs, confirming that the frozen tokenizer loader is correct.

## V1: More Paper-Matching Dynamics Objective

After V0 works, replace batch-split sampling with exact sampling:

```python
emax = log2(k_max)
step_idx = randint(0, emax + 1, size=(B, T))
is_flow = step_idx == emax
is_boot = step_idx < emax
```

Then:

- Run one main forward for all rows/timesteps.
- Apply x-space flow loss only where `is_flow`.
- Apply shortcut bootstrap loss only where `is_boot`.
- Keep ramp loss weight `0.9 * tau + 0.1`.
- Remove or de-emphasize `self_fraction`.

This is worth doing once V0 proves the Waymo latent pipeline works.

## V2: Waymo-Specific Conditioning

### Static Map Conditioning

The tokenizer already uses static map memory, but the dynamics model only sees
the compressed latent `Z`. If rollouts drift off-map or lose lane constraints,
add static map condition tokens to dynamics:

- Reuse frozen tokenizer's map encoder.
- Pool/compress map tokens into a small number of condition tokens, such as
  8-16.
- Add a new modality to `Dynamics` or prepend map condition tokens to the spatial
  block with careful attention masking.

This should come after V0, because the simplest latent-only version may already
retain enough map information for short horizons.

### Ego/Focus Action Conditioning

If the goal becomes planning/control rather than pure forecasting, derive
focus-agent actions from the trajectory:

```text
acceleration
yaw_rate
speed_delta
curvature or steering-like proxy
optional validity/mask channels
```

Align actions as:

```text
action[t - 1] produces z[t]
```

For a quick implementation, pad derived action vectors to the current 16-D
`ActionEncoder`. Later, make `Dynamics` accept configurable `action_dim`.

## Notes For Future Sessions

- Do not start by adding policy/reward/value heads. First make latent dynamics
  work.
- Do not directly train dynamics to reconstruct raw vector states in V0. Decode
  for evaluation only.
- Do not assume fixed latent slots correspond to fixed agents. Treat `Z` as
  scene-level dynamic memory.
- For representation analysis, continue to use query-based readouts from `Z`
  as described in `DYNAMIC_MEMORY_REPRESENTATION_PLAN_2026_06_25.md`.
- The current best tokenizer is strong enough for this next step, but decoded
  full-agent metrics still have around 0.6m error, so focus-agent metrics and
  horizon baselines should be interpreted separately from full-scene metrics.
