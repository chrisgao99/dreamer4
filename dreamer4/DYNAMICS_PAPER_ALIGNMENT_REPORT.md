# Dynamics Training Alignment Report

Date: 2026-05-07

This report compares `/p/yufeng/tri30/dreamer4/dreamer4/train_dynamics.py` and related code against the Dreamer 4 paper, focusing on dynamics pretraining.

Paper reference: https://arxiv.org/pdf/2509.24527

## Executive Summary

The implementation matches the paper at the high level:

- It trains on tokenizer latents from dataset sequences, not autoregressive self-rollouts.
- It corrupts clean latents by interpolating Gaussian noise and data latents.
- It uses a causal transformer, so future timesteps do not leak into earlier predictions.
- It uses x-prediction and the ramp loss weight `w(tau) = 0.9 * tau + 0.1`.
- It implements a shortcut bootstrap loss for coarser step sizes.

However, it is not an exact paper reproduction. Some differences can be reduced with command-line hyperparameters, but several require code changes.

## Important Correction: `--self_fraction 1` Is Not Paper-Matching

Changing:

```bash
--self_fraction 1 --k_max 64
```

does not make the step-size sampling match the paper.

The paper samples the step size from powers of two:

```text
d in {1, 1/2, 1/4, ..., d_min}
```

not from all fractions like `1/3`, `1/5`, etc.

If `k_max = 64`, then:

```text
d_min = 1/64
d values = {1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64}
```

There are 7 possible step sizes. A paper-like uniform distribution would use each with probability `1/7`.

In this code, the batch is split into:

```python
B_self = round(self_fraction * B)
B_emp = B - B_self
```

and then clamped so `B_self < B`. See `train_dynamics.py`, lines 756-758.

The empirical rows are forced to `d_min`, and self/bootstrap rows sample only coarser `d`, excluding `d_min`. See `train_dynamics.py`, lines 201-205 and 160-165.

So with `k_max = 64`:

```text
empirical rows: d = 1/64
self rows:      d in {1, 1/2, 1/4, 1/8, 1/16, 1/32}
```

With the current default `batch_size = 24`:

```text
self_fraction = 0.25:
  B_self = 6, B_emp = 18
  P(d = 1/64) = 18/24 = 0.75
  P(each coarser d) = (6/24) / 6 = 0.0417

self_fraction = 1:
  B_self is clamped to 23, B_emp = 1
  P(d = 1/64) = 1/24 = 0.0417
  P(each coarser d) = (23/24) / 6 = 0.1597
```

Neither matches the paper's uniform `1/7 = 0.1429` per step size.

The closest hyperparameter setting for `k_max = 64` is:

```bash
--k_max 64 --self_fraction 0.857142857 --bootstrap_start 0
```

because:

```text
self_fraction ~= log2(k_max) / (log2(k_max) + 1)
              = 6 / 7
```

With `batch_size = 24`, this gives `B_self = 21`, `B_emp = 3`:

```text
P(d = 1/64) = 3/24 = 0.125
P(each coarser d) = (21/24) / 6 = 0.1458
```

This is close but not exact because 24 is not divisible by 7. For exact batch-level proportions, use a batch size divisible by 7, for example:

```bash
--batch_size 28 --k_max 64 --self_fraction 0.857142857 --bootstrap_start 0
```

Then `B_self = 24`, `B_emp = 4`, and each of the 7 step sizes gets probability `4/28 = 1/7`.

Even then, this is only matching the marginal batch distribution. The paper's cleaner implementation is to sample `d` directly for every batch/time element and choose flow vs bootstrap loss based on whether `d == d_min`. That requires a code change.

## Differences And Fixes

| Area | Current Code | Paper | Can Hyperparameter Fix It? | Suggested Fix |
|---|---|---|---|---|
| Finest grid `k_max` | Default `--k_max 8`; `d_min = 1/8`. | Objective uses `d_min = 1/Kmax`; paper ablations compare against 64-step diffusion and shortcut sampling with 4 steps. | Mostly yes, if target is `Kmax=64`. | Run with `--k_max 64`. I did not find a single explicit paper line saying the final training `Kmax` value, but `64` is strongly implied by the sampling-step ablation. |
| Step-size sampling distribution | Batch split: empirical rows are all `d_min`; self rows sample coarser powers of two. | Sample `d` from powers of two and use flow loss only when `d = d_min`, bootstrap otherwise. | Approximate only. | For `k_max=64`, use `--self_fraction 0.857142857`, ideally with batch size divisible by 7. Exact match requires code change to sample `step_idx` uniformly from `[0, emax]` for all rows/timesteps. |
| Bootstrap warmup | Bootstrap inactive until `--bootstrap_start`, default `5000`. | Paper objective presents bootstrap as part of training, no warmup described. | Yes. | Use `--bootstrap_start 0`. |
| Loss type | Code predicts clean `z1` and uses x-space flow/bootstrap losses. | Paper explicitly uses x-prediction for long rollouts. | Already matches. | No change needed. |
| Noise corruption | Code uses `z_tilde = (1 - tau) * z0 + tau * z1`, with `z0 ~ N(0, I)`. | Same as Eq. 6. | Already matches. | No change needed. |
| Ramp loss weight | Code uses `w = 0.9 * tau + 0.1`. | Same as Eq. 8. | Already matches. | No change needed. |
| Causality | Code uses `is_causal=True` in temporal attention. | Paper says attention is causal in time. | Already matches at the causal-mask level. | No change needed for causality. |
| Inference context corruption `tau_ctx` | Code/eval/interactive treat past context as nearly clean via `signal_idxs_full = k_max - 1`, but I do not see actual Gaussian corruption applied to past inputs. | Paper says past inputs are slightly corrupted to `tau_ctx = 0.1` for robustness during inference. | No exposed hyperparameter. | Requires code change in `sample_one_timestep_packed()` in both `train_dynamics.py` and `interactive.py`: corrupt `past_packed` and set matching signal indices for context. Need resolve paper notation ambiguity around whether `tau_ctx=0.1` means signal level 0.1 or corruption level 0.1. |
| Context/window length | Training default `--seq_len 32`; eval context default `8`; interactive context default `24`. | Paper discusses longer context, e.g. Minecraft context length `C=192`, and says batch length should exceed context length. | Partly. | Increase `--seq_len`, `--eval_ctx`, and `interactive.py --ctx_window` as memory allows. Matching the paper's alternating short/long batch schedule requires code changes. |
| Alternating short/long batches | Not implemented. One `--seq_len` is used. | Paper alternates many short batches with occasional long batches, then finetunes on long batches. | No. | Requires dataset/training-loop schedule changes. |
| Temporal attention frequency | Default `--time_every 1`. | Paper says temporal attention once every 4 layers. | Yes. | Use `--time_every 4`. |
| Efficient transformer details | Code uses RMSNorm/SwiGLU-like MLP and separate space/time attention, but no RoPE, QKNorm, attention-logit soft capping, or GQA. | Paper uses pre-layer RMSNorm, RoPE, SwiGLU, QKNorm, logit soft capping, GQA, and other efficient design choices. | Mostly no. | Requires model code changes. `--time_every 4` only covers one of these differences. |
| Action encoding | Code encodes a full 16-D continuous action vector through one MLP. | Paper encodes action components separately and sums them with learned embeddings; supports continuous, categorical, and binary components. | No. | Requires code change in `ActionEncoder`. For DMControl continuous vectors, current code may be reasonable but is not paper-identical. |
| Spatial tokens/model scale | Current run log shows tokenizer `n_lat=16`, `packing=2`, so dynamics has only `n_spatial=8`. | Paper's final Minecraft model uses many more spatial tokens; ablation text mentions increasing to `Nz=128` and `Nz=256`. | Not really with only dynamics args. | Requires tokenizer architecture/checkpoint changes and likely retraining. Reducing `--packing_factor` can increase `n_spatial` only up to tokenizer `n_latents`. |
| Labeled/unlabeled video mixture | `--use_actions` uses action-conditioned `WMDataset`; without actions uses frame-only dataset. Mixed labeled/unlabeled rows in one training run are not clearly implemented. | Paper emphasizes learning from large unlabeled video plus smaller action-labeled data. | No, not cleanly. | Requires dataset/training-loop changes to mix action-labeled and unlabeled batches, using `actions=None` or masks for unlabeled samples. |
| Policy/reward/imagination training | `train_dynamics.py` only trains the dynamics model. | Paper later finetunes policy/reward heads and performs imagination training. | Not applicable to this script. | Requires additional training code; not solved by dynamics hyperparameters. |

## Recommended Minimal CLI For A More Paper-Like Dynamics Objective

If the goal is only to make the shortcut objective sampling closer to the paper while keeping this code structure:

```bash
torchrun --nproc_per_node=8 train_dynamics.py \
  --use_actions \
  --k_max 64 \
  --self_fraction 0.857142857 \
  --bootstrap_start 0 \
  --time_every 4
```

If possible, use a batch size divisible by `log2(k_max) + 1`. For `k_max=64`, use a batch size divisible by `7`.

Example:

```bash
--batch_size 28
```

This still does not implement the exact paper sampling. It only approximates the marginal distribution through the existing empirical/self batch split.

## Recommended Code Change For Exact Step-Size Sampling

To match the paper more directly, replace the empirical/self batch split with direct sampling:

```python
emax = log2(k_max)
step_idx = randint(0, emax + 1, size=(B, T))
is_flow = step_idx == emax
is_boot = step_idx < emax
```

Then:

- Run one main forward pass for all tokens.
- Apply x-space flow loss where `is_flow`.
- Apply bootstrap loss where `is_boot`.
- Weight both by `w(tau) = 0.9 * tau + 0.1`.

This would remove `--self_fraction` entirely or turn it into an optional approximation mode.

## Notes On Current Run

The current `train_dynamics.out` shows:

```text
k_max = 8
step_embed = Embedding(4, 512)
signal_embed = Embedding(9, 512)
B_self = 6
batch_size = 24
```

So the current run uses:

```text
d values possible overall: {1, 1/2, 1/4, 1/8}
flow rows: d = 1/8
self rows: d in {1, 1/2, 1/4}
```

with 75% of rows forced to `d = 1/8` and only 25% assigned to coarser bootstrap steps.

## Source Pointers

Paper:

- Eq. 6: corrupted latents and clean representation prediction.
- Eq. 7: x-space flow loss for `d = d_min`, bootstrap loss otherwise.
- Eq. 8: ramp loss weight `w(tau) = 0.9 tau + 0.1`.
- Section 3.2: autoregressive inference with `K = 4`, plus `tau_ctx`.
- Section 3.4: causal-in-time efficient transformer, temporal attention once every 4 layers, RoPE, QKNorm, logit soft capping, GQA.

Local code:

- `train_dynamics.py`: `_sample_step_excluding_dmin()`, `_sample_tau_for_step()`, `dynamics_pretrain_loss()`, `sample_one_timestep_packed()`.
- `model.py`: `TimeSelfAttention`, `Dynamics`, `ActionEncoder`.
- `interactive.py`: interactive sampling path, also missing visible `tau_ctx` corruption.
