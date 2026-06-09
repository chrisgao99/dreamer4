# Tokenizer Encoder A/B Comparison

Date: 2026-06-08

Logs:

```text
Encoder A repeat map:
  /p/yufeng/tri30/dreamer4/waymo/logs/ooi50k_lat16_d256_ep200_4a100.log

Encoder B static map query:
  /p/yufeng/tri30/dreamer4/waymo/logs/ooi50k_lat16_d256_ep200_4a100_staticmap_v2.log
```

## 2026-06-08 Reconstruction Sanity Check Update

Decoded visualization exposed a loss-shape bug in the training/eval code used by
the two completed runs.

Issue:

```text
vector_tokenizer_reconstruction_loss() converted agents to (B,T,K,F), then
normalized_agent_targets() could transpose them again when T == K == 32.
```

Because the training run used `time_window=32` and `K=32`, the old validation
loss compared decoder predictions against an agent/time-swapped target. The
reported low validation losses in the tables below should therefore be treated
as invalid for reconstruction quality. Runtime/throughput comparison is still
useful.

Fix applied:

```text
/p/yufeng/tri30/dreamer4/waymo/vector_tokenizer_decoder.py
  normalized_agent_targets(..., already_btkf=True)
  _slice_time_window now handles both (B,K,T,F) and (B,T,K,F)
```

Visualization script:

```text
/p/yufeng/tri30/dreamer4/waymo/visualize_vector_tokenizer_reconstruction.py
```

Example output:

```text
/p/yufeng/tri30/dreamer4/waymo/reports/reconstruction_vis/ooi50k_idx0_v1_v2.png
```

Corrected quick validation sanity check on the first 128 validation samples:

| metric | Encoder A repeat map | Encoder B static map |
|---|---:|---:|
| corrected loss_total | 0.2429 | 0.2420 |
| corrected loss_agent_xy | 0.0561 | 0.0555 |
| corrected loss_agent_vel | 0.0171 | 0.0171 |
| corrected loss_agent_yaw | 0.3565 | 0.3559 |
| agent_xy_mae_m | 40.0033 | 39.7916 |
| agent_speed_mae_mps | 3.6195 | 3.6565 |
| agent_yaw_mae_deg | 75.5668 | 75.4594 |

Updated conclusion:

```text
The completed v1/v2 checkpoints are useful for runtime comparison and debugging,
but not for final reconstruction-quality conclusions. Re-run tokenizer training
after the loss fix before choosing the best reconstruction architecture.
```

Retraining launch scripts updated to use separate run/checkpoint/log names:

```text
Encoder A repeat map:
  script: /p/yufeng/tri30/dreamer4/waymo/launch_ooi50k_lat16_d256_ep200_4a100_tmux.sh
  run_name: ooi50k_lat16_d256_ep200_4a100_lossfix
  ckpt_dir: /p/yufeng/tri30/dreamer4/waymo/checkpoints/ooi50k_lat16_d256_ep200_4a100_lossfix
  log: /p/yufeng/tri30/dreamer4/waymo/logs/ooi50k_lat16_d256_ep200_4a100_lossfix.log

Encoder B static map query:
  script: /p/yufeng/tri30/dreamer4/waymo/launch_ooi50k_lat16_d256_ep200_4a100_staticmap_v2_tmux.sh
  run_name: ooi50k_lat16_d256_ep200_4a100_staticmap_v2_lossfix
  ckpt_dir: /p/yufeng/tri30/dreamer4/waymo/checkpoints/ooi50k_lat16_d256_ep200_4a100_staticmap_v2_lossfix
  log: /p/yufeng/tri30/dreamer4/waymo/logs/ooi50k_lat16_d256_ep200_4a100_staticmap_v2_lossfix.log
```

## Setup

Controlled variables:

```text
dataset: waymo_vector_dataset_ooi_centered_50k train/val
world_size: 4 GPUs
batch_size: 32 per GPU
epochs: 200
time_window: 32
d_model: 256
encoder_depth: 4
decoder_depth: 4
n_latents: 16
d_bottleneck: 32
decoder: Decoder 1 latent-only
no interaction scorer
no dynamics loss
```

Changed variable:

```text
Encoder A:
  encoder_variant = repeat_map
  per-timestep tokens = 16 latents + 32 agents + 256 map + 16 lights = 320
  params = 10,648,249

Encoder B:
  encoder_variant = static_map_query
  map_depth = 2
  map_cross_every = 1
  map_query_tokens = latent_agent
  per-timestep dynamic tokens = 16 latents + 32 agents + 16 lights = 64
  static map memory = 256 tokens per scene
  params = 13,807,801
```

## Final Epoch Metrics

Epoch 200 validation:

| metric | Encoder A repeat map | Encoder B static map | B - A | relative |
|---|---:|---:|---:|---:|
| loss_total | 0.0057 | 0.0061 | +0.0004 | +7.0% |
| loss_agent_xy | 0.0006 | 0.0008 | +0.0002 | +33.3% |
| loss_agent_vel | 0.0005 | 0.0005 | 0.0000 | 0.0% |
| loss_agent_yaw | 0.0082 | 0.0090 | +0.0008 | +9.8% |
| loss_agent_valid | 0.0001 | 0.0000 | -0.0001 | better |
| loss_light_state | 0.0012 | 0.0011 | -0.0001 | better |
| loss_light_valid | 0.0005 | 0.0001 | -0.0004 | better |

## Best Validation Metrics

Best epoch by `loss_total`:

| metric | Encoder A repeat map | Encoder B static map | B - A | relative |
|---|---:|---:|---:|---:|
| best epoch | 82 | 85 | +3 | - |
| loss_total | 0.0052 | 0.0055 | +0.0003 | +5.8% |
| loss_agent_xy | 0.0006 | 0.0008 | +0.0002 | +33.3% |
| loss_agent_vel | 0.0005 | 0.0005 | 0.0000 | 0.0% |
| loss_agent_yaw | 0.0078 | 0.0085 | +0.0007 | +9.0% |
| loss_agent_valid | 0.0001 | 0.0000 | -0.0001 | better |
| loss_light_state | 0.0007 | 0.0003 | -0.0004 | better |
| loss_light_valid | 0.0008 | 0.0007 | -0.0001 | better |

Best eval line:

```text
Encoder A:
  step 52500
  loss_total 0.0051

Encoder B:
  step 22500
  loss_total 0.0056
```

## Runtime

Training speed:

```text
Encoder A:
  last-100 logged steps/sec avg: 2.038
  after epoch 10 avg: 2.038

Encoder B:
  last-100 logged steps/sec avg: 4.129
  after epoch 10 avg: 4.131
```

Speedup:

```text
4.129 / 2.038 = 2.03x faster
```

Approximate training time for 70,400 steps:

```text
Encoder A: 70,400 / 2.038 = 9.6 hours
Encoder B: 70,400 / 4.129 = 4.7 hours
```

## Metric Caveat

Encoder B logs additional physical-unit metrics such as:

```text
agent_xy_mae_m
agent_speed_mae_mps
agent_vxvy_mae_mps
agent_yaw_mae_deg
```

These metrics are useful diagnostics, but they are not used as the main A/B
comparison because Encoder A was trained before those metrics were added.

Also, the absolute `agent_xy_mae_m` values in Encoder B look large relative to
the normalized SmoothL1 loss. Treat the physical-unit metrics as diagnostic
until they are sanity-checked with decoded visualizations or both checkpoints
are re-evaluated with the same metric script.

## Convergence

First epoch reaching validation `loss_total` threshold:

| threshold | Encoder A | Encoder B |
|---:|---:|---:|
| 0.5 | 2 | 2 |
| 0.1 | 11 | 13 |
| 0.05 | 16 | 16 |
| 0.02 | 20 | 21 |
| 0.01 | 31 | 32 |
| 0.007 | 41 | 46 |
| 0.006 | 59 | 59 |
| 0.0055 | 77 | 85 |

Interpretation:

- Encoder B converges at nearly the same pace early and mid training.
- Encoder A reaches the very lowest reconstruction losses slightly sooner.
- Encoder B's final/peak reconstruction quality is close but not quite equal.

## Interpretation

Main result:

```text
Static-map-query Encoder B is much faster and only slightly worse in total reconstruction loss.
```

Strengths of Encoder B:

- About 2.0x faster training throughput.
- Removes repeated static map tokens from the per-timestep dynamic token set.
- Matches velocity reconstruction.
- Improves or matches validity and traffic-light losses.
- Reaches near-baseline total loss.

Weaknesses of Encoder B:

- Slightly worse total reconstruction loss.
- The gap is mostly in agent position and yaw:
  - best loss_agent_xy is 0.0008 vs 0.0006;
  - best loss_agent_yaw is 0.0085 vs 0.0078.
- Parameter count is larger because map self-attention and cross-attention were added.

Conclusion:

```text
Encoder B is a strong default candidate if runtime and scalability matter.
Encoder A remains the strict best reconstruction baseline.
```

For this project, Encoder B should be treated as successful enough to continue:

- the quality gap is small;
- the runtime gain is large;
- static map memory is architecturally cleaner for future dynamics and larger map contexts.

## Recommended Next Steps

Do not move to interaction scorer yet.

Recommended order:

1. Keep Encoder A as the reported reconstruction upper baseline.
2. Use Encoder B as the practical/default architecture for the next static-map experiments.
3. Run one or two targeted Encoder B ablations:

```text
map_depth = 0
map_query_tokens = all
```

4. Implement Decoder 2: latent + static map memory.
5. Compare:

```text
Encoder B + Decoder 1
Encoder B + Decoder 2
```

6. If Decoder 2 closes the agent xy/yaw gap, use Encoder B + Decoder 2 for dynamics training.
7. After tokenizer+dynamics are stable, return to OOI interaction scoring.

Most important diagnosis:

```text
Encoder B does not need to prove it beats Encoder A on strict latent-only reconstruction.
It needs to prove that static map memory supports dynamic prediction efficiently.
Decoder 2 is therefore the next high-value experiment.
```
