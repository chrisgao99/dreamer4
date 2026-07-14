# Tokenizer Interaction-Aux Representation Experiment

Date: 2026-07-10

Project path: `/scratch/baz7dy/tri30/dreamer4`

Purpose: this is a representation-improvement experiment, not only a probe.
The goal is to make tokenizer `z` encode more future interaction information
by adding a small supervised auxiliary objective during tokenizer fine-tuning.

## Motivation

The previous three-layer interactive probe analysis suggests that current
single-step raw pair geometry is already very strong. The tokenizer `z` looks
mostly like a compact scene reconstruction code, which is expected because the
tokenizer was trained for encoding and reconstruction.

The concern is:

- if `z` does not contain enough interaction dynamics, the world model and
  policy may have to learn interaction reasoning from a weak latent basis;
- waiting for the world model to finish and then probing it is a passive
  wait-and-see plan;
- an earlier intervention should encourage the tokenizer latent to carry
  pairwise interaction-relevant information.

This experiment therefore adds interaction supervision directly while
fine-tuning the tokenizer.

## Core Design

Add an auxiliary interaction module to the tokenizer wrapper:

```text
agents/lights/map -> tokenizer encoder -> z

z + learned slot pair queries -> interaction auxiliary head
z -> tokenizer decoder -> reconstruction loss
```

The auxiliary head is intentionally `z`-only. It does not receive `pair_raw`.
This avoids the previous `raw_z` probe issue where raw pair geometry could
dominate the attention query and hide whether `z` itself carries interaction
information.

For each scene and query time:

```text
focus slot:     0
candidate slot: j, for valid non-focus agents
pair:           (0, j)
```

The auxiliary head predicts the same style of labels as the interactive probe:

- relevance;
- interaction type;
- response binary labels:
  - `focus_goes_first`
  - `focus_yields_to_other`
  - `focus_decelerates_for_interaction`
- `delta_arrival_time_s` regression.

## Query Construction

The auxiliary module uses learned candidate-slot queries initialized from the
tokenizer decoder's `agent_queries`.

For each candidate slot `j`, it builds a pair query from:

```text
focus_query = agent_query[0]
candidate_query = agent_query[j]
pair_query = MLP([focus_query, candidate_query,
                  focus_query * candidate_query,
                  abs(focus_query - candidate_query)])
```

Then the pair query cross-attends to `z` latents at each timestep.

This keeps the probe aligned with agent slots without feeding raw geometry.
The candidate identity is represented by the decoder's learned slot query,
not by hand-crafted pair features.

## Loss

Total tokenizer fine-tuning loss:

```text
loss_total = loss_reconstruction + interaction_aux_weight * loss_interaction_aux
```

Where:

```text
loss_interaction_aux =
    relevance_weight * BCE(relevance)
  + type_weight * CE(type)
  + response_bin_weight * BCE(response_binary)
  + response_reg_weight * SmoothL1(delta_arrival_time)
```

Current default weights in the submitted Slurm file:

```text
interaction_aux_weight = 0.1
interaction_relevance_weight = 1.0
interaction_type_weight = 1.0
interaction_response_bin_weight = 1.0
interaction_response_reg_weight = 0.2
```

The auxiliary objective is deliberately small relative to reconstruction, so
the experiment nudges `z` toward interaction information without immediately
destroying the existing tokenizer codebook/reconstruction behavior.

## Label Source

The labels reuse the existing future-interaction label builder:

```text
waymo/interactive_probe/labels.py
```

Training uses the full scene batch to create future labels, while the tokenizer
still trains on a sliced context window. When random start is enabled, the
window start is restricted so there are enough future steps for labels.

Default:

```text
time_window = 32
interaction_query_step = -1
interaction_future_steps = 50
focus_index = 0
```

So the auxiliary target asks:

> Given the tokenizer latent at the last timestep of the 32-step context, can
> `z` support predicting future interaction labels over the next 50 steps?

## Code Changes

Main implementation:

```text
waymo/core/vector_tokenizer_decoder.py
```

Added:

- `InteractionAuxOutput`
- `TokenizerInteractionAuxHead`
- optional `interaction` output in `VectorTokenizerOutput`
- `VectorTokenizer.init_interaction_aux_from_decoder_queries()`

Training integration:

```text
waymo/training/tokenizer/train_waymo_vector_tokenizer.py
```

Added:

- interaction auxiliary args;
- full-batch future label construction;
- `compute_interaction_aux_loss`;
- `compute_total_loss`;
- metrics for interaction losses and accuracies;
- old-checkpoint compatibility when `interaction_aux.*` weights are missing.

Launch script support:

```text
waymo/training/launch_ooi50k_lat16_d256_ep200_2a100_staticmap_v2_chunk32_trajloss_randstart_tmux.sh
```

Added:

- `RESUME` override;
- interaction aux environment variables;
- forwarding of interaction aux args to the tokenizer trainer.

## Slurm Job

Submit file:

```text
waymo/training/submit_ooi50k_lat64_b64_interaction_aux_finetune.slurm
```

Base checkpoint:

```text
waymo/checkpoints/ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt
```

Output checkpoint directory:

```text
waymo/checkpoints/ooi50k_lat64_b64_d256_interaction_aux_finetune_from_best
```

Submit command:

```bash
sbatch /scratch/baz7dy/tri30/dreamer4/waymo/training/submit_ooi50k_lat64_b64_interaction_aux_finetune.slurm
```

The job requests up to three days:

```text
#SBATCH --time=3-00:00:00
```

Resume behavior:

- first run: if no `latest.pt` exists in the new output directory, resume from
  the base `best.pt`;
- later runs with the same `RUN_NAME`: automatically resume from the new
  output directory's `latest.pt`.

## Important Checkpoint Detail

The base checkpoint saved args show:

```text
decoder_attend_map = None
```

So the new Slurm file sets:

```text
DECODER_ATTEND_MAP=0
```

This is required to match the base checkpoint architecture.

## Verification Already Done

Lightweight checks passed:

- tokenizer training script argument parsing;
- launch script `bash -n`;
- Slurm script `bash -n`;
- loading the old base checkpoint into the new model with interaction aux head;
- aux slot queries initialized exactly from loaded `decoder.agent_queries`;
- one-sample smoke test for forward pass and `compute_total_loss`.

Checkpoint loading message expected on first run:

```text
Loaded ... without optimizer state because optional interaction aux head differs.
Initialized interaction aux slot queries from loaded decoder.agent_queries.
```

This is expected because the old tokenizer checkpoint does not contain
`interaction_aux.*` parameters.

## What To Watch

During training, watch:

```text
loss_reconstruction
loss_interaction_aux
loss_interaction_relevance
loss_interaction_type
loss_interaction_response_bin
loss_interaction_response_reg
interaction_relevance_acc
interaction_type_acc
interaction_response_bin_acc
```

Main concern:

- if `interaction_aux_weight` is too high, reconstruction may degrade;
- if too low, `z` may not move enough to help interaction representation.

Initial default is conservative:

```text
interaction_aux_weight = 0.1
lr = 1e-4
```

## Follow-Up Evaluation

After fine-tuning, compare:

1. reconstruction metrics against the original tokenizer;
2. z-only interactive probe performance;
3. raw-only vs z-only vs raw-z probe behavior;
4. downstream world model rollout metrics when using the interaction-aux
   tokenizer checkpoint.

The desired signal is not merely that the auxiliary head performs well during
training. The stronger evidence is:

- frozen fine-tuned `z` improves z-only probe results;
- raw-free interaction labels become easier to decode from `z`;
- world model or policy training becomes more interaction-aware or more stable.

