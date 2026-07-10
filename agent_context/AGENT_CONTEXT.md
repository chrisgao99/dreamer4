# Agent Context

Last updated: 2026-07-07

## Source

- Combined boss-meeting Google Slides deck covering 2026-04-21, 2026-04-09, and 2026-03-24:
  [Google Slides](https://docs.google.com/presentation/d/14FdjmNIs-73L64M3qDy_P66qVXiMy3ILl8cpyiaAvPg/edit?usp=sharing)

## Research Focus

Yufeng is working on learning interaction-aware representations of driving scenarios, primarily from Waymo motion data. The goal is to build representations that capture multi-agent driving dynamics well enough to support downstream planning, policy learning, and analysis of special traffic interactions.

The current implementation priority is **Dreamer 4**. This is the first major direction to execute, ahead of alternative pathways.

## Current Thesis

The current working belief is:

1. A world-model-based approach is a strong way to learn driving interaction structure because predicting future world state can force the model to capture causality that hand-designed similarity rules may miss.
2. Dreamer 4 is especially promising because its transformer-style space-time structure can preserve richer multi-agent information than Dreamer 3 / RSSM-style compression.
3. Tokenizer pretraining should focus on bottleneck latents `z` and ordinary per-agent state tokens for selected tracks. Learned scene/task/policy-agent tokens should not be part of tokenizer pretraining; add them only when finetuning the dynamics model into a policy.
4. Contrastive learning is still relevant, but likely as an enhancement on top of pooled latents, selected-agent tokens, or downstream policy finetuning tokens, not as the first standalone path.
5. As of 2026-06-25, the tokenizer bottleneck `Z` should be treated as a **scene-level dynamic memory**. Agent-specific and pair-specific meanings should be extracted with explicit queries, such as `read(Z, query(focus, candidate))`, rather than by assuming fixed latent slots correspond to fixed agents.

## Latest Representation Evaluation Plan

See `DYNAMIC_MEMORY_REPRESENTATION_PLAN_2026_06_25.md` for the current plan to evaluate tokenizer latents as interaction-aware dynamic representations.

The revised core questions are:

1. **Relevance:** Can the representation identify interaction-relevant agents?
2. **Relations:** Can it predict future pairwise relations and risks?
3. **Counterfactuality:** Does it respond to relevant-agent changes and stay stable for irrelevant changes?
4. **Temporal Sufficiency:** Is it a useful compact state for predicting future dynamics?
5. **Transfer:** Does it improve downstream policy, value, or world-model learning?
6. **Semantic Geometry:** Are similar interaction behaviors close in representation space?

The preferred probe design is:

```text
Z = encoder(scene)
q_ij = raw current or short-history pair query for agents i and j
h_pair_ij = CrossAttention(query=q_ij, key/value=Z_flat)
head(h_pair_ij) -> objective future-relation targets
```

The hidden `h_pair` is then used for clustering and nearest-neighbor analysis.
Important baselines are `pair_raw_only`, `pair_raw + Z`, and `pair_raw +
shuffled_Z`.

## Interactive Probe Implementation Note, 2026-07-07

An implemented probe now lives at:

```text
/scratch/baz7dy/tri30/dreamer4/waymo/interactive_probe
```

Its purpose is to evaluate whether tokenizer bottleneck latents `z[31]` contain
future-interaction information beyond current focus-candidate pair geometry.
The probe uses one sample per valid `(focus agent, candidate agent)` pair. The
input pair query is current-state only, while labels may use future ground-truth
trajectories.

### Cache And Data

Built cache:

```text
/scratch/baz7dy/tri30/dreamer4/waymo/cache/interactive_probe_z31_v0_besttok
```

Train/val sizes:

```text
train: 44,998 scenes, 865,398 pairs
val:    5,002 scenes,  95,493 pairs
```

Cache fields:

```text
z_current: scene-level z[31], shape (num_scenes, 64, 64)
pair_raw: current focus-candidate features, shape (num_pairs, 34)
relevance_targets / masks
type_targets / masks
response_bin_targets / masks
response_reg_targets / masks
diagnostics
```

`pair_raw` is a 34D current-state pair feature:

```text
8 focus state features:
  x, y, speed, vx, vy, sin_yaw, cos_yaw, type
8 candidate state features:
  x, y, speed, vx, vy, sin_yaw, cos_yaw, type
18 pair geometry/motion features:
  rel_dx, rel_dy, rel_dist, rel_vx, rel_vy, rel_speed,
  bearing_sin, bearing_cos, heading_diff_sin, heading_diff_cos,
  longitudinal_offset, lateral_offset, abs_lateral_offset,
  closing_speed, same_direction_proxy, crossing_angle_proxy,
  current_close_5m, current_close_10m
```

### Layer 1: Relevance

Question:

```text
Is this candidate potentially interaction-relevant to the focus agent?
```

All valid pairs are trained/evaluated for Layer 1.

Positive label if any future cue indicates relevance:

```text
future time-aligned distance is small
OR future swept/path spatial overlap with small PET
OR same-corridor leading/following headway is small
```

Default thresholds in `interactive_probe/labels.py` include:

```text
future_steps = 50
dt = 0.1
relevance_dist_m = 8.0
path_overlap_dist_m = 4.0
pet_relevant_s = 3.0
following_relevant_headway_m = 20.0
```

Current label counts:

```text
train relevance: 166,641 / 865,398
val relevance:    18,299 /  95,493
```

### Layer 2: Interaction Type

Question:

```text
For truly relevant pairs, what type of interaction is this?
```

Layer 2 uses a ground-truth mask. Only pairs with a clear type contribute to
the type loss and type metrics. Ambiguous relevant pairs are masked out rather
than forced into `none`.

Implemented type classes:

```text
other_leads_focus:
  candidate is ahead of focus in a same-direction same-corridor future relation;
  focus may need to follow/control speed.

other_follows_focus:
  candidate is behind focus in a same-direction same-corridor future relation;
  candidate is relevant but this type currently does not participate in Layer 3
  yield/deceleration response labels.

crossing_or_oncoming_conflict:
  future paths spatially conflict and headings at the conflict are substantially
  different; this covers crossing and same-space oncoming conflicts.

converging_conflict:
  future motion converges into the same corridor/space, including cut-in,
  lane merge, road narrowing, or similar conflict. This intentionally merges
  separate `merge` and `cut-in` concepts because Waymo support for cleanly
  separating them is sparse and ambiguous.
```

Type priority in the labeler:

```text
crossing_or_oncoming_conflict
then converging_conflict
then other_leads_focus
then other_follows_focus
```

Current val type supports:

```text
other_leads_focus: 5087
other_follows_focus: 4888
crossing_or_oncoming_conflict: 807
converging_conflict: 1922
```

### Layer 3: Response / Priority

Question:

```text
For clearly interactive pairs, what does focus do or who has priority?
```

Layer 3 also uses ground-truth masks. Not every type contributes to every
response. In particular, `other_follows_focus` currently only participates in
Layer 2 type classification, not Layer 3 yield/deceleration response labels.

Implemented binary response labels:

```text
focus_goes_first:
  for crossing/converging conflict pairs with a clear conflict region, focus
  reaches the closest shared/conflict region before the candidate by a margin.

focus_yields_to_other:
  for crossing/converging conflict pairs, candidate reaches the conflict region
  first and focus shows a deceleration/slowdown response.

focus_decelerates_for_interaction:
  for other_leads_focus, crossing_or_oncoming_conflict, and converging_conflict,
  focus has a meaningful future speed drop and deceleration.
```

Implemented regression response label:

```text
delta_arrival_time_s:
  focus arrival time at closest shared/conflict region minus candidate arrival
  time. Negative means focus arrives first; positive means candidate arrives
  first.
```

Current val response supports:

```text
focus_decelerates_for_interaction: 7816 pairs, positive rate 0.3358
focus_goes_first:                  2267 pairs, positive rate 0.4204
focus_yields_to_other:             2267 pairs, positive rate 0.1142
delta_arrival_time_s:              2267 pairs
```

### Probe Model Structure

Implemented modes:

```text
raw_only:
  pair_raw -> pair_encoder -> heads

raw_z:
  pair_raw -> pair_encoder -> query
  z_current -> projection -> memory
  query cross-attends to z memory
  fused representation -> heads

raw_shuffled_z:
  same as raw_z, but z_current is shuffled across batch as a control
```

Default architecture:

```text
pair_raw dim = 34
z_current shape per scene = (64 latents, 64 dim)
d_model = 128
pair_encoder = MLP(34 -> 128)
z_proj = Linear(64 -> 128) + LayerNorm
attention = MultiheadAttention(query=pair embedding, key/value=z latents)
fuse = MLP([pair embedding, attended z] -> 128)
heads:
  relevance_head: binary logit
  type_head: 4-way classification
  response_bin_head: 3 binary logits
  response_reg_head: delta_arrival_time_s regression
```

Training loss:

```text
loss = relevance BCE on all pairs
     + type cross-entropy on GT type mask
     + response binary BCE on GT response masks
     + response regression smooth-L1 on GT response masks
```

### Current Result Summary

Three modes were trained under:

```text
/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/interactive_probe_z31_v0_besttok
```

Best-checkpoint headline results:

```text
raw_only:
  relevance AP 0.9673, type macro-F1 0.8740,
  response AP 0.8175, delta MAE 0.8153

raw_z:
  relevance AP 0.9638, type macro-F1 0.8622,
  response AP 0.8262, delta MAE 0.8218

raw_shuffled_z:
  relevance AP 0.9662, type macro-F1 0.8720,
  response AP 0.8174, delta MAE 0.8146
```

Interpretation as of 2026-07-07:

```text
This interactive probe does not yet show a stable broad z advantage over
current pair geometry. raw_only is already very strong, especially for relevance
and type. The clearest positive z signal is focus_decelerates_for_interaction:
best AP raw_only 0.9063, raw_z 0.9294, shuffled_z 0.9056. Other priority/yield
and type metrics are mostly tied or worse for raw_z.
```

## Latest World Model Coding Plan

See `WORLD_MODEL_CODING_PLAN_2026_06_30.md` for the current plan to implement
the Waymo world model after tokenizer pretraining.

The near-term decision is to freeze the best current Waymo vector tokenizer
(`lat64_b64`) and train DreamerV4-style shortcut/flow dynamics on tokenizer
latents. The first version should be action-free latent forecasting with decoded
Waymo metrics; action conditioning, exact paper-style step-size sampling,
`tau_ctx` context corruption, static-map conditioning, and downstream
policy/reward/value heads should be staged after that baseline works.

## What Changed Across the Three Meetings

### 2026-03-24

The problem was framed as **offline representation learning for interaction dynamics**. Two main pathways were analyzed:

- **Trajectory sequence alignment (CLASS-style)** using DTW
- **Dynamic system representation (MATS-style)** using affine time-varying systems

Key takeaway:

- The hard part in contrastive learning is not only the loss, but defining mathematically meaningful positive pairs for "similar driving dynamics."
- DTW can align trajectories that look spatially similar while ignoring meaningful speed differences unless richer state variables are included.
- MATS offers an interpretable dynamical-system view, but raises unresolved choices about which timesteps or interaction blocks should define similarity.

This meeting ended with two next directions:

- investigate other pathways such as world models and PINNs
- filter and visualize special interaction scenes such as highway merge, multi-lane to two lanes, and unprotected left turn

### 2026-04-09

The focus shifted toward a **Dreamer world model** as a way to learn driving dynamics directly.

Key ideas introduced:

- Use all agents' positions as input
- Encode multi-agent state with self-attention
- Handle variable numbers of agents with fixed max size, zero-padding, and masking
- Add map information through a separate map encoder plus cross-attention
- Consider whether discrete latent state `z` can represent meaningful driving-interaction concepts

Important tension:

- Dreamer-style latent state may naturally compress high-level concepts
- But it is unclear whether RSSM-style latent variables are expressive enough for rich multi-agent interaction behavior

Tooling and environment notes:

- GPUDrive is attractive for tutorials and rendering, but CUDA/NVRTC compilation is a blocker
- PufferDrive scales well on CPU but seems harder to render and inspect visually

### 2026-04-21

This meeting made the direction much more concrete: **start from Dreamer 4**.

The main shift was from RSSM compression to a **block-causal transformer world model** with both space and time processing:

- Space layers process interactions among tokens within a timestep
- Time layers process the same token/entity across timesteps
- For driving, the causal time layer is attractive because it lets the model attend to the same agent over time

The key representation idea is:

- Dreamer 4 inserts task/policy tokens during downstream agent finetuning, after tokenizer/world-model pretraining.
- For this project, tokenizer pretraining should not include a learned scene token or learned policy-agent token.
- Downstream scene/task/policy-agent tokens can be inserted later when finetuning the dynamics model into a policy, where they can gather global interaction information for policy, reward, value, probing, or retrieval.

Another important benefit is that the model output can retain **selected-agent state representations**, which may be used for agent-level dynamics analysis. Learned policy-agent tokens are reserved for later dynamics-to-policy finetuning.

## Why Dreamer 4 Is the First Implementation Target

Dreamer 4 is currently favored because it appears to solve several limitations of the Dreamer 3 / RSSM route:

- It preserves structure across both **time** and **agents**, rather than compressing all history into a small latent state
- It gives more direct access to **token-level representations**, including possible agent-specific dynamics tokens
- It is more naturally compatible with variable agent counts through transformer masking/padding
- It aligns well with the actual research need: representing rich interaction dynamics rather than only reconstructing single-step latent summaries

Known tradeoffs:

- higher compute and memory cost
- still operates with a fixed history window
- no official public codebase was noted in the slides

## Current Proposed Implementation Direction

The most important near-term plan, based on the latest meeting, is:

1. Start from Dreamer 4
2. Train the tokenizer without learned scene/task/policy-agent tokens, using bottleneck latents plus the ordinary selected-agent, traffic-light, and map tokens needed to encode the scene state.
3. Train the dynamics/world model on tokenizer latents.
4. Add downstream scene/task/policy-agent tokens only when finetuning the dynamics model into a policy.
5. Explore whether bottleneck latents, selected-agent state tokens, and downstream policy finetuning tokens serve as useful interaction representations.

## World Model vs. Contrastive Learning

The current stance is not "one replaces the other," but rather:

- **World model first** to learn causally grounded structure
- **Contrastive refinement second** if needed to sharpen representation geometry

Working comparison:

- Contrastive learning can emphasize distinctions between scenes, but depends heavily on how positive and negative pairs are defined
- World-model learning can capture detailed structure and causal regularities, but may also encode noise without additional pressure

Practical interpretation:

- Dreamer 4-style tokenizer/dynamics first; downstream scene/task/policy-agent tokens only during policy finetuning
- contrastive loss is a likely add-on, not the first milestone

## Open Technical Questions

- How exactly should downstream scene/task/policy-agent tokens be trained during dynamics-to-policy finetuning: policy/reward/value losses, auxiliary probing, contrastive loss, or some combination?
- What should count as the best scene-level objective for driving interaction representation?
- How useful are the per-agent output tokens in practice for representing agent dynamics?
- How much map information should enter the world model, and at what stage?
- Which simulator or environment path is most practical for downstream RL experiments given current GPUDrive and PufferDrive limitations?
- What evaluation will best show that the learned representation is truly interaction-aware rather than only predictive?

## Environment and Data Notes

- Waymo motion data is the central dataset mentioned in the slides
- Special interaction scenes of interest include:
  - highway merge
  - multi-lane to two lanes
  - unprotected left turn
- GPUDrive has been used for visualizing special scenes
- GPUDrive currently has CUDA/NVRTC-related compilation issues
- PufferDrive is still of interest for scalable RL experimentation

## What A Future Codex Session Should Know Immediately

If a future session starts from this file, the most important facts are:

- Yufeng is studying **interaction-aware driving representations**
- The active implementation priority is **Dreamer 4**
- The tokenizer should not include learned scene/task/policy-agent tokens; add them only when finetuning the dynamics model into a policy
- Dreamer 3 / RSSM and pure contrastive approaches are important baselines or comparisons, but not the first implementation target
- The session should treat the latest plan from **2026-04-21** as the current source of truth unless newer notes override it

## Suggested Immediate Next Steps

- Gather the exact Dreamer 4 paper/code references that match the transformer world-model design described in the slides
- Translate the downstream scene/task/policy-agent-token idea into a concrete dynamics-to-policy finetuning spec for driving data
- Define the input tensor structure for multi-agent trajectories and map features
- Decide the first downstream objective for validating representation quality
- Record implementation choices and unresolved questions in a separate running note as work begins
