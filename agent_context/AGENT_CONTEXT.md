# Agent Context

Last updated: 2026-11-12

## Latest Update: Bias Controller, Ceiling Test, DPO and Exogenous-Agent Notes, 2026-11-12

- **The z route did not give order control.** Experiments 1a (frozen WM, oracle z) and 1b-v2 (dense z, unfrozen denoiser) both failed: reversing z did not flip the pass order. Results are in §8.8 of the z plan and §1 of the bias plan.
- **Replacement plan, agreed with the user:**
  - a learned bias controller reads the conflicts in this replan's draft plan;
  - it outputs δ for each conflicting pair, added in three places (encoder, `agent_attn`, `scene_cross`);
  - it is trained on closed-loop outcomes, not log likelihood.
- **Ceiling test result (frozen 85k, 512 scenes, K = 8; 356 scenes have a target pair):**
  - the best δ per scene cuts real collisions 0.537 → 0.373 (−30%), the pre-registered middle zone, so only same-direction (−36%) and crossing (−30%) pairs go forward; VRU drops only −15%;
  - global δ moves collisions to other agents;
  - most of the remaining 0.373 comes from pairs that were never steered.
- **Follow-up discussion, not decided:**
  - DPO on winner/loser plans from the same state, restricted to the colliding pair, as an alternative to labels from δ search.
  - The generate-all assumption: every agent is treated as following the joint plan, which fails for exogenous agents such as a human driver or the policy under test. Possible fixes are an exogenous flag, multi-draft prediction and closed-loop training with log-replay or perturbed agents.
  - The existing focus mode gives the model the exogenous agent's 4 s future, so it is an oracle, not a human proxy.

Files:

```text
agent_context/BIAS_CONTROLLER_PLAN_2026_11_12.md
agent_context/CLOSED_LOOP_PREFERENCE_AND_EXOGENOUS_AGENTS_2026_11_12.md
waymo/pair_order_z/runs/bias_ceiling_85k_20261112/summary.md
```

## Latest Update: Pass-Order Variable z for WM Interaction Fine-Tuning, 2026-11-09

Plan agreed with the user; not yet implemented.

- Problem: the laneconn90k MON-8 `best_ade.pt` collides 50–80% in some
  scenes regardless of resampling. Violation-aware MoN selection did not help,
  because the violating samples get no gradient. So the fix targets the
  representation, not sample selection.
- Design:
  - a supervised pair variable z_ij ∈ {i first, j first, no conflict}, from the
    logged conflict entry order, relabeled at every replan;
  - a pair head (motion/conflict/ROW pair features plus agent tokens), trained
    with CE;
  - z enters the denoiser `agent_attn` through a zero-init, bounded bias
    g(z, pair features);
  - g has no target of its own: it is trained by the WM flow loss with
    teacher-forced GT z and z-dropout, with the WM frozen first.
- Inference: route first, then r and z, then the plan. One z per pair, shared
  by both cars, re-evaluated with hysteresis at each replan.
- Off-log branches: r and z are recomputed from the rollout's own route.
  DAgger-style rule labels (car in zone = first, no gap → non-holder yields,
  can't stop → goes) cover states the WM visits but the log does not.
  Timing-shift interventions are added.
- Steps:
  1. WM failure table by ROW type × margin;
  2. standalone head, A1 (own features) vs A2 (frozen WM tokens);
  3. attach to the frozen WM;
  4. unfreeze;
  5. interventions / DPO.
- 2026-11-10 update (§8 of the plan): the first WM round is a finetune, not a
  retrain, starting from the pre-MoN stage-1 `best.pt` (laneconn scratch100k,
  step 85000, EMA weights). Experiment 1 = oracle z (logged pass order) with the WM frozen,
  training only the zero-init z→bias module g with the stage-1 flow loss, then
  512-scene closed-loop eval with z unknown / oracle / flipped. Experiment 2 =
  A2 head (frozen WM tokens, 3 classes). §8.1 lists what the small rowz models
  changed (no ROW input in round 1, no same-direction z, reaction needs §3.6).
  Paths in the plan are relative to `tri30/`.

Full plan:

```text
agent_context/ROW_PAIR_ORDER_Z_WM_FINETUNE_PLAN_2026_11_09.md
```

## Latest Update: RelativeAgentBias vs Agent-Agent Attention, 2026-11-05

Analysis of the laneconn90k MON-8 `best_ade.pt` world model.

- `RelativeAgentBias` is a pose-only geometric prior: no velocity, no time.
  It is one `(B, 8 heads, N, N)` tensor per replan, shared by all 4 encoder
  self-attention layers, all 8 denoiser `agent_attn` layers and the agent part
  of `scene_cross`.
- It accounts for ~40% of across-key logit variance, is uncorrelated with QK,
  and moves ~30% of attention mass when removed in one layer.
- It ties each head index to a consistent geometric role across layers.
- Attention weights do not track which car actually drives behavior.
- The dynbias FutureAgentBias was barely used without a matching training
  signal.

Full structure, tables, caveats and planned experiments:

```text
agent_context/RELATIVE_AGENT_BIAS_ATTENTION_ANALYSIS_2026_11_05.md
```

Scripts and raw results: `/p/yufeng/tri30/wm_interaction_probe/`.

## Latest Update: WOSAC Improvement and Anti-Collapse Fine-Tuning, 2026-08-12

The current confirmed world-model behavior is that Stage 1 retains strong
noise-dependent stochasticity, while long-horizon rollout fine-tuning lowers
ADE but collapses different noise samples toward nearly identical futures.
The current 1000-view internal WOSAC-style evaluation also identifies high
collision and offroad rates as the main physical-realism failures.

The agreed first training study keeps the architecture fixed and compares:

```text
Control: current rollout fine-tuning
A: joint Minimum-over-N (N=4)
B: MoN + all-sample collision/offroad/kinematic losses
C: B + Stage 1 diversity retention
```

Full evidence, loss definitions, controlled experiment matrix, evaluation
criteria, and implementation order:

```text
agent_context/WOSAC_IMPROVEMENT_PLAN_2026_08_12.md
```

## Source

- Combined boss-meeting Google Slides deck covering 2026-04-21, 2026-04-09, and 2026-03-24:
  [Google Slides](https://docs.google.com/presentation/d/14FdjmNIs-73L64M3qDy_P66qVXiMy3ILl8cpyiaAvPg/edit?usp=sharing)

## Research Focus

Yufeng is working on learning interaction-aware representations of driving scenarios, primarily from Waymo motion data. The goal is to build representations that capture multi-agent driving dynamics well enough to support downstream planning, policy learning, and analysis of special traffic interactions.

The current implementation priority is **Dreamer 4**. This is the first major direction to execute, ahead of alternative pathways.

## Latest Update: Matched Interaction Contrastive Learning, 2026-07-20

The current interactive-probe result is that the tested tokenizer latent `Z`
does not show stable incremental interaction information beyond the strong raw
kinematics baseline under the existing probe. This conclusion is limited to the
current probe/readout; it is not proof that `Z` contains no interaction
information.

The next representation-improvement experiment is direct **matched supervised
contrastive tokenizer fine-tuning**:

```text
learned focus/candidate agent slot queries
  -> read pair token only from Z
  -> matched supervised contrastive loss
```

Raw pair history does not enter the learned pair readout. It is used only to
find different-scene positives and hard negatives. Matching uses two seconds of
time-aligned causal history in a fixed query-time focus frame, with queries one,
two, or three seconds before a geometry-defined interaction event. Version 1
does not use DTW.

The implemented 5k screening cache contains:

```text
19,941 event-aligned pair samples
13,242 high-confidence response samples
10,189 trainable anchors with both positive and hard negative
25,584 positive edges
32,776 hard-negative edges
52,968 ordinary-negative edges
```

Code and cache:

```text
waymo/interaction_contrastive_learning
waymo/cache/interaction_contrastive_learning_5k
```

Full decisions, algorithms, counts, caveats, and the next training plan:

```text
agent_context/MATCHED_INTERACTION_CONTRASTIVE_PLAN_AND_5K_DATA_2026_07_20.md
```

This dated note supersedes older generic statements that contrastive learning
should only be considered after completing all world-model work. The current
experiment is a targeted tokenizer-representation refinement motivated by the
probe result; Dreamer 4 remains the overall project architecture.

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
