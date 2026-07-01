# Dynamic Memory Representation Plan

Date: 2026-06-25

Project path: `/scratch/baz7dy/tri30/dreamer4/waymo`

## Core Representation Framing

The current tokenizer bottleneck latent should be treated as a **scene-level
dynamic memory**:

```text
encoder(scene) -> Z
Z shape example: [T, N_latents, D_bottleneck] = [32, 64, 64]
```

`Z` is not expected to have one fixed latent slot per agent. A latent slot does
not need to mean "agent 7" or "the focus vehicle." Instead, downstream modules
should read from `Z` using explicit queries:

```text
representation = read(Z, query)
```

The query provides the semantic question, such as:

- focus agent context
- focus-candidate interaction
- pairwise future relation
- policy/value context
- behavior-clustering readout

This avoids forcing the latent slots to be agent-indexed while still allowing
agent-specific and pair-specific representations to be extracted.

## Revised Research Questions

The current evaluation should move beyond reconstruction ADE/FDE. Reconstruction
is useful, but the goal is an interaction-aware representation for future world
model training and downstream policy learning.

### Q1. Relevance

Can the representation identify interaction-relevant agents?

Concrete evaluation:

- For each focus-centered scene, score each candidate agent.
- Evaluate top-k recall, AP, AUROC, and ranking correlation against automatic
  future-interaction targets.
- Compare `pair_raw_only` versus `pair_raw + Z`.

Possible automatic targets:

- future minimum distance to focus
- future time-to-collision or close-risk proxy
- relative order change
- crossing or yielding-like interaction
- future speed change of focus caused by nearby agents

### Q2. Relations

Can the representation predict future pairwise relations and risks?

Concrete evaluation:

- For each pair `(i, j)`, predict:
  - future min distance
  - future relative `dx, dy` at 1s, 3s, 5s
  - close interaction risk
  - crossing, following, overtaking, yielding-like labels
  - future relative bearing or heading change

This tests whether `Z` contains pairwise dynamic information beyond current
geometry.

### Q3. Counterfactuality

Does the representation respond to relevant-agent changes and stay stable under
irrelevant-agent changes?

Concrete evaluation:

1. Compute `Z` for the original scene.
2. Remove or perturb a candidate agent.
3. Recompute `Z`.
4. Compare readout changes for relevant versus irrelevant perturbations.

Expected behavior:

- removing a truly relevant agent should change pair/focus predictions and
  embeddings substantially
- removing an irrelevant agent should have a smaller effect

### Q4. Temporal Sufficiency

Is the representation a useful compact state for predicting future dynamics?

Concrete evaluation:

- Train a latent transition or future-relation predictor from `Z_t`.
- Predict `Z_{t+1:t+H}`, future pairwise relations, or future scene dynamics.
- Compare capacity-matched models using raw state, pooled state tokens, and `Z`.

This question is central for world model training: `Z` should work as a compact
dynamic state, not only as a reconstruction cache.

### Q5. Transfer

Does the representation improve downstream policy, value, or world-model
training?

Concrete evaluation:

- Use `Z` or query-read representations in imitation, policy, value, or world
  model objectives.
- Compare training curves and sample efficiency against raw state and simpler
  hand-designed features.
- Check whether better tokenizer reconstruction also predicts better downstream
  performance.

### Q6. Semantic Geometry

Are similar interaction behaviors close in representation space?

Concrete evaluation:

- Build `h_pair = read(Z, query(i, j))` for many scene-agent pairs.
- Cluster `h_pair` with UMAP plus k-means or HDBSCAN.
- Inspect cluster statistics and representative trajectory visualizations.
- Compare against clustering raw pair features.

Useful cluster statistics:

- future min distance
- future TTC proxy
- relative heading change
- crossing/following/overtaking/yielding proxies
- focus speed change
- candidate speed change
- relative longitudinal order change

### Q7. Multi-Agent Interaction Sets

Can the representation discover and summarize a variable-size set of agents
that jointly affect the focus agent's decision?

Motivation:

- Some interactions are genuinely pairwise, such as a focus vehicle following a
  single lead vehicle.
- Other interactions are group-level, such as highway merging, where the focus
  vehicle may need to reason about the front vehicle, rear vehicle, adjacent
  vehicle, and the target-lane gap together.
- The number of participating agents is not known ahead of time and should not
  be hard-coded.

Concrete evaluation:

- Build pair readouts `h_pair_0j = read(Z, query(focus, j))` for all candidate
  agents around the focus.
- Predict a learned participant weight or mask `w_j` for each candidate.
- Aggregate the weighted pair representations as a set to produce a group-level
  interaction representation `h_group`.
- Use `h_group` to predict focus-level outcomes and multi-agent event outcomes.
- Evaluate whether the learned participant weights align with counterfactual
  sensitivity: perturbing or removing high-weight agents should change the
  focus prediction more than perturbing low-weight agents.

Possible targets:

- focus future trajectory
- focus speed change or braking
- lane-change / merge / gap-choice outcome
- whether a rear or adjacent vehicle yields
- whether a candidate vehicle overtakes the focus
- future relative order among focus, front vehicle, and rear vehicle

Metrics:

- focus trajectory ADE/FDE or negative log likelihood
- lane-change / merge / gap-choice accuracy
- AP/AUROC/top-k recall for participant relevance
- sparsity and calibration of the learned participant mask
- counterfactual sensitivity gap between high-weight and low-weight agents

### Analysis Dimension: Compression vs. Sufficiency

Track whether larger latents improve interaction evaluations or merely improve
trajectory reconstruction.

Important comparisons:

- `lat16_b32`, `lat16_b64`, `lat32_b64`, `lat64_b64`
- `lat16_b128`, `lat64_b32`, `lat64_b128`
- possibly `lat128_b64` if `lat64_b128` continues to improve

Desired outcome:

- reconstruction improves with capacity
- Q1-Q7 also improve or remain strong
- representation is not just a high-capacity trajectory cache

## Query And Readout Design

### Query Source

For the first probe, use raw agent state features rather than encoder agent
tokens. This gives the cleanest test of whether `Z` contains useful interaction
information.

Current agent feature order in the Waymo vector dataset:

```text
x, y, speed, vx, vy, valid, yaw, type
```

Recommended raw pair query features:

```text
agent_i:
  x, y, vx, vy, speed, sin(yaw), cos(yaw), valid, type

agent_j:
  x, y, vx, vy, speed, sin(yaw), cos(yaw), valid, type

relative i->j:
  dx, dy
  distance
  dvx, dvy
  relative speed
  bearing
  heading difference
  longitudinal offset
  lateral offset
  is_front
  is_same_direction
  is_crossing_like
```

Using encoder agent tokens can be a later ablation, but it risks letting the
probe bypass `Z`. If tested, compare:

```text
raw query only
raw query + Z
encoder agent token only
encoder agent token + Z
```

### Current-Only Query

The cleanest first version uses only the current state:

```text
q_ij = MLP([state_i_t, state_j_t, relative_state_ij_t])
h_ij = CrossAttention(query=q_ij, key/value=Z_flat)
prediction = MLP(h_ij)
```

This tests whether `Z` provides dynamic scene memory beyond current pair
geometry.

### Short-History Query

A more realistic second version uses recent history, for example the last 8
frames:

```text
hist_i = states_i[t-7:t]
hist_j = states_j[t-7:t]
rel_hist_ij = relative_states_ij[t-7:t]

q_ij = TemporalEncoder([hist_i, hist_j, rel_hist_ij])
h_ij = CrossAttention(query=q_ij, key/value=Z_flat)
prediction = MLP(h_ij)
```

This is closer to downstream policy and world-model usage, because current state
alone may not reveal intent.

### Memory Flattening

For an encoder output:

```text
Z: [B, T, N_latents, D]
```

flatten time and latent slots:

```text
Z_flat: [B, T * N_latents, D]
```

Then let pair or focus queries attend to this memory.

### Pair Representation

The hidden readout from the probe is the pair-level interaction representation:

```text
h_pair_ij = read(Z, query(i, j))
```

For focus-agent partner relevance, set `i = 0` and evaluate all candidates:

```text
score_j = head(h_pair_0j)
j = 1..K-1
```

For full scene relation prediction, evaluate all pairs:

```text
H_pair: [B, K, K, D_pair]
```

### From Pair To Multi-Agent Group Representation

Pair representation is the base case. For focus-centered prediction, first read
one pair representation for each focus-candidate pair:

```text
h_pair_0j = read(Z, query(focus, j))
j = 1..K-1
```

Then predict how relevant each candidate is to the current focus decision:

```text
score_j = relevance_head(h_pair_0j)
w_j = sparse_attention(score)_j
```

Here `sparse_attention` can be implemented with `entmax`, `sparsemax`, or
`sigmoid + sparsity regularization`. A plain softmax is a useful baseline, but
it always assigns nonzero weight and always distributes total mass across the
candidates, even when most candidates are irrelevant.

After weighting the pair representations, aggregate them as an unordered set:

```text
x_j = w_j * h_pair_0j
h_group = SetTransformer({x_j})
prediction = head(h_group)
```

`SetTransformer` means self-attention over a set of agent-pair embeddings,
without agent-index positional encoding, followed by a permutation-invariant
pooling/readout. A minimal first implementation can be:

```text
TransformerEncoder(no positional encoding, candidate mask)
+ attention pooling or masked mean pooling
```

This lets the model represent interactions that are not decomposable into
independent pairs. For example, in highway merging, the focus vehicle may need
to jointly reason about the target-lane front vehicle, target-lane rear vehicle,
and the available gap.

Important base-case interpretation:

```text
one relevant candidate:
  h_group semantically reduces to h_pair_0j
```

This equivalence is semantic rather than guaranteed numeric identity. If the
SetTransformer and pooling are designed as identity for a single valid token,
then `h_group = h_pair_0j`. Otherwise, with learned projection, residual blocks,
or pooling, the single-candidate case is better understood as:

```text
h_group = group_readout({h_pair_0j})
```

which plays the same role as the pair representation but may be a learned
transformed version of it. With multiple relevant candidates, `h_group` becomes
the natural extension of `h_pair_0j` from a two-agent interaction to a
variable-size interaction set.

### Focus Representation

A focus-level readout can be built with a focus query:

```text
h_focus = read(Z, query(focus_state))
```

This can be used for downstream policy, value, behavior clustering, or focus
future prediction.

## Probe Training As Both Evaluation And Readout Learning

Training a relation probe serves two purposes:

1. It tests whether `Z` contains interaction information.
2. It learns how to query `Z` and turns the hidden layer `h_pair` into a useful
   pair-level interaction embedding.

The probe should predict objective future-relation targets, not manually defined
scene similarity:

```text
head(h_pair) -> future relation targets
```

Recommended targets:

- future minimum distance
- future relative `dx, dy`
- close-risk logit
- TTC proxy
- relative order change
- speed-change/yielding proxy

After training, use the hidden representation `h_pair` for clustering and
nearest-neighbor analysis.

## Baselines And Sanity Checks

Always compare:

```text
Probe A: pair_raw_only -> future relation
Probe B: pair_raw + Z -> future relation
Probe C: pair_raw + shuffled_Z -> future relation
```

Interpretation:

- If B > A, `Z` contributes useful scene dynamic information.
- If C drops back near A, the probe is using the correct scene memory.
- If A ~= B, the target may be solvable from current geometry alone, or `Z` may
  not contain the needed interaction information.

For clustering, compare:

```text
cluster(raw pair features)
cluster(h_pair from Z)
```

If `h_pair` clusters align better with future interaction modes than raw pair
features, this supports Q6.

## Full-Future Leakage Warning

The current tokenizer is trained on 32-step chunks. If `Z` is encoded from a
window that includes future frames relative to the query time, a probe may be
diagnostic but not predictive.

Use two evaluation settings:

### Diagnostic Setting

```text
Z_full = encoder(scene[0:32])
query = current or early-frame pair query
target = relation inside the same 32-step chunk
```

This tests whether `Z` contains relation information.

### World-Model Setting

```text
Z_past = encoder(scene[past/current only])
query = current pair query
target = future relation after current
```

This is the clean setting for future world model and policy transfer.

## Tokenizer Direction Based On Current Experiments

Current empirical takeaway as of 2026-06-25:

- `lat64_b64` is the strongest no-agent-token bottleneck tokenizer so far.
- Increasing latent capacity improves reconstruction, suggesting bottleneck
  capacity still matters.
- `all_agent_token` decoder can produce strong early reconstruction, but it
  bypasses part of the bottleneck and is not a clean representation objective.
- `focus_agent_token` decoder variants underperform and should not be a main
  path.

Working stance:

- Keep the canonical tokenizer as `decoder_agent_token_mode=none`.
- Treat all-agent-token decoder as an upper bound or ablation, not as the main
  representation.
- Use query/readout probes to test whether the no-agent-token `Z` captures
  dynamic interaction information.

## Near-Term Experimental Plan

### Tokenizer Capacity Experiments

Run and compare:

```text
lat16_b128
lat64_b32
lat64_b128
```

Purpose:

- `lat16_b128`: same total capacity as `lat32_b64`, tests width versus slots.
- `lat64_b32`: same total capacity as `lat32_b64`, tests slots versus width.
- `lat64_b128`: tests whether the current best `lat64_b64` benefits from more
  width.

Possible next experiment:

```text
lat128_b64
```

only if `lat64_b128` improves Q1-Q7, not just reconstruction.

### First Probe Experiment

Freeze the best tokenizer encoder, likely `lat64_b64` at its best validation
checkpoint.

For each focus-centered sample:

1. Compute `Z`.
2. For each candidate `j = 1..K-1`, build raw current pair query `(focus, j)`.
3. Read `h_pair_0j = read(Z, query(0, j))`.
4. Predict:
   - future min distance
   - close-risk logit
   - final relative `dx, dy`
5. Evaluate against raw-only and shuffled-Z baselines.

Metrics:

- regression MAE for min distance and relative displacement
- AUROC/AP for close-risk
- top-1/top-3 recall for important partner ranking
- correlation with future min distance

### Multi-Agent Group Probe Experiment

After the first pair probe works, extend from two-agent pair prediction to
variable-size group prediction.

For each focus-centered sample:

1. Compute `Z`.
2. For each candidate `j = 1..K-1`, read
   `h_pair_0j = read(Z, query(focus, j))`.
3. Predict participant scores and weights:
   `score_j = relevance_head(h_pair_0j)`,
   `w_j = sparse_attention(score)_j`.
4. Build weighted set tokens `x_j = w_j * h_pair_0j`.
5. Aggregate with a SetTransformer-style set readout:
   `h_group = SetTransformer({x_j})`.
6. Predict focus-level and event-level targets from `h_group`.

Recommended targets:

- focus future trajectory
- focus speed change or braking
- lane-change / merge outcome
- target-lane gap choice, if a gap label can be derived automatically
- whether rear or adjacent vehicles yield or overtake
- future order changes among focus and high-weight candidates

Baselines:

- focus raw state only
- raw focus + hand top-k nearest candidates
- independent pair heads plus max/mean pooling
- learned pair relevance + simple weighted pooling
- learned pair relevance + SetTransformer group readout
- learned pair relevance + SetTransformer group readout + shuffled `Z`

Expected behavior:

- For truly pairwise scenes, the learned group readout should collapse toward a
  single high-weight candidate and behave like the pair representation.
- For multi-agent scenes, the group readout should outperform independent pair
  prediction and simple pooling, especially on focus future prediction,
  gap-choice, yielding, and overtaking outcomes.
- Counterfactual perturbations of high-weight candidates should cause larger
  prediction changes than perturbations of low-weight candidates.

### Clustering Experiment

After training the relation probe:

1. Collect `h_pair_0j` for many focus-candidate pairs.
2. Run PCA/UMAP and k-means or HDBSCAN.
3. For each cluster, report automatic relation statistics.
4. Save representative trajectory visualizations per cluster.
5. Compare against clustering raw pair features.

Goal:

- Check whether `h_pair` groups similar interaction behaviors, not just similar
  raw geometry.

### Counterfactual Sensitivity Experiment

For each scene:

1. Identify likely relevant and irrelevant candidates using future-relation
   targets.
2. Remove or perturb one candidate.
3. Recompute `Z`.
4. Compare changes in:
   - `h_focus`
   - `h_pair`
   - relation predictions
   - partner ranking

Expected:

- relevant perturbations produce larger changes than irrelevant perturbations

### Downstream Transfer Experiment

Use readouts from `Z` as inputs to:

- future scene/world-model prediction
- policy/value/imitation heads
- focus future trajectory prediction

Compare:

- raw state
- pooled encoder agent tokens
- `Z` pooled
- `h_focus = read(Z, focus_query)`
- `h_pair` aggregate over top-ranked partners
- `h_group = SetTransformer({w_j * h_pair_0j})`

## Important Design Principle

Do not judge the tokenizer only by reconstruction ADE/FDE.

The target representation should satisfy:

```text
high reconstruction quality
+ strong Q1-Q7 probe results
+ useful downstream transfer
+ no obvious reliance on decoder skip leakage
```

`Z` should be a compact dynamic memory for interaction, not merely a compressed
trajectory cache.

## Parallel Routes For Improving Interaction-Aware `Z`

If query-based probes show that `Z` does not contain enough interaction
information, there are several parallel ways to improve or reorganize the
representation. Contrastive learning is only one route. World-model prediction
and control-system readouts are separate routes with different goals.

### Route A: Contrastive Interaction Geometry

Purpose:

- Shape the geometry of query-conditioned embeddings.
- Make scenes, pairs, or groups with similar future interaction behavior close
  in representation space.

Core idea:

```text
h_pair = read(Z_t, query(i, j))
h_group = read(Z_t, query(focus, candidates))
```

Build an interaction signature from future behavior:

- future relative `dx, dy` sequence
- distance sequence and future minimum distance
- relative order change
- focus and candidate speed changes
- TTC, close-risk, yielding, overtaking, and crossing proxies

Then define one behavior-space similarity between signatures and train a soft
contrastive loss so embedding similarity matches that behavior similarity.

Important clarification:

- The behavior-space metric, such as DTW, soft-DTW, event distance, or a
  dynamics-aware distance, defines the target similarity.
- The contrastive loss is only a training objective that makes
  `sim(h_a, h_b)` approximate that target similarity.
- DTW is a useful baseline for timing-tolerant sequence comparison, but it is
  not necessarily the final or best notion of interaction similarity.

Training options:

- freeze the tokenizer and train only the readout/projection
- fine-tune the tokenizer with reconstruction anchor plus contrastive loss

This route mainly answers:

```text
Are similar interaction behaviors organized near each other in representation
space?
```

### Route B: World-Model Predictive Latents

Purpose:

- Use the world model to unfold future latent states instead of requiring
  current `Z_t` to directly contain all future interaction outcomes.
- Improve `Z` as a predictive dynamic state when joint fine-tuning is allowed.

Motivating diagnostic:

```text
query current Z_t -> current interaction property: good
query current Z_t -> future interaction property: poor
```

This suggests `Z_t` may encode the current scene well, but may not yet be a
sufficient predictive state. In that case, train or evaluate:

```text
Z_t -> world model -> Zhat_{t+1:t+H}
query Zhat_{t+k} -> interaction properties at future time k
```

Useful comparison:

```text
oracle future Z_{t+k} + probe
predicted future Zhat_{t+k} + same probe
```

Interpretation:

- If the probe works on oracle future `Z_{t+k}` but fails on predicted
  `Zhat_{t+k}`, the transition/world model is the main bottleneck.
- If the probe fails even on oracle future `Z_{t+k}`, the latent itself may not
  stably encode the interaction property.

Training variants:

```text
Option A: frozen tokenizer
  train only the world model and interaction heads
  tests whether current Z is sufficient

Option B: frozen tokenizer + adapter
  Z_base = frozen_encoder(scene)
  S = adapter(Z_base)
  world model operates on S
  learns a dynamics-aware state without risking reconstruction

Option C: joint fine-tuning
  update encoder/tokenizer with reconstruction + transition + interaction loss
  truly reshapes Z into a more predictive interaction-aware state
```

Possible objective for Option C:

```text
L = L_rec
  + L_latent_transition
  + L_future_interaction_property
```

This route mainly answers:

```text
Can the latent be rolled forward into future states from which interaction
properties are readable?
```

### Route C: Control-System / Local Dynamics Readout

Purpose:

- Test whether current `Z_t` contains the local interaction mechanism, not just
  current geometry.
- Extract an action-conditioned local dynamics approximation from `Z_t`.

Core idea:

```text
h_pair = read(Z_t, query(i, j))
h_group = read(Z_t, query(focus, candidates))

g(h_pair or h_group, s_t) -> A_t, B_t, c_t, Sigma_t, pi_t
```

where:

- `s_t` is the current pair or group state, such as relative position,
  relative velocity, speed, and heading
- `A_t` captures natural local interaction evolution and agent-agent influence
- `B_t` captures response to ego/focus action
- `c_t` captures residual drift
- `Sigma_t` captures uncertainty
- `pi_t` captures multimodal local dynamics

One-step local model:

```text
s_{t+1} = A_t s_t + B_t u_t + c_t + noise
```

Training:

- `s_t` can be an input because it is the current operating point.
- `s_{t+1}` and future states are targets, not inputs.
- Multi-step rollout losses can be used to make the local dynamics meaningful
  beyond one frame.

Extensions:

```text
time-varying dynamics:
  predict A_{t:t+H}, B_{t:t+H}, c_{t:t+H}

re-linearized rollout:
  read new A/B from each predicted state or latent

mixture dynamics:
  predict several modes, such as yield / no-yield / brake / pass
```

This route is not mainly for long-horizon prediction by a single fixed linear
model. It is for reading out a local, action-conditioned interaction mechanism
from `Z_t`. Long-horizon prediction should still use rollout, re-linearization,
or a world model.

This route mainly answers:

```text
Does Z_t contain local interaction dynamics and action-response structure?
```

### How The Routes Relate

These routes can share targets and readouts, but they are conceptually
different:

```text
Contrastive route:
  organize interaction embeddings by behavior similarity

World-model route:
  roll Z forward and query future interaction properties

Control-system route:
  read local action-conditioned dynamics from current Z
```

They can be used independently or combined. For improving the tokenizer itself,
the relevant loss must update the encoder/tokenizer. If the tokenizer is frozen,
these routes evaluate the current `Z` or learn an adapter/readout on top of it,
but they do not directly improve `Z_base`.
