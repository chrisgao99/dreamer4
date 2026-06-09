# OOI Focus-Conditioned Interaction Scorer Plan

Date: 2026-06-04

Project path: `/p/yufeng/tri30/dreamer4/waymo`

## Motivation

The current Waymo OOI tokenizer dataset centers each sample on an object of interest
(OOI). Slot 0 is the focus agent, and the other OOI agents are prioritized into
early slots, often slot 1. This is useful for making interaction-rich training
examples, but it leaks the answer to the model: the model can infer that the
nearest or most relevant interaction partner is usually in a fixed early slot.

The larger goal is different:

Given one focus vehicle, the model should decide which surrounding agents matter
for that focus agent's future dynamics.

Examples:

- A left-turning vehicle should attend to oncoming straight traffic.
- A lane-changing vehicle should attend to the target-lane leader/follower.
- A slowing or yielding vehicle should attend to the conflict partner that causes
  the change in behavior.

The desired contribution is therefore not only "train Dreamer4 on OOI data", but:

> A focus-conditioned interaction tokenizer that learns to identify and represent
> surrounding agents that are dynamically relevant to a focus agent.

## Step 1: Remove Slot Leakage

### Problem

The current OOI-centered preparation uses `priority_src_indices=ooi_src_indices`.
Inside `filter_scenario_around_focus`, agent selection is roughly:

1. focus agent first;
2. other priority OOI agents next;
3. other OOI agents;
4. tracks-to-predict agents;
5. nearby agents by distance.

This means the model can exploit slot order. In many two-OOI scenes, slot 1 is
implicitly the interaction partner of slot 0.

At test time, however, the model will not know which surrounding agent is the
most relevant partner. It must infer this from geometry, history, map context,
and dynamics.

### Plan

Keep only one privileged slot:

- slot 0 = focus agent;
- slots 1..K = candidate surrounding agents.

Do not use OOI membership to force candidates into early slots for the main
interaction-learning dataset.

Candidate ordering options:

- current/history distance from focus;
- front-of-focus first, then distance;
- tracks-to-predict first if this is available at test time;
- deterministic geometric ranking;
- random shuffle of non-focus slots during training.

The safest v1 is:

- slot 0 fixed as focus;
- slots 1..K selected by test-time-available geometry;
- shuffle slots 1..K during training as augmentation;
- keep OOI labels only as supervision targets, not as ordering input.

### Expected Outcome

The model can no longer solve interaction relevance from slot identity. Any
successful interaction score must come from agent state, relative geometry, map
context, and learned dynamics.

## Step 2: Add Focus-Conditioned Interaction Scores

### Definition

For each focus-centered sample, the model predicts one score per candidate agent:

```text
score[k] = P(candidate agent k is interaction-relevant to focus)
```

where:

```text
k = 1..K-1
```

Slot 0 is the focus agent and should be masked out for scoring.

At inference time, the model returns top-ranked agents:

```text
top_interaction_agents = argsort(score, descending=True)[:top_k]
```

### Interaction Scorer Architecture

Use the encoder's current-time agent tokens.

Let:

```text
h0 = encoder token for focus agent at current timestep
hk = encoder token for candidate agent k at current timestep
gk = explicit focus-candidate geometry features at current/history time
```

Then predict:

```text
score_logit[k] = MLP([h0, hk, h0 * hk, abs(h0 - hk), gk])
score[k] = sigmoid(score_logit[k])
```

The explicit geometry features should use only information available at test
time. A useful first set:

```text
dx, dy
distance
relative_vx, relative_vy
relative_speed
heading_delta
longitudinal_offset
lateral_offset
is_front
is_same_direction
is_crossing_like
```

These features make the scorer easier to train and easier to inspect. The
transformer still learns richer context from agent, map, and light tokens.

### Hard Supervision

Use Waymo OOI membership as weak interaction supervision.

For a focus sample:

```text
y_k = 1 if candidate k is another OOI agent in the same raw scenario
y_k = 0 otherwise
```

Implementation source:

- `agent_objects_of_interest[k]` in the NPZ;
- `k != 0`;
- `agent_mask[k] == True`.

This is the cleanest first target because it directly matches the current data
source. It also gives a clear evaluation target:

```text
Can the model recover the other OOI agents without being told their slots?
```

### Soft Supervision

OOI labels are useful but incomplete. Some dynamically relevant agents may not be
Waymo OOI. Add optional soft targets based on future interaction cues:

```text
target_k = max(
    OOI_target,
    close_score,
    crossing_score,
    following_score,
    cutin_score,
    yield_score,
)
```

Recommended v1 target values:

```text
1.0 if candidate is another OOI
0.7 if future min distance to focus <= 5m
0.5 if future min distance to focus <= 10m
0.5 if crossing/conflict-like geometry
0.4 if following/leading-like geometry
0.4 if cut-in/merge-like geometry
0.0 otherwise
```

Important distinction:

- The target may use future trajectories because it is a training label.
- The score head input should not use future trajectories if the intended test
  setting is online/current-time inference.

For that reason, compute the score from current/past tokens and current/history
geometry, not from future-pooled tokens.

### Loss

Start with binary cross entropy:

```text
loss_score = BCEWithLogits(score_logit[k], target_k)
```

Mask invalid candidates:

```text
candidate_mask = agent_mask & (slot_index != 0)
```

Add an optional ranking loss so positives are ranked above negatives:

```text
loss_rank = max(0, margin - score_logit[pos] + score_logit[neg])
```

Overall tokenizer objective:

```text
loss = reconstruction_loss
     + lambda_score * loss_score
     + lambda_rank * loss_rank
```

Suggested initial weights:

```text
lambda_score = 0.2
lambda_rank = 0.1
margin = 0.5
```

Keep the interaction losses modest at first so they shape the representation
without destabilizing reconstruction.

### Metrics

Report interaction retrieval metrics in addition to reconstruction loss:

```text
Recall@1
Recall@3
Recall@5
Average Precision
top-1 score positive rate
mean positive score
mean negative score
```

The key experiment:

Compare a slot-leaking dataset against the non-privileged candidate dataset. The
non-privileged setting is the real test of whether the scorer learned relevance.

## Step 3: Add Dynamics Pressure To The Representation

### Problem

Pure reconstruction does not force the latent representation to encode influence.
The model can reconstruct each agent's trajectory independently or rely on easy
scene statistics.

To claim an interaction representation, the representation should be useful for
predicting focus-candidate dynamic relationships.

### Pairwise Future Objectives

For each focus-candidate pair, predict compact future interaction quantities:

```text
future_min_distance
time_to_closest_approach
relative_displacement_at_horizon
future_heading_delta
will_cross_close
will_follow_or_lead
will_cut_in_or_merge
```

These can be generated from existing trajectories in the NPZ files.

The pairwise head can share the same pair embedding as the interaction scorer:

```text
pair_repr[k] = [h0, hk, h0 * hk, abs(h0 - hk), gk]
```

Then:

```text
pair_prediction[k] = MLP(pair_repr[k])
```

This makes the score more than a label classifier: it becomes tied to concrete
future dynamics.

### Masked-Agent Objective

Mask or drop one candidate agent from the encoder input and measure whether the
focus future reconstruction/prediction becomes worse.

This supports a counterfactual definition of relevance:

```text
An agent matters if removing it changes the model's prediction for the focus.
```

Possible training variant:

1. Run normal encoding.
2. Run encoding with candidate k masked.
3. Encourage the scorer to rank high agents whose removal causes large focus
   future error increase.

This is more expensive than the simple score head, so it should be a later step.

### Contrastive Objective

Use interaction labels to shape representation geometry:

```text
positive pair = focus and interaction-relevant candidate
negative pair = focus and non-relevant candidate
```

Train with a contrastive/ranking objective:

```text
sim(focus, positive) > sim(focus, negative)
```

This can help make the latent space interpretable:

- interacting agents cluster near the focus representation;
- non-interacting nearby agents remain separable.

### Counterfactual Evaluation

After training, evaluate whether top-ranked agents actually matter:

1. Predict focus future with all candidates.
2. Remove top-ranked candidate.
3. Remove random candidate.
4. Compare focus prediction degradation.

A strong interaction scorer should produce larger degradation when removing its
top-ranked agents than when removing random nearby agents.

## Recommended V1 Implementation

Build the smallest version that proves the idea:

1. Create an OOI-centered v2 dataset where only slot 0 is privileged.
2. Keep `agent_objects_of_interest` and add per-slot `interaction_target`.
3. Add an `InteractionScorer` head to the vector tokenizer.
4. Train with:

```text
reconstruction_loss + 0.2 * score_BCE
```

5. Log:

```text
loss_interaction_score
interaction_recall_at_1
interaction_recall_at_3
interaction_ap
```

6. Compare:

```text
baseline: current OOI-prioritized slots
v2: non-privileged candidate slots
```

If v2 recovers OOI partners with good Recall@K, the project has a clear
contribution beyond applying Dreamer4.

## Longer-Term Project Framing

The final story can be:

> We train a focus-conditioned tokenizer for autonomous driving scenes. Unlike
> generic reconstruction tokenizers, our model explicitly learns to retrieve and
> represent agents that are dynamically relevant to a given focus vehicle. We use
> Waymo OOI labels and trajectory-derived interaction cues as weak supervision,
> remove slot-order leakage, and evaluate interaction retrieval and
> counterfactual prediction sensitivity.

This makes the project direction align with the real driving question:

For this focus vehicle, who should it pay attention to, and why?
