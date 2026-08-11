# Matched Interaction Contrastive Learning Plan and 5k Data Results

Date: 2026-07-20

Project root: `/p/yufeng/tri30/dreamer4`

Implementation: `waymo/interaction_contrastive_learning`

Prepared cache: `waymo/cache/interaction_contrastive_learning_5k`

Status: sample preparation is implemented and validated. Contrastive tokenizer
training has not started.

## 1. Motivation and Current Evidence

The previous interactive probe compared one-timestep raw focus-candidate
kinematics against a raw query that cross-attends to tokenizer latent `Z`.

Main probe observations:

```text
Relevance/type accuracy:
  raw_only: 0.907
  raw_z:    0.895

Relevance AP:
  raw_only: 0.967
  raw_z:    0.964

Relation-type F1 changes from raw_only to raw_z:
  approximately -0.001 to -0.006 across the four classes

Response AP:
  yield:        0.639 -> 0.636
  deceleration: 0.906 -> 0.929
```

The precise conclusion is:

> Under the current probe architecture and evaluation, no stable incremental
> interaction information was detected in `Z` beyond the strong single-step
> raw-kinematics baseline.

This does not prove that `Z` contains no interaction information. The old
`raw_z` probe can ignore `Z` because its pair representation also receives a
strong raw query. Shuffled-`Z`, `Z`-only, multi-seed, and bootstrap-confidence
controls remain important for the final evaluation.

The agreed next experiment is direct matched supervised contrastive fine-tuning
of the tokenizer representation.

## 2. Final Representation Design

The pair representation must be read from `Z` without raw pair geometry:

```text
agent slot queries (focus slot 0, candidate slot j)
  -> pair slot query q_slot_ij
  -> CrossAttention(q_slot_ij, Z_t, Z_t)
  -> pair token h_ij
  -> projection head
  -> normalized contrastive embedding g_ij
```

The existing `TokenizerInteractionAuxHead` in
`waymo/core/vector_tokenizer_decoder.py` is the starting implementation.

Restrictions:

- Raw pair geometry must not enter `q_slot_ij`, `h_ij`, or the projector.
- Raw observed kinematics are used only offline to find matched samples.
- The contrastive projector is discarded after fine-tuning.
- The improved tokenizer encoder and its `Z` are retained.

This prevents the raw-query shortcut from satisfying the contrastive objective
without changing `Z`.

## 3. Matching History

A single pair state does not describe an interaction process. Matching uses a
causal two-second history ending at the selected query timestep.

Each timestep contains 11 features:

```text
relative longitudinal position
relative lateral position
relative longitudinal velocity
relative lateral velocity
sin(relative heading)
cos(relative heading)
focus speed
focus acceleration
focus yaw rate
candidate speed
candidate acceleration
```

Focus position is not included as an absolute feature. Focus motion is retained
through its speed, acceleration, yaw rate, and the evolution of the candidate's
relative trajectory.

## 4. Query-Time Ego Frame

The full history window uses one fixed coordinate frame defined by the focus
agent at the query timestep. For query timestep `t_q`:

```text
relative_position_tau = R(-focus_yaw_tq) * (other_position_tau - focus_position_tau)
relative_velocity_tau = R(-focus_yaw_tq) * (other_velocity_tau - focus_velocity_tau)
```

All history timesteps `tau <= t_q` use the same query-time rotation. The
sequence is not re-centered and re-rotated independently at every history
timestep.

The source NPZ files are originally centered at the Waymo current timestep
(step 10). The builder performs the additional query-time transformation for
every detected event/query pair.

## 5. Event-Relative Alignment

Scenes are not matched by absolute timestep. Each pair first receives a
geometry-defined interaction event time.

For crossing and converging conflicts, the future paths are compared to find
their closest spatial points:

```text
(i*, j*) = argmin ||focus_position_i - other_position_j||
t_event = min(i*, j*)
```

The path conflict must satisfy spatial-overlap and PET thresholds. Heading and
aligned corridor geometry determine whether the event is crossing/oncoming,
converging, or following-like.

For following, the event is the first aligned timestep at which the pair enters
the configured same-direction, same-corridor headway threshold.

Queries are selected at fixed lead times:

```text
t_query = t_event - lead_steps

lead_steps = 10, 20, 30
dt = 0.1 seconds

therefore queries are 1, 2, and 3 seconds before the event
```

Every query uses 20 history steps, or two seconds. A sample is discarded if it
does not have a complete valid history. Event detection may inspect the future
offline, but stored matching history never contains a timestep after `t_query`.

## 6. Response Labels and Pair Attribution

High-confidence response classes are:

```text
goes_first
yields
decelerates
maintains
```

Conflict arrival order is defined from the focus and candidate closest-path
arrival steps. Yield additionally requires an attributed focus deceleration.

The earlier probe labeler computed one focus deceleration flag and could assign
it to several candidate pairs. The new sample builder instead:

1. detects high-confidence focus deceleration episodes;
2. finds pair events within an attribution window;
3. assigns an episode to at most one nearest pair event;
4. marks competing close candidates as ambiguous;
5. excludes ambiguous responses from contrastive matching.

This is precision-oriented: discard uncertain data rather than train with an
incorrect pair attribution.

## 7. Version-1 Matching Metric

DTW is not used in version 1 because histories share a sample rate, queries are
event-relative aligned, and the exact timing of negotiation/deceleration may be
meaningful.

Each feature is robust-standardized:

```text
x_normalized = (x - median) / IQR
```

The aligned history receives an increasing recent-time weight. The query
endpoint is included again to make current-state similarity a strict part of
the distance. The final matching vector has 231 dimensions:

```text
20 timesteps * 11 history features + 11 query-endpoint features = 231
```

Matching uses Euclidean distance over this normalized, time-weighted vector.
Outliers are clipped. Each exact stratum has an adaptive distance caliper based
on different-scene nearest-neighbor distances. Constrained DTW remains a future
ablation.

## 8. Sample Definitions

Exact matching strata are:

```text
lead_steps
relation type
focus agent type
candidate agent type
```

All edges cross scenario IDs.

Positive:

```text
same stratum
same high-confidence response
within the distance caliper
different scene
```

Hard negative:

```text
same stratum
different high-confidence response
within the distance caliper
different scene
```

Ordinary/easy negative:

```text
same lead time
different relation type
different scene
```

Future response decides whether a nearby pair is positive or hard negative, but
future quantities do not enter the matching vector.

## 9. Implemented Code

```text
waymo/interaction_contrastive_learning/legacy/pair_samples.py
  event detection
  deceleration attribution
  query-time-frame causal history extraction
  response labeling

waymo/interaction_contrastive_learning/legacy/build_matched_samples.py
  deterministic source sampling
  robust history normalization
  per-stratum scipy cKDTree matching
  positive/hard-negative/easy-negative construction
  semantic edge validation
  NPZ/CSV/JSON output

waymo/interaction_contrastive_learning/tests/legacy/test_pair_samples.py
  synthetic crossing-event test
  query-time-frame history test
  positive/hard-negative matching test
```

Reproduction command from `waymo/`:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python \
  interaction_contrastive_learning/legacy/build_matched_samples.py \
  --data_root data/waymo_vector_dataset_ooi_centered_training_all \
  --split train \
  --max_focus_samples 5000 \
  --selection random \
  --seed 0 \
  --output_dir cache/interaction_contrastive_learning_5k
```

All three unit tests pass.

## 10. Data-Source Clarification

The requested source directory is:

```text
/p/yufeng/tri30/dreamer4/waymo/data/
  waymo_vector_dataset_ooi_centered_training_all
```

Despite this experiment being called “5k”, that directory contains:

```text
train focus samples: 367,670
val focus samples:    40,854
all focus samples:   408,524
```

The preparation run used a deterministic seed-0 reservoir sample of exactly
5,000 train focus NPZ files. Each source path is stored in the cache and CSV.

## 11. 5k Collection Results

Input processing:

```text
selected focus samples:             5,000
processed focus samples:            5,000
failed focus samples:                   0
unique scenarios producing samples: 3,773
elapsed time:                       48.4 seconds
```

Pair samples:

```text
all event-aligned pair samples:   19,941
high-confidence response samples: 13,242
```

High-confidence response distribution:

| Response | Count |
|---|---:|
| maintains | 8,010 |
| goes_first | 2,562 |
| decelerates | 2,380 |
| yields | 290 |

Eligible relation distribution:

| Relation | Count |
|---|---:|
| other_leads_focus | 5,602 |
| other_follows_focus | 4,788 |
| converging_conflict | 2,168 |
| crossing_or_oncoming_conflict | 684 |

Matching coverage:

```text
anchors with at least one positive:        12,896
anchors with at least one hard negative:   10,328
anchors with both and usable for training: 10,189
```

Stored edges:

```text
positive edges:      25,584
hard-negative edges: 32,776
ordinary negatives:  52,968
```

Trainable anchors by lead time:

| Time before event | Trainable anchors |
|---|---:|
| 1 second | 4,151 |
| 2 seconds | 3,408 |
| 3 seconds | 2,630 |

Rare-yield coverage:

```text
eligible yield samples:  290
trainable yield anchors: 261

crossing yield trainable anchors:   23
converging yield trainable anchors: 238
```

Trainable anchors by relation/response:

| Relation | Response | Count |
|---|---|---:|
| other_leads_focus | decelerates | 1,453 |
| other_leads_focus | maintains | 3,393 |
| other_follows_focus | decelerates | 749 |
| other_follows_focus | maintains | 3,020 |
| crossing_or_oncoming_conflict | goes_first | 362 |
| crossing_or_oncoming_conflict | yields | 23 |
| converging_conflict | goes_first | 951 |
| converging_conflict | yields | 238 |

## 12. Stored Artifacts

```text
waymo/cache/interaction_contrastive_learning_5k/train_samples.npz
  size: 21,174,336 bytes
  history shape: (19,941, 20, 11), float32
  matching vector shape: (19,941, 231), float16
  event/query metadata, labels, source IDs, normalization statistics

waymo/cache/interaction_contrastive_learning_5k/train_matches.npz
  size: 724,844 bytes
  positive, hard-negative, and ordinary-negative indices/distances
  trainable_anchor_mask

waymo/cache/interaction_contrastive_learning_5k/train_samples.csv
  size: 5,049,926 bytes
  human-readable per-sample inspection table

waymo/cache/interaction_contrastive_learning_5k/train_summary.json
  size: 8,458 bytes
  full configuration, counts, strata, calipers, and output paths
```

Post-write audit confirmed:

- all history and matching-vector values are finite;
- positive edges are cross-scene, same-lead, same-relation,
  same-agent-type, and same-response;
- hard negatives satisfy those controls but have a different response;
- ordinary negatives are cross-scene, same-lead, and different-relation;
- all 25,584 positive, 32,776 hard-negative, and 52,968 ordinary-negative
  edges passed semantic validation.

## 13. Planned First Training Experiment

Objective:

```text
L_total = L_reconstruction + lambda_contrastive * L_matched_SupCon
```

Recommended sequence:

1. Load the baseline tokenizer checkpoint.
2. Initialize the existing slot-query pair readout and a small projector.
3. Briefly train only the readout/projector to validate sampling and loss.
4. Freeze the tokenizer decoder.
5. Unfreeze the final one or two encoder blocks.
6. Use a smaller encoder LR than readout/projector LR.
7. Keep reconstruction loss active through the frozen decoder.
8. Ramp the contrastive weight during initial steps.
9. If stable, optionally unfreeze more encoder layers.

Suggested starting values:

```text
contrastive temperature: 0.07
readout/projector LR:     approximately 1e-4
encoder LR:               approximately 1e-5
```

Choose the contrastive weight from encoder gradient norms. Initially target
contrastive gradients around 10%–30% of reconstruction gradients.

Because yield is rare, balance response classes and require every selected
anchor to have a positive and hard negative. The stored fixed-width indices
avoid online nearest-neighbor search.

If this is initially unstable, the agreed fallback is the existing `Z`-only
supervised interaction auxiliary objective, followed by adding contrastive loss.

## 14. Evaluation and Success Criteria

After fine-tuning, discard the training projector/readout, freeze the tokenizer,
and train fresh independent probes:

```text
raw_current_only
raw_history_only
z_only
raw_history_z
raw_history_shuffled_z
```

Use at least three seeds and scene-level bootstrap confidence intervals. Report
metrics separately at one-, two-, and three-second lead times.

Success conditions:

1. `raw_history_z` exceeds `raw_history_only` on matched validation subsets.
2. `raw_history_z` exceeds `raw_history_shuffled_z`.
3. `z_only` carries readable yield/go-first and deceleration information.
4. Improvement occurs on high-confidence matched pairs, not only easy types.
5. Tokenizer reconstruction stays within a predefined degradation tolerance.
6. A world model retrained on the new tokenizer improves interaction-sensitive
   downstream metrics.

Initial ablations:

```text
baseline tokenizer
matched-SupCon tokenizer
matched-SupCon with shuffled response labels
current-only matching versus aligned-history matching
```

## 15. Known Risks and Immediate Next Step

- Yield remains rare, especially crossing yield; balanced sampling is required.
- Visually inspect deceleration attribution and event labels before training.
- Rare agent-type strata can have large adaptive calipers; consider a minimum
  stratum size or absolute maximum caliper after inspection.
- Very small distances can represent nearly stationary identical histories;
  audit them for degenerate all-zero tracks.
- OOI-centered data may retain slot-selection bias; audit candidate ordering.
- This is a 5k screening cache. Build full scale only after training smoke tests.

Before training, inspect stratified stored pairs:

```text
crossing: goes_first versus yields
converging: goes_first versus yields
following/leading: maintains versus decelerates
each 1s, 2s, and 3s lead group
```

If their observed histories are genuinely similar and future responses are
correctly different, implement the batch sampler and matched supervised
contrastive tokenizer fine-tuning.
