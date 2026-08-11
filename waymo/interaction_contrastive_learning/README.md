# Interaction Contrastive Learning: matched sample preparation

This directory prepares pair-level samples before contrastive tokenizer
fine-tuning.  It does not train a model.

## Code layout

- `latest/`: current full-pair v2 filtering and event-aligned masked-RMS
  similarity pipeline, including its runners and visual audits;
- `legacy/`: archived soft-pair v1 and discrete-label matched-pair experiments;
- `tests/latest/` and `tests/legacy/`: tests grouped by the same boundary.

## Full 91-step physical-contact pairs (v2)

The v2 collector separates pair inclusion from event/window quality:

- every original two-OOI pair is retained unconditionally;
- all samples store the complete `(2, 91, 6)` trajectories plus a `(2, 91)`
  validity mask, so partial tracks are masked rather than discarded;
- raw Waymo length/width arrays define oriented-box path-contact intervals
  with a 1 metre edge-clearance buffer;
- continuous line-segment intersections prevent crossings between 10 Hz
  samples from being missed;
- all non-OOI candidates with physical path contact are retained by default;
  they are still ranked deterministically by a soft zone-PET score, and an
  optional positive `--non_ooi_top_k_per_focus` can cap them for smaller runs;
- unordered `(scenario, agent A, agent B)` keys remove the duplicate OOI view.

Build in a detached CPU tmux session:

```bash
bash waymo/interaction_contrastive_learning/latest/run_full_pair_50k_v2_tmux.sh
```

Outputs are sharded under `waymo/cache/interaction_full_pairs_50k_v2/`.
`summary.json` enforces and reports original-OOI completeness; the run fails if
even one expected two-OOI pair is missing.

## Event-aligned RMS neighbours for full pairs (v0)

The first similarity baseline operates on the completed no-top-k v2 dataset:

- linearly align every pair to `primary_step_first` on a 60-step window from
  -1.9 to +4.0 seconds;
- fit per-agent-role median/IQR normalization on valid train states only and
  apply the same statistics to val;
- keep samples with at least 80% joint trajectory coverage;
- retrieve 1,024 coarse candidates in a 42-dimensional low-frequency DCT
  space, separately by ordered agent types and fallback/contact event class;
- exclude same-scenario matches and rerank with exact mask-aware RMS over the
  full normalized `60 x 2 x 6` representation;
- store the exact top 32 neighbours. DCT distance is never used as the final
  similarity value.

The detached runner waits until the source dataset has a `summary.json`, then
computes RMS and creates a dependency-light HTML audit of representative top-1
matches:

```bash
bash waymo/interaction_contrastive_learning/latest/run_full_pair_rms_neighbors_tmux.sh
```

Outputs are written under
`waymo/cache/interaction_full_pairs_50k_v2_no_topk_rms_v0/`:

- `{train,val}_rms_features.npz`: aligned normalized features, masks, scaler,
  and source metadata;
- `{train,val}_rms_neighbors.npz`: neighbour indices, exact RMS, and common
  valid fractions;
- `{train,val}_rms_top3.csv`: convenient top-match table;
- `rms_summary.json`: configuration, distributions, timing, and sampled exact
  top-32 recall of the coarse stage;
- `visual_audit/index.html`: side-by-side trajectory audit across the RMS
  distribution.

## Shared-time-axis soft pairs (v1)

The current label-free experiment scans the 50k OOI-centered train and val
splits.  For each focus/other pair it searches asynchronous path points with
at most 30 steps (3 seconds) arrival-time difference, retains pairs whose
constrained closest distance is at most 6 metres, and stores a continuous
geometry/PET relevance score.  It does not assign relation, response, hard
negative, or easy negative labels.

Every retained interaction is represented by one 60-step shared-time-axis
sequence.  Agents are ordered by arrival time, the first arrival is at sequence
index 19, and the sequence extends through index 59.  Each step stores the two
agents' conflict-frame position, velocity, and heading sine/cosine (12 channels).

Build train and val caches on CPU:

```bash
PYTHONPATH=waymo \
/p/yufeng/.conda/envs/dreamer4/bin/python \
  waymo/interaction_contrastive_learning/legacy/build_soft_pair_dataset.py
```

Outputs under `waymo/cache/interaction_soft_pairs_50k_v1/`:

- `{train,val}_samples.npz`: raw and globally normalized 60x12 sequences,
  source metadata, closest-point/PET values, and relevance scores;
- `{train,val}_soft_neighbors.npz`: different-scene nearest neighbours, exact
  full-sequence RMS distances, locally scaled soft similarities, and scales;
- `{train,val}_samples.csv`: human-readable sample metadata;
- `summary.json`: extraction, normalization, distance, and distribution stats.

The robust scaler is fitted on train only and applied unchanged to val.  The
neighbour graph is also built separately per split.  Candidate retrieval uses
a compact low-frequency DCT descriptor, but every stored distance and soft
score is recomputed from the complete normalized 60x12 sequence without DTW.

## Legacy discrete-label matched-pair experiment

The earlier screening experiment uses:

- a geometry-defined pair event rather than an absolute scene timestep;
- query steps at 1, 2, and 3 seconds before the event;
- 2 seconds of causal history ending at the query;
- one fixed query-time focus frame for the full history;
- no DTW;
- robust-standardized, time-aligned weighted Euclidean matching;
- exact matching strata `(lead, relation, focus type, candidate type)`;
- different-scene positives with the same response;
- different-scene hard negatives with similar history but a different response;
- easy negatives with a different relation at the same lead time.

Future-derived event/response labels are used only for supervision and for
positive/negative selection.  Matching vectors contain only states at or before
the query timestep.

## Build the 5k screening cache

From `waymo/`:

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

The source directory contains about 408k focus samples, despite the experiment
being referred to as “5k”.  The command above records the exact deterministic
5,000-file reservoir sample in the output metadata.

Outputs:

- `train_samples.npz`: histories, labels, source IDs, normalized matching vectors;
- `train_matches.npz`: fixed-width positive, hard-negative, and negative indices;
- `train_samples.csv`: human-readable sample and match counts;
- `train_summary.json`: counts, distributions, matching calipers, and configuration.

An anchor is marked `trainable_anchor_mask=True` only when it has at least one
positive and one hard negative within its stratum's distance caliper.

## Visual audit of anchors and matches

Build a deterministic, stratified gallery before training:

```bash
MPLCONFIGDIR=/tmp/matplotlib-interaction-audit \
/p/yufeng/.conda/envs/dreamer4/bin/python \
  interaction_contrastive_learning/legacy/visualize_matched_samples.py \
  --samples_npz cache/interaction_contrastive_learning_5k/train_samples.npz \
  --matches_npz cache/interaction_contrastive_learning_5k/train_matches.npz \
  --output_dir cache/interaction_contrastive_learning_5k/visual_audit_folders \
  --num_anchors 24 \
  --selection stratified \
  --seed 0
```

The default 24-anchor selection is round-robin balanced over
`(relation, response, lead_steps)`.  Every anchor receives its own directory,
with separate `anchor/`, `positive/`, `hard_negative/`, and `easy_negative/`
subdirectories.  Every sample is a standalone trajectory-and-speed image; no
samples are composited into one large image.  All scenes are aligned to their
own query-time focus frame.  Solid trajectories are the 20-step causal
matching history; dashed trajectories show post-query ground truth for label
auditing.  Images also show query/event timing, pair speeds, matching distance,
PET, and arrival-time difference.

Outputs:

- `index.html`: browsable gallery;
- `anchor_*/anchor/*.png`: standalone anchor images;
- `anchor_*/positive/*.png`, `hard_negative/*.png`, and
  `easy_negative/*.png`: standalone matched-sample images;
- `anchor_*/manifest.csv` and `.json`: per-anchor match metadata;
- `gallery_manifest.csv` and `.json`: every displayed sample and edge;
- `gallery_config.json`: reproducible selection and rendering arguments.

To render specific anchors instead of sampling them:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python \
  interaction_contrastive_learning/legacy/visualize_matched_samples.py \
  --anchor_indices 0,123,456 \
  --output_dir cache/interaction_contrastive_learning_5k/visual_audit_selected
```

Optional `--relations`, `--responses`, and `--lead_steps` filters apply to the
automatically sampled anchors.  For example,
`--relations converging_conflict --responses yields --lead_steps 20` audits
two-second converging-yield anchors.

## Tests

```bash
PYTHONPATH=waymo \
/p/yufeng/.conda/envs/dreamer4/bin/python -m unittest \
  discover -s interaction_contrastive_learning/tests -p 'test_*.py'
```
