# Interactive Probe Methods and Z-Only Probe Plan

Date: 2026-07-07

Project path: `/scratch/baz7dy/tri30/dreamer4`

Relevant code:

- `waymo/interactive_probe/build_cache.py`
- `waymo/interactive_probe/train_probe.py`
- `waymo/interactive_probe/model.py`
- `waymo/interactive_probe/labels.py`

Recent run artifacts:

- Logs: `waymo/logs/interactive_probe`
- Cache: `waymo/cache/interactive_probe_z31_v0_besttok`
- Checkpoints: `waymo/checkpoints/interactive_probe_z31_v0_besttok`

## Current Probe Setup

The current interactive probe cache stores one row per valid focus-candidate
pair. The focus agent is fixed to slot 0, and each pair has a `candidate_index`.

For each scene, the cache stores:

```text
z_current:         (num_scenes, 64, 64)
pair_scene_index:  (num_pairs,)
candidate_index:   (num_pairs,)
pair_raw:          (num_pairs, 34)
```

The `z_current` value is produced by the frozen tokenizer encoder at
`query_step=31`, after giving the encoder a 32-step context window.

The `pair_raw` value is not a 32-step trajectory. It is a single-current-step
feature vector built from focus and candidate states at `query_step=31`.

`pair_raw` includes:

- focus current position, speed, velocity, yaw, and type;
- candidate current position, speed, velocity, yaw, and type;
- relative position, distance, relative velocity, and relative speed;
- bearing and heading difference;
- longitudinal and lateral offset in the focus frame;
- closing speed;
- same-direction and crossing-angle proxy flags;
- current close-within-5m and close-within-10m flags.

This makes `pair_raw` a strong current-state pair geometry and kinematics
baseline, even though it is only a single timestep.

## Existing Probe Modes

The current model supports three modes:

```text
raw_only:
  pair_raw -> heads

raw_z:
  pair_raw -> query
  query attends to z_current
  fused hidden -> heads

raw_shuffled_z:
  pair_raw -> query
  query attends to scene-shuffled z_current
  fused hidden -> heads
```

The output heads predict:

- relevance;
- interaction type;
- response binary labels;
- delta arrival time regression.

Important interpretation:

`raw_z` is not a pure z probe. The attention query is produced from
`pair_raw`, so the model receives strong pair geometry directly. This can make
the probe rely mostly on raw features, with z contributing little or being
ignored.

## Recent Result Summary

The July 2026 interactive probe run found:

- `raw_only` already reaches strong validation performance.
- `raw_z` does not clearly and stably beat `raw_only`.
- `raw_shuffled_z` is close to `raw_only` and sometimes better on type metrics.

This suggests that the current `raw_z` comparison does not cleanly measure
whether z contains interaction information. The raw pair features may dominate
the task.

The result should be interpreted as:

> Current single-step pair geometry is a strong shortcut, and adding z through a
> raw-derived query did not produce a stable additional gain.

It should not yet be interpreted as:

> z itself is worse than raw features.

That conclusion requires a clean z-only probe.

## Why Z-Only Is Nontrivial

The label is pair-specific:

```text
relation(focus slot 0, candidate slot j)
```

But `z_current` is scene-level:

```text
z_current: (64 latent tokens, 64 dims)
```

The latent-token axis is not an agent axis. Row `j` of z is not agent slot `j`.
Therefore, a naive probe like:

```text
mean_pool(z_current) -> MLP -> pair label
```

is not well defined for pair-level prediction. The same scene-level vector would
be used for all candidates in the scene.

The probe must tell the model which candidate slot is being queried while not
revealing current raw geometry.

## Proposed Z-Only Probe

The clean z-only probe should use:

```text
Input:
  z_current
  candidate_index j

Do not input:
  pair_raw
  current x/y/speed/yaw
  distance
  relative velocity
  closing speed
  any hand-built pair geometry
```

The candidate index is allowed because otherwise the pair task is undefined.
It tells the probe which agent slot to ask about, but it does not directly
reveal where the agent is or how it is moving.

## Recommended Architecture: Slot-Query Z-Only Probe

Use a learned query for the pair `(focus=0, candidate=j)` and let that query
cross-attend to `z_current`.

Conceptually:

```text
focus_query = slot_embedding[0]
cand_query  = slot_embedding[j]
pair_query  = MLP([focus_query, cand_query, focus_query * cand_query, abs(focus_query - cand_query)])

memory = Linear(z_current)
attn_out = CrossAttention(query=pair_query, key=memory, value=memory)
hidden = Fuse(pair_query, attn_out)
heads(hidden) -> relevance/type/response/delta
```

This tests whether the scene latent contains information that a slot-aware
reader can use to recover the interaction relation for a requested candidate.

## Stronger Variant: Decoder-Query Initialization

The tokenizer decoder already uses learned agent queries to reconstruct fixed
agent slots from z. A stronger and more semantically aligned z-only probe can
initialize the slot embeddings from:

```text
tokenizer.decoder.agent_queries
```

Useful variants:

- freeze decoder agent queries and train only the probe head;
- initialize from decoder agent queries, then allow fine-tuning;
- compare against randomly initialized slot embeddings.

If decoder-query initialization helps a lot, that would indicate z is accessible
through the same slot addressing mechanism used by reconstruction.

## Alternative Architecture: All-Slot Multi-Output Probe

Instead of one row per pair, process each scene once:

```text
z_current -> candidate slot queries for all j -> per-slot predictions
```

Then use the cached `candidate_index` values to gather predictions and compute
loss only for valid focus-candidate pairs.

This is closer to the tokenizer decoder shape, but it requires a larger code
change because the current dataset is pair-row based.

## Baselines and Controls

Recommended comparisons:

1. `raw_only`
   - Existing strong current-geometry baseline.

2. `raw_z`
   - Existing raw-derived query plus real z.

3. `raw_shuffled_z`
   - Existing control for whether real scene z matters in the raw-query setup.

4. `z_only_slot_random`
   - New z-only probe with random learned slot embeddings.

5. `z_only_slot_decoder_init`
   - New z-only probe initialized from tokenizer decoder agent queries.

6. `z_only_shuffled_z`
   - Same z-only slot probe, but with scene-shuffled z.
   - This checks whether the probe is exploiting slot identity priors rather
     than scene-specific latent content.

7. `slot_id_only`
   - Candidate slot embedding only, no z.
   - This checks for remaining slot-order leakage.

## Expected Interpretations

If `z_only_slot_decoder_init` is much better than `slot_id_only` and
`z_only_shuffled_z`, then z contains pair-relevant scene information.

If `z_only_slot_random` is weak but `z_only_slot_decoder_init` is strong, then
z is readable, but only through the tokenizer's learned slot-query convention.

If all z-only variants are weak while `raw_only` is strong, then current
interaction labels are mostly recoverable from explicit current geometry, and
the tokenizer latent may not preserve pair-interaction information in an easily
probeable form.

If `z_only_shuffled_z` is close to `z_only`, then the probe is likely using
candidate slot priors or label imbalance rather than real z content.

## Implementation Notes

The current pair-row dataset already stores `candidate_index`, so a first
implementation can reuse the current cache and dataloader.

Minimal model change:

- add `mode="z_only_slot"` to `InteractiveProbe`;
- add a slot embedding table sized to the tokenizer decoder's number of agent
  slots;
- in `PairDataset.__getitem__`, return `candidate_index`;
- in `forward`, build pair query from slot 0 and candidate slot j;
- cross-attend that query to `z_current`;
- send the fused representation to the existing heads.

No new cache is required for the first z-only experiment.

The important rule is:

> Do not pass `pair_raw` or any current geometry to the z-only model.

