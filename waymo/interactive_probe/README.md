# Interactive Relation Probe

This probe asks whether `z[query_step]` contains future-interaction information
for a focus-other agent pair.

## Cache building

Build train/val caches separately. Labels use future ground-truth trajectories;
inputs use only current pair features and optional scene-level `z_current`.

```bash
python interactive_probe/build_cache.py \
  --data_dir /path/to/train \
  --split train \
  --output_dir /path/to/cache/interactive_probe \
  --tokenizer_ckpt /path/to/tokenizer/best.pt
```

Important outputs:

- `pair_raw`: current focus-other pair features, shape `(N, 34)`.
- `z_current`: scene-level latent array, shape `(num_scenes, n_latents, z_dim)`.
- `relevance_targets`: layer-1 all-pair relevance label.
- `type_targets` and `type_masks`: layer-2 interaction type, trained/evaluated only when `type_masks=1`.
- `response_*` and masks: layer-3 priority/response labels, trained/evaluated only when the corresponding mask is 1.

Layer-2 types:

- `other_leads_focus`
- `other_follows_focus`
- `crossing_or_oncoming_conflict`
- `converging_conflict`

Layer-3 response labels:

- `focus_goes_first`
- `focus_yields_to_other`
- `focus_decelerates_for_interaction`
- `delta_arrival_time_s`

## Probe training

```bash
python interactive_probe/train_probe.py \
  --mode raw_z \
  --train_cache /path/to/cache/interactive_probe/train_cache.npz \
  --val_cache /path/to/cache/interactive_probe/val_cache.npz \
  --run_dir /path/to/checkpoints/interactive_probe/raw_z
```

Supported modes:

- `raw_only`: current pair features only.
- `raw_z`: current pair features query the scene `z_current`.
- `raw_shuffled_z`: shuffled-z control.

