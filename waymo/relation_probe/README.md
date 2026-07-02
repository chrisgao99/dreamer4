# z[31] Current-State Relation Probe

This folder contains the first current-state pair representation probe for the
Waymo vector tokenizer.

## Question

Use the frozen tokenizer encoder and read only the current latent state:

```text
context frames: 0..31
query frame:    31
memory:         z_current = Z[:, 31, :, :] = [64, 64]
pair:           focus agent 0 vs candidate agents 1..31
```

Compare:

```text
raw_only
raw_z
raw_shuffled_z
```

The main interpretation is:

```text
raw_z > raw_only
raw_shuffled_z ~= raw_only
```

This suggests the current tokenizer latent contains scene context that is
readable by a pair query.

## Labels

The first version uses only current-frame labels.

Current pair geometry:

- relative dx/dy, distance
- relative velocity and speed
- longitudinal/lateral offset in the focus frame
- heading difference
- front/behind/left/right
- same-direction and crossing-angle proxies
- current close 5m/10m

Current scene context:

- nearest candidate overall
- top-3 nearest candidates
- nearest front/rear candidate in focus corridor
- left/right adjacent candidate
- same-corridor proxy
- close front/rear gap proxy
- focus-neighborhood membership
- distance rank, front/rear gap, local density

## Build Cache

The cache stores:

```text
train_cache.npz / val_cache.npz
  z_current:        [num_scenes, 64, 64] fp16
  pair_scene_index: [num_pairs]
  candidate_index:  [num_pairs]
  pair_raw:         [num_pairs, F]
  reg_targets:      [num_pairs, R]
  bin_targets:      [num_pairs, C]
```

Submit:

```bash
sbatch /scratch/baz7dy/tri30/dreamer4/waymo/relation_probe/submit_build_cache.slurm
```

Default cache path:

```text
/scratch/baz7dy/tri30/dreamer4/waymo/cache/relation_probe_z31_current_v0_besttok
```

## Train Probes

Submit the three modes:

```bash
MODE=raw_only sbatch /scratch/baz7dy/tri30/dreamer4/waymo/relation_probe/submit_train_probe.slurm
MODE=raw_z sbatch /scratch/baz7dy/tri30/dreamer4/waymo/relation_probe/submit_train_probe.slurm
MODE=raw_shuffled_z sbatch /scratch/baz7dy/tri30/dreamer4/waymo/relation_probe/submit_train_probe.slurm
```

Probe training defaults to `NUM_WORKERS=0` because the cache is already loaded
into memory and this avoids multiprocessing issues with shared array transport.

Default run path:

```text
/scratch/baz7dy/tri30/dreamer4/waymo/checkpoints/relation_probe_z31_current_v0_besttok/{mode}
```

Each run writes:

```text
best.pt
best_metrics.json
latest.pt
latest_metrics.json
final.pt
final_metrics.json
```

## Compare

```bash
cd /scratch/baz7dy/tri30/dreamer4/waymo/relation_probe

python compare_metrics.py \
  --run_dir \
    ../checkpoints/relation_probe_z31_current_v0_besttok/raw_only \
    ../checkpoints/relation_probe_z31_current_v0_besttok/raw_z \
    ../checkpoints/relation_probe_z31_current_v0_besttok/raw_shuffled_z \
  --name raw_only raw_z raw_shuffled_z \
  --metrics_name best_metrics.json \
  --output_csv ../checkpoints/relation_probe_z31_current_v0_besttok/compare_best.csv
```
