# Waymo Folder Layout

- `core/`: reusable dataset, filtering, encoder, and decoder modules.
- `data_prep/`: scripts for analyzing raw Waymo data and preparing NPZ datasets.
- `training/`: training entrypoints plus tmux/slurm launch scripts.
- `evaluation/`: reconstruction, visualization, gallery scripts, run lists, and `reports/`.
- `data/`, `checkpoints/`, `logs/`, `wandb/`: generated artifacts kept at their original paths so existing runs and checkpoints remain valid.

Most shell entrypoints still `cd /scratch/baz7dy/tri30/dreamer4` before running, so they can be launched from any directory.
