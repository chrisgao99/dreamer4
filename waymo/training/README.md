# Waymo training scripts

The experiment launchers are grouped by training target:

- `tokenizer/`: tokenizer training, overfit checks, and tokenizer Slurm jobs.
- `world_model/`: world-model training, rollout, and evaluation jobs.
- this directory: compatibility Python entry points only.

Prefer shared launchers plus environment-variable overrides over copying an
existing Slurm file. For example, the latent/context evaluation grid is kept in
`world_model/submit_eval_waymo_world_model_lat32_lat64_step850k_ctx_sweep.slurm`
as one Slurm array instead of four single-configuration wrappers.

When adding an experiment, keep the filename specific to the parameters that
actually differ and place it in the matching subdirectory.
