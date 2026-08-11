# Interactive Waymo world model

This browser game uses the action-conditioned Waymo checkpoint directly. The
human controls focus-agent slot 0; the decoded world model supplies every other
agent.

## Run

From the repository root on a CUDA machine:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python \
  waymo/interactive_world_model_game.py \
  --world-model-ckpt \
  /p/yufeng/tri30/dreamer4/waymo/checkpoints/waymo_wm_original_stmlayer_3_stage/waymo_wm_time1_mapx1_h30step10k_exact_ctx1_h90_d1_chunk32s30_b1_50k/step_00040000.pt
```

Open <http://127.0.0.1:7861>. For a remote GPU node, first forward the port
from your laptop:

```bash
ssh -L 7861:127.0.0.1:7861 USER@GPU_HOST
```

Then open <http://127.0.0.1:7861> locally. The checkpoint already records the
matching tokenizer and validation dataset paths, so they do not need to be
specified. Use `--tokenizer-ckpt` and `--data-dir` to override them.

## Controls

- Up/down: increase/decrease speed.
- Left/right: increase/decrease heading.
- Space: play or pause.
- Period (`.`): advance one model step while paused.
- `R`: reset the current scene.
- `N`: load a random validation scene.

Each tick is 0.1 simulation seconds. The controller integrates an exact focus
state and sends the model its training-time raw action layout:

```text
[delta_x, delta_y, delta_yaw, speed, velocity_x, velocity_y, valid, 0, ..., 0]
```

Defaults are 5 m/s^2 acceleration, 45 degrees/s yaw rate, one D1 dynamics pass
per frame, a 32-frame tokenizer decode window, and a 90-step episode (the
checkpoint's trained rollout horizon). All are configurable through CLI flags.

## PufferDrive 3D renderer

PufferDrive rendering is a two-step workflow. Dreamer still loads the same NPZ
files for model input; the offline manifest joins each exact NPZ/focus view to a
PufferDrive `map_000.bin`. At runtime the game converts Dreamer's local poses
back to raw Waymo world coordinates and sends them to a separate renderer
process by stable track ID.

First, convert the NPZ views in the PufferDrive environment. For example, this
converts the complete Dreamer validation directory:

```bash
cd /p/yufeng/tri30/dreamer4

PUFFER_SCENE_OUTPUT=/p/yufeng/tri30/dreamer4/waymo/cache/pufferdrive_val

/p/yufeng/.conda/envs/puffd/bin/python \
  waymo/data_prep/prepare_pufferdrive_static_scenes.py \
  data/waymo_vector_dataset_ooi_centered_50k/val \
  --output-dir "$PUFFER_SCENE_OUTPUT"
```

For a quick smoke test, replace the validation directory with one `.npz` file.
The converter resolves the corresponding cached raw Scenario proto, retains the
raw 3D map, dimensions, elevation, and trajectories, and writes:

```text
$PUFFER_SCENE_OUTPUT/manifest.csv
$PUFFER_SCENE_OUTPUT/views/<scenario_id>__focus_<track_id>/map_000.bin
```

Then run the game in the Dreamer environment and pass that conversion manifest:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python \
  waymo/interactive_world_model_game.py \
  --device cuda:0 \
  --host 127.0.0.1 \
  --port 7861 \
  --renderer puffer \
  --puffer-manifest \
  /p/yufeng/tri30/dreamer4/waymo/cache/pufferdrive_val/manifest.csv
```

This requires the Puffer worker at the default command:

```text
/p/yufeng/.conda/envs/puffd/bin/python -u \
  /p/yufeng/tri30/PufferDrive/puffer_renderer_worker.py
```

Use `--puffer-worker-command` to override it. The worker owns Raylib and the
headless display, while the Dreamer process owns model inference and the web
server. By default, a missing converted scene or worker error is reported and
that browser session falls back to the original 2D renderer. Add
`--puffer-strict` to abort instead, which is useful when validating a new
conversion or worker build.

The converter preserves raw Waymo z values in the scene asset, but the current
native PufferDrive renderer intentionally draws roads and actors on fixed
display planes, matching the supplied demo video's visual style. It is not yet
an elevation-aware renderer for overpasses or sloped roads.
