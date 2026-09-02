# Interactive Waymo world model

This browser game uses the action-conditioned Waymo checkpoint directly. The
human controls focus-agent slot 0; the decoded world model supplies every other
agent.

## Run

From the repository root on a CUDA machine:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python \
  waymo/interactive_world_model_game.py \
  --ckpt h90
```

The 2D and 3D games share two named checkpoint profiles:

```text
h30 -> best_multisample_finetuned.pt
h90 -> step_00027000.pt (default)
```

Select either with `--ckpt h30` or `--ckpt h90`. The longer spelling
`--checkpoint-profile` is equivalent. `--world-model-ckpt /path/custom.pt`
remains available for experiments and overrides the named profile. A profile
name selects weights only; both profiles use the same context protocol. The
default rollout is 80 generated frames in 2D and 150 generated frames in 3D.

Open <http://127.0.0.1:7861>. For a remote GPU node, first forward the port
from your laptop:

```bash
ssh -L 7861:127.0.0.1:7861 USER@GPU_HOST
```

Then open <http://127.0.0.1:7861> locally. The checkpoint already records the
matching tokenizer and validation dataset paths, so they do not need to be
specified. Use `--tokenizer-ckpt` and `--data-dir` to override them.

## Context and rollout timeline

All three frontends—the 2D web game, 3D web game, and local 3D
wheel/keyboard game—replay and condition on the same recorded context:

```text
recorded replay:      Waymo frames 1–11   (11 frames)
first model context:  Waymo frames 2–11   (10 past tokens)

2D prediction:       frames 12–91   (80 generated frames)
3D prediction:       frames 12–161  (150 generated frames)
```

Frame 1 is replayed and is also needed to construct frame 2's recorded motion
action. The checkpoint's `max_rollout_window=11` includes the new noisy
prediction token, so the sampler retains ten past latent tokens—frames 2
through 11—for its first prediction. It does not feed eleven past frames plus
an extra twelfth token. Driving input is ignored during the recorded replay and
takes effect on the first generated frame, frame 12.

The protocol can be overridden with `--context-frames` and `--unroll-steps`.
`--context-frames` defaults to 11; `--unroll-steps` defaults to 80 for
`--renderer 2d` and 150 for `--renderer puffer`. An explicitly supplied
`--unroll-steps` always wins. The older `--max-steps` spelling remains an
equivalent alias. `--start-frame 0` uses the frame numbering above.

Every frontend displays a reportable identifier in the form
`scene #<dataset-index> | scenario <Waymo-ID> | focus <track-ID>`. The same
identifier and exact NPZ filename are printed to the terminal whenever a scene
loads. When reporting a bad scene, record the whole label; `scene #` is the
value to pass back through `--scene-index` for the same dataset.

## Controls

- Up/down: increase/decrease speed.
- Left/right: increase/decrease heading.
- Space: play or pause.
- Period (`.`): advance one timeline frame while paused (recorded replay first,
  then model rollout).
- `R`: reset the current scene.
- `N`: load a random validation scene.

Each tick is 0.1 simulation seconds. The controller integrates an exact focus
state and sends the model its training-time raw action layout:

```text
[delta_x, delta_y, delta_yaw, speed, velocity_x, velocity_y, valid, 0, ..., 0]
```

Defaults are 5 m/s^2 acceleration, 45 degrees/s yaw rate, one D1 dynamics pass
per generated frame, a 32-frame tokenizer decode window, 11 recorded replay
frames, and a renderer-specific generated horizon: 80 in 2D or 150 in 3D. All
are configurable through CLI flags.

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
  --ckpt h90 \
  --host 127.0.0.1 \
  --port 7861 \
  --renderer puffer \
  --unroll-steps 150 \
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

## Headless 3D videos with recorded focus actions

`waymo/evaluation/render_puffer_gt_focus_rollouts.py` creates offline MP4s
without a browser or physical display. It takes scenes in the durable
`sample_order` from `val_random128_seed0_manifest.json`, displays 11 recorded
frames, computes future focus-car actions from the full NPZ trajectory, and
uses Dreamer to generate the other actors for the requested rollout length.
For the standard first-five, context-11, future-80 run:

```bash
unset DISPLAY

/p/yufeng/.conda/envs/dreamer4/bin/python -u \
  waymo/evaluation/render_puffer_gt_focus_rollouts.py \
  --device cuda:0 \
  --ckpt h90 \
  --subset-manifest waymo/evaluation/val_random128_seed0_manifest.json \
  --puffer-manifest waymo/cache/pufferdrive_val128_seed0/manifest.csv \
  --sample-start 0 \
  --num-scenes 5 \
  --context-frames 11 \
  --unroll-steps 80 \
  --fps 10 \
  --puffer-width 1280 \
  --puffer-height 720 \
  --puffer-timeout 120 \
  --output-dir waymo/eval_results/puffer_gt_focus_h90_first5_ctx11_u80
```

Do not pass `--puffer-use-inherited-display` on a headless node and do not wrap
this command in `xvfb-run`. The external Puffer worker owns an isolated Xvfb.
Each standard output contains exactly 91 frames and lasts 9.1 seconds at 10
FPS. The frame overlay and `summary.json` preserve subset order, full dataset
index, scenario ID, focus track ID, checkpoint, and random seed.

To render the same five scenes as complete 91-frame ground-truth replays, with
no action input and without loading the tokenizer, checkpoint, or world model,
use the standalone renderer and place its output in the model-run directory's
`ground_truth` child:

```bash
env -u DISPLAY PYTHONUNBUFFERED=1 \
  /p/yufeng/.conda/envs/dreamer4/bin/python -u \
  waymo/evaluation/render_puffer_ground_truth_replays.py \
  --subset-manifest waymo/evaluation/val_random128_seed0_manifest.json \
  --puffer-manifest waymo/cache/pufferdrive_val128_seed0/manifest.csv \
  --sample-start 0 \
  --num-scenes 5 \
  --fps 10 \
  --puffer-width 1280 \
  --puffer-height 720 \
  --puffer-timeout 120 \
  --output-dir \
    waymo/eval_results/puffer_gt_focus_h90_first5_ctx11_u80/ground_truth
```

This script is renderer-only and does not require a GPU. It replays all actor
poses directly from each NPZ and sends the matching converted Puffer scene's
per-frame elevation; it has no device or checkpoint argument. Puffer's current
native renderer still draws actors on fixed display planes, so elevation is
preserved in the request but is not yet visible in the video.

## Local 3D wheel/joystick or keyboard game

For a machine with a desktop display, run the Pygame frontend instead of the
browser server. A wheel is optional:

```bash
cd /p/yufeng/tri30/dreamer4

/p/yufeng/.conda/envs/dreamer4/bin/python \
  waymo/local_puffer_fanatec_game.py \
  --device cuda:0 \
  --ckpt h90 \
  --input-device auto \
  --scene-index 3155 \
  --unroll-steps 150 \
  --puffer-use-inherited-display \
  --puffer-manifest \
  /p/yufeng/tri30/dreamer4/waymo/cache/pufferdrive_val128_seed0/manifest.csv
```

This manifest is the vehicle-focus portion of the fixed random-128 validation
subset recorded in `waymo/evaluation/val_random128_seed0_manifest.json`: 118
views currently remain after removing three pedestrian-focus views, six
cyclist-focus views, and manually excluded bad scene `#4428`. Its first
recorded sample has full-validation dataset index 3155, hence the initial index
above. Five of the 118 views have a missing focus observation somewhere during
recorded frames 1–11. Because Puffer's chase camera requires a valid focus on
every displayed frame, 3D scene switching skips those five and selects among
the 113 compatible views. Original Dreamer NPZ and raw Waymo data are preserved
for all exclusions. The 2D model path retains the dataset's true per-frame
validity masks.

Input defaults to `--input-device auto`: it uses a compatible SDL
wheel/joystick when one is present and otherwise starts in keyboard mode. You
can select a source explicitly:

```bash
--input-device keyboard   # Always use keyboard
--input-device wheel      # Require a wheel/joystick
--input-device auto       # Wheel when available; keyboard fallback (default)
```

Keyboard driving controls are Up or W for throttle, Down or S for brake, and
Left/A or Right/D for steering. In `auto` mode, held driving keys temporarily
override the wheel, and unplugging the wheel safely falls back to keyboard.

The frontend requires Pygame in the Dreamer environment. On this machine
Pygame 2.5.2 is already available. If another installation reports that it is
missing, install it with:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python -m pip install pygame
```

The local window displays PufferDrive's agent-perspective JPEG frames while a
background thread owns Dreamer inference. The local entrypoint enables the
inherited X11 display by default (the explicit flag above is documentary), so
it does not require Xvfb on the desktop machine. The default Fanatec mapping comes
from `/p/liverobotics/yf_metadrive/code/pygame_test.py`:

```text
axis 0: wheel     (left -1, center 0, right +1)
axis 2: throttle  (released +1, pressed toward -1)
axis 5: brake     (released +1, pressed toward -1)
axis 1: ignored third pedal
```

Puffer/Dreamer steering is left-positive, so wheel axis 0 is inverted by default.
The overlay shows both raw axes and normalized controls. If SDL enumerates a
different device or the firmware exposes different axes, use, for example:

```bash
--joystick-index 1 --steering-axis 0 --throttle-axis 2 --brake-axis 5
```

The steering bar should follow the physical wheel direction. Use
`--no-invert-steering` only if it does not, and confirm that the vehicle itself
also turns the correct way. Available keyboard commands are Space to pause, `.`
to single-step, `R` to reset, `N` to select another converted scene, and
Q/Escape to quit. `N` selects only NPZ views present in the Puffer manifest; it
reloads the current view when a manifest contains just one scene.
