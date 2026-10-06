# SMART / TrajTok map adaptations: fresh stage one

This run uses the settings of
`waymo/checkpoints/waymo_daf_h40_b5_typeaware_vxvy_scratch100k_20260925/best.pt`
(step 95,000), with random initialization and a new output directory.
The original Huber flow objective, action statistics, measured agent lengths,
type-aware calibration, train/val split and agent selection are reused.
No lane connectivity edges or additional loss terms are added.

## Submit on the GPU host

```bash
cd /p/yufeng/tri30/dreamer4
CUDA_DEVICE=0 bash waymo/adaptations/launch_stage1_tmux.sh
```

On the other server, use `cd /scratch/baz7dy/tri30/dreamer4` and run the same
launcher command. Repository/data paths follow the script location. The original
manifest can retain `/p/yufeng/...` NPZ paths: preparation resolves each filename
under the current `DATA_ROOT/{train,val}` in memory, without rewriting the file.
Training/statistics/calibration paths are also resolved under the current data
root. Activate the dreamer4 environment before launching, or set `PYTHON`.

Change `CUDA_DEVICE` to your selected GPU. The script creates the detached tmux
session; you do not need to wrap it in another `tmux new-session` command.
It checks CUDA, builds a reusable Scenario ID index, prepares **all 50,000** map
sidecars, runs CPU preflight, then trains. Preparation can take substantial time
and disk space. It does not update original NPZ files or baseline checkpoints.
Index and map cache generation resume on repeated invocation; an existing
training checkpoint/log requires a new `RUN_NAME`.

```bash
tmux attach -t daf_map_stage1_cuda0
tail -f waymo/logs/wm/waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_scratch100k_20261006.log
```

Defaults:

- Python: automatically selected from the active conda environment, owner/home
  dreamer4 environment, or current `python`; must import torch/numpy/protobuf
- Source data: `data/waymo_vector_dataset_ooi_centered_50k_with_lengths`
- Raw Scenario: `/p/liverobotics/waymo_open_dataset_motion/scenario/training`
- Index: `data/waymo_scenario_index_v1`
- Map cache: `data/waymo_map_smarttrajtok_5m_v2`
- Output: `waymo/checkpoints/waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_scratch100k_20261006`

`PYTHON`, `DATA_ROOT`, `SCENARIO_ROOT`, `INDEX_DIR`, `MAP_CACHE_DIR`, `PREP_WORKERS`,
`RUN_NAME`, `SESSION_NAME`, and `CUDA_DEVICE` can be set as environment variables.
`SCENARIO_ROOT` must contain a `training` directory of **Scenario protobuf**
TFRecords, not tf.Example TFRecords.

## Training settings

H40/B5, history 11, type-aware execution with vx/vy input, d_model 256,
heads 8, depths history/map/scene/action/refiner = 2/2/4/8/2, dropout 0.05,
batch 4 × accumulation 4, AdamW lr 5e-5, warmup 5k, cosine endpoint 500k,
EMA 0.9999, bf16, seed 0, max steps 100k. All agents including focus are
jointly generated (`condition_focus_actions=False`). Dynamic loss balancing
stays disabled. Frozen baseline argv is in `baseline_launch_config.json`.

## Implemented geometry and attention

Map features are reconstructed from raw map geometry. Keep each whole feature
within the original 100m crop around the same focus/OOI trajectories, plus every
feature ID retained by the baseline NPZ. There is no 256-segment cap. Split by
source polyline arc length into 5m pieces, resample six points per piece and
preserve the final short piece and shared endpoints. This is an approximately
1m sampling representation; bends between samples are approximated.

Each segment keeps its original map ID. Segment position is its sampled mean;
heading is the mean tangent direction. Directed attention bias encodes distance,
sin/cos bearing in the query frame, and sin/cos heading difference. Map-to-map
attention gathers at most 32 nearest segments within 30m. Each agent queries at
most 64 nearest segments within 100m. The action decoder queries all agent
context tokens and this agent's local map neighborhood with the same geometry
bias. Neighborhood search is chunked, and attention gathers actual neighbors;
there is no dense M×M map-attention matrix. Distance search is still O(M²) work,
with bounded temporary memory, so larger coverage still increases runtime.

Rollout recomputes agent-to-map neighborhoods and relative geometry at each
replan from **executed** current poses (every B=5 steps), while the static map
frame stays fixed. It does not update geometry inside each flow solver iteration
or predict future light states inside the H=40 plan.

## Traffic lights and stop signs missing from NPZ

Existing NPZs contain `light_ids` but do not preserve stop-sign controlled-lane
relationships. Use the original Scenario fields:

- `scenario.dynamic_map_states[t].lane_states`: `lane`, `state`, `stop_point`.
  `lane` is the controlled `map_features[].id`.
- `scenario.map_features[].stop_sign`: `lane` lists controlled lane IDs;
  `position` is the sign position, not a stop-line position.
- `scenario.map_features[].lane.polyline`: lane centerline geometry.

The cache builder performs these exact ID joins. Lane segments receive the
**current observed** signal state/presence and relative stop point, plus a
stop-sign flag and relative sign position. No separate light/sign tokens are
used in the adapted encoder. If multiple signs control a lane, keep the nearest
sign to each segment. Unobserved signal states remain masked; state 0 with an
observed record remains a present UNKNOWN signal, never an assumed green light.

Both shard order and track slot order differ between tf.Example and Scenario.
`index_scenarios.py` matches by `scenario_id`, records byte offsets, and caches
source size/mtime. `prepare_map_cache.py` resolves crop source slots to track
IDs using NPZ metadata, checks scenario identity, timeline and current agent
positions, and applies the original `ego_origin_xy` / `ego_heading` transform.
Missing source scenes or identity mismatches fail loudly.

Individual preparation commands (CPU only):

```bash
PYTHON="$(command -v python)"  # activate the dreamer4 environment first
"$PYTHON" waymo/adaptations/index_scenarios.py \
  --scenario_root /p/liverobotics/waymo_open_dataset_motion/scenario \
  --output_dir data/waymo_scenario_index_v1 --workers 4
"$PYTHON" waymo/adaptations/prepare_map_cache.py \
  --data_root data/waymo_vector_dataset_ooi_centered_50k_with_lengths \
  --index_dir data/waymo_scenario_index_v1 \
  --output_dir data/waymo_map_smarttrajtok_5m_v2 --workers 4
```

The map cache is versioned and stale settings are rejected. An incomplete
`--max_files` cache cannot launch the full training run.

## Verification and evaluation

```bash
python waymo/adaptations/verify.py  # in the dreamer4 environment
```

This runs adaptation and existing action-flow/velocity regression functions
without requiring pytest. Real-data CPU forward/backward evidence is in
`verification/cpu_real_data_preflight.json`. Baseline parameter comparison is
in `verification/baseline_config_comparison.json`.

GPU training was not started during preparation. CUDA was unavailable in the
code-editing environment, so GPU/bf16 memory use and full 50k cache generation
remain for the submitting host. The launcher retains the original microbatch.

The trainer's one-shot/sample/receding validation paths load the sidecars.
For other standalone evaluators, load `MapAdaptationDataset(data_dir,
map_cache_dir)` and use the existing trainer `collate_vector_batch` / `scene_kwargs`.
For direct `rollout_receding_horizon` calls, pass `map_ids`, `map_is_lane`,
`map_stop_sign`, `map_stop_point`, and `light_id_sequence`. Legacy evaluator
loaders using only the old NPZ fields require this integration before they can
run the new checkpoints; legacy checkpoints retain their original architecture.

Indexes and map sidecars are generated separately on each server. If moving a
previously generated index/cache as well as the source data, their stored
absolute-path identity checks may reject them; regenerate them at the new path.
