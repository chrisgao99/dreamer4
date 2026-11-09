# Lane connectivity comparison on CUDA 1

This experiment adds lane graph conditioning to the CUDA 0 SMART/TrajTok map
adaptation run. It reuses the same 50k NPZs, 5m map sidecars, train/val split,
action statistics/calibration, H40/B5, batch 4 x accumulation 4, bf16, seed 0,
learning rate/schedule, depths and 100k steps. Fresh random initialization;
no warm start. The frozen comparison settings are in `baseline_cuda0_args.json`.

**Scope: lane connectivity only.** No off-road, lane-direction or transition
losses, and no generated-history closed-loop training are added. These are
separate changes to the training objective and should be another ablation.

## Data and conditioning

`prepare_topology.py` uses the existing Scenario ID byte-offset index and reads
raw `lane.exit_lanes`, `left_neighbors`, `right_neighbors`, and boundary spans.
It writes independent `data/waymo_lane_connectivity_5m_v1/{train,val}` sidecars;
original NPZs and the baseline map cache are not modified.

Segments retain original lane IDs. Directed typed edges encode hold/forward
continuation, successors, and left/right changes. A change is enabled only
inside the annotated neighbor span and where the entire 5m source segment has
known broken boundary coverage (types 1/4/5). Solid/mixed/unknown boundaries,
road edges, and missing target lanes do not create change edges. These are
conservative map-derived permissions; they do not encode restrictions absent
from the Waymo map, such as temporary restrictions or jurisdiction-specific rules.

The sparse map attention gets a learned edge-type bias. Each current agent pose
is associated with up to 4 lane segments within 8m and with heading difference
less than 90 degrees, ranked by distance and heading compatibility. Directed
paths are explored for up to 16 segment hops with a 64-node frontier cap. The
nearest 64 reachable nodes become additional geometry-aware path memory with
hop embeddings, queried by the agent encoder and action decoder. Pedestrians
use the original map context; vehicle/cyclist tokens use lane path memory when
candidates exist. No future GT route or future lane assignment is supplied.

Path conditioning is guidance, not a hard executor constraint: generated poses
can still violate map rules. Receding evaluation recomputes candidate lanes and
reachable paths from executed poses on every B-step replan. The underlying
loss and type-aware execution remain the baseline versions.

## Background preparation and training

```bash
cd /p/yufeng/tri30/dreamer4  # or /scratch/baz7dy/tri30/dreamer4
conda activate dreamer4
CUDA_DEVICE=1 bash waymo/lane_connectivity/launch_tmux.sh
```

The launcher creates a detached tmux session, prepares all 50k graph sidecars,
rechecks whether CUDA 1 is still free, runs CPU and CUDA 1 bf16 forward/backward
preflight, and starts the fresh 100k training run. The whole workflow remains
alive when the screen/SSH session disconnects. Existing valid sidecars resume;
use a new RUN_NAME if an old log/checkpoint exists.

```bash
tmux attach -t daf_laneconn_stage1_20261007_cuda1
tail -f waymo/logs/wm/waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k_20261007.log
```

Output checkpoint directory:
`waymo/checkpoints/waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_scratch100k_20261007`.

Paths follow the repository root; `PYTHON`, `DATA_ROOT`, `MAP_CACHE_DIR`,
`INDEX_DIR`, `TOPOLOGY_DIR`, `RUN_NAME`, `SESSION_NAME`, `PREP_WORKERS`,
`CUDA_DEVICE` can be overridden. Existing topology files record scenario ID and
exact segment-ID order, validated against the baseline map on loading.

## Checks

Five lane-specific regression checks cover directional successors, solid/unknown
boundary exclusion and partial-span handling, route reachability, forward/backward,
checkpoint reconstruction, closed-loop inference, and matched baseline settings.
The baseline adaptation's 28 checks also pass. Real CPU and CUDA 1 bf16 preflight
reports are in `verification/`; GPU sample microbatch 4 peaked at about 6GiB.
The submitted pipeline repeats preflight on its complete dataset before training.
