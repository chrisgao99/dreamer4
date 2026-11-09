# Lane connectivity + map losses

Matches CUDA 1 lane-connectivity scratch 100k. Adds vehicle-footprint off-road (0.1), lane-direction consistency (0.05), and illegal lane-transition (0.1) losses. Weights increase linearly during the first 5,000 updates.

Regularizes the existing flow endpoint estimate by executing H40 actions from real history. No additional model forward or generated-history feedback; closed-loop training is disabled.

Off-road supervision uses measured Waymo boundaries and vehicle widths. It approximates local lane corridors rather than exhaustive drivable-area polygons. Unknown boundary regions and invalid agents are masked. Direction uses geometric lane matching; transitions use legal successor and permitted lane-change edges.

Original dataset, map and topology caches remain unchanged. New geometry sidecars are generated in `data/waymo_lane_loss_geometry_5m_v1` using existing raw Scenario files and index. On another server, regenerate these sidecars; repository-relative paths support relocation. Set PYTHON for the target environment if necessary.

```bash
CUDA_DEVICE=2 bash waymo/lane_losses/launch_tmux.sh
tmux attach -t daf_laneconn_maploss_stage1_20261007_cuda2
```

The detached session prepares 50,000 files, checks CPU and CUDA bf16 forward/backward, then automatically trains 100k updates. Log: `waymo/logs/wm/waymo_daf_h40_b5_typeaware_vxvy_laneconn_maploss_scratch100k_20261007.log`.

Validation: 9 loss checks, 5 connectivity checks, 28 existing checks passed. Real-data CPU and CUDA 2 bf16 full-model gradients passed; CUDA preflight peak allocation approximately 6 GB. Verification reports are in `verification/`.
