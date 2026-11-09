# Whole-scene WOMD cache: loader, sparse agent attention, ADE eval

Data: `/p/yufeng/waymo_scene_prep` (one ego-independent NPZ per WOMD scenario,
full training + validation; see its README).

- `dataset.py`: `SceneCacheDataset` + `collate_scenes`. Picks a center agent
  (SDC by default), rotates into its frame and emits the same fields as
  `ConnectivityDataset`/`collate_connectivity`, so existing direct-action-flow
  models and rollouts run unchanged. All agents valid at the current frame are
  kept by default (WOSAC), padded per batch. Options: `agent_radius_m`,
  `max_agents`, `map_crop_m` (lane edges are remapped), `center="random_ttp"`.
- `sparse_agent_attention.py`: `patch_agent_attention(model, knn, radius_m)`.
  All agent-agent attention uses the pairwise `RelativeAgentBias`, so this one
  patch covers the scene encoder, decoder memory and `agent_attn`.
  `knn=0` is exact dense (row-chunked); `knn=K` keeps the K nearest valid
  agents within `radius_m`, rebuilt every replan (SMART/TrajTok radius graph).
- `eval_scene_ade.py`: closed-loop ADE with the ERD/MON validation protocol,
  per scene, bucketed by agent count, plus a legacy `val_ade_m` reproduction.
- `launch_eval_tmux.sh`: detached run, e.g. `CUDA_DEVICE=1 bash waymo/scene_cache/launch_eval_tmux.sh`.
- `tests/test_against_legacy.py`: loader vs legacy focus NPZ + sidecars (global
  coordinates, matched by track/map id) and patched-vs-original bias.
