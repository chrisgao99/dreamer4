# Dynamic agent bias (2026-11-02)

Same data, arguments and two-stage recipe as
`waymo_daf_laneconn90k_mon8_gt_unroll80_b5_detach0_10k_20261009` (laneconn stage one 100k, MON-8
finetune from step 90k), with ONE model change: `dynamic_agent_bias=True`.

## Change

`RelativeAgentBias` is computed once from current poses and reused for every future chunk and
flow step. With the flag, the denoiser's per-chunk agent attention (`JointActionDiTBlock.agent_attn`)
adds a second bias, `FutureAgentBias`: the same directed geometry features (dx, dy, sin/cos dyaw,
distance, type) evaluated on each agent's **planned pose at the end of that chunk**, plus flow time
and an availability flag. Output layer zero-initialized; encoder and decoder memory biases unchanged.

Planned poses come from self-conditioning: the clean-action estimate `x + (1-t) v` of the previous
solver step, integrated holonomically from the current pose (vectorized, detached). Step 0 has no
estimate (flag 0: all geometry zeroed, the bias is constant over keys, softmax unchanged).
Training: a no-grad pass at the same `(x_t, t)` gives the estimate; each example uses it with
probability 0.5. Implemented in `training/world_model/direct_action_flow.py`
(`FutureAgentBias`, `decode_velocity(self_condition=...)`, `sample_normalized_actions`,
`flow_matching_loss`) and `mon_long_rollout.py`. Every entry point must call
`model.set_action_normalizer(normalizer)` (base trainer, MON trainer, `scene_cache/eval_scene_ade.py`
do). Flag off = previous behavior and checkpoints.

Note: the no-grad pass inside autocast must be followed by `torch.clear_autocast_cache()`;
otherwise cached bf16 weight casts carry no autograd history and the whole decoder gets no
gradient. `tests/test_dynamic_bias.py` checks every parameter receives a gradient.

## Multi-GPU (CUDA 0,1,3), same updates

- Stage one (`train_stage1_ddp.py`): global batch 16 per update split 6/5/5 over ranks (was 4x4
  accumulation); rank losses weighted by share, gradients summed. Epoch-seeded global order.
  Rank 0 evaluates/saves exactly like `base.train` (val every 2.5k, samples 10k, receding 50k,
  `step_*.pt` every 2.5k, `best.pt` by val flow loss).
- Stage two (`train_mon80_ddp.py`): same batch on every rank; candidate k scored on rank k%3,
  each rank's share in one batched no-grad rollout with the identical per-candidate noise
  (8 sequential rollouts ~37s, batched ~5s); each distinct winner replayed (batch 2, with
  gradients) on its own rank, without activation checkpointing unless OOM (identical gradients);
  gradients summed. Verified against the single-GPU path: identical grad norms to 6 digits.
  Batched-vs-replay score difference (bf16 rounding) is logged as `replay_max_error` (relative).

Speed: stage one ~3.1 updates/s (baseline 1.34 on one GPU, ~10h for 100k); stage two ~20 s/update
(baseline ~52 s, ~2.3 days for 10k).

## Run

```bash
bash waymo/dynamic_agent_bias/launch_tmux.sh   # tmux session dynbias_20261102; survives logout
tail -f waymo/logs/wm/waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_dynbias_scratch100k_20261102.log
tail -f waymo/logs/wm/waymo_daf_laneconn_dynbias90k_mon8_gt_unroll80_b5_detach0_10k_20261102.log
cat waymo/dynamic_agent_bias/runs/waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_dynbias_scratch100k_20261102/PHASE.txt
```

The launcher freezes the code into `runs/<stage1>/source_snapshot/` (plus git HEAD and diff), runs
the unit test, stage one, then stage two from `step_000090000.pt`. Each stage retries up to three
times from its latest checkpoint; re-running the launcher after a reboot resumes. Outputs:
`waymo/checkpoints/<stage1>/` and `waymo/checkpoints/<stage2>/best_ade.pt` (same selection as before).
