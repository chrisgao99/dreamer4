# Waymo tokenizer → world model 两阶段训练代码与实验上下文

更新时间：2026-07-24

这份文件用于把当前 Waymo vector tokenizer、标准 latent world model、带 motion head 的 MotionLatent V1、H30/H90 exact rollout fine-tuning，以及正在运行的实验完整交接给新的 Codex 会话。

## 给新 Codex 会话的最短提示词

可以在新会话直接发送：

> 请先完整阅读 `/p/yufeng/tri30/dreamer4/agent_context/WORLD_MODEL_PIPELINE_CODE_CONTEXT_2026_07_24.md`。然后检查文档里两个 tmux pipeline 的当前状态和日志。不要重复提交已有阶段，不要 reset/revert 当前 dirty worktree；后续修改必须保持标准 WM 与 MotionLatent V1 的 context/window、batch、训练预算和评测协议可比较。

## 1. 当前整体流程

```text
Waymo filtered vector scene (.npz, usually 91 frames)
    |
    v
Vector tokenizer (frozen for all world-model experiments)
    |  encode scene → z: [B,T,64,64]
    |
    +-----------------------------+
    |                             |
    v                             v
Standard latent WM                MotionLatent V1
(no explicit motion head)         (explicit q + motion head)
    |                             |
Stage 1 shortcut                  Stage 1 native online rollout
    |                             |
best <=600k checkpoint            best <=600k checkpoint
    |                             |
exact ctx1/H30 50k                exact ctx1/H30 50k
    |                             |
exact ctx1/H90 50k                exact ctx1/H90 50k
    |                             |
    +------------ unified horizon evaluation ------------+
```

Repository root:

```text
/p/yufeng/tri30/dreamer4
```

Dataset used by the current experiments:

```text
/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_ooi_centered_50k/train
/p/yufeng/tri30/dreamer4/data/waymo_vector_dataset_ooi_centered_50k/val
```

## 2. Dataset and tokenizer

### 2.1 Dataset code

Primary dataset implementation:

```text
waymo/core/waymo_vector_dataset.py
```

Each item is one filtered vector scene. Dynamic tensors include agents and traffic lights over time; static map polylines are stored once per scene.

### 2.2 Tokenizer training entry points

Compatibility entry point:

```text
waymo/training/train_waymo_vector_tokenizer.py
```

It only forwards execution to the real trainer:

```text
waymo/training/tokenizer/train_waymo_vector_tokenizer.py
```

Generic historical tmux launcher used for this tokenizer family:

```text
waymo/training/tokenizer/launch_ooi50k_lat16_d256_ep200_2a100_staticmap_v2_chunk32_trajloss_randstart_tmux.sh
```

The launcher's defaults are configurable through environment variables; the exact current tokenizer configuration should be taken from the checkpoint args below, not inferred from the launcher's filename/defaults.

### 2.3 Tokenizer model code

```text
waymo/core/vector_tokenizer_encoder.py
waymo/core/vector_tokenizer_decoder.py
```

Important components:

- `VectorStaticMapQueryEncoder`: dynamic latent/agent/light tokens query a separately encoded static map.
- `TimeSelfAttention`: dynamically builds a causal `T x T` mask; there is no learned fixed maximum temporal length.
- `VectorBlockCausalTokenizerDecoder`: decodes latent sequence back to agent and traffic-light states.
- `dreamer4/model.py::add_sinusoidal_positions`: dynamically creates sinusoidal embeddings for the input `T`.

### 2.4 Frozen tokenizer used by both current WM pipelines

```text
/p/yufeng/tri30/dreamer4/waymo/checkpoints/
ooi50k_lat64_b64_d256_ep200_anygpu_staticmap_v2_chunk32_raw_kinematic_nofde_focus_randstart_noamp/best.pt
```

Parameters read directly from `best.pt`:

```text
checkpoint step:              83,500
checkpoint epoch:             118
batch_size per process:       32
time_window:                  32
random_time_window_start:     true
d_model:                      256
encoder depth:                4
decoder depth:                4
n_heads:                      4
n_latents:                    64
d_bottleneck:                 64
encoder_variant:              static_map_query
time_every:                   1
map_depth:                    2
map_cross_every:              1
map_query_tokens:             latent_agent
decoder_agent_token_mode:     none
agent_xy_loss:                smooth_l1
agent_xy_parameterization:    absolute
agent_kinematic_xy_weight:    5
agent_speed_yaw_kinematic_weight: 2
focus_agent_weight:           4
lr:                           3e-4
weight_decay:                 1e-4
```

Tokenizer output before world-model packing:

```text
z: [B,T,64,64]
```

World-model packing factor is 2:

```text
packed z: [B,T,32,128]
```

### 2.5 Critical tokenizer length caveat

`chunk32` / `time_window=32` is a tokenizer **training crop**, not an architectural maximum. The world-model code can and currently does pass an entire 91-frame scene to the frozen tokenizer in one encoder call.

Therefore H90 currently means:

```text
scene first 91 frames
→ tokenizer encode all 91 frames at once
→ z_gt[0:91]
```

There is no automatic `32 + 32 + 27` chunking in the current WM pipelines. Positions 32–90 and temporal histories longer than 32 are supported by code but are out of the tokenizer's training-length distribution. The decoder has the same length-extrapolation issue when decoding 91 latent frames.

Optional tokenizer chunk support exists in the standard trainer through:

```text
train_waymo_world_model.py::encode_batch_dynamics_inputs_for_world_model
train_waymo_world_model.py::decode_batch_z_for_world_model
--tokenizer_chunk_window
--tokenizer_chunk_stride
```

The current matched pipelines leave chunking disabled, so both methods share the same tokenizer length behavior.

## 3. Standard latent world model: no explicit motion head

### 3.1 Core code

Main trainer and shared Waymo WM utilities:

```text
waymo/training/world_model/train_waymo_world_model.py
```

Backbone model:

```text
dreamer4/model.py::Dynamics
```

There is no separate physical motion head and no explicit propagated agent state `q`. The dynamics transformer predicts packed tokenizer latents. Ego action and static map tokens are conditioning inputs.

Important functions in `train_waymo_world_model.py`:

```text
slice_time_window
encode_batch_dynamics_inputs_for_world_model
pack_bottleneck_to_spatial / unpack_spatial_to_bottleneck
build_ego_action_features
dynamics_pretrain_loss
sample_one_timestep_packed
sample_autoregressive_packed_sequence
rollout_loss
evaluate
train
```

### 3.2 Stage 1: shortcut objective

Trainer selection:

```text
--train_objective shortcut
```

Current matched Stage-1 configuration:

```text
seq_len=11
batch_size=8
self_fraction=0.5
time_every=1
dynamics_attend_map=true
map_cross_every=1
max_rollout_window=11
lr=1e-4
```

Semantics:

```text
clean latent sequence:  z[0:11]                  length 11
random Gaussian:        noise[0:11]              length 11
per-position tau/sigma: independently sampled
model input:            11 partially noised latents
model output:           11 clean-latent estimates
loss:                   all 11 positions
```

It is **not** `11 clean context + one noisy target`. It does not create a 12th temporal slot.

With batch 8 and `self_fraction=0.5`:

- 4 scenes use empirical clean endpoint supervision.
- 4 scenes use shortcut self-consistency/bootstrap.
- The self half executes additional half-step forwards to construct a detached bootstrap target.

### 3.3 Stage 2: exact autoregressive rollout objective

Trainer selection:

```text
--train_objective rollout
```

H30:

```text
seq_len=31
eval_ctx=1
eval_horizon=30
```

H90:

```text
seq_len=91
eval_ctx=1
eval_horizon=90
```

At one future step, the model input is conceptually:

```text
[past generated/known latent frames] + [current noisy latent]
```

`sample_one_timestep_packed` returns the model's last temporal output and ignores outputs for the past positions.

Current denoising schedule:

```text
eval_schedule=shortcut
eval_d=0.25
K=4 denoising/integration substeps per generated future frame
```

Thus H30 performs approximately `30 x 4 = 120` sequential model forwards per rollout, and H90 approximately `90 x 4 = 360`.

Window meaning:

```text
max_rollout_window=11
past_keep=max_rollout_window-1=10
maximum model input = 10 past + 1 current noisy = 11 temporal slots
```

`rollout_loss` compares every generated future latent against `z_gt`, averages the losses over the full horizon, and backpropagates through the generated history. There is no detach between future steps.

`self_fraction` is relevant to Stage 1 shortcut training but is not used by the rollout objective.

### 3.4 Current standard-WM CUDA0 pipeline

Launcher:

```text
waymo/training/world_model/launch_waymo_time1_mapx1_select600k_h30_h90_cuda0_tmux.sh
```

tmux:

```text
wm_time1_mapx1_select600k_h30_h90_cuda0
```

Automatic sequence:

```text
existing Stage-1 checkpoints 50k,100k,...,600k
→ evaluate each with ctx1/H30 on the same 32 validation batches
→ choose minimum focus_agent_fde_m
   tie ordering: focus ADE, then latent MSE
→ exact H30: batch1, 50k, lr=1e-5
→ exact H90: batch1, 50k, lr=5e-6
```

Existing Stage-1 run/checkpoints:

```text
waymo/checkpoints/
waymo_wm_v1_egoact_focus_raw_noclamp_win11_randstart_b8_self05_norecon_time1_mapx1_1m/
```

The original 1M pipeline was intentionally stopped around step 879k. No checkpoints were deleted. Only checkpoints through 600k participate in the matched selection.

Logs:

```text
waymo/logs/evaluation/
waymo_wm_v1_egoact_focus_raw_noclamp_win11_randstart_b8_self05_norecon_time1_mapx1_1m_select_upto600k_h30_cuda0.log

waymo/logs/wm/
waymo_wm_time1_mapx1_best600k_h30_50k_exact_ctx1_h90_b1_50k_pipeline.log
waymo_wm_time1_mapx1_best_upto600k_exact_ctx1_h30_b1_50k.log
waymo_wm_time1_mapx1_best600k_h30_50k_exact_ctx1_h90_b1_50k.log
```

## 4. MotionLatent V1: explicit q and motion head

### 4.1 Extra semantic reader

Trainer:

```text
waymo/training/world_model/train_waymo_semantic_reader.py
```

Model component:

```text
waymo/training/world_model/motion_latent_v1.py::LightweightAgentSemanticReader
```

Current frozen reader checkpoint:

```text
waymo/checkpoints/
waymo_semantic_reader_agent32_zonly_d256_depth2_b8_20k_v1/best.pt
```

The reader approximates `P(q|z)` without raw `q` or map input. It is pretrained separately and frozen during MotionLatent world-model training. It is used for latent/physical-state consistency loss; it is not another rollout stage.

### 4.2 MotionLatent model code

```text
waymo/training/world_model/motion_latent_v1.py
```

Main class:

```text
MotionLatentDynamicsV1
```

Key behavior:

1. Tokenize explicit agent state `q=(x,y,speed,vx,vy,valid,yaw,type)` into agent tokens.
2. Feed packed latent tokens and q tokens through one shared `Dynamics` backbone.
3. `motion_head` predicts 6 values per agent:
   - `dv_x, dv_y`
   - `d_yaw`
   - `xy_correction_x, xy_correction_y`
   - validity logit
4. `integrate_motion` performs hard kinematic integration and creates `q_next`.
5. The controlled focus-agent state is overwritten using the supplied ego action exactly.
6. `condition_latent_on_q` injects `q_next` back into the final latent endpoint using q-to-latent attention.

Loss helpers in the same file:

```text
motion_targets
integrate_motion
motion_residual_loss
semantic_reader_loss
```

### 4.3 MotionLatent trainer

```text
waymo/training/world_model/train_waymo_motion_latent_v1.py
```

Important functions:

```text
choose_context
make_prediction_inputs
exact_rollout_loss
train
```

It supports:

```text
--train_mode online_step
--train_mode exact_rollout
--init_ckpt
--resume
```

### 4.4 MotionLatent Stage 1: native online-step training

This is not the standard WM shortcut objective.

Current matched settings:

```text
random model initialization
train_mode=online_step
rollout_end=90
time_every=1
max_context=10
batch_size=8
max_steps=600k
lr=1e-4
map_cross_every=1
```

Native training behavior:

- Encode the first 91 scene frames to `z_gt` once for the batch.
- Choose starting context using the existing curriculum (early training favors the old 11-frame start; later it mixes ctx1 and the long-context start).
- Predict one direct `d=1` next latent and one `q_next` at a time.
- Perform one optimizer update for each target time.
- Append predicted `z_next/q_next` to rollout history, but detach them before later target steps.
- Actual transformer history is clipped by `max_context=10`, then one current noisy token is appended, so total input is at most 11.

The log line still says `ctx11 curriculum`; this is legacy wording and the starting target index may still be 11. `make_prediction_inputs` clips the history to `max_context=10` before the model call.

Unlike exact rollout, later losses cannot backpropagate through earlier predicted states because online-step mode detaches the feedback.

### 4.5 MotionLatent Stage 2: exact rollout fine-tuning

```text
train_mode=exact_rollout
fixed context=1
```

H30 and H90 use the same forward rollout path as MotionLatent evaluation:

- Start from `z_gt[0], q_gt[0]`.
- Predict one direct `d=1` `z_next, q_next` per future frame.
- Feed predictions back without detach.
- Clip history to 10 frames.
- Average latent, motion, and semantic consistency losses over the complete horizon.
- Backpropagate through the entire generated H30 or H90 chain.

MotionLatent uses one direct `d=1` model call per future frame. This differs from the standard WM's four `d=0.25` denoising substeps per future frame.

### 4.6 Current matched MotionLatent CUDA1 pipeline

Launcher:

```text
waymo/training/world_model/launch_waymo_motion_latent_v1_matched600k_h30_h90_cuda1_tmux.sh
```

tmux:

```text
wm_motion_latent_v1_matched600k_h30_h90_cuda1
```

Automatic sequence:

```text
random-init MotionLatent time_every1/max_context10
→ online-step Stage 1: batch8, 600k
→ evaluate checkpoints 50k,100k,...,600k with ctx1/H30 on 32 val batches
→ choose minimum focus_agent_fde_m
→ exact H30: batch1, 50k, lr=1e-5
→ exact H90: batch1, 50k, lr=5e-6
```

Logs:

```text
waymo/logs/wm/
waymo_motion_latent_v1_matched_time1_ctx10_b8_stage1_600k.log
waymo_motion_latent_v1_matched_time1_ctx10_best600k_h30_50k_exact_h90_b1_50k_pipeline.log
waymo_motion_latent_v1_matched_time1_ctx10_best600k_exact_h30_b1_50k.log
waymo_motion_latent_v1_matched_time1_ctx10_best600k_h30_50k_exact_h90_b1_50k.log

waymo/logs/evaluation/
waymo_motion_latent_v1_matched_time1_ctx10_b8_stage1_600k_select_upto600k_h30_cuda1.log
```

## 5. Two world-model variants at a glance

| Property | Standard latent WM | MotionLatent V1 |
|---|---|---|
| Main trainer | `train_waymo_world_model.py` | `train_waymo_motion_latent_v1.py` |
| Model | `dreamer4.model.Dynamics` | `motion_latent_v1.MotionLatentDynamicsV1` |
| Explicit physical state q | No | Yes |
| Motion head | No | Yes, 6 values per agent |
| Hard kinematic integration | No | Yes |
| Semantic reader | No | Yes, frozen |
| Stage-1 objective | Parallel shortcut denoising | Detached online autoregressive transitions |
| Stage-1 length | 11 total temporal slots | Up to 10 history + current |
| Exact rollout context | 1 GT frame | 1 GT frame/q state |
| Exact max past | 10 | 10 |
| Denoising per future | 4 substeps at d=0.25 | One direct d=1 call |
| Exact feedback detach | No | No |
| H30 output length | 1 context + 30 predicted = 31 | Same |
| H90 output length | 1 context + 90 predicted = 91 | Same |

## 6. Unified evaluation

Evaluator:

```text
waymo/evaluation/eval_waymo_world_model_horizons.py
```

It detects MotionLatent checkpoints through checkpoint format:

```text
waymo_motion_latent_world_model_v1
```

and selects either:

```text
standard: wm.evaluate / legacy_flow sampler
MotionLatent: evaluate_motion_latent_v1 / direct_d1 sampler
```

Common important arguments:

```text
--eval_ctx 1
--horizons 10 30 50 80 90
--eval_seq_len 91
--eval_schedule shortcut
--eval_d 0.25                 # standard WM only
--max_rollout_window 11       # standard WM total window
--use_ego_actions
--ego_action_source focus
--ego_action_normalization raw
--no-ego_action_clamp
```

Metrics to compare:

```text
latent_mse_future
agent_xy_mae_m
agent_fde_mae_m
focus_agent_xy_mae_m
focus_agent_fde_m
agent_speed_mae_mps
agent_vxvy_mae_mps
agent_yaw_mae_deg
agent_valid_acc
```

The current Stage-1 checkpoint-selection criterion is H30 `focus_agent_fde_m` on a fixed 32-batch validation subset. This is a model-selection choice and should be recorded in any comparison table.

## 7. Previous useful MotionLatent result

Old MotionLatent checkpoint:

```text
waymo/checkpoints/
waymo_motion_latent_v1_qkin_preader_ctx1ctx11_b8_mapx1_1m/step_00650000.pt
```

Exact H30 fine-tune from that 650k checkpoint:

```text
waymo/checkpoints/
waymo_motion_latent_v1_qkin_preader_exact_fixed_ctx1_h30_b4_init650k_50k/
final_step_00050000.pt
```

H30 comparison on 128 validation batches:

```text
                              old 650k       exact H30 50k
latent MSE                    0.0766         0.0501
all-agent ADE                 4.9590 m       3.4988 m
all-agent FDE                 6.8779 m       4.6515 m
focus ADE                     0.6726 m       0.2219 m
focus FDE                     1.2934 m       0.3548 m
```

Training-loss bins showed most H30 adaptation by roughly 20k–30k; 50k was sufficient and had reached a plateau. The matched experiments nevertheless keep 50k for both variants.

## 8. Important fairness and interpretation caveats

1. **Tokenizer length mismatch:** both H90 methods use a tokenizer trained on random 32-frame windows but encode/decode 91 frames in one call.
2. **Window terminology:** standard `max_rollout_window=11` means 10 past + 1 current. MotionLatent `max_context=10` means the same total input length 11. Old MotionLatent `max_context=11` meant total input could be 12.
3. **Step counts are not FLOPs:** the matched runs use Stage-1 batch8/600k optimizer updates and exact batch1/50k+50k. The two Stage-1 objectives do different work per update, and MotionLatent online-step reuses a scene batch across many target times. This matches nominal steps and batch, not exact GPU FLOPs or unique-scene exposure.
4. **Different future-frame sampler:** standard WM uses 4 denoising calls per future; MotionLatent uses one direct call plus explicit q integration. This is part of the method difference but changes compute.
5. **Selection subset:** best Stage-1 checkpoint is chosen on only 32 validation batches. Final claims should rerun selected checkpoints on the same larger evaluation set, e.g. 128 batches or full validation.
6. **H90 training cost/memory:** exact H90 backpropagates through a long free-running chain. Current exact batch is intentionally 1 on each 80GB A100.
7. **No need to submit later stages manually:** each current tmux script is a sequential, resumable pipeline. It selects the Stage-1 checkpoint and launches H30 then H90 automatically.

## 9. Current workspace safety

The repository worktree is dirty and contains user work plus newly added/untracked world-model files. Do not run destructive cleanup commands, `git reset --hard`, or revert unrelated changes.

Important locally modified or newly introduced code includes:

```text
waymo/training/world_model/train_waymo_world_model.py
waymo/evaluation/eval_waymo_world_model_horizons.py
waymo/training/world_model/motion_latent_v1.py
waymo/training/world_model/train_waymo_motion_latent_v1.py
waymo/training/world_model/train_waymo_semantic_reader.py
waymo/training/world_model/launch_waymo_time1_mapx1_select600k_h30_h90_cuda0_tmux.sh
waymo/training/world_model/launch_waymo_motion_latent_v1_matched600k_h30_h90_cuda1_tmux.sh
```

Before editing, always inspect `git status` and the exact overlapping diff. Preserve logs, checkpoints, and existing experiments.

## 10. Status/check commands for a new session

Run from `/p/yufeng/tri30/dreamer4`:

```bash
tmux list-sessions | rg 'wm_time1_mapx1_select600k|wm_motion_latent_v1_matched600k'

tmux capture-pane -pt wm_time1_mapx1_select600k_h30_h90_cuda0 -S -40
tmux capture-pane -pt wm_motion_latent_v1_matched600k_h30_h90_cuda1 -S -40

tail -50 waymo/logs/wm/waymo_wm_time1_mapx1_best600k_h30_50k_exact_ctx1_h90_b1_50k_pipeline.log
tail -50 waymo/logs/wm/waymo_motion_latent_v1_matched_time1_ctx10_best600k_h30_50k_exact_h90_b1_50k_pipeline.log

nvidia-smi
```

Expected automatic terminal condition for each pipeline is a final H90 checkpoint:

```text
standard WM:
waymo/checkpoints/waymo_wm_time1_mapx1_best600k_h30_50k_exact_ctx1_h90_b1_50k/final_step_00050000.pt

MotionLatent V1:
waymo/checkpoints/waymo_motion_latent_v1_matched_time1_ctx10_best600k_h30_50k_exact_h90_b1_50k/final_step_00050000.pt
```

Do not submit duplicate H30/H90 jobs while the parent tmux pipeline is alive. If a pipeline stops, inspect its pane/log and existing `latest.pt`; both launchers are written to resume completed/partial stages.
