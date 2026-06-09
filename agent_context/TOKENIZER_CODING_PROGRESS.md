# Tokenizer Coding Progress

Last updated: 2026-06-04

## Current Code Location

Waymo-related code and small debug data are now under:

```text
/p/yufeng/tri30/dreamer4/waymo
```

Filtered debug NPZ files are under:

```text
/p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset
```

The command cookbook is:

```text
/p/yufeng/tri30/dreamer4/waymo/HOW_TO_RUN.md
```

## Files Added So Far

### `waymo_vector_filter.py`

Purpose:

- Read Waymo motion tf.Example TFRecords.
- Convert each scenario into fixed-size numpy tensors for the vector tokenizer.

Important implementation detail:

- It reads TFRecords directly with a minimal protobuf definition for `tf.train.Example`.
- It does not require TensorFlow at runtime.

Current output tensors:

```text
agents:        (K, 91, 8)
agent_mask:    (K,)
agent_ids:     (K,)
map_polylines: (M, P, 6)
map_mask:      (M, P)
map_ids:       (M,)
lights:        (91, 16, 4)
light_mask:    (91, 16)
light_ids:     (91, 16)
ego_origin_xy: (2,)
ego_heading:   scalar
```

Current agent feature order:

```text
x, y, speed, vx, vy, valid, yaw, type
```

Current map feature order:

```text
x, y, dir_x, dir_y, type, valid
```

Current traffic-light feature order:

```text
x, y, state, valid
```

Filtering procedure:

1. Parse Waymo tf.Example features.
2. Build full 91-step agent arrays from past/current/future fields.
3. Identify ego/SDC from `state/is_sdc`; fallback to most-valid track if unavailable.
4. Select ego as slot 0.
5. Select the closest `K-1` other agents by minimum valid distance to ego.
6. Build a local roadgraph crop by keeping map ids with points near the ego trajectory.
7. Split kept map ids into fixed-length polyline chunks.
8. Build 91-step traffic-light tensors.
9. Normalize geometry to the ego current frame by default.
10. Save `ego_origin_xy` and `ego_heading` for future world-coordinate conversion.

Coordinate-frame decision:

- Agent positions are ego-centric.
- Agent velocities are rotated into the ego frame.
- Agent yaw subtracts ego heading.
- Map positions are ego-centric.
- Map direction vectors are rotated into the ego frame.
- Traffic-light positions are ego-centric.
- Semantic fields such as type, state, id, and valid are unchanged.

This follows the same principle as MTR center-object coordinates, but our center object is the ego/SDC.

### `waymo_vector_dataset.py`

Purpose:

- PyTorch dataset wrapper for filtered NPZ files.

Current behavior:

- Loads `.npz` files from a directory or a list of directories/files.
- Returns tensors for agents, map polylines, lights, masks, ids, and ego transform metadata.
- Current dataset keeps agents in saved layout `(K, T, F)`.
- The encoder accepts this layout and internally transposes to `(T, K, F)`.

### `inspect_waymo_vector_filter.py`

Purpose:

- Quick CLI for one raw TFRecord.
- Prints filtered tensor shapes, selected agent ids, map counts, light counts, and coordinate ranges.

Used to verify the filter on:

```text
/p/liverobotics/waymo_open_dataset_motion/tf_example/training/training_tfexample.tfrecord-00082-of-01000
```

### `visualize_waymo_vector_npz.py`

Purpose:

- Visualize one filtered NPZ file.
- Write an MP4 video and a VSCode-friendly PNG preview sheet.

Visualization contents:

- gray cropped map polylines
- ego agent in green
- other agents with heading arrows and short trails
- traffic lights colored red/yellow/green based on Waymo state
- simple HUD with scenario id, timestep, and entity counts

Dependencies:

- Uses OpenCV (`cv2`) only for drawing/writing MP4.
- Uses `ffmpeg` only if optional GIF output is requested.

### `vector_tokenizer_encoder.py`

Purpose:

- Implement Encoder A and Encoder B for the Waymo vector tokenizer.
- Encoder only. The first decoder baseline now lives separately in `vector_tokenizer_decoder.py`.

#### Encoder A: repeated map over time

Status:

- Implemented as `VectorBlockCausalEncoder`.
- Select in training with:

```text
--encoder_variant repeat_map
```

High-level procedure:

1. Load filtered tensors:

```text
agents:        (B, K, T, 8) or (B, T, K, 8)
agent_mask:    (B, K)
map_polylines: (B, M, P, 6)
map_mask:      (B, M, P)
lights:        (B, T, L, 4)
light_mask:    (B, T, L)
```

2. Convert agents internally to:

```text
(B, T, K, 8)
```

3. Encode agent states with `AgentFeatureEncoder`.

4. Encode static map polylines once with `MapFeatureEncoder`.

5. Repeat map tokens over time:

```text
(B, M, D) -> (B, T, M, D)
```

6. Encode traffic lights with `TrafficLightEncoder`.

7. Add `N_latents` learnable latent tokens per timestep.

8. Concatenate tokens per timestep:

```text
[latent_1..latent_N, agent_1..agent_K, map_1..map_M, light_1..light_L]
```

9. Add Dreamer-style sinusoidal time and slot positions.

10. Run block-causal transformer layers.

11. Project latent tokens through tanh bottleneck:

```text
z: (B, T, N_latents, D_bottleneck)
```

#### Encoder B: static map query

Status:

- Implemented on 2026-06-04 as `VectorStaticMapQueryEncoder`.
- Uses the same `MapFeatureEncoder`, `AgentFeatureEncoder`, and `TrafficLightEncoder` stems as Encoder A.
- Select in training with:

```text
--encoder_variant static_map_query
```

Pipeline:

```text
agents: (B,T,K,8) -> AgentFeatureEncoder -> (B,T,K,D)
lights: (B,T,L,4) -> TrafficLightEncoder -> (B,T,L,D)
latents: learned tokens -> (B,T,N,D)

map_polylines: (B,M,P,6) -> MapFeatureEncoder -> (B,M,D)
optional map self-attention refinement -> (B,M,D)

dynamic tokens per timestep:
[latent_1..latent_N, agent_1..agent_K, light_1..light_L]
```

Implemented dynamic block:

```text
1. dynamic space self-attention
2. static map cross-attention
3. causal time self-attention
4. MLP
```

Current configurable knobs:

```text
--map_depth        default 2
--map_cross_every  default 1
--map_query_tokens default latent_agent
```

Allowed `--map_query_tokens` values:

```text
latent
agent
latent_agent
all
```

Default behavior:

- latent and agent tokens query map memory after every dynamic space-attention block;
- traffic-light tokens do not query map directly by default;
- traffic-light tokens still communicate with agent/latent tokens through dynamic space attention;
- map tokens stay static and do not participate in temporal attention.

Main-token comparison for the intended `K=32, L=16, M=256, N=16` run:

```text
Encoder A:
  per-timestep tokens = 16 + 32 + 256 + 16 = 320

Encoder B:
  per-timestep dynamic tokens = 16 + 32 + 16 = 64
  static map memory = 256 tokens per scene
```

Feature encoder details:

`AgentFeatureEncoder`

- This is not copied from MTR.
- It is a simple repo-native per-timestep MLP stem for the filtered agent state.
- Input feature order from the filter:

```text
x, y, speed, vx, vy, valid, yaw, type
```

- It converts yaw to:

```text
sin(yaw), cos(yaw)
```

- It mildly normalizes continuous fields:

```text
x / 100
y / 100
speed / 30
vx / 30
vy / 30
valid
sin(yaw)
cos(yaw)
```

- It embeds the discrete agent type with:

```text
nn.Embedding(max_agent_type=16, hidden_dim)
```

- Then it concatenates continuous features and type embedding:

```text
8 continuous features + hidden_dim type embedding
```

- Current MLP structure:

```text
Linear(8 + hidden_dim -> hidden_dim)
SiLU
Linear(hidden_dim -> d_model)
```

- Output:

```text
(B, T, K, d_model)
```

`MapFeatureEncoder`

- This is the part that is MTR-style.
- It follows the PointNet polyline encoder pattern from MTR rather than copying MTR code verbatim.
- MTR reference file:

```text
/p/yufeng/tri30/renference_repos/MTR/mtr/models/utils/polyline_encoder.py
```

- Input map feature order from the filter:

```text
x, y, dir_x, dir_y, type, valid
```

- It mildly normalizes position fields:

```text
x / 100
y / 100
dir_x
dir_y
valid
```

- It embeds lane/map type with:

```text
nn.Embedding(max_lane_type=64, hidden_dim)
```

- Point feature per map point:

```text
5 continuous features + hidden_dim type embedding
```

- Then `PointNetPolylineEncoder` does:

```text
point MLP:
  Linear(in_dim -> hidden_dim)
  SiLU
  Linear(hidden_dim -> hidden_dim)
  SiLU

masked max-pool over points:
  (B, M, P, hidden_dim) -> (B, M, hidden_dim)

concatenate local point feature with pooled polyline feature:
  (B, M, P, hidden_dim * 2)

second point MLP:
  Linear(hidden_dim * 2 -> hidden_dim)
  SiLU
  Linear(hidden_dim -> d_model)

masked max-pool over points:
  (B, M, P, d_model) -> (B, M, d_model)
```

- Output:

```text
(B, M, d_model)
```

- Current first-pass encoder then repeats this over time:

```text
(B, M, d_model) -> (B, T, M, d_model)
```

- This repeat-over-time part is not from MTR and is marked as a current limitation. The planned revision is to keep map as static memory and let dynamic tokens query it.

`TrafficLightEncoder`

- This is not copied from MTR.
- It is a simple repo-native per-timestep MLP stem for traffic-light state.
- Input light feature order from the filter:

```text
x, y, state, valid
```

- It mildly normalizes position fields:

```text
x / 100
y / 100
valid
state_is_nonzero
```

- It embeds the discrete light state with:

```text
nn.Embedding(max_state=16, hidden_dim)
```

- Then it concatenates continuous features and state embedding:

```text
4 continuous features + hidden_dim state embedding
```

- Current MLP structure:

```text
Linear(4 + hidden_dim -> hidden_dim)
SiLU
Linear(hidden_dim -> d_model)
```

- Output:

```text
(B, T, L, d_model)
```

Current smoke-test architecture:

```text
d_model = 128
n_heads = 4
depth = 3
n_latents = 8
d_bottleneck = 32
hidden_dim = 64
time_every = 1
dropout = 0.0
```

These values are for smoke testing only. The class defaults are larger:

```text
d_model = 256
n_heads = 4
depth = 6
n_latents = 8
d_bottleneck = 32
hidden_dim = 128
dropout = 0.05
time_every = 1
```

Current outputs:

```text
z:            (B, T, N_latents, D_bottleneck)
agent_tokens: (B, T, K, D)
map_tokens:   (B, T, M, D)
light_tokens: (B, T, L, D)
token_mask:   (B, T, S)
```

Transformer structure in detail:

- The encoder is a stack of `depth` block-causal transformer blocks.
- Smoke-test setting uses `depth=3`, so there are 3 transformer blocks.
- Default class setting uses `depth=6`, so there are 6 transformer blocks.
- Each block has this order:

```text
RMSNorm
Spatial self-attention over tokens inside each timestep
Residual connection

RMSNorm
Causal time self-attention over each persistent token slot
Residual connection

RMSNorm
MLP / feed-forward network
Residual connection
```

Because `time_every=1` in both the smoke-test and default settings, every block contains one spatial attention layer and one time attention layer. Therefore:

```text
smoke test: 3 blocks = 3 spatial layers + 3 causal time layers + 3 MLPs
default:    6 blocks = 6 spatial layers + 6 causal time layers + 6 MLPs
```

If `time_every=2`, only every second block would include time attention:

```text
depth=6, time_every=2 -> 6 spatial layers + 3 causal time layers + 6 MLPs
```

Spatial layer details:

- Implemented by `SpaceSelfAttention`.
- Input shape is:

```text
(B, T, S, D)
```

- It reshapes to:

```text
(B*T, S, D)
```

- Each timestep is processed independently.
- Attention is dense over valid tokens in the same timestep.
- Token slots include:

```text
latent tokens
agent tokens
map tokens
traffic-light tokens
```

- The attention module is standard multi-head self-attention:
  - one linear layer produces Q/K/V
  - split into `n_heads`
  - PyTorch scaled-dot-product attention
  - output linear projection
- Invalid/padded tokens are masked by `token_mask`.
- Current version uses dense spatial attention rather than MTR's local/sparse attention. This is simpler for debugging but more expensive.

Time layer details:

- Implemented by `TimeSelfAttention`.
- Input shape is:

```text
(B, T, S, D)
```

- It reshapes to:

```text
(B*S, T, D)
```

- Each persistent token slot gets its own causal temporal attention stream.
- Example:

```text
agent slot 3 at time t attends to agent slot 3 at times <= t
latent slot 2 at time t attends to latent slot 2 at times <= t
map polyline slot 17 at time t attends to map polyline slot 17 at times <= t
light slot 5 at time t attends to light slot 5 at times <= t
```

- The causal mask prevents future leakage.
- The attention module is also standard multi-head self-attention with Q/K/V projection, scaled-dot-product attention, and output projection.
- Map tokens are static but currently still pass through time attention because they are repeated across time. This is simple and valid, but may be wasteful. A later version can disable map time attention or separate map memory from dynamic tokens.

MLP / feed-forward details:

- The block uses Dreamer 4's `MLP` from `dreamer4/model.py`.
- It is a gated SiLU MLP:
  - input linear projects to `2 * hidden`
  - split into two halves
  - multiply one half by `silu(other half)`
  - output linear projects back to `D`
  - dropout is applied
- Hidden size is controlled by `mlp_ratio`, default `4.0`.

Normalization:

- Each sublayer uses `RMSNorm` before attention/MLP.
- This follows the existing Dreamer 4 code style.

Positional encoding:

- Before the block stack, the encoder adds Dreamer-style sinusoidal positions with `add_sinusoidal_positions`.
- It adds both:
  - time position over `T`
  - slot position over `S`
- The positions are generated at runtime, so the model can accept variable `T` such as 11, 16, 32, or 91.

### `vector_tokenizer_decoder.py`

Purpose:

- First implementation of Decoder 1 from `TOKENIZER_PLAN.md`: a strict latent-only decoder.
- Provides a `VectorTokenizer` wrapper, decoder output dataclasses, normalized reconstruction targets, and first-pass reconstruction losses for the upcoming training script.

Current decoder inputs:

```text
z:          (B, T, N_latents, D_bottleneck)
agent_mask: (B, K)
light_mask: (B, T, L)
```

Current decoder token layout:

```text
[latent_1..latent_N, agent_queries, light_queries]
```

Current decoder outputs:

```text
agent_continuous:   (B, T, K, 7)
agent_valid_logits: (B, T, K)
light_state_logits: (B, T, L, 16)
light_valid_logits: (B, T, L)
agent_tokens:       (B, T, K, D)
light_tokens:       (B, T, L, D)
token_mask:         (B, T, S)
```

`agent_continuous` uses normalized targets:

```text
x / 100, y / 100, speed / 30, vx / 30, vy / 30, sin(yaw), cos(yaw)
```

Current reconstruction losses:

- SmoothL1 on normalized continuous agent targets, masked by valid agent timesteps.
- BCE on agent valid logits, masked by selected agent slots.
- Cross entropy on traffic-light state, masked by valid light observations.
- BCE on traffic-light valid logits.

Current smoke-test architecture matches the encoder smoke test:

```text
d_model = 128
n_heads = 4
depth = 3
n_latents = 8
d_bottleneck = 32
dropout = 0.0
time_every = 1
```

Smoke-tested commands:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/vector_tokenizer_decoder.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --batch_size 2 \
  --time_window 11
```

This runs forward, finite-value checks, reconstruction loss, and backward.

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/vector_tokenizer_decoder.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --batch_size 2 \
  --time_window 32 \
  --no_backward
```

This runs a longer forward-only window check.

## What Comes From Dreamer 4

Borrowed/adapted from existing Dreamer 4 code:

- Block-causal transformer idea.
- Separate space attention and time attention.
- Causal time attention over token slots.
- Learnable latent tokens.
- Tanh bottleneck projection for `z`.
- Sinusoidal time/slot positional embeddings.
- `RMSNorm`, `MLP`, and `add_sinusoidal_positions` utilities from `dreamer4/model.py`.

Important difference from the image tokenizer:

- The image tokenizer has image patch tokens.
- This vector tokenizer has latent, agent, map, and traffic-light tokens.
- Map tokens are static and repeated across time.
- Time length is variable at runtime.

## What Comes From MTR

Borrowed/adapted from MTR:

- Use center-frame coordinates rather than global coordinates.
- For this project, the center frame is ego/SDC current pose.
- Agent positions, velocities, headings are transformed consistently.
- Map polyline positions and direction vectors are transformed consistently.
- Map encoding uses an MTR-style PointNet polyline encoder:
  - point MLP
  - max-pool over polyline
  - concatenate local point feature with pooled polyline feature
  - second point MLP
  - max-pool to one token per polyline

Not directly reused from MTR yet:

- MTR local attention CUDA ops.
- MTR full data pipeline.
- MTR query decoder.
- MTR loss/evaluation stack.

Current design intentionally keeps dense attention first so shape/debug behavior is easy to verify.

### `train_waymo_vector_tokenizer.py`

Purpose:

- Train Decoder 1 with either Encoder A or Encoder B.
- Keep the comparison clean: no interaction scorer, no dynamics loss, no policy/scene/task tokens.

Encoder selection:

```text
--encoder_variant repeat_map
--encoder_variant static_map_query
```

Static-map-query flags:

```text
--map_depth 2
--map_cross_every 1
--map_query_tokens latent_agent
```

Logged reconstruction losses:

```text
loss_total
loss_agent_xy
loss_agent_vel
loss_agent_yaw
loss_agent_valid
loss_light_state
loss_light_valid
```

Logged physical-unit and accuracy metrics:

```text
agent_xy_mae_m
agent_speed_mae_mps
agent_vxvy_mae_mps
agent_yaw_mae_deg
agent_valid_acc
light_state_acc
light_valid_acc
```

Main fair-comparison training setup:

```text
d_model = 256
n_heads = 4
depth = 4
decoder_depth = 4
n_latents = 16
d_bottleneck = 32
z scalars per timestep = 512
time_window = 32
K = 32
L = 16
M = 256 first, then M = 512 as a static-map stress test
P = 20
```

Recommended comparison table after training:

```text
model
encoder_variant
map_depth
map_cross_every
map_query_tokens
parameter_count
train_steps_per_sec
peak_gpu_memory
val_loss_total
val_agent_xy_mae_m
val_agent_speed_mae_mps
val_agent_vxvy_mae_mps
val_agent_yaw_mae_deg
val_agent_valid_acc
val_light_state_acc
val_light_valid_acc
```

## Smoke Test Results

Encoder A command:

```bash
cd /p/yufeng/tri30/dreamer4

/p/yufeng/.conda/envs/dreamer4/bin/python waymo/vector_tokenizer_encoder.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --batch_size 2 \
  --time_window 11 \
  --encoder_variant repeat_map
```

Observed output:

```text
batch agents: (2, 16, 11, 8)
batch map: (2, 256, 20, 6)
batch lights: (2, 11, 16, 4)
encoder_variant: repeat_map
z: (2, 11, 8, 32)
agent_tokens: (2, 11, 16, 128)
map_tokens: (2, 11, 256, 128)
light_tokens: (2, 11, 16, 128)
token_mask valid: 6283/6512
```

Encoder B command:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/vector_tokenizer_encoder.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --batch_size 2 \
  --time_window 11 \
  --encoder_variant static_map_query \
  --map_depth 2 \
  --map_cross_every 1 \
  --map_query_tokens latent_agent
```

Observed output:

```text
batch agents: (2, 16, 11, 8)
batch map: (2, 256, 20, 6)
batch lights: (2, 11, 16, 4)
encoder_variant: static_map_query
z: (2, 11, 8, 32)
agent_tokens: (2, 11, 16, 128)
map_tokens: (2, 11, 256, 128)
light_tokens: (2, 11, 16, 128)
token_mask valid: 651/880
```

Trainer smoke test, Encoder B:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/train_waymo_vector_tokenizer.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --ckpt_dir /tmp/vector_tokenizer_static_map_query_smoke \
  --batch_size 2 \
  --time_window 11 \
  --max_steps 2 \
  --log_every 1 \
  --eval_every 0 \
  --save_every 0 \
  --encoder_variant static_map_query \
  --map_depth 2 \
  --map_cross_every 1 \
  --map_query_tokens latent_agent \
  --d_model 128 \
  --depth 3 \
  --decoder_depth 3 \
  --n_latents 8 \
  --d_bottleneck 32 \
  --hidden_dim 64 \
  --num_workers 0 \
  --no_amp
```

Result:

```text
Forward, reconstruction loss, backward, validation, metric logging, and checkpoint writing all completed.
```

Trainer smoke test, Encoder A:

```bash
/p/yufeng/.conda/envs/dreamer4/bin/python waymo/train_waymo_vector_tokenizer.py \
  --data_dir /p/yufeng/tri30/dreamer4/waymo/data/waymo_vector_dataset \
  --ckpt_dir /tmp/vector_tokenizer_repeat_map_smoke \
  --batch_size 2 \
  --time_window 11 \
  --max_steps 1 \
  --log_every 1 \
  --eval_every 0 \
  --save_every 0 \
  --encoder_variant repeat_map \
  --d_model 128 \
  --depth 3 \
  --decoder_depth 3 \
  --n_latents 8 \
  --d_bottleneck 32 \
  --hidden_dim 64 \
  --num_workers 0 \
  --no_amp
```

Result:

```text
Original repeat-map training path still runs with the new trainer and metric logging.
```

## Current Limitations

- Encoder B uses dense map cross-attention. Local/nearest map attention is not implemented yet.
- Decoder 1 is still latent-only. Decoder 2 with static map conditioning is intentionally not included in this A/B comparison.
- No interaction scorer or dynamics loss is attached to this tokenizer comparison.
- Dataset currently returns fixed time windows by slicing from the start; random time-window sampling is still a later training improvement.
- Agent selection can still depend on the prepared dataset's selection policy. Causal deployment should use observed/history-only selection.

## Recommended Next Step

Run the main A/B comparison:

```text
Encoder A + Decoder 1, repeated map
Encoder B + Decoder 1, static map query
```

Use identical data, loss weights, bottleneck size, depth, batch size, and training schedule. Compare the validation losses, physical-unit metrics, accuracy metrics, steps/sec, parameter count, and peak GPU memory listed above.

After the fair `M=256` comparison, run `M=512` if a larger map-token dataset is available. This should stress the difference between repeated static map tokens and static map memory.

## Encoder A/B Comparison Plan

Date: 2026-06-05

### Runs

Encoder A, completed:

```text
run_name: ooi50k_lat16_d256_ep200_4a100
log: /p/yufeng/tri30/dreamer4/waymo/logs/ooi50k_lat16_d256_ep200_4a100.log
ckpt: /p/yufeng/tri30/dreamer4/waymo/checkpoints/ooi50k_lat16_d256_ep200_4a100
encoder_variant: repeat_map
params: 10,648,249
```

Encoder B, running:

```text
run_name: ooi50k_lat16_d256_ep200_4a100_staticmap_v2
log: /p/yufeng/tri30/dreamer4/waymo/logs/ooi50k_lat16_d256_ep200_4a100_staticmap_v2.log
ckpt: /p/yufeng/tri30/dreamer4/waymo/checkpoints/ooi50k_lat16_d256_ep200_4a100_staticmap_v2
encoder_variant: static_map_query
map_depth: 2
map_cross_every: 1
map_query_tokens: latent_agent
params: 13,807,801
```

Controlled variables:

```text
dataset train/val split
4 GPUs
batch_size = 32 per GPU
epochs = 200
time_window = 32
d_model = 256
encoder depth = 4
decoder depth = 4
n_latents = 16
d_bottleneck = 32
Decoder 1 latent-only
loss weights
optimizer/lr/weight_decay/dropout/mlp_ratio defaults
no interaction scorer
no dynamics loss
```

### Metrics To Compare After Training

Use the metrics shared by both logs as the primary fair comparison:

```text
final val/loss_total
final val/loss_agent_xy
final val/loss_agent_vel
final val/loss_agent_yaw
final val/loss_agent_valid
final val/loss_light_state
final val/loss_light_valid
train steps/sec
parameter count
```

Encoder B additionally logs physical metrics:

```text
agent_xy_mae_m
agent_speed_mae_mps
agent_vxvy_mae_mps
agent_yaw_mae_deg
agent_valid_acc
light_state_acc
light_valid_acc
```

These should be used for human interpretation of Encoder B, but not as the only
direct comparison against Encoder A unless Encoder A is re-evaluated with the
new metric code.

### Current Early Signal

At the start of Encoder B training:

```text
epoch 1 val/loss_total: 0.5915
epoch 2 val/loss_total: 0.4503
epoch 5 val/loss_total: 0.2529
epoch 8 val/loss_total: 0.1915
steps/sec after warmup: roughly 4.1-4.9
```

This is not a final quality comparison, but it already shows that static-map
query is much faster than repeated-map v1 under the same batch/window setting.

### Decision Rules

If Encoder B reaches similar or better final validation loss than Encoder A:

```text
Conclusion:
  Static map memory is the better tokenizer encoder.
  It preserves reconstruction quality while reducing repeated static-token cost.

Next step:
  Use Encoder B as the default tokenizer architecture.
  Then implement Decoder 2: latent + static map memory, because z should not need
  to memorize static lane geometry.
```

If Encoder B is slightly worse on agent reconstruction but much faster:

```text
Conclusion:
  Static map memory is promising, but map conditioning needs tuning.

Next ablations:
  --map_query_tokens all
  --map_depth 0
  --map_depth 1
  --map_depth 4
  --map_cross_every 2
  n_heads = 8 if GPU memory allows

Also inspect:
  whether light losses improve faster than agent losses;
  whether agent_xy/yaw are the weak points;
  whether map cross-attention should update only agents or both agents/latents.
```

If Encoder B is clearly worse and not much faster:

```text
Conclusion:
  The simple dense static-map-query design is not enough.

Next step:
  Keep Encoder A as the tokenizer baseline.
  Improve Encoder B before using it downstream:
    add local/nearest map attention;
    add explicit agent-map relative geometry bias;
    add lane-id/light-lane association if available;
    test Decoder 2 before discarding static map memory entirely.
```

If Encoder B is much better on traffic lights but not agent motion:

```text
Conclusion:
  Static-map-query helps scene context, but agent-map geometry is not yet strong
  enough for motion reconstruction.

Next step:
  Add relative agent-map features or local map selection for cross-attention.
```

### Recommended Project Next Step After A/B

Do not add interaction scorer or dynamics until the tokenizer architecture choice
is made.

Recommended order:

```text
1. Finish Encoder B training.
2. Fill the comparison table against Encoder A.
3. Choose default encoder.
4. Implement Decoder 2 if Encoder B is competitive.
5. Only then move to dynamics training.
6. After tokenizer + dynamics are stable, return to OOI interaction scorer.
```
