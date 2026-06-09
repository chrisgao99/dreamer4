# Waymo Vector Tokenizer Plan

Last updated: 2026-06-05

## Purpose

This note records the current plan for building a Dreamer-4-style tokenizer for Waymo vector data. It should be read together with `AGENT_CONTEXT.md`.

The project goal is to learn interaction-aware driving representations from Waymo motion data. The tokenizer should convert selected agents, cropped map context, and traffic lights into compact latent tokens that can be used by a Dreamer-4-style world model. Following the Dreamer 4 phase split, tokenizer pretraining should not include learned scene/task/policy-agent tokens. Those learned control or summary tokens should be inserted only when finetuning the dynamics model into a policy.

The key tokenizer outputs are:

- ordinary per-agent state tokens for selected-agent dynamics representation
- bottleneck latent tokens `z_t` for dynamics/world-model training

Current status as of 2026-05-06:

- Data filtering, NPZ loading, visualization, and the first encoder baseline are implemented under `/p/yufeng/tri30/dreamer4/waymo`.
- The implemented encoder baseline repeats encoded static map tokens over time, then runs Dreamer-style block-causal space/time layers.
- Decoder 1, split reconstruction losses, and the first Waymo tokenizer training script are implemented; static-map query encoder and dynamics training are not implemented yet.
- The immediate next goal is to train Encoder A + Decoder 1, then implement a small set of decoder/encoder variants and compare reconstruction quality, future prediction quality, representation quality, and memory/runtime.

## Reference Repos

The user cloned two reference repositories. They are currently located under:

- `/p/yufeng/tri30/renference_repos/MTR`
- `/p/yufeng/tri30/renference_repos/SMART`

Note: the folder is spelled `renference_repos`, not `reference_repos`.

Important MTR files to inspect/adapt:

- `/p/yufeng/tri30/renference_repos/MTR/mtr/models/utils/polyline_encoder.py`
- `/p/yufeng/tri30/renference_repos/MTR/mtr/models/context_encoder/mtr_encoder.py`
- `/p/yufeng/tri30/renference_repos/MTR/mtr/models/motion_decoder/mtr_decoder.py`
- `/p/yufeng/tri30/renference_repos/MTR/mtr/datasets/waymo/data_preprocess.py`

Important SMART files to inspect/adapt:

- `/p/yufeng/tri30/renference_repos/SMART/smart/datasets/preprocess.py`
- `/p/yufeng/tri30/renference_repos/SMART/smart/modules/agent_decoder.py`
- `/p/yufeng/tri30/renference_repos/SMART/smart/modules/map_decoder.py`
- `/p/yufeng/tri30/renference_repos/SMART/smart/modules/smart_decoder.py`
- `/p/yufeng/tri30/renference_repos/SMART/smart/layers/attention_layer.py`

Design stance:

- Use MTR as the main reference for map/polyline encoding, agent-map attention, and query-style decoding.
- Use SMART as a reference for vector tokenization, map trajectory tokens, and generative/autoregressive objectives.
- Do not directly import the full MTR or SMART training stack unless there is a concrete reason. The first implementation should be a thin adaptation that fits the existing Dreamer 4 code.

## Existing Dreamer 4 Code

Main repo:

- `/p/yufeng/tri30/dreamer4`

Relevant files:

- `/p/yufeng/tri30/dreamer4/dreamer4/model.py`
- `/p/yufeng/tri30/dreamer4/dreamer4/train_tokenizer.py`
- `/p/yufeng/tri30/dreamer4/dreamer4/train_dynamics.py`
- `/p/yufeng/tri30/dreamer4/dreamer4/waymo_filter.py`

The existing `model.py` already has the core Dreamer-4 pattern:

- token layout with modalities
- space attention inside each timestep
- causal time attention across timesteps
- bottleneck tokenizer latent tokens
- dynamics model with spatial/register tokens, with learned scene/task/policy-agent tokens reserved for policy finetuning

The vector tokenizer should reuse this block-causal time-space idea instead of becoming only an MTR-style static scene encoder.

## Input Data To Extract

Waymo motion tf.Example has 91 state steps:

- 10 past
- 1 current
- 80 future

Earlier conversation sometimes said 90 timesteps. Use all 91 states first; training code can later choose 90 transitions if needed.

Agent input:

- `x`
- `y`
- `speed`
- `velocity_x`
- `velocity_y`
- `valid`
- optional but useful: `bbox_yaw`, `type`, `is_sdc`, `id`

Agent selection:

- Select ego/SDC as slot 0 using `state/is_sdc`.
- Select the closest `K=16` or `K=32` agents.
- Closeness means: across the 91 timesteps, at least one valid timestep has distance to ego below a threshold.
- Rank by minimum valid distance to ego over time.
- Pad missing agent slots and return masks.
- Current implemented debug setting is `K=16`.
- Script default supports `K=32`.
- Current implemented agent distance threshold is `80m`.
- This `80m` threshold is an engineering hyperparameter, not copied exactly from MTR. MTR keeps all objects with valid past trajectories and uses masks rather than our fixed closest-agent filter.

Map input:

- `roadgraph_samples/xyz`
- `roadgraph_samples/dir`
- `roadgraph_samples/type`
- `roadgraph_samples/valid`
- `roadgraph_samples/id`

Map filtering:

- Do not input all 30000 Waymo roadgraph samples.
- Crop enough local area around the ego trajectory.
- First version can keep roadgraph samples within a radius of the ego trajectory, e.g. 80m or 100m.
- Better version groups samples by `roadgraph_samples/id` into polylines and keeps polylines with at least one point near the ego trajectory.
- Convert to fixed polyline tensor, e.g. `(M, P, F_map)` with masks.
- Current implemented map distance threshold is `100m`.
- Current implemented map cap is `M=256` polyline chunks.
- Current implemented maximum points per polyline chunk is `P=20`.
- `P=20` matches MTR's `NUM_POINTS_EACH_POLYLINE=20`.
- MTR uses up to `NUM_OF_SRC_POLYLINES=768` nearest polylines for Waymo. It does not use our exact `100m` map radius; it uses top-k nearest polyline selection around the center object.
- Our `M=256` is a lighter first baseline because the map tokens are currently repeated over time. After static-map encoding is implemented, compare `M=256`, `M=512`, and possibly `M=768`.

Map features:

- `x`
- `y`
- lane direction `dir_x`, `dir_y`
- lane type
- valid

Traffic light input:

- all timesteps light `x`
- all timesteps light `y`
- all timesteps light `state`
- valid if available
- optionally lane/control id if available

Expected tensor shape:

- `(T=91, L=16, F_light)`

Coordinate normalization:

- Normalize all positions into an ego-centric frame.
- Use ego current pose as origin.
- Rotate by ego current heading.
- Apply the same transform to agent trajectories, map polylines, and traffic lights.

Coordinate-frame decision:

- Use ego-centric geometry for the first tokenizer implementation.
- This matches the spirit of MTR, which transforms trajectories and map polylines into a center-object coordinate frame.
- For this project the center object is the ego/SDC, so all geometric fields should be expressed relative to the ego at the current timestep.
- Agent `x/y`: subtract ego current position, then rotate by negative ego heading.
- Agent `vx/vy`: rotate by negative ego heading, without subtracting position.
- Agent `yaw`: subtract ego heading and wrap to `[-pi, pi]`.
- Map lane/polyline `x/y`: subtract ego current position, then rotate by negative ego heading.
- Map lane/polyline `dir_x/dir_y`: rotate by negative ego heading, without subtracting position.
- Traffic light `x/y`: subtract ego current position, then rotate by negative ego heading.
- Semantic fields such as agent type, lane type, traffic-light state, ids, and valid masks are unchanged.
- Save `ego_origin_xy` and `ego_heading` so model predictions can later be transformed back to world coordinates for evaluation or visualization.

## Core Tensor Structure

After filtering, a batch should look like:

```text
agents: (B, T, K, F_agent)
map:    (B, M, P, F_map)
lights: (B, T, L, F_light)
```

Map is static. For the Dreamer-4 tokenizer, map polyline tokens can be repeated across time:

```text
map_tokens: (B, M, D) -> (B, T, M, D)
```

This is the implemented baseline. It intentionally keeps the first vector tokenizer close to the original Dreamer image-token structure. Original Dreamer image tokenizer uses `128x128` RGB images with `4x4` patches:

```text
image patches: (B, T, 1024, 48) -> projected image tokens: (B, T, 1024, 256)
```

So repeating `M=256` vector map tokens over time is acceptable as a first baseline because it is still fewer static tokens than Dreamer image patches.

The tokenizer encoder should produce:

```text
z:          (B, T, N_latents, D_bottleneck)
agent_repr: (B, T, K, D)
```

The main current architectural adaptation from Dreamer 4 is not a new attention block. The adaptation is:

- image patch tokens are replaced by structured Waymo vector tokens
- map tokens are encoded with an MTR-style PointNet polyline encoder
- traffic lights are explicit dynamic tokens
- no learned scene/task/policy-agent token is used during tokenizer pretraining
- decoder targets are driving states rather than image patches

Downstream policy training can later add learned scene/task/policy-agent tokens to the dynamics/policy transformer, matching Dreamer 4's agent finetuning phase rather than the causal tokenizer phase.

## Encoder Design

The tokenizer encoder should be Dreamer-4-style, with both space and causal time layers.

Initial tokenization:

```text
agent feature MLP:
  (x, y, speed, vx, vy, valid, type, yaw?) -> agent tokens

map polyline encoder:
  P lane points -> one map/polyline token

traffic light MLP:
  (x, y, state, valid) -> light tokens

latent tokens:
  learned latent queries repeated at every timestep
```

Per timestep token layout:

```text
[latent_1..latent_N, agent_1..agent_K, map_1..map_M, light_1..light_L]
```

Overall transformer input:

```text
tokens: (B, T, S, D)
```

Each encoder block should contain:

```text
space attention within timestep t
causal time attention across timesteps for persistent slots
MLP
```

Space attention purpose:

- model interactions at the same timestep
- let agents read other agents
- let agents read nearby map context
- let agents read relevant traffic light context
- let latent tokens aggregate information for bottleneck `z_t`

Time attention purpose:

- model evolution over time without future leakage
- agent slot `i` at time `t` attends to agent slot `i` at times `<= t`
- light slot `j` at time `t` attends to light slot `j` at times `<= t`
- latent slot `k` at time `t` attends to latent slot `k` at times `<= t`
- map time attention can be disabled because map is static; repeating map tokens through time is mainly to support per-timestep space attention

Causality rule:

```text
z_t = encoder(x_<=t, map, lights_<=t)
```

Do not let tokenizer latents at time `t` see future agent/light states. Otherwise the dynamics model will learn from leaked future information.

Time-window decision:

- The full filtered scenario has 91 state timesteps, but the model should not require all 91 as a fixed input window.
- The block-causal transformer can handle different sequence lengths as long as the implementation uses runtime positional encodings or otherwise supports variable `T`.
- The existing Dreamer 4 code uses sinusoidal time positions generated from the current input length, so variable-length windows are compatible in principle.
- Batches still need a common `T`; use fixed-length sampled windows inside a batch and allow different runs/batches to use different `T`.
- Do not make the main training window only 10 timesteps. Waymo observed history is 10 past plus 1 current, i.e. 11 observed states, and many driving interactions such as merges, yielding, and lane changes unfold over 2-3 seconds.
- Recommended first training window: `T=32` states, roughly 3.2 seconds at 10 Hz, if memory allows.
- Fallback/lightweight window: `T=16` states, roughly 1.6 seconds.
- Also include `T=11` or random prefix-truncated batches during training so inference with only the initial observed history is not out of distribution.
- For evaluation/test-time forecasting, feed the available observed prefix, usually 11 states, then roll the dynamics model forward autoregressively.

Traffic light rule:

- Agent tokens should query traffic light tokens directly.
- Do not rely on an aggregate token as the only bridge from lights to agents.
- Otherwise per-agent tokens may not know whether a relevant light is red or green.

Map rule:

- Agent tokens should query map tokens directly.
- Latent tokens should have access to agent, map, and light context.

### Encoder Variants To Implement And Compare

The first set of comparisons should keep the Dreamer-style block-causal transformer as the backbone and vary how static map context enters the encoder.

#### Encoder A: repeat map input over time

Status:

- Implemented in `/p/yufeng/tri30/dreamer4/waymo/vector_tokenizer_encoder.py`.

Pipeline:

```text
agents: (B,T,K,8) -> AgentFeatureEncoder -> (B,T,K,D)
map:    (B,M,P,6) -> MTR-style MapFeatureEncoder -> (B,M,D)
map repeated over time: (B,M,D) -> (B,T,M,D)
lights: (B,T,L,4) -> TrafficLightEncoder -> (B,T,L,D)

tokens per timestep:
[latent_1..N, agent_1..K, map_1..M, light_1..L]

Block-causal transformer:
space attention within timestep
causal time attention across same token slot

latent tokens -> Linear(D -> d_bottleneck) + tanh -> z
```

Why this variant:

- It is the simplest adaptation of Dreamer 4: every timestep has a spatial token set.
- It keeps map visible to agents and latent tokens through regular space attention.
- It is a baseline, not the final intended static-map design.
- It is acceptable as a first pass because `M=256` map tokens is smaller than Dreamer image input with `1024` image patch tokens per timestep.

Expected weakness:

- Map is static but repeated for every timestep, so memory/runtime scales with `T*M`.
- Time attention over map slots is unnecessary unless disabled/masked.
- It mixes static geometry and dynamic state in one temporal token grid.

#### Encoder B: static map query during encoding

Status:

- Implemented on 2026-06-04 in `/p/yufeng/tri30/dreamer4/waymo/vector_tokenizer_encoder.py`.
- Select with `--encoder_variant static_map_query` in `/p/yufeng/tri30/dreamer4/waymo/train_waymo_vector_tokenizer.py`.
- This variant is intended for a clean Encoder A vs Encoder B tokenizer comparison only. Do not add interaction scorer or dynamics losses to this experiment.

Pipeline:

```text
agents/lights every timestep:
  agent tokens: (B,T,K,D)
  light tokens: (B,T,L,D)
  latent tokens: (B,T,N,D)

static map memory:
  map_polylines: (B,M,P,6) -> MapFeatureEncoder -> (B,M,D)

dynamic tokens run Dreamer-style space/time layers
map enters through cross-attention:
  latent and agent tokens query static map memory by default
```

Implemented Encoder B block:

```text
1. dynamic space self-attention over [latents, agents, lights]
2. static map cross-attention from selected dynamic tokens to map memory
3. causal time self-attention over dynamic slots only
4. MLP
```

Current configurable choices:

```text
--map_depth        default 2, static map self-attention refinement layers
--map_cross_every  default 1, add cross-attention every encoder block
--map_query_tokens default latent_agent, choices latent/agent/latent_agent/all
```

Use dense map cross-attention first; later compare nearest/local map attention if needed.

Why this variant:

- Map is static, so it should not need a repeated time dimension.
- It is more memory efficient: map memory scales with `B*M*D`, not `B*T*M*D`.
- It creates a cleaner separation between dynamic tokens and static context.
- It is closer to how map often acts in motion forecasting models: static memory queried by dynamic entities.

Expected weakness:

- It requires modifying the Dreamer block structure to add cross-attention.
- If cross-attention is too weak or too sparse, agents/latents may not receive enough lane context.

#### Encoder comparison metrics

The immediate comparison must keep decoder, bottleneck, losses, dataset, and training schedule fixed. Compare:

```text
Encoder A + Decoder 1: repeated map input over time
Encoder B + Decoder 1: static map query during encoding
```

Core normalized validation losses:

```text
val/loss_total
val/loss_agent_xy
val/loss_agent_vel
val/loss_agent_yaw
val/loss_agent_valid
val/loss_light_state
val/loss_light_valid
```

Physical-unit validation metrics now logged by the trainer:

```text
val/agent_xy_mae_m
val/agent_speed_mae_mps
val/agent_vxvy_mae_mps
val/agent_yaw_mae_deg
val/agent_valid_acc
val/light_state_acc
val/light_valid_acc
```

Efficiency/runtime metrics to record from training logs or `torch.cuda` profiling:

```text
learnable parameter count
train steps/sec
peak GPU memory
dynamic tokens per timestep
static map tokens per scene
estimated attention score count
```

Main fair-comparison hyperparameters:

```text
K = 32
L = 16
M = 256 first, then M = 512 as a static-map stress test
P = 20
T = 32
d_model = 256
n_heads = 4, matching current v1 run
encoder depth = 4
decoder depth = 4
n_latents = 16
d_bottleneck = 32
z scalars per timestep = 512
```

Token-count comparison for the main setting:

```text
Encoder A repeated map:
  per-timestep tokens = 16 latents + 32 agents + 256 map + 16 lights = 320

Encoder B static map query:
  per-timestep dynamic tokens = 16 latents + 32 agents + 16 lights = 64
  static map memory = 256 tokens per scene
```

Decision rules after the 200-epoch A/B run:

```text
If Encoder B matches or improves Encoder A final validation loss:
  choose Encoder B as the default tokenizer encoder;
  proceed to Decoder 2, where decoder queries can condition on static map memory.

If Encoder B is slightly worse but much faster:
  keep Encoder B as promising and run ablations:
    map_query_tokens = all
    map_depth = 0, 1, 4
    map_cross_every = 2
    n_heads = 8 if memory allows

If Encoder B is clearly worse and not faster:
  keep Encoder A as the baseline;
  redesign static-map query with local/nearest map attention or explicit
  agent-map relative geometry before using it downstream.

If Encoder B mainly improves traffic-light losses but not agent motion:
  add stronger agent-map geometry features or local map attention.
```

Important sequencing rule:

```text
Do not add interaction scorer or dynamics to this comparison.
First choose the tokenizer encoder.
Then compare Decoder 1 vs Decoder 2.
Only after tokenizer architecture is stable, move to dynamics.
Only after tokenizer+dynamics are stable, add OOI interaction scoring.
```

Initial implementation order:

1. Keep Encoder A as the working baseline.
2. Implement decoder and tokenizer training on Encoder A.
3. Implement Encoder B with static map memory and cross-attention. Done 2026-06-04.
4. Compare Encoder A vs Encoder B under the same Decoder 1/loss setup.

## Decoder Design

The decoder should also use space-time layers. The decoder itself is not meant to be the main novelty; it is the bottleneck test that tells whether `z` contains enough driving-state information.

Input:

```text
z: (B, T, N_latents, D_bottleneck)
```

Create decoder query tokens:

```text
agent queries: (B, T, K, D)
light queries: (B, T, L, D)
optional map queries: (B, T, M, D)
```

Decoder token layout:

```text
[latent_1..latent_N, agent_queries, light_queries, optional_map_queries]
```

Each decoder block:

```text
space attention within timestep
causal time attention across persistent query slots
MLP
```

Decoder heads:

- `agent_head`: reconstruct/predict `x, y, speed, vx, vy, valid`, optionally yaw
- `light_head`: predict light state classification, valid, optionally x/y
- optional `map_head`: reconstruct only the cropped input map polylines

Important decoder target decision:

- Do not decode the whole global Waymo map.
- The model cannot reconstruct map geometry it never receives.
- If map reconstruction is used, reconstruct only the cropped local map polylines that were input to the encoder.
- Map reconstruction should be auxiliary, not the main objective.

### Decoder Variants To Implement And Compare

#### Decoder 1: latent-only decoder

Status:

- Implemented in `/p/yufeng/tri30/dreamer4/waymo/vector_tokenizer_decoder.py`.

Input:

```text
z only
```

Decoder tokens:

```text
[latent_1..N, agent_queries, light_queries, optional map_queries]
```

Why this variant:

- It is the cleanest autoencoder bottleneck.
- No encoder agent/map/light tokens can bypass `z`.
- If reconstruction works, it means `z` captures the necessary dynamic scene information.

Expected weakness:

- It may force `z` to memorize static map geometry, which may not be desirable.
- It may be too hard for early training if `z` capacity is small.

Use this as the first strict baseline.

#### Decoder 2: latent + static map encoding

Status:

- Not implemented yet.

Input:

```text
z + static map memory
```

Decoder behavior:

- Decode agent and traffic-light states from `z`.
- Allow decoder queries to cross-attend to encoded static map memory.
- Do not require `z` to memorize map geometry.

Why this variant:

- In driving, map is static context. It may be better to condition on map rather than compress map into `z`.
- This is likely the most useful practical decoder if latent-only reconstruction is weak.
- It aligns with the future Encoder B design where static map memory is separate.

Expected weakness:

- It weakens the strict bottleneck because decoder sees map context outside `z`.
- Reconstruction performance must be interpreted as "dynamic state compression conditioned on map," not "full scene compression."

#### Decoder 3: latent + a few past agent states

Status:

- Not implemented yet.

Input:

```text
z + current/past observed agent states
```

Possible observed inputs:

- current state only: `t`
- short observed prefix: last `2-5` timesteps
- Waymo observed history: past 10 + current 1, i.e. 11 states

Why this variant:

- It can stabilize prediction if latent-only decoding is too difficult.
- It resembles forecasting settings where the model always has observed history and predicts future.

Expected weakness:

- It introduces a strong shortcut around `z`.
- It is less suitable for measuring tokenizer compression.
- Use only as an ablation or forecasting-oriented decoder, not the main tokenizer baseline.

#### Decoder capacity ablations

Use this as the main first baseline:

```text
N_latents: 8
d_bottleneck: 32
z scalars per timestep: 256
```

If all decoder choices are weak, compare bottleneck capacity before changing the whole design:

```text
N_latents: 8 -> 16
d_bottleneck: 32 -> 64
D model size: 128/256
decoder depth: 3/6/8
```

The first diagnosis should be whether failures come from decoder conditioning, insufficient `z` capacity, or loss balancing.

## Training Targets

Tokenizer pretraining:

- masked reconstruction of selected agent states
- masked reconstruction/classification of traffic light states
- optional masked reconstruction of cropped map polylines

Tokenizer reconstruction targets:

```text
agent continuous state:
  x, y, speed, vx, vy, optional yaw

agent validity:
  valid / invalid at each timestep

traffic-light state:
  classification over Waymo light states

traffic-light validity:
  valid / invalid for each light slot

optional map reconstruction:
  cropped input map only, not global Waymo map
```

Recommended first losses:

```text
agent xy/v/speed loss: masked L1 or SmoothL1
agent yaw loss: sin/cos or wrapped angle loss
agent valid loss: BCE
traffic-light state loss: cross entropy, masked by light validity
traffic-light valid loss: BCE
optional map point loss: masked L1/SmoothL1 on cropped map points
optional map type loss: cross entropy
```

Loss priority:

1. selected-agent future/current reconstruction
2. traffic-light state
3. optional cropped-map reconstruction as auxiliary only

Do not let map reconstruction dominate, because map is static context rather than the main driving dynamics target.

World-model / Dreamer dynamics training:

- predict future tokenizer latents:

```text
z_t -> z_{t+1} or z_{t+k}
```

- decode predicted future latents to selected-agent future states
- decode predicted future latents to traffic light states

Primary supervision should emphasize driving-relevant future state:

- selected agents' future trajectories and valid masks
- traffic light state sequence

Static map reconstruction is optional because map is mainly context.

### Tokenizer vs Full Dreamer Dynamics

The training should be staged:

1. Train tokenizer first.
2. Use tokenizer encoder to produce meaningful `z_t`.
3. Train Dreamer-style dynamics/shortcut model in latent space.
4. Decode predicted future latents back to agents/lights for evaluation or auxiliary supervision.

Why tokenizer first:

- The tokenizer defines the latent space. Dynamics training is not meaningful until `z_t` reconstructs driving state reasonably.
- Tokenizer training is cheaper and easier to debug than full rollout training.
- Reconstruction metrics reveal whether the vector tokenizer bottleneck is working before adding dynamics complexity.

At dynamics training time:

```text
observed/prefix scene -> tokenizer -> z_t
dynamics/shortcut model -> predicted future z
decoder -> future agents/lights
```

At test/forecast time:

```text
input observed Waymo history, usually 11 states
roll dynamics forward for future states
decode predicted future agents/lights
```

### Representation And Contrastive Learning Plan

The tokenizer representation targets are bottleneck latents `z` and ordinary selected-agent tokens. Contrastive learning should be added only after the reconstruction/dynamics baselines are stable.

Possible contrastive targets:

- pooled latent tokens `z`
- ego-agent token over time
- attention-pooled agent tokens
- concatenation or projection of pooled `z` + pooled agent tokens

Downstream dynamics-to-policy finetuning can later insert learned scene/task/policy-agent tokens into the dynamics or policy transformer. Those tokens should not be part of tokenizer reconstruction training.

Potential positive pairs:

- two time crops from the same scenario
- interaction-preserving augmentations of the same scenario
- scenes from the same filtered maneuver category, e.g. merge, lane change, yielding, unprotected left turn
- future/past views of the same interaction if no leakage is introduced into causal evaluation

Potential evaluation:

- retrieval: nearest neighbors should show similar interaction dynamics
- linear probing: latent/selected-agent/downstream-policy token predicts maneuver label
- clustering by interaction type
- compare representation distance with future trajectory/dynamics similarity
- compare contrastive refinement against world-model-only tokens

## Implementation Order

### Completed

Implemented under `/p/yufeng/tri30/dreamer4/waymo`:

- `waymo_vector_filter.py`
- `waymo_vector_dataset.py`
- `inspect_waymo_vector_filter.py`
- `visualize_waymo_vector_npz.py`
- `vector_tokenizer_encoder.py`
- `vector_tokenizer_decoder.py`

Completed behavior:

```text
agents:        (K, 91, 8)
agent_mask:    (K,)
map_polylines: (M, P, 6)
map_mask:      (M, P)
lights:        (91, 16, 4)
light_mask:    (91, 16)
```

The first encoder baseline supports:

```text
agents: (B,K,T,8) or (B,T,K,8)
map:    (B,M,P,6)
lights: (B,T,L,4)

output z: (B,T,N_latents,D_bottleneck)
agent_tokens: (B,T,K,D)
map_tokens: (B,T,M,D)
light_tokens: (B,T,L,D)
```

Smoke-tested windows:

```text
T=32
T=11
```

### Next Implementation Steps

1. Train Encoder A + Decoder 1 with the implemented tokenizer reconstruction training script.

```text
/p/yufeng/tri30/dreamer4/waymo/train_waymo_vector_tokenizer.py
```

Use the implemented `VectorTokenizer` wrapper and split reconstruction losses:

```text
agent xy SmoothL1
agent speed/vx/vy SmoothL1
agent yaw sin/cos SmoothL1
agent valid BCE
traffic-light state CE
traffic-light valid BCE
```

Optional map reconstruction remains a later auxiliary head.

2. Train Encoder A + Decoder 1 on the existing small debug set, then on more Waymo shards.

3. Add Decoder 2: latent + static map memory.

4. Compare Decoder 1 vs Decoder 2 with the same Encoder A.

5. Implement Encoder B: static map query during encoding.

6. Compare:

```text
Encoder A + Decoder 1
Encoder A + Decoder 2
Encoder B + Decoder 1
Encoder B + Decoder 2
```

7. After tokenizer works, adapt Dreamer 4 dynamics/shortcut model to consume vector latents and expose:

- selected-agent state tokens
- predicted future latents
- decoded future agent/light states
- downstream scene/task/policy-agent tokens inserted only during dynamics-to-policy finetuning

8. Add representation evaluation and optional contrastive training.

## Open Questions

- What exact crop radius should be used for map and agent selection? Current defaults are `80m` for agent selection and `100m` for map crop, but these should be treated as tunable hyperparameters.
- Should map crop use radius filtering, MTR-style top-k nearest polylines, or both as an ablation?
- Should `M=256` be increased to `512` or MTR-style `768` after static-map encoding reduces memory cost?
- Should agent selection rank only by current distance, minimum distance over 91 states, or minimum distance over the observed history? Current plan: minimum valid distance over all 91 for tokenizer pretraining, but this may leak future selection. For causal deployment, select from observed/history only.
- Should the tokenizer pretrain on full 91 states or only observed history plus future prediction? Current plan: build tensors for 91 states, enforce causal attention, and choose losses by experiment.
- How to best attach traffic lights to lanes: use Waymo lane/control ids if available; otherwise use nearest lane/polyline.
- Whether to implement sparse/local attention masks immediately or start with dense attention and masks for simplicity.
- Whether strict latent-only decoding is too hard. If so, compare latent + static map memory and latent + few past agent states.
- Which downstream scene/task/policy-agent-token design best supports policy, reward, value, probing, or retrieval after tokenizer and dynamics training are stable.

## Current Recommended Milestones

### Milestone 1: data/filtering layer

Status:

- Completed for small debug data.

Success criterion:

- one Waymo TFRecord can be converted into tensors:

```text
agents:      (K, 91, F_agent)
agent_mask:  (K,)
map:         (M, P, F_map)
map_mask:    (M, P)
lights:      (91, 16, F_light)
light_mask:  (91, 16)
```

- ego is always slot 0
- selected agents are within threshold by the chosen distance rule
- map is local and padded/capped deterministically
- traffic lights include full time sequence

### Milestone 2: encoder baseline

Status:

- Implemented and smoke-tested.

Success criterion:

- Encoder A runs on `T=32` training-style windows.
- Encoder A runs on `T=11` observed-prefix windows.
- Outputs are finite:

```text
z: (B,T,N_latents,D_bottleneck)
agent_tokens: (B,T,K,D)
map_tokens: (B,T,M,D)
light_tokens: (B,T,L,D)
```

### Milestone 3: tokenizer autoencoder

Status: implemented first training script; needs real training runs on more data.

Success criterion:

- Decoder reconstructs masked selected-agent states.
- Decoder predicts traffic-light state.
- Optional map reconstruction is implemented as auxiliary only.
- Compare Decoder 1 vs Decoder 2.

### Milestone 4: encoder map comparison

Status:

- Encoder B static-map-query code implemented and smoke-tested on 2026-06-04.

Success criterion:

- Encoder A: repeated map over time.
- Encoder B: static map query with cross-attention.
- Compare reconstruction, physical-unit metrics, memory, and runtime.
- Do not include interaction scorer or dynamics in this comparison.

### Milestone 5: dynamics and representation

Success criterion:

- Train latent dynamics/shortcut model using tokenizer latents.
- Decode future latent rollouts to future agents/lights.
- Insert downstream scene/task/policy-agent tokens only during dynamics-to-policy finetuning.
- Evaluate latent/selected-agent/downstream-policy tokens using probing, retrieval, policy performance, and optional contrastive learning.
