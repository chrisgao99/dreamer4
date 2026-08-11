# World Model Motion Head, Latent Head, and Lightweight Semantic Reader Plan

Date: 2026-07-21

Status: proposed design; not yet implemented

## 1. Primary Goal

The purpose of this change is to **reduce the burden on the latent world model
of learning basic agent kinematics from scratch**.

The current latent dynamics model must implicitly learn all of the following
inside an opaque tokenizer latent space:

- position integration from velocity;
- acceleration and deceleration;
- yaw evolution;
- validity transitions;
- multi-agent interactions;
- stochastic scene evolution;
- production of a future latent that remains decodable by the frozen
  tokenizer.

The proposed design separates these responsibilities without replacing the
latent world model:

- a **motion head** predicts physically meaningful motion residuals;
- hard-coded kinematic integration converts those residuals into a structured
  next-agent state;
- a **latent head** continues to predict the next tokenizer latent;
- a frozen, lightweight semantic reader `P` checks that the predicted latent
  represents an agent state consistent with the motion branch.

The lightweight reader is an auxiliary training component. It is not a new
source of ground-truth input to the tokenizer decoder.

## 2. Explicit Non-Goals and Constraints

This proposal does **not**:

- feed ground-truth `q_next` into the tokenizer decoder;
- change tokenizer reconstruction into `decoder(z, q)`;
- make the tokenizer reconstruction task trivial by exposing the target agent
  features to the decoder;
- replace the latent world model with a purely linear or constant-velocity
  model;
- force the learned motion residual to zero;
- use ADE or FDE as training losses;
- require the full frozen tokenizer decoder to run at every rollout step.

The existing tokenizer latent `z` remains the main learned scene
representation. The structured state `q` is an auxiliary physical rollout
state and supervision path.

## 3. State and Notation

For each agent, define the structured physical state:

```text
q_t = {
  x, y,
  vx, vy,
  yaw,
  valid,
  static identity/type/size fields as needed
}
```

Define:

```text
z_t:
  frozen-tokenizer latent for the scene at time t

K(q_t, action_t):
  hard-coded kinematic baseline transition

R_theta(...):
  learned nonlinear residual produced by the motion head

q_struct_next:
  the complete next physical state obtained from K plus R_theta

P_phi(z_next):
  lightweight semantic readout of agent state from the predicted latent
```

The key distinction is:

```text
q_base_next   = K(q_t, action_t)

q_struct_next = Integrate(
                  q_t,
                  action_t,
                  R_theta(q_history, z_history, map, action_t)
                )
```

Consistency is imposed against `q_struct_next`, not against the purely
kinematic `q_base_next`.

## 4. Model Data Flow

At one rollout step:

```text
(z_history, q_history, map, action)
                  |
                  v
      shared context/interaction backbone
                  |
          +-------+-------+
          |               |
          v               v
      motion head      latent head
          |               |
          v               v
  learned motion       predicted
     residuals           z_next
          |
          v
  hard-coded kinematic
      integration
          |
          v
   q_struct_next

predicted z_next
          |
          v
 frozen lightweight P
          |
          v
     q_from_z_next

q_from_z_next  <---- consistency ---->  stopgrad(q_struct_next)
```

The two branches have separate ground-truth anchors:

```text
latent branch:
  z_next -> z_gt_next

motion branch:
  predicted motion residuals -> GT-derived motion targets

semantic alignment:
  P_frozen(z_next) -> stopgrad(q_struct_next)
```

The semantic alignment loss updates the latent dynamics branch. The motion
branch is trained by its own ground-truth motion losses rather than being
pulled toward an imperfect latent prediction.

## 5. Motion Head

The motion head should not independently predict every raw agent feature.
Static features are copied, and redundant physical features should be derived
instead of separately predicted.

Recommended outputs per agent:

```text
delta_vx, delta_vy
delta_yaw or yaw_rate
small integration correction dx, dy
validity logit
```

Recommended update:

```text
v_next = v_t + delta_v

yaw_next = wrap(yaw_t + delta_yaw)

p_next = p_t
       + 0.5 * (v_t + v_next) * dt
       + position_correction

speed_next = norm(v_next)
```

For a controlled focus agent, the commanded action may replace the predicted
focus transition. Other agents continue to use learned residuals and
kinematic integration.

The learned residual can represent nonlinear behavior such as braking,
turning, yielding, and interaction response. Therefore, even exact agreement
between the motion and latent branches does not imply a linear world model.

## 6. Latent Head

The latent head retains the original responsibility of predicting a future
tokenizer latent:

```text
latent_head(context, corrupted_future_latent, shortcut_condition)
  -> z_next
```

It remains trained with the appropriate latent ground-truth objective, such as
the existing clean-latent/flow or shortcut objective.

The motion branch does not numerically add physical deltas directly to opaque
latent dimensions. There is no assumed correspondence between a latent
dimension and a specific `x`, `y`, velocity, or yaw feature.

Instead, semantic coupling is provided by the frozen reader `P` and its
consistency loss.

## 7. Lightweight Semantic Reader P

`P` is, functionally, a simplified **agent-only decoder**. Its purpose is to
read the physical agent state represented by `z` much more cheaply than the
full tokenizer decoder.

### 7.1 Inputs and outputs

Preferred definition:

```text
P_phi(q_next | z_next)
```

Operationally, `P` consumes:

- `z_next`;
- 32 learned agent-slot queries;
- optional static slot/type/mask information.

It should not consume dynamic `q_t` values such as current position and
velocity in the default design. Otherwise it could ignore `z_next` and act as
another kinematic predictor.

It outputs only the physical features needed for semantic consistency:

```text
x, y
vx, vy
sin(yaw), cos(yaw)
validity logit
```

It does not need to reconstruct traffic lights, map features, the full token
layout, or other tokenizer outputs.

### 7.2 Recommended architecture

```text
z_next: [B, N_latent, D_z]
  -> linear projection to D=256
  -> latent memory

32 learned agent queries: [32, 256]
  -> agent-query self-attention
  -> agent-query-to-latent cross-attention
  -> feed-forward network
  -> repeat for 2 decoder blocks

agent hidden: [B, 32, 256]
  -> continuous-state head
  -> yaw head
  -> validity head
```

Where compatible, initialize or reuse the frozen tokenizer decoder's:

- agent query embeddings;
- slot embeddings;
- latent input projection;
- feature normalization conventions.

For semantic consistency, a deterministic reader with fixed feature scales is
preferred over an unconstrained heteroscedastic distribution. A learned
variance can otherwise weaken the constraint by inflating its scale.

### 7.3 Reader pretraining and freezing

Train `P` separately on frozen ground-truth tokenizer latents:

```text
GT scene
  -> frozen tokenizer encoder
  -> z_gt
  -> P_phi
  -> q_pred
```

Targets come from the corresponding ground-truth agent state. Train only `P`;
keep the tokenizer frozen.

Recommended reader losses are per-feature losses, not ADE/FDE:

```text
normalized Huber for x, y, vx, vy
circular loss for yaw
BCE for validity
```

After pretraining, freeze `P` before world-model training. Freezing prevents
the reader and world model from jointly inventing an incorrect but mutually
consistent semantic mapping.

## 8. Training Objectives and Gradient Routing

The proposed objective is:

```text
L_total = L_latent_gt
        + lambda_motion * L_motion_gt
        + lambda_cons * L_consistency
        + lambda_self * L_self_aux
```

### 8.1 Motion supervision

`L_motion_gt` supervises physically meaningful local quantities derived from
ground truth:

```text
delta velocity or acceleration
yaw rate / wrapped delta yaw
integration correction
validity transition
```

Use per-feature robust regression, circular yaw loss, and BCE. Do not use ADE
or FDE as losses.

### 8.2 Semantic consistency

```text
q_from_z_next = P_frozen(z_pred_next)

L_consistency = feature_distance(
                  q_from_z_next,
                  stopgrad(q_struct_next)
                )
```

Recommended gradient routing:

- `P` is frozen;
- `q_struct_next` is stop-gradient in this loss;
- `L_consistency` updates the latent dynamics branch;
- the motion head is updated by `L_motion_gt`;
- the latent head is also anchored by `L_latent_gt`.

Consistency is an alignment regularizer, not the dominant dynamics objective.
Its gradient contribution should remain modest relative to the direct latent
and motion ground-truth objectives.

### 8.3 Interpretation of zero consistency loss

A consistency loss of zero is not itself a failure. It means:

```text
the physical state read from z_next
  ==
the physical state produced by hard kinematics plus learned motion residual
```

The world model becomes a purely linear kinematic model only if the learned
residual `R_theta` collapses toward zero. Direct ground-truth supervision of
acceleration, yaw-rate, interaction correction, and validity is what prevents
that collapse.

## 9. Decoder Usage

The full tokenizer decoder is not required at every rollout step.

During the inner rollout loop:

```text
z_next -> lightweight frozen P -> semantic consistency
```

The complete frozen decoder may still be used:

- once after a full rollout for final evaluation or visualization;
- sparsely on a small number of sampled training frames if additional
  full-decoder alignment is later needed.

This sparse optional use is not part of the core proposal and should not be
confused with per-step decode/re-encode.

## 10. Expected Benefit

The intended division of labor is:

```text
hard kinematics:
  deterministic position/velocity/yaw integration structure

motion head:
  learned acceleration, turning, validity, and interaction residuals

latent head:
  stochastic future scene representation in tokenizer latent space

lightweight P:
  semantic agreement between the latent prediction and motion prediction
```

This design should reduce the amount of basic kinematic structure that the
latent world model must rediscover inside an opaque latent space, while still
requiring the latent dynamics to represent nonlinear interaction-aware future
states.

## 11. Main Limitation

`P` is an approximation to the physical-state portion of the full tokenizer
decoder. Therefore, consistency under `P` does not mathematically guarantee
that the complete decoder will produce exactly the same state.

The following anchors reduce that risk:

- keep direct `z_pred -> z_gt` latent supervision;
- initialize `P` from compatible full-decoder components where possible;
- train `P` on frozen tokenizer latents before freezing it;
- optionally distill the full decoder's physical outputs into `P`;
- reserve sparse full-decoder checks for later only if necessary.

The lightweight reader should be treated as a low-cost semantic regularizer,
not as a replacement for the full tokenizer decoder or as proof of exact
decoder equivalence.

## 12. First Submitted Experiment Configuration

The first implementation uses one shared `Dynamics` Transformer with
`space_mode="wm_agent"`. Explicit q tokens and packed z tokens therefore mix
inside the same backbone. The motion head is followed by hard integration, and
the resulting `q_next` is injected into the latent endpoint with agent-to-latent
cross-attention before `z_next` is returned.

The lightweight reader is pretrained first and frozen for world-model training:

- reader implementation: `waymo/training/world_model/motion_latent_v1.py`;
- reader training: `waymo/training/world_model/train_waymo_semantic_reader.py`;
- world-model training: `waymo/training/world_model/train_waymo_motion_latent_v1.py`;
- CUDA 2/3 sequential launcher:
  `waymo/training/world_model/launch_waymo_motion_latent_v1_cuda23_tmux.sh`.

The submitted world-model run is **1,000,000 transition optimizer steps**, not
300k. It uses global batch 8 (local batch 4 on each of two GPUs), always rolls
to frame 90, and detaches the predicted q/z state after every transition.

Context curriculum:

- 0--30k: ctx11 only;
- 30k--150k: linearly increase ctx1 probability from 0 to 0.8;
- 150k--1M: ctx1/ctx11 = 80/20.

Every sample retains direct d=1 ground-truth latent supervision. The auxiliary
two-half-step bootstrap runs on 50% of each local batch, with weight 0 through
20k, linearly ramped to 0.1 at 60k, then held at 0.1. Semantic consistency has
weight 0.1. Motion targets are computed relative to the current predicted q.
Neither ADE nor FDE is used as a training loss, and the full tokenizer decoder
is not called inside the rollout training loop.
