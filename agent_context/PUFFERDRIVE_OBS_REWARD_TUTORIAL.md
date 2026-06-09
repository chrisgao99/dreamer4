# PufferDrive Observation and Reward Tutorial

Last updated: 2026-05-04

This report explains the current single-agent observation vector in PufferDrive, the reward logic returned by `env.step()`, and the practical ways to customize either one.

Primary source files:

- `/p/yufeng/tri30/PufferDrive/pufferlib/ocean/drive/drive.py`
- `/p/yufeng/tri30/PufferDrive/pufferlib/ocean/drive/drive.h`
- `/p/yufeng/tri30/PufferDrive/pufferlib/config/ocean/drive.ini`

## 1. Observation Shape

For the default `dynamics_model="classic"`, one controlled agent observes a fixed vector of length `1121`.

The Python wrapper computes this in `drive.py`:

```python
self.num_obs = (
    self.ego_features
    + self.max_partner_objects * self.partner_features
    + self.max_road_objects * self.road_features
)
```

The constants come from `drive.h`:

```c
#define MAX_ROAD_SEGMENT_OBSERVATIONS 128
#define MAX_AGENTS 32
#define PARTNER_FEATURES 7
#define ROAD_FEATURES 7
#define EGO_FEATURES_CLASSIC 8
#define EGO_FEATURES_JERK 11
```

So for classic dynamics:

```text
ego features:       8
partner features:  31 * 7  = 217
road features:    128 * 7  = 896
total:                       1121
```

For jerk dynamics:

```text
ego features:      11
partner features: 217
road features:    896
total:                       1124
```

Important distinction:

```text
num_agents controls batch rows.
The 1121 feature dimension is per controlled agent.
```

For example, 10 controlled agents gives:

```text
obs.shape == (10, 1121)
```

## 2. Observation Layout

For classic dynamics, the vector layout is:

```text
obs[0:8]       ego block
obs[8:225]     partner block: 31 slots * 7 features
obs[225:1121]  road/map block: 128 slots * 7 features
```

For jerk dynamics, the first block is 11, so the partner and road starts shift:

```text
obs[0:11]      ego block
obs[11:228]    partner block
obs[228:1124]  road/map block
```

## 3. Ego Features

In `compute_observations()` in `drive.h`, the classic ego block is:

```text
0  relative goal x in ego frame, scaled by 0.005
1  relative goal y in ego frame, scaled by 0.005
2  signed speed / MAX_SPEED
3  ego width / MAX_VEH_WIDTH
4  ego length / MAX_VEH_LEN
5  collision flag, 1 if collision_state > 0 else 0
6  respawn flag, 1 if respawn_timestep != -1 else 0
7  entity type / 3.0
```

The coordinate frame is ego-centric:

- subtract ego position
- rotate by ego heading
- scale relative positions by `0.005` for goals

For jerk dynamics, ego has 11 features:

```text
0  relative goal x
1  relative goal y
2  signed speed
3  width
4  length
5  collision flag
6  steering angle / pi
7  longitudinal acceleration, normalized asymmetrically
8  lateral acceleration / max lateral acceleration
9  respawn flag
10 entity type / 3.0
```

## 4. Partner Agent Features

The partner block has up to `MAX_AGENTS - 1 = 31` slots. Each slot has 7 features:

```text
0  relative x in ego frame, scaled by 0.02
1  relative y in ego frame, scaled by 0.02
2  other width / MAX_VEH_WIDTH
3  other length / MAX_VEH_LEN
4  cos(relative heading)
5  sin(relative heading)
6  signed speed / MAX_SPEED
```

Partners are taken from active controlled agents first, then static agents. The ego itself is skipped. Invalid/respawning agents are skipped. Partners farther than 50m are skipped because the code filters on squared distance:

```c
if (dist > 2500.0f)
    continue;
```

Unused partner slots are zero padded.

## 5. Road and Map Features

The road/map block has up to `128` segment slots. Each slot has 7 features:

```text
0  segment midpoint relative x in ego frame, scaled by 0.02
1  segment midpoint relative y in ego frame, scaled by 0.02
2  segment half-length / MAX_ROAD_SEGMENT_LENGTH
3  width / MAX_ROAD_SCALE
4  cos(relative segment direction)
5  sin(relative segment direction)
6  entity type - 4.0
```

The observed road segments come from a grid-neighbor cache around the ego vehicle:

```c
int grid_idx = getGridIndex(env, ego_entity->x, ego_entity->y);
int list_size = get_neighbor_cache_entities(
    env, grid_idx, entity_list, MAX_ROAD_SEGMENT_OBSERVATIONS
);
```

The road type value is shifted by `-4.0`. The effective encoding is:

```text
0  road lane
1  road line
2  road edge
3  stop sign
4  crosswalk
5  speed bump
6  driveway
```

Unused road slots are zero padded.

## 6. How To Reduce or Customize Observations

### Option A: Slice in Python

This is the safest way if you only need a custom representation for data collection or model input.

Example:

```python
def classic_obs_blocks(obs):
    ego = obs[:, :8]
    partners = obs[:, 8:225].reshape(obs.shape[0], 31, 7)
    roads = obs[:, 225:].reshape(obs.shape[0], 128, 7)
    return ego, partners, roads

ego, partners, roads = classic_obs_blocks(obs)

custom_obs = np.concatenate(
    [
        ego,
        partners[:, :8].reshape(obs.shape[0], -1),
        roads[:, :32].reshape(obs.shape[0], -1),
    ],
    axis=1,
)
```

This does not change `env.observation_space`; it only changes what your downstream model sees.

Use this first unless you need the simulator itself to emit a smaller buffer.

### Option B: Add a Python wrapper

You can create a small wrapper that calls `env.step()`, slices `obs`, and exposes a custom observation space to your policy. This avoids changing C code but keeps the training interface clean.

Good use cases:

- using only ego + closest K partners
- removing road features entirely
- replacing raw slots with learned or hand-built features
- testing a tokenizer input format before changing the simulator

### Option C: Change C constants and rebuild

If you want `Drive` itself to return a smaller observation vector, edit `drive.h`.

Examples:

- Reduce road slots:

```c
#define MAX_ROAD_SEGMENT_OBSERVATIONS 64
```

- Reduce simulation/partner capacity:

```c
#define MAX_AGENTS 16
```

Be careful: `MAX_AGENTS` affects more than the observation. It changes the maximum number of actors per scene inside the simulator. If you only want fewer observed partners, it would be cleaner to add a separate constant such as:

```c
#define MAX_PARTNER_OBSERVATIONS 16
```

Then update:

```c
int max_obs = ego_dim
    + PARTNER_FEATURES * MAX_PARTNER_OBSERVATIONS
    + ROAD_FEATURES * MAX_ROAD_SEGMENT_OBSERVATIONS;
```

and update the partner loop/padding logic in `compute_observations()`.

### Option D: Rewrite `compute_observations()`

For a truly custom observation, edit `compute_observations()` directly. You must keep the C layout and the Python `self.num_obs` calculation consistent.

Things to update together:

1. `drive.h` constants and `max_obs`
2. all writes into `obs[...]` inside `compute_observations()`
3. zero-padding logic
4. `drive.py` observation-space calculation
5. any model code that assumes the old 1121 layout

After changing `drive.h`, rebuild the native extension from the PufferDrive repo root:

```bash
python setup.py build_ext --inplace --force
```

Use the same conda/virtual environment that runs your script.

## 7. Current Reward Logic

The reward is computed in `c_step()` in `drive.h`.

At each environment step:

1. rewards, terminals, and truncations are reset to zero
2. timestep increments
3. expert/static agents are replayed
4. active agents apply actions through dynamics
5. a small smoothness penalty is added for classic dynamics
6. collision/offroad metrics are computed
7. collision/offroad penalties are added
8. goal reward is added when the agent reaches its goal
9. terminal/truncation flags are set
10. observations are recomputed

### Reward terms

For classic dynamics, every controlled agent receives a tiny jerk/smoothness penalty:

```c
float jerk_penalty =
    -0.0002f * sqrtf(delta_vx * delta_vx + delta_vy * delta_vy) / env->dt;
env->rewards[i] += jerk_penalty;
```

Collision penalty:

```c
env->rewards[i] += env->reward_vehicle_collision;
```

Offroad penalty:

```c
env->rewards[i] += env->reward_offroad_collision;
```

Goal reward:

```c
env->rewards[i] += env->reward_goal;
```

or, after respawn:

```c
env->rewards[i] += env->reward_goal_post_respawn;
```

In the default Python constructor, these are:

```python
reward_vehicle_collision = -0.1
reward_offroad_collision = -0.1
reward_goal = 1.0
reward_goal_post_respawn = 0.5
goal_radius = 2.0
goal_speed = 20.0
```

In `drive.ini`, the training config currently contains:

```ini
reward_vehicle_collision = -0.5
reward_offroad_collision = -0.5
reward_goal = 1.0
reward_goal_post_respawn = 0.25
goal_radius = 2.0
goal_speed = 100.0
```

When using `Drive(...)` directly in Python, the constructor arguments are passed into the C binding and override ini values. When training through PufferLib config, the ini/config values are typically passed through as constructor arguments.

### Collision and offroad detection

`compute_agent_metrics()` sets `agent->collision_state`.

Vehicle collision:

- skipped for pedestrians
- checks nearby active/static agents
- uses bounding box collision
- sets `VEHICLE_COLLISION`

Offroad:

- checks road-edge intersections with the agent bounding box
- skipped for pedestrians
- sets `OFFROAD`

These metrics also update logs:

```text
collision_rate
collisions_per_agent
offroad_rate
offroad_per_agent
lane_alignment_rate
```

Lane alignment is currently logged, not directly rewarded.

### Goal reward condition

An agent reaches its goal when:

```text
distance_to_goal < goal_radius
current_speed <= goal_speed
current_goal_reached == false
```

`goal_behavior` changes what happens next:

```text
0 GOAL_RESPAWN        mark terminal, respawn after reaching goal
1 GOAL_GENERATE_NEW  sample a new goal and continue
2 GOAL_STOP          stop the agent at the goal
```

One subtle detail: in the "stop at goal" branch, the code uses assignment rather than addition:

```c
env->rewards[i] = env->reward_goal;
env->logs[i].episode_return = env->reward_goal;
```

So if a collision/offroad penalty and goal reward happen in the same step, this branch can overwrite earlier reward terms. If you want all terms to accumulate consistently, change those to `+=`.

## 8. How To Modify Rewards Without Rebuilding

You can change existing scalar reward parameters in Python:

```python
env = Drive(
    reward_vehicle_collision=-1.0,
    reward_offroad_collision=-1.0,
    reward_goal=2.0,
    reward_goal_post_respawn=0.5,
    goal_radius=2.0,
    goal_speed=10.0,
    collision_behavior=1,
    offroad_behavior=1,
    goal_behavior=2,
)
```

Useful knobs:

```text
reward_vehicle_collision  penalty for vehicle collision
reward_offroad_collision  penalty for crossing road edges/offroad
reward_goal               reward for reaching the goal
reward_goal_post_respawn  reward for reaching goal after respawn
goal_radius               distance threshold for success
goal_speed                speed threshold for success
collision_behavior        0 ignore, 1 stop, 2 remove
offroad_behavior          0 ignore, 1 stop, 2 remove
goal_behavior             0 respawn, 1 generate_new_goals, 2 stop
termination_mode          0 time limit, 1 terminate after all original agents reset
```

This path is enough if you only want to tune weights or thresholds.

## 9. How To Add New Reward Terms

Structural reward changes require editing `c_step()` in `drive.h` and rebuilding.

### Example: add lane-alignment reward

Currently lane alignment is computed and logged:

```c
int lane_aligned = env->entities[agent_idx].metrics_array[LANE_ALIGNED_IDX];
env->logs[i].lane_alignment_rate = lane_aligned;
```

To reward lane alignment:

```c
float lane_reward = lane_aligned ? 0.01f : -0.01f;
env->rewards[i] += lane_reward;
env->logs[i].episode_return += lane_reward;
```

Put this after `compute_agent_metrics()` and before terminal/truncation handling.

### Example: add dense progress reward

A dense progress reward needs the previous distance to goal before dynamics and the new distance after dynamics.

Sketch:

```c
float prev_dist = relative_distance_2d(
    prev_x, prev_y,
    env->entities[agent_idx].goal_position_x,
    env->entities[agent_idx].goal_position_y
);

float new_dist = relative_distance_2d(
    env->entities[agent_idx].x,
    env->entities[agent_idx].y,
    env->entities[agent_idx].goal_position_x,
    env->entities[agent_idx].goal_position_y
);

float progress_reward = 0.01f * (prev_dist - new_dist);
env->rewards[i] += progress_reward;
env->logs[i].episode_return += progress_reward;
```

To implement this cleanly, store `prev_x` and `prev_y` before `move_dynamics()`. The current code already stores previous velocity for the smoothness penalty, so this is the natural place.

### Example: make goal reward additive

To avoid overwriting penalties when reaching a goal, change:

```c
env->rewards[i] = env->reward_goal;
env->logs[i].episode_return = env->reward_goal;
```

to:

```c
env->rewards[i] += env->reward_goal;
env->logs[i].episode_return += env->reward_goal;
```

This makes the reward algebra consistent with the collision/offroad and generate-new-goal branches.

## 10. Recommended Workflow

For your tokenizer/data-collection work:

1. Keep native observations unchanged at first.
2. Slice the 1121 vector into ego/partner/road blocks in Python.
3. Build your custom representation from those blocks.
4. Only edit `drive.h` when you need simulator-level speed/memory reductions or when training code must see a smaller `observation_space`.

For reward experiments:

1. First tune constructor/config knobs.
2. If that is not enough, edit `c_step()` in `drive.h`.
3. Rebuild with:

```bash
python setup.py build_ext --inplace --force
```

4. Run a short rollout and print reward components/logs to verify the sign and scale.

## 11. Quick Python Block Parser

For classic observations:

```python
def split_classic_drive_obs(obs):
    ego_dim = 8
    partner_slots = 31
    partner_dim = 7
    road_slots = 128
    road_dim = 7

    ego_end = ego_dim
    partner_end = ego_end + partner_slots * partner_dim

    ego = obs[:, :ego_end]
    partners = obs[:, ego_end:partner_end].reshape(obs.shape[0], partner_slots, partner_dim)
    roads = obs[:, partner_end:].reshape(obs.shape[0], road_slots, road_dim)
    return ego, partners, roads
```

For jerk observations, set `ego_dim = 11`.

