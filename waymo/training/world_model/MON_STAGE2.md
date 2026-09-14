# DirectActionFlow MoN stage-two comparison

Both runs start independently from the same generate-all best75k EMA. They use
context 11, H15 plans, B5 commitment, three differentiable replans per training
window, K8 joint scene samples. Same data split, anchors/seeds across MoN variants,
batch size 2, AdamW LR 1e-6 / WD .01, clip 1, EMA .999, bf16, deterministic CUDA.
10,000 generator updates each; CUDA1 MoN GT, CUDA2 MoN GT + physical.

## Objective

One joint winner per scene minimizes Smooth-L1(xy) + .5 Smooth-L1(vx,vy) +
.5 Smooth-L1(sin(yaw),cos(yaw)), valid GT transitions averaged equally. Both GT
and predicted velocities are computed with the same backward difference dt=.1.
No latent loss, teacher loss, shortcut retention, or velocity-position consistency.
No special ego or distance weights; all valid agents including ego are generated.

Eight no-grad candidates select the winner, then identical seeded candidates are
replayed with autograd. Exact replay is checked every iteration. Only winning
scene/candidate pairs get GT gradients. Physical losses average ALL eight samples;
candidates are backpropagated sequentially to bound memory, then ONE optimizer
step is taken. No truncation/detachment between replans in gradient replay.

Physical weights: collision .1, offroad .1, sideslip .1. Linear warmup 500 steps.
Sideslip uses midpoint-heading lateral velocity Smooth-L1 to zero for vehicles
and cyclists; pedestrians are exempt. Weight .1 is explicit adaptation, not the
old latent-model kinematic coefficient. Logged raw losses expose relative scale.

## Geometry and scope

NPZ lacks measured dimensions. Proxy rectangles: vehicle 4.8x2m, pedestrian
.8x.8m, cyclist 2x.8m, unknown 1x1m. Collision uses four-axis separating-axis gaps
for oriented rectangles (all valid pairs), warning distance 1m, softplus .2m.
Offroad uses maximum corner signed distance to nearest retained oriented road-edge
points (types15/16), margin .3m, softplus .2m, only vehicles/cyclists.
These are proxies, NOT official WOSAC collision/offroad metrics.

IMPORTANT GT audit: on first32 validation scenes/H15, unfiltered legacy-style
signed-edge proxy marks 27.3% of valid road-agent times offroad; this indicates
substantial GT/map/footprint inconsistency. Do not optimize these false positives.
The offroad training mask requires the logged footprint to be inside with .3m
margin at BOTH initial anchor and corresponding GT future frame. It depends only
on fixed data, never generated positions or predicted validity. Thus every
candidate is still penalized if it leaves the road at a supported frame. Report
road_reliable_fraction and unfiltered offroad alongside filtered offroad; never
hide discarded coverage or call the masked rate a whole-dataset official metric.
This cannot correct missing/mis-signed map geometry; it bounds the proxy's use.

## Monitoring/recovery

Every250 updates, use the SAME 32-scene/K4/H80/B5 ADE/minADE/CPD validation as ERD.
Also measure physical proxies on first8 scenes/K4/H80, with identical data-only
road masks for both methods. Log motion errors, candidate mean vs winner, physical
terms, gradient norm, exact replay error, and GPU memory/time. latest every100,
retained snapshots every1000; best_ade selected only by validation ADE. Save model,
EMA, optimizer, RNG, next step, configuration. SIGINT/TERM saves at step boundary.
Final evaluation queue: 512 scenes/K32/H80, B1/5/15. A common independent quality
analysis is still needed to judge road compliance / semantic diversity vs ERD;
CPD/minADE alone cannot establish superiority. Equal generator updates are not
equal compute: record GPU-hours and sample counts for method comparisons.

The existing ERD jobs and source are unchanged by this experiment.
