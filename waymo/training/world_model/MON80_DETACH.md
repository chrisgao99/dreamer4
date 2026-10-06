# H40 pure MoN full80 detach ablation, 2026-09-22

All runs initialize independently from the original (not 70/30) H40 step95000
EMA. Context11, plan40, execute5, sixteen replans cover 80 future frames.
CUDA1: detach0 (no truncation). CUDA2: detach5. CUDA3: detach20.
Launcher: launch_mon80_detach_tmux.sh; persistent tmux sessions.

Same data/order and candidate seeds, batch2, K8, Euler8, bf16, AdamW LR1e-6,
weight decay .01, gradient clip1, EMA .999, target10000 optimizer updates.
Dropout disabled (model.eval), gradients enabled. The 91-frame dataset provides
exactly one L11+H80 window, anchored at frame10. Partial tracks use the same
contiguous physically-valid GT mask as existing MoN; no dynamic/static weights.

Pure scene-joint MoN: Smooth-L1(xy) + .5 Smooth-L1(velocity) +
.5 Smooth-L1(sin/cos heading), averaged over all valid agent-time points.
K complete joint80 trajectories are scored without gradients; one winner per
scene is replayed with identical RNG and checked numerically before backward.
Each winning candidate is backpropagated; ONE optimizer update follows the
complete batch. No CAT-K/local winner selection, flow retention or physical loss.

Each segment is executed from its current (possibly detached) pose and the
executed poses are concatenated. Do not concatenate normalized actions and
re-integrate from the initial GT pose, which would bypass detach boundaries.
Only feedback history is detached; earlier outputs retain their own supervision.
The shared velocity loss uses differences of adjacent output poses, including
segment boundaries. Thus a boundary velocity error supervises both adjacent
poses directly; detachment cuts recurrent context paths, not these explicit
loss dependencies. Activation checkpointing recomputes each plan; it does not
truncate gradients.

Validation at initialization and every500 updates: fixed first32 validation
scenes, K4, H80/B5; mean ADE, minADE and CPD. Same subset across runs, not an
independent test set. best_ade.pt selected by mean ADE; retain snapshots every1000
so collapse/diversity tradeoffs remain inspectable. latest.pt every100; final.pt
at10000. No automatic CPD early stopping or additional loss.

Tests verify identical forward trajectories across detach settings, end-position
gradient support across all80/last5/last20 steps, preservation of early-segment
supervision, and checkpointed/non-checkpointed gradient equivalence.
