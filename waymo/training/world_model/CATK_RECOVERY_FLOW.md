# CAT-K-inspired continuous flow recovery, 2026-09-22

Scheme B: original H40 step95000 EMA, L11, plan40, execute5, rollout80.
CUDA0, tmux catk_recovery_h40_95k_k8_b5_h80_cuda0_20260922.
Launcher: launch_catk_recovery_cuda0_tmux.sh.
This samples eight joint candidates; it is NOT probability-ranked top-K CAT-K.

At each of 16 segments, sample K8 H40 plans without gradients using current
generated context. Select one candidate per whole scene using the existing
position + .5 velocity + .5 sin/cos heading Smooth-L1 distance over the next
five GT-valid steps. Execute those five generated steps, detach history, repeat.
No per-agent candidate mosaics, no GT resets, no best-of80 winner, no MoN loss.

The supervised target at each pre-execution context is separately constructed:
first action goes from the generated current pose to the next GT pose;
subsequent actions are the GT-to-GT local increments. Up to40 future steps
are supervised. Tail segments use only remaining GT, padded with invalid zeros.
Original contiguous physical GT masks are retained; disappeared tracks cannot
reappear as targets. Recovery transitions exceeding5m or.75rad are rejected.
Normalized actions outside the original +/-5 target range are rejected rather
than clipped; masks remain contiguous prefixes. Log first-action rejection
fractions and target coverage to expose excessive/implausible recovery filtering.

On each generated context, use the original Huber conditional flow-matching
objective (beta1, tmax.9), without dynamic/static reweighting. Each segment loss
is divided by16; empty segments contribute zero. The up-to40-step supervision
windows overlap: this is equal-per-context training, not equal weight per unique
future timestamp. Backward per segment bounds memory; one optimizer update after
all16. Sampling and feedback are detached; only the scene encoder/flow network
for each local recovery objective receive gradients. Selection and training use
the same frozen-within-update model parameters. Dropout disabled, bf16.

Batch2, K8, Euler8, AdamW LR1e-6/WD.01, clip1, EMA.999, seed20260911,
10000 updates. No original-GT-context retention loss, no physical loss, no
MoN loss, no autoregressive gradient through selected histories.

Validation is UNGUIDED free sampling on the same first32 validation scenes as
the three MoN ablations: K4, H80/B5, mean ADE/minADE/CPD, initially and every500
updates. Never use GT selection at evaluation. best_ade.pt selected by mean ADE;
latest every100, snapshots every1000, final at10000. CPD is monitored, not a loss.

Tests: recovery execution matches GT from a displaced context, future padding,
physical and normalized rejection (no clipping), no GT-track resurrection,
detached feedback, and whole-scene candidate selection without agent mosaics.
