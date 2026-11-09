# Evaluation defaults

Direct action flow multisample evaluation defaults to `--metric_validity holonomic`, including when evaluating type-aware or adapted-map checkpoints. Keep this scoring mask when comparing experiments. This controls logged ground-truth validity only; model execution still uses checkpoint kinematics.

The map-adaptation/connectivity launcher explicitly selects holonomic scoring. 512-scene protocol: first 512 validation files, batch 4, K32, 80-step closed loop, H40/B5, EMA, solver 8, seed 12346.

The image-only coverage launcher uses the existing fixed intersection10 manifest, random32, H40/B40, 80 steps, EMA, solver 8, seed 20260907. Logged valid GT is shown without inverse-action truncation. It saves ten scene PNGs plus a contact-sheet PNG; no JSON/NPZ artifacts. Logs are outside the image output directories.
