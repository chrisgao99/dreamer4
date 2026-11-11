#!/usr/bin/env bash
# Experiment 1b-v2 pipeline: labels v2 -> fine-tune (g v2 + unfrozen denoiser) -> 512-scene eval
# (unknown / oracle / reversed of the fine-tuned model, 16 rollouts, run in parallel) -> summary
# -> steering scene 95720e9471028705:6062 (32 rollouts). Finished stages are skipped on rerun.
#   cd dreamer4 && STEPS=10000 setsid nohup bash waymo/pair_order_z/run_exp1b_v2.sh > <run>/logs/pipeline.log 2>&1 &
set -euo pipefail
TRI30="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
D="$TRI30/dreamer4"
cd "$D"
PY="${PY:-$HOME/.conda/envs/dreamer4/bin/python}"
RUN="${RUN:-$D/waymo/pair_order_z/runs/exp1b_v2_dense_z_20261111}"
LABELS="$D/data/pair_order_z_labels_v2"
STEPS="${STEPS:-10000}"
K="${K:-16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
mkdir -p "$RUN/logs" "$RUN/eval"
say() { echo "[$(date '+%F %T')] $*"; }

if [[ ! -s "$LABELS/train_summary.json" || ! -s "$LABELS/val_summary.json" ]]; then
  say "labels v2"
  "$PY" waymo/pair_order_z/make_z_labels_v2.py --splits val train --workers 7 > "$RUN/logs/labels_v2.log" 2>&1
fi
[[ -s "$LABELS/train_summary.json" ]] || { say "labels v2 failed"; exit 1; }

if [[ ! -s "$RUN/train/done" ]]; then
  say "fine-tune: $STEPS steps -> $RUN/train"
  "$PY" waymo/pair_order_z/train_zbias_v2.py --out_dir "$RUN/train" --steps "$STEPS" > "$RUN/logs/train.log" 2>&1
  date -Is > "$RUN/train/done"
fi
WM="$RUN/train/wm_final.pt"; G="$RUN/train/zbias_final.pt"

pids=()
for mode in unknown oracle reversed; do
  if [[ ! -s "$RUN/eval/$mode/result.json" ]]; then
    rm -rf "$RUN/eval/$mode"
    say "eval $mode (K=$K) -> $RUN/eval/$mode"
    "$PY" waymo/pair_order_z/eval_zbias_v2.py --mode "$mode" --checkpoint "$WM" --zbias_ckpt "$G" \
      --num_rollouts "$K" --out_dir "$RUN/eval/$mode" > "$RUN/logs/eval_$mode.log" 2>&1 &
    pids+=($!)
  fi
done
for p in "${pids[@]}"; do wait "$p"; done

say "summary"
"$PY" waymo/pair_order_z/summarize_v2.py --run "$RUN" > "$RUN/logs/summary.log" 2>&1

say "steering scene 95720e9471028705:6062"
"$PY" waymo/pair_order_z/scene_test_1a.py --v2 --checkpoint "$WM" --zbias_ckpt "$G" \
  --out "$RUN/scene_95720e9471028705_6062" > "$RUN/logs/scene_test.log" 2>&1 || say "scene test failed (see log)"
say "done: $RUN/summary.md"
