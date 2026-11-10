#!/usr/bin/env bash
# Experiment 1 pipeline (plan §8.2-8.3): labels -> train g (frozen 85k WM) -> eval oracle & reversed
# (in parallel) -> wait for the baseline eval (unknown, started separately) -> summary.
#   bash waymo/pair_order_z/run_exp1.sh            (from dreamer4/; runs in the foreground)
# All paths are relative to tri30/ via TRI30; resumable stage by stage (finished stages are skipped).
set -euo pipefail
TRI30="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
D="$TRI30/dreamer4"
cd "$D"
PY="${PY:-$HOME/.conda/envs/dreamer4/bin/python}"
RUN="${RUN:-$D/waymo/pair_order_z/runs/exp1_oracle_z_frozen_20261110}"
LABELS="$D/data/pair_order_z_labels_v1"
STEPS="${STEPS:-6000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
mkdir -p "$RUN/logs" "$RUN/eval"
say() { echo "[$(date '+%F %T')] $*"; }

say "labels: waiting for $LABELS/{val,train}.npz"
while [[ ! -s "$LABELS/train_summary.json" || ! -s "$LABELS/val_summary.json" ]]; do sleep 30; done

if [[ ! -s "$RUN/train/done" ]]; then
  say "train g: $STEPS steps -> $RUN/train"
  "$PY" waymo/pair_order_z/train_zbias_frozen.py --out_dir "$RUN/train" --steps "$STEPS" \
    > "$RUN/logs/train.log" 2>&1
  date -Is > "$RUN/train/done"
fi
G="$RUN/train/zbias_best.pt"

pids=()
for mode in oracle reversed; do
  if [[ ! -s "$RUN/eval/$mode/result.json" ]]; then
    rm -rf "$RUN/eval/$mode"
    say "eval $mode -> $RUN/eval/$mode"
    "$PY" waymo/pair_order_z/eval_zbias.py --mode "$mode" --zbias_ckpt "$G" --out_dir "$RUN/eval/$mode" \
      > "$RUN/logs/eval_$mode.log" 2>&1 &
    pids+=($!)
  fi
done
for p in "${pids[@]}"; do wait "$p"; done

say "waiting for baseline $RUN/eval/unknown/result.json"
while [[ ! -s "$RUN/eval/unknown/result.json" ]]; do sleep 60; done

say "summary"
"$PY" waymo/pair_order_z/summarize_exp1.py --run "$RUN" > "$RUN/logs/summary.log" 2>&1
say "done: $RUN/summary.md"
