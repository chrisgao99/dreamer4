#!/bin/bash
# Bias head v1 (plan BIAS_CONTROLLER_PLAN_2026_11_12 steps 3-4): labels -> train -> closed-loop eval.
# Labels are launched separately (make_bias_labels.py, PID given as $1); this waits for them, then trains and evaluates.
set -e
cd /scratch/baz7dy/tri30/dreamer4
PY=~/.conda/envs/dreamer4/bin/python
R=waymo/pair_order_z/runs/bias_head_v1_20261112
LABEL_PID=$1
if [ -n "$LABEL_PID" ]; then
  while kill -0 "$LABEL_PID" 2>/dev/null; do sleep 60; done
fi
n=$(ls $R/labels/batches/*.npz | wc -l)
echo "$(date) labels done: $n batch files"
$PY waymo/pair_order_z/train_bias_head.py --labels_dir $R/labels --out_dir $R/train > $R/train/train.log 2>&1 || { mkdir -p $R/train; $PY waymo/pair_order_z/train_bias_head.py --labels_dir $R/labels --out_dir $R/train > $R/train/train.log 2>&1; }
echo "$(date) training done"
$PY waymo/pair_order_z/eval_bias_head.py --head $R/train/bias_head.pt --out_dir $R/eval --K 8 --taus 0.5,0.8 > $R/eval/eval.log 2>&1 || { mkdir -p $R/eval; $PY waymo/pair_order_z/eval_bias_head.py --head $R/train/bias_head.pt --out_dir $R/eval --K 8 --taus 0.5,0.8 > $R/eval/eval.log 2>&1; }
echo "$(date) eval done"
