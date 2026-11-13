#!/bin/bash
# Bias head v1, full labels (plan BIAS_CONTROLLER_PLAN_2026_11_12 steps 3-4): resume labels -> check -> train -> eval.
set -e
cd /scratch/baz7dy/tri30/dreamer4
PY=~/.conda/envs/dreamer4/bin/python
R=waymo/pair_order_z/runs/bias_head_v1_20261112
mkdir -p $R/train_full $R/eval_full
$PY waymo/pair_order_z/make_bias_labels.py --out_dir $R/labels --n_scenes 2000 --K 4 >> $R/labels/run.log 2>&1
n=$(ls $R/labels/batches/*.npz | wc -l); s=$(ls $R/labels/batches/*.skipped 2>/dev/null | wc -l)
echo "$(date) labels: $n batch files, $s skipped"
if [ $((n + s)) -lt 500 ]; then echo "labels incomplete, stopping"; exit 1; fi
$PY waymo/pair_order_z/train_bias_head.py --labels_dir $R/labels --out_dir $R/train_full > $R/train_full/train.log 2>&1
echo "$(date) training done"
$PY waymo/pair_order_z/eval_bias_head.py --head $R/train_full/bias_head.pt --out_dir $R/eval_full --K 8 --taus 0.5,0.8 > $R/eval_full/eval.log 2>&1
echo "$(date) eval done"
