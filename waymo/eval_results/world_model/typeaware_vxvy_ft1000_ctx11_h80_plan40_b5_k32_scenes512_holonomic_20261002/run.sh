#!/usr/bin/env bash
set -euo pipefail
ROOT=/p/yufeng/tri30/dreamer4
OUT="$ROOT/waymo/eval_results/world_model/typeaware_vxvy_ft1000_ctx11_h80_plan40_b5_k32_scenes512_holonomic_20261002"
PYTHON=/p/yufeng/.conda/envs/dreamer4/bin/python
cd "$ROOT"
exec > "$OUT/run.log" 2>&1
trap 'rc=$?; echo "$rc" > "$OUT/exit_code"; date -Is > "$OUT/finished_at"' EXIT
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
CKPT="$ROOT/waymo/checkpoints/waymo_daf_h40_typeaware_vxvy_best95k_mon8_gt_unroll80_b5_detach0_10k_20260929/step_0001000.pt"
date -Is > "$OUT/started_at"
"$PYTHON" - "$CKPT" "$OUT" <<'PY'
import json, sys, torch
from pathlib import Path
c=torch.load(sys.argv[1],map_location='cpu',weights_only=False,mmap=True)
assert c['step']==1000
assert c['args']['horizon']==40
assert 'ema_model' in c
assert torch.cuda.is_available()
config=dict(checkpoint=sys.argv[1], checkpoint_step=c['step'], weights='ema', plan_horizon=40, commitment_steps=5, rollout_steps=80, num_scenes=512, num_rollouts=32, eval_batch_size=4, eval_max_batches=128, solver_steps=8, seed=12346, metric_validity='holonomic', focus_mode='generate_all', cuda_index=0, metrics=['mean_ade_m','minade_m','flow_erd_cpd','first40_mean_ade_m'])
Path(sys.argv[2],'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
print('Preflight passed:',json.dumps(config),flush=True)
PY
"$PYTHON" -u waymo/evaluation/eval_waymo_direct_action_flow_multisample.py \
 --checkpoint "$CKPT" --val_data_dir "$ROOT/data/waymo_vector_dataset_ooi_centered_50k_with_lengths/val" \
 --output_json "$OUT/result.json" --rollout_ade_csv "$OUT/rollout_ade.csv" \
 --scene_metrics_csv "$OUT/scene_metrics.csv" --details_npz "$OUT/details.npz" \
 --device cuda --weights ema --focus_mode generate_all --metric_validity holonomic \
 --eval_batch_size 4 --eval_max_batches 128 --num_workers 4 --num_rollouts 32 \
 --rollout_steps 80 --commitment_steps 5 --solver_steps 8 --seed 12346 --log_every 1 \
 --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977
