#!/usr/bin/env bash
set -euo pipefail
ROOT=/p/yufeng/tri30/dreamer4
OUT_ROOT="$ROOT/waymo/eval_results/world_model/h40_95k_finetune_ctx11_h80_b5_k32_scenes512_20260925"
PYTHON=/p/yufeng/.conda/envs/dreamer4/bin/python
item="${1:?experiment}"; gpu="${2:?gpu}"
case "$item:$gpu" in
 catk:2) name=waymo_daf_h40_95k_catk_sample8_recovery_flow_unroll80_b5_10k_20260922 ;;
 mon_detach0:2) name=waymo_daf_h40_95k_mon8_gt_unroll80_b5_detach0_10k_20260922 ;;
 mon_detach5:3) name=waymo_daf_h40_95k_mon8_gt_unroll80_b5_detach5_10k_20260922 ;;
 mon_detach20:3) name=waymo_daf_h40_95k_mon8_gt_unroll80_b5_detach20_10k_20260922 ;;
 *) exit 2 ;;
esac
out="$OUT_ROOT/$item"
mkdir -p "$out"
exec > >(tee -a "$out/run.log") 2>&1
trap 'rc=$?; echo "$rc" > "$out/exit_code"; date -Is > "$out/finished_at"' EXIT
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ckpt="$ROOT/waymo/checkpoints/waymo_daf_h40_95k_finetune/$name/best_ade.pt"
echo "Started $(date -Is) CUDA=$gpu checkpoint=$ckpt"
"$PYTHON" waymo/evaluation/eval_waymo_direct_action_flow_multisample.py \
 --checkpoint "$ckpt" --val_data_dir "$ROOT/data/waymo_vector_dataset_ooi_centered_50k/val" \
 --output_json "$out/result.json" --rollout_ade_csv "$out/rollout_ade.csv" \
 --scene_metrics_csv "$out/scene_metrics.csv" --details_npz "$out/details.npz" \
 --device cuda --weights ema --focus_mode generate_all --eval_batch_size 4 --eval_max_batches 128 \
 --num_workers 4 --num_rollouts 32 --rollout_steps 80 --commitment_steps 5 --solver_steps 8 \
 --seed 12346 --log_every 1 --physical_metrics \
 --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977
"$PYTHON" "$OUT_ROOT/summarize.py" "$out"
