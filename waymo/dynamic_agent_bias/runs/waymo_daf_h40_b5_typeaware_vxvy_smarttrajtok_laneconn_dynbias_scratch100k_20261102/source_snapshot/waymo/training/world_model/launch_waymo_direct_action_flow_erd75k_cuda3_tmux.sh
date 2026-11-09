#!/usr/bin/env bash
# Alternate matched beta runs in 500-generator-update blocks on GPU3.
set -euo pipefail
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd)}"
PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
SESSION_NAME="${SESSION_NAME:-wm_erd75k_beta1_beta099_cuda3}"
MAX_GENERATOR_STEPS="${MAX_GENERATOR_STEPS:-10000}"
BLOCK_GENERATOR_STEPS=500
ROOT_NAME=waymo_direct_action_flow_generateall75k_erd_h15_b5
CKPT="$REPO_ROOT/waymo/checkpoints/waymo_direct_action_flow_v1_h15_b5_generateall_scratch100k_phys5_yaw075_huber_bounded_tmax090_lr5e5/best.pt"
DATA_ROOT="$REPO_ROOT/data/waymo_vector_dataset_ooi_centered_50k"
TRAIN_SCRIPT="$REPO_ROOT/waymo/training/world_model/train_waymo_direct_action_flow_erd.py"
LOG_DIR="$REPO_ROOT/waymo/logs/wm"
for file in "$CKPT" "$TRAIN_SCRIPT" "$PYTHON"; do [[ -f "$file" ]] || { echo "Missing: $file"; exit 1; }; done
[[ -d "$DATA_ROOT/train" && -d "$DATA_ROOT/val" ]] || { echo 'Missing dataset'; exit 1; }
[[ "$MAX_GENERATOR_STEPS" =~ ^[0-9]+$ ]] && (( MAX_GENERATOR_STEPS > 0 && MAX_GENERATOR_STEPS % 500 == 0 )) || { echo 'MAX_GENERATOR_STEPS must be a positive multiple of 500'; exit 1; }
if [[ "${RUN_INSIDE_TMUX:-0}" != 1 ]]; then
    if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then echo "Already running: $SESSION_NAME"; exit 1; fi
    printf -v command '%q ' env RUN_INSIDE_TMUX=1 REPO_ROOT="$REPO_ROOT" PYTHON="$PYTHON" CUDA_DEVICE="$CUDA_DEVICE" SESSION_NAME="$SESSION_NAME" MAX_GENERATOR_STEPS="$MAX_GENERATOR_STEPS" bash "$SCRIPT_PATH"
    tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT"
    tmux set-window-option -t "$SESSION_NAME:0" remain-on-exit on
    tmux respawn-pane -k -t "$SESSION_NAME:0.0" "$command"
    echo "Started $SESSION_NAME on GPU $CUDA_DEVICE; target $MAX_GENERATOR_STEPS generator updates per beta"
    echo "Attach: tmux attach -t $SESSION_NAME"
    exit 0
fi
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG_DIR/${ROOT_NAME}_queue.log") 2>&1
for (( target=BLOCK_GENERATOR_STEPS; target<=MAX_GENERATOR_STEPS; target+=BLOCK_GENERATOR_STEPS )); do
    for beta in 1.0 0.99; do
        if [[ "$beta" == 1.0 ]]; then tag=beta100; else tag=beta099; fi
        name="${ROOT_NAME}_${tag}_10kg_20260911"
        out="$REPO_ROOT/waymo/checkpoints/$name"
        resume=(); iteration=0
        if [[ -f "$out/latest.pt" ]]; then
            iteration=$("$PYTHON" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False,mmap=True)["iteration"])' "$out/latest.pt")
            resume=(--resume "$out/latest.pt")
        fi
        remaining=$(( target*10-iteration ))
        if (( remaining <= 0 )); then continue; fi
        echo "===== $(date -Is) beta=$beta target_generator_step=$target resume_iteration=$iteration ====="
        "$PYTHON" "$TRAIN_SCRIPT" --checkpoint "$CKPT" --train_data "$DATA_ROOT/train" --val_data "$DATA_ROOT/val" \
            --output_dir "$out" --beta "$beta" --max_generator_steps "$MAX_GENERATOR_STEPS" \
            --max_iterations "$remaining" "${resume[@]}" 2>&1 | tee -a "$LOG_DIR/$name.log"
    done
done
# Full-size final evaluations. Existing finished outputs are skipped; partial
# outputs are left intact and require deliberate recovery, never overwritten.
for tag in beta100 beta099; do
    name="${ROOT_NAME}_${tag}_10kg_20260911"
    for b in 1 5 15; do
        out="$REPO_ROOT/waymo/eval_results/world_model/${name}_final_h80_b${b}_k32_val128_cpd"
        if [[ -f "$out/result.json" ]]; then continue; fi
        mkdir -p "$out"
        "$PYTHON" "$REPO_ROOT/waymo/evaluation/eval_waymo_direct_action_flow_multisample.py" \
            --checkpoint "$REPO_ROOT/waymo/checkpoints/$name/final.pt" --val_data_dir "$DATA_ROOT/val" \
            --output_json "$out/result.json" --rollout_ade_csv "$out/rollout_ade.csv" \
            --scene_metrics_csv "$out/scene_metrics.csv" --details_npz "$out/details.npz" \
            --device cuda --weights ema --focus_mode generate_all --eval_batch_size 4 --eval_max_batches 128 \
            --num_workers 4 --num_rollouts 32 --rollout_steps 80 --commitment_steps "$b" --solver_steps 8 \
            --seed 12346 --log_every 1 --cpd_type_scales 25.423524547767016 4.925798640017075 18.77286026475977 \
            2>&1 | tee -a "$LOG_DIR/${name}_eval_b${b}.log"
    done
done
echo "===== $(date -Is) ERD training and final evaluations complete ====="
