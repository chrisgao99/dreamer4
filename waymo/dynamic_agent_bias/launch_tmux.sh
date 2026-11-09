#!/usr/bin/env bash
# Dynamic agent bias: stage one (flow matching, 100k) then MON-8 finetune (10k) from step 90k,
# on CUDA 0,1,3, in one detached tmux session. Usage:
#   bash waymo/dynamic_agent_bias/launch_tmux.sh
# Re-running the same command after an interruption resumes from the latest checkpoints.
set -euo pipefail
main() {
  REPO=/p/yufeng/tri30/dreamer4
  PYTHON="${PYTHON:-/p/yufeng/.conda/envs/dreamer4/bin/python}"
  TORCHRUN="$(dirname "$PYTHON")/torchrun"
  GPUS="${GPUS:-0,1,3}"
  NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
  STAGE1="${STAGE1:-waymo_daf_h40_b5_typeaware_vxvy_smarttrajtok_laneconn_dynbias_scratch100k_20261102}"
  STAGE2="${STAGE2:-waymo_daf_laneconn_dynbias90k_mon8_gt_unroll80_b5_detach0_10k_20261102}"
  SESSION="${SESSION:-dynbias_20261102}"
  RUN_DIR="$REPO/waymo/dynamic_agent_bias/runs/$STAGE1"
  SNAP="$RUN_DIR/source_snapshot"
  CK1="$REPO/waymo/checkpoints/$STAGE1"
  CK2="$REPO/waymo/checkpoints/$STAGE2"
  LOG1="$REPO/waymo/logs/wm/$STAGE1.log"
  LOG2="$REPO/waymo/logs/wm/$STAGE2.log"
  DATA="$REPO/data/waymo_vector_dataset_ooi_centered_50k_with_lengths"
  # Smoke tests only: shorter stages and extra arguments; defaults are the real run.
  S1_STEPS="${S1_STEPS:-100000}"; S1_SOURCE="${S1_SOURCE:-90000}"
  S1_EXTRA="${S1_EXTRA:-}"; S2_EXTRA="${S2_EXTRA:-}"
  S1_FINAL="$CK1/final_step_$(printf %09d "$S1_STEPS").pt"; S1_CKPT="$CK1/step_$(printf %09d "$S1_SOURCE").pt"

  if [[ "${1:-}" != --worker ]]; then
    if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "tmux session $SESSION exists" >&2; exit 1; fi
    for gpu in ${GPUS//,/ }; do
      uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id="$gpu")"
      # ALLOW_SHARED_GPUS=1: the user accepted sharing cards with other jobs.
      if [[ "${ALLOW_SHARED_GPUS:-0}" != 1 ]] && nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then
        echo "CUDA $gpu is in use; refusing to start" >&2; exit 1
      fi
    done
    mkdir -p "$RUN_DIR" "$REPO/waymo/logs/wm"
    if [[ ! -d "$SNAP" ]]; then
      # Freeze the code: later edits in the repository cannot change this run.
      (cd "$REPO" && find waymo -path waymo/checkpoints -prune -o -path waymo/eval_results -prune \
          -o -path waymo/logs -prune -o -path waymo/wandb -prune -o -path waymo/cache -prune \
          -o -path waymo/dynamic_agent_bias/runs -prune \
          -o \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) -size -2M -print | grep -v __pycache__ \
          | cpio -pdm --quiet "$SNAP")
      git -C "$REPO" rev-parse HEAD > "$RUN_DIR/git_head.txt"
      git -C "$REPO" diff > "$RUN_DIR/git_diff_at_launch.patch" || true
    fi
    printf -v command '%q ' env PYTHON="$PYTHON" GPUS="$GPUS" STAGE1="$STAGE1" STAGE2="$STAGE2" SESSION="$SESSION" \
      S1_STEPS="$S1_STEPS" S1_SOURCE="$S1_SOURCE" S1_EXTRA="$S1_EXTRA" S2_EXTRA="$S2_EXTRA" \
      bash "$SNAP/waymo/dynamic_agent_bias/launch_tmux.sh" --worker
    tmux new-session -d -s "$SESSION" -c "$REPO" "$command"
    echo "Started tmux session $SESSION (attach: tmux attach -t $SESSION)"
    echo "Logs: $LOG1 then $LOG2; phase: $RUN_DIR/PHASE.txt"
    exit 0
  fi

  cd "$SNAP"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
  phase() { echo "$1 $(date -Is)" | tee -a "$RUN_DIR/PHASE.txt"; }
  # Up to three attempts per stage; every retry resumes from the last saved checkpoint.
  attempt() {
    local name="$1"; shift
    for try in 1 2 3; do
      if "$@"; then return 0; fi
      phase "$name failed attempt $try"; sleep 60
    done
    return 1
  }

  phase tests
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON" waymo/dynamic_agent_bias/tests/test_dynamic_bias.py >> "$LOG1" 2>&1 \
    || { phase 'tests failed'; echo 1 > "$RUN_DIR/exit_code.txt"; exit 1; }

  stage1() {
    local resume=()
    [[ -f "$CK1/latest.pt" ]] && resume=(--resume)
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" waymo/dynamic_agent_bias/train_stage1_ddp.py \
      --run_name "$STAGE1" --num_workers 6 "${resume[@]}" $S1_EXTRA >> "$LOG1" 2>&1
  }
  if [[ ! -f "$S1_FINAL" ]]; then
    phase stage1_flow_matching
    attempt stage1 stage1 || { phase 'stage1 failed'; echo 1 > "$RUN_DIR/exit_code.txt"; exit 1; }
  fi
  [[ -f "$S1_CKPT" ]] || { phase "missing $S1_CKPT"; echo 1 > "$RUN_DIR/exit_code.txt"; exit 1; }

  stage2() {
    local resume=()
    [[ -f "$CK2/latest.pt" ]] && resume=(--resume "$CK2/latest.pt")
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" waymo/dynamic_agent_bias/train_mon80_ddp.py \
      --checkpoint "$S1_CKPT" --train_data "$DATA/train" --val_data "$DATA/val" --output_dir "$CK2" \
      --map_cache_dir "$REPO/data/waymo_map_smarttrajtok_5m_v2" --lane_topology_dir "$REPO/data/waymo_lane_connectivity_5m_v1" \
      --detach_every 0 --seed 20260911 --batch_size 2 --num_workers 4 --num_candidates 8 --solver_steps 8 \
      --max_steps 10000 --lr 1e-6 --ema_decay .999 --eval_every 500 --eval_batches 8 --eval_batch_size 4 \
      --eval_rollouts 4 --physical_eval_batches 0 --save_every 100 --snapshot_every 1000 --log_every 10 \
      "${resume[@]}" $S2_EXTRA >> "$LOG2" 2>&1
  }
  if [[ ! -f "$CK2/final.pt" ]]; then
    phase stage2_mon80
    attempt stage2 stage2 || { phase 'stage2 failed'; echo 1 > "$RUN_DIR/exit_code.txt"; exit 1; }
  fi
  phase complete
  echo 0 > "$RUN_DIR/exit_code.txt"
}
main "$@"; exit $?
