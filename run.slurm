#!/bin/bash
#SBATCH --job-name=codenames
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900          # 15 min before time limit
#SBATCH --open-mode=append           # keep logs across requeues (same job id)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Slurm email (controller-side)
#SBATCH --mail-user=henryxie@college.harvard.edu
#SBATCH --mail-type=END,FAIL         # omit REQUEUE to avoid requeue spam

set -euo pipefail

# IMPORTANT: logs/ must exist at *submit time* for the SBATCH --output/--error paths.
# Run: mkdir -p logs   (before: sbatch this_script.sh)
mkdir -p logs

SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0")"

# Stable ID for log naming if we fall back to "resubmit with dependency"
ORIG_JOB_ID="${ORIG_JOB_ID:-${SLURM_JOB_ID:-unknown}}"

REQUEUE_SENT=0
MAIN_PID=""

handle_usr1() {
  # prevent double-trigger in the same run
  if (( REQUEUE_SENT == 1 )); then
    return 0
  fi
  REQUEUE_SENT=1

  echo "[$(date)] USR1 received: requesting requeue, then stopping workload..."

  set +e
  # Prefer requeue (same job ID, restarts from top)
  scontrol requeue "$SLURM_JOB_ID"
  rc=$?
  if (( rc != 0 )); then
    echo "[$(date)] requeue failed (rc=$rc) -> resubmitting with dependency (append to original logs)"
    sbatch --dependency=afterany:"$SLURM_JOB_ID" \
          --export=ALL,ORIG_JOB_ID="$ORIG_JOB_ID" \
          --open-mode=append \
          --output="logs/%x-${ORIG_JOB_ID}.out" \
          --error="logs/%x-${ORIG_JOB_ID}.err" \
          "$SCRIPT_PATH"
  fi

  # Ask your workload to stop gracefully so it exits before walltime.
  if [[ -n "${MAIN_PID}" ]] && kill -0 "${MAIN_PID}" 2>/dev/null; then
    # If we started it with setsid, MAIN_PID is also the process-group id
    kill -TERM -- -"${MAIN_PID}" 2>/dev/null || kill -TERM "${MAIN_PID}" 2>/dev/null

    # Give it a little time to shut down cleanly
    for _ in $(seq 1 120); do
      kill -0 "${MAIN_PID}" 2>/dev/null || break
      sleep 1
    done

    # Last resort
    kill -KILL -- -"${MAIN_PID}" 2>/dev/null || true
  fi
  set -e

  exit 0
}
trap handle_usr1 USR1

module load miniforge/25.11.0-0

# --- Activate conda ---
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate codenames-lora

export VLLM_ATTENTION_BACKEND=TRITON
export VLLM_USE_PRECOMPILED=1

# Cache to local scratch (ephemeral per run; fine, but will be cold after requeue)
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# --- Run experiment ---
# Run workload in its own session so we can kill the whole group on USR1 (torchrun-friendly)
set +e
setsid torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m src.train_lora_sft --config configs/default.yml &
MAIN_PID=$!
wait "$MAIN_PID"
RC=$?
set -e

exit "$RC"