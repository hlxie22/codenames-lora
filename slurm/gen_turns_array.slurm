#!/bin/bash
#SBATCH --job-name=cn-gen
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs

MAIN_PID=""
REQUEUE_SENT=0

_job_handle_id() {
  # Prefer requeueing the specific array task if present (cluster-dependent)
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  else
    echo "${SLURM_JOB_ID}"
  fi
}

_kill_main() {
  if [[ -n "${MAIN_PID}" ]] && kill -0 "${MAIN_PID}" 2>/dev/null; then
    # Try killing the whole process group first, then the pid.
    kill -TERM -- "-${MAIN_PID}" 2>/dev/null || true
    kill -TERM "${MAIN_PID}" 2>/dev/null || true
  fi
}

_requeue_me() {
  if (( REQUEUE_SENT == 1 )); then return 0; fi
  REQUEUE_SENT=1

  local jid
  jid="$(_job_handle_id)"

  echo "[$(date)] Signal received; requesting requeue of job handle: ${jid}"
  if ! scontrol requeue "${jid}"; then
    echo "[$(date)] ERROR: scontrol requeue failed for ${jid}" >&2
    # Best-effort diagnostics
    scontrol show job "${jid}" 2>/dev/null || true
  fi

  _kill_main
  exit 0
}

# Time-limit warning (per #SBATCH --signal=...)
trap _requeue_me USR1
# Common cancel/preempt paths
trap _requeue_me TERM INT USR2

source slurm/common.sh

CONFIG="${CONFIG:-configs/default.yml}"

# Run in background so traps can interrupt/cleanup reliably
python -m src.generate_turns_data --config "$CONFIG" --no-merge &
MAIN_PID=$!

wait "$MAIN_PID"
RC=$?

exit "$RC"