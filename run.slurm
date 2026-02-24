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
#SBATCH --mail-type=NONE             # disable Slurm mails (prevents requeue spam)

set -euo pipefail

mkdir -p logs

# ====== CONFIG: put your email here (in the file) ======
MAIL_TO="henryxie@college.harvard.edu"
MAIL_SUBJECT_PREFIX="Codenames Job Done"
# =======================================================

SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0")"

# If we ever fall back to "resubmit with dependency", keep a stable ID for logs + email context.
ORIG_JOB_ID="${ORIG_JOB_ID:-${SLURM_JOB_ID:-unknown}}"

REQUEUE_SENT=0
MAIN_PID=""
START_TIME="$(date --iso-8601=seconds 2>/dev/null || date)"

send_mail() {
  local subject="$1"
  local body="$2"

  if [[ -z "$MAIL_TO" ]]; then
    echo "[$(date)] MAIL_TO empty; skipping completion email."
    return 0
  fi

  if command -v mail >/dev/null 2>&1; then
    printf "%s\n" "$body" | mail -s "$subject" "$MAIL_TO"
  elif command -v mailx >/dev/null 2>&1; then
    printf "%s\n" "$body" | mailx -s "$subject" "$MAIL_TO"
  elif command -v sendmail >/dev/null 2>&1; then
    {
      echo "To: $MAIL_TO"
      echo "Subject: $subject"
      echo
      printf "%s\n" "$body"
    } | sendmail -t
  else
    echo "[$(date)] No mail/mailx/sendmail found; cannot send completion email."
    return 1
  fi
}

on_exit() {
  local rc=$?
  local end_time
  end_time="$(date --iso-8601=seconds 2>/dev/null || date)"

  # If we requeued/resubmitted, do NOT email.
  if (( REQUEUE_SENT == 1 )); then
    echo "[$(date)] Exiting due to requeue/resubmit; not sending completion email."
    return 0
  fi

  local out_log="logs/${SLURM_JOB_NAME:-job}-${ORIG_JOB_ID}.out"
  local err_log="logs/${SLURM_JOB_NAME:-job}-${ORIG_JOB_ID}.err"

  local subject="${MAIL_SUBJECT_PREFIX} ${SLURM_JOB_NAME:-job} (${ORIG_JOB_ID}) completed (rc=${rc})"
  local body
  body=$(
    cat <<EOF
Job completed (not requeued).

Job name:   ${SLURM_JOB_NAME:-unknown}
Orig jobid: ${ORIG_JOB_ID}
This jobid: ${SLURM_JOB_ID:-unknown}
User:       ${USER:-unknown}
Node(s):    ${SLURM_NODELIST:-unknown}
Start:      ${START_TIME}
End:        ${end_time}
Exit code:  ${rc}

Logs (appended across retries if applicable):
  OUT: ${out_log}
  ERR: ${err_log}
EOF
  )

  send_mail "$subject" "$body" || true
}
trap on_exit EXIT

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
    # Fallback: new job id, but force logs to append to the ORIGINAL job's log files
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
setsid python -m src.generate_sft_data --config configs/default.yml &
MAIN_PID=$!
wait "$MAIN_PID"
RC=$?
set -e

exit "$RC"