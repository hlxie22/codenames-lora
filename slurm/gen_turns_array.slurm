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

REQUEUE_SENT=0
handle_usr1() {
  if (( REQUEUE_SENT == 1 )); then return 0; fi
  REQUEUE_SENT=1
  echo "[$(date)] USR1 received: requesting requeue..."
  scontrol requeue "$SLURM_JOB_ID" || true
  exit 0
}
trap handle_usr1 USR1

source slurm/common.sh

# For arrays, prefer shared HF cache (common.sh sets HF_HOME to SCRATCH/HOME)
CONFIG="${CONFIG:-configs/default.yml}"

# generate_turns_data detects SLURM_ARRAY_TASK_ID/COUNT and shards globally
python -m src.generate_turns_data --config "$CONFIG" --no-merge