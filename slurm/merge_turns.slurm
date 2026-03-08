#!/bin/bash
#SBATCH --job-name=cn-merge
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs

source slurm/common.sh

# Use CONFIG from pipeline export; default to configs/default.yml
CONFIG="${CONFIG:-configs/default.yml}"

# TOTAL_SHARDS must be exported by pipeline (jobs * procs_per_job)
TOTAL_SHARDS="${TOTAL_SHARDS:?need TOTAL_SHARDS exported from pipeline (jobs * procs_per_job)}"

# Optional: allow partial merges mid-run (set in pipeline if you want)
ALLOW_MISSING="${ALLOW_MISSING:-0}"

MERGE_ARGS=(--config "$CONFIG" --num-shards "$TOTAL_SHARDS")
if [[ "$ALLOW_MISSING" == "1" ]]; then
  MERGE_ARGS+=(--allow-missing)
fi

echo "[$(date)] Merging turns shards..."
python -m src.merge_turns_shards "${MERGE_ARGS[@]}"

echo "[$(date)] Building TRL DPO pairs..."
python -m src.build_dpo_pairs --config "$CONFIG"

echo "[$(date)] Done."