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

CONFIG="${CONFIG:-configs/default.yml}"
export CONFIG

TOTAL_SHARDS="${TOTAL_SHARDS:?need TOTAL_SHARDS exported from pipeline (jobs * procs_per_job)}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
CLEAN_SHARDS="${CLEAN_SHARDS:-1}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG not found: $CONFIG" >&2
  exit 1
fi

MERGE_ARGS=(--config "$CONFIG" --num-shards "$TOTAL_SHARDS")
if [[ "$ALLOW_MISSING" == "1" ]]; then
  MERGE_ARGS+=(--allow-missing)
fi

echo "[$(date)] Merging turns shards..."
echo "[$(date)] CONFIG=$CONFIG"
echo "[$(date)] TOTAL_SHARDS=$TOTAL_SHARDS"
echo "[$(date)] ALLOW_MISSING=$ALLOW_MISSING"
python -m src.merge_turns_shards "${MERGE_ARGS[@]}"

if [[ "$CLEAN_SHARDS" == "1" ]]; then
  echo "[$(date)] Cleaning shard files..."

  read -r TURNS_RAW_PATH < <(
    python - <<'PY'
import os
from src.utils import load_yaml
from src.generate_turns_data import _cfg_turns_paths

cfg = load_yaml(os.environ.get("CONFIG", "configs/default.yml"))
_, turns_raw_path = _cfg_turns_paths(cfg)
print(turns_raw_path)
PY
  )

  if [[ -z "${TURNS_RAW_PATH}" ]]; then
    echo "[$(date)] WARNING: could not resolve turns_raw_path; skipping shard cleanup."
  else
    shopt -s nullglob

    shard_files=( "${TURNS_RAW_PATH}".shard*-of-* )
    progress_files=( "${TURNS_RAW_PATH}".shard*-of-*.progress.json )

    if (( ${#shard_files[@]} > 0 )); then
      echo "[$(date)] Removing ${#shard_files[@]} shard files for ${TURNS_RAW_PATH}"
      rm -f -- "${shard_files[@]}"
    else
      echo "[$(date)] No shard files found for ${TURNS_RAW_PATH}"
    fi

    if (( ${#progress_files[@]} > 0 )); then
      echo "[$(date)] Removing ${#progress_files[@]} progress files for ${TURNS_RAW_PATH}"
      rm -f -- "${progress_files[@]}"
    fi
  fi
fi

echo "[$(date)] Done."