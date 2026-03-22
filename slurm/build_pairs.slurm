#!/bin/bash
#SBATCH --job-name=cn-build-pairs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs

source slurm/common.sh

CONFIG="${CONFIG:-configs/default.yml}"
export CONFIG

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG not found: $CONFIG" >&2
  exit 1
fi

read -r TURNS_RAW_PATH DPO_PAIRS_PATH < <(
  python - <<'PY'
import os
from src.utils import load_yaml

cfg = load_yaml(os.environ.get("CONFIG", "configs/default.yml"))
print(f"{cfg['paths']['turns_raw_path']}\t{cfg['paths']['dpo_pairs_path']}")
PY
)

if [[ -z "${TURNS_RAW_PATH}" ]]; then
  echo "ERROR: Could not resolve paths.turns_raw_path from $CONFIG" >&2
  exit 1
fi

if [[ -z "${DPO_PAIRS_PATH}" ]]; then
  echo "ERROR: Could not resolve paths.dpo_pairs_path from $CONFIG" >&2
  exit 1
fi

if [[ ! -s "$TURNS_RAW_PATH" ]]; then
  echo "ERROR: turns_raw missing or empty: $TURNS_RAW_PATH" >&2
  exit 1
fi

echo "[$(date)] Building TRL DPO pairs..."
echo "[$(date)] CONFIG=$CONFIG"
echo "[$(date)] TURNS_RAW_PATH=$TURNS_RAW_PATH"
echo "[$(date)] DPO_PAIRS_PATH=$DPO_PAIRS_PATH"

python -m src.build_dpo_pairs --config "$CONFIG"

if [[ ! -s "$DPO_PAIRS_PATH" ]]; then
  echo "ERROR: DPO pairs file was not created or is empty: $DPO_PAIRS_PATH" >&2
  exit 1
fi

echo "[$(date)] Done."