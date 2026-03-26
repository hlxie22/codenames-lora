#!/bin/bash
#SBATCH --job-name=cn-refilter
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

python - <<'PY'
import os
from src.utils import load_yaml, read_jsonl, write_jsonl
from src.generate_turns_data import filter_examples, _cfg_turns_paths

cfg = load_yaml(os.environ["CONFIG"])
turns_path, turns_raw_path = _cfg_turns_paths(cfg)

raw = read_jsonl(turns_raw_path)
filtered = filter_examples(raw, cfg)
write_jsonl(turns_path, filtered)

print(f"CONFIG={os.environ['CONFIG']}")
print(f"RAW={turns_raw_path}")
print(f"OUT={turns_path}")
print(f"raw_rows={len(raw)}")
print(f"filtered_rows={len(filtered)}")
PY

echo "[$(date)] Done."