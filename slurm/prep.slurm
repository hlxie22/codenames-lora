#!/bin/bash
#SBATCH --job-name=cn-prep
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs

source slurm/common.sh

CONFIG="${CONFIG:-configs/default.yml}"
export CONFIG
CLEAN="${CLEAN:-0}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG not found: $CONFIG" >&2
  exit 1
fi

read -r VOCAB_PATH VOCAB_MANIFEST_PATH BOARDS_TRAIN_PATH BOARDS_EVAL_PATH TURNS_PATH TURNS_RAW_PATH DPO_PAIRS_PATH < <(
  python - <<'PY'
import os
from src.utils import load_yaml

cfg = load_yaml(os.environ["CONFIG"])
vocab_manifest = str((cfg.get("vocab", {}) or {}).get("manifest_path", "") or "")

print(
    f"{cfg['paths']['vocab_path']}\t"
    f"{vocab_manifest}\t"
    f"{cfg['paths']['boards_train_path']}\t"
    f"{cfg['paths']['boards_eval_path']}\t"
    f"{cfg['paths']['turns_path']}\t"
    f"{cfg['paths']['turns_raw_path']}\t"
    f"{cfg['paths']['dpo_pairs_path']}"
)
PY
)

if [[ "$CLEAN" == "1" ]]; then
  echo "[clean] Removing existing data artifacts..."

  rm -f "$VOCAB_PATH"
  [[ -n "$VOCAB_MANIFEST_PATH" ]] && rm -f "$VOCAB_MANIFEST_PATH"

  rm -f "$BOARDS_TRAIN_PATH" "$BOARDS_EVAL_PATH"
  rm -f "$TURNS_PATH" "$TURNS_RAW_PATH" "$DPO_PAIRS_PATH"

  rm -f "${TURNS_RAW_PATH}".shard*-of-* 2>/dev/null || true
  rm -f "${TURNS_RAW_PATH}".shard*-of-*.progress.json 2>/dev/null || true
fi

if [[ ! -f "$VOCAB_PATH" ]]; then
  echo "[step] build_vocab"
  python -m src.build_vocab --config "$CONFIG"
else
  echo "[step] build_vocab: exists, skipping"
fi

if [[ ! -f "$BOARDS_TRAIN_PATH" || ! -f "$BOARDS_EVAL_PATH" ]]; then
  echo "[step] make_boards"
  python -m src.make_boards --config "$CONFIG"
else
  echo "[step] make_boards: exists, skipping"
fi
