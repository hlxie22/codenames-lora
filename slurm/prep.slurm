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
CLEAN="${CLEAN:-0}"

if [[ "$CLEAN" == "1" ]]; then
  echo "[clean] Removing existing data artifacts..."
  rm -f data/vocab.txt data/vocab_manifest.json
  rm -f data/boards_train.jsonl data/boards_eval.jsonl

  # Base + iteration-scoped turns/pairs
  rm -f data/turns.jsonl data/turns_raw.jsonl data/dpo_pairs.jsonl
  rm -f data/turns_iter*.jsonl data/turns_raw_iter*.jsonl data/dpo_pairs_iter*.jsonl

  # Shards + progress for any variant
  rm -f data/turns_raw.jsonl.shard*-of-* 2>/dev/null || true
  rm -f data/turns_raw_iter*.jsonl.shard*-of-* 2>/dev/null || true
  rm -f data/turns_raw.jsonl.*.progress.json 2>/dev/null || true
  rm -f data/turns_raw_iter*.jsonl.*.progress.json 2>/dev/null || true
fi

if [[ ! -f data/vocab.txt ]]; then
  echo "[step] build_vocab"
  python -m src.build_vocab --config "$CONFIG"
else
  echo "[step] build_vocab: exists, skipping"
fi

if [[ ! -f data/boards_train.jsonl || ! -f data/boards_eval.jsonl ]]; then
  echo "[step] make_boards"
  python -m src.make_boards --config "$CONFIG"
else
  echo "[step] make_boards: exists, skipping"
fi