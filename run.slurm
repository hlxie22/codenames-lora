#!/bin/bash
#SBATCH --job-name=codenames-sftpipe
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-user=henryxie@college.harvard.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

mkdir -p logs data outputs

SCRIPT_PATH="$(readlink -f "$0" 2>/dev/null || realpath "$0")"
ORIG_JOB_ID="${ORIG_JOB_ID:-${SLURM_JOB_ID:-unknown}}"

REQUEUE_SENT=0
MAIN_PID=""

handle_usr1() {
  if (( REQUEUE_SENT == 1 )); then return 0; fi
  REQUEUE_SENT=1

  echo "[$(date)] USR1 received: requesting requeue, then stopping workload..."

  set +e
  scontrol requeue "$SLURM_JOB_ID"
  rc=$?
  if (( rc != 0 )); then
    echo "[$(date)] requeue failed (rc=$rc) -> resubmitting with dependency"
    sbatch --dependency=afterany:"$SLURM_JOB_ID" \
          --export=ALL,ORIG_JOB_ID="$ORIG_JOB_ID" \
          --open-mode=append \
          --output="logs/%x-${ORIG_JOB_ID}.out" \
          --error="logs/%x-${ORIG_JOB_ID}.err" \
          "$SCRIPT_PATH"
  fi

  if [[ -n "${MAIN_PID}" ]] && kill -0 "${MAIN_PID}" 2>/dev/null; then
    kill -TERM -- -"${MAIN_PID}" 2>/dev/null || kill -TERM "${MAIN_PID}" 2>/dev/null
    for _ in $(seq 1 120); do
      kill -0 "${MAIN_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- -"${MAIN_PID}" 2>/dev/null || true
  fi
  set -e
  exit 0
}
trap handle_usr1 USR1

module load miniforge/25.11.0-0
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate codenames-lora

export VLLM_ATTENTION_BACKEND=TRITON
export VLLM_USE_PRECOMPILED=1
export TOKENIZERS_PARALLELISM=false

# Cache to local scratch (ephemeral per run)
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

CONFIG="configs/default.yml"

# Optional: set CLEAN=1 to wipe prior artifacts and regenerate from scratch
# (otherwise SFT generation will resume using existing raw jsonl)
CLEAN="${CLEAN:-0}"

# set +e
# setsid bash -lc "
#   set -euo pipefail

#   if [[ \"${CLEAN}\" == \"1\" ]]; then
#     echo '[clean] Removing existing data artifacts...'
#     rm -f data/vocab.txt data/vocab_manifest.json
#     rm -f data/boards_train.jsonl data/boards_eval.jsonl
#     rm -f data/sft_turns.jsonl data/sft_turns_raw.jsonl
#     rm -f data/sft_turns_raw.jsonl.shard*-of-* 2>/dev/null || true
#     rm -f data/sft_turns_raw.jsonl.*.progress.json 2>/dev/null || true
#   fi

#   # 1) Build vocab (idempotent-ish: skip if exists unless CLEAN=1)
#   if [[ ! -f data/vocab.txt ]]; then
#     echo '[step 1/3] build_vocab'
#     python -m src.build_vocab --config \"${CONFIG}\"
#   else
#     echo '[step 1/3] build_vocab: data/vocab.txt exists, skipping'
#   fi

#   # 2) Make boards (skip if exists unless CLEAN=1)
#   if [[ ! -f data/boards_train.jsonl || ! -f data/boards_eval.jsonl ]]; then
#     echo '[step 2/3] make_boards'
#     python -m src.make_boards --config \"${CONFIG}\"
#   else
#     echo '[step 2/3] make_boards: boards files exist, skipping'
#   fi

#   # 3) Generate SFT data (GPU; spawns inference.num_processes children itself)
#   echo '[step 3/3] generate_sft_data'
#   python -m src.generate_sft_data --config \"${CONFIG}\"

#   echo '[done] pipeline complete (up to SFT data)'
# " &
# MAIN_PID=$!
# wait "$MAIN_PID"
# RC=$?
# set -e

# exit "$RC"

# Run workload in its own session so we can kill the whole group on USR1 (torchrun-friendly)
set +e
setsid torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  -m src.train_lora_sft --config configs/default.yml &
MAIN_PID=$!
wait "$MAIN_PID"
RC=$?
set -e

exit "$RC"

# set +e

# setsid bash -lc "
#   python -m src.eval --config configs/default.yml --mode baseline --out outputs/eval/baseline &&
#   python -m src.eval --config configs/default.yml --mode sft      --out outputs/eval/sft
# " &
# MAIN_PID=$!

# wait "$MAIN_PID"
# RC=$?
# set -e

# exit "$RC"