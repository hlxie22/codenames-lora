#!/bin/bash
#SBATCH --job-name=codenames
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

module load miniforge/24.3.0-0

# --- Activate conda ---
source /orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh
conda activate codenames-lora

export VLLM_ATTENTION_BACKEND=TRITON
# Prefer precompiled vLLM libs
export VLLM_USE_PRECOMPILED=1

# Cache to scratch
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# --- Run experiment ---

set -e
python -m src.build_vocab --out data/vocab.txt --manifest data/vocab_manifest.json
# python -m src.make_boards --config configs/default.yaml
# python -m src.generate_sft_data --config configs/default.yaml
# torchrun --standalone --nproc_per_node=2 -m src.train_lora_sft --config configs/default.yaml
# python -m src.eval --config configs/default.yaml --mode baseline --out outputs/eval/baseline_run1
# python -m src.eval --config configs/default.yaml --mode sft --out outputs/eval/sft_run1