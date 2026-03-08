#!/bin/bash
#SBATCH --job-name=cn-dpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# -----------------------------
# User-configurable knobs
# -----------------------------
CONFIG="${CONFIG:-configs/default.yml}"
CONDA_ENV="${CONDA_ENV:-codenames-lora}"

# Optional: set to 1 to disable custom epoch eval callback
NO_EPOCH_EVAL="${NO_EPOCH_EVAL:-0}"

# Optional rendezvous port override
MASTER_PORT="${MASTER_PORT:-29500}"

# -----------------------------
# Requeue handling (USR1)
# -----------------------------
REQUEUE_SENT=0
MAIN_PID=""
handle_usr1() {
  if (( REQUEUE_SENT == 1 )); then return 0; fi
  REQUEUE_SENT=1
  echo "[$(date)] USR1 received: requesting requeue..."
  scontrol requeue "$SLURM_JOB_ID" || true
  if [[ -n "${MAIN_PID}" ]] && kill -0 "${MAIN_PID}" 2>/dev/null; then
    kill -TERM -- -"${MAIN_PID}" 2>/dev/null || kill -TERM "${MAIN_PID}" 2>/dev/null
  fi
  exit 0
}
trap handle_usr1 USR1

# -----------------------------
# Conda activation
# -----------------------------

module -q load miniforge/24.3.0-0

CONDA_BASE="/orcd/software/core/001/pkg/miniforge/24.3.0-0"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: conda.sh not found at: $CONDA_BASE/etc/profile.d/conda.sh" >&2
  echo "Edit CONDA_BASE/module load lines to match your cluster." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[$(date)] Activating conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

echo "[$(date)] Python: $(command -v python)"
python -V

# -----------------------------
# Shared caches / misc knobs
# -----------------------------
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

# Some sites set these and it can confuse launcher tooling; keep consistent behavior.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

# -----------------------------
# Validate required inputs exist (from previous steps)
# -----------------------------
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG not found: $CONFIG" >&2
  exit 1
fi

DPO_PATH="$(python - <<'PY'
import os, sys
import yaml
cfg_path = os.environ.get("CONFIG", "configs/default.yml")
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
p = cfg.get("paths", {}).get("dpo_pairs_path")
if not p:
  print("", end="")
  sys.exit(0)
print(p)
PY
)"

if [[ -z "$DPO_PATH" ]]; then
  echo "ERROR: paths.dpo_pairs_path not found in $CONFIG" >&2
  exit 1
fi

if [[ ! -s "$DPO_PATH" ]]; then
  echo "ERROR: DPO pairs file missing or empty: $DPO_PATH" >&2
  echo "This run.slurm assumes previous steps already produced it on the shared filesystem." >&2
  exit 1
fi

echo "[$(date)] Found DPO pairs: $DPO_PATH"

# -----------------------------
# Multi-node torchrun via srun
# -----------------------------
NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

echo "[$(date)] NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[$(date)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[$(date)] CONFIG=$CONFIG"

export NNODES GPUS_PER_NODE MASTER_ADDR MASTER_PORT CONFIG NO_EPOCH_EVAL

set +e
srun --export=ALL \
     --ntasks="$NNODES" \
     --ntasks-per-node=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
     --kill-on-bad-exit=1 \
     python - <<'PY' &
import os, sys, subprocess

node_rank = os.environ.get("SLURM_NODEID", "0")

nnodes = os.environ["NNODES"]
gpus_per_node = os.environ["GPUS_PER_NODE"]
master_addr = os.environ["MASTER_ADDR"]
master_port = os.environ["MASTER_PORT"]
cfg = os.environ["CONFIG"]
no_epoch_eval = os.environ.get("NO_EPOCH_EVAL", "0") == "1"

rdzv_id = os.environ.get("SLURM_JOB_ID", "0")

cmd = [
  sys.executable, "-m", "torch.distributed.run",
  f"--nnodes={nnodes}",
  f"--nproc_per_node={gpus_per_node}",
  f"--node_rank={node_rank}",
  "--rdzv_backend=c10d",
  f"--rdzv_id={rdzv_id}",
  f"--rdzv_endpoint={master_addr}:{master_port}",
  "-m", "src.train_trl_dpo",
  "--config", cfg,
]
if no_epoch_eval:
  cmd.append("--no-epoch-eval")

print(f"[{os.uname().nodename}] node_rank={node_rank} launching: {' '.join(cmd)}", flush=True)
raise SystemExit(subprocess.call(cmd))
PY

MAIN_PID=$!
wait "$MAIN_PID"
RC=$?
set -e

exit "$RC"