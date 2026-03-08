#!/bin/bash
#SBATCH --job-name=cn-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l4:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs

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

source slurm/common.sh

CONFIG="${CONFIG:-configs/default.yml}"

# ---- distributed via srun (1 task per node) + torch.distributed.run (no bash -c) ----
NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[$(date)] NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[$(date)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[$(date)] VENV python: $(which python)"

# Use the venv python resolved after sourcing common.sh
VENV_PY="$(which python)"
if [[ -z "$VENV_PY" || ! -x "$VENV_PY" ]]; then
  echo "ERROR: could not resolve an executable python after activating venv."
  exit 1
fi

export NNODES GPUS_PER_NODE MASTER_ADDR MASTER_PORT CONFIG VENV_PY

# Some sites set these and it can confuse launcher tooling; keep your original behavior.
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

set +e
srun --export=ALL \
     --ntasks="$NNODES" \
     --ntasks-per-node=1 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
     --kill-on-bad-exit=1 \
     "$VENV_PY" -c '
import os, sys, subprocess

# Per-node values provided by Slurm for each step task:
node_rank = os.environ.get("SLURM_NODEID", "0")

nnodes = os.environ["NNODES"]
gpus_per_node = os.environ["GPUS_PER_NODE"]
master_addr = os.environ["MASTER_ADDR"]
master_port = os.environ["MASTER_PORT"]
cfg = os.environ["CONFIG"]

# Use the same venv python for the torch.distributed.run launcher
py = os.environ.get("VENV_PY", sys.executable)

# Slightly cleaner: avoid nested quoting in the command list
rdzv_id = os.environ.get("SLURM_JOB_ID", "0")

print(f"[{os.uname().nodename}] node_rank={node_rank} nnodes={nnodes} gpus_per_node={gpus_per_node}", flush=True)

cmd = [
  py, "-m", "torch.distributed.run",
  f"--nnodes={nnodes}",
  f"--nproc_per_node={gpus_per_node}",
  f"--node_rank={node_rank}",
  "--rdzv_backend=c10d",
  f"--rdzv_id={rdzv_id}",
  f"--rdzv_endpoint={master_addr}:{master_port}",
  "-m", "src.train_trl_dpo",
  "--config", cfg,
]

raise SystemExit(subprocess.call(cmd))
' &
MAIN_PID=$!
wait "$MAIN_PID"
RC=$?
set -e
exit "$RC"