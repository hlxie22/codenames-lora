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
export CONFIG

CONDA_ENV="${CONDA_ENV:-codenames-lora}"

NO_EPOCH_EVAL="${NO_EPOCH_EVAL:-0}"
RUN_POST_EVAL="${RUN_POST_EVAL:-1}"
SKIP_TRAIN_IF_DONE="${SKIP_TRAIN_IF_DONE:-1}"
POST_EVAL_WHICH="${POST_EVAL_WHICH:-both}"
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
# Email notifications (manual; skips requeue exits)
# -----------------------------
MAIL_TO="henryxie@college.harvard.edu"
MAIL_SUBJECT_PREFIX="slurm"
SECONDS=0
START_TS="$(date -Iseconds)"

send_mail() {
  local subject="$1"
  local body="$2"

  [[ -n "$MAIL_TO" ]] || return 0

  if command -v mail >/dev/null 2>&1; then
    printf "%s\n" "$body" | mail -s "$subject" "$MAIL_TO"
  elif command -v mailx >/dev/null 2>&1; then
    printf "%s\n" "$body" | mailx -s "$subject" "$MAIL_TO"
  elif command -v sendmail >/dev/null 2>&1; then
    {
      printf "To: %s\n" "$MAIL_TO"
      printf "Subject: %s\n" "$subject"
      printf "\n%s\n" "$body"
    } | sendmail -t
  else
    echo "[$(date)] WARNING: no mail/mailx/sendmail found; cannot send email." >&2
    return 1
  fi
}

notify_on_exit() {
  local rc=$?

  if (( ${REQUEUE_SENT:-0} == 1 )); then
    return 0
  fi

  set +e

  local end_ts dur out_log err_log status subject body
  end_ts="$(date -Iseconds)"
  dur="$SECONDS"
  out_log="logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-unknown}.out"
  err_log="logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-unknown}.err"

  if [[ "$rc" -eq 0 ]]; then
    status="COMPLETED"
  else
    status="FAILED (rc=$rc)"
  fi

  subject="${MAIL_SUBJECT_PREFIX}: ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-} ${status}"

  body=$(
    cat <<EOF2
Job notification: $status

JobName:   ${SLURM_JOB_NAME:-}
JobID:     ${SLURM_JOB_ID:-}
User:      ${SLURM_JOB_USER:-$USER}
Cluster:   ${SLURM_CLUSTER_NAME:-}
Partition: ${SLURM_JOB_PARTITION:-}
Nodes:     ${SLURM_JOB_NODELIST:-}
Workdir:   ${SLURM_SUBMIT_DIR:-$(pwd)}

Start:     $START_TS
End:       $end_ts
Duration:  ${dur}s

CONFIG:    ${CONFIG:-}
Stdout:    $out_log
Stderr:    $err_log
EOF2
  )

  send_mail "$subject" "$body" >/dev/null 2>&1 || true
  return 0
}

trap notify_on_exit EXIT

# -----------------------------
# Conda activation (cluster-specific)
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

# -----------------------------
# Validate required inputs exist
# -----------------------------
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG not found: $CONFIG" >&2
  exit 1
fi

read -r TRAIN_OBJECTIVE TRAIN_MODULE TRAIN_DATA_PATH TRAIN_OUT_DIR < <(
  python - <<'PY'
import os
from src.utils import load_yaml, resolve_training_plan

cfg = load_yaml(os.environ.get("CONFIG", "configs/default.yml"))
plan = resolve_training_plan(cfg)
print(f"{plan['objective']}\t{plan['module']}\t{plan['data_path']}\t{plan['out_dir']}")
PY
)

if [[ -z "$TRAIN_DATA_PATH" ]]; then
  echo "ERROR: could not resolve training data path from $CONFIG" >&2
  exit 1
fi
if [[ -z "$TRAIN_OUT_DIR" ]]; then
  echo "ERROR: could not resolve training output dir from $CONFIG" >&2
  exit 1
fi

if [[ ! -s "$TRAIN_DATA_PATH" ]]; then
  echo "ERROR: training source file missing or empty: $TRAIN_DATA_PATH" >&2
  echo "This run.slurm assumes previous steps already produced it on the shared filesystem." >&2
  exit 1
fi

echo "[$(date)] TRAIN_OBJECTIVE=$TRAIN_OBJECTIVE"
echo "[$(date)] TRAIN_MODULE=$TRAIN_MODULE"
echo "[$(date)] TRAIN_DATA_PATH=$TRAIN_DATA_PATH"
echo "[$(date)] TRAIN_OUT_DIR=$TRAIN_OUT_DIR"
echo "[$(date)] ITER=${ITER:-<unset>}"

export TRAIN_OBJECTIVE TRAIN_MODULE TRAIN_OUT_DIR

# -----------------------------
# Decide whether training is already done
# -----------------------------
TRAIN_DONE=0
if [[ -s "$TRAIN_OUT_DIR/adapter_model.safetensors" && -s "$TRAIN_OUT_DIR/adapter_config.json" ]]; then
  TRAIN_DONE=1
fi

POST_EVAL_ROOT="${POST_EVAL_ROOT:-$TRAIN_OUT_DIR/post_eval}"
EVAL_BASELINE_DIR="$POST_EVAL_ROOT/baseline"
EVAL_TRAINED_DIR="$POST_EVAL_ROOT/trained"

NEED_BASELINE=0
NEED_TRAINED=0
if [[ "$POST_EVAL_WHICH" == "both" || "$POST_EVAL_WHICH" == "baseline" ]]; then
  [[ -s "$EVAL_BASELINE_DIR/metrics.json" ]] || NEED_BASELINE=1
fi
if [[ "$POST_EVAL_WHICH" == "both" || "$POST_EVAL_WHICH" == "trained" ]]; then
  [[ -s "$EVAL_TRAINED_DIR/metrics.json" ]] || NEED_TRAINED=1
fi

EVAL_NEEDED=0
if (( NEED_BASELINE == 1 || NEED_TRAINED == 1 )); then
  EVAL_NEEDED=1
fi

# -----------------------------
# Helper: run post-eval (baseline + trained)
# -----------------------------
run_post_eval() {
  mkdir -p "$POST_EVAL_ROOT"

  GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}"
  GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
  if [[ "$GPUS_PER_NODE" -lt 1 ]]; then
    GPUS_PER_NODE=1
  fi
  echo "[$(date)] Post-eval will use GPUS_PER_NODE=$GPUS_PER_NODE"

  EVAL_CONFIG="$POST_EVAL_ROOT/config_eval_autogen.yaml"
  export EVAL_CONFIG GPUS_PER_NODE

  python - <<'PY'
import os, yaml
from src.utils import load_yaml

cfg_path = os.environ.get("CONFIG", "configs/default.yml")
out_path = os.environ["EVAL_CONFIG"]
gpus = int(os.environ.get("GPUS_PER_NODE", "1"))

cfg = load_yaml(cfg_path)
objective = os.environ.get("TRAIN_OBJECTIVE", "dpo").strip().lower()
train_out_dir = os.environ.get("TRAIN_OUT_DIR", "").strip()

if objective == "sft" and train_out_dir:
    tcfg = cfg.get("training", {}) or {}
    # src.eval --mode trained reads training.output_adapter_dir.
    # Point it at the actual trained adapter dir for SFT runs.
    tcfg["output_adapter_dir"] = train_out_dir
    cfg["training"] = tcfg

inf = cfg.get("inference", {}) or {}
bs = int(os.environ.get("EVAL_BATCH_SIZE", "4"))
inf["batch_size"] = bs
inf["num_processes"] = max(1, gpus)

v = inf.get("vllm", {}) or {}
v["tensor_parallel_size"] = int(v.get("tensor_parallel_size", 1))
if inf.get("backend", "hf") == "vllm":
  v["tensor_parallel_size"] = 1
inf["vllm"] = v
cfg["inference"] = inf

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"Wrote eval config -> {out_path} (inference.num_processes={inf['num_processes']})")
PY

  if (( NEED_BASELINE == 1 )); then
    echo "[$(date)] Running baseline eval -> $EVAL_BASELINE_DIR"
    python -m src.eval --config "$EVAL_CONFIG" --mode baseline --out "$EVAL_BASELINE_DIR"
  else
    echo "[$(date)] Baseline eval already exists: $EVAL_BASELINE_DIR/metrics.json"
  fi

  if (( NEED_TRAINED == 1 )); then
    echo "[$(date)] Running trained eval -> $EVAL_TRAINED_DIR"
    python -m src.eval --config "$EVAL_CONFIG" --mode trained --out "$EVAL_TRAINED_DIR"
  else
    echo "[$(date)] Trained eval already exists: $EVAL_TRAINED_DIR/metrics.json"
  fi
}

# -----------------------------
# Skip-to-eval mode
# -----------------------------
if (( TRAIN_DONE == 1 )) && (( SKIP_TRAIN_IF_DONE == 1 )) && (( RUN_POST_EVAL == 1 )) && (( EVAL_NEEDED == 1 )); then
  echo "[$(date)] Training artifacts detected, eval missing -> skipping training and running post-eval."
  run_post_eval
  echo "[$(date)] Done (eval-only)."
  exit 0
fi

if (( TRAIN_DONE == 1 )) && (( SKIP_TRAIN_IF_DONE == 1 )) && (( RUN_POST_EVAL == 1 )) && (( EVAL_NEEDED == 0 )); then
  echo "[$(date)] Training artifacts detected and eval already done -> nothing to do."
  exit 0
fi

# -----------------------------
# Multi-node torchrun via srun (training)
# -----------------------------
NNODES="${SLURM_NNODES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-2}}"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

echo "[$(date)] NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE"
echo "[$(date)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[$(date)] CONFIG=$CONFIG"
echo "[$(date)] TRAIN_OBJECTIVE=$TRAIN_OBJECTIVE TRAIN_MODULE=$TRAIN_MODULE"
echo "[$(date)] Starting training..."

export NNODES GPUS_PER_NODE MASTER_ADDR MASTER_PORT NO_EPOCH_EVAL TRAIN_MODULE

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
cfg = os.environ.get("CONFIG", "configs/default.yml")
no_epoch_eval = os.environ.get("NO_EPOCH_EVAL", "0") == "1"
train_module = os.environ["TRAIN_MODULE"]

rdzv_id = os.environ.get("SLURM_JOB_ID", "0")

cmd = [
  sys.executable, "-m", "torch.distributed.run",
  f"--nnodes={nnodes}",
  f"--nproc_per_node={gpus_per_node}",
  f"--node_rank={node_rank}",
  "--rdzv_backend=c10d",
  f"--rdzv_id={rdzv_id}",
  f"--rdzv_endpoint={master_addr}:{master_port}",
  "-m", train_module,
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

if [[ "$RC" -ne 0 ]]; then
  echo "[$(date)] Training exited with RC=$RC"
  exit "$RC"
fi

echo "[$(date)] Training finished OK."

# -----------------------------
# Post-eval after training (optional)
# -----------------------------
NEED_BASELINE=0
NEED_TRAINED=0
if [[ "$POST_EVAL_WHICH" == "both" || "$POST_EVAL_WHICH" == "baseline" ]]; then
  [[ -s "$EVAL_BASELINE_DIR/metrics.json" ]] || NEED_BASELINE=1
fi
if [[ "$POST_EVAL_WHICH" == "both" || "$POST_EVAL_WHICH" == "trained" ]]; then
  [[ -s "$EVAL_TRAINED_DIR/metrics.json" ]] || NEED_TRAINED=1
fi

if (( RUN_POST_EVAL == 1 )) && (( NEED_BASELINE == 1 || NEED_TRAINED == 1 )); then
  echo "[$(date)] Running post-eval..."
  run_post_eval
fi

echo "[$(date)] Done."
exit 0
