#!/bin/bash
#SBATCH --job-name=cn-pipeline
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# -------------------------------------------------------------------
# Minimal deps for pipeline's inline python:
#   - needs "import yaml" (PyYAML)
#
# IMPORTANT: install to node-local temp (NOT shared FS) to avoid pip
# rename/RECORD issues on NFS/Lustre.
# -------------------------------------------------------------------
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(command -v python3.12 || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: Could not find python3.12 or python3 on PATH." >&2
  exit 1
fi

# Node-local scratch if available
LOCAL_TMP="${PIPELINE_TMP:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}"
JOB_TAG="${SLURM_JOB_ID:-$$}"
PIPE_PY_PKGS="${PIPE_PY_PKGS:-$LOCAL_TMP/cn_pipeline_pkgs_${USER}_${JOB_TAG}}"
mkdir -p "$PIPE_PY_PKGS"

# Ensure pip exists (best-effort)
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  if "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    :
  else
    echo "ERROR: pip not available for $PYTHON and ensurepip failed." >&2
    echo "Set PYTHON=/path/to/python-with-pip or load your python module first." >&2
    exit 1
  fi
fi

# Install PyYAML into node-local target dir (retry once on transient FS issues)
install_pyyaml() {
  "$PYTHON" -m pip install \
    --disable-pip-version-check \
    --no-cache-dir \
    --upgrade \
    --target "$PIPE_PY_PKGS" \
    pyyaml
}

if ! install_pyyaml >/dev/null 2>&1; then
  echo "[pipeline] pip install pyyaml failed once; retrying..." >&2
  rm -rf "$PIPE_PY_PKGS"/*
  install_pyyaml >/dev/null
fi

# Ensure repo imports + target packages are visible
export PYTHONPATH="$PIPE_PY_PKGS:$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# Sanity check for logs
"$PYTHON" - <<'PY'
import sys, yaml
print("[pipeline] python =", sys.executable)
print("[pipeline] pyyaml =", yaml.__version__)
PY

# ---- knobs you change in one place ----
CONFIG_MAIN="${CONFIG_MAIN:-configs/default.yml}"
CONFIG_GEN="${CONFIG_GEN:-$CONFIG_MAIN}"
export CONFIG_MAIN CONFIG_GEN

# If 1: skip gen+merge when final artifacts already exist (non-empty)
SKIP_IF_READY="${SKIP_IF_READY:-1}"

# How many SLURM array tasks (jobs) for turn generation.
# Total global shards = GEN_ARRAY_TASKS * inference.num_processes (usually 4).
GEN_ARRAY_TASKS="${GEN_ARRAY_TASKS:-4}"
GEN_PROCS_PER_JOB="${GEN_PROCS_PER_JOB:-4}"  # must match inference.num_processes in CONFIG_GEN

CLEAN="${CLEAN:-0}"                 # sbatch --export=ALL,CLEAN=1 slurm/pipeline.slurm
ALLOW_MISSING="${ALLOW_MISSING:-0}" # set 1 to merge partial shards (debug only)

# ---- parse TRAIN_NODES from CONFIG_MAIN (training.distributed.num_nodes) ----
TRAIN_NODES="${TRAIN_NODES:-$(
"$PYTHON" - <<'PY'
import os
from src.utils import load_yaml
cfg = load_yaml(os.environ.get("CONFIG_MAIN","configs/default.yml"))
print(int(((cfg.get("training",{}) or {}).get("distributed",{}) or {}).get("num_nodes", 1)))
PY
)}"

# ---- resolve artifact paths from CONFIG_MAIN ----
# IMPORTANT: print all values on ONE line so read works.
read -r TURNS_RAW_PATH TURNS_PATH DPO_PAIRS_PATH < <("$PYTHON" - <<'PY'
import os
from src.utils import load_yaml
cfg_path = os.environ.get("CONFIG_MAIN", "configs/default.yml")
cfg = load_yaml(cfg_path)
paths = cfg.get("paths", {}) or {}
print(f"{paths.get('turns_raw_path','')}\t{paths.get('turns_path','')}\t{paths.get('dpo_pairs_path','')}")
PY
)

if [[ -z "${DPO_PAIRS_PATH}" ]]; then
  echo "ERROR: Could not read paths.dpo_pairs_path from CONFIG_MAIN=$CONFIG_MAIN" >&2
  exit 1
fi

echo "CONFIG_MAIN=$CONFIG_MAIN"
echo "CONFIG_GEN =$CONFIG_GEN"
echo "TRAIN_NODES=$TRAIN_NODES"
echo "GEN_ARRAY_TASKS=$GEN_ARRAY_TASKS"
echo "GEN_PROCS_PER_JOB=$GEN_PROCS_PER_JOB"
echo "CLEAN=$CLEAN"
echo "ALLOW_MISSING=$ALLOW_MISSING"
echo "SKIP_IF_READY=$SKIP_IF_READY"
echo "ITER=${ITER:-<unset>}"
echo "TURNS_RAW_PATH=$TURNS_RAW_PATH"
echo "TURNS_PATH=$TURNS_PATH"
echo "DPO_PAIRS_PATH=$DPO_PAIRS_PATH"

# ---- submit prep (vocab + boards) ----
PREP_JOB=$(sbatch --parsable --export=ALL,CLEAN="$CLEAN",CONFIG="$CONFIG_MAIN" slurm/prep.slurm)
echo "prep job: $PREP_JOB"

# ---- fast path: if DPO pairs already exist, skip gen+merge and go straight to train ----
if [[ "$SKIP_IF_READY" == "1" && "$CLEAN" != "1" && -s "$DPO_PAIRS_PATH" ]]; then
  echo "[pipeline] Found existing non-empty DPO pairs ($DPO_PAIRS_PATH). Skipping gen + merge."

  TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:"$PREP_JOB" \
    --nodes="$TRAIN_NODES" \
    --ntasks-per-node=1 \
    --export=ALL,CONFIG="$CONFIG_MAIN" \
    slurm/train.slurm)

  echo "train job: $TRAIN_JOB"
  echo "Pipeline submitted (skipped gen+merge)."
  exit 0
fi

# ---- optional: if all shard files exist, skip gen array and go straight to merge ----
AUTO_SHARDS="$("$PYTHON" - <<'PY'
import os, re, glob
from pathlib import Path
from src.utils import load_yaml

cfg = load_yaml(os.environ.get("CONFIG_MAIN","configs/default.yml"))
base = (cfg.get("paths", {}) or {}).get("turns_raw_path", "")
if not base:
    print("")
    raise SystemExit(0)

base = Path(base)
files = glob.glob(str(base) + ".shard*-of-*")
if not files:
    print("")
    raise SystemExit(0)

ns = set()
for f in files:
    m = re.search(r"\.shard(\d+)-of-(\d+)$", f)
    if m:
        ns.add(int(m.group(2)))

if len(ns) != 1:
    print("")
    raise SystemExit(0)

N = next(iter(ns))

# require all shard files to exist and be non-empty
ok = True
for sid in range(N):
    tag = f"shard{sid:02d}-of-{N:02d}"
    sp = base.with_suffix(base.suffix + f".{tag}")
    if (not sp.exists()) or sp.stat().st_size == 0:
        ok = False
        break

print(str(N) if ok else "")
PY
)"

if [[ "$SKIP_IF_READY" == "1" && "$CLEAN" != "1" && -n "$AUTO_SHARDS" ]]; then
  TOTAL_SHARDS="$AUTO_SHARDS"
  echo "[pipeline] Detected all $TOTAL_SHARDS shards already exist. Skipping gen array."

  MERGE_JOB=$(sbatch --parsable \
    --dependency=afterok:"$PREP_JOB" \
    --export=ALL,CONFIG="$CONFIG_MAIN",TOTAL_SHARDS="$TOTAL_SHARDS",ALLOW_MISSING="$ALLOW_MISSING" \
    slurm/merge_turns.slurm)
  echo "merge job: $MERGE_JOB (total_shards=$TOTAL_SHARDS)"

  TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:"$MERGE_JOB" \
    --nodes="$TRAIN_NODES" \
    --ntasks-per-node=1 \
    --export=ALL,CONFIG="$CONFIG_MAIN" \
    slurm/train.slurm)
  echo "train job: $TRAIN_JOB"

  echo "Pipeline submitted (skipped gen)."
  exit 0
fi

# ---- submit generation array (depends on prep) ----
GEN_JOB=$(sbatch --parsable \
  --dependency=afterok:"$PREP_JOB" \
  --array=0-$((GEN_ARRAY_TASKS-1)) \
  --export=ALL,CONFIG="$CONFIG_GEN",CLEAN="$CLEAN" \
  slurm/gen_turns_array.slurm)
echo "gen array job: $GEN_JOB (tasks=$GEN_ARRAY_TASKS)"

# ---- submit merge + filter + build_dpo_pairs (depends on ALL array tasks finishing) ----
TOTAL_SHARDS=$((GEN_ARRAY_TASKS * GEN_PROCS_PER_JOB))
MERGE_JOB=$(sbatch --parsable \
  --dependency=afterok:"$GEN_JOB" \
  --export=ALL,CONFIG="$CONFIG_MAIN",TOTAL_SHARDS="$TOTAL_SHARDS",ALLOW_MISSING="$ALLOW_MISSING" \
  slurm/merge_turns.slurm)
echo "merge job: $MERGE_JOB (total_shards=$TOTAL_SHARDS)"

# # ---- submit training (depends on merge) ----
# TRAIN_JOB=$(sbatch --parsable \
#   --dependency=afterok:"$MERGE_JOB" \
#   --nodes="$TRAIN_NODES" \
#   --ntasks-per-node=1 \
#   --export=ALL,CONFIG="$CONFIG_MAIN" \
#   slurm/train.slurm)
# echo "train job: $TRAIN_JOB"

echo "Pipeline submitted."