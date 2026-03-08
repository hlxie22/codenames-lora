#!/bin/bash
# Common SLURM bootstrap: venv + deps + caches
# Intended to be: source slurm/common.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ---- choose python (allow override) ----
PYTHON_VERSION="${PYTHON_VERSION:-}"
PYTHON="${PYTHON:-}"

if [[ -z "${PYTHON}" && -n "${PYTHON_VERSION}" ]]; then
  PYTHON="$(command -v "python${PYTHON_VERSION}" || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(command -v python3.12 || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: Could not find a usable python on PATH (python3.12/python3)."
  exit 1
fi

echo "[$(date)] Using PYTHON=$PYTHON"
"$PYTHON" -V

# Optional strict pin: set REQUIRE_PYTHON_MINOR=3.12 (or 3.10, etc)
REQUIRE_PYTHON_MINOR="${REQUIRE_PYTHON_MINOR:-}"
if [[ -n "${REQUIRE_PYTHON_MINOR}" ]]; then
  "$PYTHON" - <<PY
import sys
want = tuple(map(int, "${REQUIRE_PYTHON_MINOR}".split(".")))
got = sys.version_info[:2]
assert got == want, f"Python {got} != required {want}"
print("Python version OK:", sys.version)
PY
fi

VENV_DIR="${VENV_DIR:-$SLURM_SUBMIT_DIR/.venv}"
REQ_FILE="${REQ_FILE:-$SLURM_SUBMIT_DIR/requirements.txt}"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: requirements file not found: $REQ_FILE"
  exit 1
fi

REQ_HASH="$("$PYTHON" - <<PY
import hashlib, pathlib
p = pathlib.Path("$REQ_FILE")
print(hashlib.sha256(p.read_bytes()).hexdigest())
PY
)"
DEPS_SENTINEL="$VENV_DIR/.deps.${REQ_HASH}"
BOOT_SENTINEL="$VENV_DIR/.bootstrap.done"

# Serialize venv mutations across array tasks
LOCKDIR="$SLURM_SUBMIT_DIR/.venv_install.lock"
while ! mkdir "$LOCKDIR" 2>/dev/null; do
  echo "[$(date)] Another task is setting up the venv; waiting..."
  sleep 5
done
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

# Create venv if missing
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  rm -rf "$VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
echo "[$(date)] Activated venv: $(which python)"
python -V

# Bootstrap pip tooling once (PIN setuptools for vLLM)
SETUPTOOLS_SPEC="${SETUPTOOLS_SPEC:-setuptools>=77.0.3,<80}"
if [[ ! -f "$BOOT_SENTINEL" ]]; then
  python -m pip install -U pip wheel "$SETUPTOOLS_SPEC"
  touch "$BOOT_SENTINEL"
fi

# Install deps once per requirements hash
if [[ ! -f "$DEPS_SENTINEL" ]]; then
  echo "[$(date)] Installing deps from $REQ_FILE ..."
  python -m pip install -r "$REQ_FILE"

  if [[ -f pyproject.toml || -f setup.py || -f setup.cfg ]]; then
    python -m pip install -e .
  fi

  touch "$DEPS_SENTINEL"
  echo "[$(date)] Deps installed; wrote $DEPS_SENTINEL"
else
  echo "[$(date)] Deps already installed for requirements hash; skipping pip install."
fi

# If repo isn't installable, ensure imports work
if [[ ! -f pyproject.toml && ! -f setup.py && ! -f setup.cfg ]]; then
  export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"
fi

trap - EXIT
rmdir "$LOCKDIR" 2>/dev/null || true

# ---- caches (shared; override if you want) ----
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

# ---- misc knobs ----
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

export VLLM_USE_V1="${VLLM_USE_V1:-0}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True