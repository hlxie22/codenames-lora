#!/bin/bash
#SBATCH --job-name=cn-pipeline
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

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
python - <<'PY'
import os, re

path = os.environ.get("CONFIG_MAIN", "configs/default.yml")

num = None
lines = open(path, "r", encoding="utf-8").read().splitlines()

# Minimal indentation-based parse for:
# training:
#   distributed:
#     num_nodes: 3
for i, line in enumerate(lines):
    if re.match(r'^\s*training\s*:\s*$', line):
        for j in range(i + 1, len(lines)):
            lj = lines[j]
            if lj.strip() and not lj.startswith(" "):
                break
            if re.match(r'^\s{2}distributed\s*:\s*$', lj):
                for k in range(j + 1, len(lines)):
                    lk = lines[k]
                    if re.match(r'^\s{2}\S', lk):
                        break
                    m = re.match(r'^\s{4}num_nodes\s*:\s*(\d+)\s*$', lk)
                    if m:
                        num = int(m.group(1))
                        break
                break
        break

print(num if num is not None else 1)
PY
)}"

# ---- resolve artifact paths from CONFIG_MAIN ----
# IMPORTANT: print all values on ONE line so read works.
read -r TURNS_RAW_PATH TURNS_PATH DPO_PAIRS_PATH < <(python - <<'PY'
import os, yaml
cfg_path = os.environ.get("CONFIG_MAIN", "configs/default.yml")
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) or {}
paths = cfg.get("paths", {}) or {}
print(f"{paths.get('turns_raw_path','')}\t{paths.get('turns_path','')}\t{paths.get('dpo_pairs_path','')}")
PY
)

if [[ -z "${DPO_PAIRS_PATH}" ]]; then
  echo "ERROR: Could not read paths.dpo_pairs_path from CONFIG_MAIN=$CONFIG_MAIN"
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
AUTO_SHARDS="$(python - <<'PY'
import os, re, glob, yaml
from pathlib import Path

cfg_path = os.environ.get("CONFIG_MAIN", "configs/default.yml")
cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8")) or {}
base = cfg.get("paths", {}).get("turns_raw_path", "")
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

# ---- submit training (depends on merge) ----
TRAIN_JOB=$(sbatch --parsable \
  --dependency=afterok:"$MERGE_JOB" \
  --nodes="$TRAIN_NODES" \
  --ntasks-per-node=1 \
  --export=ALL,CONFIG="$CONFIG_MAIN" \
  slurm/train.slurm)
echo "train job: $TRAIN_JOB"

echo "Pipeline submitted."