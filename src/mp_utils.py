# src/mp_utils.py
from __future__ import annotations

import os
import sys
import subprocess
from typing import List, Tuple, Optional

ENV_CHILD = "CODENAMES_CHILD"
ENV_SHARD = "CODENAMES_SHARD_ID"
ENV_NSHARDS = "CODENAMES_NUM_SHARDS"


def parse_visible_gpus() -> List[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [x.strip() for x in cvd.split(",") if x.strip()]
    try:
        import torch

        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return [str(i) for i in range(n)]


def is_child_process() -> bool:
    return os.environ.get(ENV_CHILD, "") == "1"


def child_shard_info() -> Tuple[int, int]:
    sid = int(os.environ.get(ENV_SHARD, "0"))
    n = int(os.environ.get(ENV_NSHARDS, "1"))
    return sid, n


def launch_children(
    module: str,
    argv: List[str],
    num_procs: int,
    *,
    shard_base: int = 0,
    total_shards: Optional[int] = None,
) -> None:
    """
    Launches `num_procs` child processes, each pinned to exactly one GPU (via CUDA_VISIBLE_DEVICES).

    Sharding semantics:
      - Each child gets a (global) shard id via CODENAMES_SHARD_ID
      - Total number of shards is CODENAMES_NUM_SHARDS
      - By default, shard ids are 0..num_procs-1 and total_shards=num_procs (old behavior)
      - If you pass shard_base and total_shards, children get:
            shard_id = shard_base + local_rank
            num_shards = total_shards
        This composes cleanly with SLURM job arrays (job-level shards) + intra-job GPU shards.
    """
    gpus = parse_visible_gpus()
    if len(gpus) < num_procs:
        raise RuntimeError(f"Need {num_procs} visible GPUs, but only see {len(gpus)}: {gpus}")

    base = int(shard_base)
    total = int(total_shards) if total_shards is not None else int(num_procs)
    if total < 1:
        total = 1

    procs: List[subprocess.Popen] = []
    for local_sid in range(num_procs):
        env = os.environ.copy()
        env[ENV_CHILD] = "1"
        env[ENV_SHARD] = str(base + local_sid)
        env[ENV_NSHARDS] = str(total)

        # Pin each child to a single GPU
        env["CUDA_VISIBLE_DEVICES"] = gpus[local_sid]

        cmd = [sys.executable, "-m", module, *argv]
        procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    for p in procs:
        rc = rc or p.wait()

    if rc != 0:
        raise SystemExit(rc)