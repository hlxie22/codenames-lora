# src/mp_utils.py
from __future__ import annotations
import os, sys, subprocess
from dataclasses import dataclass
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

def launch_children(module: str, argv: List[str], num_procs: int) -> None:
    gpus = parse_visible_gpus()
    if len(gpus) < num_procs:
        raise RuntimeError(f"Need {num_procs} visible GPUs, but only see {len(gpus)}: {gpus}")

    procs: List[subprocess.Popen] = []
    for sid in range(num_procs):
        env = os.environ.copy()
        env[ENV_CHILD] = "1"
        env[ENV_SHARD] = str(sid)
        env[ENV_NSHARDS] = str(num_procs)
        env["CUDA_VISIBLE_DEVICES"] = gpus[sid]
        cmd = [sys.executable, "-m", module, *argv]
        procs.append(subprocess.Popen(cmd, env=env))

    rc = 0
    for p in procs:
        rc = rc or p.wait()
    if rc != 0:
        raise SystemExit(rc)