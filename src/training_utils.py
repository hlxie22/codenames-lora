from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def setup_node_local_hf_cache() -> Path:
    base_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    job_id = os.environ.get("SLURM_JOB_ID") or "nojob"

    local_hf_home = Path(base_tmp) / f"hf_{job_id}"
    local_datasets_cache = local_hf_home / "datasets"
    local_hub_cache = local_hf_home / "hub"

    local_datasets_cache.mkdir(parents=True, exist_ok=True)
    local_hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(local_hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(local_datasets_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(local_hub_cache)
    return local_datasets_cache


def world_size() -> int:
    for k in ("WORLD_SIZE", "SLURM_NTASKS"):
        v = os.environ.get(k)
        if v:
            with contextlib.suppress(Exception):
                return max(1, int(v))
    return 1


def should_force_nonreentrant(cfg: Dict[str, Any]) -> bool:
    tcfg = cfg.get("training", {}) or {}
    if not bool(tcfg.get("gradient_checkpointing", True)):
        return False
    return world_size() > 1


def safe_enable_input_grads(model: Any) -> None:
    with contextlib.suppress(Exception):
        model.enable_input_require_grads()


def safe_enable_gradient_checkpointing(model: Any, *, use_reentrant: bool) -> None:
    with contextlib.suppress(Exception):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )


def resolve_training_precision(tcfg: Dict[str, Any]) -> Tuple[bool, bool, torch.dtype | None]:
    use_bf16 = bool(tcfg.get("bf16", True)) and torch.cuda.is_available()
    use_fp16 = bool(tcfg.get("fp16", False)) and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    return use_bf16, use_fp16, dtype
