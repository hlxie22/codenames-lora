import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

def now_iso() -> str:
    # Nice for logs/progress: 2026-02-20T13:45:12
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def save_progress(
    progress_path: str | Path,
    *,
    done: int,
    total: int,
    last_example_id: Optional[str] = None,
    last_board_id: Optional[str] = None,
    mean_reward: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Atomic JSON progress writer (safe for crashes).
    """
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "timestamp": now_iso(),
        "done": int(done),
        "total": int(total),
        "pct": float(done / total) if total else 0.0,
        "last_example_id": last_example_id,
        "last_board_id": last_board_id,
        "mean_reward_running": float(mean_reward) if mean_reward is not None else None,
    }
    if extra:
        payload.update(extra)

    tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, progress_path)

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML config must be a mapping/dict: {path}")
    return obj


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: str | Path, obj: Dict[str, Any], *, do_fsync: bool = True) -> None:
    path = Path(path)
    if path.parent and str(path.parent) != "":
        ensure_dir(path.parent)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        if do_fsync:
            os.fsync(f.fileno())

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def try_get_git_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def save_config_snapshot(cfg: Dict[str, Any], out_dir: str | Path, filename: str = "config_snapshot.yaml") -> None:
    out_dir = ensure_dir(out_dir)
    p = out_dir / filename
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def save_run_meta(out_dir: str | Path, command: List[str] | None = None, extra: Dict[str, Any] | None = None) -> None:
    out_dir = ensure_dir(out_dir)
    meta: Dict[str, Any] = {
        "timestamp": now_ts(),
        "git_hash": try_get_git_hash(),
        "command": command,
    }
    if extra:
        meta.update(extra)
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)