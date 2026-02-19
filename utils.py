import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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