import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def _expand_templates(obj: Any, mapping: Dict[str, Any]) -> Any:
    """
    Recursively replace occurrences of {iter}, {prev}, {root}, {iter_suffix}, {prev_suffix},
    {iter_dir}, {prev_dir} in strings.
    """
    if isinstance(obj, dict):
        return {k: _expand_templates(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_templates(v, mapping) for v in obj]
    if isinstance(obj, str):
        s = obj
        for k, v in mapping.items():
            s = s.replace("{" + k + "}", str(v))
        return s
    return obj


def _parse_iter_spec(raw_iter: Any) -> Dict[str, Any]:
    """
    Supports:
      iter:
        n: 0
        use_trained: true
        root: "outputs/dpo"

    and also:
      iter: 2
      iter: "2"
    """
    n = 0
    use_trained = False
    root = "outputs/dpo"

    if raw_iter is None:
        pass
    elif isinstance(raw_iter, dict):
        n = int(raw_iter.get("n", 0))
        use_trained = bool(raw_iter.get("use_trained", False))
        root = str(raw_iter.get("root", "outputs/dpo"))
    elif isinstance(raw_iter, int):
        n = int(raw_iter)
    elif isinstance(raw_iter, str):
        s = raw_iter.strip()
        try:
            n = int(s)
        except Exception as e:
            raise ValueError(
                f"Unsupported iter value {raw_iter!r}. "
                "Use a mapping or an integer."
            ) from e
    else:
        raise ValueError(
            f"Unsupported iter value type: {type(raw_iter)}. "
            "Use a mapping or an integer."
        )

    if n < 0:
        raise ValueError(f"iter.n must be >= 0 (got {n})")

    return {
        "n": int(n),
        "use_trained": bool(use_trained),
        "root": root,
    }


def resolve_training_objective(cfg: Dict[str, Any]) -> str:
    training = cfg.get("training", {}) or {}
    objective = str(training.get("objective", "dpo")).strip().lower() or "dpo"
    if objective not in {"sft", "dpo"}:
        raise ValueError(f"Unsupported training objective: {objective!r}. Expected 'sft' or 'dpo'.")

    cfg.setdefault("training", {})
    cfg["training"]["objective"] = objective
    return objective


def resolve_training_plan(cfg: Dict[str, Any]) -> Dict[str, str]:
    objective = resolve_training_objective(cfg)
    tcfg = cfg.get("training", {}) or {}
    paths = cfg.get("paths", {}) or {}

    if objective == "sft":
        sft_source = str(tcfg.get("sft_source", "turns_raw")).strip().lower()
        if sft_source not in {"turns_raw", "turns"}:
            raise ValueError(
                f"Unsupported training.sft_source: {sft_source!r}. Expected 'turns_raw' or 'turns'."
            )

        if sft_source == "turns":
            data_path = str(paths.get("turns_path", ""))
        else:
            data_path = str(paths.get("turns_raw_path", ""))

        out_dir = str(tcfg.get("sft_output_adapter_dir", tcfg.get("output_adapter_dir", "")))
        module = "src.train_sft"
    else:
        sft_source = ""
        data_path = str(paths.get("dpo_pairs_path", ""))
        out_dir = str(tcfg.get("output_adapter_dir", ""))
        module = "src.train_trl_dpo"

    return {
        "objective": objective,
        "module": module,
        "data_path": data_path,
        "out_dir": out_dir,
        "sft_source": sft_source,
    }


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Loads YAML and applies iteration templating + objective normalization.

    Supported YAML forms:
      iter:
        n: 0
        use_trained: true
        root: outputs/dpo

      iter: 2

    Supported placeholders in YAML strings:
      {iter}        -> current iteration (int)
      {prev}        -> previous iteration index, clamped at 0
      {root}        -> iter.root
      {iter_suffix} -> "" if iter==0 else "_iter{iter}"
      {prev_suffix} -> "" if prev==0 else "_iter{prev}"
      {iter_dir}    -> "" if iter==0 else "/iter{iter}"
      {prev_dir}    -> "" if prev==0 else "/iter{prev}"

    Environment override:
      ITER=<int>    -> overrides iter.n

    Effective adapter behavior:
      - When iter.n > 0 and iter.use_trained=true, previous-iteration adapters stay enabled
        for rollouts and training initialization.
      - Otherwise rollout_adapter_dir / init_adapter_dir / sft_init_adapter_dir are disabled.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML config must be a mapping/dict: {path}")

    cfg: Dict[str, Any] = obj

    raw_iter = cfg.get("iter", {})
    parsed = _parse_iter_spec(raw_iter)

    iter_n = int(parsed["n"])
    requested_use_trained = bool(parsed["use_trained"])
    root = str(parsed["root"])

    env_iter = os.environ.get("ITER")
    if env_iter is not None:
        try:
            iter_n = int(env_iter)
        except Exception as e:
            raise ValueError(
                f"Invalid ITER environment override {env_iter!r}. Use an integer."
            ) from e

    if iter_n < 0:
        raise ValueError(f"iter.n must be >= 0 after overrides (got {iter_n})")

    prev = max(0, iter_n - 1)

    iter_suffix = "" if iter_n == 0 else f"_iter{iter_n}"
    prev_suffix = "" if prev == 0 else f"_iter{prev}"
    iter_dir = "" if iter_n == 0 else f"/iter{iter_n}"
    prev_dir = "" if prev == 0 else f"/iter{prev}"

    mapping = {
        "iter": iter_n,
        "prev": prev,
        "root": root,
        "iter_suffix": iter_suffix,
        "prev_suffix": prev_suffix,
        "iter_dir": iter_dir,
        "prev_dir": prev_dir,
    }

    cfg = _expand_templates(cfg, mapping)
    objective = resolve_training_objective(cfg)

    effective_use_trained = bool(iter_n > 0 and requested_use_trained)

    cfg["iter"] = {
        "n": iter_n,
        "prev": prev,
        "use_trained": effective_use_trained,
        "root": root,
    }

    if not effective_use_trained:
        cfg.setdefault("inference", {})
        cfg["inference"]["rollout_adapter_dir"] = None
        cfg.setdefault("training", {})
        cfg["training"]["init_adapter_dir"] = None
        if "sft_init_adapter_dir" in cfg["training"]:
            cfg["training"]["sft_init_adapter_dir"] = None

    cfg["training"]["objective"] = objective
    return cfg


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
    try:
        import numpy as np  # lazy import
        np.random.seed(seed)
    except Exception:
        pass
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
