# src/io_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from .utils import read_jsonl


def shard_tag(shard_id: int, num_shards: int) -> str:
    return f"shard{shard_id:02d}-of-{num_shards:02d}"


def shard_path_insert_before_suffix(path: str | Path, shard_id: int, num_shards: int) -> Path:
    """
    per_board.jsonl -> per_board.shard00-of-02.jsonl
    (matches eval.py's existing convention)
    """
    p = Path(path)
    tag = shard_tag(shard_id, num_shards)
    if p.suffix:
        return p.with_name(f"{p.stem}.{tag}{p.suffix}")
    return p.with_name(f"{p.name}.{tag}")


def shard_path_append_to_suffix(path: str | Path, shard_id: int, num_shards: int) -> Path:
    """
    foo.jsonl -> foo.jsonl.shard00-of-02
    (matches generate_sft_data.py's existing convention)
    """
    p = Path(path)
    tag = shard_tag(shard_id, num_shards)
    if p.suffix:
        return p.with_suffix(p.suffix + f".{tag}")
    return p.with_name(f"{p.name}.{tag}")


def merge_jsonl_shards(
    shard_paths: Iterable[str | Path],
    *,
    sort_key: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    for sp in shard_paths:
        p = Path(sp)
        if p.exists():
            combined.extend(read_jsonl(p))
    if sort_key is not None:
        combined.sort(key=sort_key)
    return combined