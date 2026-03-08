# src/merge_turns_shards.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import shard_path_append_to_suffix, merge_jsonl_shards
from .utils import load_yaml, write_jsonl
from .generate_turns_data import filter_examples, _cfg_turns_paths


def _sort_key(r: Dict[str, Any]) -> str:
    return str(r.get("board_id") or r.get("example_id") or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-shards", type=int, required=True, help="TOTAL global shards (jobs * procs_per_job)")
    ap.add_argument("--allow-missing", action="store_true", help="Merge whatever shards exist (useful mid-run).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    turns_path, turns_raw_path = _cfg_turns_paths(cfg)

    n = int(args.num_shards)
    if n < 1:
        raise ValueError("--num-shards must be >= 1")

    base_raw = Path(turns_raw_path)
    shard_paths = [shard_path_append_to_suffix(base_raw, sid, n) for sid in range(n)]

    existing = [p for p in shard_paths if p.exists()]
    missing = [p for p in shard_paths if not p.exists()]

    if missing and not args.allow_missing:
        raise SystemExit(
            f"Missing {len(missing)}/{n} shard files. "
            f"Pass --allow-missing to merge partial results."
        )

    print(f"Found {len(existing)}/{n} shard files. Merging...")

    combined: List[Dict[str, Any]] = merge_jsonl_shards(existing, sort_key=_sort_key)

    # Dedupe defensively by example_id / board_id (shouldn't happen if sharding is configured correctly)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for r in combined:
        k = str(r.get("example_id") or r.get("board_id") or "")
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)

    write_jsonl(base_raw, deduped)
    filtered = filter_examples(deduped, cfg)
    write_jsonl(turns_path, filtered)

    print(f"Wrote merged raw     -> {base_raw} ({len(deduped)} examples)")
    print(f"Wrote filtered turns -> {turns_path} ({len(filtered)} examples)")


if __name__ == "__main__":
    main()