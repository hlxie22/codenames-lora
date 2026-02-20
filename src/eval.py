from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .io_utils import shard_path_insert_before_suffix, merge_jsonl_shards
from .metrics import aggregate
from .model_wrappers import Embedder, make_text_generator, load_lora_on_generator
from .mp_utils import is_child_process, child_shard_info, launch_children
from .rollout import run_turns_batched
from .utils import (
    ensure_dir,
    load_yaml,
    read_jsonl,
    save_config_snapshot,
    save_run_meta,
    set_global_seed,
    write_jsonl,
)


def _shard_out_path(out_dir: str | Path, shard_id: int, num_shards: int) -> Path:
    out_dir = Path(out_dir)
    return shard_path_insert_before_suffix(out_dir / "per_board.jsonl", shard_id, num_shards)


def _merge_shards(out_dir: str | Path, num_shards: int) -> List[Dict[str, Any]]:
    out_dir = Path(out_dir)
    shard_paths = [_shard_out_path(out_dir, sid, num_shards) for sid in range(num_shards)]
    combined = merge_jsonl_shards(shard_paths, sort_key=lambda r: str(r.get("board_id", "")))
    return combined


# -------------------------
# vLLM LoRA toggling proxy (so we can share one engine)
# -------------------------

class _VLLMProxy:
    """
    Delegate to a base generator but force a particular LoRA request for calls.
    Works with VLLMTextGenerator which stores LoRARequest on `._lora_request`.
    """

    def __init__(self, base, lora_request):
        self._base = base
        self._lora_request = lora_request
        self.model_id = getattr(base, "model_id", "unknown")

    def format_chat(self, *args, **kwargs):
        return self._base.format_chat(*args, **kwargs)

    def generate(self, *args, **kwargs):
        old = getattr(self._base, "_lora_request", None)
        try:
            setattr(self._base, "_lora_request", self._lora_request)
            return self._base.generate(*args, **kwargs)
        finally:
            setattr(self._base, "_lora_request", old)

    def generate_batch(self, *args, **kwargs):
        old = getattr(self._base, "_lora_request", None)
        try:
            setattr(self._base, "_lora_request", self._lora_request)
            return self._base.generate_batch(*args, **kwargs)
        finally:
            setattr(self._base, "_lora_request", old)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["baseline", "sft"], required=True)
    ap.add_argument("--out", required=True, help="Output directory (e.g., outputs/baselines/run1)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["training"].get("seed", 0)))

    num_procs = int(cfg.get("inference", {}).get("num_processes", 1))
    batch_size = int(cfg.get("inference", {}).get("batch_size", 1))

    out_dir = ensure_dir(args.out)

    # Master process: write run meta, spawn children, then merge + aggregate
    if num_procs > 1 and not is_child_process():
        save_config_snapshot(cfg, out_dir)
        save_run_meta(
            out_dir,
            command=["python", "-m", "src.eval", "--config", args.config, "--mode", args.mode, "--out", args.out],
        )

        if cfg.get("inference", {}).get("backend") == "vllm":
            tp = int(cfg.get("inference", {}).get("vllm", {}).get("tensor_parallel_size", 1))
            if tp != 1:
                print(
                    f"[warn] inference.vllm.tensor_parallel_size={tp} with inference.num_processes={num_procs}. "
                    f"Usually you want tensor_parallel_size=1 when using multiple processes."
                )

        # mp_utils.launch_children(module, argv, num_procs)
        launch_children("src.eval", ["--config", args.config, "--mode", args.mode, "--out", args.out], num_procs)

        # Merge shard outputs and compute metrics once
        per_board = _merge_shards(out_dir, num_procs)
        per_path = Path(out_dir) / "per_board.jsonl"
        write_jsonl(per_path, per_board)

        metrics = aggregate(per_board)
        with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Wrote per-board -> {per_path}")
        print(f"Wrote metrics   -> {Path(out_dir) / 'metrics.json'}")
        return

    # Worker (or single process)
    if num_procs == 1 and not is_child_process():
        save_config_snapshot(cfg, out_dir)
        save_run_meta(
            out_dir,
            command=["python", "-m", "src.eval", "--config", args.config, "--mode", args.mode, "--out", args.out],
        )

    boards = read_jsonl(cfg["paths"]["boards_eval_path"])

    shard_id, num_shards = child_shard_info() if is_child_process() else (0, 1)
    if num_shards > 1:
        boards = [b for i, b in enumerate(boards) if (i % num_shards) == shard_id]
        print(f"[shard {shard_id}/{num_shards}] boards={len(boards)}")

    # Build generators (avoid two vLLM engines when model_ids match)
    backend = cfg.get("inference", {}).get("backend", "hf")
    sp_id = cfg["models"]["spymaster_model_id"]
    g_id = cfg["models"]["guesser_model_id"]

    if backend == "vllm" and sp_id == g_id:
        base = make_text_generator(sp_id, cfg)

        if args.mode == "sft":
            adapter_dir = cfg["training"]["output_adapter_dir"]
            base = load_lora_on_generator(base, adapter_dir)
            # capture LoRA request then disable it for guesser calls
            lora_req = getattr(base, "_lora_request", None)
            setattr(base, "_lora_request", None)

            spymaster = _VLLMProxy(base, lora_req)
            guesser = _VLLMProxy(base, None)
        else:
            spymaster = base
            guesser = base
    else:
        # General case: separate generators
        spymaster = make_text_generator(sp_id, cfg)
        if args.mode == "sft":
            adapter_dir = cfg["training"]["output_adapter_dir"]
            spymaster = load_lora_on_generator(spymaster, adapter_dir)
        guesser = make_text_generator(g_id, cfg)

    # Embedder
    use_embed = bool(cfg.get("constraints", {}).get("enable_directness_check", True))
    embedder = None
    if use_embed:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        embedder = Embedder(cfg["models"]["embedding_model_id"], device=device)

    # Evaluate
    per_board: List[Dict[str, Any]] = []

    # Keep n_candidates=1 in eval
    n_candidates = 1

    for start in range(0, len(boards), max(1, batch_size)):
        batch = boards[start : start + max(1, batch_size)]

        bests, metas = run_turns_batched(
            batch,
            spymaster,
            guesser,
            embedder,
            cfg,
            n_candidates=n_candidates,
        )

        for b, best, meta in zip(batch, bests, metas):
            rec = {
                "board_id": b["board_id"],
                "reward": float(best.reward),
                "clue": best.clue,
                "num": int(best.num),
                "guess_words": best.guess_words,
                "stats": {**best.stats, "directness": float(best.directness)},
                "clue_meta": {
                    "valid": bool(best.valid),
                    "rejected_total": int(meta["rejected_total"]),
                    "rejection_counts": meta["rejection_counts"],
                },
            }
            per_board.append(rec)

        done = min(start + len(batch), len(boards))
        if done % 50 == 0 or done == len(boards):
            mr = float(np.mean([r["reward"] for r in per_board])) if per_board else 0.0
            print(f"[shard {shard_id}/{num_shards}] [{done}/{len(boards)}] mean_reward={mr:.3f}")

    # Write outputs
    if num_shards > 1 and is_child_process():
        shard_path = _shard_out_path(out_dir, shard_id, num_shards)
        write_jsonl(shard_path, per_board)
        print(f"[shard {shard_id}/{num_shards}] Wrote per-board shard -> {shard_path}")
        return

    # Single-process final outputs
    per_path = Path(out_dir) / "per_board.jsonl"
    write_jsonl(per_path, per_board)

    metrics = aggregate(per_board)
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote per-board -> {per_path}")
    print(f"Wrote metrics   -> {Path(out_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()