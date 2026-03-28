from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .io_utils import shard_path_insert_before_suffix, merge_jsonl_shards
from .metrics import aggregate, aggregate_paired, prefix_metric_keys
from .model_wrappers import Embedder, build_codenames_generators
from .mp_utils import is_child_process, child_shard_info, launch_children
from .rollout import run_turns_batched
from .utils import (
    ensure_dir,
    load_yaml,
    read_jsonl,
    resolve_trained_adapter_dir,
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


def _write_metrics_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _augment_trained_metrics_with_paired(trained_metrics_path: Path, paired: Dict[str, Any]) -> None:
    payload: Dict[str, Any] = {}
    if trained_metrics_path.exists():
        try:
            payload = json.loads(trained_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    payload.update(prefix_metric_keys(paired, "paired_baseline"))
    payload["paired_vs_baseline"] = paired
    _write_metrics_json(trained_metrics_path, payload)


def _maybe_write_post_eval_paired_metrics(root_dir: str | Path) -> Optional[Dict[str, Any]]:
    root = Path(root_dir)
    baseline_per = root / "baseline" / "per_board.jsonl"
    trained_per = root / "trained" / "per_board.jsonl"
    if not baseline_per.exists() or not trained_per.exists():
        return None

    baseline = read_jsonl(baseline_per)
    trained = read_jsonl(trained_per)
    paired = aggregate_paired(trained, baseline)

    paired_path = root / "paired_metrics.json"
    _write_metrics_json(paired_path, paired)

    trained_metrics_path = root / "trained" / "metrics.json"
    _augment_trained_metrics_with_paired(trained_metrics_path, paired)
    return paired


def _finalize_eval_outputs(out_dir: str | Path, per_board: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    per_path = out_dir / "per_board.jsonl"
    write_jsonl(per_path, per_board)

    metrics = aggregate(per_board)
    _write_metrics_json(out_dir / "metrics.json", metrics)

    if out_dir.name in {"baseline", "trained"}:
        _maybe_write_post_eval_paired_metrics(out_dir.parent)
        if out_dir.name == "trained":
            try:
                metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            except Exception:
                pass

    print(f"Wrote per-board -> {per_path}")
    print(f"Wrote metrics   -> {out_dir / 'metrics.json'}")
    if out_dir.name in {"baseline", "trained"}:
        paired_path = out_dir.parent / "paired_metrics.json"
        if paired_path.exists():
            print(f"Wrote paired    -> {paired_path}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["baseline", "trained"], required=True)
    ap.add_argument("--out", required=True, help="Output directory (e.g., outputs/baselines/run1)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["training"].get("seed", 0)))
    trained_adapter_dir = resolve_trained_adapter_dir(cfg)

    num_procs = int(cfg.get("inference", {}).get("num_processes", 1))
    batch_size = int(cfg.get("inference", {}).get("batch_size", 1))

    out_dir = ensure_dir(args.out)

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

        launch_children("src.eval", ["--config", args.config, "--mode", args.mode, "--out", args.out], num_procs)

        per_board = _merge_shards(out_dir, num_procs)
        _finalize_eval_outputs(out_dir, per_board)
        return

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

    use_trained_adapter = trained_adapter_dir if args.mode == "trained" else None
    spymaster, guesser = build_codenames_generators(
        cfg,
        spymaster_adapter_dir=use_trained_adapter,
    )

    use_embed = bool(cfg.get("constraints", {}).get("enable_directness_check", True))
    embedder = None
    if use_embed:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        embedder = Embedder(cfg["models"]["embedding_model_id"], device=device)

    per_board: List[Dict[str, Any]] = []
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
            max_resamples=1,
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
                    "parse_valid": bool(best.parse_valid),
                    "rejected_total": int(meta["rejected_total"]),
                    "rejection_counts": meta["rejection_counts"],
                },
            }
            per_board.append(rec)

        done = min(start + len(batch), len(boards))
        if done % 50 == 0 or done == len(boards):
            mr = float(np.mean([r["reward"] for r in per_board])) if per_board else 0.0
            print(f"[shard {shard_id}/{num_shards}] [{done}/{len(boards)}] mean_reward={mr:.3f}")

    if num_shards > 1 and is_child_process():
        shard_path = _shard_out_path(out_dir, shard_id, num_shards)
        write_jsonl(shard_path, per_board)
        print(f"[shard {shard_id}/{num_shards}] Wrote per-board shard -> {shard_path}")
        return

    _finalize_eval_outputs(out_dir, per_board)


if __name__ == "__main__":
    main()
