from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .model_wrappers import HFTextGenerator, Embedder, load_lora_on_generator
from .rollout import run_turn, select_best_candidate
from .metrics import aggregate
from .utils import load_yaml, read_jsonl, write_jsonl, ensure_dir, save_config_snapshot, save_run_meta, set_global_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["baseline", "sft"], required=True)
    ap.add_argument("--out", required=True, help="Output directory (e.g., outputs/baselines/run1)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["training"].get("seed", 0)))

    out_dir = ensure_dir(args.out)
    save_config_snapshot(cfg, out_dir)
    save_run_meta(out_dir, command=["python", "-m", "src.eval", "--config", args.config, "--mode", args.mode, "--out", args.out])

    boards = read_jsonl(cfg["paths"]["boards_eval_path"])

    # models
    spymaster = HFTextGenerator(cfg["models"]["spymaster_model_id"], device_map="auto")
    if args.mode == "sft":
        adapter_dir = cfg["training"]["output_adapter_dir"]
        spymaster = load_lora_on_generator(spymaster, adapter_dir)

    guesser = HFTextGenerator(cfg["models"]["guesser_model_id"], device_map="auto")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    embedder = Embedder(cfg["models"]["embedding_model_id"], device=device)

    per_board: List[Dict[str, Any]] = []
    for i, b in enumerate(boards):
        seed = int(b.get("seed", 0))
        candidates, meta = run_turn(b, spymaster, guesser, embedder, cfg, n_candidates=1, seed=seed)
        best = select_best_candidate(candidates)

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

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(boards)}] mean_reward={np.mean([r['reward'] for r in per_board]):.3f}")

    per_path = Path(out_dir) / "per_board.jsonl"
    write_jsonl(per_path, per_board)

    metrics = aggregate(per_board)
    with open(Path(out_dir) / "metrics.json", "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2)

    print(f"Wrote per-board -> {per_path}")
    print(f"Wrote metrics   -> {Path(out_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()