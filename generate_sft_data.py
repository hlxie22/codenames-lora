from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .model_wrappers import HFTextGenerator, Embedder
from .rollout import run_turn, select_best_candidate
from .spymaster_prompt import build_spymaster_messages
from .utils import load_yaml, read_jsonl, write_jsonl, ensure_dir, set_global_seed

import re
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)

def extract_think_block(text: str) -> str:
    # Keep the model-produced think block if present; otherwise return "".
    lo = text.lower().rfind("<think>")
    hi = text.lower().rfind("</think>")
    if lo != -1 and hi != -1 and hi > lo:
        return text[lo : hi + len("</think>")].strip()
    # Sometimes you might only see </think>; keep everything up to it as "thinking".
    if hi != -1:
        return text[: hi + len("</think>")].strip()
    return ""

def extract_think(text: str) -> str:
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else ""

def filter_examples(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    fcfg = cfg["filtering"]
    mode = fcfg["mode"]

    if not records:
        return []

    if mode == "top_percent":   
        top_p = float(fcfg["top_percent"])
        rewards = np.array([r["reward"] for r in records], dtype=np.float32)
        thr = float(np.quantile(rewards, 1.0 - top_p))
        kept = [r for r in records if float(r["reward"]) >= thr]
        return kept

    if mode == "rule_based":
        min_team = int(fcfg.get("min_team_correct", 0))
        require_no_assassin = bool(fcfg.get("require_no_assassin", False))
        kept = []
        for r in records:
            st = r.get("stats", {})
            if int(st.get("n_team", 0)) < min_team:
                continue
            if require_no_assassin and int(st.get("assassin", 0)) > 0:
                continue
            kept.append(r)
        return kept

    raise ValueError(f"Unknown filtering mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["training"].get("seed", 0)))

    boards = read_jsonl(cfg["paths"]["boards_train_path"])

    # Models
    spymaster = HFTextGenerator(cfg["models"]["spymaster_model_id"], device_map="auto")
    guesser = HFTextGenerator(cfg["models"]["guesser_model_id"], device_map="auto")

    # Embedder (use cuda if available)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    embedder = Embedder(cfg["models"]["embedding_model_id"], device=device)

    n_candidates = int(cfg["decoding"]["n_candidates"])

    raw_records: List[Dict[str, Any]] = []
    for i, b in enumerate(boards):
        # seed per board for stability
        seed = int(b.get("seed", 0))

        candidates, meta = run_turn(
            b,
            spymaster,
            guesser,
            embedder,
            cfg,
            n_candidates=n_candidates,
            seed=seed,
        )
        best = select_best_candidate(candidates)

        # prompt must match exactly the spymaster prompt used in rollouts
        revealed = [False] * len(b["board_words"])
        msgs = build_spymaster_messages(b["board_words"], b["labels"], revealed, cfg)

        use_chat = bool(cfg.get("qwen", {}).get("use_chat_template", False))
        sp_think = bool(cfg.get("qwen", {}).get("enable_thinking_spymaster", True))
        prompt = spymaster.format_chat(msgs, add_generation_prompt=True, enable_thinking=sp_think) if use_chat else msgs[-1]["content"]

        think_block = extract_think_block(best.raw_spymaster_text)
        completion = ""
        if think_block:
            completion += think_block + "\n"
        completion += f"CLUE: {best.clue}\nNUM: {best.num}\n"

        rec = {
            "example_id": f"turn_{i+1:06d}",
            "board_id": b["board_id"],
            "prompt": prompt,
            "completion": completion,
            "reward": float(best.reward),
            "stats": {
                **best.stats,
                "directness": float(best.directness),
            },
            "clue_meta": {
                "clue": best.clue,
                "num": int(best.num),
                "valid": bool(best.valid),
                "rejected_candidates": int(meta["rejected_total"]),
                "rejection_counts": meta["rejection_counts"],
            },
            "debug": {
                "guess_words": best.guess_words,
            },
        }
        raw_records.append(rec)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(boards)}] mean_reward={np.mean([r['reward'] for r in raw_records]):.3f}")

    # write raw
    raw_path = cfg["paths"].get("sft_turns_raw_path")
    if raw_path:
        write_jsonl(raw_path, raw_records)
        print(f"Wrote raw SFT records -> {raw_path}")

    # filter and write final
    filtered = filter_examples(raw_records, cfg)
    write_jsonl(cfg["paths"]["sft_turns_path"], filtered)
    print(f"Filtered {len(filtered)}/{len(raw_records)} -> {cfg['paths']['sft_turns_path']}")


if __name__ == "__main__":
    main()