# src/build_dpo_pairs.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .utils import load_yaml, read_jsonl, write_jsonl


def build_pairs(
    turns_raw: List[Dict[str, Any]],
    *,
    margin: float,
    max_rejected_per_prompt: int = 1,
) -> List[Dict[str, str]]:
    """
    Flatten your per-board record into TRL DPO pairs:
      {prompt: str, chosen: str, rejected: str}

    chosen = r["completion"] (your best)
    rejected = one or more candidates with (chosen_reward - cand_reward) >= margin
               (lowest rewards first), capped per prompt.
    """
    out: List[Dict[str, str]] = []
    max_rej = max(1, int(max_rejected_per_prompt))

    for r in turns_raw:
        prompt = r.get("prompt")
        chosen = r.get("completion")
        if not isinstance(prompt, str) or not isinstance(chosen, str):
            continue

        best_reward = float(r.get("reward", 0.0))
        cands = r.get("candidates") or []
        rej_list: List[Tuple[float, str]] = []

        for c in cands:
            try:
                rej_reward = float(c.get("reward", 0.0))
                rej_text = c.get("completion")
                if not isinstance(rej_text, str) or not rej_text:
                    continue
                if (best_reward - rej_reward) >= float(margin):
                    rej_list.append((rej_reward, rej_text))
            except Exception:
                continue

        if not rej_list:
            continue

        rej_list.sort(key=lambda t: t[0])  # lowest reward first
        for _, rejected in rej_list[:max_rej]:
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--turns-raw", default=None, help="Override paths.turns_raw_path")
    ap.add_argument("--out", default=None, help="Override paths.dpo_pairs_path")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    turns_raw_path = args.turns_raw or cfg["paths"]["turns_raw_path"]
    out_path = args.out or cfg["paths"]["dpo_pairs_path"]

    margin = float(cfg.get("training", {}).get("dpo_reward_margin", 2.0))
    max_rej = int(cfg.get("training", {}).get("dpo_max_rejected_per_prompt", 1))

    turns_raw = read_jsonl(turns_raw_path)
    pairs = build_pairs(turns_raw, margin=margin, max_rejected_per_prompt=max_rej)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, pairs)
    print(f"Wrote {len(pairs)} DPO pairs -> {out_path}")


if __name__ == "__main__":
    main()