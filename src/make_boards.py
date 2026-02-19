from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import load_yaml, set_global_seed, write_jsonl, ensure_dir


def load_vocab(path: str | Path) -> List[str]:
    vocab: List[str] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w:
                continue
            wl = w.lower()
            if wl in seen:
                continue
            seen.add(wl)
            vocab.append(w)
    return vocab


def sample_board(vocab: List[str], rng: np.random.Generator, board_size: int, allow_duplicates: bool) -> List[str]:
    if allow_duplicates:
        idx = rng.integers(0, len(vocab), size=board_size).tolist()
        return [vocab[i] for i in idx]
    else:
        if len(vocab) < board_size:
            raise ValueError(f"Vocab too small: {len(vocab)} < {board_size}")
        idx = rng.choice(len(vocab), size=board_size, replace=False).tolist()
        return [vocab[i] for i in idx]


def assign_labels(
    rng: np.random.Generator,
    board_size: int,
    n_team: int,
    n_opp: int,
    n_neu: int,
    n_assassin: int,
) -> List[str]:
    labels = (["TEAM"] * n_team) + (["OPP"] * n_opp) + (["NEU"] * n_neu) + (["ASSASSIN"] * n_assassin)
    if len(labels) != board_size:
        raise ValueError("Label counts do not sum to board_size.")
    rng.shuffle(labels)
    return labels


def board_hash(board_words: List[str]) -> str:
    s = "||".join(sorted([w.lower() for w in board_words]))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def make_split(
    split_name: str,
    n_boards: int,
    vocab: List[str],
    cfg: Dict[str, Any],
    seed: int,
    global_seen_hashes: set[str],
) -> List[Dict[str, Any]]:
    bcfg = cfg["boards"]
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Any]] = []
    local_seen = set()

    for i in range(n_boards):
        # try until unique (if requested)
        while True:
            board_seed = int(rng.integers(0, 2**31 - 1))
            brng = np.random.default_rng(board_seed)
            words = sample_board(vocab, brng, int(bcfg["board_size"]), bool(bcfg["allow_duplicates"]))
            h = board_hash(words)

            if bcfg.get("avoid_repeat_boards_within_split", True) and h in local_seen:
                continue
            if bcfg.get("avoid_repeat_boards_across_splits", True) and h in global_seen_hashes:
                continue

            local_seen.add(h)
            global_seen_hashes.add(h)
            labels = assign_labels(
                brng,
                int(bcfg["board_size"]),
                int(bcfg["n_team"]),
                int(bcfg["n_opp"]),
                int(bcfg["n_neu"]),
                int(bcfg["n_assassin"]),
            )
            rec = {
                "board_id": f"{split_name}_{i+1:06d}",
                "board_words": words,
                "labels": labels,
                "seed": board_seed,
                "board_hash": h,
            }
            records.append(rec)
            break

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    vocab = load_vocab(cfg["paths"]["vocab_path"])

    set_global_seed(0)

    seen_hashes: set[str] = set()
    train = make_split(
        "train",
        int(cfg["boards"]["n_train_boards"]),
        vocab,
        cfg,
        int(cfg["boards"]["seed_train"]),
        seen_hashes,
    )
    eval_ = make_split(
        "eval",
        int(cfg["boards"]["n_eval_boards"]),
        vocab,
        cfg,
        int(cfg["boards"]["seed_eval"]),
        seen_hashes,
    )

    write_jsonl(cfg["paths"]["boards_train_path"], train)
    write_jsonl(cfg["paths"]["boards_eval_path"], eval_)

    print(f"Wrote {len(train)} train boards -> {cfg['paths']['boards_train_path']}")
    print(f"Wrote {len(eval_)} eval boards  -> {cfg['paths']['boards_eval_path']}")


if __name__ == "__main__":
    main()