#!/usr/bin/env python3
"""
Build a clean Codenames-style vocabulary file using wordfreq.

Approach (hybrid):
- remove ultra-common stopwords: top_n_list('en', STOP_N)
- keep only words within a Zipf frequency window: MIN_ZIPF <= zipf <= MAX_ZIPF
- keep only lowercase alphabetic tokens (default: ^[a-z]+$)
- filter by length
- apply a small curated "ban list" for months/days/numbers/meta-words
- write data/vocab.txt (one word per line)
- optionally write a manifest with stats

Usage:
  pip install wordfreq
  python -m src.build_vocab --out data/vocab.txt --manifest data/vocab_manifest.json

Notes:
- Output words are lowercased.
- Ordering is deterministic: sorted by (-zipf, word).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from wordfreq import iter_wordlist, top_n_list, zipf_frequency


DEFAULT_BAN = {
    # days
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # months
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    # numbers (word forms)
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "first", "second", "third",
    # ultra-generic/meta (often unfun boards)
    "thing", "things", "stuff", "object", "objects", "person", "people", "someone", "somebody",
    "anything", "everything", "nothing", "something",
    # directions (optional; remove if you like them)
    "north", "south", "east", "west", "left", "right",
}


def load_extra_banlist(path: str | None) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Banlist file not found: {path}")
    banned = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if not w or w.startswith("#"):
            continue
        banned.add(w)
    return banned


def build_vocab(
    *,
    lang: str,
    stop_n: int,
    min_zipf: float,
    max_zipf: float,
    min_len: int,
    max_len: int,
    token_regex: str,
    extra_ban: Set[str],
) -> Tuple[List[str], dict]:
    token_re = re.compile(token_regex)

    stop = set(top_n_list(lang, stop_n))
    banned = set(DEFAULT_BAN) | set(extra_ban)

    kept: List[Tuple[str, float]] = []
    seen = set()

    n_total = 0
    n_stop = 0
    n_badshape = 0
    n_len = 0
    n_zipf = 0
    n_banned = 0

    for w in iter_wordlist(lang):
        n_total += 1
        w = w.strip().lower()
        if not w:
            continue

        if w in seen:
            continue
        seen.add(w)

        if w in stop:
            n_stop += 1
            continue

        if w in banned:
            n_banned += 1
            continue

        if not token_re.match(w):
            n_badshape += 1
            continue

        if len(w) < min_len or len(w) > max_len:
            n_len += 1
            continue

        z = float(zipf_frequency(w, lang))
        if z < min_zipf or z > max_zipf:
            n_zipf += 1
            continue

        kept.append((w, z))

    # deterministic ordering: more frequent first, then alpha
    kept.sort(key=lambda t: (-t[1], t[0]))
    vocab = [w for (w, _) in kept]

    zs = [z for (_, z) in kept]
    manifest = {
        "lang": lang,
        "stop_n": stop_n,
        "min_zipf": min_zipf,
        "max_zipf": max_zipf,
        "min_len": min_len,
        "max_len": max_len,
        "token_regex": token_regex,
        "counts": {
            "total_seen": n_total,
            "kept": len(vocab),
            "filtered_stopwords": n_stop,
            "filtered_banned": n_banned,
            "filtered_badshape": n_badshape,
            "filtered_length": n_len,
            "filtered_zipf": n_zipf,
        },
        "zipf_stats": {
            "min": min(zs) if zs else None,
            "max": max(zs) if zs else None,
            "mean": (sum(zs) / len(zs)) if zs else None,
            "p10": percentile(zs, 10) if zs else None,
            "p50": percentile(zs, 50) if zs else None,
            "p90": percentile(zs, 90) if zs else None,
        },
        "examples_top20": vocab[:20],
    }
    return vocab, manifest


def percentile(values: List[float], p: float) -> float:
    # small deterministic percentile helper without numpy
    if not values:
        return float("nan")
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d = k - f
    return xs[f] * (1 - d) + xs[c] * d


def write_vocab(out_path: str, vocab: List[str]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(vocab) + "\n", encoding="utf-8")


def write_manifest(path: str, manifest: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/vocab.txt", help="Output vocab path")
    ap.add_argument("--manifest", default=None, help="Optional JSON manifest path")

    ap.add_argument("--lang", default="en", help="Language code for wordfreq")
    ap.add_argument("--stop-n", type=int, default=300, help="Remove top-N most frequent words as stopwords")
    ap.add_argument("--min-zipf", type=float, default=3.0, help="Minimum Zipf frequency to keep")
    ap.add_argument("--max-zipf", type=float, default=6.0, help="Maximum Zipf frequency to keep (drops ultra-common)")
    ap.add_argument("--min-len", type=int, default=3, help="Minimum token length")
    ap.add_argument("--max-len", type=int, default=12, help="Maximum token length")
    ap.add_argument(
        "--token-regex",
        default=r"^[a-z]+$",
        help="Regex a token must match (default keeps only lowercase alphabetic words).",
    )
    ap.add_argument(
        "--extra-banlist",
        default=None,
        help="Optional path to a newline-separated banlist (lowercased). Lines starting with # are ignored.",
    )

    args = ap.parse_args()

    extra_ban = load_extra_banlist(args.extra_banlist)

    vocab, manifest = build_vocab(
        lang=args.lang,
        stop_n=args.stop_n,
        min_zipf=args.min_zipf,
        max_zipf=args.max_zipf,
        min_len=args.min_len,
        max_len=args.max_len,
        token_regex=args.token_regex,
        extra_ban=extra_ban,
    )

    write_vocab(args.out, vocab)
    print(f"Wrote {len(vocab)} words -> {args.out}")

    if args.manifest:
        write_manifest(args.manifest, manifest)
        print(f"Wrote manifest -> {args.manifest}")
    else:
        # quick console summary
        print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()