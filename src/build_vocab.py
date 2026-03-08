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

from .utils import load_yaml

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
    max_words: int | None = None,
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

    n_pre_cap = len(kept)
    if max_words is not None:
        if int(max_words) <= 0:
            raise ValueError(f"max_words must be > 0 (got {max_words})")
        kept = kept[: int(max_words)]

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
        "max_words": int(max_words) if max_words is not None else None,
        "counts": {
            "total_seen": n_total,
            "kept_pre_cap": n_pre_cap,
            "kept": len(vocab),
            "capped": max(0, n_pre_cap - len(vocab)),
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
    # Internal defaults (used if neither CLI nor config provide values)
    OUT_DEFAULT = "data/vocab.txt"
    LANG_DEFAULT = "en"
    STOP_N_DEFAULT = 300
    MIN_ZIPF_DEFAULT = 3.0
    MAX_ZIPF_DEFAULT = 6.0
    MIN_LEN_DEFAULT = 3
    MAX_LEN_DEFAULT = 12
    TOKEN_REGEX_DEFAULT = r"^[a-z]+$"

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Optional YAML config path (e.g., configs/default.yaml)")

    # If provided, CLI overrides config; if omitted, config (then defaults) are used.
    ap.add_argument("--out", default=None, help="Output vocab path (overrides config paths.vocab_path)")
    ap.add_argument("--manifest", default=None, help="Optional JSON manifest path (overrides config vocab.manifest_path)")

    ap.add_argument("--lang", default=None, help="Language code for wordfreq (overrides config vocab.lang)")
    ap.add_argument("--stop-n", type=int, default=None, help="Remove top-N most frequent words as stopwords")
    ap.add_argument("--min-zipf", type=float, default=None, help="Minimum Zipf frequency to keep")
    ap.add_argument("--max-zipf", type=float, default=None, help="Maximum Zipf frequency to keep (drops ultra-common)")
    ap.add_argument("--min-len", type=int, default=None, help="Minimum token length")
    ap.add_argument("--max-len", type=int, default=None, help="Maximum token length")
    ap.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional cap on vocab size after filtering/sorting (top-N by Zipf). Overrides config vocab.max_words.",
    )
    ap.add_argument(
        "--token-regex",
        default=None,
        help="Regex a token must match (default keeps only lowercase alphabetic words).",
    )
    ap.add_argument(
        "--extra-banlist",
        default=None,
        help="Optional path to a newline-separated banlist (overrides config vocab.extra_banlist).",
    )

    args = ap.parse_args()

    cfg = load_yaml(args.config) if args.config else {}
    vcfg = cfg.get("vocab", {}) if isinstance(cfg, dict) else {}
    pcfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    out_path = (
        args.out
        or pcfg.get("vocab_path")
        or vcfg.get("out")
        or OUT_DEFAULT
    )
    manifest_path = (
        args.manifest
        or vcfg.get("manifest_path")
        or None
    )

    lang = args.lang or vcfg.get("lang") or LANG_DEFAULT
    stop_n = args.stop_n if args.stop_n is not None else int(vcfg.get("stop_n", STOP_N_DEFAULT))
    min_zipf = args.min_zipf if args.min_zipf is not None else float(vcfg.get("min_zipf", MIN_ZIPF_DEFAULT))
    max_zipf = args.max_zipf if args.max_zipf is not None else float(vcfg.get("max_zipf", MAX_ZIPF_DEFAULT))
    min_len = args.min_len if args.min_len is not None else int(vcfg.get("min_len", MIN_LEN_DEFAULT))
    max_len = args.max_len if args.max_len is not None else int(vcfg.get("max_len", MAX_LEN_DEFAULT))
    token_regex = args.token_regex or vcfg.get("token_regex") or TOKEN_REGEX_DEFAULT
    max_words = args.max_words if args.max_words is not None else vcfg.get("max_words", None)
    if max_words is not None:
        max_words = int(max_words)

    extra_banlist_path = args.extra_banlist or vcfg.get("extra_banlist") or None
    extra_ban = load_extra_banlist(extra_banlist_path)

    vocab, manifest = build_vocab(
        lang=lang,
        stop_n=stop_n,
        min_zipf=min_zipf,
        max_zipf=max_zipf,
        min_len=min_len,
        max_len=max_len,
        token_regex=token_regex,
        extra_ban=extra_ban,
        max_words=max_words,
    )

    write_vocab(out_path, vocab)
    print(f"Wrote {len(vocab)} words -> {out_path}")

    if manifest_path:
        write_manifest(manifest_path, manifest)
        print(f"Wrote manifest -> {manifest_path}")
    else:
        # quick console summary
        print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()