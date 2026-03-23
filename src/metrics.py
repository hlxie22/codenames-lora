from __future__ import annotations

from typing import Any, Dict, List, Tuple
from collections import Counter
import numpy as np


def compute_clue_diversity(clues: List[str]) -> Dict[str, Any]:
    clues = [c.strip().lower() for c in clues if c and c.strip()]
    n = len(clues)
    if n == 0:
        return {"n": 0, "distinct": 0, "distinct_rate": 0.0, "entropy": 0.0}

    ctr = Counter(clues)
    distinct = len(ctr)
    probs = np.array([v / n for v in ctr.values()], dtype=np.float64)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {"n": n, "distinct": distinct, "distinct_rate": float(distinct / n), "entropy": entropy}


def bootstrap_ci(values: List[float], n: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n):
        samp = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(samp)))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def aggregate(per_board_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(r["reward"]) for r in per_board_records]
    n_team = [int(r["stats"].get("n_team", 0)) for r in per_board_records]
    n_opp = [int(r["stats"].get("n_opp", 0)) for r in per_board_records]
    n_neu = [int(r["stats"].get("n_neu", 0)) for r in per_board_records]
    assassin = [int(r["stats"].get("assassin", 0)) for r in per_board_records]
    directness = [float(r["stats"].get("directness", 0.0)) for r in per_board_records]
    clues = [r.get("clue", "") for r in per_board_records]

    parse_valid_flags: List[float] = []
    nonempty_clue_flags: List[float] = []
    rejected_totals: List[int] = []
    rejection_reason_counts: Counter[str] = Counter()

    for r in per_board_records:
        clue = str(r.get("clue", "") or "")
        nonempty_clue_flags.append(1.0 if clue.strip() else 0.0)

        clue_meta = r.get("clue_meta") or {}
        parse_valid = clue_meta.get("parse_valid", None)
        if parse_valid is None:
            parse_valid = bool(clue.strip()) and int(r.get("num", 0) or 0) > 0
        parse_valid_flags.append(1.0 if bool(parse_valid) else 0.0)

        try:
            rejected_totals.append(int(clue_meta.get("rejected_total", 0)))
        except Exception:
            pass

        rej_counts = clue_meta.get("rejection_counts") or {}
        if isinstance(rej_counts, dict):
            for k, v in rej_counts.items():
                try:
                    rejection_reason_counts[str(k)] += int(v)
                except Exception:
                    continue

    assassin_rate = float(np.mean([1.0 if a > 0 else 0.0 for a in assassin])) if assassin else 0.0
    parse_valid_rate = float(np.mean(parse_valid_flags)) if parse_valid_flags else 0.0

    out: Dict[str, Any] = {
        "n_boards": len(per_board_records),
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_median": float(np.median(rewards)) if rewards else 0.0,
        "reward_ci95": bootstrap_ci(rewards, n=500, seed=1) if rewards else (0.0, 0.0),
        "team_mean": float(np.mean(n_team)) if n_team else 0.0,
        "opp_mean": float(np.mean(n_opp)) if n_opp else 0.0,
        "neu_mean": float(np.mean(n_neu)) if n_neu else 0.0,
        "assassin_rate": assassin_rate,
        "directness_mean": float(np.mean(directness)) if directness else 0.0,
        "directness_p90": float(np.quantile(directness, 0.9)) if directness else 0.0,
        "parse_valid_rate": parse_valid_rate,
        "parse_fail_rate": float(1.0 - parse_valid_rate),
        "nonempty_clue_rate": float(np.mean(nonempty_clue_flags)) if nonempty_clue_flags else 0.0,
        "rejected_candidates_mean": float(np.mean(rejected_totals)) if rejected_totals else 0.0,
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "clue_diversity": compute_clue_diversity(clues),
    }
    return out