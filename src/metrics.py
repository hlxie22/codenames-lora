from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

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


def prefix_metric_keys(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    pre = prefix if prefix.endswith("_") else f"{prefix}_"
    return {f"{pre}{k}": v for k, v in metrics.items()}


def _clue_meta(record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(record.get("clue_meta") or {})


def _parse_valid_flag(record: Dict[str, Any]) -> float:
    clue = str(record.get("clue", "") or "")
    clue_meta = _clue_meta(record)
    parse_valid = clue_meta.get("parse_valid", None)
    if parse_valid is None:
        parse_valid = bool(clue.strip()) and int(record.get("num", 0) or 0) > 0
    return 1.0 if bool(parse_valid) else 0.0


def _nonempty_clue_flag(record: Dict[str, Any]) -> float:
    clue = str(record.get("clue", "") or "")
    return 1.0 if clue.strip() else 0.0


def _rejected_total(record: Dict[str, Any]) -> float:
    clue_meta = _clue_meta(record)
    try:
        return float(clue_meta.get("rejected_total", 0))
    except Exception:
        return 0.0


def _assassin_hit_flag(record: Dict[str, Any]) -> float:
    stats = record.get("stats") or {}
    return 1.0 if int(stats.get("assassin", 0)) > 0 else 0.0


def _board_id(record: Dict[str, Any]) -> str:
    return str(record.get("board_id") or record.get("example_id") or "")


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
        nonempty_clue_flags.append(_nonempty_clue_flag(r))
        parse_valid_flags.append(_parse_valid_flag(r))

        try:
            rejected_totals.append(int(_rejected_total(r)))
        except Exception:
            pass

        rej_counts = _clue_meta(r).get("rejection_counts") or {}
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


def aggregate_paired(
    current_records: List[Dict[str, Any]],
    reference_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Board-aligned paired comparison. Deltas are always:

        current - reference

    Matching is done by board_id (or example_id as a fallback).
    """
    cur_by_id = {_board_id(r): r for r in current_records if _board_id(r)}
    ref_by_id = {_board_id(r): r for r in reference_records if _board_id(r)}

    shared_ids = sorted(set(cur_by_id) & set(ref_by_id))
    if not shared_ids:
        return {
            "n_paired": 0,
            "reward_current_mean": 0.0,
            "reward_reference_mean": 0.0,
            "reward_delta_mean": 0.0,
            "reward_delta_median": 0.0,
            "reward_delta_ci95": (0.0, 0.0),
            "reward_win_rate": 0.0,
            "reward_tie_rate": 0.0,
            "reward_loss_rate": 0.0,
            "team_current_mean": 0.0,
            "team_reference_mean": 0.0,
            "team_delta_mean": 0.0,
            "opp_current_mean": 0.0,
            "opp_reference_mean": 0.0,
            "opp_delta_mean": 0.0,
            "neu_current_mean": 0.0,
            "neu_reference_mean": 0.0,
            "neu_delta_mean": 0.0,
            "assassin_rate_current": 0.0,
            "assassin_rate_reference": 0.0,
            "assassin_rate_delta": 0.0,
            "parse_valid_rate_current": 0.0,
            "parse_valid_rate_reference": 0.0,
            "parse_valid_rate_delta": 0.0,
            "nonempty_clue_rate_current": 0.0,
            "nonempty_clue_rate_reference": 0.0,
            "nonempty_clue_rate_delta": 0.0,
            "rejected_candidates_current_mean": 0.0,
            "rejected_candidates_reference_mean": 0.0,
            "rejected_candidates_delta_mean": 0.0,
            "directness_current_mean": 0.0,
            "directness_reference_mean": 0.0,
            "directness_delta_mean": 0.0,
            "clue_changed_rate": 0.0,
        }

    reward_cur: List[float] = []
    reward_ref: List[float] = []
    reward_delta: List[float] = []

    team_cur: List[float] = []
    team_ref: List[float] = []
    team_delta: List[float] = []

    opp_cur: List[float] = []
    opp_ref: List[float] = []
    opp_delta: List[float] = []

    neu_cur: List[float] = []
    neu_ref: List[float] = []
    neu_delta: List[float] = []

    assassin_cur: List[float] = []
    assassin_ref: List[float] = []
    assassin_delta: List[float] = []

    parse_cur: List[float] = []
    parse_ref: List[float] = []
    parse_delta: List[float] = []

    nonempty_cur: List[float] = []
    nonempty_ref: List[float] = []
    nonempty_delta: List[float] = []

    rejected_cur: List[float] = []
    rejected_ref: List[float] = []
    rejected_delta: List[float] = []

    directness_cur: List[float] = []
    directness_ref: List[float] = []
    directness_delta: List[float] = []

    reward_wins = 0
    reward_ties = 0
    reward_losses = 0
    clue_changed = 0

    for board_id in shared_ids:
        cur = cur_by_id[board_id]
        ref = ref_by_id[board_id]

        cur_reward = float(cur.get("reward", 0.0))
        ref_reward = float(ref.get("reward", 0.0))
        delta_reward = cur_reward - ref_reward
        reward_cur.append(cur_reward)
        reward_ref.append(ref_reward)
        reward_delta.append(delta_reward)

        if np.isclose(delta_reward, 0.0, atol=1e-9):
            reward_ties += 1
        elif delta_reward > 0.0:
            reward_wins += 1
        else:
            reward_losses += 1

        cur_stats = cur.get("stats") or {}
        ref_stats = ref.get("stats") or {}

        cur_team = float(cur_stats.get("n_team", 0))
        ref_team = float(ref_stats.get("n_team", 0))
        team_cur.append(cur_team)
        team_ref.append(ref_team)
        team_delta.append(cur_team - ref_team)

        cur_opp = float(cur_stats.get("n_opp", 0))
        ref_opp = float(ref_stats.get("n_opp", 0))
        opp_cur.append(cur_opp)
        opp_ref.append(ref_opp)
        opp_delta.append(cur_opp - ref_opp)

        cur_neu = float(cur_stats.get("n_neu", 0))
        ref_neu = float(ref_stats.get("n_neu", 0))
        neu_cur.append(cur_neu)
        neu_ref.append(ref_neu)
        neu_delta.append(cur_neu - ref_neu)

        cur_assassin = _assassin_hit_flag(cur)
        ref_assassin = _assassin_hit_flag(ref)
        assassin_cur.append(cur_assassin)
        assassin_ref.append(ref_assassin)
        assassin_delta.append(cur_assassin - ref_assassin)

        cur_parse = _parse_valid_flag(cur)
        ref_parse = _parse_valid_flag(ref)
        parse_cur.append(cur_parse)
        parse_ref.append(ref_parse)
        parse_delta.append(cur_parse - ref_parse)

        cur_nonempty = _nonempty_clue_flag(cur)
        ref_nonempty = _nonempty_clue_flag(ref)
        nonempty_cur.append(cur_nonempty)
        nonempty_ref.append(ref_nonempty)
        nonempty_delta.append(cur_nonempty - ref_nonempty)

        cur_rejected = _rejected_total(cur)
        ref_rejected = _rejected_total(ref)
        rejected_cur.append(cur_rejected)
        rejected_ref.append(ref_rejected)
        rejected_delta.append(cur_rejected - ref_rejected)

        cur_directness = float(cur_stats.get("directness", 0.0))
        ref_directness = float(ref_stats.get("directness", 0.0))
        directness_cur.append(cur_directness)
        directness_ref.append(ref_directness)
        directness_delta.append(cur_directness - ref_directness)

        cur_clue = str(cur.get("clue", "") or "").strip().lower()
        ref_clue = str(ref.get("clue", "") or "").strip().lower()
        if cur_clue != ref_clue:
            clue_changed += 1

    n = len(shared_ids)
    out: Dict[str, Any] = {
        "n_paired": n,
        "reward_current_mean": float(np.mean(reward_cur)),
        "reward_reference_mean": float(np.mean(reward_ref)),
        "reward_delta_mean": float(np.mean(reward_delta)),
        "reward_delta_median": float(np.median(reward_delta)),
        "reward_delta_ci95": bootstrap_ci(reward_delta, n=500, seed=7),
        "reward_win_rate": float(reward_wins / n),
        "reward_tie_rate": float(reward_ties / n),
        "reward_loss_rate": float(reward_losses / n),
        "team_current_mean": float(np.mean(team_cur)),
        "team_reference_mean": float(np.mean(team_ref)),
        "team_delta_mean": float(np.mean(team_delta)),
        "opp_current_mean": float(np.mean(opp_cur)),
        "opp_reference_mean": float(np.mean(opp_ref)),
        "opp_delta_mean": float(np.mean(opp_delta)),
        "neu_current_mean": float(np.mean(neu_cur)),
        "neu_reference_mean": float(np.mean(neu_ref)),
        "neu_delta_mean": float(np.mean(neu_delta)),
        "assassin_rate_current": float(np.mean(assassin_cur)),
        "assassin_rate_reference": float(np.mean(assassin_ref)),
        "assassin_rate_delta": float(np.mean(assassin_delta)),
        "parse_valid_rate_current": float(np.mean(parse_cur)),
        "parse_valid_rate_reference": float(np.mean(parse_ref)),
        "parse_valid_rate_delta": float(np.mean(parse_delta)),
        "nonempty_clue_rate_current": float(np.mean(nonempty_cur)),
        "nonempty_clue_rate_reference": float(np.mean(nonempty_ref)),
        "nonempty_clue_rate_delta": float(np.mean(nonempty_delta)),
        "rejected_candidates_current_mean": float(np.mean(rejected_cur)),
        "rejected_candidates_reference_mean": float(np.mean(rejected_ref)),
        "rejected_candidates_delta_mean": float(np.mean(rejected_delta)),
        "directness_current_mean": float(np.mean(directness_cur)),
        "directness_reference_mean": float(np.mean(directness_ref)),
        "directness_delta_mean": float(np.mean(directness_delta)),
        "clue_changed_rate": float(clue_changed / n),
    }
    return out
