#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make repo root importable even when running: python scripts/dpo_pair_stats.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import load_yaml, read_jsonl


def mean(xs: List[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def std(xs: List[float]) -> float | None:
    if not xs:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def quantile(xs: List[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    i = (len(ys) - 1) * q
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return ys[lo]
    w = i - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def numeric_summary(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
        }
    return {
        "n": len(xs),
        "mean": mean(xs),
        "std": std(xs),
        "min": min(xs),
        "p25": quantile(xs, 0.25),
        "p50": quantile(xs, 0.50),
        "p75": quantile(xs, 0.75),
        "max": max(xs),
    }


def discrete_distribution(xs: List[float]) -> List[Dict[str, Any]]:
    if not xs:
        return []
    ctr = Counter(xs)
    total = len(xs)
    return [
        {"value": k, "count": v, "frac": v / total}
        for k, v in sorted(ctr.items(), key=lambda kv: kv[0])
    ]


def _valid_completion_text(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def reconstruct_dpo_pairs(
    turns_raw: List[Dict[str, Any]],
    *,
    margin: float,
    max_rejected_per_prompt: int,
) -> List[Dict[str, Any]]:
    """
    Reconstruct the same chosen/rejected pairing logic as src.build_dpo_pairs.build_pairs,
    while preserving rewards/stats for analysis.
    """
    out: List[Dict[str, Any]] = []
    max_rej = max(1, int(max_rejected_per_prompt))

    for r in turns_raw:
        prompt = r.get("prompt")
        chosen_text = r.get("completion")
        if not isinstance(prompt, str) or not _valid_completion_text(chosen_text):
            continue

        chosen_reward = float(r.get("reward", 0.0))
        chosen_stats = dict(r.get("stats") or {})

        rej_list: List[Tuple[float, Dict[str, Any]]] = []
        for c in (r.get("candidates") or []):
            try:
                rej_reward = float(c.get("reward", 0.0))
                rej_text = c.get("completion")
                if not _valid_completion_text(rej_text):
                    continue
                if (chosen_reward - rej_reward) >= float(margin):
                    rej_list.append((rej_reward, c))
            except Exception:
                continue

        if not rej_list:
            continue

        # Same ordering as src.build_dpo_pairs
        rej_list.sort(key=lambda t: t[0])

        for rej_reward, c in rej_list[:max_rej]:
            rejected_stats = dict(c.get("stats") or {})
            out.append(
                {
                    "prompt": prompt,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": float(rej_reward),
                    "chosen_stats": chosen_stats,
                    "rejected_stats": rejected_stats,
                    "reward_gap": chosen_reward - float(rej_reward),
                }
            )

    return out


def collect_metric(pairs: List[Dict[str, Any]], side: str, key: str) -> List[float]:
    assert side in ("chosen", "rejected")
    out: List[float] = []
    stats_key = f"{side}_stats"
    for p in pairs:
        stats = p.get(stats_key) or {}
        try:
            out.append(float(stats.get(key, 0.0)))
        except Exception:
            out.append(0.0)
    return out


def build_report(
    pairs: List[Dict[str, Any]],
    *,
    turns_raw_path: str,
    iter_n: int,
    margin: float,
    max_rejected_per_prompt: int,
) -> Dict[str, Any]:
    chosen_rewards = [float(p["chosen_reward"]) for p in pairs]
    rejected_rewards = [float(p["rejected_reward"]) for p in pairs]
    reward_gaps = [float(p["reward_gap"]) for p in pairs]

    metrics = ["n_team", "n_opp", "n_neu", "assassin"]

    metric_report: Dict[str, Any] = {}
    for m in metrics:
        chosen_vals = collect_metric(pairs, "chosen", m)
        rejected_vals = collect_metric(pairs, "rejected", m)
        metric_report[m] = {
            "chosen": {
                "average": mean(chosen_vals),
                "summary": numeric_summary(chosen_vals),
                "distribution": discrete_distribution(chosen_vals),
            },
            "rejected": {
                "average": mean(rejected_vals),
                "summary": numeric_summary(rejected_vals),
                "distribution": discrete_distribution(rejected_vals),
            },
        }

    return {
        "meta": {
            "iter": iter_n,
            "turns_raw_path": turns_raw_path,
            "num_pairs": len(pairs),
            "dpo_reward_margin": margin,
            "dpo_max_rejected_per_prompt": max_rejected_per_prompt,
            "weighting": "pair_weighted",
        },
        "reward": {
            "chosen": {
                "average": mean(chosen_rewards),
                "summary": numeric_summary(chosen_rewards),
                "distribution": discrete_distribution(chosen_rewards),
            },
            "rejected": {
                "average": mean(rejected_rewards),
                "summary": numeric_summary(rejected_rewards),
                "distribution": discrete_distribution(rejected_rewards),
            },
            "gap": {
                "average": mean(reward_gaps),
                "summary": numeric_summary(reward_gaps),
                "distribution": discrete_distribution(reward_gaps),
            },
        },
        "metrics": metric_report,
    }


def print_human_report(report: Dict[str, Any]) -> None:
    meta = report["meta"]
    reward = report["reward"]
    metrics = report["metrics"]

    print(f"iter: {meta['iter']}")
    print(f"turns_raw_path: {meta['turns_raw_path']}")
    print(f"num_pairs: {meta['num_pairs']}")
    print(f"dpo_reward_margin: {meta['dpo_reward_margin']}")
    print(f"dpo_max_rejected_per_prompt: {meta['dpo_max_rejected_per_prompt']}")
    print()

    print("Reward averages")
    print(f"  chosen:   {reward['chosen']['average']}")
    print(f"  rejected: {reward['rejected']['average']}")
    print(f"  gap:      {reward['gap']['average']}")
    print()

    print("Reward summaries")
    for side in ("chosen", "rejected", "gap"):
        s = reward[side]["summary"]
        print(
            f"  {side:8s} n={s['n']} mean={s['mean']} std={s['std']} "
            f"min={s['min']} p25={s['p25']} p50={s['p50']} p75={s['p75']} max={s['max']}"
        )
    print()

    for metric_name, payload in metrics.items():
        print(f"{metric_name}")
        for side in ("chosen", "rejected"):
            s = payload[side]["summary"]
            print(
                f"  {side:8s} avg={payload[side]['average']} "
                f"n={s['n']} min={s['min']} p25={s['p25']} p50={s['p50']} p75={s['p75']} max={s['max']}"
            )
        print("  distribution_chosen:", payload["chosen"]["distribution"])
        print("  distribution_rejected:", payload["rejected"]["distribution"])
        print()


def save_histogram(values: List[float], title: str, out_path: Path, bins: int = 30) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not values:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(values, bins=bins)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_bar_from_distribution(
    dist: List[Dict[str, Any]],
    title: str,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not dist:
        return

    xs = [d["value"] for d in dist]
    ys = [d["count"] for d in dist]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(xs, ys, width=0.8)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_plots(report: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    reward = report["reward"]
    metrics = report["metrics"]

    # Reward histograms
    for side in ("chosen", "rejected", "gap"):
        dist = reward[side]["distribution"]
        values: List[float] = []
        for d in dist:
            values.extend([float(d["value"])] * int(d["count"]))
        save_histogram(
            values,
            title=f"reward_{side}",
            out_path=out_dir / f"reward_{side}_hist.png",
            bins=30,
        )

    # Discrete count metrics as bar charts
    for metric_name, payload in metrics.items():
        for side in ("chosen", "rejected"):
            save_bar_from_distribution(
                payload[side]["distribution"],
                title=f"{metric_name}_{side}",
                out_path=out_dir / f"{metric_name}_{side}_bar.png",
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute chosen/rejected DPO-pair stats for a particular iteration."
    )
    ap.add_argument("--config", default="configs/default.yml")
    ap.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Iteration number. If set, overrides ITER env var before loading config.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON report.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print JSON report to stdout instead of human-readable text.",
    )
    ap.add_argument(
        "--plots-dir",
        default=None,
        help="Optional directory to save PNG distribution plots.",
    )
    args = ap.parse_args()

    if args.iter is not None:
        os.environ["ITER"] = str(args.iter)

    cfg = load_yaml(args.config)
    iter_n = int((cfg.get("iter", {}) or {}).get("n", 0))
    turns_raw_path = str(cfg["paths"]["turns_raw_path"])
    margin = float((cfg.get("training", {}) or {}).get("dpo_reward_margin", 2.0))
    max_rej = int((cfg.get("training", {}) or {}).get("dpo_max_rejected_per_prompt", 1))

    turns_raw = read_jsonl(turns_raw_path)
    pairs = reconstruct_dpo_pairs(
        turns_raw,
        margin=margin,
        max_rejected_per_prompt=max_rej,
    )

    report = build_report(
        pairs,
        turns_raw_path=turns_raw_path,
        iter_n=iter_n,
        margin=margin,
        max_rejected_per_prompt=max_rej,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.plots_dir:
        save_plots(report, Path(args.plots_dir))

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(report)
        if args.plots_dir:
            print(f"plots_dir: {args.plots_dir}")


if __name__ == "__main__":
    main()