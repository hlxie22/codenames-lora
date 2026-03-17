from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .utils import load_yaml, read_jsonl, write_jsonl


def _passes_chosen_filter(
    record: Dict[str, Any],
    filt: Dict[str, Any] | None,
) -> bool:
    if not filt:
        return True

    stats = record.get("stats") or {}
    reward = float(record.get("reward", 0.0))

    min_reward = filt.get("min_reward", None)
    if min_reward is not None and reward < float(min_reward):
        return False

    min_team = filt.get("min_team_correct", None)
    if min_team is not None and int(stats.get("n_team", 0)) < int(min_team):
        return False

    max_opp = filt.get("max_opp_wrong", None)
    if max_opp is not None and int(stats.get("n_opp", 0)) > int(max_opp):
        return False

    max_neu = filt.get("max_neu_wrong", None)
    if max_neu is not None and int(stats.get("n_neu", 0)) > int(max_neu):
        return False

    if bool(filt.get("require_no_assassin", False)) and int(stats.get("assassin", 0)) > 0:
        return False

    return True


def build_pairs(
    turns_raw: List[Dict[str, Any]],
    *,
    margin: float,
    max_rejected_per_prompt: int = 1,
    chosen_filter: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    """
    Flatten per-board records into TRL DPO pairs:
      {prompt: str, chosen: str, rejected: str}

    chosen = r["completion"]
    rejected = candidates with (chosen_reward - cand_reward) >= margin
               sorted by lowest reward first, capped per prompt

    Only keeps prompts whose chosen response passes chosen_filter.
    """
    out: List[Dict[str, str]] = []
    max_rej = max(1, int(max_rejected_per_prompt))

    for r in turns_raw:
        prompt = r.get("prompt")
        chosen = r.get("completion")
        if not isinstance(prompt, str) or not isinstance(chosen, str):
            continue

        if not _passes_chosen_filter(r, chosen_filter):
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


def _load_cfg_for_iter(config_path: str, iter_n: int) -> Dict[str, Any]:
    prev_iter_env = os.environ.get("ITER")
    try:
        os.environ["ITER"] = str(iter_n)
        return load_yaml(config_path)
    finally:
        if prev_iter_env is None:
            os.environ.pop("ITER", None)
        else:
            os.environ["ITER"] = prev_iter_env


def _pairs_from_turns_raw(
    turns_raw_path: str,
    *,
    margin: float,
    max_rejected_per_prompt: int,
    chosen_filter: Dict[str, Any] | None,
) -> List[Dict[str, str]]:
    rows = read_jsonl(turns_raw_path)
    return build_pairs(
        rows,
        margin=margin,
        max_rejected_per_prompt=max_rejected_per_prompt,
        chosen_filter=chosen_filter,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--turns-raw", default=None, help="Override paths.turns_raw_path for current iter")
    ap.add_argument("--out", default=None, help="Override paths.dpo_pairs_path")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg.get("training", {}) or {}

    turns_raw_path = args.turns_raw or cfg["paths"]["turns_raw_path"]
    out_path = args.out or cfg["paths"]["dpo_pairs_path"]

    margin = float(tcfg.get("dpo_reward_margin", 2.0))
    max_rej = int(tcfg.get("dpo_max_rejected_per_prompt", 1))
    chosen_filter = dict(tcfg.get("dpo_chosen_filter") or {})

    cur_iter = int((cfg.get("iter", {}) or {}).get("n", 0))
    mix_previous = bool(tcfg.get("dpo_mix_previous_pairs", False))
    mix_num_previous = tcfg.get("dpo_mix_num_previous_iters", None)

    current_pairs = _pairs_from_turns_raw(
        turns_raw_path,
        margin=margin,
        max_rejected_per_prompt=max_rej,
        chosen_filter=chosen_filter,
    )
    all_pairs = list(current_pairs)

    prev_added = 0
    if mix_previous and cur_iter > 0:
        if mix_num_previous is None:
            start_iter = 0
        else:
            start_iter = max(0, cur_iter - int(mix_num_previous))

        for prev_iter in range(start_iter, cur_iter):
            prev_cfg = _load_cfg_for_iter(args.config, prev_iter)
            prev_turns_raw_path = prev_cfg["paths"]["turns_raw_path"]

            if not Path(prev_turns_raw_path).exists():
                print(f"[warn] Previous turns_raw missing for iter {prev_iter}: {prev_turns_raw_path}")
                continue

            prev_pairs = _pairs_from_turns_raw(
                prev_turns_raw_path,
                margin=margin,
                max_rejected_per_prompt=max_rej,
                chosen_filter=chosen_filter,
            )
            all_pairs.extend(prev_pairs)
            prev_added += len(prev_pairs)
            print(f"Included {len(prev_pairs)} filtered previous pairs from iter {prev_iter} -> {prev_turns_raw_path}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, all_pairs)

    print(f"Wrote {len(all_pairs)} DPO pairs -> {out_path}")
    print(f"  current iter pairs kept: {len(current_pairs)}")
    if mix_previous:
        print(f"  previous iter pairs added: {prev_added}")


if __name__ == "__main__":
    main()