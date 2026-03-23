from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

from .utils import load_yaml, read_jsonl, resolve_training_objective, write_jsonl


def _record_valid(record: Dict[str, Any]) -> bool:
    if "valid" in record:
        return bool(record.get("valid", False))
    clue_meta = record.get("clue_meta") or {}
    return bool(clue_meta.get("valid", False))


def _passes_chosen_filter(
    record: Dict[str, Any],
    filt: Dict[str, Any] | None,
) -> bool:
    if not filt:
        return True

    stats = record.get("stats") or {}
    reward = float(record.get("reward", 0.0))

    if bool(filt.get("require_valid", False)) and not _record_valid(record):
        return False

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


def _passes_rejected_filter(
    cand: Dict[str, Any],
    filt: Dict[str, Any] | None,
) -> bool:
    if not filt:
        return True

    stats = cand.get("stats") or {}
    reward = float(cand.get("reward", 0.0))

    if bool(filt.get("require_valid", False)) and not bool(cand.get("valid", False)):
        return False

    min_reward = filt.get("min_reward", None)
    if min_reward is not None and reward < float(min_reward):
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
    rejected_filter: Dict[str, Any] | None = None,
) -> List[Dict[str, str]]:
    """
    Flatten per-board records into TRL DPO pairs:
      {prompt: str, chosen: str, rejected: str}

    chosen = r["completion"]
    rejected = candidates that:
      1) pass rejected_filter
      2) satisfy (chosen_reward - cand_reward) >= margin

    Among eligible rejecteds, keep the hardest ones first:
      highest reward first, capped per prompt.

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
                if not _passes_rejected_filter(c, rejected_filter):
                    continue

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

        # hardest eligible negatives first
        rej_list.sort(key=lambda t: t[0], reverse=True)

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
    rejected_filter: Dict[str, Any] | None,
) -> List[Dict[str, str]]:
    rows = read_jsonl(turns_raw_path)
    return build_pairs(
        rows,
        margin=margin,
        max_rejected_per_prompt=max_rejected_per_prompt,
        chosen_filter=chosen_filter,
        rejected_filter=rejected_filter,
    )


def _n_tokens(tokenizer: Any, text: str, *, add_special_tokens: bool) -> int:
    ids = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    return int(len(ids))


def _filter_overlong_pairs(
    pairs: List[Dict[str, str]],
    *,
    tokenizer: Any,
    max_length: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Drop DPO pairs that would exceed TRL's max_length instead of letting TRL truncate.

    We track:
      - completion-only lengths (helps debug overly verbose responses)
      - prompt+chosen and prompt+rejected lengths (actual training-time constraint)
    """
    kept: List[Dict[str, str]] = []
    stats: Dict[str, Any] = {
        "max_length": int(max_length),
        "input_pairs": int(len(pairs)),
        "kept_pairs": 0,
        "dropped_pairs_total": 0,
        "dropped_prompt_plus_chosen": 0,
        "dropped_prompt_plus_rejected": 0,
        "chosen_completion_gt_max": 0,
        "rejected_completion_gt_max": 0,
        "max_prompt_plus_chosen_tokens_seen": 0,
        "max_prompt_plus_rejected_tokens_seen": 0,
        "max_chosen_completion_tokens_seen": 0,
        "max_rejected_completion_tokens_seen": 0,
    }

    for pair in pairs:
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]

        chosen_completion_len = _n_tokens(tokenizer, chosen, add_special_tokens=False)
        rejected_completion_len = _n_tokens(tokenizer, rejected, add_special_tokens=False)
        chosen_pair_len = _n_tokens(tokenizer, prompt + chosen, add_special_tokens=True)
        rejected_pair_len = _n_tokens(tokenizer, prompt + rejected, add_special_tokens=True)

        stats["max_chosen_completion_tokens_seen"] = max(
            int(stats["max_chosen_completion_tokens_seen"]), chosen_completion_len
        )
        stats["max_rejected_completion_tokens_seen"] = max(
            int(stats["max_rejected_completion_tokens_seen"]), rejected_completion_len
        )
        stats["max_prompt_plus_chosen_tokens_seen"] = max(
            int(stats["max_prompt_plus_chosen_tokens_seen"]), chosen_pair_len
        )
        stats["max_prompt_plus_rejected_tokens_seen"] = max(
            int(stats["max_prompt_plus_rejected_tokens_seen"]), rejected_pair_len
        )

        overlong = False

        if chosen_completion_len > max_length:
            stats["chosen_completion_gt_max"] += 1
            overlong = True
        if rejected_completion_len > max_length:
            stats["rejected_completion_gt_max"] += 1
            overlong = True

        if chosen_pair_len > max_length:
            stats["dropped_prompt_plus_chosen"] += 1
            overlong = True
        if rejected_pair_len > max_length:
            stats["dropped_prompt_plus_rejected"] += 1
            overlong = True

        if overlong:
            stats["dropped_pairs_total"] += 1
            continue

        kept.append(pair)

    stats["kept_pairs"] = int(len(kept))
    return kept, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--turns-raw", default=None, help="Override paths.turns_raw_path for current iter")
    ap.add_argument("--out", default=None, help="Override paths.dpo_pairs_path")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    objective = resolve_training_objective(cfg)
    if objective == "sft":
        raise SystemExit(
            "build_dpo_pairs is disabled for training.objective=sft. "
            "Set training.objective=dpo before building DPO pairs."
        )

    tcfg = cfg.get("training", {}) or {}

    turns_raw_path = args.turns_raw or cfg["paths"]["turns_raw_path"]
    out_path = args.out or cfg["paths"]["dpo_pairs_path"]

    margin = float(tcfg.get("dpo_reward_margin", 2.0))
    max_rej = int(tcfg.get("dpo_max_rejected_per_prompt", 1))
    max_len = int(tcfg.get("trl_max_length", tcfg.get("max_seq_len", 4096)))
    chosen_filter = dict(tcfg.get("dpo_chosen_filter") or {})
    rejected_filter = dict(tcfg.get("dpo_rejected_filter") or {})

    cur_iter = int((cfg.get("iter", {}) or {}).get("n", 0))
    mix_previous = bool(tcfg.get("dpo_mix_previous_pairs", False))
    mix_num_previous = tcfg.get("dpo_mix_num_previous_iters", None)

    current_pairs = _pairs_from_turns_raw(
        turns_raw_path,
        margin=margin,
        max_rejected_per_prompt=max_rej,
        chosen_filter=chosen_filter,
        rejected_filter=rejected_filter,
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
                rejected_filter=rejected_filter,
            )
            all_pairs.extend(prev_pairs)
            prev_added += len(prev_pairs)
            print(f"Included {len(prev_pairs)} filtered previous pairs from iter {prev_iter} -> {prev_turns_raw_path}")

    tok = AutoTokenizer.from_pretrained(
        cfg["models"]["spymaster_model_id"],
        use_fast=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    all_pairs_before_length_filter = len(all_pairs)
    all_pairs, length_stats = _filter_overlong_pairs(
        all_pairs,
        tokenizer=tok,
        max_length=max_len,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, all_pairs)

    stats_path = Path(out_path).with_suffix(Path(out_path).suffix + ".stats.json")
    stats_payload = {
        "out_path": str(out_path),
        "turns_raw_path": str(turns_raw_path),
        "current_iter_pairs_before_length_filter": int(len(current_pairs)),
        "previous_iter_pairs_added_before_length_filter": int(prev_added),
        "all_pairs_before_length_filter": int(all_pairs_before_length_filter),
        **length_stats,
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(all_pairs)} DPO pairs -> {out_path}")
    print(f"  current iter pairs kept before length filter: {len(current_pairs)}")
    if mix_previous:
        print(f"  previous iter pairs added before length filter: {prev_added}")
    print(f"  dropped for length: {length_stats['dropped_pairs_total']}")
    print(f"  length stats: {stats_path}")


if __name__ == "__main__":
    main()
