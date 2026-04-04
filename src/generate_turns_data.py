# src/generate_turns_data.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .io_utils import shard_path_append_to_suffix, merge_jsonl_shards
from .model_wrappers import Embedder, build_codenames_generators
from .mp_utils import launch_children, is_child_process, child_shard_info
from .prompting import render_prompt, get_spymaster_reasoning_mode
from .rollout import run_turns_batched
from .spymaster_prompt import build_spymaster_messages
from .think_utils import split_think_and_rest
from .utils import (
    load_yaml,
    read_jsonl,
    resolve_training_objective,
    set_global_seed,
    write_jsonl,
    save_progress,
)


# -------------------------
# Rollout generator helpers
# -------------------------

def _rollout_adapter_dir(cfg: Dict[str, Any]) -> Optional[str]:
    """
    Rollout adapter dir comes from inference.rollout_adapter_dir.
    (load_yaml() will set it to None when iter.use_trained=false.)
    """
    inf = cfg.get("inference", {}) or {}
    ad = inf.get("rollout_adapter_dir")
    if not ad:
        return None
    return str(ad)


def build_rollout_generators(cfg: Dict[str, Any]):
    """
    Returns (spymaster_generator, guesser_generator) for rollouts.
    If rollout_adapter_dir exists, apply it to the spymaster only.
    """
    adapter_dir = _rollout_adapter_dir(cfg)
    if adapter_dir and not Path(adapter_dir).exists():
        adapter_dir = None
    return build_codenames_generators(cfg, spymaster_adapter_dir=adapter_dir)


# -------------------------
# Resume helpers
# -------------------------

def load_done_example_ids(raw_path: str | Path) -> set[str]:
    """Return example_ids already present in an existing raw jsonl (for resume)."""
    raw_path = Path(raw_path)
    done: set[str] = set()
    if not raw_path.exists():
        return done

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                eid = obj.get("example_id") or obj.get("board_id")
                if eid:
                    done.add(str(eid))
            except Exception:
                continue
    return done


def _try_load_progress(progress_path: Path) -> Optional[Dict[str, Any]]:
    try:
        if progress_path.exists():
            return json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _final_answer_completion(clue: str, num: int) -> str:
    return f"CLUE: {clue}\nNUM: {int(num)}\n"


def _safe_completion_raw(raw_spymaster_text: str, clue: str, num: int) -> str:
    raw = (raw_spymaster_text or "").strip()
    if raw:
        return raw
    return _final_answer_completion(clue, num).strip()


def _spymaster_completion_fields(
    cfg: Dict[str, Any],
    *,
    raw_spymaster_text: str,
    clue: str,
    num: int,
) -> Dict[str, Any]:
    """
    Normalize stored completions so native Qwen3 reasoning traces stay in raw fields,
    while the training-facing `completion` field stays visible/final-only.
    """
    raw = _safe_completion_raw(raw_spymaster_text, clue, num)
    final = _final_answer_completion(clue, int(num)).strip()
    reasoning_trace, visible = split_think_and_rest(raw, which="first", allow_partial=True)
    visible = visible.strip() or final

    mode = get_spymaster_reasoning_mode(cfg)
    if mode == "visible":
        completion = raw
    else:
        completion = final

    out: Dict[str, Any] = {
        "completion": completion,
        "completion_raw": raw,
        "completion_final": final,
        "visible_completion": visible,
    }
    if reasoning_trace:
        out["reasoning_trace"] = reasoning_trace
    return out


# -------------------------
# Filtering / materialization
# -------------------------

def _record_valid(record: Dict[str, Any]) -> bool:
    if "valid" in record:
        return bool(record.get("valid", False))
    clue_meta = record.get("clue_meta") or {}
    return bool(clue_meta.get("valid", False))


def _passes_structured_filter(
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


def _candidate_sort_key(row: Dict[str, Any]) -> tuple[float, int, int, int, int]:
    stats = row.get("stats") or {}
    return (
        float(row.get("reward", 0.0)),
        int(stats.get("n_team", 0)),
        -int(stats.get("assassin", 0)),
        -int(stats.get("n_opp", 0)),
        -int(stats.get("n_neu", 0)),
    )


def _promote_candidate_to_parent(parent: Dict[str, Any], cand: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(parent)

    for k in (
        "clue",
        "num",
        "parse_valid",
        "valid",
        "rejection_reason",
        "completion",
        "completion_raw",
        "completion_final",
        "visible_completion",
        "reasoning_trace",
        "raw_spymaster_text",
        "raw_guesser_text",
        "reward",
        "guess_words",
        "sft_uplift_raw",
        "sft_score",
        "sft_rank",
    ):
        if k in cand:
            out[k] = cand[k]

    out["stats"] = {
        **(cand.get("stats") or {}),
        "directness": float(cand.get("directness", 0.0)),
    }

    clue_meta = dict(parent.get("clue_meta") or {})
    clue_meta.update(
        {
            "clue": cand.get("clue", clue_meta.get("clue")),
            "num": int(cand.get("num", clue_meta.get("num", 0) or 0)),
            "parse_valid": bool(cand.get("parse_valid", clue_meta.get("parse_valid", False))),
            "valid": bool(cand.get("valid", clue_meta.get("valid", False))),
        }
    )
    out["clue_meta"] = clue_meta
    out["debug"] = {"guess_words": list(cand.get("guess_words") or [])}
    return out


def _score_sft_candidates_against_raw_pool(
    parent: Dict[str, Any],
    cfg: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_pool = [dict(c) for c in list(parent.get("candidates") or [])]
    if not raw_pool:
        return [], {
            "raw_pool_size": 0,
            "raw_reward_mean": 0.0,
            "raw_reward_std": 0.0,
            "raw_valid_frac": 0.0,
        }

    rewards = [float(c.get("reward", 0.0)) for c in raw_pool]
    raw_mean = float(np.mean(rewards)) if rewards else 0.0
    raw_std = float(np.std(rewards)) if rewards else 0.0
    raw_valid_frac = (
        float(np.mean([1.0 if _record_valid(c) else 0.0 for c in raw_pool]))
        if raw_pool else 0.0
    )

    creativity_cfg = ((cfg.get("training", {}) or {}).get("sft_creativity") or {})
    uplift_coef = float(creativity_cfg.get("uplift_coef", 1.0))
    std_coef = float(creativity_cfg.get("std_coef", 0.0))

    scored: List[Dict[str, Any]] = []
    for cand in raw_pool:
        cand_reward = float(cand.get("reward", 0.0))
        cand_uplift_raw = cand_reward - raw_mean
        sft_score = uplift_coef * cand_uplift_raw + std_coef * raw_std

        scored_cand = dict(cand)
        scored_cand["sft_uplift_raw"] = float(cand_uplift_raw)
        scored_cand["sft_score"] = float(sft_score)
        scored.append(scored_cand)

    raw_stats = {
        "raw_pool_size": int(len(raw_pool)),
        "raw_reward_mean": float(raw_mean),
        "raw_reward_std": float(raw_std),
        "raw_valid_frac": float(raw_valid_frac),
    }
    return scored, raw_stats


def _apply_sft_creativity_gate(
    cands: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    creativity_cfg = ((cfg.get("training", {}) or {}).get("sft_creativity") or {})
    if not bool(creativity_cfg.get("enabled", True)):
        return [dict(c) for c in cands]

    min_score = float(creativity_cfg.get("min_score", 0.0))
    return [dict(c) for c in cands if float(c.get("sft_score", 0.0)) >= min_score]


def _apply_sft_structured_filter(
    cands: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sft_filter = dict((cfg.get("training", {}) or {}).get("sft_filter") or {})
    return [dict(c) for c in cands if _passes_structured_filter(c, sft_filter)]


def _sort_sft_pool(
    cands: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    creativity_cfg = ((cfg.get("training", {}) or {}).get("sft_creativity") or {})
    sort_by_score = bool(creativity_cfg.get("sort_by_score", False))

    def _score_key(row: Dict[str, Any]) -> tuple[float, float, float, int, int, int, int]:
        stats = row.get("stats") or {}
        return (
            float(row.get("sft_score", 0.0)),
            float(row.get("reward", 0.0)),
            float(row.get("reward", 0.0)),
            int(stats.get("n_team", 0)),
            -int(stats.get("assassin", 0)),
            -int(stats.get("n_opp", 0)),
            -int(stats.get("n_neu", 0)),
        )

    sorted_pool = sorted(
        [dict(c) for c in cands],
        key=_score_key if sort_by_score else _candidate_sort_key,
        reverse=True,
    )

    for idx, cand in enumerate(sorted_pool):
        cand["sft_rank"] = int(idx)

    return sorted_pool


def _build_sft_pool_row(
    parent: Dict[str, Any],
    final_pool: List[Dict[str, Any]],
    raw_stats: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    default_idx = 0
    default_cand = final_pool[default_idx]

    row = {
        "example_id": str(parent.get("example_id") or parent.get("board_id") or ""),
        "board_id": str(parent.get("board_id") or parent.get("example_id") or ""),
        "prompt": parent.get("prompt"),
    }
    row = _promote_candidate_to_parent(row, default_cand)

    row["sft_pool"] = [dict(c) for c in final_pool]
    row["sft_pool_size"] = int(len(final_pool))
    row["sft_default_idx"] = int(default_idx)

    row["sft_raw_pool_size"] = int(raw_stats.get("raw_pool_size", 0))
    row["sft_raw_reward_mean"] = float(raw_stats.get("raw_reward_mean", 0.0))
    row["sft_raw_reward_std"] = float(raw_stats.get("raw_reward_std", 0.0))
    row["sft_raw_valid_frac"] = float(raw_stats.get("raw_valid_frac", 0.0))

    # Backward-compatibility mirrors for one transition cycle.
    row["candidate_reward_mean"] = float(raw_stats.get("raw_reward_mean", 0.0))
    row["candidate_reward_std"] = float(raw_stats.get("raw_reward_std", 0.0))
    row["candidate_valid_frac"] = float(raw_stats.get("raw_valid_frac", 0.0))
    row["best_minus_avg_reward"] = float(default_cand.get("sft_uplift_raw", 0.0))

    for stale_key in ("candidates", "best_candidate_idx"):
        row.pop(stale_key, None)

    return row


def filter_examples(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    DPO/default:
      - keep old board-level filtering behavior from cfg["filtering"]

    SFT:
      - materialize canonical board-level SFT pools in turns.jsonl
      - score creativity relative to the full raw candidate distribution per board
      - apply creativity threshold first, then hard SFT filtering
      - drop boards with no final SFT-approved completions
    """
    if not records:
        return []

    objective = resolve_training_objective(cfg)

    if objective == "sft":
        kept: List[Dict[str, Any]] = []

        for parent in records:
            scored_candidates, raw_stats = _score_sft_candidates_against_raw_pool(parent, cfg)
            if not scored_candidates:
                continue

            creativity_survivors = _apply_sft_creativity_gate(scored_candidates, cfg)
            if not creativity_survivors:
                continue

            final_pool = _apply_sft_structured_filter(creativity_survivors, cfg)
            if not final_pool:
                continue

            final_pool = _sort_sft_pool(final_pool, cfg)
            kept.append(_build_sft_pool_row(parent, final_pool, raw_stats, cfg))

        return kept

    # ---- existing default / DPO behavior ----
    fcfg = cfg["filtering"]
    mode = fcfg["mode"]

    min_reward = fcfg.get("min_reward", None)
    min_reward = float(min_reward) if min_reward is not None else None

    def passes_min_reward(r: Dict[str, Any]) -> bool:
        if min_reward is None:
            return True
        return float(r.get("reward", 0.0)) >= min_reward

    if mode == "top_percent":
        top_p = float(fcfg["top_percent"])
        rewards = np.array([r["reward"] for r in records], dtype=np.float32)
        thr = float(np.quantile(rewards, 1.0 - top_p))
        return [r for r in records if float(r["reward"]) >= thr and passes_min_reward(r)]

    if mode == "rule_based":
        min_team = int(fcfg.get("min_team_correct", 0))
        require_no_assassin = bool(fcfg.get("require_no_assassin", False))
        kept = []
        for r in records:
            if not passes_min_reward(r):
                continue
            st = r.get("stats", {})
            if int(st.get("n_team", 0)) < min_team:
                continue
            if require_no_assassin and int(st.get("assassin", 0)) > 0:
                continue
            kept.append(r)
        return kept

    raise ValueError(f"Unknown filtering mode: {mode}")


def _merge_shards_to_base(base_raw_path: str | Path, num_shards: int) -> List[Dict[str, Any]]:
    shard_paths = [shard_path_append_to_suffix(base_raw_path, sid, num_shards) for sid in range(num_shards)]
    combined = merge_jsonl_shards(shard_paths)
    write_jsonl(base_raw_path, combined)
    return combined


def _cfg_turns_paths(cfg: Dict[str, Any]) -> tuple[str, str]:
    paths = cfg.get("paths", {}) or {}
    turns_path = paths.get("turns_path") or paths.get("sft_turns_path")
    turns_raw_path = paths.get("turns_raw_path") or paths.get("sft_turns_raw_path")
    if not turns_path or not turns_raw_path:
        raise RuntimeError(
            "Config must set paths.turns_path and paths.turns_raw_path "
            "(or legacy paths.sft_turns_path / paths.sft_turns_raw_path)."
        )
    return str(turns_path), str(turns_raw_path)


# -------------------------
# SLURM array helpers (job-level sharding)
# -------------------------

def _slurm_array_info() -> tuple[int, int]:
    tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if tid is None:
        return (0, 1)
    sid = int(tid)

    tcount = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if tcount is not None:
        return (sid, max(1, int(tcount)))

    tmin = os.environ.get("SLURM_ARRAY_TASK_MIN")
    tmax = os.environ.get("SLURM_ARRAY_TASK_MAX")
    if tmin is not None and tmax is not None:
        n = int(tmax) - int(tmin) + 1
        return (sid, max(1, n))

    return (sid, 1)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--job-shard-id", type=int, default=None, help="Override SLURM_ARRAY_TASK_ID")
    ap.add_argument("--job-num-shards", type=int, default=None, help="Override SLURM_ARRAY_TASK_COUNT")
    ap.add_argument("--no-merge", action="store_true", help="Do not merge/filter (write only shard outputs).")

    args = ap.parse_args()

    cfg = load_yaml(args.config)
    turns_path, turns_raw_path = _cfg_turns_paths(cfg)

    num_procs = int(cfg.get("inference", {}).get("num_processes", 1))
    batch_size = int(cfg.get("inference", {}).get("batch_size", 1))

    sl_sid, sl_n = _slurm_array_info()
    job_sid = int(args.job_shard_id) if args.job_shard_id is not None else int(sl_sid)
    job_n = int(args.job_num_shards) if args.job_num_shards is not None else int(sl_n)

    if job_n < 1:
        job_n = 1
    if job_sid < 0 or job_sid >= job_n:
        raise ValueError(f"Invalid job shard {job_sid} for job_num_shards={job_n}")

    if num_procs > 1 and not is_child_process():
        if cfg.get("inference", {}).get("backend") == "vllm":
            tp = int(cfg.get("inference", {}).get("vllm", {}).get("tensor_parallel_size", 1))
            if tp != 1:
                print(
                    f"[warn] inference.vllm.tensor_parallel_size={tp} with inference.num_processes={num_procs}. "
                    f"Usually you want tensor_parallel_size=1 when using multiple processes."
                )

        total_shards = int(job_n) * int(num_procs)
        shard_base = int(job_sid) * int(num_procs)

        launch_children(
            "src.generate_turns_data",
            ["--config", args.config],
            num_procs,
            shard_base=shard_base,
            total_shards=total_shards,
        )

        if job_n > 1 or bool(args.no_merge):
            print(
                f"[master] finished job shard {job_sid}/{job_n} "
                f"(children covered global shards {shard_base}..{shard_base + num_procs - 1} of {total_shards}). "
                f"Skipping merge/filter."
            )
            return

        combined = _merge_shards_to_base(turns_raw_path, total_shards)
        filtered = filter_examples(combined, cfg)
        write_jsonl(turns_path, filtered)
        print(f"Merged {len(combined)} raw -> Filtered {len(filtered)} -> {turns_path}")
        return

    set_global_seed(int(cfg["training"].get("seed", 0)))

    boards = read_jsonl(cfg["paths"]["boards_train_path"])

    maxb = cfg.get("boards", {}).get("sft_max_train_boards", None)
    if maxb is not None:
        boards = boards[: int(maxb)]
        print(f"Using only first {len(boards)} train boards (boards.sft_max_train_boards={maxb}).")

    shard_id, num_shards = child_shard_info() if is_child_process() else (job_sid, job_n)

    if num_shards > 1:
        boards = [b for i, b in enumerate(boards) if (i % num_shards) == shard_id]
        print(f"[shard {shard_id}/{num_shards}] boards={len(boards)}")

    shard_total = len(boards)

    raw_path = Path(turns_raw_path)
    if num_shards > 1:
        raw_path = shard_path_append_to_suffix(raw_path, shard_id, num_shards)

    progress_path = raw_path.with_suffix(raw_path.suffix + ".progress.json")

    done_ids = load_done_example_ids(raw_path)
    if done_ids:
        before = len(boards)
        boards = [b for b in boards if str(b.get("board_id")) not in done_ids]
        after = len(boards)
        print(
            f"[shard {shard_id}/{num_shards}] Resuming: found {len(done_ids)} already-written examples in {raw_path}. "
            f"Remaining={after} (was {before}), shard_total={shard_total}."
        )
    else:
        print(f"[shard {shard_id}/{num_shards}] Starting fresh: shard_total={shard_total} -> {raw_path}")

    prev_prog = _try_load_progress(progress_path) or {}
    prev_mean = prev_prog.get("mean_reward_running", None)
    prev_done = prev_prog.get("done", None)

    running_sum = 0.0
    running_n = 0
    uplift_sum = 0.0
    uplift_n = 0
    cand_avg_sum = 0.0
    cand_std_sum = 0.0
    cand_valid_frac_sum = 0.0
    uplift_valid_sum = 0.0
    uplift_valid_n = 0
    cand_avg_valid_sum = 0.0

    try:
        if prev_mean is not None and prev_done is not None and int(prev_done) == len(done_ids):
            running_n = int(prev_done)
            running_sum = float(prev_mean) * float(prev_done)

            prev_cand_avg = prev_prog.get("mean_candidate_reward_running", None)
            prev_uplift = prev_prog.get("mean_best_minus_avg_reward_running", None)
            prev_cand_std = prev_prog.get("candidate_reward_std_running", None)
            prev_valid_frac = prev_prog.get("candidate_valid_frac_running", None)

            if prev_cand_avg is not None and prev_uplift is not None:
                uplift_n = int(prev_done)
                cand_avg_sum = float(prev_cand_avg) * float(prev_done)
                uplift_sum = float(prev_uplift) * float(prev_done)

            if prev_cand_std is not None:
                cand_std_sum = float(prev_cand_std) * float(prev_done)

            if prev_valid_frac is not None:
                cand_valid_frac_sum = float(prev_valid_frac) * float(prev_done)

            prev_cand_avg_valid = prev_prog.get("mean_candidate_reward_valid_running", None)
            prev_uplift_valid = prev_prog.get("mean_best_minus_avg_reward_valid_running", None)
            prev_uplift_valid_done = prev_prog.get("uplift_valid_done", None)
            if (
                prev_cand_avg_valid is not None
                and prev_uplift_valid is not None
                and prev_uplift_valid_done is not None
            ):
                uplift_valid_n = int(prev_uplift_valid_done)
                cand_avg_valid_sum = float(prev_cand_avg_valid) * float(uplift_valid_n)
                uplift_valid_sum = float(prev_uplift_valid) * float(uplift_valid_n)
    except Exception:
        running_sum = 0.0
        running_n = 0
        uplift_sum = 0.0
        uplift_n = 0
        cand_avg_sum = 0.0
        cand_std_sum = 0.0
        cand_valid_frac_sum = 0.0
        uplift_valid_sum = 0.0
        uplift_valid_n = 0
        cand_avg_valid_sum = 0.0

    save_progress(
        progress_path,
        done=len(done_ids),
        total=shard_total,
        last_example_id=prev_prog.get("last_example_id"),
        last_board_id=prev_prog.get("last_board_id"),
        mean_reward=(running_sum / running_n) if running_n else None,
        extra={
            "status": "running",
            "n_candidates": int(cfg["decoding"]["n_candidates"]),
            "max_resamples": int(cfg["decoding"]["max_resamples"]),
            "batch_size": int(batch_size),
            "shard_id": int(shard_id),
            "num_shards": int(num_shards),
            "preserve_spymaster_raw_text": True,
            "uplift_done": int(uplift_n),
            "mean_candidate_reward_running": (cand_avg_sum / uplift_n) if uplift_n else None,
            "mean_best_minus_avg_reward_running": (uplift_sum / uplift_n) if uplift_n else None,
            "candidate_reward_std_running": (cand_std_sum / uplift_n) if uplift_n else None,
            "candidate_valid_frac_running": (cand_valid_frac_sum / uplift_n) if uplift_n else None,
            "uplift_valid_done": int(uplift_valid_n),
            "mean_candidate_reward_valid_running": (cand_avg_valid_sum / uplift_valid_n) if uplift_valid_n else None,
            "mean_best_minus_avg_reward_valid_running": (uplift_valid_sum / uplift_valid_n) if uplift_valid_n else None,
        },
    )

    spymaster, guesser = build_rollout_generators(cfg)

    use_embed = bool(cfg.get("constraints", {}).get("enable_directness_check", True))
    embedder = None
    if use_embed:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        embedder = Embedder(cfg["models"]["embedding_model_id"], device=device)

    n_candidates = int(cfg["decoding"]["n_candidates"])
    progress_every = int(cfg.get("decoding", {}).get("progress_every", 50))

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    f_raw = open(raw_path, "a", encoding="utf-8")
    writes_since_flush = 0
    FLUSH_EVERY = int(cfg.get("decoding", {}).get("flush_every", 10))

    try:
        if not boards and len(done_ids) >= shard_total:
            print(f"[shard {shard_id}/{num_shards}] Nothing to do: done={len(done_ids)}/{shard_total}")
        else:
            for start in range(0, len(boards), max(1, batch_size)):
                batch = boards[start: start + max(1, batch_size)]

                bests, metas, all_cands = run_turns_batched(
                    batch,
                    spymaster,
                    guesser,
                    embedder,
                    cfg,
                    n_candidates=n_candidates,
                    return_candidates=True,
                )

                for b, best, meta, cands in zip(batch, bests, metas, all_cands):
                    example_id = str(b["board_id"])
                    if example_id in done_ids:
                        continue

                    revealed = [False] * len(b["board_words"])
                    msgs = build_spymaster_messages(b["board_words"], b["labels"], revealed, cfg)
                    prompt = render_prompt(spymaster, msgs, cfg, role="spymaster")

                    best_completion_fields = _spymaster_completion_fields(
                        cfg,
                        raw_spymaster_text=best.raw_spymaster_text,
                        clue=best.clue,
                        num=int(best.num),
                    )

                    best_idx = None
                    for j, c in enumerate(cands):
                        if c is best:
                            best_idx = j
                            break
                    if best_idx is None:
                        for j, c in enumerate(cands):
                            if (
                                c.clue == best.clue
                                and int(c.num) == int(best.num)
                                and float(c.reward) == float(best.reward)
                                and bool(c.valid) == bool(best.valid)
                            ):
                                best_idx = j
                                break

                    cand_rewards = [float(c.reward) for c in cands] if cands else []
                    avg_all = float(np.mean(cand_rewards)) if cand_rewards else 0.0
                    std_all = float(np.std(cand_rewards)) if cand_rewards else 0.0
                    uplift_all = float(best.reward) - float(avg_all)

                    valid_rewards = [float(c.reward) for c in cands if bool(getattr(c, "valid", False))] if cands else []
                    avg_valid = float(np.mean(valid_rewards)) if valid_rewards else None
                    uplift_valid = (float(best.reward) - float(avg_valid)) if avg_valid is not None else None

                    valid_frac = (
                        float(np.mean([1.0 if bool(getattr(c, "valid", False)) else 0.0 for c in cands]))
                        if cands else 0.0
                    )

                    cand_payload: List[Dict[str, Any]] = []
                    for c in cands:
                        cand_completion_fields = _spymaster_completion_fields(
                            cfg,
                            raw_spymaster_text=c.raw_spymaster_text,
                            clue=c.clue,
                            num=int(c.num),
                        )
                        cd: Dict[str, Any] = {
                            "clue": c.clue,
                            "num": int(c.num),
                            "parse_valid": bool(c.parse_valid),
                            "valid": bool(c.valid),
                            "rejection_reason": c.rejection_reason,
                            "directness": float(c.directness),
                            "reward": float(c.reward),
                            "stats": c.stats,
                            "guess_words": c.guess_words,
                            **cand_completion_fields,
                            "raw_spymaster_text": c.raw_spymaster_text,
                            "raw_guesser_text": c.raw_guesser_text,
                        }
                        cand_payload.append(cd)

                    rec = {
                        "example_id": example_id,
                        "board_id": str(b["board_id"]),
                        "prompt": prompt,
                        **best_completion_fields,
                        "raw_spymaster_text": best.raw_spymaster_text,
                        "raw_guesser_text": best.raw_guesser_text,
                        "reward": float(best.reward),
                        "stats": {**best.stats, "directness": float(best.directness)},
                        "clue_meta": {
                            "clue": best.clue,
                            "num": int(best.num),
                            "parse_valid": bool(best.parse_valid),
                            "valid": bool(best.valid),
                            "rejected_candidates": int(meta["rejected_total"]),
                            "rejection_counts": meta["rejection_counts"],
                        },
                        "debug": {"guess_words": best.guess_words},
                        "best_candidate_idx": best_idx,
                        "candidates": cand_payload,
                        "candidate_reward_mean": float(avg_all),
                        "candidate_reward_std": float(std_all),
                        "candidate_valid_frac": float(valid_frac),
                        "best_minus_avg_reward": float(uplift_all),
                        "candidate_reward_mean_valid": float(avg_valid) if avg_valid is not None else None,
                        "best_minus_avg_reward_valid": float(uplift_valid) if uplift_valid is not None else None,
                    }

                    f_raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    writes_since_flush += 1
                    if writes_since_flush >= FLUSH_EVERY:
                        f_raw.flush()
                        writes_since_flush = 0

                    done_ids.add(example_id)

                    running_sum += float(best.reward)
                    running_n += 1
                    cand_avg_sum += float(avg_all)
                    cand_std_sum += float(std_all)
                    cand_valid_frac_sum += float(valid_frac)
                    uplift_sum += float(uplift_all)
                    uplift_n += 1

                    if avg_valid is not None and uplift_valid is not None:
                        cand_avg_valid_sum += float(avg_valid)
                        uplift_valid_sum += float(uplift_valid)
                        uplift_valid_n += 1

                    if len(done_ids) % progress_every == 0:
                        mean_r = (running_sum / running_n) if running_n else None
                        mean_cand = (cand_avg_sum / uplift_n) if uplift_n else None
                        mean_uplift = (uplift_sum / uplift_n) if uplift_n else None
                        mean_std = (cand_std_sum / uplift_n) if uplift_n else None
                        mean_vfrac = (cand_valid_frac_sum / uplift_n) if uplift_n else None
                        mean_cand_valid = (cand_avg_valid_sum / uplift_valid_n) if uplift_valid_n else None
                        mean_uplift_valid = (uplift_valid_sum / uplift_valid_n) if uplift_valid_n else None

                        save_progress(
                            progress_path,
                            done=len(done_ids),
                            total=shard_total,
                            last_example_id=rec["example_id"],
                            last_board_id=rec["board_id"],
                            mean_reward=mean_r,
                            extra={
                                "n_candidates": int(n_candidates),
                                "max_resamples": int(cfg["decoding"]["max_resamples"]),
                                "batch_size": int(batch_size),
                                "shard_id": int(shard_id),
                                "num_shards": int(num_shards),
                                "preserve_spymaster_raw_text": True,
                                "status": "running",
                                "uplift_done": int(uplift_n),
                                "mean_candidate_reward_running": mean_cand,
                                "mean_best_minus_avg_reward_running": mean_uplift,
                                "candidate_reward_std_running": mean_std,
                                "candidate_valid_frac_running": mean_vfrac,
                                "uplift_valid_done": int(uplift_valid_n),
                                "mean_candidate_reward_valid_running": mean_cand_valid,
                                "mean_best_minus_avg_reward_valid_running": mean_uplift_valid,
                            },
                        )

        f_raw.flush()

    finally:
        try:
            f_raw.flush()
        except Exception:
            pass
        try:
            f_raw.close()
        except Exception:
            pass

    if num_shards > 1:
        print(f"[shard {shard_id}/{num_shards}] done; skipping global filter/merge (merge job will handle).")

        mean_r = (running_sum / running_n) if running_n else None
        mean_cand = (cand_avg_sum / uplift_n) if uplift_n else None
        mean_uplift = (uplift_sum / uplift_n) if uplift_n else None
        mean_std = (cand_std_sum / uplift_n) if uplift_n else None
        mean_vfrac = (cand_valid_frac_sum / uplift_n) if uplift_n else None
        mean_cand_valid = (cand_avg_valid_sum / uplift_valid_n) if uplift_valid_n else None
        mean_uplift_valid = (uplift_valid_sum / uplift_valid_n) if uplift_valid_n else None

        save_progress(
            progress_path,
            done=len(done_ids),
            total=shard_total,
            last_example_id=prev_prog.get("last_example_id"),
            last_board_id=prev_prog.get("last_board_id"),
            mean_reward=mean_r,
            extra={
                "status": "finished_shard",
                "uplift_done": int(uplift_n),
                "mean_candidate_reward_running": mean_cand,
                "mean_best_minus_avg_reward_running": mean_uplift,
                "candidate_reward_std_running": mean_std,
                "candidate_valid_frac_running": mean_vfrac,
                "uplift_valid_done": int(uplift_valid_n),
                "mean_candidate_reward_valid_running": mean_cand_valid,
                "mean_best_minus_avg_reward_valid_running": mean_uplift_valid,
            },
        )
        return

    raw_for_filter = read_jsonl(raw_path) if raw_path.exists() else []
    filtered = filter_examples(raw_for_filter, cfg)
    write_jsonl(turns_path, filtered)
    print(f"Filtered {len(filtered)}/{len(raw_for_filter)} -> {turns_path}")

    mean_r = (running_sum / running_n) if running_n else None
    mean_cand = (cand_avg_sum / uplift_n) if uplift_n else None
    mean_uplift = (uplift_sum / uplift_n) if uplift_n else None
    mean_std = (cand_std_sum / uplift_n) if uplift_n else None
    mean_vfrac = (cand_valid_frac_sum / uplift_n) if uplift_n else None
    mean_cand_valid = (cand_avg_valid_sum / uplift_valid_n) if uplift_valid_n else None
    mean_uplift_valid = (uplift_valid_sum / uplift_valid_n) if uplift_valid_n else None

    save_progress(
        progress_path,
        done=len(done_ids),
        total=shard_total,
        last_example_id=None,
        last_board_id=None,
        mean_reward=mean_r,
        extra={
            "status": "finished",
            "uplift_done": int(uplift_n),
            "mean_candidate_reward_running": mean_cand,
            "mean_best_minus_avg_reward_running": mean_uplift,
            "candidate_reward_std_running": mean_std,
            "candidate_valid_frac_running": mean_vfrac,
            "uplift_valid_done": int(uplift_valid_n),
            "mean_candidate_reward_valid_running": mean_cand_valid,
            "mean_best_minus_avg_reward_valid_running": mean_uplift_valid,
        },
    )


if __name__ == "__main__":
    main()
