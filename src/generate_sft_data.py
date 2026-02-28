# src/generate_sft_data.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .io_utils import shard_path_append_to_suffix, merge_jsonl_shards
from .model_wrappers import Embedder, make_text_generator
from .mp_utils import launch_children, is_child_process, child_shard_info
from .prompting import render_prompt
from .rollout import run_turns_batched
from .spymaster_prompt import build_spymaster_messages
from .think_utils import extract_think as _extract_think
from .utils import load_yaml, read_jsonl, set_global_seed, write_jsonl, save_progress


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
                # If a partial/corrupt last line exists, ignore it
                continue
    return done


def _try_load_progress(progress_path: Path) -> Optional[Dict[str, Any]]:
    try:
        if progress_path.exists():
            return json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def extract_think_block(text: str) -> str:
    """
    Back-compat wrapper (this file previously had a custom "last block + partial tolerant" parser).
    Returns the FULL <think>...</think> block (or best-effort partial) from the LAST block.
    """
    return _extract_think(text, mode="full", which="last", allow_partial=True)


# -------------------------
# Filtering
# -------------------------

def filter_examples(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    fcfg = cfg["filtering"]
    mode = fcfg["mode"]

    if not records:
        return []

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


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    num_procs = int(cfg.get("inference", {}).get("num_processes", 1))
    batch_size = int(cfg.get("inference", {}).get("batch_size", 1))

    # Master: spawn workers and merge/filter when done
    if num_procs > 1 and not is_child_process():
        if cfg.get("inference", {}).get("backend") == "vllm":
            tp = int(cfg.get("inference", {}).get("vllm", {}).get("tensor_parallel_size", 1))
            if tp != 1:
                print(
                    f"[warn] inference.vllm.tensor_parallel_size={tp} with inference.num_processes={num_procs}. "
                    f"Usually you want tensor_parallel_size=1 when using multiple processes."
                )

        launch_children("src.generate_sft_data", ["--config", args.config], num_procs)

        base_raw_path = cfg.get("paths", {}).get("sft_turns_raw_path")
        if not base_raw_path:
            raise RuntimeError("paths.sft_turns_raw_path must be set when using inference.num_processes > 1")

        combined = _merge_shards_to_base(base_raw_path, num_procs)
        filtered = filter_examples(combined, cfg)
        write_jsonl(cfg["paths"]["sft_turns_path"], filtered)
        print(f"Merged {len(combined)} raw -> Filtered {len(filtered)} -> {cfg['paths']['sft_turns_path']}")
        return

    # Worker or single-process mode
    set_global_seed(int(cfg["training"].get("seed", 0)))

    boards = read_jsonl(cfg["paths"]["boards_train_path"])

    # Apply cap BEFORE sharding so total dataset size is bounded
    maxb = cfg.get("boards", {}).get("sft_max_train_boards", None)
    if maxb is not None:
        boards = boards[: int(maxb)]
        print(f"Using only first {len(boards)} train boards for SFT generation (boards.sft_max_train_boards={maxb}).")

    shard_id, num_shards = child_shard_info() if is_child_process() else (0, 1)

    # Shard boards by index
    if num_shards > 1:
        boards = [b for i, b in enumerate(boards) if (i % num_shards) == shard_id]
        print(f"[shard {shard_id}/{num_shards}] boards={len(boards)}")

    # IMPORTANT: per-shard total is defined BEFORE resume filtering
    shard_total = len(boards)

    # Paths (use per-shard raw file to avoid concurrent appends)
    base_raw_path = cfg.get("paths", {}).get("sft_turns_raw_path")
    if not base_raw_path:
        raise RuntimeError("paths.sft_turns_raw_path must be set")

    raw_path = Path(base_raw_path)
    if num_shards > 1:
        raw_path = shard_path_append_to_suffix(raw_path, shard_id, num_shards)

    progress_path = raw_path.with_suffix(raw_path.suffix + ".progress.json")

    # Resume: detect already-written examples
    done_ids = load_done_example_ids(raw_path)
    if done_ids:
        # CRITICAL FIX:
        # Skip compute for already-done boards BEFORE calling run_turns_batched
        before = len(boards)
        boards = [b for b in boards if str(b.get("board_id")) not in done_ids]
        after = len(boards)
        print(
            f"[shard {shard_id}/{num_shards}] Resuming: found {len(done_ids)} already-written examples in {raw_path}. "
            f"Remaining={after} (was {before}), shard_total={shard_total}."
        )
    else:
        print(f"[shard {shard_id}/{num_shards}] Starting fresh: shard_total={shard_total} -> {raw_path}")

    # Refresh / initialize progress at startup so timestamp updates even before new writes
    prev_prog = _try_load_progress(progress_path) or {}
    prev_mean = prev_prog.get("mean_reward_running", None)
    prev_done = prev_prog.get("done", None)

    # Running mean reward of BEST (one per board)
    running_sum = 0.0
    running_n = 0

    # Running stats for best-of-N uplift vs average candidate
    uplift_sum = 0.0
    uplift_n = 0
    cand_avg_sum = 0.0
    cand_std_sum = 0.0
    cand_valid_frac_sum = 0.0

    # Valid-only variants
    uplift_valid_sum = 0.0
    uplift_valid_n = 0
    cand_avg_valid_sum = 0.0

    # If prior progress is consistent with what we found, carry forward mean approx
    try:
        if prev_mean is not None and prev_done is not None and int(prev_done) == len(done_ids):
            running_n = int(prev_done)
            running_sum = float(prev_mean) * float(prev_done)

            # Carry forward uplift stats if they exist in the progress file.
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
            "include_raw_texts": False,

            # best-of-N vs avg candidate stats (running)
            "uplift_done": int(uplift_n),
            "mean_candidate_reward_running": (cand_avg_sum / uplift_n) if uplift_n else None,
            "mean_best_minus_avg_reward_running": (uplift_sum / uplift_n) if uplift_n else None,
            "candidate_reward_std_running": (cand_std_sum / uplift_n) if uplift_n else None,
            "candidate_valid_frac_running": (cand_valid_frac_sum / uplift_n) if uplift_n else None,

            # valid-only variant
            "uplift_valid_done": int(uplift_valid_n),
            "mean_candidate_reward_valid_running": (cand_avg_valid_sum / uplift_valid_n) if uplift_valid_n else None,
            "mean_best_minus_avg_reward_valid_running": (uplift_valid_sum / uplift_valid_n) if uplift_valid_n else None,
        },
    )

    # Models
    gen = make_text_generator(cfg["models"]["spymaster_model_id"], cfg)
    spymaster = gen
    guesser = gen

    # Embedder (OPTIONAL via constraints.enable_directness_check)
    use_embed = bool(cfg.get("constraints", {}).get("enable_directness_check", True))
    embedder = None
    if use_embed:
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        embedder = Embedder(cfg["models"]["embedding_model_id"], device=device)

    n_candidates = int(cfg["decoding"]["n_candidates"])
    progress_every = int(cfg.get("decoding", {}).get("progress_every", 50))

    # -------------------------
    # Buffered writer
    # -------------------------
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    f_raw = open(raw_path, "a", encoding="utf-8")
    writes_since_flush = 0
    FLUSH_EVERY = int(cfg.get("decoding", {}).get("flush_every", 10))

    # NOTE: To keep files small, we do NOT store raw texts for each candidate by default.
    INCLUDE_RAW_TEXTS = False

    try:
        # If everything is already done, exit quickly but still mark progress
        if not boards and len(done_ids) >= shard_total:
            print(f"[shard {shard_id}/{num_shards}] Nothing to do: done={len(done_ids)}/{shard_total}")
        else:
            for start in range(0, len(boards), max(1, batch_size)):
                batch = boards[start: start + max(1, batch_size)]

                # With the resume fix, batch should contain only not-yet-done examples.
                bests, metas, all_cands = run_turns_batched(
                    batch,
                    spymaster,
                    guesser,
                    embedder,  # may be None if directness check disabled
                    cfg,
                    n_candidates=n_candidates,
                    return_candidates=True,  # <-- variance logging
                )

                for b, best, meta, cands in zip(batch, bests, metas, all_cands):
                    example_id = str(b["board_id"])
                    # Safety check (should rarely hit after filtering)
                    if example_id in done_ids:
                        continue

                    revealed = [False] * len(b["board_words"])
                    msgs = build_spymaster_messages(b["board_words"], b["labels"], revealed, cfg)

                    # Centralized prompt rendering (no duplicated chat-template/thinking logic)
                    prompt = render_prompt(spymaster, msgs, cfg, role="spymaster")

                    # Consolidated think parsing (shared util)
                    think_block = extract_think_block(best.raw_spymaster_text)

                    completion = ""
                    if think_block:
                        completion += think_block + "\n"
                    completion += f"CLUE: {best.clue}\nNUM: {best.num}\n"

                    # Best candidate index (object identity should match; fallback to value match)
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

                    # --- best-of-N uplift stats for this example ---
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
                        cd: Dict[str, Any] = {
                            "clue": c.clue,
                            "num": int(c.num),
                            "valid": bool(c.valid),
                            "rejection_reason": c.rejection_reason,
                            "directness": float(c.directness),
                            "reward": float(c.reward),
                            "stats": c.stats,
                            "guess_words": c.guess_words,
                        }
                        if INCLUDE_RAW_TEXTS:
                            cd["raw_spymaster_text"] = c.raw_spymaster_text
                            cd["raw_guesser_text"] = c.raw_guesser_text
                        cand_payload.append(cd)

                    rec = {
                        "example_id": example_id,
                        "board_id": str(b["board_id"]),
                        "prompt": prompt,
                        "completion": completion,
                        "reward": float(best.reward),
                        "stats": {**best.stats, "directness": float(best.directness)},
                        "clue_meta": {
                            "clue": best.clue,
                            "num": int(best.num),
                            "valid": bool(best.valid),
                            "rejected_candidates": int(meta["rejected_total"]),
                            "rejection_counts": meta["rejection_counts"],
                        },
                        "debug": {"guess_words": best.guess_words},
                        # variance logging
                        "best_candidate_idx": best_idx,
                        "candidates": cand_payload,

                        # per-example best-of-N uplift fields
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

                    # Update running best reward mean
                    running_sum += float(best.reward)
                    running_n += 1

                    # Update running uplift stats
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
                            total=shard_total,  # keep per-shard total stable across resume
                            last_example_id=rec["example_id"],
                            last_board_id=rec["board_id"],
                            mean_reward=mean_r,
                            extra={
                                "n_candidates": int(n_candidates),
                                "max_resamples": int(cfg["decoding"]["max_resamples"]),
                                "batch_size": int(batch_size),
                                "shard_id": int(shard_id),
                                "num_shards": int(num_shards),
                                "include_raw_texts": bool(INCLUDE_RAW_TEXTS),
                                "status": "running",

                                # best-of-N vs avg stats (running)
                                "uplift_done": int(uplift_n),
                                "mean_candidate_reward_running": mean_cand,
                                "mean_best_minus_avg_reward_running": mean_uplift,
                                "candidate_reward_std_running": mean_std,
                                "candidate_valid_frac_running": mean_vfrac,

                                # valid-only variant
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

    # Child workers stop here; master will merge + filter.
    if is_child_process() and num_shards > 1:
        print(f"[shard {shard_id}/{num_shards}] done; skipping global filter/merge (master will handle).")

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

    # Single-process mode: filter and write final file
    raw_for_filter = read_jsonl(raw_path) if raw_path.exists() else []
    filtered = filter_examples(raw_for_filter, cfg)
    write_jsonl(cfg["paths"]["sft_turns_path"], filtered)
    print(f"Filtered {len(filtered)}/{len(raw_for_filter)} -> {cfg['paths']['sft_turns_path']}")

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