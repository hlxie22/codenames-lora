from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .io_utils import shard_path_append_to_suffix, merge_jsonl_shards
from .model_wrappers import Embedder, make_text_generator
from .mp_utils import launch_children, is_child_process, child_shard_info
from .prompting import render_prompt
from .rollout import run_turns_batched
from .spymaster_prompt import build_spymaster_messages
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


def extract_think_block(text: str) -> str:
    # Keep the model-produced think block if present; otherwise return "".
    lo = text.lower().rfind("<think>")
    hi = text.lower().rfind("</think>")
    if lo != -1 and hi != -1 and hi > lo:
        return text[lo : hi + len("</think>")].strip()
    # Sometimes you might only see </think>; keep everything up to it as "thinking".
    if hi != -1:
        return text[: hi + len("</think>")].strip()
    return ""


# -------------------------
# Filtering
# -------------------------

def filter_examples(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    fcfg = cfg["filtering"]
    mode = fcfg["mode"]

    if not records:
        return []

    if mode == "top_percent":
        top_p = float(fcfg["top_percent"])
        rewards = np.array([r["reward"] for r in records], dtype=np.float32)
        thr = float(np.quantile(rewards, 1.0 - top_p))
        return [r for r in records if float(r["reward"]) >= thr]

    if mode == "rule_based":
        min_team = int(fcfg.get("min_team_correct", 0))
        require_no_assassin = bool(fcfg.get("require_no_assassin", False))
        kept = []
        for r in records:
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

    # Paths (use per-shard raw file to avoid concurrent appends)
    base_raw_path = cfg.get("paths", {}).get("sft_turns_raw_path")
    if not base_raw_path:
        raise RuntimeError("paths.sft_turns_raw_path must be set")

    raw_path = Path(base_raw_path)
    if num_shards > 1:
        raw_path = shard_path_append_to_suffix(raw_path, shard_id, num_shards)

    done_ids = load_done_example_ids(raw_path)
    if done_ids:
        print(f"[shard {shard_id}/{num_shards}] Resuming: found {len(done_ids)} already-written examples in {raw_path}")

    progress_path = raw_path.with_suffix(raw_path.suffix + ".progress.json")

    # -------------------------
    # Buffered writer (NO fsync-per-line)
    # -------------------------
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    f_raw = open(raw_path, "a", encoding="utf-8")
    writes_since_flush = 0
    FLUSH_EVERY = 200  # tune to taste

    # Track running reward without storing all records
    running_sum = 0.0
    running_n = 0

    try:
        for start in range(0, len(boards), max(1, batch_size)):
            batch = boards[start : start + max(1, batch_size)]

            bests, metas = run_turns_batched(
                batch,
                spymaster,
                guesser,
                embedder,  # may be None if you applied the optional-embedder change in rules/rollout
                cfg,
                n_candidates=n_candidates,
            )

            for b, best, meta in zip(batch, bests, metas):
                example_id = b["board_id"]
                if example_id in done_ids:
                    continue

                revealed = [False] * len(b["board_words"])
                msgs = build_spymaster_messages(b["board_words"], b["labels"], revealed, cfg)

                # Centralized prompt rendering (no duplicated chat-template/thinking logic)
                prompt = render_prompt(spymaster, msgs, cfg, role="spymaster")

                think_block = extract_think_block(best.raw_spymaster_text)
                completion = ""
                if think_block:
                    completion += think_block + "\n"
                completion += f"CLUE: {best.clue}\nNUM: {best.num}\n"

                rec = {
                    "example_id": example_id,
                    "board_id": b["board_id"],
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
                }

                f_raw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                writes_since_flush += 1
                if writes_since_flush >= FLUSH_EVERY:
                    f_raw.flush()
                    writes_since_flush = 0

                done_ids.add(example_id)

                running_sum += float(best.reward)
                running_n += 1

                if len(done_ids) % progress_every == 0:
                    mean_r = (running_sum / running_n) if running_n else None
                    save_progress(
                        progress_path,
                        done=len(done_ids),
                        total=len(boards),
                        last_example_id=rec["example_id"],
                        last_board_id=rec["board_id"],
                        mean_reward=mean_r,
                        extra={
                            "n_candidates": int(n_candidates),
                            "max_resamples": int(cfg["decoding"]["max_resamples"]),
                            "batch_size": int(batch_size),
                            "shard_id": int(shard_id),
                            "num_shards": int(num_shards),
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
        return

    # Single-process mode: filter and write final file
    raw_for_filter = read_jsonl(raw_path) if raw_path.exists() else []
    filtered = filter_examples(raw_for_filter, cfg)
    write_jsonl(cfg["paths"]["sft_turns_path"], filtered)
    print(f"Filtered {len(filtered)}/{len(raw_for_filter)} -> {cfg['paths']['sft_turns_path']}")

    mean_r = (running_sum / running_n) if running_n else None
    save_progress(
        progress_path,
        done=len(done_ids),
        total=len(boards),
        last_example_id=None,
        last_board_id=None,
        mean_reward=mean_r,
        extra={"status": "finished"},
    )


if __name__ == "__main__":
    main()