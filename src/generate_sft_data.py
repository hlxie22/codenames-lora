from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .model_wrappers import make_text_generator, Embedder
from .rollout import run_turn, select_best_candidate
from .spymaster_prompt import build_spymaster_messages
from .utils import load_yaml, read_jsonl, write_jsonl, ensure_dir, set_global_seed

import re
import json
import os
import time

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def save_progress(
    progress_path: str | Path,
    *,
    done: int,
    total: int,
    last_example_id: Optional[str] = None,
    last_board_id: Optional[str] = None,
    mean_reward: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    progress_path = Path(progress_path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "timestamp": now_iso(),
        "done": int(done),
        "total": int(total),
        "pct": float(done / total) if total else 0.0,
        "last_example_id": last_example_id,
        "last_board_id": last_board_id,
        "mean_reward_running": float(mean_reward) if mean_reward is not None else None,
    }
    if extra:
        payload.update(extra)

    tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, progress_path)  # atomic write

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)

def load_done_example_ids(raw_path: str) -> set[str]:
    """Return example_ids already present in an existing raw jsonl (for resume)."""
    done = set()
    if not raw_path or not os.path.exists(raw_path):
        return done
    with open(raw_path, "r", encoding="utf-8") as f:
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
                # if a partial/corrupt last line exists, ignore it
                continue
    return done

def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

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

def extract_think(text: str) -> str:
    m = _THINK_RE.search(text)
    return m.group(1).strip() if m else ""

def filter_examples(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    fcfg = cfg["filtering"]
    mode = fcfg["mode"]

    if not records:
        return []

    if mode == "top_percent":   
        top_p = float(fcfg["top_percent"])
        rewards = np.array([r["reward"] for r in records], dtype=np.float32)
        thr = float(np.quantile(rewards, 1.0 - top_p))
        kept = [r for r in records if float(r["reward"]) >= thr]
        return kept

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["training"].get("seed", 0)))

    boards = read_jsonl(cfg["paths"]["boards_train_path"])

    maxb = cfg.get("boards", {}).get("sft_max_train_boards", None)
    if maxb is not None:
        boards = boards[: int(maxb)]
        print(f"Using only first {len(boards)} train boards for SFT generation (boards.sft_max_train_boards={maxb}).")

    # Models
    spymaster = make_text_generator(cfg["models"]["spymaster_model_id"], cfg)
    guesser = make_text_generator(cfg["models"]["guesser_model_id"], cfg)

    # Embedder (use cuda if available)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    embedder = Embedder(cfg["models"]["embedding_model_id"], device=device)

    n_candidates = int(cfg["decoding"]["n_candidates"])

    raw_path = cfg["paths"].get("sft_turns_raw_path")  # we will append here during generation
    done_ids = load_done_example_ids(raw_path) if raw_path else set()
    if raw_path and done_ids:
        print(f"Resuming: found {len(done_ids)} already-written examples in {raw_path}")

    raw_path = cfg["paths"].get("sft_turns_raw_path")  
    progress_path = Path(raw_path).with_suffix(".progress.json") if raw_path else None
    progress_every = int(cfg.get("decoding", {}).get("progress_every", 50))

    raw_records: List[Dict[str, Any]] = [] 

    for i, b in enumerate(boards):
        example_id = b["board_id"]
        if raw_path and example_id in done_ids:
            continue

        # seed per board for stability
        seed = int(b.get("seed", 0))

        candidates, meta = run_turn(
            b,
            spymaster,
            guesser,
            embedder,
            cfg,
            n_candidates=n_candidates,
            seed=seed,
        )
        best = select_best_candidate(candidates)

        revealed = [False] * len(b["board_words"])
        msgs = build_spymaster_messages(b["board_words"], b["labels"], revealed, cfg)

        use_chat = bool(cfg.get("qwen", {}).get("use_chat_template", False))
        sp_think = bool(cfg.get("qwen", {}).get("enable_thinking_spymaster", True))
        prompt = spymaster.format_chat(msgs, add_generation_prompt=True, enable_thinking=sp_think) if use_chat else msgs[-1]["content"]

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

        # Write immediately (checkpoint)
        if raw_path:
            append_jsonl(raw_path, rec)
            done_ids.add(example_id)

        raw_records.append(rec)   

        if progress_path and (len(done_ids) % progress_every == 0):
            mean_r = float(np.mean([r["reward"] for r in raw_records])) if raw_records else None
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
                },
            )

        if (len(done_ids) if raw_path else (i + 1)) % 50 == 0:
            mean_r = float(np.mean([r["reward"] for r in raw_records])) if raw_records else 0.0
            print(f"[done={len(done_ids) if raw_path else (i+1)}/{len(boards)}] mean_reward={mean_r:.3f}")

    raw_path = cfg["paths"].get("sft_turns_raw_path")

    if raw_path and os.path.exists(raw_path):
        raw_for_filter = read_jsonl(raw_path)
    else:
        raw_for_filter = raw_records

    filtered = filter_examples(raw_for_filter, cfg)
    write_jsonl(cfg["paths"]["sft_turns_path"], filtered)
    print(f"Filtered {len(filtered)}/{len(raw_for_filter)} -> {cfg['paths']['sft_turns_path']}")

    if progress_path:
        mean_r = float(np.mean([r["reward"] for r in raw_records])) if raw_records else None
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