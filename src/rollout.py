from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .gen_cfg import guesser_gen_cfg, spymaster_gen_cfg
from .model_wrappers import Embedder, TextGenerator
from .prompting import render_prompt
from .rules import is_valid_clue, normalize_word, score_turn
from .spymaster_prompt import build_spymaster_messages
from .guesser_prompt import build_guesser_messages

# -------------------------
# Parsing helpers
# -------------------------

_CLUE_RE = re.compile(r"CLUE\s*:\s*(.+)", re.IGNORECASE)
_NUM_RE = re.compile(r"NUM\s*:\s*([0-9]+)", re.IGNORECASE)
_GUESSES_RE = re.compile(r"GUESSES\s*:\s*(.+)", re.IGNORECASE)


def parse_spymaster_output(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract the last CLUE/NUM pair from a completion.
    """
    clues = _CLUE_RE.findall(text)
    nums = _NUM_RE.findall(text)
    clue = clues[-1].strip().splitlines()[0].strip() if clues else None
    num = int(nums[-1]) if nums else None
    return clue, num


def parse_guesser_output(text: str) -> List[str]:
    """
    Extract guesses from "GUESSES: a, b, c" (last match wins),
    otherwise fall back to first non-empty line.
    """
    ms = _GUESSES_RE.findall(text)
    raw = ms[-1] if ms else ""
    if not raw:
        raw = text.strip().splitlines()[0] if text.strip() else ""

    parts = [p.strip() for p in raw.replace("\n", " ").split(",")]
    return [p for p in parts if p]


def map_guesses_to_board(guesses: List[str], board_words: List[str]) -> List[str]:
    """
    Normalize and keep only guesses that match board words exactly after normalization.
    Ensures uniqueness while preserving order.

    Returns normalized board-word strings (consistent with score_turn / labels_by_word keys).
    """
    board_norm = {normalize_word(w) for w in board_words}
    out: List[str] = []
    seen = set()
    for g in guesses:
        gn = normalize_word(g)
        if gn in board_norm and gn not in seen:
            out.append(gn)
            seen.add(gn)
    return out


# -------------------------
# Core data structure
# -------------------------

@dataclass
class CandidateResult:
    clue: str
    num: int
    valid: bool
    rejection_reason: str
    directness: float
    reward: float
    stats: Dict[str, Any]
    guess_words: List[str]
    raw_spymaster_text: str
    raw_guesser_text: str


# -------------------------
# Single-candidate rollout
# -------------------------

def run_one_candidate(
    board_record: Dict[str, Any],
    spymaster: TextGenerator,
    guesser: TextGenerator,
    embedder: Optional[Embedder],
    cfg: Dict[str, Any],
    seed: Optional[int] = None,
) -> CandidateResult:
    board_words = board_record["board_words"]
    labels = board_record["labels"]
    revealed_mask = [False] * len(board_words)

    target_words = [w for w, lab in zip(board_words, labels) if lab == "TEAM"]
    labels_by_word = {normalize_word(w): lab for w, lab in zip(board_words, labels)}

    # --- spymaster
    sp_msgs = build_spymaster_messages(board_words, labels, revealed_mask, cfg)
    sp_gen = spymaster_gen_cfg(cfg)

    use_chat = bool(cfg.get("qwen", {}).get("use_chat_template", False))
    sp_think = bool(cfg.get("qwen", {}).get("enable_thinking_spymaster", True))

    sp_text = spymaster.generate(
        sp_msgs if use_chat else sp_msgs[-1]["content"],
        sp_gen,
        seed=seed,
        use_chat_template=use_chat,
        enable_thinking=sp_think,
    )
    clue, num = parse_spymaster_output(sp_text)

    if clue is None or num is None:
        return CandidateResult(
            clue=clue or "",
            num=num or 0,
            valid=False,
            rejection_reason="parse_fail",
            directness=0.0,
            reward=0.0,
            stats={"n_team": 0, "n_opp": 0, "n_neu": 0, "assassin": 0},
            guess_words=[],
            raw_spymaster_text=sp_text,
            raw_guesser_text="",
        )

    valid, reason, d = is_valid_clue(clue, board_words, target_words, embedder, cfg)
    if not valid:
        return CandidateResult(
            clue=clue,
            num=int(num),
            valid=False,
            rejection_reason=reason,
            directness=float(d),
            reward=0.0,
            stats={"n_team": 0, "n_opp": 0, "n_neu": 0, "assassin": 0},
            guess_words=[],
            raw_spymaster_text=sp_text,
            raw_guesser_text="",
        )

    # --- guesser
    g_msgs = build_guesser_messages(board_words, revealed_mask, clue, int(num), cfg)
    g_gen = guesser_gen_cfg(cfg)
    g_think = bool(cfg.get("qwen", {}).get("enable_thinking_guesser", True))

    g_text = guesser.generate(
        g_msgs if use_chat else g_msgs[-1]["content"],
        g_gen,
        seed=seed,
        use_chat_template=use_chat,
        enable_thinking=g_think,
    )

    guesses_raw = parse_guesser_output(g_text)
    guesses = map_guesses_to_board(guesses_raw, board_words)

    k = max(0, min(int(num), len(guesses)))
    scored = guesses[:k]
    reward, stats = score_turn(scored, labels_by_word, cfg)

    return CandidateResult(
        clue=clue,
        num=int(num),
        valid=True,
        rejection_reason="ok",
        directness=float(d),
        reward=float(reward),
        stats=stats,
        guess_words=scored,
        raw_spymaster_text=sp_text,
        raw_guesser_text=g_text,
    )


def run_turn(
    board_record: Dict[str, Any],
    spymaster: TextGenerator,
    guesser: TextGenerator,
    embedder: Optional[Embedder],
    cfg: Dict[str, Any],
    n_candidates: int = 1,
    seed: Optional[int] = None,
) -> Tuple[List[CandidateResult], Dict[str, Any]]:
    """
    Samples up to n_candidates, with validity-enforced resampling.
    Returns candidates and metadata including rejected counts.
    """
    max_resamples = int(cfg["decoding"]["max_resamples"])
    candidates: List[CandidateResult] = []
    rejection_counts: Dict[str, int] = {}
    total_rejected = 0

    for ci in range(n_candidates):
        got: Optional[CandidateResult] = None
        last_res: Optional[CandidateResult] = None

        for ri in range(max_resamples):
            s = None if seed is None else (seed + ci * 1000 + ri)
            res = run_one_candidate(board_record, spymaster, guesser, embedder, cfg, seed=s)
            last_res = res

            if res.valid:
                got = res
                break

            total_rejected += 1
            rejection_counts[res.rejection_reason] = rejection_counts.get(res.rejection_reason, 0) + 1

        if got is None:
            # record the last invalid result if we never got a valid one
            got = last_res if last_res is not None else CandidateResult(
                clue="",
                num=0,
                valid=False,
                rejection_reason="exhausted",
                directness=0.0,
                reward=0.0,
                stats={"n_team": 0, "n_opp": 0, "n_neu": 0, "assassin": 0},
                guess_words=[],
                raw_spymaster_text="",
                raw_guesser_text="",
            )
        candidates.append(got)

    meta = {"rejected_total": total_rejected, "rejection_counts": rejection_counts}
    return candidates, meta


def select_best_candidate(candidates: List[CandidateResult]) -> CandidateResult:
    """
    Primary: reward
    Secondary: lower directness (more indirect) if tie
    """
    best = candidates[0]
    for c in candidates[1:]:
        if c.reward > best.reward:
            best = c
        elif c.reward == best.reward and c.directness < best.directness:
            best = c
    return best


# -------------------------
# Batched rollout (shared by eval + SFT generation)
# -------------------------

def run_turns_batched(
    boards_batch: List[Dict[str, Any]],
    spymaster: TextGenerator,
    guesser: TextGenerator,
    embedder: Optional[Embedder],
    cfg: Dict[str, Any],
    *,
    n_candidates: int,
) -> Tuple[List[CandidateResult], List[Dict[str, Any]]]:
    """
    Returns:
      - best CandidateResult per board
      - meta per board: rejected_total + rejection_counts (matching run_turn)
    Falls back to sequential if generator doesn't support generate_batch.
    """
    # Sequential fallback if no batch API
    if not (hasattr(spymaster, "generate_batch") and hasattr(guesser, "generate_batch")):
        bests: List[CandidateResult] = []
        metas: List[Dict[str, Any]] = []
        for b in boards_batch:
            seed = int(b.get("seed", 0))
            cands, meta = run_turn(b, spymaster, guesser, embedder, cfg, n_candidates=n_candidates, seed=seed)
            bests.append(select_best_candidate(cands))
            metas.append(meta)
        return bests, metas

    sp_gen = spymaster_gen_cfg(cfg)
    g_gen = guesser_gen_cfg(cfg)
    max_resamples = int(cfg["decoding"]["max_resamples"])

    # Precompute prompts + bookkeeping per board
    sp_prompts: List[str] = []
    board_words_list: List[List[str]] = []
    target_words_list: List[List[str]] = []
    labels_by_word_list: List[Dict[str, str]] = []
    seeds0: List[int] = []

    for b in boards_batch:
        board_words = b["board_words"]
        labels = b["labels"]
        revealed_mask = [False] * len(board_words)

        sp_msgs = build_spymaster_messages(board_words, labels, revealed_mask, cfg)
        sp_prompts.append(render_prompt(spymaster, sp_msgs, cfg, role="spymaster"))

        target_words = [w for w, lab in zip(board_words, labels) if lab == "TEAM"]
        labels_by_word = {normalize_word(w): lab for w, lab in zip(board_words, labels)}

        board_words_list.append(board_words)
        target_words_list.append(target_words)
        labels_by_word_list.append(labels_by_word)
        seeds0.append(int(b.get("seed", 0)))

    # Storage
    all_candidates_per_board: List[List[CandidateResult]] = [[] for _ in boards_batch]
    rej_counts_per_board: List[Dict[str, int]] = [{} for _ in boards_batch]
    rej_total_per_board: List[int] = [0 for _ in boards_batch]

    for ci in range(n_candidates):
        # tuple: (clue, num, is_valid, reason, directness, raw_spymaster_text, used_seed)
        got: List[Optional[Tuple[str, int, bool, str, float, str, int]]] = [None] * len(boards_batch)

        pending = list(range(len(boards_batch)))
        last_text: Dict[int, str] = {}
        last_seed: Dict[int, int] = {}

        # Batched spymaster resampling
        for ri in range(max_resamples):
            if not pending:
                break

            prompts = [sp_prompts[i] for i in pending]
            seeds = [seeds0[i] + ci * 1000 + ri for i in pending]
            texts = spymaster.generate_batch(prompts, sp_gen, seeds)  # type: ignore[attr-defined]

            for idx_in_list, bi in enumerate(pending):
                txt = texts[idx_in_list]
                s_used = seeds[idx_in_list]
                last_text[bi] = txt
                last_seed[bi] = s_used

                clue, num = parse_spymaster_output(txt)
                if clue is None or num is None:
                    reason = "parse_fail"
                    rej_total_per_board[bi] += 1
                    rej_counts_per_board[bi][reason] = rej_counts_per_board[bi].get(reason, 0) + 1
                    continue

                valid, reason, d = is_valid_clue(clue, board_words_list[bi], target_words_list[bi], embedder, cfg)
                if not valid:
                    rej_total_per_board[bi] += 1
                    rej_counts_per_board[bi][reason] = rej_counts_per_board[bi].get(reason, 0) + 1
                    continue

                got[bi] = (clue, int(num), True, "ok", float(d), txt, s_used)

            pending = [bi for bi in pending if got[bi] is None]

        # Fill any missing with last attempt (invalid)
        for bi in range(len(boards_batch)):
            if got[bi] is None:
                txt = last_text.get(bi, "")
                s_used = last_seed.get(bi, seeds0[bi] + ci * 1000)
                clue, num = parse_spymaster_output(txt)
                got[bi] = (clue or "", int(num) if num is not None else 0, False, "exhausted", 0.0, txt, s_used)

        # Batched guesser for valid ones only
        valid_bis = [bi for bi in range(len(boards_batch)) if got[bi][2]]  # type: ignore[index]
        g_prompts: List[str] = []
        g_seeds: List[int] = []

        for bi in valid_bis:
            clue, num, _, _, _, _, s_used = got[bi]  # type: ignore[misc]
            revealed_mask = [False] * len(board_words_list[bi])
            g_msgs = build_guesser_messages(board_words_list[bi], revealed_mask, clue, int(num), cfg)
            g_prompts.append(render_prompt(guesser, g_msgs, cfg, role="guesser"))
            g_seeds.append(int(s_used))

        g_texts = guesser.generate_batch(g_prompts, g_gen, g_seeds) if g_prompts else []  # type: ignore[attr-defined]

        gi = 0
        for bi in range(len(boards_batch)):
            clue, num, is_valid, reason, d, sp_txt, _ = got[bi]  # type: ignore[misc]

            if not is_valid:
                all_candidates_per_board[bi].append(
                    CandidateResult(
                        clue=clue,
                        num=int(num),
                        valid=False,
                        rejection_reason=reason,
                        directness=float(d),
                        reward=0.0,
                        stats={"n_team": 0, "n_opp": 0, "n_neu": 0, "assassin": 0},
                        guess_words=[],
                        raw_spymaster_text=sp_txt,
                        raw_guesser_text="",
                    )
                )
                continue

            g_txt = g_texts[gi]
            gi += 1

            guesses_raw = parse_guesser_output(g_txt)
            guesses = map_guesses_to_board(guesses_raw, board_words_list[bi])

            k = max(0, min(int(num), len(guesses)))
            scored = guesses[:k]
            reward, stats = score_turn(scored, labels_by_word_list[bi], cfg)

            all_candidates_per_board[bi].append(
                CandidateResult(
                    clue=clue,
                    num=int(num),
                    valid=True,
                    rejection_reason="ok",
                    directness=float(d),
                    reward=float(reward),
                    stats=stats,
                    guess_words=scored,
                    raw_spymaster_text=sp_txt,
                    raw_guesser_text=g_txt,
                )
            )

    bests = [select_best_candidate(cands) for cands in all_candidates_per_board]
    metas = [
        {"rejected_total": rej_total_per_board[i], "rejection_counts": rej_counts_per_board[i]}
        for i in range(len(boards_batch))
    ]
    return bests, metas