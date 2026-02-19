from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .model_wrappers import Embedder, TextGenerator, GenerationConfig
from .rules import is_valid_clue, normalize_word, score_turn
from .spymaster_prompt import build_spymaster_messages
from .guesser_prompt import build_guesser_messages


_CLUE_RE = re.compile(r"CLUE\s*:\s*(.+)", re.IGNORECASE)
_NUM_RE = re.compile(r"NUM\s*:\s*([0-9]+)", re.IGNORECASE)
_GUESSES_RE = re.compile(r"GUESSES\s*:\s*(.+)", re.IGNORECASE)


def parse_spymaster_output(text: str) -> Tuple[Optional[str], Optional[int]]:
    clues = _CLUE_RE.findall(text)
    nums = _NUM_RE.findall(text)
    clue = clues[-1].strip().splitlines()[0].strip() if clues else None
    num = int(nums[-1]) if nums else None

    return clue, num


def parse_guesser_output(text: str) -> List[str]:
    ms = _GUESSES_RE.findall(text)
    m = ms[-1] if ms else None
    if not m:
        # fallback: take first line
        line = text.strip().splitlines()[0] if text.strip() else ""
        raw = line
    else:
        raw = m

    parts = [p.strip() for p in raw.replace("\n", " ").split(",")]
    parts = [p for p in parts if p]
    return parts


def map_guesses_to_board(guesses: List[str], board_words: List[str]) -> List[str]:
    """
    Normalize and keep only guesses that match board words exactly after normalization.
    Ensures uniqueness preserving order.
    """
    board_norm = {normalize_word(w): w for w in board_words}
    out: List[str] = []
    seen = set()
    for g in guesses:
        gn = normalize_word(g)
        if gn in board_norm and gn not in seen:
            out.append(gn)
            seen.add(gn)
    return out


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


def run_one_candidate(
    board_record: Dict[str, Any],
    spymaster: TextGenerator,
    guesser: TextGenerator,
    embedder: Embedder,
    cfg: Dict[str, Any],
    seed: Optional[int] = None,
) -> CandidateResult:
    board_words = board_record["board_words"]
    labels = board_record["labels"]
    revealed_mask = [False] * len(board_words)

    target_words = [w for w, lab in zip(board_words, labels) if lab == "TEAM"]
    labels_by_word = {normalize_word(w): lab for w, lab in zip(board_words, labels)}

    sp_msgs = build_spymaster_messages(board_words, labels, revealed_mask, cfg)
    sp_gen = GenerationConfig(
        temperature=float(cfg["decoding"]["spymaster_temperature"]),
        top_p=float(cfg["decoding"]["spymaster_top_p"]),
        top_k=int(cfg["decoding"].get("spymaster_top_k", 20)),
        max_new_tokens=int(cfg["decoding"]["spymaster_max_new_tokens"]),
    )
    use_chat = bool(cfg.get("qwen", {}).get("use_chat_template", False))
    sp_think = bool(cfg.get("qwen", {}).get("enable_thinking_spymaster", True))
    sp_text = spymaster.generate(sp_msgs, sp_gen, seed=seed, use_chat_template=use_chat, enable_thinking=sp_think)
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
            num=num,
            valid=False,
            rejection_reason=reason,
            directness=d,
            reward=0.0,
            stats={"n_team": 0, "n_opp": 0, "n_neu": 0, "assassin": 0},
            guess_words=[],
            raw_spymaster_text=sp_text,
            raw_guesser_text="",
        )

    # Run guesser
    g_msgs = build_guesser_messages(board_words, revealed_mask, clue, num, cfg)
    g_gen = GenerationConfig(
        temperature=float(cfg["decoding"]["guesser_temperature"]),
        top_p=float(cfg["decoding"]["guesser_top_p"]),
        top_k=int(cfg["decoding"].get("guesser_top_k", 20)),
        max_new_tokens=int(cfg["decoding"]["guesser_max_new_tokens"]),
    )
    g_think = bool(cfg.get("qwen", {}).get("enable_thinking_guesser", True))
    g_text = guesser.generate(g_msgs, g_gen, seed=seed, use_chat_template=use_chat, enable_thinking=g_think)
    guesses_raw = parse_guesser_output(g_text)
    guesses = map_guesses_to_board(guesses_raw, board_words)

    # Score using top-K where K = num (clamped)
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
    embedder: Embedder,
    cfg: Dict[str, Any],
    n_candidates: int = 1,
    seed: Optional[int] = None,
) -> Tuple[List[CandidateResult], Dict[str, Any]]:
    """
    Samples up to n_candidates *with validity-enforced resampling*.
    Returns candidates and metadata including rejected counts.
    """
    max_resamples = int(cfg["decoding"]["max_resamples"])
    candidates: List[CandidateResult] = []
    rejection_counts: Dict[str, int] = {}
    total_rejected = 0

    # We produce n_candidates valid candidates (or exhaust resamples)
    for ci in range(n_candidates):
        got = None
        for ri in range(max_resamples):
            # deterministic-ish seed derivation
            s = None if seed is None else (seed + ci * 1000 + ri)
            res = run_one_candidate(board_record, spymaster, guesser, embedder, cfg, seed=s)

            if res.valid:
                got = res
                break
            total_rejected += 1
            rejection_counts[res.rejection_reason] = rejection_counts.get(res.rejection_reason, 0) + 1

        if got is None:
            # record the last invalid result if we never got a valid one
            got = res
        candidates.append(got)

    meta = {
        "rejected_total": total_rejected,
        "rejection_counts": rejection_counts,
    }
    return candidates, meta


def select_best_candidate(candidates: List[CandidateResult]) -> CandidateResult:
    # Primary: reward, Secondary: lower directness (more indirect) if tie
    best = candidates[0]
    for c in candidates[1:]:
        if c.reward > best.reward:
            best = c
        elif c.reward == best.reward and c.directness < best.directness:
            best = c
    return best
