from __future__ import annotations

import re
from typing import Dict, List, Tuple, Any, Optional

from .model_wrappers import Embedder


_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def normalize_word(s: str) -> str:
    return s.strip().lower()


def is_single_word(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # “single word” in the practical sense: no whitespace
    return len(s.split()) == 1


def violates_board_overlap(clue: str, board_words: List[str]) -> bool:
    c = normalize_word(clue)
    bw = {normalize_word(w) for w in board_words}
    return c in bw


def violates_substring_ban(clue: str, board_words: List[str]) -> bool:
    c = normalize_word(clue)
    for w in board_words:
        ww = normalize_word(w)
        if not c or not ww:
            continue
        if c in ww or ww in c:
            return True
    return False


def looks_like_word_token(clue: str) -> bool:
    # avoid punctuation / multi-token-ish outputs
    c = clue.strip()
    return bool(_WORD_RE.match(c))


def directness_score(clue: str, target_words: List[str], embedder: Embedder) -> float:
    if embedder is None:
        return 0.0
    cvec = embedder.embed(clue)
    sims = []
    for t in target_words:
        tvec = embedder.embed(t)
        sims.append(Embedder.cosine(cvec, tvec))
    return float(max(sims) if sims else 0.0)


def is_valid_clue(
    clue: str,
    board_words: List[str],
    target_words: List[str],
    embedder: Optional[Embedder],
    cfg: Dict[str, Any],
) -> Tuple[bool, str, float]:
    cons = cfg["constraints"]
    clue = clue.strip()

    if cons.get("single_word_only", True) and not is_single_word(clue):
        return False, "not_single_word", 0.0

    if not looks_like_word_token(clue):
        return False, "bad_token_shape", 0.0

    if cons.get("ban_board_words", True) and violates_board_overlap(clue, board_words):
        return False, "board_word", 0.0

    if cons.get("ban_substrings", True) and violates_substring_ban(clue, board_words):
        return False, "substring", 0.0

    # Optional embedding-based directness constraint
    if not bool(cons.get("enable_directness_check", True)):
        return True, "ok", 0.0

    d = directness_score(clue, target_words, embedder)
    if d >= float(cons["tau_direct"]):
        return False, "too_direct", float(d)

    return True, "ok", float(d)


def score_turn(
    guesses: List[str],
    labels_by_word: Dict[str, str],
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    guesses: normalized board words guessed in order (unique)
    labels_by_word: normalized_word -> label
    """
    r = cfg["reward"]
    stats = {"n_team": 0, "n_opp": 0, "n_neu": 0, "assassin": 0}

    reward = 0.0
    for g in guesses:
        lab = labels_by_word.get(g)
        if lab is None:
            continue
        if lab == "TEAM":
            reward += float(r["team_correct"])
            stats["n_team"] += 1
        elif lab == "OPP":
            reward += float(r["opp_wrong"])
            stats["n_opp"] += 1
        elif lab == "NEU":
            reward += float(r["neu_wrong"])
            stats["n_neu"] += 1
        elif lab == "ASSASSIN":
            reward += float(r["assassin_wrong"])
            stats["assassin"] += 1

    # optional shaping (off by default)
    reward += float(r.get("repeat_penalty", 0.0)) * 0.0
    reward += float(r.get("brevity_bonus", 0.0)) * 0.0

    return float(reward), stats