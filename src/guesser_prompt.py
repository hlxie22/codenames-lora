from __future__ import annotations

from typing import Dict, List, Any

from .prompting import approx_visible_reasoning_word_cap, get_guesser_reasoning_mode


def _visible_reasoning_header(cfg: Dict[str, Any]) -> str:
    max_tokens = int((cfg.get("qwen", {}) or {}).get("guesser_visible_reasoning_max_tokens", 40))
    max_words = approx_visible_reasoning_word_cap(max_tokens)
    return f"""
Use the rationale to reason through likely matches, alternatives, and risks before settling on the final guesses.

OUTPUT FORMAT (exactly):
RATIONALE: <reasoning>
GUESSES: word1, word2, word3
"""


def _native_reasoning_header() -> str:
    return """
Think carefully, then return only the final answer in the format below.

OUTPUT FORMAT (exactly):
GUESSES: word1, word2, word3
"""


def _no_reasoning_header() -> str:
    return """
OUTPUT FORMAT (exactly):
GUESSES: word1, word2, word3
"""


def build_guesser_messages(
    board_words: List[str],
    revealed_mask: List[bool],
    clue: str,
    num: int,
    cfg: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = cfg.get(
        "qwen", {}
    ).get(
        "system_guesser",
        "You are the Guesser in a Codenames-like game. Reason carefully and thoroughly before answering.",
    )
    visible = [w for w, rev in zip(board_words, revealed_mask) if not rev]

    mode = get_guesser_reasoning_mode(cfg)
    if mode == "visible":
        reasoning_block = _visible_reasoning_header(cfg)
    elif mode == "native":
        reasoning_block = _native_reasoning_header()
    else:
        reasoning_block = _no_reasoning_header()

    user = f"""You only see the board words and the clue.

BOARD WORDS:
{", ".join(visible)}

CLUE: {clue}
NUM: {num}

Choose the words most likely intended by the clue.
Return at least {num} guesses if possible.
List guesses in descending confidence.

{reasoning_block}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]