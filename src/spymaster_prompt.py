from __future__ import annotations

from typing import Any, Dict, List

from .prompting import approx_visible_reasoning_word_cap, get_spymaster_reasoning_mode


def _visible_reasoning_header(cfg: Dict[str, Any]) -> str:
    max_tokens = int((cfg.get("qwen", {}) or {}).get("spymaster_visible_reasoning_max_tokens", 40))
    max_words = approx_visible_reasoning_word_cap(max_tokens)
    return f"""
Use the rationale to reason through options, tradeoffs, and risks before settling on a final answer.

OUTPUT FORMAT (exactly):
RATIONALE: <reasoning>
CLUE: <one_word>
NUM: <integer>
"""


def _native_reasoning_header() -> str:
    return """
Think carefully, then return only the final answer in the format below.

OUTPUT FORMAT (exactly):
CLUE: <one_word>
NUM: <integer>
"""


def _no_reasoning_header() -> str:
    return """
OUTPUT FORMAT (exactly):
CLUE: <one_word>
NUM: <integer>
"""


def build_spymaster_messages(
    board_words: List[str],
    labels: List[str],
    revealed_mask: List[bool],
    cfg: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = cfg.get("qwen", {}).get(
        "system_spymaster",
        "You are the Spymaster in a Codenames-like game. Reason carefully and thoroughly before answering.",
    )

    hidden_words = [w for w, rev in zip(board_words, revealed_mask) if not rev]
    team_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if lab == "TEAM" and not rev]
    opp_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if lab == "OPP" and not rev]
    neu_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if lab == "NEU" and not rev]
    ass_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if lab == "ASSASSIN" and not rev]

    mode = get_spymaster_reasoning_mode(cfg)
    if mode == "visible":
        reasoning_block = _visible_reasoning_header(cfg)
    elif mode == "native":
        reasoning_block = _native_reasoning_header()
    else:
        reasoning_block = _no_reasoning_header()

    user = f"""
ROLE:
You are the Spymaster. Give one clue word and a number that help the Guesser select TEAM words.

HIDDEN TEAM WORDS:
{", ".join(team_words)}

HIDDEN OPP WORDS:
{", ".join(opp_words)}

HIDDEN NEUTRAL WORDS:
{", ".join(neu_words)}

HIDDEN ASSASSIN WORDS:
{", ".join(ass_words)}

CONSTRAINTS:
- Output exactly ONE clue word.
- The clue must NOT be any hidden board word.
- Avoid substrings of hidden board words and vice versa.

GOAL:
Produce an indirect but helpful clue that helps the Guesser pick TEAM words while avoiding OPP, NEU, and especially ASSASSIN.

{reasoning_block}
Now produce your clue.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]