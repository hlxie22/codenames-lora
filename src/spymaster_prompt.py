from __future__ import annotations

from typing import Any, Dict, List

from .prompting import get_spymaster_reasoning_mode


def _visible_reasoning_header(cfg: Dict[str, Any]) -> str:
    max_words = int((cfg.get("qwen", {}) or {}).get("spymaster_visible_reasoning_max_words", 40))
    return f"""
VISIBLE RATIONALE INSTRUCTIONS:
- Include one short visible rationale line.
- Keep it concise and decision-focused.
- Do not restate the whole board.
- Keep the rationale under {max_words} words.
- Then give the final clue and number.

OUTPUT FORMAT (exactly):
RATIONALE: <short rationale>
CLUE: <one_word>
NUM: <integer>
"""


def _native_reasoning_header() -> str:
    return """
REASONING INSTRUCTIONS:
- Think privately if helpful before answering.
- Do not expose your chain-of-thought.
- Return only the final clue and number.

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
        "You are the Spymaster in a Codenames-like game.",
    )

    words_lines = []
    for i, w in enumerate(board_words):
        tag = "REVEALED" if revealed_mask[i] else "HIDDEN"
        words_lines.append(f"{i+1:02d}. {w} [{tag}]")

    team_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if (lab == "TEAM" and not rev)]
    opp_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if (lab == "OPP" and not rev)]
    neu_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if (lab == "NEU" and not rev)]
    ass_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if (lab == "ASSASSIN" and not rev)]

    mode = get_spymaster_reasoning_mode(cfg)
    if mode == "visible":
        reasoning_block = _visible_reasoning_header(cfg)
    elif mode == "native":
        reasoning_block = _native_reasoning_header()
    else:
        reasoning_block = _no_reasoning_header()

    user = f"""
BOARD WORDS (25):
{chr(10).join(words_lines)}

ROLE:
You are the Spymaster. Your job is to give one clue word and a number that help the Guesser select TEAM words.

LABEL MEANINGS:
- TEAM: words you want the Guesser to pick.
- OPP: opposing-team words; these are bad guesses.
- NEU: neutral bystander words; these are also bad guesses.
- ASSASSIN: the worst possible guess; hitting it is catastrophic and must be avoided.

YOUR TARGETS (TEAM) — choose a clue that points to some of these:
{", ".join(team_words)}

DANGEROUS WORDS TO AVOID ATTRACTING THE GUESSER TO:
OPP: {", ".join(opp_words)}
NEU: {", ".join(neu_words)}
ASSASSIN: {", ".join(ass_words)}

CONSTRAINTS:
- Output exactly ONE clue word (single token-like word).
- The clue must NOT be any board word.
- Avoid substrings of board words and vice versa.

GOAL:
Produce an indirect but helpful clue that helps a frozen Guesser pick TEAM words while avoiding OPP, NEU, and especially ASSASSIN.
{reasoning_block}
Now produce your clue.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
