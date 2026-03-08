from __future__ import annotations

from typing import Dict, List, Any


def build_spymaster_messages(
    board_words: List[str],
    labels: List[str],
    revealed_mask: List[bool],
    cfg: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = cfg.get("qwen", {}).get("system_spymaster", "You are the Spymaster in a Codenames-like game.")

    words_lines = []
    for i, w in enumerate(board_words):
        tag = "REVEALED" if revealed_mask[i] else "HIDDEN"
        words_lines.append(f"{i+1:02d}. {w} [{tag}]")

    team_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if (lab == "TEAM" and not rev)]
    opp_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if (lab == "OPP" and not rev)]
    neu_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if (lab == "NEU" and not rev)]
    ass_words = [w for w, lab, rev in zip(board_words, labels, revealed_mask) if (lab == "ASSASSIN" and not rev)]

    tau = cfg["constraints"]["tau_direct"]

    user = f"""
BOARD WORDS (25):
{chr(10).join(words_lines)}

YOUR TARGETS (TEAM) — choose a clue that points to some of these:
{", ".join(team_words)}

DANGEROUS / FORBIDDEN (avoid leading the guesser to these):
OPP: {", ".join(opp_words)}
NEU: {", ".join(neu_words)}
ASSASSIN: {", ".join(ass_words)}

CONSTRAINTS:
- Output exactly ONE clue word (single token-like word).
- The clue must NOT be any board word.
- Avoid substrings of board words and vice versa.

GOAL:
Produce an indirect but helpful clue that helps a frozen Guesser pick TEAM words while avoiding OPP/NEU/ASSASSIN.

OUTPUT FORMAT (exactly):
<think>
...
</think>
CLUE: <one_word>
NUM: <integer>

Now produce your clue.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]