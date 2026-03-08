from typing import Dict, List, Any

def build_guesser_messages(
    board_words: List[str],
    revealed_mask: List[bool],
    clue: str,
    num: int,
    cfg: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = cfg.get("qwen", {}).get("system_guesser", "You are the Guesser in a Codenames-like game.")
    visible = [w for w, rev in zip(board_words, revealed_mask) if not rev]

    user = f"""You only see the board words and the clue.
 
 BOARD WORDS:
 {", ".join(visible)}

CLUE: {clue}
NUM: {num}

Return at least {num} guesses if possible.

OUTPUT FORMAT (exactly):
GUESSES: word1, word2, word3
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
