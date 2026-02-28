# src/chat_formatting.py
from __future__ import annotations

from typing import Any, Dict, List, Literal

Role = Literal["spymaster", "guesser", "generic"]


def cfg_use_chat_template(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("qwen", {}).get("use_chat_template", False))


def cfg_enable_thinking(cfg: Dict[str, Any], role: Role) -> bool:
    q = cfg.get("qwen", {}) or {}
    if role == "spymaster":
        return bool(q.get("enable_thinking_spymaster", True))
    if role == "guesser":
        return bool(q.get("enable_thinking_guesser", True))
    return bool(q.get("enable_thinking", True))


def apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """
    One canonical place to handle tokenizer.apply_chat_template signature drift.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )