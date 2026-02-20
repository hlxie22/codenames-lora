# src/prompting.py
from typing import Any, Dict, List

def render_prompt(generator, messages: List[Dict[str, str]], cfg: Dict[str, Any], *, role: str) -> str:
    use_chat = bool(cfg.get("qwen", {}).get("use_chat_template", False))
    if not use_chat:
        return messages[-1]["content"]

    think_key = "enable_thinking_spymaster" if role == "spymaster" else "enable_thinking_guesser"
    enable_thinking = bool(cfg.get("qwen", {}).get(think_key, True))
    return generator.format_chat(messages, add_generation_prompt=True, enable_thinking=enable_thinking)