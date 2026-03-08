from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from .model_wrappers import GenerationConfig, TextGenerator

Role = Literal["spymaster", "guesser", "generic"]


def _use_chat_template(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("qwen", {}).get("use_chat_template", False))


def _enable_thinking(cfg: Dict[str, Any], role: Role) -> bool:
    q = cfg.get("qwen", {}) or {}
    if role == "spymaster":
        return bool(q.get("enable_thinking_spymaster", True))
    if role == "guesser":
        return bool(q.get("enable_thinking_guesser", True))
    return bool(q.get("enable_thinking", True))


def render_prompt(generator: Any, messages: List[Dict[str, str]], cfg: Dict[str, Any], *, role: Role) -> str:
    """
    Canonical prompt rendering for both batched & non-batched paths.
    If chat templates are disabled, returns the raw user content.
    Otherwise uses generator.format_chat(...) with role-specific enable_thinking.
    """
    if not _use_chat_template(cfg):
        return messages[-1]["content"]
    return generator.format_chat(
        messages,
        add_generation_prompt=True,
        enable_thinking=_enable_thinking(cfg, role),
    )


def generate_from_messages(
    generator: TextGenerator,
    messages: List[Dict[str, str]],
    gen_cfg: GenerationConfig,
    cfg: Dict[str, Any],
    *,
    role: Role,
    seed: Optional[int] = None,
) -> str:
    """
    Canonical "messages -> generator.generate(...)" wrapper.

    This removes duplicated logic scattered around the codebase:
      - chat_template vs raw string prompt
      - role-specific enable_thinking flags
      - passing the correct flags into generator.generate()
    """
    use_chat = _use_chat_template(cfg)
    if use_chat:
        return generator.generate(
            messages,
            gen_cfg,
            seed=seed,
            use_chat_template=True,
            enable_thinking=_enable_thinking(cfg, role),
        )
    return generator.generate(
        messages[-1]["content"],
        gen_cfg,
        seed=seed,
        use_chat_template=False,
        enable_thinking=False,  # ignored for raw prompts
    )