from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from .model_wrappers import GenerationConfig, TextGenerator

Role = Literal["spymaster", "guesser", "generic"]
ReasoningMode = Literal["none", "visible", "native"]


def _use_chat_template(cfg: Dict[str, Any]) -> bool:
    return bool(cfg.get("qwen", {}).get("use_chat_template", False))


def get_spymaster_reasoning_mode(cfg: Dict[str, Any]) -> ReasoningMode:
    """
    Spymaster reasoning modes:
      - none:    final answer only, no native hidden thinking
      - visible: visible rationale in completion text
      - native:  model-native hidden thinking; visible output should remain final-only
    """
    q = cfg.get("qwen", {}) or {}
    mode = str(q.get("spymaster_reasoning_mode", "none")).strip().lower()
    if mode not in {"none", "visible", "native"}:
        raise ValueError(
            f"Unsupported qwen.spymaster_reasoning_mode={mode!r}. "
            "Expected one of: none, visible, native."
        )
    return mode  # type: ignore[return-value]


def get_guesser_reasoning_mode(cfg: Dict[str, Any]) -> ReasoningMode:
    """
    Guesser reasoning modes:
      - none:    final answer only, no native hidden thinking
      - visible: visible rationale in completion text
      - native:  model-native hidden thinking; visible output should remain final-only
    """
    q = cfg.get("qwen", {}) or {}
    mode = str(q.get("guesser_reasoning_mode", "none")).strip().lower()
    if mode not in {"none", "visible", "native"}:
        raise ValueError(
            f"Unsupported qwen.guesser_reasoning_mode={mode!r}. "
            "Expected one of: none, visible, native."
        )
    return mode  # type: ignore[return-value]


def approx_visible_reasoning_word_cap(max_tokens: int) -> int:
    """
    Conservative token->word conversion for prompt instructions.

    We keep the config in tokens, but LLMs usually follow word caps more naturally.
    Using a conservative ratio leaves some token headroom.
    """
    return max(1, int(max_tokens * 0.6))


def _enable_thinking(cfg: Dict[str, Any], role: Role) -> bool:
    if role == "spymaster":
        return get_spymaster_reasoning_mode(cfg) == "native"
    if role == "guesser":
        return get_guesser_reasoning_mode(cfg) == "native"
    return False


def _render_plain_messages(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"{role.upper()}:\n{content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts).strip()


def render_prompt(generator: Any, messages: List[Dict[str, str]], cfg: Dict[str, Any], *, role: Role) -> str:
    if not isinstance(messages, list):
        raise TypeError(
            f"render_prompt expected a list of chat messages, got {type(messages).__name__}: {messages!r}"
        )

    if not _use_chat_template(cfg):
        return _render_plain_messages(messages)

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
    Canonical messages -> generator.generate(...) wrapper.

    Handles:
      - chat_template vs plain-text prompting
      - role-specific native hidden thinking
      - correct flags into generator.generate()
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

    prompt = _render_plain_messages(messages)
    return generator.generate(
        prompt,
        gen_cfg,
        seed=seed,
        use_chat_template=False,
        enable_thinking=False,
    )