# src/think_utils.py
from __future__ import annotations

import re
from typing import Literal, Tuple

# Matches a CLOSED think block. Non-greedy so multiple blocks work.
_THINK_CLOSED_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


Which = Literal["first", "last"]
Mode = Literal["inner", "full"]


def _pick_match(matches: list[re.Match[str]], which: Which) -> re.Match[str]:
    return matches[0] if which == "first" else matches[-1]


def _split_partial_think(text: str, *, which: Which) -> Tuple[str, str, bool]:
    """
    Best-effort handling for model-native traces when tags are partial or asymmetric.

    Qwen3 native thinking can appear in one of these forms in generated text:
      1) <think> ... </think> FINAL
      2) REASONING ... </think> FINAL      (no opening tag in decoded output)
      3) <think> ...                       (dangling opening tag)

    Returns (think, rest, found_partial_or_split).
    """
    t = text or ""
    if not t:
        return "", "", False

    opens = list(_THINK_OPEN_RE.finditer(t))
    closes = list(_THINK_CLOSE_RE.finditer(t))

    # Closing-only form: everything before the delimiter is hidden reasoning.
    if closes and not opens:
        m = _pick_match(closes, which)
        return t[: m.start()].strip(), t[m.end() :].strip(), True

    # Opening-only form: treat everything after the tag as reasoning.
    if opens and not closes:
        m = _pick_match(opens, which)
        return t[m.end() :].strip(), t[: m.start()].strip(), True

    # Both tags exist but the closed regex didn't match cleanly (odd nesting/spacing).
    if opens and closes:
        if which == "first":
            open_m = opens[0]
            close_m = next((c for c in closes if c.start() > open_m.start()), None)
        else:
            close_m = closes[-1]
            rev = [o for o in opens if o.start() < close_m.start()]
            open_m = rev[-1] if rev else None

        if open_m is not None and close_m is not None and close_m.start() > open_m.start():
            think = t[open_m.end() : close_m.start()].strip()
            rest = (t[: open_m.start()] + t[close_m.end() :]).strip()
            return think, rest, True

    return "", t, False


def extract_think(
    text: str | None,
    *,
    mode: Mode = "inner",
    which: Which = "first",
    allow_partial: bool = False,
) -> str:
    """
    Extract content from <think>...</think>.

    mode:
      - "inner": returns content inside tags
      - "full": returns the whole block including <think>...</think>

    which:
      - "first" or "last" match when multiple blocks exist

    allow_partial:
      - If True, tolerates outputs with only <think> OR only </think>.
        * only </think>: treat everything up to </think> as thinking
        * only <think>: treat everything after <think> as thinking
        * Qwen3-native closing-only traces are handled the same way
    """
    t = text or ""
    if not t:
        return ""

    matches = list(_THINK_CLOSED_RE.finditer(t))
    if matches:
        m = _pick_match(matches, which)
        if mode == "inner":
            return (m.group(1) or "").strip()
        return t[m.start() : m.end()].strip()

    if not allow_partial:
        return ""

    think, _, found = _split_partial_think(t, which=which)
    if not found:
        return ""

    if mode == "inner":
        return think.strip()

    # Reconstruct best-effort "full" block for partial traces.
    if "</think>" in t.lower() and "<think>" not in t.lower():
        return (think + "</think>").strip()
    if "<think>" in t.lower() and "</think>" not in t.lower():
        return ("<think>" + think).strip()
    return t.strip()


def split_think_and_rest(
    text: str | None,
    *,
    which: Which = "first",
    allow_partial: bool = False,
) -> Tuple[str, str]:
    """
    Returns (think_inner, rest_without_the_selected_think_trace).

    For closed blocks, removes the selected block and keeps surrounding text.
    For Qwen3-native closing-only traces, splits on </think> and returns the
    visible post-trace text as rest.
    """
    t = text or ""
    if not t:
        return "", ""

    matches = list(_THINK_CLOSED_RE.finditer(t))
    if matches:
        m = _pick_match(matches, which)
        think = (m.group(1) or "").strip()
        rest = (t[: m.start()] + t[m.end() :]).strip()
        rest = re.sub(r"</?think>", "", rest, flags=re.IGNORECASE).strip()
        return think, rest

    if allow_partial:
        think, rest, found = _split_partial_think(t, which=which)
        if found:
            return think.strip(), rest.strip()
        t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)

    return "", t.strip()


def strip_think_blocks(text: str | None, *, remove_dangling_tags: bool = True) -> str:
    """
    Remove reasoning traces from text.

    This handles both standard <think>...</think> blocks and Qwen3-native
    closing-only traces like:

        reasoning text ... </think>\n\nFINAL ANSWER
    """
    t = text or ""
    if not t:
        return ""

    # First remove any closed blocks.
    t = _THINK_CLOSED_RE.sub("", t)

    if remove_dangling_tags:
        think, rest = split_think_and_rest(t, which="first", allow_partial=True)
        if think:
            t = rest
        t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)

    return t.strip()


def has_think_trace(text: str | None, *, allow_partial: bool = True) -> bool:
    t = text or ""
    if not t:
        return False
    if _THINK_CLOSED_RE.search(t):
        return True
    if allow_partial:
        think, _, found = _split_partial_think(t, which="first")
        return bool(found and think.strip())
    return False
