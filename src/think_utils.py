# src/think_utils.py
from __future__ import annotations

import re
from typing import Literal, Optional, Tuple

# Matches a CLOSED think block. Non-greedy so multiple blocks work.
_THINK_CLOSED_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


Which = Literal["first", "last"]
Mode = Literal["inner", "full"]


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
    """
    t = text or ""
    if not t:
        return ""

    matches = list(_THINK_CLOSED_RE.finditer(t))
    if matches:
        m = matches[0] if which == "first" else matches[-1]
        if mode == "inner":
            return (m.group(1) or "").strip()
        return t[m.start() : m.end()].strip()

    if not allow_partial:
        return ""

    low = t.lower()
    lo = low.rfind("<think>")
    hi = low.rfind("</think>")

    # both present but regex didn't match (weird nesting/spacing) -> best-effort slice
    if lo != -1 and hi != -1 and hi > lo:
        start = lo if mode == "full" else (lo + len("<think>"))
        end = (hi + len("</think>")) if mode == "full" else hi
        return t[start:end].strip()

    # only closing tag present: keep everything up to it
    if hi != -1:
        end = hi + (len("</think>") if mode == "full" else 0)
        return t[:end].strip()

    # only opening tag present: keep everything after it
    if lo != -1:
        start = lo if mode == "full" else (lo + len("<think>"))
        return t[start:].strip()

    return ""


def strip_think_blocks(text: str | None, *, remove_dangling_tags: bool = True) -> str:
    """
    Remove all CLOSED <think>...</think> blocks.
    Optionally also remove dangling <think> or </think> tags.
    """
    t = text or ""
    if not t:
        return ""

    t = _THINK_CLOSED_RE.sub("", t)

    if remove_dangling_tags:
        # If a model emits a single dangling tag, remove it.
        t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)

    return t


def split_think_and_rest(
    text: str | None,
    *,
    which: Which = "first",
    allow_partial: bool = False,
) -> Tuple[str, str]:
    """
    Returns (think_inner, rest_without_closed_think_blocks).
    Useful when you want to log think separately but score/eval on the rest.
    """
    t = text or ""
    think = extract_think(t, mode="inner", which=which, allow_partial=allow_partial)
    rest = strip_think_blocks(t)
    return think, rest