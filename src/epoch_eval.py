# src/epoch_eval.py
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# --------- small helpers ---------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_FINAL_RE = re.compile(r"FINAL\s*:\s*(.+)", re.IGNORECASE)
_GSM8K_REF_RE = re.compile(r"####\s*([^\n]+)")

def extract_think_span(text: str) -> str:
    m = _THINK_RE.search(text or "")
    return (m.group(1).strip() if m else "")

def strip_think_blocks(text: str) -> str:
    return _THINK_RE.sub("", text or "")

def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def normalize_number_str(s: str) -> str:
    s = (s or "").strip()
    # keep leading '-' and digits / decimal
    s = s.replace(",", "")
    # common wrappers
    s = s.strip().strip("`").strip()
    # if it looks like an int/float, normalize int-ish floats
    try:
        if re.fullmatch(r"[-+]?\d+(\.\d+)?", s):
            if "." in s:
                f = float(s)
                if abs(f - round(f)) < 1e-9:
                    return str(int(round(f)))
                # keep limited precision to avoid float noise
                return str(f).rstrip("0").rstrip(".")
            return str(int(s))
    except Exception:
        pass
    return s

def parse_gsm8k_ref(answer: str) -> str:
    m = _GSM8K_REF_RE.search(answer or "")
    if not m:
        return normalize_number_str(answer)
    return normalize_number_str(m.group(1))

def parse_gsm8k_pred(text: str) -> Optional[str]:
    text = text or ""
    m = _FINAL_RE.search(text)
    if m:
        # take first token on that line
        line = m.group(1).strip().splitlines()[0].strip()
        # sometimes model outputs "42." etc
        return normalize_number_str(line.strip().rstrip("."))
    # fallback: last number in text
    nums = re.findall(r"[-+]?\d+(\.\d+)?", text.replace(",", ""))
    if nums:
        # re.findall with groups returns tuples sometimes; redo without groups:
        nums2 = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        if nums2:
            return normalize_number_str(nums2[-1])
    return None

def extract_code_from_completion(text: str) -> str:
    """
    HumanEval: return python code only.
    - removes <think> blocks
    - extracts from ```python ... ``` if present
    - else extracts starting at first 'def '
    """
    t = strip_think_blocks(text or "").strip()

    # fenced code
    fence = re.search(r"```(?:python)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    # start at first def
    idx = t.find("def ")
    if idx != -1:
        return t[idx:].strip()

    return t

# --------- HumanEval sandbox runner ---------

_ALLOWED_IMPORTS = {
    "math", "itertools", "functools", "collections", "re", "string",
    "heapq", "bisect", "typing", "statistics", "fractions", "decimal"
}

def _limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = (name or "").split(".")[0]
    if root in _ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import blocked: {name}")

def _safe_builtins() -> Dict[str, Any]:
    # keep this minimal; HumanEval tasks usually need basic python only
    allowed = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "enumerate": enumerate,
        "float": float, "int": int, "len": len, "list": list, "max": max, "min": min,
        "range": range, "reversed": reversed, "round": round, "set": set, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        "__import__": _limited_import,
    }
    return allowed

def _humaneval_worker(payload: Dict[str, Any], q) -> None:
    """
    Runs in a subprocess. Returns (passed: bool, err: str|None)
    """
    try:
        # hard timeout via alarm
        import signal
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("timeout")))
        signal.alarm(int(payload.get("timeout_s", 3)))

        env: Dict[str, Any] = {"__builtins__": _safe_builtins()}
        code = payload["code"]
        test = payload["test"]
        entry_point = payload["entry_point"]

        exec(code, env, env)
        if entry_point not in env:
            q.put((False, f"missing entry_point: {entry_point}"))
            return

        exec(test, env, env)

        # Most HumanEval tests define check(candidate)
        check = env.get("check", None)
        if check is None:
            q.put((False, "missing check() in test"))
            return

        check(env[entry_point])
        q.put((True, None))
    except Exception as e:
        q.put((False, f"{type(e).__name__}: {e}"))

def run_humaneval_case(code: str, test: str, entry_point: str, timeout_s: int = 3) -> Tuple[bool, Optional[str]]:
    """
    Executes in a separate process with:
      - import whitelist
      - limited builtins
      - hard timeout
    """
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(
        target=_humaneval_worker,
        args=({"code": code, "test": test, "entry_point": entry_point, "timeout_s": timeout_s}, q),
    )
    p.start()
    p.join(timeout_s + 1)
    if p.is_alive():
        p.kill()
        return (False, "Timeout: killed")
    try:
        passed, err = q.get_nowait()
        return (bool(passed), err)
    except Exception:
        return (False, "No result")

# --------- Model wrapper for codenames rollout ---------

class InMemoryHFGenerator:
    """
    Minimal TextGenerator-like wrapper around an in-memory (Peft) model + tokenizer.
    Implements generate() and generate_batch() (sequential per-item seeds).
    """
    def __init__(self, model, tokenizer, *, disable_adapter: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = getattr(getattr(model, "config", None), "_name_or_path", "in_memory")
        self._disable_adapter = disable_adapter

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_chat(self, messages: List[Dict[str, str]], *, add_generation_prompt=True, enable_thinking=True) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def _maybe_disable_adapter_ctx(self):
        m = self.model
        if not self._disable_adapter:
            return _NullCtx()
        # PEFT usually provides a context manager
        if hasattr(m, "disable_adapter"):
            try:
                return m.disable_adapter()
            except TypeError:
                # some versions require calling without args; still a ctx
                return m.disable_adapter()
        return _NullCtx()

    @torch.no_grad()
    def generate(self, prompt: str, *, temperature: float, top_p: float, top_k: int, max_new_tokens: int, seed: Optional[int] = None) -> str:
        if seed is not None:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = temperature is not None and float(temperature) > 1e-6
        gen_kwargs = dict(
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            top_k=int(top_k) if (do_sample and top_k is not None) else None,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with self._maybe_disable_adapter_ctx():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def generate_batch(self, prompts: List[str], *, temperature: float, top_p: float, top_k: int, max_new_tokens: int, seeds: Optional[List[Optional[int]]] = None) -> List[str]:
        # sequential per prompt so seeds are per-example
        if seeds is None:
            seeds = [None] * len(prompts)
        outs: List[str] = []
        for p, sd in zip(prompts, seeds):
            outs.append(self.generate(p, temperature=temperature, top_p=top_p, top_k=top_k, max_new_tokens=max_new_tokens, seed=sd))
        return outs

class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False

# --------- Eval routines ---------

@dataclass
class ProbeSamples:
    gsm8k: List[Dict[str, Any]]
    humaneval: List[Dict[str, Any]]
    wikitext: List[str]

def load_probe_samples(
    *,
    seed: int,
    gsm8k_n: int,
    humaneval_n: int,
    wikitext_n: int,
) -> ProbeSamples:
    from datasets import load_dataset

    rng = random.Random(seed)

    gsm = load_dataset("gsm8k", "main", split="test")
    gsm_idx = rng.sample(range(len(gsm)), k=min(gsm8k_n, len(gsm)))
    gsm_samp = [gsm[i] for i in gsm_idx]

    he = load_dataset("openai/openai_humaneval", split="test")
    he_idx = rng.sample(range(len(he)), k=min(humaneval_n, len(he)))
    he_samp = [he[i] for i in he_idx]

    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in wt["text"] if isinstance(t, str) and t.strip()]
    wt_idx = rng.sample(range(len(texts)), k=min(wikitext_n, len(texts)))
    wt_samp = [texts[i] for i in wt_idx]

    return ProbeSamples(gsm8k=gsm_samp, humaneval=he_samp, wikitext=wt_samp)

@torch.no_grad()
def eval_gsm8k(
    *,
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 384,
) -> Dict[str, Any]:
    model.eval()
    correct = 0
    n = 0
    think_tokens: List[int] = []

    for ex in samples:
        q = ex["question"]
        ref = parse_gsm8k_ref(ex["answer"])

        messages = [
            {"role": "system", "content": "You are a careful math assistant."},
            {"role": "user", "content": (
                "Solve the problem.\n"
                "Put your reasoning inside <think>...</think>.\n"
                "Then output exactly one line: FINAL: <number>\n\n"
                f"PROBLEM:\n{q}\n"
            )},
        ]
        prompt = None
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        think = extract_think_span(gen)
        think_tokens.append(count_tokens(tokenizer, think))

        pred = parse_gsm8k_pred(gen)
        if pred is not None and pred == ref:
            correct += 1
        n += 1

    acc = correct / max(1, n)
    return {
        "gsm8k_n": n,
        "gsm8k_acc": float(acc),
        "gsm8k_think_tokens_mean": float(sum(think_tokens) / max(1, len(think_tokens))),
        "gsm8k_think_tokens_p90": float(sorted(think_tokens)[int(0.9 * (len(think_tokens) - 1))]) if think_tokens else 0.0,
    }

@torch.no_grad()
def eval_humaneval(
    *,
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 384,
    timeout_s: int = 3,
) -> Dict[str, Any]:
    model.eval()
    passed = 0
    n = 0
    think_tokens: List[int] = []

    for ex in samples:
        prompt = ex["prompt"]
        test = ex["test"]
        entry = ex["entry_point"]

        messages = [
            {"role": "system", "content": "You write correct Python functions."},
            {"role": "user", "content": (
                "Write the Python function described below.\n"
                "Optional: if you think, put it in <think>...</think>.\n"
                "Then output ONLY valid Python code (no backticks, no extra text).\n\n"
                f"{prompt}"
            )},
        ]
        chat = None
        try:
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(chat, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        think = extract_think_span(gen)
        think_tokens.append(count_tokens(tokenizer, think))

        code_only = extract_code_from_completion(gen)
        full_code = prompt + "\n" + code_only + "\n"
        ok, _err = run_humaneval_case(full_code, test, entry_point=entry, timeout_s=timeout_s)

        passed += 1 if ok else 0
        n += 1

    return {
        "humaneval_n": n,
        "humaneval_pass_rate": float(passed / max(1, n)),
        "humaneval_think_tokens_mean": float(sum(think_tokens) / max(1, len(think_tokens))),
        "humaneval_think_tokens_p90": float(sorted(think_tokens)[int(0.9 * (len(think_tokens) - 1))]) if think_tokens else 0.0,
    }

@torch.no_grad()
def eval_wikitext2_ppl(
    *,
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    block_size: int = 512,
    max_blocks: int = 80,  # cap runtime
) -> Dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    blocks_done = 0
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        if input_ids.numel() < 2:
            continue

        # chunk
        seq_len = input_ids.shape[1]
        for i in range(0, seq_len, block_size):
            if blocks_done >= max_blocks:
                break
            chunk = input_ids[:, i : i + block_size]
            if chunk.shape[1] < 2:
                continue
            labels = chunk.clone()
            out = model(chunk, labels=labels)
            # loss is mean over tokens
            n_tokens = labels.numel()
            total_nll += float(out.loss) * n_tokens
            total_tokens += n_tokens
            blocks_done += 1
        if blocks_done >= max_blocks:
            break

    ppl = math.exp(total_nll / max(1, total_tokens))
    return {
        "wikitext2_blocks": int(blocks_done),
        "wikitext2_tokens": int(total_tokens),
        "wikitext2_ppl": float(ppl),
    }

def eval_codenames_subset(
    *,
    cfg: Dict[str, Any],
    model,
    tokenizer,
    device: torch.device,
    n_boards: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Computes the same key fields as src/metrics.py aggregate() but on a sampled subset of eval boards.
    """
    from .utils import read_jsonl
    from .rollout import run_turns_batched
    from .metrics import aggregate

    rng = random.Random(seed)

    boards = read_jsonl(cfg["paths"]["boards_eval_path"])
    if not boards:
        return {"codenames_n_boards": 0}

    idx = rng.sample(range(len(boards)), k=min(n_boards, len(boards)))
    subset = [boards[i] for i in idx]

    # speed overrides: keep eval cheap
    # (your config defaults are huge max_new_tokens)
    sp_max = min(int(cfg["decoding"].get("spymaster_max_new_tokens", 512)), 512)
    g_max = min(int(cfg["decoding"].get("guesser_max_new_tokens", 256)), 256)

    # build in-memory generators
    sp_gen = InMemoryHFGenerator(model, tokenizer, disable_adapter=False)
    g_gen = InMemoryHFGenerator(model, tokenizer, disable_adapter=True)

    # n_candidates=1 for cheap epoch eval
    bests, metas = run_turns_batched(
        subset,
        sp_gen,
        g_gen,
        embedder=None,   # config has enable_directness_check=false, so keep it off here
        cfg={
            **cfg,
            "decoding": {
                **cfg["decoding"],
                "n_candidates": 1,
                "spymaster_max_new_tokens": sp_max,
                "guesser_max_new_tokens": g_max,
            },
        },
        n_candidates=1,
    )

    # construct per-board records like eval.py
    per_board = []
    think_tok = []
    for b, best, meta in zip(subset, bests, metas):
        think = extract_think_span(best.raw_spymaster_text)
        think_tok.append(count_tokens(tokenizer, think))
        per_board.append({
            "board_id": b["board_id"],
            "reward": float(best.reward),
            "clue": best.clue,
            "stats": {**best.stats, "directness": float(best.directness)},
        })

    m = aggregate(per_board)
    return {
        "codenames_n_boards": int(m.get("n_boards", len(per_board))),
        "reward_mean": float(m.get("reward_mean", 0.0)),
        "reward_median": float(m.get("reward_median", 0.0)),
        "reward_ci95_lo": float(m.get("reward_ci95", (0.0, 0.0))[0]),
        "reward_ci95_hi": float(m.get("reward_ci95", (0.0, 0.0))[1]),
        "assassin_rate": float(m.get("assassin_rate", 0.0)),
        "team_mean": float(m.get("team_mean", 0.0)),
        "opp_mean": float(m.get("opp_mean", 0.0)),
        "neu_mean": float(m.get("neu_mean", 0.0)),
        "codenames_spymaster_think_tokens_mean": float(sum(think_tok) / max(1, len(think_tok))),
        "codenames_spymaster_think_tokens_p90": float(sorted(think_tok)[int(0.9 * (len(think_tok) - 1))]) if think_tok else 0.0,
    }

# --------- plotting ---------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out

def plot_epoch_history(history_path: Path, out_dir: Path) -> None:
    """
    Writes a small set of PNG plots vs epoch.
    """
    import matplotlib.pyplot as plt

    rows = _read_jsonl(history_path)
    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [r["epoch"] for r in rows]

    def series(key: str) -> List[float]:
        xs = []
        for r in rows:
            v = r.get(key, None)
            xs.append(float(v) if v is not None else float("nan"))
        return xs

    # 1) Correctness metrics
    plt.figure()
    plt.plot(epochs, series("gsm8k_acc"), label="GSM8K acc")
    plt.plot(epochs, series("humaneval_pass_rate"), label="HumanEval pass rate")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title("Correctness vs epoch")
    plt.tight_layout()
    plt.savefig(out_dir / "correctness.png")
    plt.close()

    # 2) Chain-of-thought verbosity (token count inside <think>)
    plt.figure()
    plt.plot(epochs, series("gsm8k_think_tokens_mean"), label="GSM8K think tok (mean)")
    plt.plot(epochs, series("humaneval_think_tokens_mean"), label="HumanEval think tok (mean)")
    plt.plot(epochs, series("codenames_spymaster_think_tokens_mean"), label="Codenames spymaster think tok (mean)")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("tokens")
    plt.title("<think> verbosity vs epoch")
    plt.tight_layout()
    plt.savefig(out_dir / "think_verbosity.png")
    plt.close()

    # 3) WikiText-2 perplexity
    plt.figure()
    plt.plot(epochs, series("wikitext2_ppl"))
    plt.xlabel("epoch")
    plt.ylabel("perplexity")
    plt.title("WikiText-2 PPL vs epoch")
    plt.tight_layout()
    plt.savefig(out_dir / "wikitext2_ppl.png")
    plt.close()

    # 4) Train loss + EMA
    plt.figure()
    plt.plot(epochs, series("train_loss_epoch_mean"), label="train loss (epoch mean)")
    if any(not math.isnan(x) for x in series("train_loss_epoch_ema")):
        plt.plot(epochs, series("train_loss_epoch_ema"), label="train loss (EMA)")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Train loss vs epoch")
    plt.tight_layout()
    plt.savefig(out_dir / "train_loss.png")
    plt.close()

    # 5) Grad norm (if present)
    grad = series("grad_norm_epoch_mean")
    if any(not math.isnan(x) for x in grad):
        plt.figure()
        plt.plot(epochs, grad)
        plt.xlabel("epoch")
        plt.ylabel("grad_norm")
        plt.title("Gradient norm vs epoch")
        plt.tight_layout()
        plt.savefig(out_dir / "grad_norm.png")
        plt.close()

    # 6) Codenames: reward with CI band
    plt.figure()
    rm = series("reward_mean")
    rlo = series("reward_ci95_lo")
    rhi = series("reward_ci95_hi")
    plt.plot(epochs, rm, label="reward_mean")
    # CI band
    try:
        import numpy as np
        x = np.array(epochs, dtype=float)
        lo = np.array(rlo, dtype=float)
        hi = np.array(rhi, dtype=float)
        plt.fill_between(x, lo, hi, alpha=0.2, label="CI95")
    except Exception:
        pass
    plt.plot(epochs, series("reward_median"), label="reward_median")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("reward")
    plt.title("Codenames reward vs epoch (subset eval)")
    plt.tight_layout()
    plt.savefig(out_dir / "codenames_reward.png")
    plt.close()

    # 7) Codenames behavior profile
    plt.figure()
    plt.plot(epochs, series("assassin_rate"), label="assassin_rate")
    plt.plot(epochs, series("team_mean"), label="team_mean")
    plt.plot(epochs, series("opp_mean"), label="opp_mean")
    plt.plot(epochs, series("neu_mean"), label="neu_mean")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Codenames behavior metrics vs epoch (subset eval)")
    plt.tight_layout()
    plt.savefig(out_dir / "codenames_behavior.png")
    plt.close()