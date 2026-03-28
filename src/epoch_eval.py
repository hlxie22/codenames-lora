# src/epoch_eval.py
from __future__ import annotations

import contextlib
import copy
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import TrainerCallback
except Exception:
    TrainerCallback = object  # type: ignore[misc]

from .metrics import aggregate as aggregate_codenames
from .metrics import aggregate_paired, prefix_metric_keys
from .model_wrappers import apply_chat_template, Embedder, make_text_generator
from .prompting import get_spymaster_reasoning_mode
from .rollout import run_turns_batched
from .think_utils import split_think_and_rest
from .utils import read_jsonl, write_jsonl


def _dist_available() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _dist_rank_world() -> Tuple[int, int]:
    if not _dist_available():
        return (0, 1)
    import torch.distributed as dist
    return (dist.get_rank(), dist.get_world_size())


def _barrier() -> None:
    if not _dist_available():
        return
    import torch.distributed as dist
    dist.barrier()


def _all_gather_objects(obj: Any) -> List[Any]:
    if not _dist_available():
        return [obj]
    import torch.distributed as dist
    world = dist.get_world_size()
    out: List[Any] = [None] * world
    dist.all_gather_object(out, obj)
    return out


def _all_reduce_sum_float(x: float, device: torch.device) -> float:
    if not _dist_available():
        return float(x)
    import torch.distributed as dist
    t = torch.tensor([float(x)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _all_reduce_sum_int(x: int, device: torch.device) -> int:
    if not _dist_available():
        return int(x)
    import torch.distributed as dist
    t = torch.tensor([int(x)], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs))) if xs else 0.0


def _p90(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = int(0.9 * (len(ys) - 1))
    return float(ys[i])


def _shard_list(items: List[Any], rank: int, world: int) -> List[Any]:
    return [it for i, it in enumerate(items) if (i % world) == rank]


def _infer_device_from_model(model: Any) -> torch.device:
    try:
        p = next(model.parameters())
        return p.device
    except Exception:
        pass
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        return torch.device("cuda", lr)
    return torch.device("cpu")


def _resolve_model_tokenizer_device(
    trainer: Any,
    kwargs: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[Any], torch.device]:
    if trainer is not None:
        m = getattr(trainer, "model", None)
        acc = getattr(trainer, "accelerator", None)
        if acc is not None:
            try:
                m = acc.unwrap_model(m)
            except Exception:
                pass
        if hasattr(m, "module"):
            try:
                m = m.module
            except Exception:
                pass

        tok = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
        dev = getattr(getattr(trainer, "args", None), "device", None)
        if dev is None:
            dev = _infer_device_from_model(m)
        return m, tok, dev

    m = kwargs.get("model", None)
    tok = kwargs.get("tokenizer", None) or kwargs.get("processing_class", None)
    dev = _infer_device_from_model(m) if m is not None else _infer_device_from_model(object())
    return m, tok, dev


@contextlib.contextmanager
def _maybe_disable_adapter(model: Any, disable: bool):
    if not disable:
        yield
        return

    cm = getattr(model, "disable_adapter", None)
    if callable(cm):
        with cm():
            yield
        return

    yield


@dataclass
class _GenCfg:
    temperature: float
    top_p: float
    top_k: Optional[int]
    max_new_tokens: int


class _InTrainerGenerator:
    def __init__(self, model: Any, tokenizer: Any, device: torch.device, *, disable_adapter: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.disable_adapter = bool(disable_adapter)
        self.model_id = getattr(getattr(model, "config", None), "_name_or_path", "trainer_model")

    def format_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        return apply_chat_template(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def generate(
        self,
        prompt_or_messages: str | List[Dict[str, str]],
        gen_cfg: Any,
        seed: Optional[int] = None,
        *,
        use_chat_template: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if use_chat_template:
            assert isinstance(prompt_or_messages, list)
            prompt = self.format_chat(
                prompt_or_messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            assert isinstance(prompt_or_messages, str)
            prompt = prompt_or_messages

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        temperature = float(getattr(gen_cfg, "temperature", 0.0))
        top_p = float(getattr(gen_cfg, "top_p", 1.0))
        top_k = getattr(gen_cfg, "top_k", None)
        max_new_tokens = int(getattr(gen_cfg, "max_new_tokens", 256))

        do_sample = temperature > 1e-6
        gen_kwargs = dict(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=int(top_k) if (do_sample and top_k is not None) else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            with _maybe_disable_adapter(self.model, self.disable_adapter):
                out = self.model.generate(**gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def generate_batch(
        self,
        prompts: List[str],
        gen_cfg: Any,
        seeds: Optional[List[Optional[int]]] = None,
    ) -> List[str]:
        if seeds is None:
            seeds = [None] * len(prompts)
        return [
            self.generate(p, gen_cfg, seed=s, use_chat_template=False, enable_thinking=True)
            for p, s in zip(prompts, seeds)
        ]


def sample_codenames_eval_boards(cfg: Dict[str, Any], *, n_boards: int, seed: int) -> List[Dict[str, Any]]:
    boards_eval_path = cfg["paths"]["boards_eval_path"]
    boards_all = read_jsonl(boards_eval_path)
    if n_boards <= 0:
        return []
    rng = random.Random(int(seed))
    if n_boards >= len(boards_all):
        return boards_all
    idxs = list(range(len(boards_all)))
    rng.shuffle(idxs)
    take = sorted(idxs[:n_boards])
    return [boards_all[i] for i in take]


def _count_spymaster_reasoning_tokens(cfg: Dict[str, Any], tokenizer: Any, text: str) -> int:
    mode = get_spymaster_reasoning_mode(cfg)
    if mode not in {"visible", "native"}:
        return 0

    t = text or ""
    rationale = ""

    if mode == "native":
        rationale, _ = split_think_and_rest(t, which="first", allow_partial=True)
    else:
        m = list(re.finditer(r"^RATIONALE\s*:\s*(.+)$", t, flags=re.IGNORECASE | re.MULTILINE))
        if m:
            rationale = m[-1].group(1).strip()
        else:
            clue_match = list(re.finditer(r"^CLUE\s*:", t, flags=re.IGNORECASE | re.MULTILINE))
            if clue_match:
                rationale = t[: clue_match[-1].start()].strip()

    if not rationale:
        return 0
    try:
        ids = tokenizer(rationale, return_tensors=None, add_special_tokens=False)["input_ids"]
        return int(len(ids))
    except Exception:
        try:
            return int(len(tokenizer.encode(rationale, add_special_tokens=False)))
        except Exception:
            return 0


def eval_codenames_subset_raw(
    cfg: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    boards: List[Dict[str, Any]],
    *,
    guesser_override: Any | None = None,
) -> Dict[str, Any]:
    if not boards:
        return {"per_board": [], "reasoning_tokens": []}

    spymaster = _InTrainerGenerator(model, tokenizer, device, disable_adapter=False)
    guesser = guesser_override if guesser_override is not None else _InTrainerGenerator(model, tokenizer, device, disable_adapter=True)

    use_embed = bool(cfg.get("constraints", {}).get("enable_directness_check", True))
    embedder = None
    if use_embed:
        try:
            embedder = Embedder(cfg["models"]["embedding_model_id"], device="cpu")
        except Exception:
            embedder = None

    bs = int(cfg.get("inference", {}).get("batch_size", 1))
    bs = max(1, bs)

    per_board: List[Dict[str, Any]] = []
    reasoning_tokens: List[int] = []

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
    except Exception:
        pass

    try:
        for start in range(0, len(boards), bs):
            batch = boards[start : start + bs]
            bests, metas = run_turns_batched(
                batch,
                spymaster,
                guesser,
                embedder,
                cfg,
                n_candidates=1,
                max_resamples=1,
            )

            for b, best, meta in zip(batch, bests, metas):
                per_board.append(
                    {
                        "board_id": b["board_id"],
                        "reward": float(best.reward),
                        "clue": best.clue,
                        "num": int(best.num),
                        "guess_words": best.guess_words,
                        "stats": {**best.stats, "directness": float(best.directness)},
                        "clue_meta": {
                            "parse_valid": bool(best.parse_valid),
                            "valid": bool(best.valid),
                            "rejected_total": int(meta["rejected_total"]),
                            "rejection_counts": meta["rejection_counts"],
                        },
                    }
                )
                reasoning_tokens.append(_count_spymaster_reasoning_tokens(cfg, tokenizer, best.raw_spymaster_text or ""))
    finally:
        try:
            if was_training:
                model.train()
        except Exception:
            pass

    return {"per_board": per_board, "reasoning_tokens": reasoning_tokens}


def eval_wikitext2_ppl_raw(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    *,
    seed: int,
    n_texts: int,
    block_size: int,
    max_blocks: int,
) -> Dict[str, Any]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts_all = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
    rng = random.Random(int(seed))
    rng.shuffle(texts_all)
    texts = texts_all[: max(1, int(n_texts))]

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
    except Exception:
        pass

    total_nll = 0.0
    total_tokens = 0
    blocks_done = 0

    try:
        with torch.no_grad():
            for t in texts:
                enc = tokenizer(t, return_tensors="pt", truncation=False)
                ids = enc["input_ids"][0]
                for start in range(0, int(ids.shape[0]), int(block_size)):
                    if blocks_done >= int(max_blocks):
                        break
                    chunk = ids[start : start + int(block_size)]
                    if chunk.numel() < 2:
                        continue
                    inp = chunk.unsqueeze(0).to(device)
                    out = model(input_ids=inp, labels=inp)
                    loss = float(out.loss)
                    n_tok = int(inp.numel())
                    total_nll += loss * n_tok
                    total_tokens += n_tok
                    blocks_done += 1
                if blocks_done >= int(max_blocks):
                    break
    finally:
        try:
            if was_training:
                model.train()
        except Exception:
            pass

    return {"total_nll": total_nll, "total_tokens": total_tokens, "blocks_done": blocks_done}


_GSM8K_GOLD_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_GSM8K_FINAL_RE = re.compile(r"(?:FINAL|ANSWER)\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_GSM8K_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_ANY_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _norm_num_str(s: str) -> str:
    s = s.strip().replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    return s


def _extract_gsm8k_gold(answer: str) -> str:
    m = _GSM8K_GOLD_RE.search(answer or "")
    if m:
        return _norm_num_str(m.group(1))

    ms = _ANY_NUM_RE.findall(answer or "")
    if ms:
        return _norm_num_str(ms[-1])

    return ""


def _extract_gsm8k_pred(text: str) -> str:
    t = text or ""

    m = _GSM8K_FINAL_RE.search(t)
    if m:
        return _norm_num_str(m.group(1))

    m = _GSM8K_BOXED_RE.search(t)
    if m:
        inner = m.group(1)
        ms = _ANY_NUM_RE.findall(inner)
        if ms:
            return _norm_num_str(ms[-1])

    ms = _ANY_NUM_RE.findall(t)
    if ms:
        return _norm_num_str(ms[-1])

    return ""


def load_gsm8k_subset(*, n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = [{"question": q, "answer": a, "idx": i} for i, (q, a) in enumerate(zip(ds["question"], ds["answer"]))]

    if n <= 0:
        return []
    if n >= len(rows):
        return rows

    rng = random.Random(int(seed))
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    take = sorted(idxs[:n])
    return [rows[i] for i in take]


def eval_gsm8k_raw(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
) -> Dict[str, Any]:
    if not examples:
        return {"n": 0, "correct": 0, "records": []}

    gen = _InTrainerGenerator(model, tokenizer, device)
    gen_cfg = _GenCfg(
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_new_tokens=int(max_new_tokens),
    )

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
    except Exception:
        pass

    records: List[Dict[str, Any]] = []
    correct = 0

    try:
        for ex in examples:
            messages = [
                {
                    "role": "system",
                    "content": "Solve the math word problem carefully. End with exactly one line: FINAL: <number>",
                },
                {
                    "role": "user",
                    "content": ex["question"],
                },
            ]

            out = gen.generate(
                messages,
                gen_cfg,
                seed=1234 + int(ex["idx"]),
                use_chat_template=True,
                enable_thinking=True,
            )

            gold = _extract_gsm8k_gold(ex["answer"])
            pred = _extract_gsm8k_pred(out)
            ok = (gold != "" and pred == gold)
            correct += int(ok)

            records.append(
                {
                    "idx": int(ex["idx"]),
                    "gold": gold,
                    "pred": pred,
                    "correct": bool(ok),
                }
            )
    finally:
        try:
            if was_training:
                model.train()
        except Exception:
            pass

    return {"n": len(examples), "correct": correct, "records": records}


def load_humaneval_subset(*, n: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("openai/openai_humaneval", split="test")
    rows: List[Dict[str, Any]] = []
    for i in range(len(ds)):
        rows.append(
            {
                "task_id": ds[i]["task_id"],
                "prompt": ds[i]["prompt"],
                "test": ds[i]["test"],
                "entry_point": ds[i]["entry_point"],
                "idx": i,
            }
        )

    if n <= 0:
        return []
    if n >= len(rows):
        return rows

    rng = random.Random(int(seed))
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    take = sorted(idxs[:n])
    return [rows[i] for i in take]


def _strip_code_fences(text: str) -> str:
    t = text or ""
    m = re.search(r"```(?:python)?\n(.*?)```", t, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t.strip()


def _cleanup_humaneval_completion(text: str) -> str:
    try:
        from .think_utils import strip_think_blocks
        text = strip_think_blocks(text, remove_dangling_tags=True)
    except Exception:
        pass

    text = _strip_code_fences(text)

    lines = text.splitlines()
    cleaned: List[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip("\n")


def _run_humaneval_check(
    prompt: str,
    completion: str,
    test_code: str,
    entry_point: str,
    timeout_s: int,
) -> Tuple[bool, str]:
    program = (
        prompt
        + completion
        + "\n\n"
        + test_code
        + f"\n\ncheck({entry_point})\nprint('HUMANEVAL_PASS')\n"
    )

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "humaneval_check.py"
        p.write_text(program, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(p)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(timeout_s),
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"

        ok = (proc.returncode == 0) and ("HUMANEVAL_PASS" in proc.stdout)
        if ok:
            return True, "ok"

        detail = (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip()
        if len(detail) > 500:
            detail = detail[:500]
        return False, detail


def eval_humaneval_raw(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: List[Dict[str, Any]],
    *,
    max_new_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    if not examples:
        return {"n": 0, "passed": 0, "records": []}

    gen = _InTrainerGenerator(model, tokenizer, device)
    gen_cfg = _GenCfg(
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_new_tokens=int(max_new_tokens),
    )

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
    except Exception:
        pass

    passed = 0
    records: List[Dict[str, Any]] = []

    try:
        for ex in examples:
            out = gen.generate(
                ex["prompt"],
                gen_cfg,
                seed=4321 + int(ex["idx"]),
                use_chat_template=False,
                enable_thinking=False,
            )

            completion = _cleanup_humaneval_completion(out)
            ok, detail = _run_humaneval_check(
                ex["prompt"],
                completion,
                ex["test"],
                ex["entry_point"],
                timeout_s=int(timeout_s),
            )
            passed += int(ok)

            records.append(
                {
                    "task_id": ex["task_id"],
                    "passed": bool(ok),
                    "detail": "ok" if ok else detail,
                }
            )
    finally:
        try:
            if was_training:
                model.train()
        except Exception:
            pass

    return {"n": len(examples), "passed": passed, "records": records}


def plot_epoch_history(history_path: Path, plots_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    if not history_path.exists():
        return

    rows: List[Dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    def series(key: str) -> Tuple[List[int], List[float]]:
        xs, ys = [], []
        for r in rows:
            if key in r and isinstance(r.get(key), (int, float)) and isinstance(r.get("epoch"), int):
                xs.append(int(r["epoch"]))
                ys.append(float(r[key]))
        return xs, ys

    keys = [
        "reward_mean",
        "parse_valid_rate",
        "nonempty_clue_rate",
        "assassin_rate",
        "directness_mean",
        "wikitext2_ppl",
        "gsm8k_exact_match",
        "humaneval_pass_at_1",
        "codenames_spymaster_reasoning_tokens_mean",
        "codenames_spymaster_reasoning_tokens_p90",
        "paired_ref_reward_delta_mean",
        "paired_ref_reward_win_rate",
        "paired_ref_reward_loss_rate",
        "paired_ref_team_delta_mean",
        "paired_ref_opp_delta_mean",
        "paired_ref_neu_delta_mean",
        "paired_ref_assassin_rate_delta",
        "paired_ref_rejected_candidates_delta_mean",
        "paired_ref_clue_changed_rate",
    ]

    for k in keys:
        xs, ys = series(k)
        if len(xs) < 2:
            continue
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("epoch")
        plt.ylabel(k)
        plt.title(k)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{k}.png")
        plt.close()


class EpochEvalCallback(TrainerCallback):
    def __init__(self, cfg: Dict[str, Any], out_dir: str | Path):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.history_path = self.out_dir / "epoch_eval_history.jsonl"
        self.plots_dir = self.out_dir / "epoch_eval_plots"
        self.reference_per_board_path = self.out_dir / "epoch_eval_reference_per_board.jsonl"
        self.reference_metrics_path = self.out_dir / "epoch_eval_reference_metrics.json"
        self.per_epoch_board_dir = self.out_dir / "epoch_eval_per_board"
        self._last_epoch_logged: Optional[int] = None
        self._fixed_codenames_boards: Optional[List[Dict[str, Any]]] = None
        self._fixed_gsm8k_examples: Optional[List[Dict[str, Any]]] = None
        self._fixed_humaneval_examples: Optional[List[Dict[str, Any]]] = None
        self._reference_per_board_cache: Optional[List[Dict[str, Any]]] = None
        self._external_guesser = None

    def _get_external_guesser(self):
        sp_id = self.cfg["models"]["spymaster_model_id"]
        g_id = self.cfg["models"]["guesser_model_id"]

        if g_id == sp_id:
            return None

        if self._external_guesser is None:
            gcfg = copy.deepcopy(self.cfg)
            inf = gcfg.get("inference", {}) or {}
            inf["num_processes"] = 1
            if inf.get("backend") == "vllm":
                vcfg = inf.get("vllm", {}) or {}
                vcfg["tensor_parallel_size"] = 1
                inf["vllm"] = vcfg
            gcfg["inference"] = inf

            self._external_guesser = make_text_generator(g_id, gcfg)

        return self._external_guesser

    def on_train_begin(self, args, state, control, **kwargs):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.per_epoch_board_dir.mkdir(parents=True, exist_ok=True)

        ecfg = (self.cfg.get("epoch_eval", {}) or {})
        rank, world = _dist_rank_world()

        n_boards = int(ecfg.get("codenames_n_boards", 0))
        subset_seed = int(ecfg.get("codenames_subset_seed", 0))

        if n_boards > 0:
            self._fixed_codenames_boards = sample_codenames_eval_boards(
                self.cfg,
                n_boards=n_boards,
                seed=subset_seed,
            )

            if rank == 0:
                ids_path = self.out_dir / "epoch_eval_board_ids.json"
                ids = [b["board_id"] for b in self._fixed_codenames_boards]
                ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")

        gsm8k_n = int(ecfg.get("gsm8k_n", 0))
        if gsm8k_n > 0:
            gsm8k_seed = int(ecfg.get("gsm8k_seed", 2027))
            self._fixed_gsm8k_examples = load_gsm8k_subset(n=gsm8k_n, seed=gsm8k_seed)

        humaneval_n = int(ecfg.get("humaneval_n", 0))
        if humaneval_n > 0:
            humaneval_seed = int(ecfg.get("humaneval_seed", 2028))
            self._fixed_humaneval_examples = load_humaneval_subset(n=humaneval_n, seed=humaneval_seed)

        trainer = kwargs.get("trainer", None)
        model, tok, device = _resolve_model_tokenizer_device(trainer, kwargs)
        if model is None or tok is None or n_boards <= 0 or not self._fixed_codenames_boards:
            return control

        if self.reference_per_board_path.exists():
            if rank == 0:
                try:
                    self._reference_per_board_cache = read_jsonl(self.reference_per_board_path)
                except Exception:
                    self._reference_per_board_cache = None
            _barrier()
            return control

        guesser_override = self._get_external_guesser()
        subset_local = _shard_list(self._fixed_codenames_boards, rank, world)
        ref_raw = eval_codenames_subset_raw(
            self.cfg,
            model,
            tok,
            device,
            subset_local,
            guesser_override=guesser_override,
        )
        gathered_records = _all_gather_objects(list(ref_raw.get("per_board", []) or []))

        if rank == 0:
            per_board: List[Dict[str, Any]] = []
            for part in gathered_records:
                per_board.extend(part or [])
            per_board.sort(key=lambda r: str(r.get("board_id", "")))
            self._reference_per_board_cache = per_board
            write_jsonl(self.reference_per_board_path, per_board)
            ref_metrics = aggregate_codenames(per_board)
            self.reference_metrics_path.write_text(json.dumps(ref_metrics, indent=2), encoding="utf-8")

        _barrier()
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_f = getattr(state, "epoch", None)
        epoch_i = int(epoch_f) if epoch_f is not None else 0
        if self._last_epoch_logged is not None and epoch_i == self._last_epoch_logged:
            return control
        self._last_epoch_logged = epoch_i

        rank, world = _dist_rank_world()
        host = getattr(os, "uname", lambda: type("x", (), {"nodename": "unknown"})())().nodename

        def _errlog(msg: str) -> None:
            print(msg, file=sys.stderr, flush=True)
            try:
                p = self.out_dir / f"epoch_eval_rank{rank}.err"
                with p.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                    f.flush()
            except Exception:
                pass

        err_msgs: List[str] = []

        trainer = kwargs.get("trainer", None)
        model, tok, device = _resolve_model_tokenizer_device(trainer, kwargs)
        if model is None or tok is None:
            msg = f"[epoch_eval][rank {rank} host={host}] Could not resolve model/tokenizer; skipping."
            err_msgs.append(msg)
            all_errs = _all_gather_objects("\n\n".join(err_msgs))
            if rank == 0:
                for r, s in enumerate(all_errs):
                    if s:
                        _errlog(f"[epoch_eval] errors from rank {r}:\n{s}")
            _barrier()
            return control

        ecfg = (self.cfg.get("epoch_eval", {}) or {})
        tcfg = (self.cfg.get("training", {}) or {})

        metrics: Dict[str, Any] = {}
        t0 = time.time()

        per_board_local: List[Dict[str, Any]] = []
        reasoning_local: List[int] = []

        try:
            n_boards = int(ecfg.get("codenames_n_boards", 0))
            if n_boards > 0:
                subset = self._fixed_codenames_boards or []
                subset_local = _shard_list(subset, rank, world)
                guesser_override = self._get_external_guesser()

                raw = eval_codenames_subset_raw(
                    self.cfg,
                    model,
                    tok,
                    device,
                    subset_local,
                    guesser_override=guesser_override,
                )
                per_board_local = list(raw.get("per_board", []) or [])
                reasoning_local = [int(x) for x in (raw.get("reasoning_tokens", []) or [])]
        except Exception as e:
            err_msgs.append(
                f"[epoch_eval][rank {rank} host={host}] Codenames subset eval failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

        gathered_records = _all_gather_objects(per_board_local)
        per_board: List[Dict[str, Any]] = []
        for part in gathered_records:
            per_board.extend(part or [])
        per_board.sort(key=lambda r: str(r.get("board_id", "")))

        gathered_reasoning = _all_gather_objects(reasoning_local)
        reasoning_tokens: List[int] = []
        for part in gathered_reasoning:
            reasoning_tokens.extend([int(x) for x in (part or [])])

        if rank == 0 and per_board:
            try:
                m = aggregate_codenames(per_board)
                metrics["codenames_n_boards"] = int(m.get("n_boards", len(per_board)))
                metrics["reward_mean"] = float(m.get("reward_mean", 0.0))
                metrics["reward_median"] = float(m.get("reward_median", 0.0))
                ci = m.get("reward_ci95", (0.0, 0.0))
                metrics["reward_ci95_lo"] = float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else 0.0
                metrics["reward_ci95_hi"] = float(ci[1]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else 0.0
                metrics["assassin_rate"] = float(m.get("assassin_rate", 0.0))
                metrics["team_mean"] = float(m.get("team_mean", 0.0))
                metrics["opp_mean"] = float(m.get("opp_mean", 0.0))
                metrics["neu_mean"] = float(m.get("neu_mean", 0.0))
                metrics["directness_mean"] = float(m.get("directness_mean", 0.0))
                metrics["parse_valid_rate"] = float(m.get("parse_valid_rate", 0.0))
                metrics["parse_fail_rate"] = float(m.get("parse_fail_rate", 0.0))
                metrics["nonempty_clue_rate"] = float(m.get("nonempty_clue_rate", 0.0))
                metrics["rejected_candidates_mean"] = float(m.get("rejected_candidates_mean", 0.0))

                rej_counts = dict(m.get("rejection_reason_counts", {}) or {})
                metrics["rejection_reason_counts"] = rej_counts
                for k, v in sorted(rej_counts.items()):
                    safe_k = re.sub(r"[^0-9A-Za-z_]+", "_", str(k)).strip("_") or "unknown"
                    metrics[f"rejection_reason_count__{safe_k}"] = int(v)

                metrics["codenames_spymaster_reasoning_tokens_mean"] = float(_mean([float(x) for x in reasoning_tokens]))
                metrics["codenames_spymaster_reasoning_tokens_p90"] = float(_p90([float(x) for x in reasoning_tokens]))

                ref_records = self._reference_per_board_cache
                if ref_records is None and self.reference_per_board_path.exists():
                    try:
                        ref_records = read_jsonl(self.reference_per_board_path)
                        self._reference_per_board_cache = ref_records
                    except Exception:
                        ref_records = None

                if ref_records:
                    paired = aggregate_paired(per_board, ref_records)
                    metrics.update(prefix_metric_keys(paired, "paired_ref"))
                    paired_ci = paired.get("reward_delta_ci95", (0.0, 0.0))
                    if isinstance(paired_ci, (list, tuple)) and len(paired_ci) == 2:
                        metrics["paired_ref_reward_delta_ci95_lo"] = float(paired_ci[0])
                        metrics["paired_ref_reward_delta_ci95_hi"] = float(paired_ci[1])
            except Exception as e:
                err_msgs.append(
                    f"[epoch_eval][rank {rank} host={host}] Aggregation/metrics failed: {type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                )

        _barrier()

        wt_total_nll = 0.0
        wt_total_tokens = 0
        wt_blocks_done = 0

        try:
            wt_n = int(ecfg.get("wikitext_n", 0))
            if wt_n > 0:
                raw = eval_wikitext2_ppl_raw(
                    model=model,
                    tokenizer=tok,
                    device=device,
                    seed=int(tcfg.get("seed", 0)) + 777 + epoch_i,
                    n_texts=wt_n,
                    block_size=int(ecfg.get("wikitext_block_size", 512)),
                    max_blocks=int(ecfg.get("wikitext_max_blocks", 80)),
                )
                wt_total_nll = float(raw.get("total_nll", 0.0))
                wt_total_tokens = int(raw.get("total_tokens", 0))
                wt_blocks_done = int(raw.get("blocks_done", 0))
        except Exception as e:
            err_msgs.append(
                f"[epoch_eval][rank {rank} host={host}] WikiText-2 eval failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

        total_nll = _all_reduce_sum_float(float(wt_total_nll), device)
        total_tokens = _all_reduce_sum_int(int(wt_total_tokens), device)
        blocks_done = _all_reduce_sum_int(int(wt_blocks_done), device)

        if rank == 0 and int(ecfg.get("wikitext_n", 0)) > 0:
            try:
                ppl = math.exp(total_nll / max(1, total_tokens))
                metrics["wikitext2_blocks"] = int(blocks_done)
                metrics["wikitext2_tokens"] = int(total_tokens)
                metrics["wikitext2_ppl"] = float(ppl)
            except Exception as e:
                err_msgs.append(
                    f"[epoch_eval][rank {rank} host={host}] WikiText-2 metric compute failed: {type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                )

        _barrier()

        gsm_correct_local = 0
        gsm_total_local = 0

        try:
            gsm_n = int(ecfg.get("gsm8k_n", 0))
            if gsm_n > 0:
                subset = self._fixed_gsm8k_examples or []
                subset_local = _shard_list(subset, rank, world)

                raw = eval_gsm8k_raw(
                    model=model,
                    tokenizer=tok,
                    device=device,
                    examples=subset_local,
                    max_new_tokens=int(ecfg.get("gsm8k_max_new_tokens", 1024)),
                )
                gsm_correct_local = int(raw.get("correct", 0))
                gsm_total_local = int(raw.get("n", 0))
        except Exception as e:
            err_msgs.append(
                f"[epoch_eval][rank {rank} host={host}] GSM8K eval failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

        gsm_correct = _all_reduce_sum_int(gsm_correct_local, device)
        gsm_total = _all_reduce_sum_int(gsm_total_local, device)

        if rank == 0 and int(ecfg.get("gsm8k_n", 0)) > 0:
            metrics["gsm8k_n"] = int(gsm_total)
            metrics["gsm8k_exact_match"] = float(gsm_correct / max(1, gsm_total))

        _barrier()

        he_pass_local = 0
        he_total_local = 0

        try:
            he_n = int(ecfg.get("humaneval_n", 0))
            if he_n > 0:
                subset = self._fixed_humaneval_examples or []
                subset_local = _shard_list(subset, rank, world)

                raw = eval_humaneval_raw(
                    model=model,
                    tokenizer=tok,
                    device=device,
                    examples=subset_local,
                    max_new_tokens=int(ecfg.get("humaneval_max_new_tokens", 1024)),
                    timeout_s=int(ecfg.get("humaneval_timeout_s", 3)),
                )
                he_pass_local = int(raw.get("passed", 0))
                he_total_local = int(raw.get("n", 0))
        except Exception as e:
            err_msgs.append(
                f"[epoch_eval][rank {rank} host={host}] HumanEval failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

        he_pass = _all_reduce_sum_int(he_pass_local, device)
        he_total = _all_reduce_sum_int(he_total_local, device)

        if rank == 0 and int(ecfg.get("humaneval_n", 0)) > 0:
            metrics["humaneval_n"] = int(he_total)
            metrics["humaneval_pass_at_1"] = float(he_pass / max(1, he_total))

        _barrier()

        all_errs = _all_gather_objects("\n\n".join([m for m in err_msgs if m.strip()]))
        if rank == 0:
            for r, s in enumerate(all_errs):
                if s:
                    _errlog(f"[epoch_eval] errors from rank {r}:\n{s}")

        _barrier()

        metrics["epoch"] = int(epoch_i)
        metrics["global_step"] = int(getattr(state, "global_step", 0))
        metrics["epoch_eval_seconds"] = float(time.time() - t0)

        if rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            per_path = self.per_epoch_board_dir / f"epoch_{epoch_i:03d}_step_{int(getattr(state, 'global_step', 0)):06d}.jsonl"
            try:
                write_jsonl(per_path, per_board)
            except Exception as e:
                _errlog(
                    f"[epoch_eval][rank {rank} host={host}] Failed writing per-board epoch dump: {type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                )

            tr = trainer
            if tr is not None:
                try:
                    tr.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                except Exception:
                    pass

            try:
                plot_epoch_history(self.history_path, self.plots_dir)
            except Exception as e:
                _errlog(
                    f"[epoch_eval][rank {rank} host={host}] plot_epoch_history failed: {type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                )

            print(f"[epoch_eval] wrote epoch {epoch_i} metrics -> {self.history_path}", flush=True)

        _barrier()
        return control
