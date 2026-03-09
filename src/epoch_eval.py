# src/epoch_eval.py
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import TrainerCallback
except Exception:  # very defensive
    TrainerCallback = object  # type: ignore[misc]

from .metrics import aggregate as aggregate_codenames
from .model_wrappers import apply_chat_template, Embedder
from .rollout import run_turns_batched
from .utils import read_jsonl


# -------------------------
# Distributed helpers
# -------------------------

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


# -------------------------
# Small stats helpers
# -------------------------

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
    # Best-effort: parameters device
    try:
        p = next(model.parameters())
        return p.device
    except Exception:
        pass
    # Fall back to LOCAL_RANK cuda if available
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        return torch.device("cuda", lr)
    return torch.device("cpu")


def _resolve_model_tokenizer_device(
    trainer: Any,
    kwargs: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[Any], torch.device]:
    """
    Transformers/TRL callbacks often do NOT include `trainer` in kwargs.
    So:
      - If trainer is present, unwrap + take its tokenizer/device.
      - Else, fall back to kwargs["model"] and kwargs["tokenizer"] (or processing_class).
    """
    if trainer is not None:
        # unwrap model (accelerate/deepspeed)
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

    # no trainer: use kwargs
    m = kwargs.get("model", None)
    tok = kwargs.get("tokenizer", None) or kwargs.get("processing_class", None)
    dev = _infer_device_from_model(m) if m is not None else _infer_device_from_model(object())
    return m, tok, dev


# -------------------------
# In-trainer generator (no second model load)
# -------------------------

@dataclass
class _GenCfg:
    temperature: float
    top_p: float
    top_k: Optional[int]
    max_new_tokens: int


class _InTrainerGenerator:
    """
    Minimal TextGenerator-like adapter around the *current trainer model + tokenizer*.

    Must support:
      - format_chat(messages, add_generation_prompt, enable_thinking)
      - generate(prompt_or_messages, gen_cfg, seed, use_chat_template, enable_thinking)
      - generate_batch(prompts, gen_cfg, seeds)
    """

    def __init__(self, model: Any, tokenizer: Any, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
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

        # gen_cfg is model_wrappers.GenerationConfig in your codebase
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
            try:
                out = self.model.generate(**gen_kwargs, synced_gpus=_dist_available())
            except TypeError:
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


# -------------------------
# Eval pieces
# -------------------------

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


def _count_think_tokens(tokenizer: Any, text: str) -> int:
    """
    Count tokens inside the LAST <think>...</think> block if present, else 0.
    Uses your think_utils if available; falls back to regex.
    """
    try:
        from .think_utils import extract_think
        think = extract_think(text, mode="inner", which="last", allow_partial=True)
    except Exception:
        import re
        m = list(re.finditer(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL))
        think = (m[-1].group(1).strip() if m else "")

    if not think:
        return 0
    try:
        ids = tokenizer(think, return_tensors=None, add_special_tokens=False)["input_ids"]
        return int(len(ids))
    except Exception:
        try:
            return int(len(tokenizer.encode(think, add_special_tokens=False)))
        except Exception:
            return 0


def eval_codenames_subset_raw(
    cfg: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    boards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run Codenames gameplay on a subset of boards using the in-trainer model for both roles.
    Returns raw per-board records + think token stats.
    """
    if not boards:
        return {"per_board": [], "think_tokens": []}

    # same model for spymaster + guesser (avoids loading second model)
    base_gen = _InTrainerGenerator(model, tokenizer, device)
    spymaster = base_gen
    guesser = base_gen

    # Optional directness embedder; keep it on CPU to avoid OOM during training
    use_embed = bool(cfg.get("constraints", {}).get("enable_directness_check", True))
    embedder = None
    if use_embed:
        try:
            embedder = Embedder(cfg["models"]["embedding_model_id"], device="cpu")
        except Exception:
            embedder = None

    # eval loop (batched by inference.batch_size)
    bs = int(cfg.get("inference", {}).get("batch_size", 1))
    bs = max(1, bs)

    per_board: List[Dict[str, Any]] = []
    think_tokens: List[int] = []

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
    except Exception:
        pass

    try:
        for start in range(0, len(boards), bs):
            batch = boards[start : start + bs]
            # n_candidates=1 for eval
            bests, metas = run_turns_batched(
                batch,
                spymaster,
                guesser,
                embedder,
                cfg,
                n_candidates=1,
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
                            "valid": bool(best.valid),
                            "rejected_total": int(meta["rejected_total"]),
                            "rejection_counts": meta["rejection_counts"],
                        },
                    }
                )
                think_tokens.append(_count_think_tokens(tokenizer, best.raw_spymaster_text or ""))
    finally:
        try:
            if was_training:
                model.train()
        except Exception:
            pass

    return {"per_board": per_board, "think_tokens": think_tokens}


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
    """
    Best-effort Wikitext-2 perplexity on a small sample.
    If datasets loading fails, caller should catch.
    """
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
                # chunk into blocks
                for start in range(0, int(ids.shape[0]), int(block_size)):
                    if blocks_done >= int(max_blocks):
                        break
                    chunk = ids[start : start + int(block_size)]
                    if chunk.numel() < 2:
                        continue
                    inp = chunk.unsqueeze(0).to(device)
                    out = model(input_ids=inp, labels=inp)
                    # HF returns mean loss over tokens; convert to sum nll
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


def plot_epoch_history(history_path: Path, plots_dir: Path) -> None:
    """
    Simple plotting utility: makes line plots for a few common metrics.
    Safe to call repeatedly; overwrites PNGs.
    """
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
        "assassin_rate",
        "directness_mean",
        "wikitext2_ppl",
        "codenames_spymaster_think_tokens_mean",
        "codenames_spymaster_think_tokens_p90",
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


# -------------------------
# Callback
# -------------------------

class EpochEvalCallback(TrainerCallback):
    """
    End-of-epoch eval callback that does not assume `trainer` is passed in kwargs.

    Writes (rank0):
      - <out_dir>/epoch_eval_history.jsonl
      - <out_dir>/epoch_eval_plots/*.png
    """

    def __init__(self, cfg: Dict[str, Any], out_dir: str | Path):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.history_path = self.out_dir / "epoch_eval_history.jsonl"
        self.plots_dir = self.out_dir / "epoch_eval_plots"
        self._last_epoch_logged: Optional[int] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_f = getattr(state, "epoch", None)
        epoch_i = int(epoch_f) if epoch_f is not None else 0
        if self._last_epoch_logged is not None and epoch_i == self._last_epoch_logged:
            return control
        self._last_epoch_logged = epoch_i

        rank, world = _dist_rank_world()

        trainer = kwargs.get("trainer", None)
        model, tok, device = _resolve_model_tokenizer_device(trainer, kwargs)
        if model is None or tok is None:
            if rank == 0:
                print("[epoch_eval] Could not resolve model/tokenizer from callback kwargs; skipping.")
            return control

        ecfg = (self.cfg.get("epoch_eval", {}) or {})
        tcfg = (self.cfg.get("training", {}) or {})

        metrics: Dict[str, Any] = {}
        t0 = time.time()

        # ---- Codenames subset ----
        try:
            n_boards = int(ecfg.get("codenames_n_boards", 0))
            if n_boards > 0:
                seed = int(tcfg.get("seed", 0))
                subset = sample_codenames_eval_boards(self.cfg, n_boards=n_boards, seed=seed + epoch_i)

                subset_local = _shard_list(subset, rank, world)
                raw = eval_codenames_subset_raw(self.cfg, model, tok, device, subset_local)

                gathered_records = _all_gather_objects(raw.get("per_board", []))
                per_board: List[Dict[str, Any]] = []
                for part in gathered_records:
                    per_board.extend(part or [])

                gathered_think = _all_gather_objects([int(x) for x in raw.get("think_tokens", [])])
                think_tokens: List[int] = []
                for part in gathered_think:
                    think_tokens.extend([int(x) for x in (part or [])])

                if rank == 0 and per_board:
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

                    metrics["codenames_spymaster_think_tokens_mean"] = float(_mean([float(x) for x in think_tokens]))
                    metrics["codenames_spymaster_think_tokens_p90"] = float(_p90([float(x) for x in think_tokens]))
        except Exception as e:
            if rank == 0:
                print(f"[epoch_eval] Codenames subset eval failed: {type(e).__name__}: {e}")

        _barrier()

        # ---- Wikitext-2 ppl (best-effort) ----
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
                total_nll = _all_reduce_sum_float(float(raw["total_nll"]), device)
                total_tokens = _all_reduce_sum_int(int(raw["total_tokens"]), device)
                blocks_done = _all_reduce_sum_int(int(raw["blocks_done"]), device)
                ppl = math.exp(total_nll / max(1, total_tokens))

                if rank == 0:
                    metrics["wikitext2_blocks"] = int(blocks_done)
                    metrics["wikitext2_tokens"] = int(total_tokens)
                    metrics["wikitext2_ppl"] = float(ppl)
        except Exception as e:
            if rank == 0:
                print(f"[epoch_eval] WikiText-2 eval failed (best-effort): {type(e).__name__}: {e}")

        _barrier()

        # ---- write + plot (rank0) ----
        metrics["epoch"] = int(epoch_i)
        metrics["global_step"] = int(getattr(state, "global_step", 0))
        metrics["epoch_eval_seconds"] = float(time.time() - t0)

        if rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            # If trainer exists, also log into trainer
            tr = trainer
            if tr is not None:
                try:
                    tr.log(metrics)
                except Exception:
                    pass

            try:
                plot_epoch_history(self.history_path, self.plots_dir)
            except Exception as e:
                print(f"[epoch_eval] plot_epoch_history failed: {type(e).__name__}: {e}")

            print(f"[epoch_eval] wrote epoch {epoch_i} metrics -> {self.history_path}")

        _barrier()
        return control