from __future__ import annotations

import contextlib
import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from .model_wrappers import apply_chat_template, make_text_generator


def dist_available() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def dist_rank_world() -> Tuple[int, int]:
    if not dist_available():
        return (0, 1)
    import torch.distributed as dist
    return (dist.get_rank(), dist.get_world_size())


def barrier() -> None:
    if not dist_available():
        return
    import torch.distributed as dist
    dist.barrier()


def all_gather_objects(obj: Any) -> List[Any]:
    if not dist_available():
        return [obj]
    import torch.distributed as dist
    world = dist.get_world_size()
    out: List[Any] = [None] * world
    dist.all_gather_object(out, obj)
    return out


def all_reduce_sum_float(x: float, device: torch.device) -> float:
    if not dist_available():
        return float(x)
    import torch.distributed as dist
    t = torch.tensor([float(x)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def all_reduce_sum_int(x: int, device: torch.device) -> int:
    if not dist_available():
        return int(x)
    import torch.distributed as dist
    t = torch.tensor([int(x)], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def shard_list(items: List[Any], rank: int, world: int) -> List[Any]:
    return [it for i, it in enumerate(items) if (i % world) == rank]


def infer_device_from_model(model: Any) -> torch.device:
    try:
        p = next(model.parameters())
        return p.device
    except Exception:
        pass
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        return torch.device("cuda", lr)
    return torch.device("cpu")


def resolve_model_tokenizer_device(
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
            dev = infer_device_from_model(m)
        return m, tok, dev

    m = kwargs.get("model", None)
    tok = kwargs.get("tokenizer", None) or kwargs.get("processing_class", None)
    dev = infer_device_from_model(m) if m is not None else infer_device_from_model(object())
    return m, tok, dev


@contextlib.contextmanager
def maybe_disable_adapter(model: Any, disable: bool):
    if not disable:
        yield
        return

    cm = getattr(model, "disable_adapter", None)
    if callable(cm):
        with cm():
            yield
        return

    yield


class InTrainerGenerator:
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
            with maybe_disable_adapter(self.model, self.disable_adapter):
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


def build_external_guesser(cfg: Dict[str, Any]):
    sp_id = cfg["models"]["spymaster_model_id"]
    g_id = cfg["models"]["guesser_model_id"]
    if g_id == sp_id:
        return None

    gcfg = copy.deepcopy(cfg)
    inf = gcfg.get("inference", {}) or {}
    inf["num_processes"] = 1
    if inf.get("backend") == "vllm":
        vcfg = inf.get("vllm", {}) or {}
        vcfg["tensor_parallel_size"] = 1
        inf["vllm"] = vcfg
    gcfg["inference"] = inf
    return make_text_generator(g_id, gcfg)


def build_codenames_record(board: Dict[str, Any], best: Any, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "board_id": board["board_id"],
        "reward": float(best.reward),
        "clue": best.clue,
        "num": int(best.num),
        "guess_words": best.guess_words,
        "stats": {**best.stats, "directness": float(best.directness)},
        "clue_meta": {
            "valid": bool(best.valid),
            "parse_valid": bool(best.parse_valid),
            "rejected_total": int(meta["rejected_total"]),
            "rejection_counts": meta["rejection_counts"],
        },
    }
