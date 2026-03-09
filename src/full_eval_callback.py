# src/full_eval_callback.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers import TrainerCallback
except Exception:
    TrainerCallback = object  # type: ignore[misc]

from .metrics import aggregate
from .model_wrappers import apply_chat_template, Embedder
from .rollout import run_turns_batched
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


def _shard_list(items: List[Any], rank: int, world: int) -> List[Any]:
    return [it for i, it in enumerate(items) if (i % world) == rank]


def _unwrap_trainer_model(trainer: Any) -> Any:
    m = getattr(trainer, "model", None)
    if m is None:
        return None

    acc = getattr(trainer, "accelerator", None)
    if acc is not None:
        try:
            return acc.unwrap_model(m)
        except Exception:
            pass

    if hasattr(m, "module"):
        try:
            return m.module
        except Exception:
            pass

    return m


def _get_tokenizer(trainer: Any) -> Any:
    tok = getattr(trainer, "processing_class", None)
    if tok is not None:
        return tok
    tok = getattr(trainer, "tokenizer", None)
    if tok is not None:
        return tok
    return None


def _trainer_device(trainer: Any) -> torch.device:
    args = getattr(trainer, "args", None)
    if args is not None and getattr(args, "device", None) is not None:
        return args.device
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        return torch.device("cuda", lr)
    return torch.device("cpu")


def _infer_device_from_model(model: Any) -> torch.device:
    try:
        p = next(model.parameters())
        return p.device
    except Exception:
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            return torch.device("cuda", lr)
        return torch.device("cpu")


class _InTrainerGenerator:
    """
    Minimal TextGenerator-like adapter around the *current trainer model + tokenizer*.

    Works with rollout.run_turns_batched via generate() / generate_batch() / format_chat().
    Avoids loading a second model (which is the usual OOM trap during training).
    """

    def __init__(self, model: Any, tokenizer: Any, device: torch.device, *, enable_thinking_default: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_id = getattr(getattr(model, "config", None), "_name_or_path", "trainer_model")
        self.enable_thinking_default = bool(enable_thinking_default)

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
            self.generate(p, gen_cfg, seed=s, use_chat_template=False, enable_thinking=self.enable_thinking_default)
            for p, s in zip(prompts, seeds)
        ]


class FullCodenamesEvalCallback(TrainerCallback):
    """
    Full eval (boards_eval gameplay) at epoch end.

    Outputs:
      <out_dir>/full_eval/epoch_XXX_step_YYY/per_board.jsonl
      <out_dir>/full_eval/epoch_XXX_step_YYY/metrics.json
    """

    def __init__(self, cfg: Dict[str, Any], out_dir: str | Path, *, every_epochs: int = 1, batch_size: int = 1):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.every_epochs = max(1, int(every_epochs))
        self.batch_size = max(1, int(batch_size))
        self._last_epoch_logged: Optional[int] = None

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_f = state.epoch
        epoch_i = int(epoch_f) if epoch_f is not None else 0

        # avoid double fire
        if self._last_epoch_logged is not None and epoch_i == self._last_epoch_logged:
            return control
        self._last_epoch_logged = epoch_i

        if (epoch_i % self.every_epochs) != 0:
            return control

        rank, world = _dist_rank_world()

        # ---- resolve trainer/model/tokenizer/device ----
        trainer = kwargs.get("trainer", None)

        if trainer is not None:
            model = _unwrap_trainer_model(trainer)
            tok = _get_tokenizer(trainer)
            device = _trainer_device(trainer)
        else:
            # Fallback: many Trainer/TRL versions do NOT pass `trainer` in kwargs.
            model = kwargs.get("model", None)
            tok = kwargs.get("tokenizer", None) or kwargs.get("processing_class", None)
            device = _infer_device_from_model(model) if model is not None else _infer_device_from_model(object())

        if model is None or tok is None:
            if rank == 0:
                print("[full_eval] Could not resolve model/tokenizer from callback kwargs; skipping.")
            return control

        # Load boards (small enough) + shard across ranks
        boards_eval_path = self.cfg["paths"]["boards_eval_path"]
        boards_all = read_jsonl(boards_eval_path)
        boards_local = _shard_list(boards_all, rank, world)

        # Generators (same model for spymaster+guesser to avoid loading a second model during training)
        base_gen = _InTrainerGenerator(model, tok, device, enable_thinking_default=True)

        spymaster = base_gen
        guesser = base_gen

        # Optional: directness embedder, but put it on CPU to avoid GPU OOM during training.
        use_embed = bool(self.cfg.get("constraints", {}).get("enable_directness_check", True))
        embedder = None
        if use_embed:
            try:
                embedder = Embedder(self.cfg["models"]["embedding_model_id"], device="cpu")
            except Exception:
                embedder = None

        # eval loop
        was_training = bool(getattr(model, "training", False))
        try:
            model.eval()
        except Exception:
            pass

        t0 = time.time()
        per_board_local: List[Dict[str, Any]] = []

        # Always eval with n_candidates=1 (like src.eval)
        n_candidates = 1

        for start in range(0, len(boards_local), self.batch_size):
            batch = boards_local[start : start + self.batch_size]

            bests, metas = run_turns_batched(
                batch,
                spymaster,
                guesser,
                embedder,
                self.cfg,
                n_candidates=n_candidates,
            )

            for b, best, meta in zip(batch, bests, metas):
                rec = {
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
                per_board_local.append(rec)

        # gather all per-board to rank0
        gathered = _all_gather_objects(per_board_local)
        per_board_all: List[Dict[str, Any]] = []
        for part in gathered:
            per_board_all.extend(part or [])

        # restore mode
        try:
            if was_training:
                model.train()
        except Exception:
            pass

        _barrier()

        if rank == 0:
            run_dir = self.out_dir / "full_eval" / f"epoch_{epoch_i:03d}_step_{int(getattr(state, 'global_step', 0)):06d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            per_path = run_dir / "per_board.jsonl"
            write_jsonl(per_path, per_board_all)

            metrics = aggregate(per_board_all)
            metrics["epoch"] = int(epoch_i)
            metrics["global_step"] = int(getattr(state, "global_step", 0))
            metrics["eval_seconds"] = float(time.time() - t0)
            metrics["n_boards_eval"] = int(len(per_board_all))

            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            # best-effort Trainer logging (only if we actually have trainer)
            if trainer is not None:
                try:
                    trainer.log({f"full_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
                except Exception:
                    pass

            print(f"[full_eval] wrote {len(per_board_all)} boards -> {per_path}")
            print(f"[full_eval] metrics -> {run_dir / 'metrics.json'}")

        _barrier()
        return control