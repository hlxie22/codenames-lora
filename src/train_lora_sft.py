# src/train_lora_sft.py
from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers.trainer_utils import get_last_checkpoint

from .utils import (
    load_yaml,
    read_jsonl,
    ensure_dir,
    save_config_snapshot,
    save_run_meta,
    set_global_seed,
)


class SFTDataset(Dataset):
    """
    Dataset backed by pre-tokenized examples:
      item = {"input_ids": List[int], "labels": List[int]}
    """

    def __init__(self, tokenized: List[Dict[str, Any]]):
        self.items = tokenized

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        attn = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


@dataclass
class PadCollator:
    tokenizer: Any

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attn = [b["attention_mask"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def _load_or_build_tokcache(
    *,
    cache_path: Path,
    records: List[Dict[str, Any]],
    tokenizer,
    model_id: str,
    max_len: int,
) -> List[Dict[str, Any]]:
    """
    Cache format:
      {"model_id": str, "max_len": int, "tokenized": [{"input_ids": [...], "labels": [...]}, ...]}
    """
    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        if (
            isinstance(obj, dict)
            and obj.get("model_id") == model_id
            and int(obj.get("max_len", -1)) == int(max_len)
            and isinstance(obj.get("tokenized"), list)
        ):
            print(f"Loaded tokenization cache -> {cache_path}")
            return obj["tokenized"]

    tokenized: List[Dict[str, Any]] = []
    eos = tokenizer.eos_token_id

    for r in records:
        p = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tokenizer(r["completion"], add_special_tokens=False)["input_ids"]

        input_ids = p + c + [eos]
        labels = [-100] * len(p) + c + [eos]

        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

        tokenized.append({"input_ids": input_ids, "labels": labels})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": model_id, "max_len": int(max_len), "tokenized": tokenized}, cache_path)
    print(f"Saved tokenization cache -> {cache_path}")
    return tokenized


def _disable_kv_cache(model: Any) -> None:
    """
    Disable KV cache during training to reduce VRAM usage.
    For decoder-only LMs, use_cache is useful for generation, not teacher-forced training.
    """
    try:
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    except Exception:
        pass

    for attr in ("base_model", "model", "module"):
        try:
            m = getattr(model, attr, None)
            if m is not None and hasattr(m, "config") and hasattr(m.config, "use_cache"):
                m.config.use_cache = False
        except Exception:
            continue


def _normalize_attn_impl(raw: Any) -> str | None:
    """
    Transformers expects one of: eager, sdpa, flex_attention, flash_attention_2, flash_attention_3.
    Back-compat: map old kernels hub identifiers to the supported string.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    low = s.lower()

    # Back-compat mapping (your old setting)
    if low in {"kernels-community/flash-attn2", "flash-attn2", "flash_attn2", "flashattn2"}:
        return "flash_attention_2"
    if low in {"kernels-community/flash-attn3", "flash-attn3", "flash_attn3", "flashattn3"}:
        return "flash_attention_3"

    if low in {"auto", "none", "null"}:
        return None

    # Pass-through for valid values
    allowed = {"eager", "sdpa", "flex_attention", "flash_attention_2", "flash_attention_3"}
    if s in allowed:
        return s
    if low in allowed:
        # preserve canonical lowercase form
        return low

    print(f"[warn] Unknown training.attn_implementation={s!r}; omitting attn_implementation.")
    return None


def _enable_gradient_checkpointing(model: Any, gc_kwargs: Dict[str, Any] | None) -> None:
    """
    Enable GC with compatibility across Transformers versions.
    Also ensures inputs require grads (important for PEFT+GC on some models).
    """
    try:
        if gc_kwargs is not None:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        else:
            model.gradient_checkpointing_enable()
    except TypeError:
        # older signature: no kwargs
        model.gradient_checkpointing_enable()

    # PEFT + checkpointing often needs this for a grad path
    try:
        model.enable_input_require_grads()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["training"]
    set_global_seed(int(tcfg.get("seed", 0)))

    # -------- Gradient checkpointing config --------
    gc_enabled = bool(tcfg.get("gradient_checkpointing", False))
    gc_kwargs = None
    if gc_enabled:
        gc_kwargs = {"use_reentrant": bool(tcfg.get("gradient_checkpointing_use_reentrant", False))}

    # Load data
    records = read_jsonl(cfg["paths"]["sft_turns_path"])
    if not records:
        raise RuntimeError("No SFT records found. Did generate_sft_data produce an empty filtered set?")

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model

    model_id = cfg["models"]["spymaster_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dtype
    torch_dtype = None
    if bool(tcfg.get("bf16", False)) and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif bool(tcfg.get("fp16", False)) and torch.cuda.is_available():
        torch_dtype = torch.float16

    # Attention implementation (fixes your earlier crash)
    attn_impl = _normalize_attn_impl(tcfg.get("attn_implementation", "sdpa"))

    # Build model
    model_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    # Disable KV cache for training (and for gradient checkpointing)
    _disable_kv_cache(model)

    # Enable gradient checkpointing on the base model (before PEFT wrap)
    if gc_enabled:
        _enable_gradient_checkpointing(model, gc_kwargs)

    # LoRA
    lora_cfg = LoraConfig(
        r=int(tcfg["r"]),
        lora_alpha=int(tcfg["alpha"]),
        lora_dropout=float(tcfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=tcfg.get("target_modules", None),
    )
    model = get_peft_model(model, lora_cfg)

    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)

    # Disable KV cache again after wrapping + re-enable GC (some wrappers/models need it)
    _disable_kv_cache(model)
    if gc_enabled:
        _enable_gradient_checkpointing(model, gc_kwargs)

    out_dir = ensure_dir(tcfg["output_adapter_dir"])

    # -------------------------
    # Pre-tokenize once + cache
    # -------------------------
    max_len = int(tcfg["max_seq_len"])
    sft_path = Path(cfg["paths"]["sft_turns_path"])
    cache_path = sft_path.with_suffix(sft_path.suffix + f".tokcache.maxlen{max_len}.pt")

    tokenized = _load_or_build_tokcache(
        cache_path=cache_path,
        records=records,
        tokenizer=tokenizer,
        model_id=model_id,
        max_len=max_len,
    )

    ds = SFTDataset(tokenized)
    collator = PadCollator(tokenizer)

    # TrainingArguments: be compatible across versions wrt gradient_checkpointing_kwargs
    ta_kwargs: Dict[str, Any] = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg["batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_accum"]),
        num_train_epochs=float(tcfg["epochs"]),
        learning_rate=float(tcfg["lr"]),
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        report_to=[],
        bf16=bool(tcfg.get("bf16", False)),
        fp16=bool(tcfg.get("fp16", False)),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=gc_enabled,
    )

    # Only pass gradient_checkpointing_kwargs if TrainingArguments supports it
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        if gc_enabled and gc_kwargs is not None and "gradient_checkpointing_kwargs" in sig.parameters:
            ta_kwargs["gradient_checkpointing_kwargs"] = gc_kwargs
    except Exception:
        # If signature introspection fails, just don't pass kwargs
        pass

    args_tr = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    last_ckpt = get_last_checkpoint(str(out_dir))
    trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)

    # Save adapter
    if trainer.is_world_process_zero():
        m = trainer.model
        if hasattr(m, "module"):  # DDP wrap
            m = m.module
        m.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        # Save config + run meta
        save_config_snapshot(cfg, out_dir)
        save_run_meta(out_dir, command=["python", "-m", "src.train_lora_sft", "--config", args.config])

    print(f"Saved LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()