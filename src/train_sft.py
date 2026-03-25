from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from .build_dpo_pairs import _passes_chosen_filter
from .epoch_eval import EpochEvalCallback
from .think_utils import split_think_and_rest
from .utils import (
    ensure_dir,
    load_yaml,
    read_jsonl,
    save_config_snapshot,
    save_run_meta,
    set_global_seed,
)


def _setup_node_local_hf_cache() -> None:
    base_tmp = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    job_id = os.environ.get("SLURM_JOB_ID") or "nojob"

    local_hf_home = Path(base_tmp) / f"hf_{job_id}"
    local_datasets_cache = local_hf_home / "datasets"
    local_hub_cache = local_hf_home / "hub"

    local_datasets_cache.mkdir(parents=True, exist_ok=True)
    local_hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(local_hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(local_datasets_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(local_hub_cache)


def _world_size() -> int:
    for k in ("WORLD_SIZE", "SLURM_NTASKS"):
        v = os.environ.get(k)
        if v:
            with contextlib.suppress(Exception):
                return max(1, int(v))
    return 1


def _should_force_nonreentrant(cfg: Dict[str, Any]) -> bool:
    tcfg = cfg.get("training", {}) or {}
    if not bool(tcfg.get("gradient_checkpointing", True)):
        return False
    return _world_size() > 1


def _safe_enable_input_grads(model: Any) -> None:
    with contextlib.suppress(Exception):
        model.enable_input_require_grads()


def _safe_enable_gradient_checkpointing(model: Any, *, use_reentrant: bool) -> None:
    with contextlib.suppress(Exception):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )


class CausalLMCompletionCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_inputs = self.tokenizer.pad(
            [
                {
                    "input_ids": f["input_ids"],
                    "attention_mask": f["attention_mask"],
                }
                for f in features
            ],
            padding=True,
            return_tensors="pt",
        )

        max_len = int(batch_inputs["input_ids"].shape[1])
        labels: List[List[int]] = []
        for f in features:
            lab = list(f["labels"])
            if len(lab) < max_len:
                lab = lab + ([-100] * (max_len - len(lab)))
            labels.append(lab)

        batch_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch_inputs


def _resolve_sft_source_path(cfg: Dict[str, Any]) -> str:
    tcfg = cfg.get("training", {}) or {}
    source = str(tcfg.get("sft_source", "turns_raw")).strip().lower()
    if source == "turns":
        return str(cfg["paths"]["turns_path"])
    return str(cfg["paths"]["turns_raw_path"])


def _resolve_sft_completion_field(cfg: Dict[str, Any]) -> str:
    tcfg = cfg.get("training", {}) or {}
    return str(tcfg.get("sft_completion_field", "completion"))


def _sort_sft_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]):
        stats = r.get("stats") or {}
        return (
            -float(r.get("reward", 0.0)),
            -int(stats.get("n_team", 0)),
            int(stats.get("assassin", 0)),
            str(r.get("board_id") or r.get("example_id") or ""),
        )

    return sorted(rows, key=key)


def _resolve_completion_text(row: Dict[str, Any], field: str) -> str | None:
    val = row.get(field)
    if isinstance(val, str) and val:
        text = val
    elif field != "completion":
        val = row.get("completion")
        if isinstance(val, str) and val:
            text = val
        else:
            return None
    else:
        return None

    if field == "completion":
        final = row.get("completion_final")
        raw = row.get("completion_raw") or row.get("raw_spymaster_text") or text
        reasoning_trace, stripped = split_think_and_rest(str(raw), which="first", allow_partial=True)
        if reasoning_trace and isinstance(final, str) and final:
            return final
        if reasoning_trace and stripped:
            return stripped

    return text


def _build_sft_dataset(
    cfg: Dict[str, Any],
    tok: Any,
) -> tuple[Dataset, Dict[str, Any]]:
    tcfg = cfg.get("training", {}) or {}
    source_path = _resolve_sft_source_path(cfg)
    completion_field = _resolve_sft_completion_field(cfg)
    rows = read_jsonl(source_path)

    sft_filter = dict(tcfg.get("sft_filter") or {})
    filtered = [r for r in rows if _passes_chosen_filter(r, sft_filter)]

    sorted_rows = _sort_sft_rows(filtered)

    max_examples = tcfg.get("sft_max_examples", None)
    if max_examples is not None:
        sorted_rows = sorted_rows[: int(max_examples)]

    max_len = int(tcfg.get("sft_max_length", tcfg.get("max_seq_len", 4096)))

    examples: List[Dict[str, Any]] = []
    dropped_missing = 0
    dropped_overlong = 0

    for r in sorted_rows:
        prompt = r.get("prompt")
        completion = _resolve_completion_text(r, completion_field)

        if not isinstance(prompt, str) or not isinstance(completion, str):
            dropped_missing += 1
            continue
        if not prompt or not completion:
            dropped_missing += 1
            continue

        prompt_ids = tok(prompt, add_special_tokens=False, truncation=False)["input_ids"]

        completion_text = completion
        if tok.eos_token:
            completion_text = completion_text + tok.eos_token

        completion_ids = tok(
            completion_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        input_ids = list(prompt_ids) + list(completion_ids)
        if len(input_ids) > max_len:
            dropped_overlong += 1
            continue

        labels = ([-100] * len(prompt_ids)) + list(completion_ids)
        attention_mask = [1] * len(input_ids)

        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    if not examples:
        raise RuntimeError(
            f"No SFT examples survived filtering. source_path={source_path} filter={json.dumps(sft_filter)}"
        )

    stats = {
        "source_path": source_path,
        "completion_field": completion_field,
        "source_rows": int(len(rows)),
        "filtered_rows": int(len(filtered)),
        "selected_rows_before_length": int(len(sorted_rows)),
        "dropped_missing_prompt_or_completion": int(dropped_missing),
        "dropped_overlong": int(dropped_overlong),
        "train_examples": int(len(examples)),
        "max_length": int(max_len),
        "sft_filter": sft_filter,
    }

    return Dataset.from_list(examples), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-epoch-eval", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg.get("training", {}) or {}

    set_global_seed(int(tcfg.get("seed", 0)))
    _setup_node_local_hf_cache()

    model_id = cfg["models"]["spymaster_model_id"]

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds, sft_stats = _build_sft_dataset(cfg, tok)

    use_bf16 = bool(tcfg.get("bf16", True)) and torch.cuda.is_available()
    use_fp16 = bool(tcfg.get("fp16", False)) and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)

    attn_impl = tcfg.get("attn_implementation", None)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.pad_token_id = tok.pad_token_id

    out_dir = ensure_dir(
        str(tcfg.get("sft_output_adapter_dir", tcfg.get("output_adapter_dir", "outputs/sft/lora_spymaster")))
    )

    init_adapter_dir = (
        tcfg.get("sft_init_adapter_dir")
        or tcfg.get("init_adapter_dir")
        or os.environ.get("INIT_ADAPTER_DIR")
    )
    init_adapter_dir = str(init_adapter_dir) if init_adapter_dir else ""

    if init_adapter_dir and Path(init_adapter_dir).exists():
        print(f"[train_sft] Initializing from LoRA adapter: {init_adapter_dir}")
        model = PeftModel.from_pretrained(model, init_adapter_dir, is_trainable=True)
    else:
        peft_cfg = LoraConfig(
            r=int(tcfg.get("r", 16)),
            lora_alpha=int(tcfg.get("alpha", 32)),
            lora_dropout=float(tcfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=tcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        )
        model = get_peft_model(model, peft_cfg)

    gc_on = bool(tcfg.get("gradient_checkpointing", True))
    user_use_reentrant = bool(tcfg.get("gradient_checkpointing_use_reentrant", False))
    use_reentrant = user_use_reentrant
    if _should_force_nonreentrant(cfg) and user_use_reentrant:
        print(
            "[train_sft] WARNING: forcing gradient_checkpointing_use_reentrant=False "
            "for distributed safety."
        )
        use_reentrant = False

    _safe_enable_input_grads(model)
    if gc_on:
        _safe_enable_gradient_checkpointing(model, use_reentrant=use_reentrant)

    gc_kwargs: Optional[Dict[str, Any]] = {"use_reentrant": bool(use_reentrant)} if gc_on else None

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg.get("sft_batch_size", tcfg.get("batch_size", 1))),
        gradient_accumulation_steps=int(tcfg.get("sft_grad_accum", tcfg.get("grad_accum", 1))),
        num_train_epochs=float(tcfg.get("sft_epochs", 1)),
        learning_rate=float(tcfg.get("sft_lr", 5e-5)),
        bf16=use_bf16,
        fp16=use_fp16,
        deepspeed=str(tcfg.get("deepspeed_config")) if tcfg.get("deepspeed_config") else None,
        gradient_checkpointing=gc_on,
        gradient_checkpointing_kwargs=gc_kwargs,
        ddp_find_unused_parameters=False,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tok,
        data_collator=CausalLMCompletionCollator(tok),
    )

    if not args.no_epoch_eval:
        trainer.add_callback(EpochEvalCallback(cfg, out_dir))

    stats_path = Path(out_dir) / "sft_train_stats.json"
    stats_path.write_text(json.dumps(sft_stats, indent=2), encoding="utf-8")
    print(f"[train_sft] wrote dataset stats -> {stats_path}")
    print(f"[train_sft] completion_field={sft_stats['completion_field']}")
    print(f"[train_sft] train_examples={sft_stats['train_examples']}")

    last_ckpt = get_last_checkpoint(str(out_dir)) if Path(out_dir).exists() else None
    if last_ckpt:
        print(f"[train_sft] Resuming from checkpoint: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    if trainer.is_world_process_zero():
        save_config_snapshot(cfg, out_dir)
        save_run_meta(out_dir, command=["python", "-m", "src.train_sft", "--config", args.config])

    print(f"Saved SFT LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()
