from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

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


def _encode_sft_example(
    prompt: str,
    completion: str,
    tok: Any,
    *,
    max_len: int,
) -> Dict[str, Any] | None:
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
        return None

    labels = ([-100] * len(prompt_ids)) + list(completion_ids)
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class OnePositivePerBoardPerEpochDataset(TorchDataset):
    """
    Holds a pool of encoded positive examples for each board and exposes exactly one
    sampled example per board for the current epoch.

    The epoch-specific sample is deterministic given:
      - training.sft_sampling_seed
      - epoch index
      - board_id
    """

    def __init__(
        self,
        board_to_examples: Dict[str, List[Dict[str, Any]]],
        *,
        board_ids: List[str],
        seed: int,
    ):
        self.board_to_examples = board_to_examples
        self.board_ids = list(board_ids)
        self.seed = int(seed)
        self.epoch = 0
        self._epoch_examples: List[Dict[str, Any]] = []
        self.set_epoch(0)

    def _choice_index(self, board_id: str, n: int, epoch: int) -> int:
        blob = f"{self.seed}:{epoch}:{board_id}".encode("utf-8")
        h = hashlib.sha256(blob).digest()
        return int.from_bytes(h[:8], "big") % max(1, n)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        picked: List[Dict[str, Any]] = []
        for board_id in self.board_ids:
            pool = self.board_to_examples[board_id]
            idx = self._choice_index(board_id, len(pool), self.epoch)
            picked.append(pool[idx])
        self._epoch_examples = picked

    def __len__(self) -> int:
        return len(self._epoch_examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self._epoch_examples[idx]
        return {
            "input_ids": list(ex["input_ids"]),
            "attention_mask": list(ex["attention_mask"]),
            "labels": list(ex["labels"]),
        }


class ResampleOnePositivePerBoardPerEpochCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        train_dataloader = kwargs.get("train_dataloader", None)
        ds = getattr(train_dataloader, "dataset", None) if train_dataloader is not None else None
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(int(state.epoch or 0))
            print(
                f"[train_sft] resampled one-positive-per-board dataset for epoch={int(state.epoch or 0)}"
            )
        return control


def _resolve_sft_source_path(cfg: Dict[str, Any]) -> str:
    tcfg = cfg.get("training", {}) or {}
    source = str(tcfg.get("sft_source", "turns")).strip().lower()
    if source == "turns":
        return str(cfg["paths"]["turns_path"])
    return str(cfg["paths"]["turns_raw_path"])


def _resolve_sft_completion_field(cfg: Dict[str, Any]) -> str:
    tcfg = cfg.get("training", {}) or {}
    return str(tcfg.get("sft_completion_field", "completion"))


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


def _stable_board_hash(seed: int, board_id: str) -> int:
    blob = f"{int(seed)}:{board_id}".encode("utf-8")
    h = hashlib.sha256(blob).digest()
    return int.from_bytes(h[:8], "big")


def _select_board_rows(
    rows: List[Dict[str, Any]],
    *,
    max_boards: int | None,
    seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if max_boards is None:
        return list(rows), {
            "board_rows_available": int(len(rows)),
            "board_rows_selected": int(len(rows)),
            "board_cap_applied": False,
        }

    cap = max(0, int(max_boards))
    if cap >= len(rows):
        return list(rows), {
            "board_rows_available": int(len(rows)),
            "board_rows_selected": int(len(rows)),
            "board_cap_applied": False,
            "board_cap": int(cap),
        }

    ranked: List[tuple[int, int, str]] = []
    for idx, row in enumerate(rows):
        board_id = str(row.get("board_id") or row.get("example_id") or f"row_{idx:08d}")
        ranked.append((_stable_board_hash(seed, board_id), idx, board_id))

    ranked.sort()
    chosen_idxs = {idx for _, idx, _ in ranked[:cap]}
    selected = [row for idx, row in enumerate(rows) if idx in chosen_idxs]

    return selected, {
        "board_rows_available": int(len(rows)),
        "board_rows_selected": int(len(selected)),
        "board_cap_applied": True,
        "board_cap": int(cap),
    }


def _require_turns_sft_pool_row(parent: Dict[str, Any]) -> List[Dict[str, Any]]:
    pool = parent.get("sft_pool")
    if not isinstance(pool, list) or not pool:
        raise RuntimeError(
            "training.sft_source='turns' requires canonical board-pool rows with non-empty sft_pool. "
            f"Bad board_id={parent.get('board_id')!r}"
        )
    return [dict(c) for c in pool]


def _resolve_default_sft_candidate(parent: Dict[str, Any]) -> Dict[str, Any]:
    pool = _require_turns_sft_pool_row(parent)
    idx = int(parent.get("sft_default_idx", 0) or 0)
    if idx < 0 or idx >= len(pool):
        raise RuntimeError(
            "Invalid sft_default_idx in turns row. "
            f"board_id={parent.get('board_id')!r} sft_default_idx={idx} pool_size={len(pool)}"
        )
    return pool[idx]


def _build_sft_dataset(
    cfg: Dict[str, Any],
    tok: Any,
) -> tuple[TorchDataset | HFDataset, Dict[str, Any]]:
    tcfg = cfg.get("training", {}) or {}
    source_kind = str(tcfg.get("sft_source", "turns")).strip().lower()
    source_path = _resolve_sft_source_path(cfg)
    completion_field = _resolve_sft_completion_field(cfg)
    rows = read_jsonl(source_path)

    max_len = int(tcfg.get("sft_max_length", tcfg.get("max_seq_len", 4096)))
    max_examples = tcfg.get("sft_max_examples", None)
    sampling_mode = str(tcfg.get("sft_sampling_mode", "fixed")).strip().lower()
    sampling_seed = int(tcfg.get("sft_sampling_seed", tcfg.get("seed", 0)))

    selected_rows, selection_stats = _select_board_rows(
        rows,
        max_boards=(int(max_examples) if max_examples is not None else None),
        seed=sampling_seed,
    )

    if source_kind == "turns_raw":
        print(
            "[train_sft] WARNING: training.sft_source=turns_raw is treated as a dev-only fallback. "
            "The intended path is training.sft_source=turns with canonical turns.jsonl sft_pool rows."
        )
        if sampling_mode != "fixed":
            raise RuntimeError(
                "training.sft_source='turns_raw' is only supported with training.sft_sampling_mode='fixed'. "
                "Use canonical training.sft_source='turns' for pool-based sampling."
            )

        examples: List[Dict[str, Any]] = []
        dropped_missing = 0
        dropped_overlong = 0

        for r in selected_rows:
            prompt = r.get("prompt")
            completion = _resolve_completion_text(r, completion_field)

            if not isinstance(prompt, str) or not isinstance(completion, str) or not prompt or not completion:
                dropped_missing += 1
                continue

            ex = _encode_sft_example(prompt, completion, tok, max_len=max_len)
            if ex is None:
                dropped_overlong += 1
                continue

            examples.append(ex)

        if not examples:
            raise RuntimeError(
                f"No SFT examples survived prompt/completion and length checks. source_path={source_path}"
            )

        stats = {
            "source_path": source_path,
            "source_kind": source_kind,
            "completion_field": completion_field,
            "sampling_mode": sampling_mode,
            "source_rows": int(len(rows)),
            **selection_stats,
            "semantic_filtering_applied_in_train_sft": False,
            "selected_rows_before_length": int(len(selected_rows)),
            "dropped_missing_prompt_or_completion": int(dropped_missing),
            "dropped_overlong": int(dropped_overlong),
            "train_examples": int(len(examples)),
            "max_length": int(max_len),
        }
        return HFDataset.from_list(examples), stats

    if source_kind != "turns":
        raise RuntimeError(
            f"Unsupported training.sft_source={source_kind!r}. Expected 'turns' or 'turns_raw'."
        )

    if sampling_mode == "one_per_board_per_epoch":
        board_to_examples: Dict[str, List[Dict[str, Any]]] = {}
        board_ids_in_order: List[str] = []

        dropped_missing = 0
        dropped_overlong = 0
        eligible_candidates_before_dedupe = 0
        eligible_candidates_after_dedupe = 0
        dropped_empty_pool_rows = 0

        for parent in selected_rows:
            prompt = parent.get("prompt")
            board_id = str(parent.get("board_id") or parent.get("example_id") or "")
            if not isinstance(prompt, str) or not prompt or not board_id:
                dropped_missing += 1
                continue

            try:
                pool_rows = _require_turns_sft_pool_row(parent)
            except RuntimeError:
                dropped_empty_pool_rows += 1
                continue

            seen_texts = set()
            encoded_pool: List[Dict[str, Any]] = []
            for cand in pool_rows:
                completion = _resolve_completion_text(cand, completion_field)
                if not isinstance(completion, str) or not completion:
                    dropped_missing += 1
                    continue

                eligible_candidates_before_dedupe += 1
                dedupe_key = completion.strip()
                if dedupe_key in seen_texts:
                    continue
                seen_texts.add(dedupe_key)

                ex = _encode_sft_example(prompt, completion, tok, max_len=max_len)
                if ex is None:
                    dropped_overlong += 1
                    continue

                encoded_pool.append(ex)
                eligible_candidates_after_dedupe += 1

            if not encoded_pool:
                continue

            if board_id not in board_to_examples:
                board_ids_in_order.append(board_id)
                board_to_examples[board_id] = []
            board_to_examples[board_id].extend(encoded_pool)

        if not board_to_examples:
            raise RuntimeError(
                f"No SFT examples survived prompt/completion and length checks. source_path={source_path}"
            )

        ds = OnePositivePerBoardPerEpochDataset(
            board_to_examples,
            board_ids=board_ids_in_order,
            seed=sampling_seed,
        )

        pool_sizes = [len(board_to_examples[bid]) for bid in board_ids_in_order]
        stats = {
            "source_path": source_path,
            "source_kind": source_kind,
            "completion_field": completion_field,
            "sampling_mode": sampling_mode,
            "source_rows": int(len(rows)),
            **selection_stats,
            "semantic_filtering_applied_in_train_sft": False,
            "boards_with_any_pool": int(len(board_to_examples)),
            "boards_selected_per_epoch": int(len(board_ids_in_order)),
            "eligible_candidates_before_dedupe": int(eligible_candidates_before_dedupe),
            "eligible_candidates_after_dedupe": int(eligible_candidates_after_dedupe),
            "avg_pool_size_per_selected_board": float(sum(pool_sizes) / max(1, len(pool_sizes))),
            "max_pool_size_per_selected_board": int(max(pool_sizes) if pool_sizes else 0),
            "dropped_empty_pool_rows": int(dropped_empty_pool_rows),
            "dropped_missing_prompt_or_completion": int(dropped_missing),
            "dropped_overlong": int(dropped_overlong),
            "train_examples": int(len(board_ids_in_order)),
            "max_length": int(max_len),
        }
        return ds, stats

    if sampling_mode != "fixed":
        raise RuntimeError(
            f"Unsupported training.sft_sampling_mode={sampling_mode!r}. Expected 'fixed' or 'one_per_board_per_epoch'."
        )

    examples: List[Dict[str, Any]] = []
    dropped_missing = 0
    dropped_overlong = 0
    dropped_empty_pool_rows = 0

    for parent in selected_rows:
        prompt = parent.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            dropped_missing += 1
            continue

        try:
            default_cand = _resolve_default_sft_candidate(parent)
        except RuntimeError:
            dropped_empty_pool_rows += 1
            continue

        completion = _resolve_completion_text(default_cand, completion_field)
        if not isinstance(completion, str) or not completion:
            dropped_missing += 1
            continue

        ex = _encode_sft_example(prompt, completion, tok, max_len=max_len)
        if ex is None:
            dropped_overlong += 1
            continue

        examples.append(ex)

    if not examples:
        raise RuntimeError(
            f"No SFT examples survived prompt/completion and length checks. source_path={source_path}"
        )

    stats = {
        "source_path": source_path,
        "source_kind": source_kind,
        "completion_field": completion_field,
        "sampling_mode": sampling_mode,
        "source_rows": int(len(rows)),
        **selection_stats,
        "semantic_filtering_applied_in_train_sft": False,
        "selected_rows_before_length": int(len(selected_rows)),
        "dropped_empty_pool_rows": int(dropped_empty_pool_rows),
        "dropped_missing_prompt_or_completion": int(dropped_missing),
        "dropped_overlong": int(dropped_overlong),
        "train_examples": int(len(examples)),
        "max_length": int(max_len),
    }

    return HFDataset.from_list(examples), stats


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
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "linear")),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.05)),
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

    if str(tcfg.get("sft_sampling_mode", "fixed")).strip().lower() == "one_per_board_per_epoch":
        trainer.add_callback(ResampleOnePositivePerBoardPerEpochCallback())

    if not args.no_epoch_eval:
        trainer.add_callback(EpochEvalCallback(cfg, out_dir))

    stats_path = Path(out_dir) / "sft_train_stats.json"
    stats_path.write_text(json.dumps(sft_stats, indent=2), encoding="utf-8")
    print(f"[train_sft] wrote dataset stats -> {stats_path}")
    print(f"[train_sft] completion_field={sft_stats['completion_field']}")
    print(f"[train_sft] train_examples={sft_stats['train_examples']}")
    print(f"[train_sft] sampling_mode={sft_stats.get('sampling_mode', 'fixed')}")

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
