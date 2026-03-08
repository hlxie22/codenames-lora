# src/train_trl_dpo.py
from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from trl import DPOConfig, DPOTrainer

from .epoch_eval import EpochEvalCallback
from .full_eval_callback import FullCodenamesEvalCallback
from .utils import ensure_dir, load_yaml, save_config_snapshot, save_run_meta, set_global_seed


def _setup_node_local_hf_cache() -> Path:
    """
    Put HF caches on node-local storage to avoid shared FS filelock/stale handle issues.
    """
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

    return local_datasets_cache


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "True", "yes", "YES")


def _world_size() -> int:
    for k in ("WORLD_SIZE", "SLURM_NTASKS"):
        v = os.environ.get(k)
        if v:
            with contextlib.suppress(Exception):
                return max(1, int(v))
    return 1


def _should_force_nonreentrant(cfg: Dict[str, Any]) -> bool:
    """
    DDP + (re-entrant) gradient checkpointing can crash with:
      "Expected to mark a variable ready only once"
    DPO is especially likely to trigger it because it runs multiple forward passes
    per step through the same modules.
    """
    tcfg = cfg.get("training", {}) or {}
    if not bool(tcfg.get("gradient_checkpointing", True)):
        return False
    # If distributed, be conservative.
    return _world_size() > 1


def _safe_enable_input_grads(model: Any) -> None:
    """
    For PEFT/LoRA + gradient checkpointing, this prevents:
      "None of the inputs have requires_grad=True. Gradients will be None"
    """
    with contextlib.suppress(Exception):
        model.enable_input_require_grads()


def _safe_enable_gradient_checkpointing(model: Any, *, use_reentrant: bool) -> None:
    """
    Some model classes require calling this explicitly (best-effort).
    """
    with contextlib.suppress(Exception):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})


def _make_trainer(
    *,
    model: Any,
    tok: Any,
    train_dataset: Any,
    dpo_args: Any,
    peft_cfg: Any,
) -> Any:
    """
    Robust init across minor TRL signature drift:
      - new API: processing_class=...
      - older API: tokenizer=...
    """
    # Try newest-style first
    try:
        return DPOTrainer(
            model=model,
            args=dpo_args,
            train_dataset=train_dataset,
            processing_class=tok,
            peft_config=peft_cfg,
        )
    except TypeError as e1:
        # Fall back to tokenizer=...
        try:
            return DPOTrainer(
                model=model,
                args=dpo_args,
                train_dataset=train_dataset,
                tokenizer=tok,
                peft_config=peft_cfg,
            )
        except TypeError:
            # Last resort: older TRL variants sometimes don't accept peft_config kw name
            # (rare, but cheap to try)
            try:
                return DPOTrainer(
                    model=model,
                    args=dpo_args,
                    train_dataset=train_dataset,
                    tokenizer=tok,
                )
            except TypeError:
                raise e1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-epoch-eval", action="store_true", help="Disable lightweight epoch eval callback.")

    # Full eval (boards_eval gameplay) between epochs
    ap.add_argument("--full-eval", action="store_true", help="Run full Codenames eval on boards_eval at epoch end.")
    ap.add_argument("--full-eval-every", type=int, default=None, help="Run full eval every N epochs (default: 1).")
    ap.add_argument(
        "--full-eval-batch-size",
        type=int,
        default=None,
        help="Override eval batch size (default: inference.batch_size).",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg.get("training", {}) or {}
    set_global_seed(int(tcfg.get("seed", 0)))

    model_id = cfg["models"]["spymaster_model_id"]

    # ----- Node-local HF caches -----
    local_datasets_cache = _setup_node_local_hf_cache()

    # ----- Dataset -----
    from datasets import load_dataset

    dpo_path = cfg["paths"]["dpo_pairs_path"]
    ds = load_dataset(
        "json",
        data_files=str(dpo_path),
        split="train",
        cache_dir=str(local_datasets_cache),
    )

    # ----- Tokenizer -----
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ----- Model -----
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

    # Important for checkpointing + trainer
    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.pad_token_id = tok.pad_token_id

    # ----- LoRA -----
    peft_cfg = LoraConfig(
        r=int(tcfg.get("r", 16)),
        lora_alpha=int(tcfg.get("alpha", 32)),
        lora_dropout=float(tcfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=tcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )

    out_dir = ensure_dir(tcfg["output_adapter_dir"])

    # ----- Gradient checkpointing safety -----
    gc_on = bool(tcfg.get("gradient_checkpointing", True))
    user_use_reentrant = bool(tcfg.get("gradient_checkpointing_use_reentrant", False))

    # Force non-reentrant in distributed to avoid DDP "marked ready twice" crashes in DPO+LoRA.
    force_nonreentrant = _should_force_nonreentrant(cfg)
    use_reentrant = user_use_reentrant
    if force_nonreentrant and user_use_reentrant:
        print(
            "[train_trl_dpo] WARNING: gradient_checkpointing_use_reentrant=True with distributed training. "
            "For DPO(+LoRA) this commonly triggers DDP 'marked ready twice' crashes. Forcing use_reentrant=False."
        )
        use_reentrant = False

    # Make sure LoRA grads exist under checkpointing.
    _safe_enable_input_grads(model)
    if gc_on:
        _safe_enable_gradient_checkpointing(model, use_reentrant=use_reentrant)

    # ----- TRL DPOConfig -----
    max_len = int(tcfg.get("trl_max_length", tcfg.get("max_seq_len", 4096)))
    trunc_mode = str(tcfg.get("trl_truncation_mode", "keep_start"))

    # Only pass checkpointing kwargs when checkpointing is enabled.
    gc_kwargs: Optional[Dict[str, Any]] = {"use_reentrant": bool(use_reentrant)} if gc_on else None

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        beta=float(tcfg.get("dpo_beta", 0.1)),
        max_length=max_len,
        truncation_mode=trunc_mode,
        per_device_train_batch_size=int(tcfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("grad_accum", 16)),
        num_train_epochs=float(tcfg.get("epochs", 1)),
        learning_rate=float(tcfg.get("lr", 2e-4)),
        bf16=use_bf16,
        fp16=use_fp16,
        deepspeed=str(tcfg.get("deepspeed_config")) if tcfg.get("deepspeed_config") else None,
        gradient_checkpointing=gc_on,
        gradient_checkpointing_kwargs=gc_kwargs,
        # With LoRA, it's common/expected that many base params are unused; disable extra DDP traversal.
        ddp_find_unused_parameters=False,
        logging_strategy="epoch",
        save_strategy="steps",
        save_steps=30,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = _make_trainer(model=model, tok=tok, train_dataset=ds, dpo_args=dpo_args, peft_cfg=peft_cfg)

    # Optional eval callbacks
    if not args.no_epoch_eval:
        trainer.add_callback(EpochEvalCallback(cfg, out_dir))

    full_eval_on = bool(args.full_eval) or _env_flag("FULL_EVAL", "0") or bool(
        cfg.get("epoch_eval", {}).get("full_eval", False)
    )
    if full_eval_on:
        every = args.full_eval_every
        if every is None:
            every = int(os.environ.get("FULL_EVAL_EVERY", str(cfg.get("epoch_eval", {}).get("full_eval_every", 1))))
        every = max(1, int(every))

        bs = args.full_eval_batch_size
        if bs is None:
            bs = int(os.environ.get("FULL_EVAL_BATCH_SIZE", str(cfg.get("inference", {}).get("batch_size", 1))))
        bs = max(1, int(bs))

        trainer.add_callback(FullCodenamesEvalCallback(cfg, out_dir, every_epochs=every, batch_size=bs))

    # Resume
    last_ckpt = get_last_checkpoint(str(out_dir)) if Path(out_dir).exists() else None
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    if trainer.is_world_process_zero():
        save_config_snapshot(cfg, out_dir)
        save_run_meta(out_dir, command=["python", "-m", "src.train_trl_dpo", "--config", args.config])

    print(f"Saved TRL DPO LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()