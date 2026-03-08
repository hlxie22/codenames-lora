# src/train_trl_dpo.py
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOConfig, DPOTrainer

from .epoch_eval import EpochEvalCallback  # optional: your existing lightweight eval callback
from .full_eval_callback import FullCodenamesEvalCallback  # NEW: full boards_eval between epochs
from .utils import ensure_dir, load_yaml, save_config_snapshot, save_run_meta, set_global_seed


def _setup_node_local_hf_cache() -> Path:
    """
    Put HF datasets cache (and optionally HF_HOME) on node-local storage.
    This avoids FileLock issues (e.g., Errno 116 stale file handle) on shared/NFS filesystems.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-epoch-eval", action="store_true", help="Disable custom lightweight epoch eval callback.")

    # Full eval (boards_eval gameplay) between epochs
    ap.add_argument("--full-eval", action="store_true", help="Run full Codenames eval on boards_eval at epoch end.")
    ap.add_argument("--full-eval-every", type=int, default=None, help="Run full eval every N epochs (default: 1).")
    ap.add_argument("--full-eval-batch-size", type=int, default=None, help="Override eval batch size (default: inference.batch_size).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["training"]
    set_global_seed(int(tcfg.get("seed", 0)))

    model_id = cfg["models"]["spymaster_model_id"]

    # ----- Node-local HF datasets cache -----
    local_datasets_cache = _setup_node_local_hf_cache()

    from datasets import load_dataset

    # ----- Dataset (TRL expects prompt/chosen/rejected) -----
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

    # ----- Base model -----
    dtype = torch.bfloat16 if (bool(tcfg.get("bf16", True)) and torch.cuda.is_available()) else None
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

    # ----- LoRA config -----
    peft_cfg = LoraConfig(
        r=int(tcfg.get("r", 16)),
        lora_alpha=int(tcfg.get("alpha", 32)),
        lora_dropout=float(tcfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=tcfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    out_dir = ensure_dir(tcfg["output_adapter_dir"])

    # ----- TRL Training config -----
    max_len = int(tcfg.get("trl_max_length", tcfg.get("max_seq_len", 4096)))
    trunc_mode = str(tcfg.get("trl_truncation_mode", "keep_start"))

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        beta=float(tcfg.get("dpo_beta", 0.1)),
        max_length=max_len,
        truncation_mode=trunc_mode,
        per_device_train_batch_size=int(tcfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("grad_accum", 16)),
        num_train_epochs=float(tcfg.get("epochs", 1)),
        learning_rate=float(tcfg.get("lr", 2e-4)),
        bf16=bool(tcfg.get("bf16", True)) and torch.cuda.is_available(),
        fp16=bool(tcfg.get("fp16", False)) and torch.cuda.is_available(),
        deepspeed=str(tcfg.get("deepspeed_config")) if tcfg.get("deepspeed_config") else None,
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": bool(tcfg.get("gradient_checkpointing_use_reentrant", False))},
        logging_strategy="epoch",
        save_strategy="steps",
        save_steps=30,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_cfg,
    )

    # Lightweight eval callback (your existing one)
    if not args.no_epoch_eval:
        trainer.add_callback(EpochEvalCallback(cfg, out_dir))

    # Full boards_eval gameplay eval callback (OPTIONAL; default off)
    full_eval_on = bool(args.full_eval) or _env_flag("FULL_EVAL", "0") or bool(cfg.get("epoch_eval", {}).get("full_eval", False))
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