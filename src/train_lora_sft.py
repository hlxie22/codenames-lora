from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from transformers.trainer_utils import get_last_checkpoint

from .utils import load_yaml, read_jsonl, ensure_dir, save_config_snapshot, save_run_meta, set_global_seed

class SFTDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_len: int):
        self.records = records
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        prompt = r["prompt"]
        completion = r["completion"]

        # Tokenize prompt and completion separately so we can mask prompt loss.
        p = self.tok(prompt, add_special_tokens=False)
        c = self.tok(completion, add_special_tokens=False)

        input_ids = p["input_ids"] + c["input_ids"] + [self.tok.eos_token_id]
        # labels: -100 for prompt tokens, normal ids for completion + eos
        labels = [-100] * len(p["input_ids"]) + c["input_ids"] + [self.tok.eos_token_id]

        input_ids = input_ids[: self.max_len]
        labels = labels[: self.max_len]

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
        # left-pad vs right-pad: use tokenizer padding side; default right-pad
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attn = [b["attention_mask"] for b in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["training"]

    set_global_seed(int(tcfg.get("seed", 0)))

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

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )

    lora_cfg = LoraConfig(
        r=int(tcfg["r"]),
        lora_alpha=int(tcfg["alpha"]),
        lora_dropout=float(tcfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=tcfg.get("target_modules", None),
    )
    model = get_peft_model(model, lora_cfg)

    max_len = int(tcfg["max_seq_len"])
    ds = SFTDataset(records, tokenizer, max_len=max_len)
    collator = PadCollator(tokenizer)

    out_dir = ensure_dir(tcfg["output_adapter_dir"])

    args_tr = TrainingArguments(
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
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    last_ckpt = get_last_checkpoint(str(out_dir))
    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save adapter
    if trainer.is_world_process_zero():
        m = trainer.model
        if hasattr(m, "module"):  # DDP wrap
            m = m.module
        m.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    # Save config + run meta
    if trainer.is_world_process_zero():
        save_config_snapshot(cfg, out_dir)
        save_run_meta(out_dir, command=["python", "-m", "src.train_lora_sft", "--config", args.config])

    print(f"Saved LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()