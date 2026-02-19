from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GenerationConfig:
    temperature: float
    top_p: float
    top_k: int | None = None
    max_new_tokens: int


class HFTextGenerator:
    """
    Thin wrapper around transformers generation for reproducible calls.
    """
    def __init__(self, model_id: str, device_map: str = "auto", torch_dtype: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.model_id = model_id

        dtype = None
        if torch_dtype:
            dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
        )
        self.model.eval()

    def format_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str:
        # Qwen3 supports enable_thinking in apply_chat_template; fall back if tokenizer doesn't accept it.
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    def generate(
        self,
        prompt_or_messages: str | List[Dict[str, str]],
        gen_cfg: GenerationConfig,
        seed: Optional[int] = None,
        *,
        use_chat_template: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if use_chat_template:
            assert isinstance(prompt_or_messages, list)
            text = self.format_chat(prompt_or_messages, add_generation_prompt=True, enable_thinking=enable_thinking)
        else:
            assert isinstance(prompt_or_messages, str)
            text = prompt_or_messages

        inputs = self.tokenizer(text, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = gen_cfg.temperature is not None and gen_cfg.temperature > 1e-6
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=float(gen_cfg.temperature) if do_sample else None,
                top_p=float(gen_cfg.top_p) if do_sample else None,
                top_k=int(gen_cfg.top_k) if (do_sample and gen_cfg.top_k is not None) else None,
                max_new_tokens=int(gen_cfg.max_new_tokens),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


def load_lora_on_generator(base_generator: HFTextGenerator, adapter_dir: str) -> HFTextGenerator:
    """
    Mutates base_generator.model to a PeftModel-wrapped model for inference.
    """
    from peft import PeftModel
    base_generator.model = PeftModel.from_pretrained(base_generator.model, adapter_dir)
    base_generator.model.eval()
    return base_generator


class Embedder:
    """
    Embeds single tokens/words/short strings with caching.
    Uses sentence-transformers if available; otherwise uses a HF encoder with mean pooling.
    """
    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._cache: Dict[str, np.ndarray] = {}

        self._mode = None
        self._st_model = None
        self._hf_tok = None
        self._hf_model = None

        # Prefer sentence-transformers if installed and model_id looks like one
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._st_model = SentenceTransformer(model_id, device=device)
            self._mode = "st"
        except Exception:
            self._mode = "hf"
            from transformers import AutoModel, AutoTokenizer
            import torch
            self._hf_tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            self._hf_model = AutoModel.from_pretrained(model_id)
            self._hf_model.to(device)
            self._hf_model.eval()

    def embed(self, text: str) -> np.ndarray:
        key = text.strip().lower()
        if key in self._cache:
            return self._cache[key]

        if self._mode == "st":
            vec = np.asarray(self._st_model.encode([text], normalize_embeddings=True)[0], dtype=np.float32)
        else:
            # HF encoder mean pooling
            import torch
            assert self._hf_tok is not None and self._hf_model is not None
            with torch.no_grad():
                toks = self._hf_tok([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
                out = self._hf_model(**toks)
                last = out.last_hidden_state  # (B,T,H)
                mask = toks["attention_mask"].unsqueeze(-1)  # (B,T,1)
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vec = pooled[0].detach().cpu().numpy().astype(np.float32)
                # normalize
                n = np.linalg.norm(vec) + 1e-12
                vec = vec / n

        self._cache[key] = vec
        return vec

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))