from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


# -------------------------
# One canonical chat-template helper
# -------------------------

def apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> str:
    """
    Centralized compatibility wrapper for tokenizer.apply_chat_template signature drift.

    Some tokenizers (e.g., Qwen3) accept enable_thinking=...
    Others do not. This wrapper tries with enable_thinking and falls back cleanly.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


@dataclass
class GenerationConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    top_k: int | None = None


@runtime_checkable
class TextGenerator(Protocol):
    model_id: str

    def format_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
    ) -> str: ...

    def generate(
        self,
        prompt_or_messages: str | List[Dict[str, str]],
        gen_cfg: GenerationConfig,
        seed: Optional[int] = None,
        *,
        use_chat_template: bool = False,
        enable_thinking: bool = True,
    ) -> str: ...


class VLLMTextGenerator:
    """
    Thin wrapper around vLLM offline inference.
    """

    def __init__(
        self,
        model_id: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # vLLM engine (offline)
        self.llm = LLM(
            model=model_id,
            tokenizer=model_id,
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enable_lora=bool(enable_lora),
            max_lora_rank=int(max_lora_rank),
        )

        self._lora_request = None  # set by load_lora_on_generator()

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
        gen_cfg: GenerationConfig,
        seed: Optional[int] = None,
        *,
        use_chat_template: bool = False,
        enable_thinking: bool = True,
    ) -> str:
        from vllm import SamplingParams

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

        sp = SamplingParams(
            temperature=float(gen_cfg.temperature),
            top_p=float(gen_cfg.top_p),
            top_k=int(gen_cfg.top_k) if gen_cfg.top_k is not None else -1,
            max_tokens=int(gen_cfg.max_new_tokens),
            seed=seed,
        )

        outs = self.llm.generate([prompt], sp, lora_request=self._lora_request)
        return outs[0].outputs[0].text

    def generate_batch(
        self,
        prompts: List[str],
        gen_cfg: GenerationConfig,
        seeds: Optional[List[Optional[int]]] = None,
    ) -> List[str]:
        from vllm import SamplingParams

        if seeds is None:
            seeds = [None] * len(prompts)
        assert len(seeds) == len(prompts)

        sps = [
            SamplingParams(
                temperature=float(gen_cfg.temperature),
                top_p=float(gen_cfg.top_p),
                top_k=int(gen_cfg.top_k) if gen_cfg.top_k is not None else -1,
                max_tokens=int(gen_cfg.max_new_tokens),
                seed=sd,
            )
            for sd in seeds
        ]

        outs = self.llm.generate(prompts, sps, lora_request=self._lora_request)
        return [o.outputs[0].text for o in outs]


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
        return apply_chat_template(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
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


def load_lora_on_generator(base_generator: TextGenerator, adapter_dir: str) -> TextGenerator:
    """
    HF path: wraps with PeftModel
    vLLM path: sets a LoRARequest that will be used on each generate() call
    """
    if isinstance(base_generator, HFTextGenerator):
        from peft import PeftModel

        base_generator.model = PeftModel.from_pretrained(base_generator.model, adapter_dir)
        base_generator.model.eval()
        return base_generator

    if isinstance(base_generator, VLLMTextGenerator):
        from vllm.lora.request import LoRARequest

        lora_int_id = abs(hash(adapter_dir)) % (2**31)
        base_generator._lora_request = LoRARequest("codenames-lora", int(lora_int_id), adapter_dir)
        return base_generator

    raise TypeError(f"Unsupported generator type: {type(base_generator)}")


def make_text_generator(model_id: str, cfg: Dict[str, Any]) -> TextGenerator:
    backend = cfg.get("inference", {}).get("backend", "hf")
    if backend == "vllm":
        vcfg = cfg.get("inference", {}).get("vllm", {})
        return VLLMTextGenerator(model_id, **vcfg)
    return HFTextGenerator(model_id, device_map="auto")


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

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(model_id, device=device)
            self._mode = "st"
        except Exception:
            self._mode = "hf"
            from transformers import AutoModel, AutoTokenizer

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
            import torch

            assert self._hf_tok is not None and self._hf_model is not None
            with torch.no_grad():
                toks = self._hf_tok([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
                out = self._hf_model(**toks)
                last = out.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1)
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                vec = pooled[0].detach().cpu().numpy().astype(np.float32)
                n = np.linalg.norm(vec) + 1e-12
                vec = vec / n

        self._cache[key] = vec
        return vec

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))