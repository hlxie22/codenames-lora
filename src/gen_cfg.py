# src/gen_cfg.py
from .model_wrappers import GenerationConfig

def spymaster_gen_cfg(cfg) -> GenerationConfig:
    d = cfg["decoding"]
    return GenerationConfig(
        temperature=float(d["spymaster_temperature"]),
        top_p=float(d["spymaster_top_p"]),
        top_k=int(d.get("spymaster_top_k", 20)),
        max_new_tokens=int(d["spymaster_max_new_tokens"]),
    )

def guesser_gen_cfg(cfg) -> GenerationConfig:
    d = cfg["decoding"]
    return GenerationConfig(
        temperature=float(d["guesser_temperature"]),
        top_p=float(d["guesser_top_p"]),
        top_k=int(d.get("guesser_top_k", 20)),
        max_new_tokens=int(d["guesser_max_new_tokens"]),
    )