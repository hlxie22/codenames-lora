# src/train_lora_sft.py
from __future__ import annotations

import argparse
import inspect
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
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

from transformers import TrainerCallback

from .epoch_eval import (
    load_probe_samples,
    eval_gsm8k_raw,
    eval_humaneval_raw,
    eval_wikitext2_raw,
    sample_codenames_eval_boards,
    eval_codenames_boards_records,
    plot_epoch_history,
)

from .metrics import aggregate as aggregate_codenames


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

    if low in {"kernels-community/flash-attn2", "flash-attn2", "flash_attn2", "flashattn2"}:
        return "flash_attention_2"
    if low in {"kernels-community/flash-attn3", "flash-attn3", "flash_attn3", "flashattn3"}:
        return "flash_attention_3"

    if low in {"auto", "none", "null"}:
        return None

    allowed = {"eager", "sdpa", "flex_attention", "flash_attention_2", "flash_attention_3"}
    if s in allowed:
        return s
    if low in allowed:
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
        model.gradient_checkpointing_enable()

    try:
        model.enable_input_require_grads()
    except Exception:
        pass


# -------------------------
# Epoch eval callback (distributed, sharded across ranks)
# -------------------------

class EpochEvalCallback(TrainerCallback):
    """
    - Collects train loss / grad norm from on_log() (rank0 only)
    - Runs GSM8K, HumanEval, WikiText-2 raw totals, and Codenames subset eval
      SHARDED ACROSS RANKS at on_epoch_end()
    - Reduces/gathers results to rank0, appends metrics_history.jsonl, writes plots
    - Logs timing per eval (max wall time across ranks + mean per-rank time) + running averages

    IMPORTANT: We keep DDP in lock-step:
      barrier -> all ranks eval shard -> reduce/gather -> rank0 writes -> barrier
    """

    def __init__(self, cfg: Dict[str, Any], out_dir: Path):
        self.cfg = cfg
        self.out_dir = out_dir
        self.eval_dir = out_dir / "epoch_eval"
        self.history_path = self.eval_dir / "metrics_history.jsonl"
        self.plots_dir = self.eval_dir / "plots"
        self._tok = None

        self._epoch_losses: List[float] = []
        self._epoch_grad_norms: List[float] = []

        self._ema = None
        self._ema_beta = 0.7

        self.samples = None

        # timing running averages (rank0 only)
        self._time_sum_max = defaultdict(float)
        self._time_sum_mean = defaultdict(float)
        self._time_epochs = 0

        # object collectives group (gloo)
        self._gloo_group = None

    def _barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _rank_world(self) -> Tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _is_rank0(self) -> bool:
        return bool(getattr(dist, "is_initialized", lambda: False)() and dist.get_rank() == 0) if (
            dist.is_available() and dist.is_initialized()
        ) else True

    def _sync_cuda(self, device: torch.device) -> None:
        try:
            if device is not None and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
        except Exception:
            pass

    def _time_block(self, device: torch.device, fn):
        self._sync_cuda(device)
        t0 = time.perf_counter()
        out = fn()
        self._sync_cuda(device)
        dt = time.perf_counter() - t0
        return out, float(dt)

    def _reduce_sum_3(self, a: float, b: float, c: float, device: torch.device) -> Tuple[float, float, float]:
        if not (dist.is_available() and dist.is_initialized()):
            return float(a), float(b), float(c)
        t = torch.tensor([a, b, c], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t[0].item()), float(t[1].item()), float(t[2].item())

    def _reduce_sum_2(self, a: float, b: float, device: torch.device) -> Tuple[float, float]:
        if not (dist.is_available() and dist.is_initialized()):
            return float(a), float(b)
        t = torch.tensor([a, b], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float(t[0].item()), float(t[1].item())

    def _reduce_time_stats(self, dt: float, device: torch.device) -> Tuple[float, float]:
        """
        Returns (dt_max, dt_mean) across ranks.
        dt_max = wall time blocking the job
        dt_mean = average compute time per rank (load balance indicator)
        """
        if not (dist.is_available() and dist.is_initialized()):
            return float(dt), float(dt)

        tsum = torch.tensor([dt, 1.0], device=device, dtype=torch.float64)
        dist.all_reduce(tsum, op=dist.ReduceOp.SUM)
        dt_sum, n = float(tsum[0].item()), float(tsum[1].item())
        dt_mean = dt_sum / max(1.0, n)

        tmax = torch.tensor([dt], device=device, dtype=torch.float64)
        dist.all_reduce(tmax, op=dist.ReduceOp.MAX)
        dt_max = float(tmax.item())
        return dt_max, dt_mean

    def _gather_list_ints(self, local: List[int]) -> List[int]:
        """
        Gather lists of ints across ranks (using gloo group).
        Returns concatenated list on rank0; empty list on other ranks.
        """
        rank, world = self._rank_world()
        if world == 1:
            return list(local)

        assert self._gloo_group is not None, "gloo group not initialized"

        gathered: List[Optional[List[int]]] = [None for _ in range(world)]
        dist.all_gather_object(gathered, local, group=self._gloo_group)

        if rank != 0:
            return []
        out: List[int] = []
        for part in gathered:
            if part:
                out.extend(part)
        return out

    def _gather_list_objects(self, local: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Gather lists of dicts across ranks (using gloo group).
        Returns concatenated list on rank0; empty list on other ranks.
        """
        rank, world = self._rank_world()
        if world == 1:
            return list(local)

        assert self._gloo_group is not None, "gloo group not initialized"

        gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world)]
        dist.all_gather_object(gathered, local, group=self._gloo_group)

        if rank != 0:
            return []
        out: List[Dict[str, Any]] = []
        for part in gathered:
            if part:
                out.extend(part)
        return out

    def on_train_begin(self, args, state, control, **kwargs):
        # Create dirs once
        self._barrier()
        try:
            if self._is_rank0():
                self.eval_dir.mkdir(parents=True, exist_ok=True)
                self.plots_dir.mkdir(parents=True, exist_ok=True)
        finally:
            self._barrier()

        # Load probe samples in a download-safe way:
        # rank0 downloads/caches, barrier, others load from cache.
        seed = int(self.cfg["training"].get("seed", 0)) + 12345
        gsm_n = int(self.cfg.get("epoch_eval", {}).get("gsm8k_n", 50))
        he_n = int(self.cfg.get("epoch_eval", {}).get("humaneval_n", 20))
        wt_n = int(self.cfg.get("epoch_eval", {}).get("wikitext_n", 50))

        self._barrier()
        try:
            if self._is_rank0():
                self.samples = load_probe_samples(seed=seed, gsm8k_n=gsm_n, humaneval_n=he_n, wikitext_n=wt_n)
        finally:
            self._barrier()

        if not self._is_rank0():
            self.samples = load_probe_samples(seed=seed, gsm8k_n=gsm_n, humaneval_n=he_n, wikitext_n=wt_n)

        # capture tokenizer/processing_class reference
        self._tok = kwargs.get("tokenizer") or kwargs.get("processing_class") or self._tok

        # init a gloo group for object collectives
        if dist.is_available() and dist.is_initialized() and self._gloo_group is None:
            self._gloo_group = dist.new_group(backend="gloo")

        self._barrier()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._is_rank0():
            return
        if not logs:
            return
        if "loss" in logs:
            try:
                self._epoch_losses.append(float(logs["loss"]))
            except Exception:
                pass
        if "grad_norm" in logs:
            try:
                self._epoch_grad_norms.append(float(logs["grad_norm"]))
            except Exception:
                pass

    def _append_history(self, row: Dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        # All ranks must enter/exit together
        self._barrier()
        try:
            if self.samples is None:
                return

            model = kwargs.get("model", None)
            tokenizer = self._tok or kwargs.get("tokenizer") or kwargs.get("processing_class")
            if model is None or tokenizer is None:
                return

            if hasattr(model, "module"):
                model = model.module

            device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

            rank, world = self._rank_world()

            # Temporarily enable use_cache for faster generation
            orig_use_cache = None
            try:
                if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                    orig_use_cache = model.config.use_cache
                    model.config.use_cache = True
            except Exception:
                pass

            # ---- shard probe samples by global index ----
            gsm_all = self.samples.gsm8k
            he_all = self.samples.humaneval
            wt_all = self.samples.wikitext

            gsm_local = [ex for i, ex in enumerate(gsm_all) if (i % world) == rank]
            gsm_gidx = [i for i in range(len(gsm_all)) if (i % world) == rank]

            he_local = [ex for i, ex in enumerate(he_all) if (i % world) == rank]
            he_gidx = [i for i in range(len(he_all)) if (i % world) == rank]

            wt_local = [t for i, t in enumerate(wt_all) if (i % world) == rank]

            # deterministic seed base for probe eval
            probe_seed_base = int(self.cfg["training"].get("seed", 0)) + 99991

            # ---- time + run eval shards ----
            gsm_raw, t_gsm = self._time_block(
                device,
                lambda: eval_gsm8k_raw(
                    model=model,
                    tokenizer=tokenizer,
                    samples=gsm_local,
                    global_indices=gsm_gidx,
                    device=device,
                    max_new_tokens=int(self.cfg.get("epoch_eval", {}).get("gsm8k_max_new_tokens", 384)),
                    seed_base=probe_seed_base,
                ),
            )
            t_gsm_max, t_gsm_mean = self._reduce_time_stats(t_gsm, device)

            he_raw, t_he = self._time_block(
                device,
                lambda: eval_humaneval_raw(
                    model=model,
                    tokenizer=tokenizer,
                    samples=he_local,
                    global_indices=he_gidx,
                    device=device,
                    max_new_tokens=int(self.cfg.get("epoch_eval", {}).get("humaneval_max_new_tokens", 384)),
                    timeout_s=int(self.cfg.get("epoch_eval", {}).get("humaneval_timeout_s", 3)),
                    seed_base=probe_seed_base + 100000,
                ),
            )
            t_he_max, t_he_mean = self._reduce_time_stats(t_he, device)

            wt_raw, t_wt = self._time_block(
                device,
                lambda: eval_wikitext2_raw(
                    model=model,
                    tokenizer=tokenizer,
                    texts=wt_local,
                    device=device,
                    block_size=int(self.cfg.get("epoch_eval", {}).get("wikitext_block_size", 512)),
                    max_blocks=int(self.cfg.get("epoch_eval", {}).get("wikitext_max_blocks", 80)),
                ),
            )
            t_wt_max, t_wt_mean = self._reduce_time_stats(t_wt, device)

            # Codenames subset: sample deterministically on every rank, then shard by rank
            cn_seed = int(self.cfg["training"].get("seed", 0)) + 999
            cn_n = int(self.cfg.get("epoch_eval", {}).get("codenames_n_boards", 100))
            cn_subset = sample_codenames_eval_boards(cfg=self.cfg, n_boards=cn_n, seed=cn_seed)
            cn_local = [b for i, b in enumerate(cn_subset) if (i % world) == rank]

            cn_raw, t_cn = self._time_block(
                device,
                lambda: eval_codenames_boards_records(
                    cfg=self.cfg,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    boards=cn_local,
                ),
            )
            t_cn_max, t_cn_mean = self._reduce_time_stats(t_cn, device)

            # restore cache flag
            try:
                if orig_use_cache is not None and hasattr(model, "config") and hasattr(model.config, "use_cache"):
                    model.config.use_cache = orig_use_cache
            except Exception:
                pass

            # ---- reduce numeric metrics via all_reduce ----
            # GSM8K: correct, n, think_sum
            gsm_correct = float(gsm_raw["correct"])
            gsm_n = float(gsm_raw["n"])
            gsm_think_sum = float(sum(gsm_raw["think_tokens"]))
            gsm_correct_all, gsm_n_all, gsm_think_sum_all = self._reduce_sum_3(gsm_correct, gsm_n, gsm_think_sum, device)
            gsm_acc = gsm_correct_all / max(1.0, gsm_n_all)
            gsm_think_mean = gsm_think_sum_all / max(1.0, gsm_n_all)

            # HumanEval: passed, n, think_sum
            he_passed = float(he_raw["passed"])
            he_n = float(he_raw["n"])
            he_think_sum = float(sum(he_raw["think_tokens"]))
            he_passed_all, he_n_all, he_think_sum_all = self._reduce_sum_3(he_passed, he_n, he_think_sum, device)
            he_pass_rate = he_passed_all / max(1.0, he_n_all)
            he_think_mean = he_think_sum_all / max(1.0, he_n_all)

            # WikiText: total_nll, total_tokens; blocks_done (sum)
            wt_nll_all, wt_tok_all = self._reduce_sum_2(float(wt_raw["total_nll"]), float(wt_raw["total_tokens"]), device)
            wt_blocks_all, _ = self._reduce_sum_2(float(wt_raw["blocks_done"]), 0.0, device)
            wt_ppl = math.exp(wt_nll_all / max(1.0, wt_tok_all))

            # ---- gather lists/records for p90 + codenames aggregate ----
            gsm_think_all = self._gather_list_ints(list(gsm_raw["think_tokens"]))
            he_think_all = self._gather_list_ints(list(he_raw["think_tokens"]))

            cn_per_board_all = self._gather_list_objects(list(cn_raw["per_board"]))
            cn_think_all = self._gather_list_ints(list(cn_raw["think_tokens"]))

            # total eval time stats (max/mean across ranks)
            t_total_local = t_gsm + t_he + t_wt + t_cn
            t_total_max, t_total_mean = self._reduce_time_stats(t_total_local, device)

            # ---- rank0: compute p90s, codenames aggregate, write history ----
            if self._is_rank0():
                gsm_p90 = float(sorted(gsm_think_all)[int(0.9 * (len(gsm_think_all) - 1))]) if gsm_think_all else 0.0
                he_p90 = float(sorted(he_think_all)[int(0.9 * (len(he_think_all) - 1))]) if he_think_all else 0.0
                cn_p90 = float(sorted(cn_think_all)[int(0.9 * (len(cn_think_all) - 1))]) if cn_think_all else 0.0
                cn_mean = float(sum(cn_think_all) / max(1, len(cn_think_all))) if cn_think_all else 0.0

                cn_metrics = aggregate_codenames(cn_per_board_all) if cn_per_board_all else {"n_boards": 0}

                # ---- train loss aggregation (rank0 only) ----
                loss_mean = (
                    float(sum(self._epoch_losses) / max(1, len(self._epoch_losses)))
                    if self._epoch_losses
                    else float("nan")
                )
                self._epoch_losses.clear()

                if self._ema is None or math.isnan(self._ema):
                    self._ema = loss_mean
                else:
                    self._ema = self._ema_beta * self._ema + (1.0 - self._ema_beta) * loss_mean

                grad_mean = (
                    float(sum(self._epoch_grad_norms) / max(1, len(self._epoch_grad_norms)))
                    if self._epoch_grad_norms
                    else float("nan")
                )
                self._epoch_grad_norms.clear()

                # ---- timing running averages ----
                self._time_epochs += 1
                def upd(name: str, dt_max: float, dt_mean: float):
                    self._time_sum_max[name] += float(dt_max)
                    self._time_sum_mean[name] += float(dt_mean)

                upd("gsm8k", t_gsm_max, t_gsm_mean)
                upd("humaneval", t_he_max, t_he_mean)
                upd("wikitext", t_wt_max, t_wt_mean)
                upd("codenames", t_cn_max, t_cn_mean)
                upd("total", t_total_max, t_total_mean)

                def avg_max(name: str) -> float:
                    return float(self._time_sum_max[name] / max(1, self._time_epochs))

                def avg_mean(name: str) -> float:
                    return float(self._time_sum_mean[name] / max(1, self._time_epochs))

                row = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "epoch": float(state.epoch) if state.epoch is not None else None,

                    "train_loss_epoch_mean": loss_mean,
                    "train_loss_epoch_ema": float(self._ema) if self._ema is not None else None,
                    "grad_norm_epoch_mean": grad_mean if not math.isnan(grad_mean) else None,

                    # GSM8K
                    "gsm8k_n": int(round(gsm_n_all)),
                    "gsm8k_acc": float(gsm_acc),
                    "gsm8k_think_tokens_mean": float(gsm_think_mean),
                    "gsm8k_think_tokens_p90": float(gsm_p90),

                    # HumanEval
                    "humaneval_n": int(round(he_n_all)),
                    "humaneval_pass_rate": float(he_pass_rate),
                    "humaneval_think_tokens_mean": float(he_think_mean),
                    "humaneval_think_tokens_p90": float(he_p90),

                    # WikiText
                    "wikitext2_blocks": int(round(wt_blocks_all)),
                    "wikitext2_tokens": int(round(wt_tok_all)),
                    "wikitext2_ppl": float(wt_ppl),

                    # Codenames aggregate
                    "codenames_n_boards": int(cn_metrics.get("n_boards", 0)),
                    "reward_mean": float(cn_metrics.get("reward_mean", 0.0)),
                    "reward_median": float(cn_metrics.get("reward_median", 0.0)),
                    "reward_ci95_lo": float(cn_metrics.get("reward_ci95", (0.0, 0.0))[0]),
                    "reward_ci95_hi": float(cn_metrics.get("reward_ci95", (0.0, 0.0))[1]),
                    "assassin_rate": float(cn_metrics.get("assassin_rate", 0.0)),
                    "team_mean": float(cn_metrics.get("team_mean", 0.0)),
                    "opp_mean": float(cn_metrics.get("opp_mean", 0.0)),
                    "neu_mean": float(cn_metrics.get("neu_mean", 0.0)),
                    "codenames_spymaster_think_tokens_mean": float(cn_mean),
                    "codenames_spymaster_think_tokens_p90": float(cn_p90),

                    # timing (per-epoch)
                    "t_gsm8k_s_max": float(t_gsm_max),
                    "t_gsm8k_s_mean": float(t_gsm_mean),
                    "t_humaneval_s_max": float(t_he_max),
                    "t_humaneval_s_mean": float(t_he_mean),
                    "t_wikitext_s_max": float(t_wt_max),
                    "t_wikitext_s_mean": float(t_wt_mean),
                    "t_codenames_s_max": float(t_cn_max),
                    "t_codenames_s_mean": float(t_cn_mean),
                    "t_eval_total_s_max": float(t_total_max),
                    "t_eval_total_s_mean": float(t_total_mean),

                    # timing (running averages)
                    "t_gsm8k_s_max_avg": avg_max("gsm8k"),
                    "t_humaneval_s_max_avg": avg_max("humaneval"),
                    "t_wikitext_s_max_avg": avg_max("wikitext"),
                    "t_codenames_s_max_avg": avg_max("codenames"),
                    "t_eval_total_s_max_avg": avg_max("total"),
                    "t_gsm8k_s_mean_avg": avg_mean("gsm8k"),
                    "t_humaneval_s_mean_avg": avg_mean("humaneval"),
                    "t_wikitext_s_mean_avg": avg_mean("wikitext"),
                    "t_codenames_s_mean_avg": avg_mean("codenames"),
                    "t_eval_total_s_mean_avg": avg_mean("total"),
                }

                self._append_history(row)
                plot_epoch_history(self.history_path, self.plots_dir)

                print(
                    f"[epoch {row['epoch']}] eval wall(max) "
                    f"gsm={t_gsm_max:.1f}s he={t_he_max:.1f}s wt={t_wt_max:.1f}s cn={t_cn_max:.1f}s "
                    f"total={t_total_max:.1f}s | avg_total={avg_max('total'):.1f}s"
                )

        finally:
            self._barrier()


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

    # Attention implementation
    attn_impl = _normalize_attn_impl(tcfg.get("attn_implementation", "sdpa"))

    # Build model
    model_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    _disable_kv_cache(model)

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

        logging_strategy="epoch",
        logging_first_step=True,

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

        # IMPORTANT: even with sharded eval, keep a generous timeout for safety
        ddp_timeout=int(tcfg.get("ddp_timeout_s", 7200)),
    )

    try:
        sig = inspect.signature(TrainingArguments.__init__)
        if gc_enabled and gc_kwargs is not None and "gradient_checkpointing_kwargs" in sig.parameters:
            ta_kwargs["gradient_checkpointing_kwargs"] = gc_kwargs
    except Exception:
        pass

    args_tr = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.add_callback(EpochEvalCallback(cfg, out_dir))

    last_ckpt = get_last_checkpoint(str(out_dir))
    trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)

    if trainer.is_world_process_zero():
        m = trainer.model
        if hasattr(m, "module"):
            m = m.module
        m.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        save_config_snapshot(cfg, out_dir)
        save_run_meta(out_dir, command=["python", "-m", "src.train_lora_sft", "--config", args.config])

    print(f"Saved LoRA adapter -> {out_dir}")


if __name__ == "__main__":
    main()