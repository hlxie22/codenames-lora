from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import math
import time

import torch

try:
    from transformers import TrainerCallback
except Exception as e:  # very defensive
    TrainerCallback = object  # type: ignore[misc]


def _dist_available() -> bool:
    try:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _dist_rank_world() -> Tuple[int, int]:
    if not _dist_available():
        return (0, 1)
    import torch.distributed as dist

    return (dist.get_rank(), dist.get_world_size())


def _all_reduce_sum_float(x: float, device: torch.device) -> float:
    if not _dist_available():
        return float(x)
    import torch.distributed as dist

    t = torch.tensor([float(x)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _all_reduce_sum_int(x: int, device: torch.device) -> int:
    if not _dist_available():
        return int(x)
    import torch.distributed as dist

    t = torch.tensor([int(x)], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def _all_gather_objects(obj: Any) -> List[Any]:
    if not _dist_available():
        return [obj]
    import torch.distributed as dist

    world = dist.get_world_size()
    out: List[Any] = [None] * world
    dist.all_gather_object(out, obj)
    return out


def _barrier() -> None:
    if not _dist_available():
        return
    import torch.distributed as dist

    dist.barrier()


def _p90(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = int(0.9 * (len(ys) - 1))
    return float(ys[i])


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / max(1, len(xs))) if xs else 0.0


def _shard_enumerated(items: List[Any], rank: int, world: int) -> Tuple[List[Any], List[int]]:
    """Return (items_for_rank, global_indices_for_rank) using i % world == rank sharding."""
    local: List[Any] = []
    idxs: List[int] = []
    for i, it in enumerate(items):
        if (i % world) == rank:
            local.append(it)
            idxs.append(i)
    return local, idxs


def _unwrap_model_from_trainer(trainer: Any) -> Any:
    """
    Try to get the underlying HF model (works for Accelerate/DeepSpeed wrapping).
    We keep this conservative and best-effort.
    """
    m = getattr(trainer, "model", None)
    if m is None:
        return None

    acc = getattr(trainer, "accelerator", None)
    if acc is not None:
        try:
            return acc.unwrap_model(m)
        except Exception:
            pass

    # DeepSpeedEngine often exposes .module
    if hasattr(m, "module"):
        try:
            return m.module
        except Exception:
            pass

    return m


def _trainer_device(trainer: Any) -> torch.device:
    # Prefer Trainer/Accelerate device if present
    a = getattr(trainer, "args", None)
    if a is not None and getattr(a, "device", None) is not None:
        return a.device
    if torch.cuda.is_available():
        # if LOCAL_RANK is set, pick that
        lr = int(os.environ.get("LOCAL_RANK", "0")) if "os" in globals() else 0
        return torch.device("cuda", lr)
    return torch.device("cpu")


def _get_tokenizer_from_trainer(trainer: Any) -> Any:
    tok = getattr(trainer, "processing_class", None)
    if tok is not None:
        return tok
    tok = getattr(trainer, "tokenizer", None)
    if tok is not None:
        return tok
    return None


class EpochEvalCallback(TrainerCallback):
    """
    End-of-epoch eval callback:
      - GSM8K accuracy + think token stats
      - HumanEval pass rate + think token stats
      - WikiText-2 perplexity
      - Codenames subset reward metrics

    Writes:
      - <out_dir>/epoch_eval_history.jsonl  (append-only)
      - <out_dir>/epoch_eval_plots/*.png    (rank0 only)
    """

    def __init__(self, cfg: Dict[str, Any], out_dir: str | Path):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.history_path = self.out_dir / "epoch_eval_history.jsonl"
        self.plots_dir = self.out_dir / "epoch_eval_plots"
        self._probes = None  # loaded once
        self._last_epoch_logged: Optional[int] = None

    def _ensure_probes(self) -> None:
        if self._probes is not None:
            return
        ecfg = (self.cfg.get("epoch_eval", {}) or {})
        seed = int(self.cfg.get("training", {}).get("seed", 0))
        self._probes = load_probe_samples(
            seed=seed,
            gsm8k_n=int(ecfg.get("gsm8k_n", 10)),
            humaneval_n=int(ecfg.get("humaneval_n", 10)),
            wikitext_n=int(ecfg.get("wikitext_n", 20)),
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        # Avoid double-firing for the same epoch in some Trainer schedules
        epoch_f = state.epoch
        epoch_i = int(epoch_f) if epoch_f is not None else 0
        if self._last_epoch_logged is not None and epoch_i == self._last_epoch_logged:
            return control
        self._last_epoch_logged = epoch_i

        rank, world = _dist_rank_world()

        trainer = kwargs.get("trainer", None)
        if trainer is None:
            # Some Trainer versions don't pass it; bail safely.
            return control

        model = _unwrap_model_from_trainer(trainer)
        tok = _get_tokenizer_from_trainer(trainer)
        if model is None or tok is None:
            return control

        device = _trainer_device(trainer)

        # Load probes once (each rank does it; they are small)
        try:
            self._ensure_probes()
        except Exception as e:
            if rank == 0:
                print(f"[epoch_eval] Failed to load probe datasets: {type(e).__name__}: {e}")
            return control

        ecfg = (self.cfg.get("epoch_eval", {}) or {})
        tcfg = (self.cfg.get("training", {}) or {})

        # Put model in eval, then restore mode after
        was_training = bool(getattr(model, "training", False))
        try:
            model.eval()
        except Exception:
            pass

        metrics: Dict[str, Any] = {}
        t0 = time.time()

        # ---------- GSM8K (distributed shard + reduce) ----------
        try:
            gsm_all = list(self._probes.gsm8k)  # type: ignore[union-attr]
            gsm_local, gsm_gi = _shard_enumerated(gsm_all, rank, world)

            raw = eval_gsm8k_raw(
                model=model,
                tokenizer=tok,
                samples=gsm_local,
                global_indices=gsm_gi,
                device=device,
                max_new_tokens=int(ecfg.get("gsm8k_max_new_tokens", 384)),
                seed_base=int(tcfg.get("seed", 0)) + 12345,
            )

            n = _all_reduce_sum_int(int(raw["n"]), device)
            correct = _all_reduce_sum_int(int(raw["correct"]), device)

            # gather think tokens for p90/mean
            gathered = _all_gather_objects([int(x) for x in raw.get("think_tokens", [])])
            think_tokens: List[int] = []
            for part in gathered:
                think_tokens.extend([int(x) for x in (part or [])])

            metrics["gsm8k_n"] = int(n)
            metrics["gsm8k_acc"] = float(correct / max(1, n))
            metrics["gsm8k_think_tokens_mean"] = float(_mean([float(x) for x in think_tokens]))
            metrics["gsm8k_think_tokens_p90"] = float(_p90([float(x) for x in think_tokens]))
        except Exception as e:
            if rank == 0:
                print(f"[epoch_eval] GSM8K eval failed: {type(e).__name__}: {e}")

        _barrier()

        # ---------- HumanEval (distributed shard + reduce) ----------
        try:
            he_all = list(self._probes.humaneval)  # type: ignore[union-attr]
            he_local, he_gi = _shard_enumerated(he_all, rank, world)

            raw = eval_humaneval_raw(
                model=model,
                tokenizer=tok,
                samples=he_local,
                global_indices=he_gi,
                device=device,
                max_new_tokens=int(ecfg.get("humaneval_max_new_tokens", 384)),
                timeout_s=int(ecfg.get("humaneval_timeout_s", 3)),
                seed_base=int(tcfg.get("seed", 0)) + 23456,
            )

            n = _all_reduce_sum_int(int(raw["n"]), device)
            passed = _all_reduce_sum_int(int(raw["passed"]), device)

            gathered = _all_gather_objects([int(x) for x in raw.get("think_tokens", [])])
            think_tokens: List[int] = []
            for part in gathered:
                think_tokens.extend([int(x) for x in (part or [])])

            metrics["humaneval_n"] = int(n)
            metrics["humaneval_pass_rate"] = float(passed / max(1, n))
            metrics["humaneval_think_tokens_mean"] = float(_mean([float(x) for x in think_tokens]))
            metrics["humaneval_think_tokens_p90"] = float(_p90([float(x) for x in think_tokens]))
        except Exception as e:
            if rank == 0:
                print(f"[epoch_eval] HumanEval eval failed: {type(e).__name__}: {e}")

        _barrier()

        # ---------- WikiText-2 PPL (distributed sum reduce) ----------
        try:
            wt_all = list(self._probes.wikitext)  # type: ignore[union-attr]
            wt_local, _ = _shard_enumerated(wt_all, rank, world)

            raw = eval_wikitext2_raw(
                model=model,
                tokenizer=tok,
                texts=wt_local,
                device=device,
                block_size=int(ecfg.get("wikitext_block_size", 512)),
                max_blocks=int(ecfg.get("wikitext_max_blocks", 80)),
            )

            total_nll = _all_reduce_sum_float(float(raw["total_nll"]), device)
            total_tokens = _all_reduce_sum_int(int(raw["total_tokens"]), device)
            blocks_done = _all_reduce_sum_int(int(raw["blocks_done"]), device)

            ppl = math.exp(total_nll / max(1, total_tokens))
            metrics["wikitext2_blocks"] = int(blocks_done)
            metrics["wikitext2_tokens"] = int(total_tokens)
            metrics["wikitext2_ppl"] = float(ppl)
        except Exception as e:
            if rank == 0:
                print(f"[epoch_eval] WikiText-2 eval failed: {type(e).__name__}: {e}")

        _barrier()

        # ---------- Codenames subset (distributed gather per-board then aggregate on rank0) ----------
        try:
            n_boards = int(ecfg.get("codenames_n_boards", 0))
            if n_boards > 0:
                seed = int(tcfg.get("seed", 0))
                subset = sample_codenames_eval_boards(cfg=self.cfg, n_boards=n_boards, seed=seed)

                # shard the subset across ranks
                subset_local, _ = _shard_enumerated(subset, rank, world)

                raw = eval_codenames_boards_records(
                    cfg=self.cfg,
                    model=model,
                    tokenizer=tok,
                    device=device,
                    boards=subset_local,
                )

                gathered_records = _all_gather_objects(raw.get("per_board", []))
                per_board: List[Dict[str, Any]] = []
                for part in gathered_records:
                    per_board.extend(part or [])

                gathered_think = _all_gather_objects([int(x) for x in raw.get("think_tokens", [])])
                think_tokens: List[int] = []
                for part in gathered_think:
                    think_tokens.extend([int(x) for x in (part or [])])

                if rank == 0 and per_board:
                    from .metrics import aggregate as _agg  # local import to avoid cycles

                    m = _agg(per_board)
                    metrics["codenames_n_boards"] = int(m.get("n_boards", len(per_board)))
                    metrics["reward_mean"] = float(m.get("reward_mean", 0.0))
                    metrics["reward_median"] = float(m.get("reward_median", 0.0))
                    ci = m.get("reward_ci95", (0.0, 0.0))
                    metrics["reward_ci95_lo"] = float(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else 0.0
                    metrics["reward_ci95_hi"] = float(ci[1]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else 0.0
                    metrics["assassin_rate"] = float(m.get("assassin_rate", 0.0))
                    metrics["team_mean"] = float(m.get("team_mean", 0.0))
                    metrics["opp_mean"] = float(m.get("opp_mean", 0.0))
                    metrics["neu_mean"] = float(m.get("neu_mean", 0.0))

                    metrics["codenames_spymaster_think_tokens_mean"] = float(_mean([float(x) for x in think_tokens]))
                    metrics["codenames_spymaster_think_tokens_p90"] = float(_p90([float(x) for x in think_tokens]))
        except Exception as e:
            if rank == 0:
                print(f"[epoch_eval] Codenames subset eval failed: {type(e).__name__}: {e}")

        # Restore mode
        try:
            if was_training:
                model.train()
        except Exception:
            pass

        metrics["epoch"] = int(epoch_i)
        metrics["global_step"] = int(getattr(state, "global_step", 0))
        metrics["epoch_eval_seconds"] = float(time.time() - t0)

        # Rank0: write + plot + log to trainer
        if rank == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)

            with self.history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            # best-effort Trainer logging (so it appears in console/loggers)
            try:
                trainer.log(metrics)
            except Exception:
                pass

            # plots
            try:
                plot_epoch_history(self.history_path, self.plots_dir)
            except Exception as e:
                print(f"[epoch_eval] plot_epoch_history failed: {type(e).__name__}: {e}")

            print(f"[epoch_eval] wrote epoch {epoch_i} metrics -> {self.history_path}")

        _barrier()
        return control