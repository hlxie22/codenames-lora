# src/full_eval_callback.py
from __future__ import annotations

import json
import os
import time
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from transformers import TrainerCallback
except Exception:
    TrainerCallback = object  # type: ignore[misc]

from .eval_shared import (
    InTrainerGenerator,
    all_gather_objects,
    barrier,
    build_codenames_record,
    build_external_guesser,
    dist_rank_world,
    resolve_model_tokenizer_device,
    shard_list,
)
from .metrics import aggregate
from .model_wrappers import Embedder
from .rollout import run_turns_batched
from .utils import read_jsonl, write_jsonl


class FullCodenamesEvalCallback(TrainerCallback):
    def __init__(self, cfg: Dict[str, Any], out_dir: str | Path, *, every_epochs: int = 1, batch_size: int = 1):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.every_epochs = max(1, int(every_epochs))
        self.batch_size = max(1, int(batch_size))
        self._last_epoch_logged: Optional[int] = None
        self._external_guesser = None

    def _get_external_guesser(self):
        if self._external_guesser is None:
            self._external_guesser = build_external_guesser(self.cfg)
        return self._external_guesser

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_f = state.epoch
        epoch_i = int(epoch_f) if epoch_f is not None else 0

        if self._last_epoch_logged is not None and epoch_i == self._last_epoch_logged:
            return control
        self._last_epoch_logged = epoch_i

        if (epoch_i % self.every_epochs) != 0:
            return control

        rank, world = dist_rank_world()
        host = getattr(os, "uname", lambda: type("x", (), {"nodename": "unknown"})())().nodename

        def _errlog(msg: str) -> None:
            print(msg, file=sys.stderr, flush=True)
            try:
                p = self.out_dir / f"full_eval_rank{rank}.err"
                with p.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                    f.flush()
            except Exception:
                pass

        err_msgs: List[str] = []

        trainer = kwargs.get("trainer", None)
        model, tok, device = resolve_model_tokenizer_device(trainer, kwargs)

        if model is None or tok is None:
            err_msgs.append(f"[full_eval][rank {rank} host={host}] Could not resolve model/tokenizer; skipping.")
            all_errs = all_gather_objects("\n\n".join(err_msgs))
            if rank == 0:
                for r, s in enumerate(all_errs):
                    if s:
                        _errlog(f"[full_eval] errors from rank {r}:\n{s}")
            barrier()
            return control

        boards_local: List[Dict[str, Any]] = []
        try:
            boards_eval_path = self.cfg["paths"]["boards_eval_path"]
            boards_all = read_jsonl(boards_eval_path)
            boards_local = shard_list(boards_all, rank, world)
        except Exception as e:
            err_msgs.append(
                f"[full_eval][rank {rank} host={host}] Failed to load/shard boards_eval: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            boards_local = []

        spymaster = InTrainerGenerator(
            model,
            tok,
            device,
            disable_adapter=False,
        )

        guesser_override = self._get_external_guesser()
        if guesser_override is not None:
            guesser = guesser_override
        else:
            guesser = InTrainerGenerator(
                model,
                tok,
                device,
                disable_adapter=True,
            )

        embedder = None
        try:
            use_embed = bool(self.cfg.get("constraints", {}).get("enable_directness_check", True))
            if use_embed:
                embedder = Embedder(self.cfg["models"]["embedding_model_id"], device="cpu")
        except Exception as e:
            err_msgs.append(
                f"[full_eval][rank {rank} host={host}] Embedder init failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            embedder = None

        was_training = bool(getattr(model, "training", False))
        try:
            model.eval()
        except Exception:
            pass

        t0 = time.time()
        per_board_local: List[Dict[str, Any]] = []

        try:
            n_candidates = 1

            for start in range(0, len(boards_local), self.batch_size):
                batch = boards_local[start : start + self.batch_size]

                bests, metas = run_turns_batched(
                    batch,
                    spymaster,
                    guesser,
                    embedder,
                    self.cfg,
                    n_candidates=n_candidates,
                    max_resamples=1,
                )

                for b, best, meta in zip(batch, bests, metas):
                    per_board_local.append(build_codenames_record(b, best, meta))

        except Exception as e:
            err_msgs.append(
                f"[full_eval][rank {rank} host={host}] Full eval loop failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            per_board_local = []

        try:
            if was_training:
                model.train()
        except Exception:
            pass

        gathered = all_gather_objects(per_board_local)
        per_board_all: List[Dict[str, Any]] = []
        for part in gathered:
            per_board_all.extend(part or [])

        barrier()

        all_errs = all_gather_objects("\n\n".join([m for m in err_msgs if m.strip()]))
        if rank == 0:
            for r, s in enumerate(all_errs):
                if s:
                    _errlog(f"[full_eval] errors from rank {r}:\n{s}")

        barrier()

        if rank == 0:
            try:
                run_dir = self.out_dir / "full_eval" / f"epoch_{epoch_i:03d}_step_{int(getattr(state, 'global_step', 0)):06d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                per_path = run_dir / "per_board.jsonl"
                write_jsonl(per_path, per_board_all)

                metrics = aggregate(per_board_all)
                metrics["epoch"] = int(epoch_i)
                metrics["global_step"] = int(getattr(state, "global_step", 0))
                metrics["eval_seconds"] = float(time.time() - t0)
                metrics["n_boards_eval"] = int(len(per_board_all))

                (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

                if trainer is not None:
                    try:
                        trainer.log({f"full_eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
                    except Exception:
                        pass

                print(f"[full_eval] wrote {len(per_board_all)} boards -> {per_path}", flush=True)
                print(f"[full_eval] metrics -> {run_dir / 'metrics.json'}", flush=True)

            except Exception as e:
                _errlog(
                    f"[full_eval][rank {rank} host={host}] Writing metrics/output failed: {type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                )

        barrier()
        return control
