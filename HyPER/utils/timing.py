"""Lightweight timing helpers for HyPER training runs."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_s": None, "median_s": None, "min_s": None, "max_s": None}
    return {
        "count": len(values),
        "mean_s": float(statistics.fmean(values)),
        "median_s": float(statistics.median(values)),
        "min_s": float(min(values)),
        "max_s": float(max(values)),
    }


class TrainingTimingCallback(Callback):
    """Collect coarse training/validation timing without synchronising CUDA by default."""

    def __init__(
        self,
        log_every_n_steps: int = 50,
        cuda_synchronize: bool = False,
        output_json: str | None = None,
    ) -> None:
        super().__init__()
        self.log_every_n_steps = int(log_every_n_steps or 0)
        self.cuda_synchronize = bool(cuda_synchronize)
        self.output_json = output_json
        self.epoch_start = None
        self.validation_start = None
        self.batch_start = None
        self.previous_batch_end = None
        self.train_step_times: list[float] = []
        self.batch_fetch_times: list[float] = []
        self.epoch_times: list[float] = []
        self.validation_times: list[float] = []

    def _sync(self) -> None:
        if self.cuda_synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._sync()
        now = time.perf_counter()
        self.epoch_start = now
        self.previous_batch_end = now

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._sync()
        now = time.perf_counter()
        if self.previous_batch_end is not None:
            self.batch_fetch_times.append(now - self.previous_batch_end)
        self.batch_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._sync()
        now = time.perf_counter()
        if self.batch_start is not None:
            elapsed = now - self.batch_start
            self.train_step_times.append(elapsed)
            if self.log_every_n_steps and (batch_idx + 1) % self.log_every_n_steps == 0:
                print(f"[HYPER_TIMING] train_batch={batch_idx + 1} step_s={elapsed:.4f}")
        self.previous_batch_end = now

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._sync()
        if self.epoch_start is not None:
            elapsed = time.perf_counter() - self.epoch_start
            self.epoch_times.append(elapsed)
            print(f"[HYPER_TIMING] epoch={trainer.current_epoch} train_epoch_s={elapsed:.3f}")

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._sync()
        self.validation_start = time.perf_counter()

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._sync()
        if self.validation_start is not None:
            elapsed = time.perf_counter() - self.validation_start
            self.validation_times.append(elapsed)
            print(f"[HYPER_TIMING] epoch={trainer.current_epoch} validation_epoch_s={elapsed:.3f}")

    def _output_path(self, trainer) -> Path:
        if self.output_json:
            return Path(self.output_json)
        log_dir = getattr(trainer.logger, "log_dir", None) or os.getcwd()
        return Path(log_dir) / "training_timing_summary.json"

    def on_fit_end(self, trainer, pl_module) -> None:
        datamodule = getattr(trainer, "datamodule", None)
        setup_timings = getattr(datamodule, "setup_timings", {}) if datamodule is not None else {}
        n_train = len(getattr(datamodule, "train_data", [])) if datamodule is not None and getattr(datamodule, "train_data", None) is not None else None
        payload: dict[str, Any] = {
            "batch_size": getattr(datamodule, "batch_size", None) if datamodule is not None else None,
            "num_workers": getattr(datamodule, "num_workers", None) if datamodule is not None else None,
            "pin_memory": getattr(datamodule, "pin_memory", None) if datamodule is not None else None,
            "persistent_workers": getattr(datamodule, "persistent_workers", None) if datamodule is not None else None,
            "prefetch_factor": getattr(datamodule, "prefetch_factor", None) if datamodule is not None else None,
            "n_train_events": n_train,
            "setup_timings": setup_timings,
            "train_step": _summary(self.train_step_times),
            "batch_fetch_gap": _summary(self.batch_fetch_times),
            "train_epoch": _summary(self.epoch_times),
            "validation_epoch": _summary(self.validation_times),
        }
        mean_epoch = payload["train_epoch"].get("mean_s") if isinstance(payload["train_epoch"], dict) else None
        if mean_epoch and n_train:
            payload["events_per_second_epoch"] = float(n_train) / float(mean_epoch)
        output_path = self._output_path(trainer)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[HYPER_TIMING] wrote {output_path}")
