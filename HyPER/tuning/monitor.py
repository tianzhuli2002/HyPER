"""Best-observed validation objective and Optuna pruning callback."""

from __future__ import annotations

import math

import torch
from lightning.pytorch.callbacks import Callback


class BestObservedValidation(Callback):
    def __init__(self, monitor: str, mode: str, trial=None):
        super().__init__()
        self.monitor = str(monitor)
        self.mode = str(mode)
        if self.mode not in {"min", "max"}:
            raise ValueError("Monitor mode must be 'min' or 'max'.")
        self.trial = trial
        self.observations: list[dict] = []
        self.best_metrics: dict = {}
        self.stopped_early = False
        self.stopping_epoch = None
        self.pruned = False
        self.pruning_epoch = None

    @staticmethod
    def _scalar(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("Monitored validation metric must be scalar.")
            return float(value.detach().cpu())
        return float(value)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self.monitor not in trainer.callback_metrics:
            available = sorted(str(key) for key in trainer.callback_metrics.keys())
            raise RuntimeError(
                f"Required tuning monitor {self.monitor!r} was not logged. "
                f"Available callback metrics: {available}"
            )
        value = self._scalar(trainer.callback_metrics[self.monitor])
        if value is None or not math.isfinite(value):
            return
        epoch = int(trainer.current_epoch)
        if self.observations and self.observations[-1]["epoch"] == epoch:
            return
        metrics = {}
        for name, raw in trainer.callback_metrics.items():
            try:
                scalar = self._scalar(raw)
            except (TypeError, ValueError):
                continue
            if scalar is not None and math.isfinite(scalar):
                metrics[str(name)] = scalar
        self.observations.append({"epoch": epoch, "value": value, "metrics": metrics})
        if self.trial is not None:
            self.trial.report(value, step=epoch)
            if self.trial.should_prune():
                self.pruned = True
                self.pruning_epoch = epoch
                self._write_trial_attributes()
                import optuna
                raise optuna.TrialPruned(f"Pruned at validation epoch {epoch}.")

    def on_train_end(self, trainer, pl_module):
        early = getattr(trainer, "early_stopping_callback", None)
        if early is not None and bool(getattr(early, "stopped_epoch", 0)):
            self.stopped_early = True
            self.stopping_epoch = int(early.stopped_epoch)

    def summary(self) -> dict:
        if not self.observations:
            raise RuntimeError(f"No finite real-validation observations were recorded for {self.monitor!r}.")
        choose = min if self.mode == "min" else max
        best = choose(self.observations, key=lambda item: item["value"])
        final = self.observations[-1]
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_observed_monitor_value": best["value"],
            "best_observed_monitor_epoch": best["epoch"],
            "best_epoch_metrics": best["metrics"],
            "final_observed_monitor_value": final["value"],
            "final_observed_monitor_epoch": final["epoch"],
            "final_epoch_metrics": final["metrics"],
            "number_of_valid_monitor_observations": len(self.observations),
            "stopped_early": self.stopped_early,
            "stopping_epoch": self.stopping_epoch,
            "pruned": self.pruned,
            "pruning_epoch": self.pruning_epoch,
        }

    def _write_trial_attributes(self):
        if self.trial is None or not self.observations:
            return
        summary = self.summary()
        for key in (
            "best_observed_monitor_value", "best_observed_monitor_epoch",
            "final_observed_monitor_value", "final_observed_monitor_epoch",
            "number_of_valid_monitor_observations", "stopped_early", "stopping_epoch",
            "pruned", "pruning_epoch",
        ):
            self.trial.set_user_attr(key, summary[key])
