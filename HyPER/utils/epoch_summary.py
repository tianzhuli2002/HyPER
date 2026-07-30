"""Durable, newline-terminated Lightning epoch summaries for batch logs."""

from __future__ import annotations

import math

from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only


class PersistentEpochSummary(Callback):
    """Print one aggregated summary after each completed validation epoch."""

    METRICS = (
        ("loss/train_loss_epoch", "train_loss"),
        ("loss/train_loss", "train_loss"),
        ("val_loss", "val_loss"),
        ("val_edge_loss", "edge_loss"),
        ("val_hyperedge_loss", "hyperedge_loss"),
        ("val_classification_loss", "classification_loss"),
        ("val_auc", "val_auc"),
        ("val_reco_mean_role_top1", "reco_mean_role_top1"),
    )

    @staticmethod
    def _finite_scalar(value):
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            if hasattr(value, "numel") and value.numel() != 1:
                return None
            result = float(value)
        except (TypeError, ValueError, RuntimeError):
            return None
        return result if math.isfinite(result) else None

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        self._completed_validation_epoch = int(trainer.current_epoch)

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if getattr(self, "_completed_validation_epoch", None) != int(trainer.current_epoch):
            return
        self._completed_validation_epoch = None
        fields = []
        used = set()
        for metric_name, label in self.METRICS:
            if label in used or metric_name not in trainer.callback_metrics:
                continue
            value = self._finite_scalar(trainer.callback_metrics[metric_name])
            if value is not None:
                fields.append(f"{label}={value:.6f}")
                used.add(label)
        optimizers = getattr(trainer, "optimizers", ())
        if optimizers and optimizers[0].param_groups:
            learning_rate = self._finite_scalar(optimizers[0].param_groups[0].get("lr"))
            if learning_rate is not None:
                fields.append(f"lr={learning_rate:.3e}")
        print(f"Epoch {int(trainer.current_epoch):03d} | " + " | ".join(fields), flush=True)
