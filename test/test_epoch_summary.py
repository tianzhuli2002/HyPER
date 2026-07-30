from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from lightning.pytorch.utilities import rank_zero_only

from HyPER.utils.epoch_summary import PersistentEpochSummary


def trainer(metrics, epoch=3, sanity=False):
    optimizer = SimpleNamespace(param_groups=[{"lr": 1.25e-4}])
    return SimpleNamespace(
        callback_metrics={name: torch.tensor(value) for name, value in metrics.items()},
        current_epoch=epoch, sanity_checking=sanity, optimizers=[optimizer],
    )


def emit(callback, instance):
    with patch("builtins.print") as printer:
        callback.on_validation_end(instance, None)
        callback.on_train_epoch_end(instance, None)
    return printer


class EpochSummaryTest(unittest.TestCase):
    def test_classification_summary_is_one_flushed_newline_line(self):
        printer = emit(PersistentEpochSummary(), trainer({
            "loss/train_loss_epoch": 0.34, "val_loss": 0.35,
            "val_classification_loss": 0.42, "val_auc": 0.889,
        }, epoch=38))
        printer.assert_called_once()
        line = printer.call_args.args[0]
        self.assertTrue(line.startswith("Epoch 038 | train_loss=0.340000 | val_loss=0.350000"))
        self.assertIn("classification_loss=0.420000", line)
        self.assertIn("val_auc=0.889000", line)
        self.assertEqual(printer.call_args.kwargs, {"flush": True})
        self.assertNotIn("\n", line)

    def test_reconstruction_and_joint_fields_and_unavailable_omission(self):
        callback = PersistentEpochSummary()
        reconstruction = emit(callback, trainer({
            "val_loss": 0.08, "val_edge_loss": 0.05, "val_hyperedge_loss": 0.15,
            "val_reco_mean_role_top1": 0.77,
        }))
        line = reconstruction.call_args.args[0]
        for field in ("edge_loss", "hyperedge_loss", "reco_mean_role_top1"):
            self.assertIn(f"{field}=", line)
        self.assertNotIn("classification_loss=", line)
        self.assertNotIn("val_auc=", line)
        joint = emit(callback, trainer({
            "val_loss": 0.09, "val_edge_loss": 0.06, "val_hyperedge_loss": 0.16,
            "val_classification_loss": 0.4, "val_auc": 0.88, "val_reco_mean_role_top1": 0.75,
        }))
        for field in ("edge_loss", "hyperedge_loss", "classification_loss", "val_auc", "reco_mean_role_top1"):
            self.assertIn(f"{field}=", joint.call_args.args[0])

    def test_sanity_validation_emits_nothing_and_resume_epoch_is_used(self):
        callback = PersistentEpochSummary()
        self.assertEqual(emit(callback, trainer({"val_loss": 1.0}, sanity=True)).call_count, 0)
        printer = emit(callback, trainer({"val_loss": 1.0}, epoch=17))
        self.assertTrue(printer.call_args.args[0].startswith("Epoch 017 |"))

    def test_rank_zero_decorator_suppresses_nonzero_rank(self):
        callback = PersistentEpochSummary()
        previous = rank_zero_only.rank
        try:
            rank_zero_only.rank = 1
            self.assertEqual(emit(callback, trainer({"val_loss": 1.0})).call_count, 0)
        finally:
            rank_zero_only.rank = previous


if __name__ == "__main__":
    unittest.main()
