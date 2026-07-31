import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from HyPER.analysis.metrics import (
    bootstrap_strata,
    poisson_stratified_weights,
    prepare_descending_score_groups,
    weighted_grouped_metrics,
)
from HyPER.analysis.representations import StreamingLinearCKA, linear_cka
from HyPER.topology.plot_style import configure_matplotlib, save_figure


class RepresentationProductionTest(unittest.TestCase):
    def test_streaming_cka_matches_direct(self):
        rng = np.random.default_rng(13)
        x = rng.normal(size=(517, 17))
        y = rng.normal(size=(517, 11))
        accumulator = StreamingLinearCKA(17, 11)
        for start in range(0, len(x), 73):
            accumulator.update(x[start:start + 73], y[start:start + 73])
        self.assertAlmostEqual(accumulator.value(), linear_cka(x, y), places=12)
        self.assertEqual(accumulator.count, len(x))

    def test_weighted_grouped_auc_matches_sklearn_with_ties(self):
        rng = np.random.default_rng(21)
        labels = np.r_[np.zeros(80, dtype=np.int8), np.ones(120, dtype=np.int8)]
        scores = rng.integers(0, 15, size=len(labels)) / 14.0
        weights = rng.poisson(1.0, size=(8, len(labels))).astype(float)
        prepared = prepare_descending_score_groups(labels, scores)
        calculated = weighted_grouped_metrics(prepared, weights)["roc_auc"]
        expected = np.asarray([
            roc_auc_score(labels, scores, sample_weight=row)
            for row in weights
        ])
        np.testing.assert_allclose(calculated, expected, rtol=1e-12, atol=1e-12)

    def test_paired_stratified_poisson_bootstrap_is_deterministic(self):
        labels = np.asarray([0] * 20 + [1] * 12 + [1] * 8, dtype=np.int8)
        fm = np.asarray([0] * 20 + [1] * 12 + [0] * 8, dtype=bool)
        strata = bootstrap_strata(labels, fm)
        first = poisson_stratified_weights(40, strata, 7, np.random.default_rng(42))
        second = poisson_stratified_weights(40, strata, 7, np.random.default_rng(42))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (7, 40))
        self.assertTrue(np.all(first[:, strata.background].sum(axis=1) > 0))
        self.assertTrue(np.all(first[:, strata.signal_fully_matched].sum(axis=1) > 0))
        self.assertTrue(np.all(first[:, strata.signal_non_fully_matched].sum(axis=1) > 0))

    def test_alignment_control_tool_writes_unique_shuffles_without_loading_labels(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(60, 6))
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        target = source @ q + rng.normal(size=6)
        indices = np.arange(60, dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.npz"
            target_path = root / "target.npz"
            np.savez_compressed(
                source_path,
                source_event_index=indices,
                final_event=source.astype(np.float32),
                truth_class=np.ones(60, dtype=np.int8),
            )
            np.savez_compressed(
                target_path,
                source_event_index=indices,
                classification_head_input=target.astype(np.float32),
                truth_class=np.zeros(60, dtype=np.int8),
            )
            output = root / "alignment"
            command = [
                sys.executable,
                "tools/fit_hyper_alignment_controls.py",
                "--source", str(source_path),
                "--target", str(target_path),
                "--direction", "synthetic",
                "--num-shuffles", "3",
                "--expected-event-count", "60",
                "--output-dir", str(output),
            ]
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            reused = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("REUSE verified paired alignment", reused.stdout)
            self.assertEqual(reused.stdout.count("REUSE verified shuffled alignment"), 3)
            summary = json.loads((output / "alignment_summary.json").read_text())
            self.assertFalse(summary["labels_loaded_for_fitting"])
            self.assertEqual(summary["num_shuffled_alignments"], 3)
            hashes = [item["target_permutation_hash"] for item in summary["shuffled"]]
            self.assertEqual(len(hashes), len(set(hashes)))
            self.assertTrue((output / "paired.npz").is_file())

    def test_shared_plot_style_writes_pdf_and_png(self):
        import matplotlib.pyplot as plt

        configure_matplotlib()
        with tempfile.TemporaryDirectory() as directory:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            save_figure(fig, directory, "test")
            self.assertGreater((Path(directory) / "test.pdf").stat().st_size, 0)
            self.assertGreater((Path(directory) / "test.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
