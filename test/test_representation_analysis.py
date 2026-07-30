import numpy as np
import unittest

from HyPER.analysis.representations import (
    align_exports,
    apply_procrustes,
    fit_procrustes,
    linear_cka,
)


class RepresentationAnalysisTest(unittest.TestCase):
    def test_linear_cka_identity_orthogonal_symmetry_and_range(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(200, 12))
        y = rng.normal(size=(200, 8))
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        self.assertAlmostEqual(linear_cka(x, x), 1.0, places=12)
        self.assertAlmostEqual(linear_cka(x, x @ q), 1.0, places=12)
        self.assertAlmostEqual(linear_cka(x, y), linear_cka(y, x), places=12)
        self.assertTrue(0.0 <= linear_cka(x, y) <= 1.0)

    def test_linear_cka_rejects_degenerate_input(self):
        with self.assertRaisesRegex(ValueError, "zero-variance"):
            linear_cka(np.ones((20, 3)), np.arange(60).reshape(20, 3))

    def test_event_alignment_reorders_and_rejects_mismatch(self):
        left = {
            "source_event_index": np.asarray([4, 1, 9]),
            "truth_class": np.asarray([0, 1, 1]),
            "truth_fully_matched": np.asarray([0, 1, 0]),
        }
        right = {
            "source_event_index": np.asarray([9, 4, 1]),
            "truth_class": np.asarray([1, 0, 1]),
            "truth_fully_matched": np.asarray([0, 0, 1]),
        }
        indices, _, right_rows = align_exports(left, right)
        np.testing.assert_array_equal(indices, [4, 1, 9])
        np.testing.assert_array_equal(right_rows, [1, 2, 0])
        bad = dict(right, source_event_index=np.asarray([9, 4, 2]))
        with self.assertRaisesRegex(ValueError, "sets differ"):
            align_exports(left, bad)
        common, _, _ = align_exports(left, bad, common_events_only=True)
        np.testing.assert_array_equal(common, [4, 9])

    def test_procrustes_recovers_held_out_orthogonal_map_and_shuffle_is_worse(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(600, 16))
        q, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        translation = rng.normal(size=16)
        y = x @ q + translation
        fitted = fit_procrustes(x[:500], y[:500])
        aligned = apply_procrustes(
            x[500:], fitted["source_mean"], fitted["target_mean"], fitted["rotation"]
        )
        residual = np.linalg.norm(aligned - y[500:]) / np.linalg.norm(y[500:] - y[500:].mean(0))
        self.assertLess(fitted["orthogonality_error"], 1e-10)
        self.assertLess(fitted["normalised_residual"], 1e-12)
        self.assertLess(residual, 1e-12)
        shuffled = fit_procrustes(x[:500], y[rng.permutation(500)])
        self.assertGreater(shuffled["normalised_residual"], fitted["normalised_residual"] + 0.5)


if __name__ == "__main__":
    unittest.main()
