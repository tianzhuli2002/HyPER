"""Translate tuning-level alpha/beta values into production loss weights."""

from __future__ import annotations

import math


def loss_coefficients(alpha: float, beta: float) -> dict[str, float]:
    alpha = float(alpha)
    beta = float(beta)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1].")
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be finite and in [0, 1].")
    values = {
        "edge_weight": (1.0 - beta) * (1.0 - alpha),
        "hyperedge_weight": (1.0 - beta) * alpha,
        "classification_weight": beta,
    }
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ArithmeticError("Translated HyPER loss coefficients do not sum to one.")
    return values

