"""Fixed, label-free event reconstruction confidence definitions."""

from __future__ import annotations

import numpy as np


def ttbar_sl_event_reconstruction_score(component_scores) -> float:
    """Product of the four selected ttbar-SL reconstruction-role probabilities."""
    values = np.asarray(component_scores, dtype=float).reshape(-1)
    if values.size != 4 or not np.isfinite(values).all():
        return float("nan")
    return float(np.prod(values))
