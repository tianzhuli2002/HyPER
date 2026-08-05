"""Fixed, label-free event reconstruction confidence definitions."""

from __future__ import annotations

import numpy as np


_TOPOLOGY_COMPONENT_COUNTS = {
    "ttbar1L": 4,
    "ttH": 5,
}


def normalise_reconstruction_topology(topology: str) -> str:
    """Return the canonical representation-transfer topology name."""
    value = str(topology).strip()
    aliases = {
        "ttbar1l": "ttbar1L",
        "ttbar_single_lep": "ttbar1L",
        "tth": "ttH",
    }
    canonical = aliases.get(value.lower(), value)
    if canonical not in {"ttbar1L", "ttH"}:
        raise ValueError(
            f"Unsupported reconstruction topology {topology!r}; expected ttbar1L or ttH."
        )
    return canonical


def required_truth_role_ids(topology: str) -> frozenset[int]:
    """Truth role IDs required for intrinsic full reconstructibility."""
    canonical = normalise_reconstruction_topology(topology)
    return {
        "ttbar1L": frozenset({1, 2, 3, 4}),
        "ttH": frozenset({1, 2, 3, 4, 5, 6}),
    }[canonical]


def event_reconstruction_score(component_scores, topology: str) -> float:
    """Product of the selected reconstruction-role probabilities.

    ttbar1L uses ``(top1, top2, W1, W2)``.  ttH uses
    ``(tlep, thad, Wlep, Whad, H)``.  The definition is fixed and label-free.
    """
    canonical = normalise_reconstruction_topology(topology)
    expected = _TOPOLOGY_COMPONENT_COUNTS[canonical]
    values = np.asarray(component_scores, dtype=float).reshape(-1)
    if values.size != expected or not np.isfinite(values).all():
        return float("nan")
    return float(np.prod(values))


def ttbar_sl_event_reconstruction_score(component_scores) -> float:
    """Product of the four selected ttbar-SL reconstruction-role probabilities."""
    return event_reconstruction_score(component_scores, "ttbar1L")


def tth_event_reconstruction_score(component_scores) -> float:
    """Product of the five selected ttH reconstruction-role probabilities."""
    return event_reconstruction_score(component_scores, "ttH")
