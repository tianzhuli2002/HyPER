"""Metrics, operating points and paired bootstrap helpers for HyPER analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


SIGNAL_EFFICIENCIES = (0.5, 0.7, 0.8)


def validate_binary_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError(f"Labels and scores differ in shape: {labels.shape} vs {scores.shape}.")
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Binary metrics require both signal and background labels.")
    return labels, scores


def operating_point(labels, scores, signal_efficiency: float) -> dict[str, float]:
    labels, scores = validate_binary_scores(labels, scores)
    signal = scores[labels == 1]
    background = scores[labels == 0]
    target = float(signal_efficiency)
    if not 0 < target < 1:
        raise ValueError("Signal efficiency must lie strictly between zero and one.")
    threshold = float(np.quantile(signal, 1.0 - target, method="linear"))
    actual_signal_efficiency = float(np.mean(signal >= threshold))
    background_efficiency = float(np.mean(background >= threshold))
    rejection = float(np.inf if background_efficiency == 0 else 1.0 / background_efficiency)
    return {
        "target_signal_efficiency": target,
        "threshold": threshold,
        "signal_efficiency": actual_signal_efficiency,
        "background_efficiency": background_efficiency,
        "background_rejection": rejection,
    }


def binary_metric_summary(labels, scores, *, include_roc: bool = True) -> dict[str, object]:
    labels, scores = validate_binary_scores(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    result: dict[str, object] = {
        "event_count": int(len(labels)),
        "signal_count": int(np.sum(labels == 1)),
        "background_count": int(np.sum(labels == 0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "operating_points": {},
    }
    if include_roc:
        result["roc"] = {
            "background_efficiency": fpr.tolist(),
            "signal_efficiency": tpr.tolist(),
            "threshold": thresholds.tolist(),
        }
    for efficiency in SIGNAL_EFFICIENCIES:
        result["operating_points"][f"signal_efficiency_{efficiency:.1f}"] = operating_point(
            labels, scores, efficiency
        )
    return result


def weighted_auc_from_order(labels_sorted, weights_sorted) -> np.ndarray:
    """Exact weighted AUC for one or many bootstrap weight vectors.

    Inputs must already be sorted by score in ascending order. Ties are handled by
    assigning half credit to signal/background pairs in the same score group.
    """
    labels = np.asarray(labels_sorted, dtype=np.int8).reshape(-1)
    weights = np.asarray(weights_sorted, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.shape[1] != len(labels):
        raise ValueError("Bootstrap weights do not match sorted labels.")
    signal = weights * (labels == 1)
    background = weights * (labels == 0)
    signal_total = signal.sum(axis=1)
    background_total = background.sum(axis=1)
    if np.any(signal_total <= 0) or np.any(background_total <= 0):
        raise ValueError("A bootstrap replicate contains no signal or no background weight.")
    cumulative_background = np.cumsum(background, axis=1)
    background_before = cumulative_background - background
    favourable = np.sum(signal * (background_before + 0.5 * background), axis=1)
    return favourable / (signal_total * background_total)


@dataclass(frozen=True)
class BootstrapStrata:
    background: np.ndarray
    signal_fully_matched: np.ndarray
    signal_non_fully_matched: np.ndarray


def bootstrap_strata(labels, fully_matched) -> BootstrapStrata:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    fully_matched = np.asarray(fully_matched, dtype=bool).reshape(-1)
    if labels.shape != fully_matched.shape:
        raise ValueError("labels and fully_matched must have the same shape.")
    return BootstrapStrata(
        background=np.flatnonzero(labels == 0),
        signal_fully_matched=np.flatnonzero((labels == 1) & fully_matched),
        signal_non_fully_matched=np.flatnonzero((labels == 1) & ~fully_matched),
    )


def poisson_stratified_weights(
    event_count: int,
    strata: BootstrapStrata,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate paired stratified Poisson(1) bootstrap weights.

    Each category is generated independently. Empty or zero-total strata are
    regenerated so every replicate retains all requested categories.
    """
    weights = np.zeros((int(replicates), int(event_count)), dtype=np.float32)
    for indices in (strata.background, strata.signal_fully_matched, strata.signal_non_fully_matched):
        if len(indices) == 0:
            raise ValueError("A requested bootstrap stratum is empty.")
        block = rng.poisson(1.0, size=(int(replicates), len(indices))).astype(np.float32)
        zero = np.flatnonzero(block.sum(axis=1) == 0)
        while len(zero):
            block[zero] = rng.poisson(1.0, size=(len(zero), len(indices))).astype(np.float32)
            zero = zero[block[zero].sum(axis=1) == 0]
        weights[:, indices] = block
    return weights


def prepare_descending_score_groups(labels, scores) -> dict[str, np.ndarray]:
    labels, scores = validate_binary_scores(labels, scores)
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1].astype(np.int64)
    return {
        "order": order.astype(np.int64),
        "labels": sorted_labels,
        "scores": sorted_scores,
        "group_starts": starts,
        "group_scores": sorted_scores[starts],
    }


def weighted_grouped_metrics(
    prepared: dict[str, np.ndarray],
    weights,
    signal_efficiencies=SIGNAL_EFFICIENCIES,
) -> dict[str, np.ndarray]:
    """Exact weighted AUC and rejection metrics for pre-sorted score groups."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    order = prepared["order"]
    if weights.shape[1] != len(order):
        raise ValueError("Bootstrap weights do not match prepared score events.")
    sorted_weights = weights[:, order]
    labels = prepared["labels"]
    starts = prepared["group_starts"]
    signal_group = np.add.reduceat(sorted_weights * (labels == 1), starts, axis=1)
    background_group = np.add.reduceat(sorted_weights * (labels == 0), starts, axis=1)
    signal_total = signal_group.sum(axis=1)
    background_total = background_group.sum(axis=1)
    if np.any(signal_total <= 0) or np.any(background_total <= 0):
        raise ValueError("A bootstrap replicate contains no signal or no background weight.")
    cumulative_signal = np.cumsum(signal_group, axis=1)
    cumulative_background = np.cumsum(background_group, axis=1)
    background_below = background_total[:, None] - cumulative_background
    favourable = np.sum(signal_group * (background_below + 0.5 * background_group), axis=1)
    result = {"roc_auc": favourable / (signal_total * background_total)}
    group_scores = prepared["group_scores"]
    rows = np.arange(weights.shape[0])
    for efficiency in signal_efficiencies:
        targets = float(efficiency) * signal_total
        reached = cumulative_signal >= targets[:, None]
        indices = np.argmax(reached, axis=1)
        background_efficiency = cumulative_background[rows, indices] / background_total
        result[f"threshold_at_signal_efficiency_{efficiency:.1f}"] = group_scores[indices]
        result[f"background_efficiency_at_signal_efficiency_{efficiency:.1f}"] = background_efficiency
        result[f"background_rejection_at_signal_efficiency_{efficiency:.1f}"] = np.divide(
            1.0,
            background_efficiency,
            out=np.full_like(background_efficiency, np.inf),
            where=background_efficiency > 0,
        )
    return result
