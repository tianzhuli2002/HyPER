"""Shared category diagnostics for HyPER joint reconstruction/classification plots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from HyPER.topology.plot_style import (
    BACKGROUND_COLOUR,
    FM_COLOUR,
    NONFM_COLOUR,
    REFERENCE_COLOUR,
    SIGNAL_COLOUR,
    configure_matplotlib,
    decorate_axis,
    save_figure as shared_save_figure,
)


CATEGORY_LABELS = {
    "signal_fm": "signal fully matched",
    "signal_partial": "signal partially matched",
    "signal_unmatched": "signal unmatched",
    "signal_nonfm": "signal non-fully matched",
    "background": "background",
}

PREDICTION_LABEL_CANDIDATES = (
    "HyPER_CLS_T",
    "truth_label",
    "label",
    "cls_t",
)

H5_LABEL_CANDIDATES = (
    "LABELS/GLOBAL",
    "LABELS/SIGNAL",
    "LABELS/Y",
    "LABELS/CLASS",
    "LABELS/label",
)


def _binary_labels(values: pd.Series | np.ndarray) -> np.ndarray:
    labels = pd.to_numeric(pd.Series(np.asarray(values).reshape(-1)), errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(labels)):
        raise ValueError("Binary event labels contain non-finite values.")
    if not np.isin(np.unique(labels), [0, 1]).all():
        raise ValueError(f"Event labels must be exactly 0/1, found {np.unique(labels).tolist()}.")
    return labels.astype(int)


def h5_label_candidates(label_field: str | None = None) -> tuple[str, ...]:
    candidates = (label_field, *H5_LABEL_CANDIDATES) if label_field else H5_LABEL_CANDIDATES
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return tuple(out)


def flatten_labels(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if getattr(arr.dtype, "names", None):
        arr = arr[arr.dtype.names[0]]
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(arr.shape[0], -1)[:, 0]


def binary_labels_from_h5_data(
    data: dict[str, np.ndarray],
    label_field: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    for candidate in h5_label_candidates(label_field):
        if candidate in data:
            labels = flatten_labels(data[candidate])
            return _binary_labels(labels), candidate
    return None, None


def binary_labels_from_h5_handle(
    handle: Any,
    start: int,
    stop: int,
    label_field: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    for candidate in h5_label_candidates(label_field):
        if candidate in handle:
            labels = flatten_labels(handle[candidate][start:stop])
            return _binary_labels(labels), candidate
    return None, None


def binary_labels_from_prediction(
    predictions: pd.DataFrame,
    label_field: str | None = None,
) -> tuple[np.ndarray | None, str | None]:
    candidates = (label_field, *PREDICTION_LABEL_CANDIDATES) if label_field else PREDICTION_LABEL_CANDIDATES
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in predictions.columns:
            return _binary_labels(predictions[candidate]), f"prediction:{candidate}"
    return None, None


def resolve_event_labels(
    predictions: pd.DataFrame,
    h5_labels: np.ndarray | None,
    h5_label_source: str | None,
    label_field: str | None,
    warnings_list: list[str],
    context: str = "",
) -> tuple[np.ndarray | None, str | None]:
    prediction_labels, prediction_source = binary_labels_from_prediction(predictions, label_field)
    if prediction_labels is not None:
        n = len(prediction_labels)
        if h5_labels is not None:
            h5_slice = np.asarray(h5_labels[:n], dtype=int)
            if len(h5_slice) != n:
                raise ValueError("Prediction and H5 event-label arrays have different lengths.")
            disagreements = int(np.sum(prediction_labels != h5_slice))
            if disagreements:
                raise ValueError(
                    f"Prediction labels from {prediction_source} disagree with H5 labels "
                    f"from {h5_label_source} for {disagreements}/{len(h5_slice)} rows"
                    f"{' in ' + context if context else ''}."
                )
            return prediction_labels, prediction_source
        return prediction_labels, prediction_source

    if h5_labels is not None:
        return h5_labels, h5_label_source
    raise ValueError(
        "No explicit event-level classification label is available in the prediction or H5 input. "
        "Reconstruction matching is never a signal/background-label fallback."
    )


def category_masks(evaluation: pd.DataFrame, min_jets: int | None = None) -> dict[str, np.ndarray]:
    is_signal = evaluation["is_signal"].to_numpy(dtype=int) == 1
    fully_matched = evaluation["fully_matched"].to_numpy(dtype=int) == 1
    jet_selection = np.ones(len(evaluation), dtype=bool)
    if min_jets is not None:
        jet_selection = evaluation["n_jets"].to_numpy(dtype=int) >= int(min_jets)
    if "n_truth_roles_matched" in evaluation.columns:
        n_truth_roles_matched = evaluation["n_truth_roles_matched"].to_numpy(dtype=int)
        signal_partial = is_signal & ~fully_matched & (n_truth_roles_matched > 0) & jet_selection
        signal_unmatched = is_signal & ~fully_matched & (n_truth_roles_matched == 0) & jet_selection
    else:
        signal_partial = is_signal & ~fully_matched & jet_selection
        signal_unmatched = np.zeros(len(evaluation), dtype=bool)
    signal_nonfm = signal_partial | signal_unmatched
    return {
        "signal_fm": is_signal & fully_matched & jet_selection,
        "signal_partial": signal_partial,
        "signal_unmatched": signal_unmatched,
        "signal_nonfm": signal_nonfm,
        "background": ~is_signal & jet_selection,
    }


def category_summary_from_counts(
    n_total: int,
    n_signal: int,
    n_background: int,
    n_signal_fm: int,
    n_signal_nonfm: int,
    n_reco_eval_event: int,
    n_signal_partial: int | None = None,
    n_signal_unmatched: int | None = None,
    label_source: str | None = None,
    fallback_fully_matched_used: bool = False,
) -> dict[str, Any]:
    if n_signal_partial is None:
        n_signal_partial = n_signal_nonfm
    if n_signal_unmatched is None:
        n_signal_unmatched = 0
    return {
        "label_source": label_source,
        "fallback_fully_matched_used": bool(fallback_fully_matched_used),
        "n_total": int(n_total),
        "n_total_rows": int(n_total),
        "n_signal": int(n_signal),
        "n_background": int(n_background),
        "n_fully_matched_signal": int(n_signal_fm),
        "n_non_fully_matched_signal": int(n_signal_nonfm),
        "n_partially_matched_signal": int(n_signal_partial),
        "n_unmatched_signal": int(n_signal_unmatched),
        "n_reco_eval": int(n_reco_eval_event),
        "n_signal_fm": int(n_signal_fm),
        "n_signal_nonfm": int(n_signal_nonfm),
        "n_signal_partial": int(n_signal_partial),
        "n_signal_unmatched": int(n_signal_unmatched),
        "n_reco_eval_event": int(n_reco_eval_event),
        "signal_fraction": float(n_signal / n_total) if n_total else None,
        "background_fraction": float(n_background / n_total) if n_total else None,
        "fully_matched_fraction_among_signal": (
            float(n_signal_fm / n_signal) if n_signal else None
        ),
        "partially_matched_fraction_among_signal": (
            float(n_signal_partial / n_signal) if n_signal else None
        ),
        "unmatched_fraction_among_signal": (
            float(n_signal_unmatched / n_signal) if n_signal else None
        ),
        "fm_fraction_among_signal": (
            float(n_signal_fm / n_signal) if n_signal else None
        ),
        "nonfm_fraction_among_signal": (
            float(n_signal_nonfm / n_signal) if n_signal else None
        ),
        "background_fraction_among_all_rows": (
            float(n_background / n_total) if n_total else None
        ),
    }


def save_figure(output_dir: Path, stem: str, formats: list[str]) -> None:
    shared_save_figure(plt.gcf(), output_dir, stem, formats)


def write_category_diagnostics(
    summary: dict[str, Any],
    output_dir: Path,
    formats: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count_rows = [
        ("signal_fm", int(summary["n_signal_fm"])),
        ("signal_partial", int(summary.get("n_signal_partial", summary["n_signal_nonfm"]))),
        ("signal_unmatched", int(summary.get("n_signal_unmatched", 0))),
        ("background", int(summary["n_background"])),
    ]
    pd.DataFrame(count_rows, columns=["category", "count"]).to_csv(
        output_dir / "category_counts.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "fm_fraction_among_signal": summary["fm_fraction_among_signal"],
                "nonfm_fraction_among_signal": summary["nonfm_fraction_among_signal"],
                "partial_fraction_among_signal": summary.get(
                    "partially_matched_fraction_among_signal"
                ),
                "unmatched_fraction_among_signal": summary.get(
                    "unmatched_fraction_among_signal"
                ),
                "background_fraction_among_all_rows": summary[
                    "background_fraction_among_all_rows"
                ],
            }
        ]
    ).to_csv(output_dir / "category_fractions.csv", index=False)
    with (output_dir / "category_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    plt.figure(figsize=(7.0, 4.8))
    labels = [CATEGORY_LABELS[key] for key, _ in count_rows]
    counts = [value for _, value in count_rows]
    bars = plt.bar(labels, counts, color=["tab:orange", "tab:green", "tab:red", "tab:blue"])
    for bar, value in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
        )
    plt.ylabel("Rows")
    plt.title("Reconstruction category counts")
    plt.xticks(rotation=12, ha="right")
    plt.tight_layout()
    save_figure(output_dir, "category_counts", formats)


def _binary_roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    keep = np.isfinite(scores) & np.isin(labels, [0, 1])
    labels = labels[keep].astype(int)
    scores = scores[keep].astype(float)
    n_signal = int(np.sum(labels == 1))
    n_background = int(np.sum(labels == 0))
    if n_signal == 0 or n_background == 0:
        return np.asarray([]), np.asarray([]), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    tpr = np.concatenate(
        ([0.0], np.cumsum(ordered_labels == 1) / n_signal, [1.0])
    )
    fpr = np.concatenate(
        ([0.0], np.cumsum(ordered_labels == 0) / n_background, [1.0])
    )
    return fpr, tpr, float(np.trapz(tpr, fpr))


def plot_joint_sb(
    evaluation: pd.DataFrame,
    score_field: str,
    output_dir: Path,
    formats: list[str],
    *,
    topology_label: str,
    plot_set: str = "essential",
) -> dict[str, Any]:
    configure_matplotlib()
    if score_field not in evaluation.columns:
        return {"available": False, "reason": f"missing score field {score_field}"}
    scores = evaluation[score_field].to_numpy(dtype=float)
    finite_scores = np.isfinite(scores)
    if not np.any(finite_scores):
        return {"available": False, "reason": f"score field {score_field} is all NaN"}

    masks = category_masks(evaluation)
    is_signal = evaluation["is_signal"].to_numpy(dtype=int) == 1
    is_background = ~is_signal
    bins = np.linspace(0.0, 1.0, 41)

    def filled_hist(ax, mask: np.ndarray, label: str, colour: str) -> None:
        values = scores[mask & finite_scores]
        if not len(values):
            return
        ax.hist(values, bins=bins, density=True, histtype="stepfilled",
                color=colour, alpha=0.20, linewidth=0)
        ax.hist(values, bins=bins, density=True, histtype="step",
                color=colour, linewidth=2.0, label=f"{label} ($N={len(values):,}$)")

    if plot_set == "full":
        fig, ax = plt.subplots(figsize=(7.4, 5.5))
        filled_hist(ax, is_background, "Background", BACKGROUND_COLOUR)
        filled_hist(ax, is_signal, "All signal", SIGNAL_COLOUR)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("HyPER classification score")
        ax.set_ylabel("Probability density")
        decorate_axis(ax, title=f"{topology_label} — all signal vs background")
        ax.legend(loc="best")
        fig.tight_layout()
        shared_save_figure(fig, output_dir, "classification_scores_all_signal", formats)

    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    filled_hist(ax, masks["background"], "Background", BACKGROUND_COLOUR)
    filled_hist(ax, masks["signal_fm"], "Fully matched signal", FM_COLOUR)
    filled_hist(ax, masks["signal_nonfm"], "Not fully matched signal", NONFM_COLOUR)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("HyPER classification score")
    ax.set_ylabel("Probability density")
    decorate_axis(ax, title=f"{topology_label} — classification score")
    ax.legend(loc="best")
    fig.tight_layout()
    shared_save_figure(fig, output_dir, "classification_scores", formats)

    all_labels = is_signal.astype(int)
    fpr_all, tpr_all, auc_all = _binary_roc(all_labels, scores)
    fm_or_background = masks["signal_fm"] | masks["background"]
    fm_labels = masks["signal_fm"][fm_or_background].astype(int)
    fm_scores = scores[fm_or_background]
    fpr_fm, tpr_fm, auc_fm = _binary_roc(fm_labels, fm_scores)
    nonfm_or_background = masks["signal_nonfm"] | masks["background"]
    nonfm_labels = masks["signal_nonfm"][nonfm_or_background].astype(int)
    nonfm_scores = scores[nonfm_or_background]
    fpr_nonfm, tpr_nonfm, auc_nonfm = _binary_roc(nonfm_labels, nonfm_scores)

    if plot_set == "full":
        roc_specs = (
            ("classifier_roc_all_signal", fpr_all, tpr_all, auc_all,
             "Signal efficiency", f"{topology_label} — all signal vs background", SIGNAL_COLOUR),
            ("classifier_roc_fully_matched", fpr_fm, tpr_fm, auc_fm,
             "Fully matched signal efficiency", f"{topology_label} — fully matched signal", FM_COLOUR),
            ("classifier_roc_not_fully_matched", fpr_nonfm, tpr_nonfm, auc_nonfm,
             "Not fully matched signal efficiency", f"{topology_label} — not fully matched signal", NONFM_COLOUR),
        )
        for stem, fpr, tpr, auc, ylabel, title, colour in roc_specs:
            if not len(fpr):
                continue
            fig, ax = plt.subplots(figsize=(6.3, 5.8))
            ax.plot(fpr, tpr, color=colour, linewidth=2.2, label=f"AUC={auc:.4f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color=REFERENCE_COLOUR, linewidth=1.2)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Background efficiency")
            ax.set_ylabel(ylabel)
            decorate_axis(ax, title=title)
            ax.legend(loc="lower right")
            fig.tight_layout()
            shared_save_figure(fig, output_dir, stem, formats)

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    for fpr, tpr, auc, label, colour in (
        (fpr_all, tpr_all, auc_all, "All signal", SIGNAL_COLOUR),
        (fpr_fm, tpr_fm, auc_fm, "Fully matched signal", FM_COLOUR),
        (fpr_nonfm, tpr_nonfm, auc_nonfm, "Not fully matched signal", NONFM_COLOUR),
    ):
        if len(fpr):
            ax.plot(fpr, tpr, color=colour, linewidth=2.0, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=REFERENCE_COLOUR, linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    decorate_axis(ax, title=f"{topology_label} — classification ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    shared_save_figure(fig, output_dir, "classifier_roc", formats)

    result = {
        "available": True,
        "score_field": score_field,
        "score_scope": [
            "all_signal_vs_background",
            "signal_fm_vs_signal_nonfm_vs_background",
        ],
        "n_rows_with_finite_score": int(np.sum(finite_scores)),
        "n_signal_with_finite_score": int(np.sum(is_signal & finite_scores)),
        "n_background_with_finite_score": int(np.sum(is_background & finite_scores)),
        "n_signal_fm_with_finite_score": int(np.sum(masks["signal_fm"] & finite_scores)),
        "n_signal_nonfm_with_finite_score": int(np.sum(masks["signal_nonfm"] & finite_scores)),
        "n_signal_partial_with_finite_score": int(
            np.sum(masks.get("signal_partial", np.zeros(len(evaluation), dtype=bool)) & finite_scores)
        ),
        "n_signal_unmatched_with_finite_score": int(
            np.sum(masks.get("signal_unmatched", np.zeros(len(evaluation), dtype=bool)) & finite_scores)
        ),
        "auc_all_signal_vs_background": auc_all if math.isfinite(auc_all) else None,
        "auc_fm_signal_vs_background": auc_fm if math.isfinite(auc_fm) else None,
        "auc_nonfm_signal_vs_background": auc_nonfm if math.isfinite(auc_nonfm) else None,
    }
    with (output_dir / "sb_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result

def plot_observable_pair(
    evaluation: pd.DataFrame,
    column: str,
    output_name: str,
    xlabel: str,
    title: str,
    min_jets: int,
    output_dir: Path,
    formats: list[str],
    *,
    include_category_split: bool = True,
) -> dict[str, int]:
    values = evaluation[column].to_numpy(dtype=float)
    masks = category_masks(evaluation, min_jets=min_jets)
    finite_values = np.isfinite(values)

    fm = values[masks["signal_fm"] & finite_values]
    if len(fm):
        plt.figure(figsize=(6.2, 4.4))
        plt.hist(fm, bins=50, histtype="step", linewidth=1.5, color="tab:orange")
        plt.xlabel(xlabel)
        plt.ylabel("Events")
        plt.title(f"{title} (fully matched signal)")
        plt.tight_layout()
        save_figure(output_dir, f"observable_{output_name}_fm_only", formats)

    if include_category_split:
        plt.figure(figsize=(6.2, 4.4))
        plotted = False
        for key, color in (
            ("signal_fm", "tab:orange"),
            ("signal_partial", "tab:green"),
            ("signal_unmatched", "tab:red"),
            ("background", "tab:blue"),
        ):
            scoped = values[masks[key] & finite_values]
            if not len(scoped):
                continue
            plotted = True
            plt.hist(
                scoped,
                bins=50,
                histtype="step",
                linewidth=1.5,
                label=CATEGORY_LABELS[key],
                color=color,
            )
        if plotted:
            plt.xlabel(xlabel)
            plt.ylabel("Events")
            plt.title(f"{title} (diagnostic category comparison)")
            plt.legend()
            plt.tight_layout()
            save_figure(output_dir, f"observable_{output_name}_category_split", formats)
        else:
            plt.close()

    return {
        "fm_only": int(np.sum(masks["signal_fm"] & finite_values)),
        "category_split": (
            int(
                np.sum(
                    (masks["signal_fm"] | masks["signal_nonfm"] | masks["background"])
                    & finite_values
                )
            )
            if include_category_split
            else 0
        ),
    }
