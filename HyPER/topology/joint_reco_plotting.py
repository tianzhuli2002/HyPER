"""Shared category diagnostics for HyPER joint reconstruction/classification plots."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    return np.where(labels > 0.5, 1, 0).astype(int)


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
            disagreements = int(np.sum(prediction_labels[: len(h5_slice)] != h5_slice))
            if disagreements:
                frac = disagreements / max(1, len(h5_slice))
                message = (
                    f"Prediction labels from {prediction_source} disagree with H5 labels "
                    f"from {h5_label_source} for {disagreements}/{len(h5_slice)} rows"
                    f"{' in ' + context if context else ''}."
                )
                if frac > 0.01:
                    raise ValueError(message)
                warnings_list.append(message)
            return prediction_labels, prediction_source
        return prediction_labels, prediction_source

    if h5_labels is not None:
        return h5_labels, h5_label_source
    return None, None


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
    for fmt in formats:
        plt.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close()


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
    title = "Reconstruction category counts"
    if summary.get("fallback_fully_matched_used"):
        title += " (fallback labels)"
    plt.title(title)
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
) -> dict[str, Any]:
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

    plt.figure(figsize=(6.4, 4.5))
    for mask, label, color in (
        (is_signal, "signal", "tab:orange"),
        (is_background, "background", "tab:blue"),
    ):
        values = scores[mask & finite_scores]
        if len(values):
            plt.hist(values, bins=bins, histtype="step", density=True, label=label, color=color)
    plt.xlabel(score_field)
    plt.ylabel("Density")
    plt.title("Classifier score: all signal vs background")
    plt.legend()
    plt.tight_layout()
    save_figure(output_dir, "sb_score_all_signal_vs_background", formats)

    plt.figure(figsize=(6.4, 4.5))
    for key, color in (
        ("signal_fm", "tab:orange"),
        ("signal_nonfm", "tab:green"),
        ("background", "tab:blue"),
    ):
        values = scores[masks[key] & finite_scores]
        if len(values):
            plt.hist(
                values,
                bins=bins,
                histtype="step",
                density=True,
                label=CATEGORY_LABELS[key],
                color=color,
            )
    plt.xlabel(score_field)
    plt.ylabel("Density")
    plt.title("Classifier score by reconstruction category")
    plt.legend()
    plt.tight_layout()
    save_figure(output_dir, "sb_score_signal_fm_nonfm_background", formats)

    all_labels = is_signal.astype(int)
    fpr_all, tpr_all, auc_all = _binary_roc(all_labels, scores)
    if len(fpr_all):
        plt.figure(figsize=(5.4, 5.2))
        plt.plot(fpr_all, tpr_all, label=f"AUC={auc_all:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="0.5")
        plt.xlabel("Background efficiency")
        plt.ylabel("Signal efficiency")
        plt.title("ROC: all signal vs background")
        plt.legend(loc="lower right")
        plt.tight_layout()
        save_figure(output_dir, "sb_roc_all_signal_vs_background", formats)

    fm_or_background = masks["signal_fm"] | masks["background"]
    fm_labels = masks["signal_fm"][fm_or_background].astype(int)
    fm_scores = scores[fm_or_background]
    fpr_fm, tpr_fm, auc_fm = _binary_roc(fm_labels, fm_scores)
    if len(fpr_fm):
        plt.figure(figsize=(5.4, 5.2))
        plt.plot(fpr_fm, tpr_fm, label=f"AUC={auc_fm:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="0.5")
        plt.xlabel("Background efficiency")
        plt.ylabel("FM signal efficiency")
        plt.title("ROC: fully matched signal vs background")
        plt.legend(loc="lower right")
        plt.tight_layout()
        save_figure(output_dir, "sb_roc_fm_signal_vs_background", formats)

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
        "n_signal_nonfm_with_finite_score": int(
            np.sum(masks["signal_nonfm"] & finite_scores)
        ),
        "n_signal_partial_with_finite_score": int(
            np.sum(masks.get("signal_partial", np.zeros(len(evaluation), dtype=bool)) & finite_scores)
        ),
        "n_signal_unmatched_with_finite_score": int(
            np.sum(masks.get("signal_unmatched", np.zeros(len(evaluation), dtype=bool)) & finite_scores)
        ),
        "auc_all_signal_vs_background": (
            auc_all if math.isfinite(auc_all) else None
        ),
        "auc_fm_signal_vs_background": auc_fm if math.isfinite(auc_fm) else None,
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
        "category_split": int(
            np.sum(
                (masks["signal_fm"] | masks["signal_nonfm"] | masks["background"])
                & finite_values
            )
        ),
    }
