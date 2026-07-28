#!/usr/bin/env python3
"""Lightweight S/B classifier-only validation plots for HyPER predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prediction_io import load_hyper_prediction_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--score-field", default="HyPER_CLS_PROB")
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    return parser.parse_args()


def finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def binary_labels(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    labels = np.full(arr.shape, -1, dtype=int)
    labels[finite] = (arr[finite] > 0.5).astype(int)
    return labels


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    keep = np.isfinite(scores) & np.isin(labels, [0, 1])
    scores = scores[keep]
    labels = labels[keep]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.asarray([]), np.asarray([]), float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr = np.concatenate([[0.0], tp / float(n_pos), [1.0]])
    fpr = np.concatenate([[0.0], fp / float(n_neg), [1.0]])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def save_all(fig, output_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_hyper_prediction_output(args.prediction_output, max_events=args.max_events)
    if args.score_field not in predictions.columns:
        raise KeyError(f"Score field not found: {args.score_field}")

    scores_all = np.asarray(predictions[args.score_field], dtype=float).reshape(-1)
    finite = np.isfinite(scores_all)
    scores = scores_all[finite]

    label_field = args.label_field
    if label_field is None:
        for candidate in ("HyPER_CLS_T", "truth_label", "label", "cls_t"):
            if candidate in predictions.columns:
                label_field = candidate
                break

    labels = None
    if label_field is not None and label_field in predictions.columns:
        labels = binary_labels(predictions[label_field])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores, bins=50, histtype="stepfilled", alpha=0.75)
    ax.set_xlabel(args.score_field)
    ax.set_ylabel("Events")
    ax.set_title("Classifier score")
    save_all(fig, output_dir, "classifier_score_hist", args.formats)

    summary = {
        "prediction_output": str(args.prediction_output),
        "n_rows": int(len(predictions)),
        "score_field": args.score_field,
        "score_finite": int(finite.sum()),
        "score_finite_fraction": float(finite.mean()) if len(finite) else float("nan"),
        "score_min": float(np.min(scores)) if scores.size else float("nan"),
        "score_max": float(np.max(scores)) if scores.size else float("nan"),
        "score_mean": float(np.mean(scores)) if scores.size else float("nan"),
        "score_std": float(np.std(scores)) if scores.size else float("nan"),
        "label_field": label_field,
    }

    if labels is not None:
        labelled = finite & np.isin(labels, [0, 1])
        y = labels[labelled]
        s = scores_all[labelled]
        n_signal = int((y == 1).sum())
        n_background = int((y == 0).sum())
        pred = (s >= 0.5).astype(int)
        accuracy = float((pred == y).mean()) if y.size else float("nan")
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fpr, tpr, auc = roc_auc(scores_all, labels)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(s[y == 0], bins=50, alpha=0.55, label="background")
        ax.hist(s[y == 1], bins=50, alpha=0.55, label="signal")
        ax.set_xlabel(args.score_field)
        ax.set_ylabel("Events")
        ax.set_title("Classifier score by truth class")
        ax.legend()
        save_all(fig, output_dir, "classifier_score_by_truth", args.formats)

        if fpr.size and tpr.size and math.isfinite(auc):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("Classifier ROC")
            ax.legend(loc="lower right")
            save_all(fig, output_dir, "classifier_roc", args.formats)

        summary.update(
            {
                "n_labelled": int(labelled.sum()),
                "n_signal": n_signal,
                "n_background": n_background,
                "signal_fraction": float(n_signal / y.size) if y.size else float("nan"),
                "auc": auc,
                "accuracy_at_0p5": accuracy,
                "confusion_at_0p5": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "signal_efficiency_at_0p5": float(tp / (tp + fn)) if tp + fn else float("nan"),
                "background_rejection_at_0p5": float((tn + fp) / fp) if fp else float("inf"),
                "label_source": "prediction:HyPER_CLS_T",
                "fallback_fully_matched_used": False,
            }
        )
        rejection = {}
        for efficiency in (0.7, 0.8, 0.9):
            signal_scores = s[y == 1]
            background_scores = s[y == 0]
            threshold = float(np.quantile(signal_scores, 1.0 - efficiency))
            background_efficiency = float((background_scores >= threshold).mean())
            rejection[str(efficiency)] = {
                "threshold": threshold,
                "background_rejection": float(1.0 / background_efficiency) if background_efficiency else float("inf"),
            }
        summary["background_rejection_at_signal_efficiency"] = rejection

        fig, ax = plt.subplots(figsize=(5, 5))
        matrix = np.asarray([[tn, fp], [fn, tp]])
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
        ax.set_xticks([0, 1], labels=["background", "signal"])
        ax.set_yticks([0, 1], labels=["background", "signal"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Truth")
        ax.set_title("Confusion matrix at probability 0.5")
        fig.colorbar(image, ax=ax)
        save_all(fig, output_dir, "classifier_confusion_matrix", args.formats)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [f"{key}: {value}" for key, value in summary.items()]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
