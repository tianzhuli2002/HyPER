#!/usr/bin/env python3
"""Make ttbar single-lepton S/B plots from a decorated HyPER H5."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prediction_io import load_hyper_prediction_output  # noqa: E402


LOGGER = logging.getLogger(__name__)
CUTS = ("all", "reco_valid", "thad_valid", "whad_valid", "b_lep_valid")
LABEL_CANDIDATES = (
    "LABELS/GLOBAL",
    "LABELS/SIGNAL",
    "LABELS/Y",
    "LABELS/CLASS",
    "LABELS/label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-field", default="HyPER_CLS_PROB")
    parser.add_argument(
        "--prediction-output",
        default=None,
        help="Optional HyPER output file to use when the score is not in RECO_GLOBAL.",
    )
    parser.add_argument(
        "--label-field",
        default=None,
        help="Optional H5 dataset path for binary truth labels.",
    )
    parser.add_argument(
        "--cuts",
        nargs="+",
        default=["all"],
        choices=CUTS,
        help="Cut selections to plot and summarize.",
    )
    return parser.parse_args()


def flatten_dataset(dataset: h5py.Dataset) -> np.ndarray:
    values = dataset[()]
    if getattr(values.dtype, "names", None):
        first = values.dtype.names[0]
        values = values[first]
    return np.asarray(values).reshape(-1)


def read_labels(h5: h5py.File, label_field: str | None) -> tuple[np.ndarray, str]:
    candidates = (label_field,) if label_field else LABEL_CANDIDATES
    for candidate in candidates:
        if candidate and candidate in h5:
            labels = flatten_dataset(h5[candidate]).astype(float)
            labels = np.where(labels > 0.5, 1, 0).astype(int)
            return labels, candidate
    raise KeyError(
        "Could not find truth labels. Tried: "
        + ", ".join(candidate for candidate in candidates if candidate)
    )


def read_reco_global(h5: h5py.File) -> np.ndarray:
    path = "INPUTS/RECO_GLOBAL"
    if path not in h5:
        raise KeyError(f"Missing required decorated dataset {path}")
    reco = h5[path][()]
    if reco.dtype.names is None:
        raise TypeError(f"{path} must be a structured dataset.")
    return reco


def read_scores(
    reco_global: np.ndarray,
    score_field: str,
    prediction_output: str | None,
    n_events: int,
) -> tuple[np.ndarray, str]:
    if score_field in (reco_global.dtype.names or ()):
        return np.asarray(reco_global[score_field]).reshape(-1).astype(float), (
            f"INPUTS/RECO_GLOBAL/{score_field}"
        )

    if prediction_output is None:
        raise KeyError(
            f"Score field {score_field!r} is not in INPUTS/RECO_GLOBAL and "
            "--prediction-output was not provided."
        )

    predictions = load_hyper_prediction_output(prediction_output, max_events=n_events)
    if score_field not in predictions.columns:
        raise KeyError(
            f"Score field {score_field!r} is not in prediction output columns: "
            + ", ".join(map(str, predictions.columns))
        )
    scores = predictions[score_field].to_numpy(dtype=float)
    return scores[:n_events], f"{prediction_output}:{score_field}"


def binary_roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    finite = np.isfinite(scores)
    labels = labels[finite].astype(int)
    scores = scores[finite].astype(float)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]), float("nan")

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)
    tpr = np.concatenate(([0.0], tps / positives, [1.0]))
    fpr = np.concatenate(([0.0], fps / negatives, [1.0]))
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def cut_mask(reco_global: np.ndarray, cut: str) -> np.ndarray:
    n_events = len(reco_global)
    if cut == "all":
        return np.ones(n_events, dtype=bool)
    if cut not in (reco_global.dtype.names or ()):
        LOGGER.warning("Cut field %s is missing; selection is empty.", cut)
        return np.zeros(n_events, dtype=bool)
    return np.asarray(reco_global[cut]).reshape(-1) == 1


def finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float("nan")


def plot_score_distribution(
    labels: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    cut: str,
    output_path: Path,
) -> None:
    selected_labels = labels[mask]
    selected_scores = scores[mask]
    finite = np.isfinite(selected_scores)
    selected_labels = selected_labels[finite]
    selected_scores = selected_scores[finite]

    plt.figure(figsize=(6.0, 4.5))
    bins = np.linspace(0.0, 1.0, 41)
    for label, title, color in ((0, "Background", "tab:blue"), (1, "Signal", "tab:orange")):
        values = selected_scores[selected_labels == label]
        if len(values):
            plt.hist(values, bins=bins, histtype="step", density=True, label=title, color=color)
    plt.xlabel("S/B score")
    plt.ylabel("Density")
    plt.title("Score distribution" if cut == "all" else f"Score distribution ({cut})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc(rows: list[dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(5.2, 5.0))
    plotted = False
    for row in rows:
        fpr = row.pop("_fpr")
        tpr = row.pop("_tpr")
        auc = row.get("auc")
        if len(fpr) and len(tpr) and math.isfinite(float(auc)):
            plt.plot(fpr, tpr, label=f"{row['cut']} AUC={float(auc):.3f}")
            plotted = True
    plt.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    plt.xlabel("Background efficiency")
    plt.ylabel("Signal efficiency")
    plt.title("ROC curve")
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_cut(
    cut: str,
    labels: np.ndarray,
    scores: np.ndarray,
    reco_global: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    selected_labels = labels[mask]
    selected_scores = scores[mask]
    finite = np.isfinite(selected_scores)
    selected_labels = selected_labels[finite]
    selected_scores = selected_scores[finite]

    fpr, tpr, auc = binary_roc(selected_labels, selected_scores)
    row: dict[str, object] = {
        "cut": cut,
        "n_events": int(len(selected_scores)),
        "n_signal": int(np.sum(selected_labels == 1)),
        "n_background": int(np.sum(selected_labels == 0)),
        "auc": auc,
        "mean_score_signal": finite_mean(selected_scores[selected_labels == 1]),
        "mean_score_background": finite_mean(selected_scores[selected_labels == 0]),
        "_fpr": fpr,
        "_tpr": tpr,
    }
    for valid_field in ("reco_valid", "thad_valid", "whad_valid", "b_lep_valid"):
        if valid_field in (reco_global.dtype.names or ()):
            valid = np.asarray(reco_global[valid_field]).reshape(-1)[mask]
            row[f"{valid_field}_fraction"] = float(np.mean(valid == 1)) if len(valid) else float("nan")
    return row


def write_summary(rows: list[dict[str, object]], output_dir: Path, metadata: dict[str, object]) -> None:
    def clean_value(value: object) -> object:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    clean_rows = [
        {key: clean_value(value) for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    fieldnames: list[str] = []
    for row in clean_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "cuts": clean_rows}, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input_h5, "r") as h5:
        reco_global = read_reco_global(h5)
        labels, label_source = read_labels(h5, args.label_field)
        scores, score_source = read_scores(
            reco_global,
            args.score_field,
            args.prediction_output,
            n_events=len(labels),
        )

    n_events = min(len(labels), len(scores), len(reco_global))
    labels = labels[:n_events]
    scores = scores[:n_events]
    reco_global = reco_global[:n_events]

    rows = []
    for cut in args.cuts:
        mask = cut_mask(reco_global, cut)
        rows.append(summarize_cut(cut, labels, scores, reco_global, mask))
        plot_name = "score_distribution.pdf" if cut == "all" else f"score_distribution_{cut}.pdf"
        plot_score_distribution(labels, scores, mask, cut, output_dir / plot_name)

    plot_roc(rows, output_dir / "roc_curve.pdf")
    write_summary(
        rows,
        output_dir,
        {
            "input_h5": str(args.input_h5),
            "score_source": score_source,
            "label_source": label_source,
            "n_events_available": int(n_events),
        },
    )
    LOGGER.info("Wrote plots and summaries to %s", output_dir)


if __name__ == "__main__":
    main()
