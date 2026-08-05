#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


from HyPER.topology.plot_style import (
    BACKGROUND_COLOUR,
    SIGNAL_COLOUR,
    FM_COLOUR,
    NONFM_COLOUR,
    REFERENCE_COLOUR,
    configure_matplotlib,
    save_figure as shared_save_figure,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-quality HyPER classification plots."
    )
    parser.add_argument(
        "--prediction-output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--score-field",
        default="HyPER_CLS_PROB",
    )
    parser.add_argument(
        "--label-field",
        default="HyPER_CLS_T",
    )
    parser.add_argument(
        "--fm-field",
        default="truth_fully_matched",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Optional plot title; inferred from the output path when omitted.",
    )
    parser.add_argument(
        "--experiment-label",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--formats", nargs="+", default=("pdf", "png")
    )
    parser.add_argument(
        "--plot-set", choices=("essential", "full"), default="essential"
    )
    return parser.parse_args()


def load_prediction_columns(
    prediction_path: Path,
    columns: Iterable[str],
) -> pd.DataFrame:
    columns = list(columns)

    if prediction_path.is_dir() or prediction_path.name.endswith(".pkl.parts"):
        parts = sorted(prediction_path.glob("part_*.pkl"))

        if not parts:
            raise FileNotFoundError(
                f"No part_*.pkl files found under {prediction_path}"
            )

        frames = []

        for part in parts:
            frame = pd.read_pickle(part)

            missing = [column for column in columns if column not in frame.columns]
            if missing:
                raise KeyError(
                    f"{part} is missing required columns: {missing}"
                )

            frames.append(frame.loc[:, columns].copy())

        return pd.concat(frames, ignore_index=True)

    frame = pd.read_pickle(prediction_path)

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{prediction_path} is missing required columns: {missing}"
        )

    return frame.loc[:, columns].copy()



def infer_model_label(prediction_path: Path) -> str:
    value = str(prediction_path).lower()

    if "ttbar1l" in value or "ttbar_sl" in value:
        topology = "ttbar single-lepton"
    elif "tth" in value:
        topology = "ttH single-lepton"
    else:
        topology = "HyPER classification"

    if "classification_only" in value:
        mode = "classification-only"
    elif "joint" in value:
        mode = "joint reconstruction and classification"
    else:
        mode = ""

    return f"{topology} — {mode}" if mode else topology


def decorate_axis(
    ax: plt.Axes,
    experiment_label: str,
    model_label: str,
) -> None:
    del experiment_label

    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if model_label:
        ax.set_title(model_label, pad=12)


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
) -> None:
    shared_save_figure(fig, output_dir, stem, formats)


def draw_filled_histogram(
    ax: plt.Axes,
    values: np.ndarray,
    bins: np.ndarray,
    label: str,
    colour: str,
    alpha: float = 0.12,
) -> None:
    if len(values) == 0:
        return

    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="stepfilled",
        color=colour,
        alpha=alpha,
        linewidth=0,
    )
    ax.hist(
        values,
        bins=bins,
        density=True,
        histtype="step",
        color=colour,
        linewidth=2.0,
        label=f"{label}  ($N={len(values):,}$)",
    )


def make_score_plot(
    output_dir: Path,
    score_groups: list[tuple[str, np.ndarray, str]],
    bins: np.ndarray,
    experiment_label: str,
    model_label: str,
    stem: str,
    formats: Iterable[str],
    logarithmic: bool = False,
) -> None:
    fig, ax = plt.subplots()

    for label, values, colour in score_groups:
        draw_filled_histogram(
            ax=ax,
            values=values,
            bins=bins,
            label=label,
            colour=colour,
        )

    ax.set_xlabel("HyPER classification score")
    ax.set_ylabel("Probability density")
    ax.set_xlim(0.0, 1.0)

    if logarithmic:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1.0e-3)

    decorate_axis(ax, experiment_label, model_label)

    ax.legend(
        loc="best",
        handlelength=2.4,
    )

    fig.tight_layout()
    save_figure(fig, output_dir, stem, formats)


def make_standard_roc(
    output_dir: Path,
    curves: list[tuple[str, np.ndarray, np.ndarray, float, str]],
    experiment_label: str,
    model_label: str,
    stem: str,
    formats: Iterable[str],
) -> None:
    fig, ax = plt.subplots()

    for label, fpr, tpr, auc_value, colour in curves:
        ax.plot(
            fpr,
            tpr,
            linewidth=2.1,
            color=colour,
            label=f"{label}  (AUC = {auc_value:.4f})",
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.2,
        color=REFERENCE_COLOUR,
        label="Random classifier",
    )

    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)

    decorate_axis(ax, experiment_label, model_label)
    ax.legend(loc="lower right")

    fig.tight_layout()
    save_figure(fig, output_dir, stem, formats)


def make_background_rejection_plot(
    output_dir: Path,
    curves: list[tuple[str, np.ndarray, np.ndarray, float, str, int]],
    experiment_label: str,
    model_label: str,
    formats: Iterable[str],
) -> None:
    fig, ax = plt.subplots()

    for label, fpr, tpr, auc_value, colour, n_background in curves:
        minimum_efficiency = 1.0 / max(n_background, 1)
        rejection = 1.0 / np.maximum(fpr, minimum_efficiency)

        ax.plot(
            tpr,
            rejection,
            linewidth=2.1,
            color=colour,
            label=f"{label}  (AUC = {auc_value:.4f})",
        )

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection  $1/\\epsilon_{\\mathrm{bkg}}$")
    ax.set_xlim(0.2, 0.9)
    ax.set_yscale("log")

    decorate_axis(ax, experiment_label, model_label)
    ax.legend(loc="upper right")

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "classifier_background_rejection",
        formats,
    )


def make_confusion_matrix(
    output_dir: Path,
    truth: np.ndarray,
    prediction: np.ndarray,
    experiment_label: str,
    model_label: str,
    formats: Iterable[str],
) -> None:
    matrix = confusion_matrix(
        truth,
        prediction,
        labels=[0, 1],
        normalize="true",
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )

    labels = ["Background", "Signal"]

    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            colour = "white" if value > 0.55 else "black"

            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=colour,
            )

    decorate_axis(ax, experiment_label, model_label)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "classifier_confusion_matrix",
        formats,
    )


def make_category_fractions(
    output_dir: Path,
    counts: dict[str, int],
    experiment_label: str,
    model_label: str,
    formats: Iterable[str],
) -> None:
    labels = [
        "Background",
        "Signal FM",
        "Signal non-FM",
    ]
    values = np.asarray(
        [
            counts["background"],
            counts["signal_fm"],
            counts["signal_nonfm"],
        ],
        dtype=float,
    )
    fractions = values / values.sum()

    fig, ax = plt.subplots(figsize=(6.8, 5.4))

    bars = ax.bar(
        labels,
        fractions,
        color=[
            BACKGROUND_COLOUR,
            FM_COLOUR,
            NONFM_COLOUR,
        ],
        alpha=0.78,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, count, fraction in zip(bars, values, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{fraction:.1%}\n$N={int(count):,}$",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    ax.set_ylabel("Fraction of test events")
    ax.set_ylim(0.0, max(fractions) * 1.28)
    ax.tick_params(axis="x", which="minor", bottom=False)

    decorate_axis(ax, experiment_label, model_label)

    fig.tight_layout()
    save_figure(
        fig,
        output_dir,
        "classifier_category_fractions",
        formats,
    )


def rejection_at_efficiency(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_efficiency: float,
    n_background: int,
) -> float:
    index = int(np.argmin(np.abs(tpr - target_efficiency)))
    minimum_efficiency = 1.0 / max(n_background, 1)

    return float(1.0 / max(float(fpr[index]), minimum_efficiency))


def binary_subset_roc(
    background_score: np.ndarray,
    signal_score: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(background_score) == 0 or len(signal_score) == 0:
        raise RuntimeError(
            f"{label} ROC requires at least one background and one signal event."
        )
    truth = np.concatenate(
        [np.zeros(len(background_score), dtype=np.int8),
         np.ones(len(signal_score), dtype=np.int8)]
    )
    scores = np.concatenate([background_score, signal_score])
    fpr, tpr, _ = roc_curve(truth, scores)
    return fpr, tpr, float(roc_auc_score(truth, scores))


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_label = (
        args.model_label
        if args.model_label
        else infer_model_label(args.prediction_output)
    )

    frame = load_prediction_columns(
        args.prediction_output,
        (
            args.score_field,
            args.label_field,
            args.fm_field,
        ),
    )

    score = np.asarray(frame[args.score_field], dtype=float)
    truth = (
        np.asarray(frame[args.label_field], dtype=float) > 0.5
    ).astype(np.int8)
    fully_matched = np.asarray(
        frame[args.fm_field],
        dtype=bool,
    )

    finite = np.isfinite(score)

    if not np.all(finite):
        score = score[finite]
        truth = truth[finite]
        fully_matched = fully_matched[finite]

    score = np.clip(score, 0.0, 1.0)

    background_mask = truth == 0
    signal_mask = truth == 1
    signal_fm_mask = signal_mask & fully_matched
    signal_nonfm_mask = signal_mask & ~fully_matched

    background_score = score[background_mask]
    signal_score = score[signal_mask]
    signal_fm_score = score[signal_fm_mask]
    signal_nonfm_score = score[signal_nonfm_mask]

    if len(background_score) == 0 or len(signal_score) == 0:
        raise RuntimeError(
            "Both signal and background events are required."
        )

    auc_all = float(roc_auc_score(truth, score))
    fpr_all, tpr_all, _ = roc_curve(truth, score)

    fpr_fm, tpr_fm, auc_fm = binary_subset_roc(
        background_score, signal_fm_score, "Fully matched signal"
    )

    prediction = (score >= 0.5).astype(np.int8)
    accuracy = float(accuracy_score(truth, prediction))

    bins = np.linspace(0.0, 1.0, args.bins + 1)

    all_groups = [
        ("Background", background_score, BACKGROUND_COLOUR),
        ("All signal", signal_score, SIGNAL_COLOUR),
    ]

    category_groups = [
        ("Background", background_score, BACKGROUND_COLOUR),
        ("Fully matched signal", signal_fm_score, FM_COLOUR),
        ("Not fully matched signal", signal_nonfm_score, NONFM_COLOUR),
    ]

    make_score_plot(
        args.output_dir,
        category_groups,
        bins,
        args.experiment_label,
        model_label,
        "classification_scores",
        args.formats,
    )

    fpr_nonfm, tpr_nonfm, auc_nonfm = binary_subset_roc(
        background_score, signal_nonfm_score, "Not fully matched signal"
    )

    roc_curves = [
        ("All signal", fpr_all, tpr_all, auc_all, SIGNAL_COLOUR),
        ("Fully matched signal", fpr_fm, tpr_fm, auc_fm, FM_COLOUR),
        ("Not fully matched signal", fpr_nonfm, tpr_nonfm, auc_nonfm, NONFM_COLOUR),
    ]

    make_standard_roc(
        args.output_dir,
        roc_curves,
        args.experiment_label,
        model_label,
        "classifier_roc",
        args.formats,
    )


    make_background_rejection_plot(
        args.output_dir,
        [
            (label, fpr, tpr, auc_value, colour, len(background_score))
            for label, fpr, tpr, auc_value, colour in roc_curves
        ],
        args.experiment_label,
        model_label,
        args.formats,
    )

    if args.plot_set == "full":
        make_confusion_matrix(
            args.output_dir, truth, prediction, args.experiment_label, model_label, args.formats
        )

    counts = {
        "all": int(len(score)),
        "background": int(background_mask.sum()),
        "signal": int(signal_mask.sum()),
        "signal_fm": int(signal_fm_mask.sum()),
        "signal_nonfm": int(signal_nonfm_mask.sum()),
    }

    if args.plot_set == "full":
        make_category_fractions(
            args.output_dir, counts, args.experiment_label, model_label, args.formats
        )

    summary = {
        "prediction_output": str(args.prediction_output.resolve()),
        "score_field": args.score_field,
        "label_field": args.label_field,
        "fm_field": args.fm_field,
        "model_label": model_label,
        "counts": counts,
        # Canonical fields retained for workflow-suite compatibility.
        "roc_auc": auc_all,
        "auc": auc_all,
        "accuracy": accuracy,
        "classification_accuracy": accuracy,

        # Additional detailed fields.
        "auc_all_signal": auc_all,
        "auc_fm_signal": auc_fm,
        "auc_nonfm_signal": auc_nonfm,
        "accuracy_at_threshold_0p5": accuracy,
        "background_rejection": {
            "all_signal_at_50_percent": rejection_at_efficiency(
                fpr_all,
                tpr_all,
                0.50,
                len(background_score),
            ),
            "all_signal_at_70_percent": rejection_at_efficiency(
                fpr_all,
                tpr_all,
                0.70,
                len(background_score),
            ),
            "all_signal_at_80_percent": rejection_at_efficiency(
                fpr_all,
                tpr_all,
                0.80,
                len(background_score),
            ),
            "fm_signal_at_50_percent": rejection_at_efficiency(
                fpr_fm,
                tpr_fm,
                0.50,
                len(background_score),
            ),
            "fm_signal_at_70_percent": rejection_at_efficiency(
                fpr_fm,
                tpr_fm,
                0.70,
                len(background_score),
            ),
            "fm_signal_at_80_percent": rejection_at_efficiency(
                fpr_fm, tpr_fm, 0.80, len(background_score)
            ),
            "nonfm_signal_at_50_percent": rejection_at_efficiency(
                fpr_nonfm, tpr_nonfm, 0.50, len(background_score)
            ),
            "nonfm_signal_at_70_percent": rejection_at_efficiency(
                fpr_nonfm, tpr_nonfm, 0.70, len(background_score)
            ),
            "nonfm_signal_at_80_percent": rejection_at_efficiency(
                fpr_nonfm, tpr_nonfm, 0.80, len(background_score)
            ),
        },
    }

    # summary.json is the canonical workflow output consumed by
    # validate_workflow_suite.py.
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    (args.output_dir / "metrics.json").write_text(summary_text, encoding="utf-8")
    metrics_rows = []
    for subset, auc_value, fpr, tpr, signal_count in (
        ("all_signal", auc_all, fpr_all, tpr_all, len(signal_score)),
        ("fully_matched_signal", auc_fm, fpr_fm, tpr_fm, len(signal_fm_score)),
        ("not_fully_matched_signal", auc_nonfm, fpr_nonfm, tpr_nonfm, len(signal_nonfm_score)),
    ):
        metrics_rows.append(
            {
                "subset": subset,
                "signal_events": int(signal_count),
                "background_events": int(len(background_score)),
                "roc_auc": float(auc_value),
                "background_rejection_at_signal_efficiency_0.5": rejection_at_efficiency(
                    fpr, tpr, 0.5, len(background_score)
                ),
                "background_rejection_at_signal_efficiency_0.7": rejection_at_efficiency(
                    fpr, tpr, 0.7, len(background_score)
                ),
                "background_rejection_at_signal_efficiency_0.8": rejection_at_efficiency(
                    fpr, tpr, 0.8, len(background_score)
                ),
            }
        )
    pd.DataFrame(metrics_rows).to_csv(args.output_dir / "metrics.csv", index=False)

    print(summary_text, end="")


if __name__ == "__main__":
    main()
