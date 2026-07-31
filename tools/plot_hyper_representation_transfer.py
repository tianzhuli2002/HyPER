#!/usr/bin/env python3
"""Produce the final HyPER CKA, zero-shot, transfer and control plot suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from HyPER.topology.plot_style import (
    ALIGNED_COLOUR,
    BACKGROUND_COLOUR,
    DIRECT_COLOUR,
    FM_COLOUR,
    JOINT_COLOUR,
    METHOD_COLOURS,
    NONFM_COLOUR,
    REFERENCE_COLOUR,
    SHUFFLED_COLOUR,
    SIGNAL_COLOUR,
    ZERO_SHOT_COLOUR,
    configure_matplotlib,
    decorate_axis,
    save_figure,
)

METHOD_LABELS = {
    "native_classification_only_score": "Native classification-only",
    "native_joint_score": "Native joint",
    "reconstruction_zero_shot_score": "Reconstruction zero-shot",
    "joint_reconstruction_zero_shot_score": "Joint reconstruction zero-shot",
    "reconstruction_to_classification_direct_score": "Reco→class direct",
    "reconstruction_to_classification_paired_score": "Reco→class paired",
    "reconstruction_to_joint_direct_score": "Reco→joint direct",
    "reconstruction_to_joint_paired_score": "Reco→joint paired",
    "joint_to_classification_direct_score": "Joint→class direct",
    "joint_to_classification_paired_score": "Joint→class paired",
}
PAIR_LABELS = {
    "classification_vs_reconstruction": "Classification–reconstruction",
    "classification_vs_joint": "Classification–joint",
    "reconstruction_vs_joint": "Reconstruction–joint",
}
REPRESENTATIONS = ("block_0", "block_1", "block_2", "final_event")
SUBSET_LABELS = {
    "all": "All events",
    "background": "Background",
    "signal_fully_matched": "Signal FM",
    "signal_non_fully_matched": "Signal non-FM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--controls-summary-dir", required=True)
    parser.add_argument("--cka-root", required=True)
    parser.add_argument("--full-test-cka-dir", required=True)
    parser.add_argument("--alignments-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="ttbar single-lepton representation transfer")
    return parser.parse_args()


def load_array(directory: Path, stem: str, mmap_mode="r"):
    path = directory / f"{stem}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode=mmap_mode)


def subset_masks(labels: np.ndarray, fm: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "inclusive_all_signal": np.ones(len(labels), dtype=bool),
        "fully_matched_signal_vs_background": (labels == 0) | ((labels == 1) & fm),
        "non_fully_matched_signal_vs_background": (labels == 0) | ((labels == 1) & ~fm),
    }


def category_histograms(ax, scores: np.ndarray, labels: np.ndarray, fm: np.ndarray, xlabel: str, title: str) -> None:
    bins = np.linspace(0.0, 1.0, 41)
    categories = (
        (labels == 0, "Background", BACKGROUND_COLOUR),
        ((labels == 1) & fm, "Signal FM", FM_COLOUR),
        ((labels == 1) & ~fm, "Signal non-FM", NONFM_COLOUR),
    )
    for mask, label, colour in categories:
        ax.hist(
            scores[mask],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.20,
            edgecolor=colour,
            facecolor=colour,
            linewidth=2.0,
            label=f"{label} ($N={int(mask.sum()):,}$)",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalised events")
    decorate_axis(ax, title=title)
    ax.legend(loc="best")


def add_roc(ax, labels, scores, label, colour, linestyle="-") -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))
    ax.plot(fpr, tpr, color=colour, linestyle=linestyle, linewidth=2.0, label=f"{label} (AUC={auc:.3f})")
    return fpr, tpr, auc


def plot_zero_shot(scores, labels, fm, output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True)
    category_histograms(
        axes[0], scores["reconstruction_zero_shot_score"], labels, fm,
        r"$S_{\mathrm{reco}}$", "Reconstruction-only",
    )
    category_histograms(
        axes[1], scores["joint_reconstruction_zero_shot_score"], labels, fm,
        r"$S_{\mathrm{reco}}$", "Joint model",
    )
    fig.suptitle(f"{title}: reconstruction-confidence score", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, "zero_shot_score_distributions")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True, sharey=True)
    for ax, field, panel_title in (
        (axes[0], "reconstruction_zero_shot_score", "Reconstruction-only"),
        (axes[1], "joint_reconstruction_zero_shot_score", "Joint model"),
    ):
        values = scores[field]
        add_roc(ax, labels, values, "All signal", SIGNAL_COLOUR)
        fm_mask = (labels == 0) | ((labels == 1) & fm)
        nonfm_mask = (labels == 0) | ((labels == 1) & ~fm)
        add_roc(ax, labels[fm_mask], values[fm_mask], "Signal FM", FM_COLOUR)
        add_roc(ax, labels[nonfm_mask], values[nonfm_mask], "Signal non-FM", NONFM_COLOUR)
        ax.plot([0, 1], [0, 1], linestyle="--", color=REFERENCE_COLOUR, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Background efficiency")
        ax.set_ylabel("Signal efficiency")
        decorate_axis(ax, title=panel_title)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{title}: zero-shot ROC", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, "zero_shot_roc")

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    for field, label, colour in (
        ("reconstruction_zero_shot_score", "Reconstruction-only", ZERO_SHOT_COLOUR),
        ("joint_reconstruction_zero_shot_score", "Joint model", JOINT_COLOUR),
    ):
        fpr, tpr, _ = roc_curve(labels, scores[field])
        minimum = 1.0 / max(int(np.sum(labels == 0)), 1)
        rejection = 1.0 / np.maximum(fpr, minimum)
        ax.plot(tpr, rejection, linewidth=2.0, color=colour, label=label)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    decorate_axis(ax, title=f"{title}: zero-shot background rejection")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output, "zero_shot_background_rejection")


def plot_transfer_rocs(scores, labels, output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    main = (
        "native_classification_only_score",
        "native_joint_score",
        "reconstruction_zero_shot_score",
        "reconstruction_to_classification_direct_score",
        "reconstruction_to_classification_paired_score",
    )
    for field in main:
        add_roc(ax, labels, scores[field], METHOD_LABELS[field], METHOD_COLOURS.get(field, None))
    ax.plot([0, 1], [0, 1], linestyle="--", color=REFERENCE_COLOUR, linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    decorate_axis(ax, title=f"{title}: principal comparison")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output, "main_transfer_roc")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            (
                "native_joint_score",
                "reconstruction_to_joint_direct_score",
                "reconstruction_to_joint_paired_score",
            ),
            "Reconstruction→joint head",
        ),
        (
            axes[1],
            (
                "native_classification_only_score",
                "joint_to_classification_direct_score",
                "joint_to_classification_paired_score",
            ),
            "Joint→classification-only head",
        ),
    )
    for ax, fields, panel_title in panels:
        for field in fields:
            add_roc(ax, labels, scores[field], METHOD_LABELS[field], METHOD_COLOURS.get(field, None))
        ax.plot([0, 1], [0, 1], linestyle="--", color=REFERENCE_COLOUR, linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Background efficiency")
        ax.set_ylabel("Signal efficiency")
        decorate_axis(ax, title=panel_title)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{title}: bridge transfers", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, "bridge_transfer_roc")


def plot_correlations(scores, output: Path, title: str) -> None:
    pairs = (
        ("native_classification_only_score", "reconstruction_to_classification_paired_score", "Reco→class paired"),
        ("native_classification_only_score", "joint_to_classification_paired_score", "Joint→class paired"),
        ("native_joint_score", "reconstruction_to_joint_paired_score", "Reco→joint paired"),
    )
    for native, transferred, panel_title in pairs:
        x = np.asarray(scores[native], dtype=np.float64)
        y = np.asarray(scores[transferred], dtype=np.float64)
        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)
        fig, ax = plt.subplots(figsize=(6.7, 6.0))
        image = ax.hexbin(x, y, gridsize=75, mincnt=1, cmap="viridis")
        ax.plot([0, 1], [0, 1], linestyle="--", color="white", linewidth=1.0, alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(METHOD_LABELS[native])
        ax.set_ylabel(METHOD_LABELS[transferred])
        decorate_axis(ax, title=f"{panel_title}\nPearson={pearson:.3f}, Spearman={spearman:.3f}")
        fig.colorbar(image, ax=ax, label="Events per hexagonal bin")
        fig.tight_layout()
        save_figure(fig, output, f"score_correlation_{transferred.replace('_score', '')}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_bootstrap(metrics_dir: Path, output: Path, title: str) -> None:
    summary_rows = read_csv(metrics_dir / "bootstrap_summary.csv")
    inclusive = [
        row for row in summary_rows
        if row["subset"] == "inclusive_all_signal" and row["metric"] == "roc_auc"
    ]
    order = [name for name in METHOD_LABELS if any(row["method"] == name for row in inclusive)]
    lookup = {row["method"]: row for row in inclusive}
    x = np.arange(len(order))
    values = np.asarray([float(lookup[name]["point_estimate"]) for name in order])
    low = np.asarray([float(lookup[name]["interval_95_low"]) for name in order])
    high = np.asarray([float(lookup[name]["interval_95_high"]) for name in order])
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for index, name in enumerate(order):
        ax.errorbar(
            index,
            values[index],
            yerr=[[values[index] - low[index]], [high[index] - values[index]]],
            fmt="o",
            markersize=6,
            capsize=3,
            color=METHOD_COLOURS.get(name, SIGNAL_COLOUR),
        )
    ax.set_xticks(x, [METHOD_LABELS[name] for name in order], rotation=35, ha="right")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(max(0.0, low.min() - 0.03), min(1.0, high.max() + 0.03))
    decorate_axis(ax, title=f"{title}: full-test AUC with 95% bootstrap intervals")
    fig.tight_layout()
    save_figure(fig, output, "auc_bootstrap_summary")

    difference_path = metrics_dir / "bootstrap_auc_differences.npz"
    difference_rows = read_csv(metrics_dir / "bootstrap_differences.csv")
    with np.load(difference_path, allow_pickle=False) as loaded:
        names = [row["difference"] for row in difference_rows if row["subset"] == "inclusive_all_signal"]
        fig, axes = plt.subplots(len(names), 1, figsize=(9.0, 2.5 * len(names)), squeeze=False)
        for ax, name in zip(axes.flat, names):
            key = f"inclusive_all_signal__{name}"
            values = loaded[key]
            ax.hist(values, bins=45, density=True, color=ALIGNED_COLOUR, alpha=0.35, edgecolor=ALIGNED_COLOUR)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
            ax.axvline(np.median(values), color=ALIGNED_COLOUR, linewidth=1.8)
            ax.set_xlabel("Paired AUC difference")
            ax.set_ylabel("Density")
            decorate_axis(ax, title=name.replace("_", " "))
        fig.tight_layout()
        save_figure(fig, output, "bootstrap_auc_differences")


def plot_control_nulls(controls_dir: Path, output: Path, title: str) -> None:
    rows = read_csv(controls_dir / "shuffled_null_summary.csv")
    with np.load(controls_dir / "alignment_control_auc_distributions.npz", allow_pickle=False) as loaded:
        directions = (
            "reconstruction_to_classification",
            "reconstruction_to_joint",
            "joint_to_classification",
        )
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
        for ax, direction in zip(axes, directions):
            row = next(
                item for item in rows
                if item["direction"] == direction
                and item["control_type"] == "shuffled"
                and item["subset"] == "inclusive_all_signal"
            )
            values = loaded[f"{direction}__shuffled__inclusive_all_signal"]
            ax.hist(values, bins=max(10, min(20, len(values))), density=True,
                    color=SHUFFLED_COLOUR, alpha=0.55, edgecolor=BACKGROUND_COLOUR)
            ax.axvline(float(row["paired_auc"]), color=ALIGNED_COLOUR, linewidth=2.2, label="Paired")
            ax.axvline(float(row["direct_auc"]), color=DIRECT_COLOUR, linewidth=2.0, linestyle="--", label="Direct")
            ax.axvline(float(row["median"]), color=BACKGROUND_COLOUR, linewidth=1.6, linestyle=":", label="Shuffled median")
            ax.set_xlabel("Full-test ROC AUC")
            decorate_axis(
                ax,
                title=f"{direction.replace('_', ' ')}\nempirical $p={float(row['empirical_p_value']):.3f}$",
            )
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Control density")
        fig.suptitle(f"{title}: shuffled-alignment nulls", y=1.02)
        fig.tight_layout()
        save_figure(fig, output, "shuffled_alignment_nulls")


def load_cka_summary(cka_root: Path, pair: str) -> dict:
    path = cka_root / pair / "cka_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def plot_cka(cka_root: Path, full_test_dir: Path, output: Path, title: str) -> None:
    summaries = {pair: load_cka_summary(cka_root, pair) for pair in PAIR_LABELS}
    for pair, summary in summaries.items():
        left_names = [name for name in summary["left_representation_names"] if name != "classification_head_input"]
        right_names = [name for name in summary["right_representation_names"] if name != "classification_head_input"]
        left_indices = [summary["left_representation_names"].index(name) for name in left_names]
        right_indices = [summary["right_representation_names"].index(name) for name in right_names]
        matrix = np.asarray(summary["full_cross_layer_cka"]["all"])[np.ix_(left_indices, right_indices)]
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(right_names)), right_names, rotation=35, ha="right")
        ax.set_yticks(range(len(left_names)), left_names)
        ax.set_xlabel("Right representation")
        ax.set_ylabel("Left representation")
        decorate_axis(ax, title=f"{PAIR_LABELS[pair]}: all events")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                ax.text(column, row, f"{value:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if value < 0.55 else "black")
        fig.colorbar(image, ax=ax, label="Linear centred CKA")
        fig.tight_layout()
        save_figure(fig, output, f"cka_inclusive_{pair}")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True, sharey=True)
    for ax, subset in zip(axes.flat, SUBSET_LABELS):
        for pair, summary in summaries.items():
            values = [summary["corresponding_layer_cka"][subset][name] for name in REPRESENTATIONS]
            ax.plot(REPRESENTATIONS, values, marker="o", linewidth=1.8, label=PAIR_LABELS[pair])
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Linear centred CKA")
        ax.tick_params(axis="x", rotation=25)
        decorate_axis(ax, title=SUBSET_LABELS[subset])
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"{title}: corresponding-layer CKA", y=1.01)
    fig.tight_layout()
    save_figure(fig, output, "cka_corresponding_layers")

    full = json.loads((full_test_dir / "full_test_final_event_cka.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharey=True)
    x = np.arange(len(SUBSET_LABELS))
    for ax, (pair, pair_label) in zip(axes, PAIR_LABELS.items()):
        sample_values = [summaries[pair]["corresponding_layer_cka"][subset]["final_event"] for subset in SUBSET_LABELS]
        full_values = [full["values"][pair][subset] for subset in SUBSET_LABELS]
        ax.plot(x, sample_values, marker="o", linewidth=1.8, label="100k test")
        ax.plot(x, full_values, marker="s", linewidth=1.8, label="Full test")
        ax.set_xticks(x, [SUBSET_LABELS[name].replace("Signal ", "Signal\n") for name in SUBSET_LABELS], rotation=20)
        ax.set_ylim(0, 1.02)
        decorate_axis(ax, title=pair_label)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Final-event CKA")
    fig.suptitle(f"{title}: CKA sample-size stability", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, "cka_100k_vs_full_test")


def plot_alignment_diagnostics(root: Path, output: Path, title: str) -> None:
    directions = (
        "reconstruction_to_classification",
        "reconstruction_to_joint",
        "joint_to_classification",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for index, direction in enumerate(directions):
        summary = json.loads((root / direction / "alignment_summary.json").read_text(encoding="utf-8"))
        paired = summary["paired"]
        shuffled = summary["shuffled"]
        residuals = np.asarray([row["normalised_alignment_residual"] for row in shuffled])
        cosines = np.asarray([row["mean_aligned_cosine_similarity"] for row in shuffled])
        x = index + np.linspace(-0.16, 0.16, len(shuffled))
        axes[0].scatter(x, residuals, s=18, color=SHUFFLED_COLOUR, alpha=0.75)
        axes[0].scatter(index, paired["normalised_alignment_residual"], s=65, marker="D", color=ALIGNED_COLOUR)
        axes[1].scatter(x, cosines, s=18, color=SHUFFLED_COLOUR, alpha=0.75)
        axes[1].scatter(index, paired["mean_aligned_cosine_similarity"], s=65, marker="D", color=ALIGNED_COLOUR)
    labels = [direction.replace("_", "\n") for direction in directions]
    for ax, ylabel, panel_title in (
        (axes[0], "Normalised residual", "Alignment residual"),
        (axes[1], "Mean aligned cosine similarity", "Aligned cosine similarity"),
    ):
        ax.set_xticks(range(len(directions)), labels)
        ax.set_ylabel(ylabel)
        decorate_axis(ax, title=panel_title)
    fig.suptitle(f"{title}: paired and shuffled alignment diagnostics", y=1.02)
    fig.tight_layout()
    save_figure(fig, output, "alignment_diagnostics")


def scientific_summary(
    metrics_dir: Path,
    controls_dir: Path,
    cka_root: Path,
    full_test_dir: Path,
    alignments_root: Path,
    output: Path,
) -> None:
    bootstrap = read_csv(metrics_dir / "bootstrap_summary.csv")
    auc_rows = {
        row["method"]: row
        for row in bootstrap
        if row["subset"] == "inclusive_all_signal" and row["metric"] == "roc_auc"
    }
    controls = read_csv(controls_dir / "shuffled_null_summary.csv")
    p_values = {
        row["direction"]: float(row["empirical_p_value"])
        for row in controls
        if row["control_type"] == "shuffled" and row["subset"] == "inclusive_all_signal"
    }
    full = json.loads(
        (full_test_dir / "full_test_final_event_cka.json").read_text(encoding="utf-8")
    )
    rows = []
    for method in METHOD_LABELS:
        if method not in auc_rows:
            continue
        row = auc_rows[method]
        rows.append(
            {
                "quantity": f"AUC: {METHOD_LABELS[method]}",
                "value": float(row["point_estimate"]),
                "interval_95_low": float(row["interval_95_low"]),
                "interval_95_high": float(row["interval_95_high"]),
            }
        )
    for direction, value in p_values.items():
        rows.append(
            {
                "quantity": f"Shuffled empirical p: {direction}",
                "value": value,
                "interval_95_low": "",
                "interval_95_high": "",
            }
        )
    for pair, label in PAIR_LABELS.items():
        sample = load_cka_summary(cka_root, pair)
        sample_value = float(sample["corresponding_layer_cka"]["all"]["final_event"])
        full_value = float(full["values"][pair]["all"])
        rows.extend(
            [
                {
                    "quantity": f"100k-test final-event CKA: {label}",
                    "value": sample_value,
                    "interval_95_low": "",
                    "interval_95_high": "",
                },
                {
                    "quantity": f"Full-test final-event CKA: {label}",
                    "value": full_value,
                    "interval_95_low": "",
                    "interval_95_high": "",
                },
                {
                    "quantity": f"Absolute CKA difference: {label}",
                    "value": abs(full_value - sample_value),
                    "interval_95_low": "",
                    "interval_95_high": "",
                },
            ]
        )
    for direction in (
        "reconstruction_to_classification",
        "reconstruction_to_joint",
        "joint_to_classification",
    ):
        alignment = json.loads(
            (alignments_root / direction / "alignment_summary.json").read_text(encoding="utf-8")
        )["paired"]
        rows.extend(
            [
                {
                    "quantity": f"Paired Procrustes residual: {direction}",
                    "value": float(alignment["normalised_alignment_residual"]),
                    "interval_95_low": "",
                    "interval_95_high": "",
                },
                {
                    "quantity": f"Paired aligned cosine: {direction}",
                    "value": float(alignment["mean_aligned_cosine_similarity"]),
                    "interval_95_low": "",
                    "interval_95_high": "",
                },
            ]
        )
    with (output / "scientific_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "scientific_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fig, ax = plt.subplots(figsize=(11.5, max(6.0, 0.36 * len(rows))))
    ax.axis("off")
    table_data = []
    for row in rows:
        interval = ""
        if row["interval_95_low"] != "":
            interval = f"[{row['interval_95_low']:.5f}, {row['interval_95_high']:.5f}]"
        table_data.append([row["quantity"], f"{row['value']:.6g}", interval])
    table = ax.table(
        cellText=table_data,
        colLabels=["Quantity", "Value", "95% interval"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.22)
    ax.set_title("HyPER representation-transfer production summary", pad=18)
    save_figure(fig, output, "scientific_summary")


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    scores_dir = Path(args.scores_dir).expanduser().resolve()
    metrics_dir = Path(args.metrics_dir).expanduser().resolve()
    controls_dir = Path(args.controls_summary_dir).expanduser().resolve()
    cka_root = Path(args.cka_root).expanduser().resolve()
    full_test_dir = Path(args.full_test_cka_dir).expanduser().resolve()
    alignments_root = Path(args.alignments_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(load_array(scores_dir, "test_truth_class"), dtype=np.int8)
    fm = np.asarray(load_array(scores_dir, "test_truth_fully_matched"), dtype=bool)
    scores = {field: np.asarray(load_array(scores_dir, field), dtype=np.float64) for field in METHOD_LABELS}
    plot_zero_shot(scores, labels, fm, output, args.title)
    plot_transfer_rocs(scores, labels, output, args.title)
    plot_correlations(scores, output, args.title)
    plot_bootstrap(metrics_dir, output, args.title)
    plot_control_nulls(controls_dir, output, args.title)
    plot_cka(cka_root, full_test_dir, output, args.title)
    plot_alignment_diagnostics(alignments_root, output, args.title)
    scientific_summary(metrics_dir, controls_dir, cka_root, full_test_dir, alignments_root, output)
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
