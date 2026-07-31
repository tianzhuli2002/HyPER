#!/usr/bin/env python3
"""Compute source-aligned linear centred CKA between HyPER representation exports."""

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

from HyPER.analysis.representations import align_exports, event_index_sha256, linear_cka
from HyPER.topology.plot_style import configure_matplotlib, decorate_axis, save_figure


NON_REPRESENTATIONS = {
    "source_event_index", "truth_class", "truth_fully_matched", "prediction_split",
    "model_name", "config_path", "checkpoint_path", "native_classification_logit",
    "native_classification_probability", "native_reconstruction_score",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--common-events-only", action="store_true")
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=("all", "background", "signal_fully_matched", "signal_non_fully_matched"),
        default=("all", "background", "signal_fully_matched", "signal_non_fully_matched"),
    )
    return parser.parse_args()


def load_export(path):
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def representation_names(export):
    names = [
        name for name, values in export.items()
        if name not in NON_REPRESENTATIONS and np.asarray(values).ndim == 2
    ]
    if "final_event" in names and "classification_head_input" in names:
        names.remove("classification_head_input")
    preferred = ["block_0", "block_1", "block_2", "final_event"]
    return [name for name in preferred if name in names] + [name for name in names if name not in preferred]


def main() -> int:
    args = parse_args()
    left, right = load_export(args.left), load_export(args.right)
    indices, left_rows, right_rows = align_exports(
        left, right, common_events_only=args.common_events_only
    )
    left_names, right_names = representation_names(left), representation_names(right)
    if not left_names or not right_names:
        raise ValueError("Both exports must contain at least one 2D representation array.")
    truth = np.asarray(left["truth_class"])[left_rows]
    fully_matched = np.asarray(left["truth_fully_matched"])[left_rows].astype(bool)
    all_subsets = {
        "all": np.ones(len(indices), dtype=bool),
        "background": truth == 0,
        "signal_fully_matched": (truth == 1) & fully_matched,
        "signal_non_fully_matched": (truth == 1) & ~fully_matched,
    }
    subsets = {name: all_subsets[name] for name in args.subsets}
    matrices, rows = {}, []
    for subset_name, mask in subsets.items():
        count = int(mask.sum())
        if count < 2:
            raise ValueError(f"CKA subset {subset_name!r} has only {count} events.")
        matrix = np.empty((len(left_names), len(right_names)), dtype=np.float64)
        for i, left_name in enumerate(left_names):
            for j, right_name in enumerate(right_names):
                value = linear_cka(
                    np.asarray(left[left_name])[left_rows][mask],
                    np.asarray(right[right_name])[right_rows][mask],
                )
                matrix[i, j] = value
                rows.append((subset_name, count, left_name, right_name, value))
        matrices[subset_name] = matrix
    common_names = [name for name in left_names if name in right_names]
    summary = {
        "title": args.title,
        "left_path": str(Path(args.left).resolve()), "right_path": str(Path(args.right).resolve()),
        "left_model": str(left.get("model_name", "left")), "right_model": str(right.get("model_name", "right")),
        "left_representation_names": left_names, "right_representation_names": right_names,
        "event_count": int(len(indices)), "subset_counts": {k: int(v.sum()) for k, v in subsets.items()},
        "event_index_hash": event_index_sha256(indices),
        "corresponding_layer_cka": {
            subset: {name: float(matrix[left_names.index(name), right_names.index(name)]) for name in common_names}
            for subset, matrix in matrices.items()
        },
        "full_cross_layer_cka": {subset: matrix.tolist() for subset, matrix in matrices.items()},
        "definition": "||X_centered.T @ Y_centered||_F^2 / sqrt(||X_centered.T @ X_centered||_F^2 ||Y_centered.T @ Y_centered||_F^2)",
        "stored_feature_dtype": "float32", "calculation_dtype": "float64",
        "classification_head_input_alias": "final_event",
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cka_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "cka_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subset", "event_count", "left_representation", "right_representation", "cka"))
        writer.writerows(rows)
    configure_matplotlib()
    nplots = len(matrices)
    ncols = min(2, nplots)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.0 * nrows), constrained_layout=True, squeeze=False)
    image = None
    for ax, (subset_name, matrix) in zip(axes.flat, matrices.items()):
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(right_names)), right_names, rotation=35, ha="right")
        ax.set_yticks(range(len(left_names)), left_names)
        ax.set_xlabel("Right representation")
        ax.set_ylabel("Left representation")
        decorate_axis(ax, title=f"{subset_name.replace('_', ' ')} ($N={subsets[subset_name].sum():,}$)", minor_ticks=False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.55 else "black", fontsize=8)
    for ax in axes.flat[len(matrices):]:
        ax.set_visible(False)
    fig.suptitle(args.title)
    fig.colorbar(image, ax=[ax for ax in axes.flat if ax.get_visible()], label="Linear centred CKA", shrink=0.85)
    save_figure(fig, output_dir, "cka_heatmap")

    inclusive = matrices["all"]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(inclusive, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(right_names)), right_names, rotation=35, ha="right")
    ax.set_yticks(range(len(left_names)), left_names)
    ax.set_xlabel("Right representation")
    ax.set_ylabel("Left representation")
    decorate_axis(ax, title=f"{args.title}: all events", minor_ticks=False)
    for i in range(inclusive.shape[0]):
        for j in range(inclusive.shape[1]):
            ax.text(j, i, f"{inclusive[i, j]:.3f}", ha="center", va="center",
                    color="white" if inclusive[i, j] < 0.55 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="Linear centred CKA")
    fig.tight_layout()
    save_figure(fig, output_dir, "cka_inclusive_heatmap")

    with (output_dir / "cka_corresponding_layers.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subset", "event_count", "representation", "cka"))
        for subset_name, values in summary["corresponding_layer_cka"].items():
            for representation, value in values.items():
                writer.writerow((subset_name, summary["subset_counts"][subset_name], representation, value))
    print(f"wrote={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
