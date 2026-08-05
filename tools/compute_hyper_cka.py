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
from matplotlib.patches import Patch

from HyPER.analysis.representations import (
    align_exports,
    cka_undefined_reason,
    event_index_sha256,
    linear_cka,
    representation_diagnostics,
)
from HyPER.analysis.runtime import resource_diagnostics, write_resource_diagnostics
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
    parser.add_argument("--commit", default=None)
    parser.add_argument("--profile-hash", default=None)
    parser.add_argument(
        "--representations",
        nargs="+",
        default=None,
        help="Optional representation keys to retain; the default uses all exported representations.",
    )
    parser.add_argument("--common-events-only", action="store_true")
    parser.add_argument(
        "--subsets",
        nargs="+",
        choices=("all", "background", "signal_fully_matched", "signal_non_fully_matched"),
        default=("all", "background", "signal_fully_matched", "signal_non_fully_matched"),
    )
    return parser.parse_args()


def representation_names(export):
    names = list(export.files) if hasattr(export, "files") else list(export)
    names = [name for name in names if name not in NON_REPRESENTATIONS]
    names = [name for name in names if name not in {"classification_head_input"}]
    preferred = ["block_0", "block_1", "block_2", "final_event"]
    return [name for name in preferred if name in names] + [
        name for name in names if name not in preferred
    ]


def load_export(path, requested=None):
    with np.load(path, allow_pickle=False) as loaded:
        names = representation_names(loaded)
        if requested is not None:
            missing = [name for name in requested if name not in names]
            if missing:
                raise KeyError(f"{path}: requested representations are missing: {missing}")
            names = [name for name in names if name in requested]
        required = ("source_event_index", "truth_class", "truth_fully_matched")
        result = {name: loaded[name] for name in required}
        result.update({name: loaded[name] for name in names})
        for name in ("model_name", "config_path", "checkpoint_path"):
            if name in loaded:
                result[name] = loaded[name]
        return result


def _matrix_for_plot(matrix):
    values = np.full((len(matrix), len(matrix[0])), np.nan, dtype=np.float64)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value is not None:
                values[i, j] = float(value)
    return np.ma.masked_invalid(values)


def _draw_matrix(ax, matrix, left_names, right_names):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#bdbdbd")
    image = ax.imshow(_matrix_for_plot(matrix), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(right_names)), right_names, rotation=35, ha="right")
    ax.set_yticks(range(len(left_names)), left_names)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value is None:
                ax.text(j, i, "N/A", ha="center", va="center", color="black", fontsize=8)
            else:
                value = float(value)
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    return image


def main() -> int:
    args = parse_args()
    import time
    started = time.perf_counter()
    left, right = load_export(args.left, args.representations), load_export(args.right, args.representations)
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
    diagnostics_by_subset = {}
    undefined_cells = []
    for subset_name, mask in subsets.items():
        count = int(mask.sum())
        left_subset = {
            name: np.asarray(left[name])[left_rows][mask] for name in left_names
        }
        right_subset = {
            name: np.asarray(right[name])[right_rows][mask] for name in right_names
        }
        left_diagnostics = {
            name: representation_diagnostics(value) if count >= 2 else {
                "event_count": count,
                "feature_dimension": int(value.shape[1]),
                "finite_fraction": float(np.mean(np.isfinite(value))) if value.size else 1.0,
                "centred_frobenius_norm": None,
                "active_dimension_count": None,
                "minimum_feature_standard_deviation": None,
                "maximum_feature_standard_deviation": None,
                "unique_row_count": None,
                "numerical_rank": None,
                "effective_rank": None,
                "variance_threshold": None,
                "zero_variance": False,
                "zero_variance_detection": "insufficient_events",
            }
            for name, value in left_subset.items()
        }
        right_diagnostics = {
            name: representation_diagnostics(value) if count >= 2 else {
                "event_count": count,
                "feature_dimension": int(value.shape[1]),
                "finite_fraction": float(np.mean(np.isfinite(value))) if value.size else 1.0,
                "centred_frobenius_norm": None,
                "active_dimension_count": None,
                "minimum_feature_standard_deviation": None,
                "maximum_feature_standard_deviation": None,
                "unique_row_count": None,
                "numerical_rank": None,
                "effective_rank": None,
                "variance_threshold": None,
                "zero_variance": False,
                "zero_variance_detection": "insufficient_events",
            }
            for name, value in right_subset.items()
        }
        diagnostics_by_subset[subset_name] = {"left": left_diagnostics, "right": right_diagnostics}
        matrix = [[None for _ in right_names] for _ in left_names]
        for i, left_name in enumerate(left_names):
            for j, right_name in enumerate(right_names):
                reason = "insufficient_events" if count < 2 else cka_undefined_reason(
                    left_diagnostics[left_name], right_diagnostics[right_name]
                )
                if reason is None:
                    value = linear_cka(left_subset[left_name], right_subset[right_name])
                    matrix[i][j] = value
                    status = "defined"
                else:
                    value = None
                    status = "undefined"
                    undefined_cells.append(
                        {
                            "subset": subset_name,
                            "left_representation": left_name,
                            "right_representation": right_name,
                            "reason": reason,
                            "left_centered_norm": left_diagnostics[left_name]["centred_frobenius_norm"],
                            "right_centered_norm": right_diagnostics[right_name]["centred_frobenius_norm"],
                            "left_active_dimensions": left_diagnostics[left_name]["active_dimension_count"],
                            "right_active_dimensions": right_diagnostics[right_name]["active_dimension_count"],
                        }
                    )
                rows.append({
                    "subset": subset_name,
                    "event_count": count,
                    "left_representation": left_name,
                    "right_representation": right_name,
                    "cka": value,
                    "status": status,
                    "reason": "" if reason is None else reason,
                    "left_centered_norm": left_diagnostics[left_name]["centred_frobenius_norm"],
                    "right_centered_norm": right_diagnostics[right_name]["centred_frobenius_norm"],
                    "left_active_dimensions": left_diagnostics[left_name]["active_dimension_count"],
                    "right_active_dimensions": right_diagnostics[right_name]["active_dimension_count"],
                })
        matrices[subset_name] = matrix
    common_names = [name for name in left_names if name in right_names]
    summary = {
        "title": args.title,
        "commit": args.commit,
        "profile_hash": args.profile_hash,
        "left_path": str(Path(args.left).resolve()), "right_path": str(Path(args.right).resolve()),
        "left_model": str(left.get("model_name", "left")), "right_model": str(right.get("model_name", "right")),
        "left_representation_names": left_names, "right_representation_names": right_names,
        "event_count": int(len(indices)), "subset_counts": {k: int(v.sum()) for k, v in subsets.items()},
        "event_index_hash": event_index_sha256(indices),
        "corresponding_layer_cka": {
            subset: {name: matrices[subset][left_names.index(name)][right_names.index(name)] for name in common_names}
            for subset, matrix in matrices.items()
        },
        "full_cross_layer_cka": matrices,
        "representation_diagnostics": diagnostics_by_subset,
        "undefined_cells": undefined_cells,
        "variance_policy": {
            "relative_tolerance": 1e-12,
            "roundoff_factor": 64.0,
            "exact_repeated_rows_are_zero_variance": True,
        },
        "definition": "||X_centered.T @ Y_centered||_F^2 / sqrt(||X_centered.T @ X_centered||_F^2 ||Y_centered.T @ Y_centered||_F^2)",
        "stored_feature_dtype": "float32", "calculation_dtype": "float64",
        "classification_head_input_alias": "final_event",
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cka_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "cka_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        fields = tuple(rows[0])
        writer.writerow(fields)
        for row in rows:
            writer.writerow([row[field] if row[field] is not None else "" for field in fields])
    configure_matplotlib()
    nplots = len(matrices)
    ncols = min(2, nplots)
    nrows = int(np.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.0 * nrows), constrained_layout=True, squeeze=False)
    image = None
    for ax, (subset_name, matrix) in zip(axes.flat, matrices.items()):
        image = _draw_matrix(ax, matrix, left_names, right_names)
        ax.set_xlabel("Right representation")
        ax.set_ylabel("Left representation")
        decorate_axis(ax, title=f"{subset_name.replace('_', ' ')} ($N={subsets[subset_name].sum():,}$)", minor_ticks=False)
        ax.legend(handles=[Patch(facecolor="#bdbdbd", edgecolor="none", label="N/A: zero event-to-event variance")], loc="upper left", fontsize=7, frameon=False)
    for ax in axes.flat[len(matrices):]:
        ax.set_visible(False)
    fig.suptitle(args.title)
    fig.text(0.01, 0.01, "Grey / N/A cells are mathematically undefined because a representation has no event-to-event variance.", fontsize=8)
    fig.colorbar(image, ax=[ax for ax in axes.flat if ax.get_visible()], label="Linear centred CKA", shrink=0.85)
    save_figure(fig, output_dir, "cka_heatmap")

    inclusive = matrices["all"]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = _draw_matrix(ax, inclusive, left_names, right_names)
    ax.set_xlabel("Right representation")
    ax.set_ylabel("Left representation")
    decorate_axis(ax, title=f"{args.title}: all events", minor_ticks=False)
    ax.legend(handles=[Patch(facecolor="#bdbdbd", edgecolor="none", label="N/A: zero event-to-event variance")], loc="upper left", fontsize=7, frameon=False)
    fig.colorbar(image, ax=ax, label="Linear centred CKA")
    fig.tight_layout()
    save_figure(fig, output_dir, "cka_inclusive_heatmap")

    with (output_dir / "cka_corresponding_layers.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subset", "event_count", "representation", "cka", "status", "reason", "centered_norm"))
        for subset_name, values in summary["corresponding_layer_cka"].items():
            for representation, value in values.items():
                cell = next(item for item in rows if item["subset"] == subset_name and item["left_representation"] == representation and item["right_representation"] == representation)
                writer.writerow((subset_name, summary["subset_counts"][subset_name], representation, "" if value is None else value, cell["status"], cell["reason"], "" if cell["left_centered_norm"] is None else cell["left_centered_norm"]))
    diagnostics = resource_diagnostics(stage="cka", started=started, events_processed=len(indices), output_root=output_dir)
    diagnostics.update({"left_path": str(Path(args.left).resolve()), "right_path": str(Path(args.right).resolve()), "undefined_cell_count": len(undefined_cells)})
    write_resource_diagnostics(output_dir, diagnostics)
    print(f"wrote={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
