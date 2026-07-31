#!/usr/bin/env python3
"""Calculate full-test metrics and paired stratified bootstrap intervals from saved scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from HyPER.analysis.metrics import (
    SIGNAL_EFFICIENCIES,
    binary_metric_summary,
    bootstrap_strata,
    poisson_stratified_weights,
    prepare_descending_score_groups,
    weighted_grouped_metrics,
)

METHOD_LABELS = {
    "native_classification_only_score": "Native classification-only",
    "native_joint_score": "Native joint",
    "reconstruction_zero_shot_score": "Reconstruction zero-shot",
    "joint_reconstruction_zero_shot_score": "Joint reconstruction zero-shot",
    "reconstruction_to_classification_direct_score": "Reconstruction→classification direct",
    "reconstruction_to_classification_paired_score": "Reconstruction→classification paired",
    "reconstruction_to_joint_direct_score": "Reconstruction→joint direct",
    "reconstruction_to_joint_paired_score": "Reconstruction→joint paired",
    "joint_to_classification_direct_score": "Joint→classification direct",
    "joint_to_classification_paired_score": "Joint→classification paired",
}

DIFFERENCES = {
    "reco_to_class_paired_minus_direct": (
        "reconstruction_to_classification_paired_score",
        "reconstruction_to_classification_direct_score",
    ),
    "reco_to_joint_paired_minus_direct": (
        "reconstruction_to_joint_paired_score",
        "reconstruction_to_joint_direct_score",
    ),
    "joint_to_class_paired_minus_direct": (
        "joint_to_classification_paired_score",
        "joint_to_classification_direct_score",
    ),
    "native_joint_minus_native_classification": (
        "native_joint_score",
        "native_classification_only_score",
    ),
    "joint_zero_shot_minus_reconstruction_zero_shot": (
        "joint_reconstruction_zero_shot_score",
        "reconstruction_zero_shot_score",
    ),
    "joint_to_class_paired_minus_native_classification": (
        "joint_to_classification_paired_score",
        "native_classification_only_score",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-event-count", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=list(METHOD_LABELS))
    return parser.parse_args()


def load_array(directory: Path, stem: str, mmap_mode="r") -> np.ndarray:
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


def interval_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "interval_68_low": float(np.quantile(values, 0.16)),
        "interval_68_high": float(np.quantile(values, 0.84)),
        "interval_95_low": float(np.quantile(values, 0.025)),
        "interval_95_high": float(np.quantile(values, 0.975)),
    }


def main() -> int:
    args = parse_args()
    if args.replicates <= 0 or args.chunk_size <= 0:
        raise ValueError("--replicates and --chunk-size must be positive.")
    scores_dir = Path(args.scores_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(load_array(scores_dir, "test_truth_class"), dtype=np.int8)
    fm = np.asarray(load_array(scores_dir, "test_truth_fully_matched"), dtype=bool)
    if args.expected_event_count is not None and len(labels) != args.expected_event_count:
        raise RuntimeError(f"Loaded {len(labels)} events, expected {args.expected_event_count}.")
    methods = list(args.methods)
    unknown = [name for name in methods if name not in METHOD_LABELS]
    if unknown:
        raise KeyError(f"Unknown methods: {unknown}")
    score_arrays = {name: load_array(scores_dir, name) for name in methods}
    masks = subset_masks(labels, fm)
    point_rows = []
    prepared = {}
    point_lookup = {}
    for subset, mask in masks.items():
        subset_labels = labels[mask]
        for method, scores in score_arrays.items():
            values = np.asarray(scores[mask], dtype=np.float64)
            metrics = binary_metric_summary(subset_labels, values, include_roc=False)
            point_lookup[(subset, method)] = metrics
            row = {
                "subset": subset,
                "method": method,
                "label": METHOD_LABELS[method],
                "event_count": metrics["event_count"],
                "roc_auc": metrics["roc_auc"],
            }
            for efficiency in SIGNAL_EFFICIENCIES:
                op = metrics["operating_points"][f"signal_efficiency_{efficiency:.1f}"]
                row[f"threshold_at_signal_efficiency_{efficiency:.1f}"] = op["threshold"]
                row[f"background_efficiency_at_signal_efficiency_{efficiency:.1f}"] = op["background_efficiency"]
                row[f"background_rejection_at_signal_efficiency_{efficiency:.1f}"] = op["background_rejection"]
            point_rows.append(row)
            prepared[(subset, method)] = prepare_descending_score_groups(subset_labels, values)
    with (output / "full_test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    operating_rows = []
    for row in point_rows:
        for efficiency in SIGNAL_EFFICIENCIES:
            operating_rows.append(
                {
                    "subset": row["subset"],
                    "method": row["method"],
                    "label": row["label"],
                    "target_signal_efficiency": efficiency,
                    "threshold": row[f"threshold_at_signal_efficiency_{efficiency:.1f}"],
                    "background_efficiency": row[
                        f"background_efficiency_at_signal_efficiency_{efficiency:.1f}"
                    ],
                    "background_rejection": row[
                        f"background_rejection_at_signal_efficiency_{efficiency:.1f}"
                    ],
                }
            )
    with (output / "operating_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(operating_rows[0]))
        writer.writeheader()
        writer.writerows(operating_rows)
    (output / "full_test_metrics.json").write_text(
        json.dumps(
            {
                "event_count": len(labels),
                "methods": METHOD_LABELS,
                "metrics": point_rows,
                "operating_points": operating_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    strata = bootstrap_strata(labels, fm)
    rng = np.random.default_rng(args.seed)
    bootstrap_values = {
        (subset, method, metric): np.empty(args.replicates, dtype=np.float64)
        for subset in masks
        for method in methods
        for metric in (
            "roc_auc",
            *[f"background_rejection_at_signal_efficiency_{efficiency:.1f}" for efficiency in SIGNAL_EFFICIENCIES],
        )
    }
    started = time.perf_counter()
    completed = 0
    while completed < args.replicates:
        count = min(args.chunk_size, args.replicates - completed)
        weights = poisson_stratified_weights(len(labels), strata, count, rng)
        for subset, mask in masks.items():
            subset_weights = weights[:, mask]
            for method in methods:
                values = weighted_grouped_metrics(prepared[(subset, method)], subset_weights)
                for metric, result in values.items():
                    if metric.startswith("threshold") or metric.startswith("background_efficiency"):
                        continue
                    bootstrap_values[(subset, method, metric)][completed:completed + count] = result
        completed += count
        elapsed = time.perf_counter() - started
        print(
            f"bootstrap={completed}/{args.replicates} elapsed_seconds={elapsed:.1f} "
            f"replicates_per_second={completed / elapsed:.3f}",
            flush=True,
        )
    np.savez_compressed(
        output / "bootstrap_distributions.npz",
        **{
            f"{subset}__{method}__{metric}": values.astype(np.float32)
            for (subset, method, metric), values in bootstrap_values.items()
        },
    )
    summary_rows = []
    for (subset, method, metric), values in bootstrap_values.items():
        point = (
            point_lookup[(subset, method)]["roc_auc"]
            if metric == "roc_auc"
            else point_lookup[(subset, method)]["operating_points"][
                metric.replace("background_rejection_at_", "")
            ]["background_rejection"]
        )
        summary_rows.append(
            {
                "subset": subset,
                "method": method,
                "metric": metric,
                "point_estimate": point,
                **interval_summary(values),
            }
        )
    difference_rows = []
    difference_payload = {}
    for subset in masks:
        for name, (left, right) in DIFFERENCES.items():
            if left not in methods or right not in methods:
                continue
            values = (
                bootstrap_values[(subset, left, "roc_auc")]
                - bootstrap_values[(subset, right, "roc_auc")]
            )
            point = point_lookup[(subset, left)]["roc_auc"] - point_lookup[(subset, right)]["roc_auc"]
            row = {
                "subset": subset,
                "difference": name,
                "left_method": left,
                "right_method": right,
                "point_estimate": float(point),
                **interval_summary(values),
                "fraction_above_zero": float(np.mean(values > 0)),
            }
            difference_rows.append(row)
            difference_payload[f"{subset}__{name}"] = values.astype(np.float32)
    np.savez_compressed(output / "bootstrap_auc_differences.npz", **difference_payload)
    with (output / "bootstrap_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "bootstrap_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(difference_rows[0]))
        writer.writeheader()
        writer.writerows(difference_rows)
    summary = {
        "event_count": len(labels),
        "replicates": args.replicates,
        "seed": args.seed,
        "bootstrap_kind": "paired stratified Poisson(1) bootstrap",
        "stratum_counts": {
            "background": int(len(strata.background)),
            "signal_fully_matched": int(len(strata.signal_fully_matched)),
            "signal_non_fully_matched": int(len(strata.signal_non_fully_matched)),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "metric_summary": summary_rows,
        "paired_auc_differences": difference_rows,
    }
    (output / "bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote={output} replicates={args.replicates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
