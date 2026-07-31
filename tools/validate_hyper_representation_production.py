#!/usr/bin/env python3
"""Validate final HyPER representation-transfer production outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

DIRECTIONS = (
    "reconstruction_to_classification",
    "reconstruction_to_joint",
    "joint_to_classification",
)
PAIRS = (
    "classification_vs_reconstruction",
    "classification_vs_joint",
    "reconstruction_vs_joint",
)
REQUIRED_PLOTS = (
    "zero_shot_score_distributions",
    "zero_shot_roc",
    "zero_shot_background_rejection",
    "main_transfer_roc",
    "bridge_transfer_roc",
    "auc_bootstrap_summary",
    "bootstrap_auc_differences",
    "shuffled_alignment_nulls",
    "cka_corresponding_layers",
    "cka_100k_vs_full_test",
    "alignment_diagnostics",
    "score_correlation_reconstruction_to_classification_paired",
    "score_correlation_joint_to_classification_paired",
    "score_correlation_reconstruction_to_joint_paired",
    "cka_inclusive_classification_vs_reconstruction",
    "cka_inclusive_classification_vs_joint",
    "cka_inclusive_reconstruction_vs_joint",
    "scientific_summary",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--alignment-event-count", type=int, default=100000)
    parser.add_argument("--cka-event-count", type=int, default=100000)
    parser.add_argument("--test-event-count", type=int, default=912666)
    parser.add_argument("--shuffled-alignments", type=int, default=50)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--random-controls", type=int, default=20)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required non-empty file is missing: {path}")
    return path


def load_json(path: Path) -> dict:
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    representations = root / "representations"
    for split, count in (("val", args.alignment_event_count), ("test", args.cka_event_count)):
        for mode in ("classification", "reconstruction", "joint"):
            path = require_file(representations / f"{mode}_{split}.npz")
            with np.load(path, allow_pickle=False) as loaded:
                required = {
                    "source_event_index", "truth_class", "truth_fully_matched",
                    "block_0", "block_1", "block_2", "final_event", "classification_head_input",
                }
                missing = required - set(loaded.files)
                if missing:
                    raise KeyError(f"{path} is missing {sorted(missing)}.")
                if len(loaded["source_event_index"]) != count:
                    raise RuntimeError(f"{path} has {len(loaded['source_event_index'])} events, expected {count}.")
                if not np.array_equal(loaded["final_event"], loaded["classification_head_input"]):
                    raise RuntimeError(f"{path}: final_event and classification_head_input differ.")
                indices = loaded["source_event_index"]
                if len(np.unique(indices)) != count:
                    raise RuntimeError(f"{path}: source_event_index is not unique.")
                expected_dimensions = {
                    "block_0": 128, "block_1": 128, "block_2": 128,
                    "final_event": 1024, "classification_head_input": 1024,
                }
                for name, dimension in expected_dimensions.items():
                    value = loaded[name]
                    if value.shape != (count, dimension):
                        raise RuntimeError(f"{path}: {name} shape is {value.shape}.")
                    if not np.isfinite(value).all():
                        raise RuntimeError(f"{path}: {name} contains non-finite values.")

    alignments_root = root / "alignments"
    for direction in DIRECTIONS:
        summary = load_json(alignments_root / direction / "alignment_summary.json")
        if summary["fit_event_count"] != args.alignment_event_count:
            raise RuntimeError(f"{direction}: wrong fit event count.")
        if summary["num_shuffled_alignments"] != args.shuffled_alignments:
            raise RuntimeError(f"{direction}: wrong shuffled alignment count.")
        if summary.get("labels_loaded_for_fitting") is not False:
            raise RuntimeError(f"{direction}: labels_loaded_for_fitting is not false.")
        seeds = summary["shuffle_seeds"]
        if len(seeds) != len(set(seeds)) or len(seeds) != args.shuffled_alignments:
            raise RuntimeError(f"{direction}: shuffled seeds are incomplete or duplicated.")
        paired_summary = summary["paired"]
        if not np.isfinite(paired_summary["orthogonality_error"]) or paired_summary["orthogonality_error"] > 1e-7:
            raise RuntimeError(f"{direction}: paired rotation is not sufficiently orthogonal.")
        require_file(alignments_root / direction / "paired.npz")
        require_file(alignments_root / direction / "paired.json")
        for seed in seeds:
            require_file(alignments_root / direction / "shuffled" / f"seed_{int(seed):03d}.npz")
            shuffled_summary = load_json(
                alignments_root / direction / "shuffled" / f"seed_{int(seed):03d}.json"
            )
            if shuffled_summary.get("labels_loaded_for_fitting") is not False:
                raise RuntimeError(f"{direction}/seed {seed}: labels flag is not false.")
            if shuffled_summary.get("target_permutation_hash") in (None, ""):
                raise RuntimeError(f"{direction}/seed {seed}: permutation hash is missing.")
            if not np.isfinite(shuffled_summary["orthogonality_error"]) or shuffled_summary["orthogonality_error"] > 1e-7:
                raise RuntimeError(f"{direction}/seed {seed}: rotation is not sufficiently orthogonal.")

    cka_root = root / "cka"
    for pair in PAIRS:
        summary = load_json(cka_root / pair / "cka_summary.json")
        if summary["event_count"] != args.cka_event_count:
            raise RuntimeError(f"{pair}: CKA event count is {summary['event_count']}.")
        if summary.get("classification_head_input_alias") != "final_event":
            raise RuntimeError(f"{pair}: final-event alias metadata is missing.")
        for matrix in summary["full_cross_layer_cka"].values():
            values = np.asarray(matrix, dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < -1e-10) or np.any(values > 1.0 + 1e-10):
                raise RuntimeError(f"{pair}: CKA matrix contains invalid values.")
        require_file(cka_root / pair / "cka_matrix.csv")
        require_file(cka_root / pair / "cka_corresponding_layers.csv")
        for stem in ("cka_heatmap", "cka_inclusive_heatmap"):
            require_file(cka_root / pair / f"{stem}.pdf")
            require_file(cka_root / pair / f"{stem}.png")

    full_cka_root = root / "full_test_cka"
    full_cka = load_json(full_cka_root / "full_test_final_event_cka.json")
    if full_cka["event_count"] != args.test_event_count:
        raise RuntimeError(f"Full-test CKA has {full_cka['event_count']} events.")
    if full_cka["subset_counts"]["all"] != args.test_event_count:
        raise RuntimeError("Full-test CKA all-event count is inconsistent.")
    full_values = np.asarray(
        [value for pair in full_cka["values"].values() for value in pair.values()],
        dtype=np.float64,
    )
    if not np.isfinite(full_values).all() or np.any(full_values < -1e-10) or np.any(full_values > 1.0 + 1e-10):
        raise RuntimeError("Full-test CKA contains invalid values.")
    for name in ("full_test_final_event_cka.csv", "full_test_final_event_cka.pdf", "full_test_final_event_cka.png"):
        require_file(full_cka_root / name)

    score_root = root / "scores"
    evaluation = load_json(score_root / "evaluation_summary.json")
    if evaluation["event_count"] != args.test_event_count:
        raise RuntimeError(f"Evaluation has {evaluation['event_count']} events.")
    if evaluation.get("labels_used_for_alignment") is not False:
        raise RuntimeError("Evaluation does not identify the alignment as label-free.")
    if evaluation.get("num_random_controls") != args.random_controls:
        raise RuntimeError("Evaluation uses the wrong random-control count.")
    canonical = score_root / "scores"
    source_indices = np.load(require_file(canonical / "test_source_event_index.npy"), mmap_mode="r")
    if len(source_indices) != args.test_event_count or len(np.unique(source_indices)) != len(source_indices):
        raise RuntimeError("Canonical full-test source-event index is incomplete or non-unique.")
    for stem in ("test_truth_class", "test_truth_fully_matched", *evaluation["principal_score_fields"]):
        values = np.load(require_file(canonical / f"{stem}.npy"), mmap_mode="r")
        if values.shape != (args.test_event_count,):
            raise RuntimeError(f"{stem} shape is {values.shape}.")
        if stem in evaluation["principal_score_fields"] and not np.isfinite(values).all():
            raise RuntimeError(f"{stem} contains non-finite values.")
    labels = np.load(canonical / "test_truth_class.npy", mmap_mode="r")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise RuntimeError("Full-test truth labels do not contain both classes.")
    controls = score_root / "controls"
    for direction in DIRECTIONS:
        shuffled = np.load(require_file(controls / f"{direction}_shuffled_scores.npy"), mmap_mode="r")
        random = np.load(require_file(controls / f"{direction}_random_scores.npy"), mmap_mode="r")
        if shuffled.shape != (args.shuffled_alignments, args.test_event_count):
            raise RuntimeError(f"{direction} shuffled score shape is {shuffled.shape}.")
        if random.shape != (args.random_controls, args.test_event_count):
            raise RuntimeError(f"{direction} random score shape is {random.shape}.")

    statistics = root / "metrics"
    require_file(statistics / "operating_points.csv")
    bootstrap = load_json(statistics / "bootstrap_summary.json")
    if bootstrap["replicates"] != args.bootstrap_replicates:
        raise RuntimeError(f"Bootstrap has {bootstrap['replicates']} replicates.")
    if bootstrap["event_count"] != args.test_event_count:
        raise RuntimeError("Bootstrap uses the wrong event count.")
    null = load_json(statistics / "shuffled_null_summary.json")
    if null["expected_shuffled_alignments_per_direction"] != args.shuffled_alignments:
        raise RuntimeError("Null summary uses the wrong shuffled alignment count.")
    if null["event_count"] != args.test_event_count:
        raise RuntimeError("Null summary uses the wrong event count.")

    plots = root / "plots"
    for stem in REQUIRED_PLOTS:
        require_file(plots / f"{stem}.pdf")
        require_file(plots / f"{stem}.png")
    require_file(plots / "scientific_summary.csv")
    require_file(plots / "scientific_summary.json")

    report = {
        "output_root": str(root),
        "alignment_event_count": args.alignment_event_count,
        "cka_event_count": args.cka_event_count,
        "test_event_count": args.test_event_count,
        "shuffled_alignments_per_direction": args.shuffled_alignments,
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_controls_per_direction": args.random_controls,
        "status": "valid",
    }
    (root / "production_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
