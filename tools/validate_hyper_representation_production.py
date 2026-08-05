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

ALLOWED_UNDEFINED_REASONS = {
    "left_zero_variance",
    "right_zero_variance",
    "both_zero_variance",
    "insufficient_events",
}
FORBIDDEN_UNDEFINED_REPRESENTATIONS = {"final_event", "classification_head_input"}

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
ESSENTIAL_PLOTS = (
    "native_multitask_effect",
    "zero_shot_score_distributions",
    "shuffled_alignment_nulls",
    "representation_geometry",
)
FULL_EXTRA_PLOTS = (
    "zero_shot_roc",
    "zero_shot_background_rejection",
    "main_transfer_roc",
    "bridge_transfer_roc",
    "auc_bootstrap_summary",
    "bootstrap_auc_differences",
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
    parser.add_argument("--topology", choices=("ttbar1L", "ttH"), required=True)
    parser.add_argument("--plot-set", choices=("essential", "full"), default="essential")
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--expected-profile-hash", default=None)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required non-empty file is missing: {path}")
    return path


def load_json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"Non-standard JSON constant {value!r} in {path}.")
    return json.loads(require_file(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def _finite_or_none(value) -> bool:
    return value is None or bool(np.isfinite(float(value)))


def validate_cka_summary(
    directory: Path,
    expected_events: int,
    *,
    reference_dimensions: dict,
    expected_commit: str | None = None,
    expected_profile_hash: str | None = None,
) -> int:
    summary = load_json(directory / "cka_summary.json")
    if summary.get("event_count") != expected_events:
        raise RuntimeError(f"{directory}: CKA event count is {summary.get('event_count')}.")
    if summary.get("classification_head_input_alias") != "final_event":
        raise RuntimeError(f"{directory}: final-event alias metadata is missing.")
    if summary.get("variance_policy", {}).get("exact_repeated_rows_are_zero_variance") is not True:
        raise RuntimeError(f"{directory}: variance policy is missing exact-degeneracy handling.")
    if expected_commit is not None and summary.get("commit") != expected_commit:
        raise RuntimeError(f"{directory}: CKA commit metadata is incompatible.")
    if expected_profile_hash is not None and summary.get("profile_hash") != expected_profile_hash:
        raise RuntimeError(f"{directory}: CKA profile metadata is incompatible.")
    names_left = list(summary.get("left_representation_names", []))
    names_right = list(summary.get("right_representation_names", []))
    matrices = summary.get("full_cross_layer_cka")
    diagnostics = summary.get("representation_diagnostics", {})
    documented = {}
    for cell in summary.get("undefined_cells", []):
        key = (cell.get("subset"), cell.get("left_representation"), cell.get("right_representation"))
        if key in documented:
            raise RuntimeError(f"{directory}: duplicate undefined-cell documentation for {key}.")
        documented[key] = cell
        if cell.get("reason") not in ALLOWED_UNDEFINED_REASONS:
            raise RuntimeError(f"{directory}: unsupported undefined reason {cell.get('reason')!r}.")
        if cell.get("left_representation") == "classification_head_input" or cell.get("right_representation") == "classification_head_input":
            raise RuntimeError(f"{directory}: classification-head input CKA is undefined.")
        if not _finite_or_none(cell.get("left_centered_norm")) or not _finite_or_none(cell.get("right_centered_norm")):
            raise RuntimeError(f"{directory}: undefined-cell diagnostics are non-finite.")
    expected_undefined = set()
    for subset, matrix in matrices.items():
        if len(matrix) != len(names_left) or any(len(row) != len(names_right) for row in matrix):
            raise RuntimeError(f"{directory}: CKA matrix shape is inconsistent with representation names.")
        if subset not in diagnostics:
            raise RuntimeError(f"{directory}: diagnostics are missing subset {subset!r}.")
        count = int(summary.get("subset_counts", {}).get(subset, -1))
        for i, left_name in enumerate(names_left):
            for j, right_name in enumerate(names_right):
                value = matrix[i][j]
                key = (subset, left_name, right_name)
                if value is None:
                    expected_undefined.add(key)
                    cell = documented.get(key)
                    if cell is None:
                        raise RuntimeError(f"{directory}: null CKA cell {key} has no documented reason.")
                    reason = cell["reason"]
                    if reason == "insufficient_events":
                        if count >= 2:
                            raise RuntimeError(f"{directory}: insufficient-event reason on subset with {count} events.")
                    else:
                        left_diag = diagnostics[subset]["left"][left_name]
                        right_diag = diagnostics[subset]["right"][right_name]
                        left_zero = bool(left_diag.get("zero_variance"))
                        right_zero = bool(right_diag.get("zero_variance"))
                        expected_reason = (
                            "both_zero_variance" if left_zero and right_zero else
                            "left_zero_variance" if left_zero else
                            "right_zero_variance" if right_zero else None
                        )
                        if reason != expected_reason:
                            raise RuntimeError(f"{directory}: reason {reason!r} disagrees with diagnostics for {key}.")
                        if (left_name in FORBIDDEN_UNDEFINED_REPRESENTATIONS and left_zero) or (right_name in FORBIDDEN_UNDEFINED_REPRESENTATIONS and right_zero):
                            raise RuntimeError(f"{directory}: a final/head representation is itself zero-variance for {key}.")
                        for side, diag, field in (("left", left_diag, "left_centered_norm"), ("right", right_diag, "right_centered_norm")):
                            if bool(diag.get("zero_variance")) and float(diag["centred_frobenius_norm"]) > float(diag["variance_threshold"]) * np.sqrt(max(1, count * int(diag["feature_dimension"]))):
                                raise RuntimeError(f"{directory}: {side} zero-variance reason has a nonzero norm for {key}.")
                    continue
                if not np.isfinite(float(value)) or float(value) < -1e-10 or float(value) > 1.0 + 1e-10:
                    raise RuntimeError(f"{directory}: defined CKA cell {key} is invalid: {value!r}.")
    if expected_undefined != set(documented):
        raise RuntimeError(f"{directory}: undefined-cell documentation does not match matrix nulls.")

    with require_file(directory / "cka_matrix.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = len(names_left) * len(names_right) * len(matrices)
    if len(rows) != expected_rows:
        raise RuntimeError(f"{directory}: CSV has {len(rows)} rows, expected {expected_rows}.")
    for row in rows:
        key = (row["subset"], row["left_representation"], row["right_representation"])
        value = matrices[row["subset"]][names_left.index(row["left_representation"])][names_right.index(row["right_representation"])]
        csv_value = row.get("cka", "")
        if value is None:
            if csv_value != "" or row.get("status") != "undefined" or row.get("reason") != documented[key]["reason"]:
                raise RuntimeError(f"{directory}: CSV/JSON disagreement for undefined cell {key}.")
        else:
            if row.get("status") != "defined" or row.get("reason", "") or not csv_value or not np.isclose(float(csv_value), float(value), rtol=1e-12, atol=1e-12):
                raise RuntimeError(f"{directory}: CSV/JSON disagreement for defined cell {key}.")
    return len(expected_undefined)


def main() -> int:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    submission_summary_path = root / "submission_summary.json"
    if args.expected_commit is not None or args.expected_profile_hash is not None:
        submission = load_json(submission_summary_path)
        if args.expected_commit is not None and submission.get("commit") != args.expected_commit:
            raise RuntimeError(
                f"Submission commit {submission.get('commit')!r} differs from expected {args.expected_commit!r}."
            )
        if args.expected_profile_hash is not None and submission.get("profile_hash") != args.expected_profile_hash:
            raise RuntimeError("Submission profile hash differs from the expected profile hash.")
        if submission.get("topology") != args.topology or submission.get("plot_set") != args.plot_set:
            raise RuntimeError("Submission metadata is incompatible with the validation request.")
    representations = root / "representations"
    reference_dimensions = None
    split_indices = {}
    health_reports = {}
    for split, count in (("val", args.alignment_event_count), ("test", args.cka_event_count)):
        split_dimensions = {}
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
                if split in split_indices and not np.array_equal(indices, split_indices[split]):
                    raise RuntimeError(f"{path}: source_event_index order differs from the other model exports.")
                split_indices[split] = np.asarray(indices)
                dimensions = {}
                for name in ("block_0", "block_1", "block_2", "final_event", "classification_head_input"):
                    value = loaded[name]
                    if value.ndim != 2 or value.shape[0] != count or value.shape[1] <= 0:
                        raise RuntimeError(f"{path}: {name} shape is {value.shape}.")
                    if not np.isfinite(value).all():
                        raise RuntimeError(f"{path}: {name} contains non-finite values.")
                    dimensions[name] = int(value.shape[1])
                if dimensions["classification_head_input"] != dimensions["final_event"]:
                    raise RuntimeError(f"{path}: classification-head and final-event dimensions differ.")
                health_path = representations / f"{mode}_{split}_health.json"
                health = load_json(health_path)
                if health.get("event_count") != count or health.get("split") != split:
                    raise RuntimeError(f"{health_path}: health metadata does not match {split} export.")
                for name, diagnostic in health.get("representations", {}).items():
                    if diagnostic.get("finite_fraction") != 1.0:
                        raise RuntimeError(f"{health_path}: {name} is not fully finite.")
                if health.get("representations", {}).get("final_event", {}).get("zero_variance"):
                    raise RuntimeError(f"{health_path}: complete final_event representation is globally degenerate.")
                health_reports[f"{mode}_{split}"] = health
                split_dimensions[mode] = dimensions
        if len({tuple(item.items()) for item in split_dimensions.values()}) != 1:
            raise RuntimeError(f"{split}: model representation dimensions differ: {split_dimensions}.")
        dimensions = next(iter(split_dimensions.values()))
        if reference_dimensions is None:
            reference_dimensions = dimensions
        elif dimensions != reference_dimensions:
            raise RuntimeError(
                f"Validation/test representation dimensions differ: {reference_dimensions} vs {dimensions}."
            )

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
    undefined_count = 0
    for pair in PAIRS:
        undefined_count += validate_cka_summary(cka_root / pair, args.cka_event_count, reference_dimensions=reference_dimensions)
        require_file(cka_root / pair / "cka_matrix.csv")
        require_file(cka_root / pair / "cka_corresponding_layers.csv")
        for stem in ("cka_heatmap", "cka_inclusive_heatmap"):
            require_file(cka_root / pair / f"{stem}.pdf")
            require_file(cka_root / pair / f"{stem}.png")

    full_cka_root = root / "full_test_cka"
    full_cka = load_json(full_cka_root / "full_test_final_event_cka.json")
    if full_cka["event_count"] != args.test_event_count:
        raise RuntimeError(f"Full-test CKA has {full_cka['event_count']} events.")
    if int(full_cka.get("representation_dimension", -1)) != int(reference_dimensions["final_event"]):
        raise RuntimeError(
            f"Full-test CKA dimension {full_cka.get('representation_dimension')} differs from "
            f"the exported final-event dimension {reference_dimensions['final_event']}."
        )
    if full_cka["subset_counts"]["all"] != args.test_event_count:
        raise RuntimeError("Full-test CKA all-event count is inconsistent.")
    full_values = np.asarray([value for pair in full_cka["values"].values() for value in pair.values()], dtype=np.float64)
    if not np.isfinite(full_values).all() or np.any(full_values < -1e-10) or np.any(full_values > 1.0 + 1e-10):
        raise RuntimeError("Full-test CKA contains invalid values.")
    for name in ("full_test_final_event_cka.csv", "full_test_final_event_cka.pdf", "full_test_final_event_cka.png"):
        require_file(full_cka_root / name)

    score_root = root / "scores"
    evaluation = load_json(score_root / "evaluation_summary.json")
    if evaluation["event_count"] != args.test_event_count:
        raise RuntimeError(f"Evaluation has {evaluation['event_count']} events.")
    if evaluation.get("topology", "ttbar1L") != args.topology:
        raise RuntimeError(
            f"Evaluation topology {evaluation.get('topology', 'ttbar1L')!r} differs from {args.topology!r}."
        )
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
    required_plots = ESSENTIAL_PLOTS if args.plot_set == "essential" else ESSENTIAL_PLOTS + FULL_EXTRA_PLOTS
    for stem in required_plots:
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
        "topology": args.topology,
        "plot_set": args.plot_set,
        "undefined_cka_cells": undefined_count,
        "representation_health_reports": sorted(health_reports),
        "status": "valid",
    }
    (root / "production_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
