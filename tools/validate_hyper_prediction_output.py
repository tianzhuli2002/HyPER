#!/usr/bin/env python3
"""Validate HyPER prediction outputs against source H5 row alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HyPER.topology.prediction_io import load_hyper_prediction_output  # noqa: E402


PROVENANCE_AND_TRUTH_REQUIRED = {
    "prediction_row_index", "source_event_index", "prediction_split", "source_dataset_name",
    "HyPER_CLS_T", "number_of_nodes", "node_truth_ids", "edge_reconstruction_active",
    "hyperedge_reconstruction_active", "selected_reconstruction_local_indices",
    "selected_reconstruction_scores", "truth_fully_matched", "truth_partially_matched",
    "truth_unmatched",
}

TTH_REQUIRED = PROVENANCE_AND_TRUTH_REQUIRED | {
    "source_event_index",
    "HyPER_best_tlep",
    "HyPER_best_thad",
    "HyPER_best_wlep",
    "HyPER_best_whad",
    "HyPER_best_higgs",
    "HyPER_best_blep",
    "HyPER_best_bhad",
    "HyPER_best_whad_j1",
    "HyPER_best_whad_j2",
    "HyPER_best_higgs_b1",
    "HyPER_best_higgs_b2",
    "reco_valid",
    "tlep_valid",
    "thad_valid",
    "wlep_valid",
    "whad_valid",
    "higgs_valid",
}

TTBAR_REQUIRED = PROVENANCE_AND_TRUTH_REQUIRED | {
    "source_event_index",
    "HyPER_best_top1",
    "HyPER_best_top2",
    "HyPER_best_w1",
    "HyPER_best_w2",
    "HyPER_best_top1_prob",
    "HyPER_best_top2_prob",
    "HyPER_best_w1_prob",
    "HyPER_best_w2_prob",
}

CLASSIFIER_REQUIRED = {
    "source_event_index", "prediction_row_index", "prediction_split", "source_dataset_name",
    "HyPER_CLS_T", "HyPER_CLS_LOGIT", "HyPER_CLS_PROB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-output", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--topology", required=True, choices=["ttH", "tth", "ttbar1L", "ttbar_sl", "ttbar_single_lep", "classifier"])
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--expected-split", default=None)
    parser.add_argument("--expected-source-indices", default=None)
    return parser.parse_args()


def h5_length(path: Path) -> int:
    with h5py.File(path, "r") as handle:
        if "INPUTS" not in handle or "GLOBAL" not in handle["INPUTS"]:
            raise KeyError(f"{path} does not contain INPUTS/GLOBAL")
        return int(len(handle["INPUTS"]["GLOBAL"]))


def read_parts_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not (path.name.endswith(".pkl.parts") or path.is_dir()):
        return None, warnings
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None, [f"Missing parts manifest: {manifest_path}"]
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for part_name in manifest.get("part_files", []):
        if not (path / part_name).exists():
            warnings.append(f"Missing listed part file: {part_name}")
    return manifest, warnings


def required_columns(topology: str) -> set[str]:
    norm = topology.lower()
    if norm in {"tth", "tth"}:
        return TTH_REQUIRED
    if norm in {"ttbar1l", "ttbar_sl", "ttbar_single_lep"}:
        return TTBAR_REQUIRED
    if norm == "classifier":
        return CLASSIFIER_REQUIRED
    raise ValueError(topology)


def finite_fraction(values) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return float(np.isfinite(arr).mean()) if arr.size else float("nan")


def score_range_summary(values) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    return {
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "count_below_0": int((finite_values < 0.0).sum()),
        "count_above_1": int((finite_values > 1.0).sum()),
    }


def main() -> int:
    args = parse_args()
    prediction_path = Path(args.prediction_output)
    h5_path = Path(args.h5)
    errors: list[str] = []
    warnings: list[str] = []

    if not prediction_path.exists():
        errors.append(f"Prediction output does not exist: {prediction_path}")
    if not h5_path.exists():
        errors.append(f"H5 does not exist: {h5_path}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2

    parts_manifest, manifest_warnings = read_parts_manifest(prediction_path)
    warnings.extend(manifest_warnings)
    is_parts = prediction_path.name.endswith(".pkl.parts") or prediction_path.is_dir()

    frame = load_hyper_prediction_output(prediction_path, max_events=args.max_events)
    total_rows_attr = int(frame.attrs.get("hyper_total_rows", len(frame)))
    loaded_rows = int(len(frame))
    source_rows = h5_length(h5_path)

    required = required_columns(args.topology)
    missing = sorted(required.difference(frame.columns))
    if missing:
        errors.append("Missing required columns: " + ", ".join(missing))

    for required_common in ("prediction_row_index", "prediction_split", "source_dataset_name", "source_event_index"):
        if required_common not in frame.columns:
            errors.append(f"Prediction output must contain {required_common}")

    if parts_manifest is not None:
        manifest_rows = parts_manifest.get("number_of_prediction_rows")
        # The parts manifest records the complete output size, while this
        # validator may deliberately materialise only --max-events rows.
        if (
            manifest_rows is not None
            and loaded_rows == int(manifest_rows)
            and int(manifest_rows) != total_rows_attr
        ):
            errors.append(
                f"Manifest n_events={manifest_rows} does not match "
                f"loaded total rows={total_rows_attr}"
            )
    if "prediction_row_index" in frame.columns:
        order_values = np.asarray(frame["prediction_row_index"], dtype=np.int64)
        if not np.array_equal(order_values, np.arange(len(frame))):
            errors.append("prediction_row_index is not contiguous from zero")

    has_source_index = "source_event_index" in frame.columns
    source_index_summary = None
    if has_source_index:
        source_index = np.asarray(frame["source_event_index"], dtype=np.int64)
        if source_index.size:
            unique = bool(len(np.unique(source_index)) == len(source_index))
            in_bounds = bool(source_index.min() >= 0 and source_index.max() < source_rows)
            source_index_summary = {
                "min": int(source_index.min()),
                "max": int(source_index.max()),
                "unique": unique,
                "in_bounds": in_bounds,
            }
        else:
            unique = in_bounds = True
            source_index_summary = {"min": None, "max": None, "unique": True, "in_bounds": True}
        if not unique:
            errors.append("source_event_index contains duplicates")
        if not in_bounds:
            errors.append("source_event_index contains values outside H5 row range")
        if unique and in_bounds and "HyPER_CLS_T" in frame.columns:
            order = np.argsort(source_index)
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            with h5py.File(h5_path, "r") as handle:
                h5_sorted = np.asarray(handle["LABELS/GLOBAL"][source_index[order]]).reshape(-1)
            h5_labels = h5_sorted[inverse].astype(np.int8)
            prediction_labels = np.asarray(frame["HyPER_CLS_T"], dtype=np.int8)
            if not np.array_equal(h5_labels, prediction_labels):
                errors.append("HyPER_CLS_T disagrees with canonical H5 LABELS/GLOBAL at source_event_index")
    elif total_rows_attr != source_rows:
        errors.append(
            "Subset prediction has no source_event_index. Re-run prediction with source-index export enabled, "
            "or predict the full H5."
        )

    if total_rows_attr > source_rows:
        errors.append(f"Prediction rows ({total_rows_attr}) exceed H5 rows ({source_rows})")
    if args.strict and total_rows_attr not in {source_rows, min(source_rows, total_rows_attr)}:
        warnings.append("Strict length check could not prove full-sample equality")

    score_summaries = {}
    for score_field in ("HyPER_CLS_PROB", "HyPER_best_top1_prob", "HyPER_best_tlep_prob"):
        if score_field in frame.columns:
            frac = finite_fraction(frame[score_field])
            if frac == 0.0:
                errors.append(f"Score column is all NaN/non-finite: {score_field}")
            score_summaries[score_field] = score_range_summary(frame[score_field])

    if "HyPER_CLS_PROB" in score_summaries:
        cls_summary = score_summaries["HyPER_CLS_PROB"]
        if cls_summary["nonfinite_count"]:
            errors.append("Classifier score HyPER_CLS_PROB contains non-finite values")
        if cls_summary["count_below_0"] or cls_summary["count_above_1"]:
            errors.append("Classifier score HyPER_CLS_PROB is outside probability range [0, 1]")

    validity_summary = {}
    for column in ("reco_valid", "tlep_valid", "thad_valid", "wlep_valid", "whad_valid", "higgs_valid"):
        if column in frame.columns:
            values = np.asarray(frame[column], dtype=float)
            finite = values[np.isfinite(values)]
            validity_summary[column] = float(np.mean(finite > 0.5)) if finite.size else None
            if finite.size == 0:
                errors.append(f"Validity column has no finite values: {column}")

    if args.expected_split:
        if "prediction_split" not in frame.columns or not bool((frame["prediction_split"] == args.expected_split).all()):
            errors.append(f"prediction_split is not uniformly {args.expected_split!r}")
    if args.expected_source_indices:
        expected = np.load(args.expected_source_indices, allow_pickle=False).astype(np.int64, copy=False)
        if total_rows_attr != len(expected):
            errors.append(f"Prediction row count {total_rows_attr} differs from requested source-index count {len(expected)}")
        if has_source_index and not np.array_equal(source_index, expected[:len(source_index)]):
            errors.append("Prediction source_event_index order differs from the requested source-index order")

    manifest = parts_manifest
    if manifest is None:
        specific = Path(str(prediction_path) + ".manifest.json")
        if specific.exists():
            manifest = json.loads(specific.read_text(encoding="utf-8"))
    if manifest is not None:
        recorded_source = manifest.get("source_h5_path")
        if recorded_source and Path(recorded_source).resolve() != h5_path.resolve():
            errors.append("Prediction manifest canonical source H5 disagrees with validator H5")
        if args.expected_split and manifest.get("prediction_split") != args.expected_split:
            errors.append("Prediction manifest split disagrees with expected split")

    summary = {
        "prediction_output": str(prediction_path),
        "h5": str(h5_path),
        "topology": args.topology,
        "is_parts": is_parts,
        "loaded_rows": loaded_rows,
        "total_rows": total_rows_attr,
        "h5_rows": source_rows,
        "columns": list(map(str, frame.columns)),
        "parts_manifest_present": parts_manifest is not None,
        "has_source_event_index": has_source_index,
        "source_index_summary": source_index_summary,
        "source_index_sample_sha256": (
            hashlib.sha256(np.asarray(source_index, dtype=np.int64).tobytes()).hexdigest()
            if has_source_index else None
        ),
        "label_sample_sha256": (
            hashlib.sha256(np.asarray(frame["HyPER_CLS_T"], dtype=np.int8).tobytes()).hexdigest()
            if "HyPER_CLS_T" in frame.columns else None
        ),
        "score_ranges": score_summaries,
        "validity_fractions": validity_summary,
        "labels_match_canonical_h5": not any("HyPER_CLS_T disagrees" in item for item in errors),
        "requested_order_preserved": not any("requested source-index order" in item for item in errors),
        "warnings": warnings,
        "errors": errors,
    }

    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args.summary_json:
        Path(args.summary_json).write_text(output + "\n", encoding="utf-8")

    if errors and args.strict:
        return 2
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
