#!/usr/bin/env python3
"""Summarise shuffled and random alignment-control AUC distributions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score

DIRECTION_TO_PAIRED = {
    "reconstruction_to_classification": "reconstruction_to_classification_paired_score",
    "reconstruction_to_joint": "reconstruction_to_joint_paired_score",
    "joint_to_classification": "joint_to_classification_paired_score",
}
DIRECTION_TO_DIRECT = {
    "reconstruction_to_classification": "reconstruction_to_classification_direct_score",
    "reconstruction_to_joint": "reconstruction_to_joint_direct_score",
    "joint_to_classification": "joint_to_classification_direct_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", required=True)
    parser.add_argument("--controls-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-shuffles", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def load(path: Path, mmap_mode="r"):
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode=mmap_mode)


def masks(labels: np.ndarray, fm: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "inclusive_all_signal": np.ones(len(labels), dtype=bool),
        "fully_matched_signal_vs_background": (labels == 0) | ((labels == 1) & fm),
        "non_fully_matched_signal_vs_background": (labels == 0) | ((labels == 1) & ~fm),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "percentile_05": float(np.quantile(values, 0.05)),
        "percentile_16": float(np.quantile(values, 0.16)),
        "percentile_84": float(np.quantile(values, 0.84)),
        "percentile_95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }



def control_auc_rows(
    matrix: np.ndarray,
    labels: np.ndarray,
    fully_matched: np.ndarray,
    workers: int,
) -> list[dict[str, float]]:
    subset_masks = masks(labels, fully_matched)

    def calculate(row: int) -> dict[str, float]:
        score = np.asarray(matrix[row], dtype=np.float64)
        return {
            subset: float(roc_auc_score(labels[mask], score[mask]))
            for subset, mask in subset_masks.items()
        }

    if workers == 1:
        return [calculate(row) for row in range(matrix.shape[0])]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(calculate, range(matrix.shape[0])))

def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    scores_dir = Path(args.scores_dir).expanduser().resolve()
    controls_dir = Path(args.controls_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(load(scores_dir / "test_truth_class.npy"), dtype=np.int8)
    fm = np.asarray(load(scores_dir / "test_truth_fully_matched.npy"), dtype=bool)
    metadata_path = controls_dir / "control_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    rows = []
    distributions = {}
    for direction, paired_field in DIRECTION_TO_PAIRED.items():
        paired = np.asarray(load(scores_dir / f"{paired_field}.npy"), dtype=np.float64)
        direct = np.asarray(load(scores_dir / f"{DIRECTION_TO_DIRECT[direction]}.npy"), dtype=np.float64)
        for control_type in ("shuffled", "random"):
            matrix = load(controls_dir / f"{direction}_{control_type}_scores.npy")
            if control_type == "shuffled" and matrix.shape[0] != args.expected_shuffles:
                raise RuntimeError(
                    f"{direction} has {matrix.shape[0]} shuffled controls, expected {args.expected_shuffles}."
                )
            seeds = [entry["seed"] for entry in metadata.get(direction, {}).get(control_type, [])]
            if seeds and len(seeds) != matrix.shape[0]:
                raise RuntimeError(f"Control metadata count differs for {direction}/{control_type}.")
            control_results = control_auc_rows(matrix, labels, fm, args.workers)
            for subset, mask in masks(labels, fm).items():
                subset_labels = labels[mask]
                paired_auc = float(roc_auc_score(subset_labels, paired[mask]))
                direct_auc = float(roc_auc_score(subset_labels, direct[mask]))
                aucs = np.asarray(
                    [result[subset] for result in control_results],
                    dtype=np.float64,
                )
                key = f"{direction}__{control_type}__{subset}"
                distributions[key] = aucs.astype(np.float32)
                exceed = int(np.sum(aucs >= paired_auc))
                row = {
                    "direction": direction,
                    "control_type": control_type,
                    "subset": subset,
                    "control_count": int(len(aucs)),
                    "paired_auc": paired_auc,
                    "direct_auc": direct_auc,
                    **distribution_summary(aucs),
                    "controls_at_or_above_paired": exceed,
                    "empirical_p_value": float((1 + exceed) / (1 + len(aucs))),
                }
                rows.append(row)
    np.savez_compressed(output / "alignment_control_auc_distributions.npz", **distributions)
    with (output / "shuffled_null_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "event_count": int(len(labels)),
        "expected_shuffled_alignments_per_direction": args.expected_shuffles,
        "rows": rows,
        "interpretation": {
            "shuffled": "no-correspondence Procrustes null fitted on validation representations",
            "random": "random orthogonal-map control with paired source and target means",
        },
    }
    (output / "shuffled_null_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
