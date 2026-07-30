#!/usr/bin/env python3
"""Fit paired label-free orthogonal Procrustes alignment between HyPER exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from HyPER.analysis.representations import align_exports, event_index_sha256, fit_procrustes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source-key", default="final_event")
    parser.add_argument("--target-key", default="classification_head_input")
    parser.add_argument("--shuffle-target", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--common-events-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args()


def _load(path, representation_key):
    with np.load(path, allow_pickle=False) as loaded:
        required = ("source_event_index", representation_key)
        missing = [name for name in required if name not in loaded]
        if missing:
            raise KeyError(f"Representation export {path} is missing {missing}.")
        output = {name: loaded[name] for name in required}
        if "checkpoint_path" in loaded:
            output["checkpoint_path"] = loaded["checkpoint_path"]
        return output


def main() -> int:
    args = parse_args()
    source = _load(args.source, args.source_key)
    target = _load(args.target, args.target_key)
    indices, source_rows, target_rows = align_exports(
        source, target, common_events_only=args.common_events_only
    )
    if args.source_key not in source or args.target_key not in target:
        raise KeyError(f"Missing representation key; source={sorted(source)}, target={sorted(target)}")
    x = np.asarray(source[args.source_key][source_rows], dtype=np.float64)
    y = np.asarray(target[args.target_key][target_rows], dtype=np.float64)
    if args.shuffle_target:
        y = y[np.random.default_rng(args.seed).permutation(len(y))]
    fitted = fit_procrustes(x, y)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_mean=fitted["source_mean"], target_mean=fitted["target_mean"],
        rotation=fitted["rotation"], singular_values=fitted["singular_values"],
    )
    summary = {
        "source_representation_path": str(Path(args.source).resolve()),
        "target_representation_path": str(Path(args.target).resolve()),
        "source_representation_name": args.source_key,
        "target_representation_name": args.target_key,
        "source_checkpoint": str(source.get("checkpoint_path", "")),
        "target_checkpoint": str(target.get("checkpoint_path", "")),
        "fit_event_count": int(len(indices)),
        "fit_source_event_index_hash": event_index_sha256(indices),
        "source_dimension": int(x.shape[1]), "target_dimension": int(y.shape[1]),
        "normalised_alignment_residual": fitted["normalised_residual"],
        "mean_aligned_cosine_similarity": fitted["mean_aligned_cosine_similarity"],
        "orthogonality_error": fitted["orthogonality_error"],
        "target_pairs_shuffled": bool(args.shuffle_target), "shuffle_seed": int(args.seed),
        "labels_used_for_fitting": False,
    }
    summary_path = Path(args.summary).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
