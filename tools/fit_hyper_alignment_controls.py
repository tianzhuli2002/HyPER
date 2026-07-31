#!/usr/bin/env python3
"""Fit one paired and an ensemble of shuffled orthogonal Procrustes maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from HyPER.analysis.representations import align_exports, event_index_sha256, fit_procrustes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source-key", default="final_event")
    parser.add_argument("--target-key", default="classification_head_input")
    parser.add_argument("--direction", required=True)
    parser.add_argument("--num-shuffles", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--expected-event-count", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_minimal(path: str, key: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = ("source_event_index", key)
        missing = [name for name in required if name not in loaded]
        if missing:
            raise KeyError(f"Representation export {path} is missing {missing}.")
        result = {name: loaded[name] for name in required}
        if "checkpoint_path" in loaded:
            result["checkpoint_path"] = loaded["checkpoint_path"]
        return result


def permutation_sha256(permutation: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(permutation, dtype=np.int64).tobytes()
    ).hexdigest()


def save_alignment(path: Path, fitted: dict[str, object]) -> None:
    np.savez(
        path,
        source_mean=fitted["source_mean"],
        target_mean=fitted["target_mean"],
        rotation=fitted["rotation"],
        singular_values=fitted["singular_values"],
    )


def valid_saved_alignment(
    alignment_path: Path,
    summary_path: Path,
    *,
    alignment_type: str,
    fit_event_count: int,
    source_hash: str,
    seed: int | None,
    permutation_hash: str | None,
    direction: str,
    source_key: str,
    target_key: str,
    dimension: int,
) -> dict | None:
    if not alignment_path.is_file() or not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(alignment_path, allow_pickle=False) as loaded:
            required = ("source_mean", "target_mean", "rotation", "singular_values")
            if any(name not in loaded for name in required):
                return None
            if not all(np.isfinite(loaded[name]).all() for name in required):
                return None
            if loaded["source_mean"].shape != (dimension,):
                return None
            if loaded["target_mean"].shape != (dimension,):
                return None
            if loaded["rotation"].shape != (dimension, dimension):
                return None
            if loaded["singular_values"].shape != (dimension,):
                return None
        expected = (
            summary.get("direction") == direction
            and summary.get("alignment_type") == alignment_type
            and summary.get("fit_event_count") == int(fit_event_count)
            and summary.get("fit_source_event_index_hash") == source_hash
            and summary.get("seed") == seed
            and summary.get("target_permutation_hash") == permutation_hash
            and summary.get("source_representation_name") == source_key
            and summary.get("target_representation_name") == target_key
            and summary.get("source_dimension") == dimension
            and summary.get("target_dimension") == dimension
            and summary.get("labels_loaded_for_fitting") is False
        )
        return summary if expected else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def make_summary(
    *,
    args: argparse.Namespace,
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    indices: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fitted: dict[str, object],
    alignment_type: str,
    seed: int | None,
    permutation_hash: str | None,
) -> dict[str, object]:
    return {
        "direction": args.direction,
        "alignment_type": alignment_type,
        "seed": seed,
        "fit_event_count": int(len(indices)),
        "fit_source_event_index_hash": event_index_sha256(indices),
        "target_permutation_hash": permutation_hash,
        "source_representation_path": str(Path(args.source).resolve()),
        "target_representation_path": str(Path(args.target).resolve()),
        "source_representation_name": args.source_key,
        "target_representation_name": args.target_key,
        "source_checkpoint": str(source.get("checkpoint_path", "")),
        "target_checkpoint": str(target.get("checkpoint_path", "")),
        "source_dimension": int(x.shape[1]),
        "target_dimension": int(y.shape[1]),
        "normalised_alignment_residual": fitted["normalised_residual"],
        "mean_aligned_cosine_similarity": fitted["mean_aligned_cosine_similarity"],
        "orthogonality_error": fitted["orthogonality_error"],
        "labels_loaded_for_fitting": False,
    }


def main() -> int:
    args = parse_args()
    if args.num_shuffles <= 0:
        raise ValueError("--num-shuffles must be positive.")
    source = load_minimal(args.source, args.source_key)
    target = load_minimal(args.target, args.target_key)
    indices, source_rows, target_rows = align_exports(source, target)
    if args.expected_event_count is not None and len(indices) != args.expected_event_count:
        raise RuntimeError(
            f"Alignment has {len(indices)} events, expected {args.expected_event_count}."
        )
    x = np.asarray(source[args.source_key][source_rows], dtype=np.float64)
    y = np.asarray(target[args.target_key][target_rows], dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Alignment representation shapes differ: {x.shape} vs {y.shape}.")
    source_hash = event_index_sha256(indices)
    output = Path(args.output_dir).expanduser().resolve()
    shuffled_dir = output / "shuffled"
    output.mkdir(parents=True, exist_ok=True)
    shuffled_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    paired_path = output / "paired.npz"
    paired_summary_path = output / "paired.json"
    paired_summary = valid_saved_alignment(
        paired_path,
        paired_summary_path,
        alignment_type="paired",
        fit_event_count=len(indices),
        source_hash=source_hash,
        seed=None,
        permutation_hash=None,
        direction=args.direction,
        source_key=args.source_key,
        target_key=args.target_key,
        dimension=x.shape[1],
    )
    if paired_summary is None or args.overwrite:
        if (
            not args.overwrite
            and (paired_path.exists() or paired_summary_path.exists())
        ):
            raise RuntimeError(
                f"Existing paired alignment is incomplete or incompatible: {paired_path}; "
                "pass --overwrite to replace it."
            )
        paired = fit_procrustes(x, y)
        save_alignment(paired_path, paired)
        paired_summary = make_summary(
            args=args,
            source=source,
            target=target,
            indices=indices,
            x=x,
            y=y,
            fitted=paired,
            alignment_type="paired",
            seed=None,
            permutation_hash=None,
        )
        paired_summary_path.write_text(
            json.dumps(paired_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(f"REUSE verified paired alignment {paired_path}")
    rows.append(paired_summary)

    seen_hashes: set[str] = set()
    for offset in range(args.num_shuffles):
        seed = args.seed_start + offset
        permutation = np.random.default_rng(seed).permutation(len(y))
        if np.array_equal(permutation, np.arange(len(y))):
            raise RuntimeError(f"Shuffle seed {seed} produced the identity permutation.")
        permutation_hash = permutation_sha256(permutation)
        if permutation_hash in seen_hashes:
            raise RuntimeError(f"Duplicate target permutation generated for seed {seed}.")
        seen_hashes.add(permutation_hash)
        stem = f"seed_{seed:03d}"
        alignment_path = shuffled_dir / f"{stem}.npz"
        summary_path = shuffled_dir / f"{stem}.json"
        summary = valid_saved_alignment(
            alignment_path,
            summary_path,
            alignment_type="shuffled",
            fit_event_count=len(indices),
            source_hash=source_hash,
            seed=int(seed),
            permutation_hash=permutation_hash,
            direction=args.direction,
            source_key=args.source_key,
            target_key=args.target_key,
            dimension=x.shape[1],
        )
        if summary is None or args.overwrite:
            if (
                not args.overwrite
                and (alignment_path.exists() or summary_path.exists())
            ):
                raise RuntimeError(
                    f"Existing shuffled alignment is incomplete or incompatible: {alignment_path}; "
                    "pass --overwrite to replace it."
                )
            fitted = fit_procrustes(x, y[permutation])
            save_alignment(alignment_path, fitted)
            summary = make_summary(
                args=args,
                source=source,
                target=target,
                indices=indices,
                x=x,
                y=y,
                fitted=fitted,
                alignment_type="shuffled",
                seed=int(seed),
                permutation_hash=permutation_hash,
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"direction={args.direction} shuffle={offset + 1}/{args.num_shuffles} "
                f"seed={seed} residual={fitted['normalised_residual']:.6g}",
                flush=True,
            )
        else:
            print(f"REUSE verified shuffled alignment {alignment_path}")
        rows.append(summary)

    fields = (
        "direction",
        "alignment_type",
        "seed",
        "fit_event_count",
        "fit_source_event_index_hash",
        "target_permutation_hash",
        "source_dimension",
        "target_dimension",
        "normalised_alignment_residual",
        "mean_aligned_cosine_similarity",
        "orthogonality_error",
        "labels_loaded_for_fitting",
    )
    with (output / "alignment_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    ensemble_summary = {
        "direction": args.direction,
        "fit_event_count": int(len(indices)),
        "fit_source_event_index_hash": source_hash,
        "num_shuffled_alignments": args.num_shuffles,
        "shuffle_seeds": list(
            range(args.seed_start, args.seed_start + args.num_shuffles)
        ),
        "paired": paired_summary,
        "shuffled": rows[1:],
        "elapsed_seconds": time.perf_counter() - started,
        "labels_loaded_for_fitting": False,
    }
    (output / "alignment_summary.json").write_text(
        json.dumps(ensemble_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote={output} direction={args.direction} shuffles={args.num_shuffles}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
