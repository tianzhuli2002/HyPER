#!/usr/bin/env python3
"""Stream a split through frozen HyPER models and compute final-event linear CKA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from HyPER.analysis.representations import StreamingLinearCKA
from HyPER.analysis.runtime import (
    build_analysis_datamodule,
    forward_representations,
    load_analysis_config,
    load_frozen_model,
    truth_fully_matched,
)
from HyPER.topology.plot_style import configure_matplotlib, decorate_axis, save_figure

MODES = ("classification", "reconstruction", "joint")
PAIRS = (
    ("classification", "reconstruction"),
    ("classification", "joint"),
    ("reconstruction", "joint"),
)
SUBSETS = ("all", "background", "signal_fully_matched", "signal_non_fully_matched")
PAIR_LABELS = {
    ("classification", "reconstruction"): "Classification–reconstruction",
    ("classification", "joint"): "Classification–joint",
    ("reconstruction", "joint"): "Reconstruction–joint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument(f"--{mode}-config", required=True)
        parser.add_argument(f"--{mode}-checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-cache", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-event-count", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Full-test final-event CKA")
    return parser.parse_args()


def deterministic_limit(indices: np.ndarray, count: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if count is None or count >= len(indices):
        return indices
    if count <= 0:
        raise ValueError("--max-events must be positive.")
    positions = np.sort(np.random.default_rng(seed).choice(len(indices), count, replace=False))
    return indices[positions]


def subset_masks(labels: np.ndarray, fully_matched: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    fully_matched = np.asarray(fully_matched, dtype=bool)
    return {
        "all": np.ones(len(labels), dtype=bool),
        "background": labels == 0,
        "signal_fully_matched": (labels == 1) & fully_matched,
        "signal_non_fully_matched": (labels == 1) & ~fully_matched,
    }


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU acceleration was requested but PyTorch cannot access CUDA.")
    device = torch.device("cuda:0" if args.accelerator == "gpu" else "cpu")
    cfg = load_analysis_config(args.reconstruction_config)
    module = build_analysis_datamodule(
        cfg,
        h5=args.h5,
        split_cache=args.split_cache,
        split=args.split,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    selected = deterministic_limit(module.predict_data.indices, args.max_events, args.seed)
    module.predict_data.indices = selected
    if args.expected_event_count is not None and len(selected) != args.expected_event_count:
        raise RuntimeError(
            f"Selected {len(selected)} events but --expected-event-count={args.expected_event_count}."
        )
    models = {
        mode: load_frozen_model(getattr(args, f"{mode}_checkpoint"), device)
        for mode in MODES
    }
    accumulators: dict[tuple[str, str, str], StreamingLinearCKA] = {}
    subset_counts = {name: 0 for name in SUBSETS}
    event_hash = hashlib.sha256()
    batch_count = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in module.predict_dataloader():
            batch = batch.to(device)
            representations = {}
            for mode, model in models.items():
                _, reps = forward_representations(model, batch)
                value = reps["final_event"].detach().float().cpu().numpy()
                if value.ndim != 2 or value.shape[1] != 1024:
                    raise RuntimeError(f"{mode} final_event has unexpected shape {value.shape}.")
                if not np.isfinite(value).all():
                    raise RuntimeError(f"{mode} final_event contains non-finite values.")
                representations[mode] = value
            indices = batch.source_event_index.detach().cpu().reshape(-1).numpy().astype(np.int64)
            event_hash.update(np.ascontiguousarray(indices).tobytes())
            labels = batch.cls_t.detach().cpu().reshape(-1).numpy().astype(np.int8)
            fm = truth_fully_matched(batch).astype(bool)
            masks = subset_masks(labels, fm)
            for subset, mask in masks.items():
                count = int(mask.sum())
                subset_counts[subset] += count
                if count == 0:
                    continue
                for left, right in PAIRS:
                    key = (left, right, subset)
                    if key not in accumulators:
                        accumulators[key] = StreamingLinearCKA(
                            representations[left].shape[1], representations[right].shape[1]
                        )
                    accumulators[key].update(representations[left][mask], representations[right][mask])
            batch_count += 1
    elapsed = time.perf_counter() - started
    if subset_counts["all"] != len(selected):
        raise RuntimeError(
            f"Streamed event count {subset_counts['all']} differs from selected {len(selected)}."
        )
    rows = []
    values = {}
    for left, right in PAIRS:
        pair_name = f"{left}_vs_{right}"
        values[pair_name] = {}
        for subset in SUBSETS:
            key = (left, right, subset)
            if key not in accumulators:
                raise RuntimeError(f"No events accumulated for {pair_name}/{subset}.")
            value = accumulators[key].value()
            values[pair_name][subset] = value
            rows.append(
                {
                    "left_model": left,
                    "right_model": right,
                    "subset": subset,
                    "event_count": accumulators[key].count,
                    "cka": value,
                }
            )
    summary = {
        "title": args.title,
        "split": args.split,
        "event_count": int(len(selected)),
        "subset_counts": subset_counts,
        "event_index_hash": event_hash.hexdigest(),
        "representation": "final_event",
        "representation_dimension": 1024,
        "classification_head_input_alias": "final_event",
        "model_checkpoints": {
            mode: str(Path(getattr(args, f"{mode}_checkpoint")).expanduser().resolve())
            for mode in MODES
        },
        "values": values,
        "calculation": "streamed sufficient statistics in float64; no N x N kernel",
        "batch_count": batch_count,
        "elapsed_seconds": elapsed,
        "events_per_second": float(len(selected) / elapsed) if elapsed > 0 else None,
    }
    (output / "full_test_final_event_cka.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "full_test_final_event_cka.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    x = np.arange(len(SUBSETS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(PAIRS))
    for offset, pair in zip(offsets, PAIRS):
        pair_name = f"{pair[0]}_vs_{pair[1]}"
        ax.plot(
            x + offset,
            [values[pair_name][subset] for subset in SUBSETS],
            marker="o",
            linewidth=1.8,
            label=PAIR_LABELS[pair],
        )
    ax.set_xticks(x, [name.replace("signal_", "signal\n").replace("_", " ") for name in SUBSETS])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Linear centred CKA")
    ax.set_xlabel("Event subset")
    decorate_axis(ax, title=args.title)
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, output, "full_test_final_event_cka")
    print(f"wrote={output} events={len(selected)} elapsed_seconds={elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
