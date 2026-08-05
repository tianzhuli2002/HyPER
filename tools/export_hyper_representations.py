#!/usr/bin/env python3
"""Export deterministic, source-aligned frozen HyPER event representations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from HyPER.analysis.runtime import (
    build_analysis_datamodule,
    forward_representations,
    load_analysis_config,
    load_frozen_model,
    truth_fully_matched,
    resource_diagnostics,
    write_resource_diagnostics,
)
from HyPER.analysis.representations import representation_diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--topology", choices=("ttbar1L", "ttH"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-cache", required=True)
    parser.add_argument("--dataset-root", help="Exact directory containing the on-disk graph DB.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--max-events", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--representations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--profile-hash", default=None)
    return parser.parse_args()


def deterministic_indices(split_cache: str, split: str, count: int, seed: int) -> np.ndarray:
    key = "val_idx" if split == "val" else "test_idx"
    with np.load(split_cache, allow_pickle=False) as loaded:
        if key not in loaded:
            raise KeyError(f"Split cache {split_cache} has no {key} array.")
        indices = loaded[key].astype(np.int64)
    if count <= 0:
        raise ValueError("--max-events must be positive.")
    if count >= len(indices):
        return indices
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(len(indices), size=int(count), replace=False))
    return indices[positions]


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU acceleration was requested but PyTorch cannot access CUDA.")
    device = torch.device("cuda:0" if args.accelerator == "gpu" else "cpu")
    selected = deterministic_indices(args.split_cache, args.split, args.max_events, args.seed)
    cfg = load_analysis_config(args.config)
    module = build_analysis_datamodule(
        cfg, h5=args.h5, split_cache=args.split_cache, split=args.split,
        dataset_root=args.dataset_root, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    module.predict_data.indices = selected
    model = load_frozen_model(args.checkpoint, device)
    print(f"config={Path(args.config).resolve()}")
    print(f"checkpoint={Path(args.checkpoint).resolve()}")
    print(f"source_h5={Path(args.h5).resolve()}")
    print(f"split_cache={Path(args.split_cache).resolve()} split={args.split} events={len(selected)}")
    print(f"device={device} representations={args.representations}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stored: dict[str, np.ndarray] | None = None
    row_start = 0
    with torch.inference_mode():
        for batch in module.predict_dataloader():
            batch = batch.to(device)
            outputs, representations = forward_representations(model, batch)
            missing = [name for name in args.representations if name not in representations]
            if missing:
                raise KeyError(f"Requested representations are unavailable: {missing}; available={sorted(representations)}")
            source = batch.source_event_index.detach().cpu().reshape(-1).numpy().astype(np.int64)
            truth = batch.cls_t.detach().cpu().reshape(-1).numpy().astype(np.int8)
            batch_values: dict[str, np.ndarray] = {
                "source_event_index": source,
                "truth_class": truth,
                "truth_fully_matched": truth_fully_matched(batch, args.topology),
            }
            for name in args.representations:
                if name == "classification_head_input" and "final_event" in batch_values:
                    batch_values[name] = batch_values["final_event"]
                else:
                    batch_values[name] = representations[name].detach().float().cpu().numpy()
            logits = outputs[3]
            if logits is not None:
                logits = logits.detach().float().cpu().reshape(-1).numpy()
                reconstructed = model.Classification.mlp_class(
                    representations["classification_head_input"]
                ).detach().float().cpu().reshape(-1).numpy()
                np.testing.assert_allclose(reconstructed, logits, rtol=1e-5, atol=1e-6)
                batch_values["native_classification_logit"] = logits
                batch_values["native_classification_probability"] = torch.sigmoid(
                    torch.from_numpy(logits)
                ).numpy()
            if stored is None:
                stored = {
                    name: np.empty((len(selected),) + np.asarray(value).shape[1:], dtype=(
                        np.int64 if name == "source_event_index" else
                        np.int8 if name in {"truth_class", "truth_fully_matched"} else np.float32
                    ))
                    for name, value in batch_values.items()
                }
            missing = set(stored) - set(batch_values)
            unexpected = set(batch_values) - set(stored)
            if missing or unexpected:
                raise RuntimeError(f"Batch export fields changed: missing={sorted(missing)}, unexpected={sorted(unexpected)}.")
            row_stop = row_start + len(source)
            if row_stop > len(selected):
                raise RuntimeError("Exporter received more events than the selected deterministic sample.")
            for name, value in batch_values.items():
                value = np.asarray(value)
                if value.shape[0] != len(source) or value.shape[1:] != stored[name].shape[1:]:
                    raise RuntimeError(f"Export field {name!r} shape changed: {value.shape} vs {stored[name].shape}.")
                stored[name][row_start:row_stop] = value
            row_start = row_stop
    if stored is None or row_start != len(selected):
        raise RuntimeError(f"Exported {row_start} events but expected {len(selected)}.")
    output = dict(stored)
    if not np.array_equal(output["source_event_index"], selected):
        raise RuntimeError("Export order or source-event identity differs from deterministic selection.")
    if len(np.unique(output["source_event_index"])) != len(output["source_event_index"]):
        raise RuntimeError("Exported source_event_index values are not unique.")
    for name, values in output.items():
        if name.startswith("truth") or name == "source_event_index":
            continue
        output[name] = values.astype(np.float32, copy=False)
        if not np.isfinite(output[name]).all():
            raise RuntimeError(f"Exported array {name!r} contains non-finite values.")
    output.update({
        "prediction_split": np.full(len(selected), args.split),
        "model_name": np.asarray(args.model_name or Path(args.checkpoint).parent.parent.parent.name),
        "config_path": np.asarray(str(Path(args.config).resolve())),
        "checkpoint_path": np.asarray(str(Path(args.checkpoint).resolve())),
    })
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output)
    temporary.replace(destination)
    health_names = [name for name in args.representations if name != "classification_head_input"]
    health = {
        "model_name": str(output["model_name"]),
        "topology": args.topology,
        "split": args.split,
        "event_count": int(len(selected)),
        "commit": args.commit,
        "profile_hash": args.profile_hash,
        "variance_policy": {
            "relative_tolerance": 1e-12,
            "roundoff_factor": 64.0,
            "exact_repeated_rows_are_zero_variance": True,
        },
        "representations": {
            name: representation_diagnostics(output[name]) for name in health_names
        },
        "aliases": {
            "classification_head_input": {
                "target": "final_event",
                "equal_values": bool(np.array_equal(output["classification_head_input"], output["final_event"])),
                "physically_stored_twice_in_npz": True,
                "duplicate_raw_bytes": int(output["final_event"].nbytes),
            }
        },
    }
    (destination.parent / f"{destination.stem}_health.json").write_text(
        json.dumps(health, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (destination.parent / f"{destination.stem}_health.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("model_name", "topology", "split", "representation", *health["representations"][health_names[0]].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in health_names:
            writer.writerow({"model_name": health["model_name"], "topology": args.topology, "split": args.split, "representation": name, **health["representations"][name]})
    diagnostics = resource_diagnostics(stage="export", started=started, events_processed=len(selected), output_root=destination.parent)
    diagnostics.update({"model_name": str(output["model_name"]), "split": args.split, "representation_file": str(destination)})
    write_resource_diagnostics(destination.parent, diagnostics)
    for name in args.representations:
        print(f"{name}: {output[name].shape} {output[name].dtype}")
    print(f"wrote={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
