#!/usr/bin/env python3
"""Export deterministic, source-aligned frozen HyPER event representations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from HyPER.analysis.runtime import (
    build_analysis_datamodule,
    forward_representations,
    load_analysis_config,
    load_frozen_model,
    truth_fully_matched,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
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
    device = torch.device("cuda:0" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
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

    stored: dict[str, list[np.ndarray]] = {
        "source_event_index": [], "truth_class": [], "truth_fully_matched": []
    }
    for name in args.representations:
        stored[name] = []
    if model.classification_enabled:
        stored["native_classification_logit"] = []
        stored["native_classification_probability"] = []
    with torch.inference_mode():
        for batch in module.predict_dataloader():
            batch = batch.to(device)
            outputs, representations = forward_representations(model, batch)
            missing = [name for name in args.representations if name not in representations]
            if missing:
                raise KeyError(f"Requested representations are unavailable: {missing}; available={sorted(representations)}")
            source = batch.source_event_index.detach().cpu().reshape(-1).numpy().astype(np.int64)
            truth = batch.cls_t.detach().cpu().reshape(-1).numpy().astype(np.int8)
            stored["source_event_index"].append(source)
            stored["truth_class"].append(truth)
            stored["truth_fully_matched"].append(truth_fully_matched(batch))
            for name in args.representations:
                stored[name].append(representations[name].detach().float().cpu().numpy())
            logits = outputs[3]
            if logits is not None:
                logits = logits.detach().float().cpu().reshape(-1).numpy()
                reconstructed = model.Classification.mlp_class(
                    representations["classification_head_input"]
                ).detach().float().cpu().reshape(-1).numpy()
                np.testing.assert_allclose(reconstructed, logits, rtol=1e-5, atol=1e-6)
                stored["native_classification_logit"].append(logits)
                stored["native_classification_probability"].append(
                    torch.sigmoid(torch.from_numpy(logits)).numpy()
                )
    output = {name: np.concatenate(chunks) for name, chunks in stored.items() if chunks}
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
    np.savez_compressed(destination, **output)
    for name in args.representations:
        print(f"{name}: {output[name].shape} {output[name].dtype}")
    print(f"wrote={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
