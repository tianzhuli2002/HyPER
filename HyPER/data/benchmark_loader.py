"""Small CPU benchmark for the production HyPER graph DataLoader."""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from HyPER.data.datamodule import HyPERDataModule, SourceIndexSubset
from HyPER.train import _graph_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--dataset-name")
    parser.add_argument("--split-cache")
    parser.add_argument("--source-indices")
    parser.add_argument("--task-mode", choices=("reconstruction", "classification", "joint"), default="joint")
    parser.add_argument(
        "--loader-path",
        choices=("baseline", "optimised"),
        default="optimised",
        help="Use per-event __getitem__ reads or the production batched __getitems__ path.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measured-batches", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _mode_flags(mode: str) -> tuple[bool, bool]:
    return mode in {"classification", "joint"}, mode in {"reconstruction", "joint"}


class _PerEventFetchDataset(Dataset):
    """Expose only ``__getitem__`` to reproduce the pre-optimisation loader path."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def main() -> None:
    args = _parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive.")
    if args.warmup_batches < 0 or args.measured_batches <= 0 or args.repeats <= 0:
        raise ValueError("Warm-up must be non-negative; measured batches and repeats must be positive.")

    cfg = OmegaConf.load(args.config)
    classification_enabled, reconstruction_enabled = _mode_flags(args.task_mode)
    dataset_root = str(args.dataset_root or cfg.dataset.root)
    dataset_name = str(args.dataset_name or cfg.dataset.train_set)
    split_cfg = OmegaConf.to_container(cfg.dataset.split, resolve=True)
    if args.split_cache:
        split_cfg["enabled"] = True
        split_cfg["cache_path"] = str(Path(args.split_cache).expanduser().resolve())
        split_cfg["require_existing"] = True

    datamodule = HyPERDataModule(
        root=dataset_root,
        train_set=dataset_name,
        val_set=None if cfg.dataset.val_set is None else str(cfg.dataset.val_set),
        predict_set=None,
        batch_size=int(args.batch_size or cfg.dataset.batch_size),
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
        persistent_workers=bool(cfg.dataset.persistent_workers),
        prefetch_factor=int(args.prefetch_factor),
        force_reload=False,
        use_ondisk=True,
        graph_config=_graph_config(cfg),
        split_config=split_cfg,
        predict_split=None,
        source_indices_file=None,
        source_h5_path=cfg.dataset.get("source_h5_path"),
        require_two_event_classes=False,
        tuning_mode=False,
        seed=int(cfg.general.seed),
        classification_enabled=classification_enabled,
        reconstruction_enabled=reconstruction_enabled,
        verify_source_identity_per_event=False,
        source_identity_setup_samples=8,
    )
    datamodule.setup("fit")

    if args.source_indices:
        indices = np.load(args.source_indices, allow_pickle=False)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("--source-indices must contain a one-dimensional integer array.")
        base = getattr(datamodule.train_data, "dataset", None)
        if base is None:
            raise RuntimeError("Could not resolve the underlying source-indexed training dataset.")
        datamodule.train_data = SourceIndexSubset(
            base,
            indices.astype(np.int64, copy=False),
            split_name="benchmark",
            verify_source_identity_per_event=False,
        )
        datamodule.train_data.validate_source_identity(8)

    if args.loader_path == "baseline":
        datamodule.train_data = _PerEventFetchDataset(datamodule.train_data)

    loader = datamodule.train_dataloader()
    runs = []
    for repeat in range(args.repeats):
        torch.manual_seed(int(cfg.general.seed))
        iterator = iter(loader)
        for _ in range(args.warmup_batches):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError("DataLoader ended during warm-up.") from exc

        durations = []
        events = nodes = edges = hyperedges = 0
        for _ in range(args.measured_batches):
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            durations.append(time.perf_counter() - started)
            events += int(batch.num_graphs)
            nodes += int(batch.num_nodes)
            edges += int(batch.edge_index.size(1))
            hyperedges += int(batch.hyperedge_index.size(1))
        if not durations:
            raise RuntimeError("No DataLoader batches were measured.")
        wall = sum(durations)
        ordered = sorted(durations)
        p95_index = min(len(ordered) - 1, max(0, int(np.ceil(0.95 * len(ordered))) - 1))
        runs.append(
            {
                "repeat": repeat,
                "measured_batches": len(durations),
                "events": events,
                "nodes": nodes,
                "directed_edges": edges,
                "hyperedges": hyperedges,
                "wall_seconds": wall,
                "mean_batch_seconds": statistics.mean(durations),
                "median_batch_seconds": statistics.median(durations),
                "p95_batch_seconds": ordered[p95_index],
                "batches_per_second": len(durations) / wall,
                "events_per_second": events / wall,
                "nodes_per_second": nodes / wall,
                "directed_edges_per_second": edges / wall,
                "hyperedges_per_second": hyperedges / wall,
            }
        )

    throughputs = [run["events_per_second"] for run in runs]
    representative = runs[-1]
    result = {
        "config": str(Path(args.config).expanduser().resolve()),
        "loader_path": args.loader_path,
        "task_mode": args.task_mode,
        "batch_size": int(args.batch_size or cfg.dataset.batch_size),
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "warmup_batches": args.warmup_batches,
        "repeats": args.repeats,
        "runs": runs,
        "mean_events_per_second": statistics.mean(throughputs),
        "events_per_second_stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "relative_events_per_second_variation": (
            statistics.stdev(throughputs) / statistics.mean(throughputs)
            if len(throughputs) > 1 and statistics.mean(throughputs) else 0.0
        ),
        "peak_resident_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        **{key: value for key, value in representative.items() if key != "repeat"},
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        output = Path(args.output_json).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
