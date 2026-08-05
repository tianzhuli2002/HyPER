"""Direct loading and inference helpers for already-trained HyPER models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from HyPER.configuration import configured_topology, task_spec
from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel
from HyPER.topology.reconstruction_score import required_truth_role_ids


def resource_diagnostics(
    *,
    stage: str,
    started: float,
    events_processed: int | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    """Return lightweight process/resource measurements for production artifacts.

    The keyword-only arguments are intentional: callers should identify the
    measurements explicitly at each production stage.  ``output_root`` is
    recorded even when it does not exist yet; in that case ``output_bytes`` is
    left as ``None`` rather than making diagnostics prevent a stage from
    completing.
    """
    if events_processed is not None and events_processed < 0:
        raise ValueError("events_processed must be non-negative or None")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports ru_maxrss in KiB; retaining the unit in the artifact avoids
    # ambiguity when the file is compared with Slurm's MaxRSS.
    payload: dict[str, object] = {
        "stage": str(stage),
        "elapsed_wall_seconds": float(max(0.0, time.perf_counter() - started)),
        "peak_rss_bytes": int(usage.ru_maxrss * 1024),
        "peak_gpu_allocated_bytes": None,
        "peak_gpu_reserved_bytes": None,
        "events_processed": None if events_processed is None else int(events_processed),
        "events_per_second": None,
        "output_location": None,
        "output_bytes": None,
        "process_id": os.getpid(),
    }
    if events_processed is not None:
        elapsed = float(payload["elapsed_wall_seconds"])
        payload["events_per_second"] = (
            0.0
            if events_processed == 0
            else float(events_processed / elapsed) if elapsed > 0.0 else None
        )
    if torch.cuda.is_available():
        payload["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        payload["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    if output_root is not None:
        root = Path(output_root)
        payload["output_location"] = str(root.expanduser().resolve())
        if root.exists():
            payload["output_bytes"] = int(
                sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            )
    return payload


def write_resource_diagnostics(
    output_root: str | Path,
    diagnostics: dict[str, object],
    *,
    filename: str = "runtime_diagnostics.json",
) -> None:
    """Write strict JSON diagnostics below an output directory."""
    if not filename or Path(filename).name != filename:
        raise ValueError(f"Diagnostic filename must be a simple filename: {filename!r}")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_analysis_config(path: str):
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = OmegaConf.load(config_path)
    required = ("dataset", "input", "target", "model")
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError(f"Analysis config {config_path} is missing sections {missing}.")
    configured_topology(cfg)
    task_spec(cfg)
    return cfg


def build_analysis_datamodule(
    cfg, *, h5: str, split_cache: str, split: str, dataset_root: str | None,
    batch_size: int, num_workers: int, source_indices_file: str | None = None,
    pin_memory: bool = False,
):
    h5_path = Path(h5).expanduser().resolve()
    split_path = Path(split_cache).expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    root = Path(dataset_root or str(cfg.dataset.root)).expanduser().resolve()
    db_path = root / f"{cfg.dataset.predict_set}.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"Required exact graph cache does not exist: {db_path}")
    graph_config = OmegaConf.to_container(
        OmegaConf.create({"input": cfg.input, "target": cfg.target}), resolve=True
    )
    graph_config.setdefault("target", {})["encoding"] = "typed"
    split_config = OmegaConf.to_container(cfg.dataset.split, resolve=True)
    split_config.update({"cache_path": str(split_path), "require_existing": True})
    module = HyPERDataModule(
        root=str(root), train_set=str(cfg.dataset.train_set),
        predict_set=str(cfg.dataset.predict_set), batch_size=int(batch_size), drop_last=False,
        num_workers=int(num_workers), pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0, prefetch_factor=int(cfg.dataset.prefetch_factor),
        graph_config=graph_config,
        split_config=split_config, predict_split=str(split), source_indices_file=source_indices_file,
        source_h5_path=str(h5_path), require_two_event_classes=True,
        seed=int(cfg.general.seed), classification_enabled=True, reconstruction_enabled=True,
        verify_source_identity_per_event=True, source_identity_setup_samples=32,
    )
    module.setup("predict")
    return module


def load_frozen_model(checkpoint: str, device: torch.device) -> HyPERModel:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    model = HyPERModel.load_from_checkpoint(str(path), map_location="cpu")
    model.eval().requires_grad_(False).to(device)
    return model


def truth_fully_matched(batch, topology: str = "ttbar1L") -> np.ndarray:
    """Return intrinsic truth reconstructibility for the requested topology."""
    node_truth = batch.node_truth_ids.detach().cpu().reshape(-1).numpy()
    node_types = batch.x[:, -1].detach().cpu().reshape(-1).numpy()
    node_batch = batch.batch.detach().cpu().reshape(-1).numpy()
    result = np.zeros(int(batch.num_graphs), dtype=np.int8)
    required = required_truth_role_ids(topology)
    for event in range(int(batch.num_graphs)):
        keep = (node_batch == event) & (node_types == 1)
        result[event] = int(required.issubset(set(node_truth[keep].astype(int).tolist())))
    return result


def forward_representations(model: HyPERModel, batch):
    outputs, representations = model(
        batch.x, batch.edge_index, batch.edge_attr, batch.u, batch.batch,
        batch.hyperedge_index, return_representations=True,
    )
    return outputs, representations
