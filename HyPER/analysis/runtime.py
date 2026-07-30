"""Direct loading and inference helpers for already-trained HyPER models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel


def load_analysis_config(path: str):
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = OmegaConf.load(config_path)
    required = ("dataset", "input", "target", "model", "classification", "reconstruction")
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError(f"Analysis config {config_path} is missing sections {missing}.")
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
    if bool(cfg.dataset.use_ondisk) and not db_path.is_file():
        raise FileNotFoundError(f"Required exact graph cache does not exist: {db_path}")
    graph_config = OmegaConf.to_container(OmegaConf.create({"input": cfg.input, "target": cfg.target}), resolve=True)
    split_config = OmegaConf.to_container(cfg.dataset.split, resolve=True)
    split_config.update({"enabled": True, "cache_path": str(split_path), "require_existing": True})
    module = HyPERDataModule(
        root=str(root), train_set=str(cfg.dataset.train_set), val_set=None,
        predict_set=str(cfg.dataset.predict_set), batch_size=int(batch_size), drop_last=False,
        num_workers=int(num_workers), pin_memory=bool(pin_memory),
        persistent_workers=int(num_workers) > 0, prefetch_factor=int(cfg.dataset.prefetch_factor),
        force_reload=False, use_ondisk=bool(cfg.dataset.use_ondisk), graph_config=graph_config,
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


def truth_fully_matched(batch) -> np.ndarray:
    node_truth = batch.node_truth_ids.detach().cpu().reshape(-1).numpy()
    node_types = batch.x[:, -1].detach().cpu().reshape(-1).numpy()
    node_batch = batch.batch.detach().cpu().reshape(-1).numpy()
    result = np.zeros(int(batch.num_graphs), dtype=np.int8)
    required = {1, 2, 3, 4}
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
