"""One explicit data-isolation contract for every HyPER tuning stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


_CANONICAL_ARRAYS = ("train_idx", "val_idx", "test_idx")


def _required_file(path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"Required {description} is absent or empty: {resolved}")
    return resolved


def configure_effective_graph_dataset(cfg, *, dataset_root, dataset_name) -> dict:
    """Point a tuning config at the exact graph DB validated/staged by the wrapper."""
    root = Path(dataset_root).expanduser().resolve()
    name = str(dataset_name).strip()
    if not root.is_dir():
        raise FileNotFoundError(f"Effective graph dataset root is absent: {root}")
    if not name or Path(name).name != name:
        raise ValueError(f"Effective graph dataset name must be a plain non-empty name: {dataset_name!r}")
    database = _required_file(root / f"{name}.db", "effective graph database")
    manifest = _required_file(
        root / f"{name}.db.manifest.json", "effective graph database manifest"
    )
    OmegaConf.update(cfg, "dataset.root", str(root), merge=False, force_add=True)
    OmegaConf.update(cfg, "dataset.train_set", name, merge=False, force_add=True)
    OmegaConf.update(cfg, "dataset.predict_set", name, merge=False, force_add=True)
    return {
        "dataset_root": str(root),
        "dataset_name": name,
        "database_path": str(database),
        "manifest_path": str(manifest),
    }


def _load_indices(path, description: str) -> np.ndarray:
    values = np.load(_required_file(path, description), allow_pickle=False)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{description} must be a one-dimensional integer NumPy array: {path}")
    values = values.astype(np.int64, copy=False)
    if values.size == 0:
        raise ValueError(f"{description} must not be empty: {path}")
    if np.any(values < 0):
        raise ValueError(f"{description} contains negative source-event indices: {path}")
    if np.unique(values).size != values.size:
        raise ValueError(f"{description} contains duplicate source-event indices: {path}")
    return values


def _event_count_from_cfg(cfg) -> int | None:
    """Return an available source event range without guessing a substitute input."""
    source_h5 = cfg.get("dataset", {}).get("source_h5_path")
    if not source_h5:
        return None
    source = _required_file(source_h5, "tuning source H5")
    try:
        import h5py
        with h5py.File(source, "r") as handle:
            return int(handle["LABELS/GLOBAL"].shape[0])
    except KeyError as exc:
        raise KeyError(f"Tuning source H5 lacks LABELS/GLOBAL for event-range validation: {source}") from exc


def validate_tuning_data_isolation(cfg, *, canonical_split_path, train_indices_path,
                                   validation_indices_path) -> dict:
    """Validate canonical split membership and the persisted tuning subsets."""
    split_path = _required_file(canonical_split_path, "canonical tuning split cache")
    with np.load(split_path, allow_pickle=False) as archive:
        missing = [name for name in _CANONICAL_ARRAYS if name not in archive.files]
        if missing:
            raise KeyError(f"Canonical split cache {split_path} lacks required arrays: {missing}")
        canonical = {
            name: _load_archive_indices(archive[name], f"canonical split {name}", split_path)
            for name in _CANONICAL_ARRAYS
        }
    for first, second in (("train_idx", "val_idx"), ("train_idx", "test_idx"),
                          ("val_idx", "test_idx")):
        overlap = np.intersect1d(canonical[first], canonical[second]).size
        if overlap:
            raise ValueError(f"Canonical split cache has {overlap} overlapping {first}/{second} indices: {split_path}")
    train = _load_indices(train_indices_path, "tuning training indices")
    validation = _load_indices(validation_indices_path, "tuning validation indices")
    overlaps = {
        "train_validation_overlap": int(np.intersect1d(train, validation).size),
        "train_test_overlap": int(np.intersect1d(train, canonical["test_idx"]).size),
        "validation_test_overlap": int(np.intersect1d(validation, canonical["test_idx"]).size),
    }
    if any(overlaps.values()):
        raise ValueError("Tuning split isolation failed: " + ", ".join(f"{key}={value}" for key, value in overlaps.items()))
    outside_train = np.setdiff1d(train, canonical["train_idx"])
    outside_validation = np.setdiff1d(validation, canonical["val_idx"])
    if outside_train.size or outside_validation.size:
        raise ValueError(
            "Tuning subsets escape their canonical partitions: "
            f"outside_train={outside_train[:10].tolist()}, "
            f"outside_validation={outside_validation[:10].tolist()}, split={split_path}"
        )
    n_events = _event_count_from_cfg(cfg)
    if n_events is not None:
        maximum = max(int(train.max()), int(validation.max()), *(int(values.max()) for values in canonical.values()))
        if maximum >= n_events:
            raise IndexError(f"Tuning index {maximum} is outside source event range [0, {n_events})")
    return {
        "canonical_split": str(split_path), "training_indices": str(Path(train_indices_path).resolve()),
        "validation_indices": str(Path(validation_indices_path).resolve()),
        "training_count": int(train.size), "validation_count": int(validation.size),
        "canonical_counts": {name: int(values.size) for name, values in canonical.items()},
        "event_count": n_events, **overlaps,
    }


def _load_archive_indices(values, description: str, split_path: Path) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{description} must be a one-dimensional integer array: {split_path}")
    values = values.astype(np.int64, copy=False)
    if values.size == 0:
        raise ValueError(f"{description} must not be empty: {split_path}")
    if np.any(values < 0) or np.unique(values).size != values.size:
        raise ValueError(f"{description} contains invalid or duplicate indices: {split_path}")
    return values


def configure_tuning_data_isolation(cfg, *, canonical_split_path, train_indices_path,
                                    validation_indices_path) -> dict:
    """Apply and validate the one permissible data contract for tuning."""
    split_path = _required_file(canonical_split_path, "canonical tuning split cache")
    train_path = _required_file(train_indices_path, "tuning training indices")
    validation_path = _required_file(validation_indices_path, "tuning validation indices")
    OmegaConf.update(cfg, "dataset.split.cache_path", str(split_path), merge=False, force_add=True)
    OmegaConf.update(cfg, "dataset.split.require_existing", True, merge=False, force_add=True)
    OmegaConf.update(cfg, "dataset.split.predict_split", None, merge=False, force_add=True)
    OmegaConf.update(cfg, "predicting.split", None, merge=False, force_add=True)
    OmegaConf.update(cfg, "predicting.source_indices_file", None, merge=False, force_add=True)
    OmegaConf.update(cfg, "tuning.train_indices_file", str(train_path), merge=False, force_add=True)
    OmegaConf.update(cfg, "tuning.validation_indices_file", str(validation_path), merge=False, force_add=True)
    return validate_tuning_data_isolation(
        cfg, canonical_split_path=split_path, train_indices_path=train_path,
        validation_indices_path=validation_path,
    )
