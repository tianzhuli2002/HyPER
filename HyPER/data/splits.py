"""Deterministic train/validation/test split helpers for HyPER H5 datasets."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = 4
CLASSIFICATION_LABEL_FIELD = "label"


def dataset_h5_path(root: str | Path, name: str) -> Path:
    return Path(root) / "raw" / f"{name}.h5"


def _normalise_fraction(value: Any, name: str) -> float:
    out = float(value)
    if out < 0:
        raise ValueError(f"dataset.split.{name} must be non-negative, got {out}.")
    return out


def decode_classification_labels(
    values: np.ndarray,
    *,
    source_h5_path: str | Path,
    dataset_name: str = "LABELS/GLOBAL",
    expected_field: str = CLASSIFICATION_LABEL_FIELD,
) -> np.ndarray:
    """Decode the one explicit event-classification label per source event."""
    raw = np.asarray(values)
    context = (
        f"source={Path(source_h5_path).expanduser().resolve()}, dataset={dataset_name}, "
        f"dtype={raw.dtype}, expected_field={expected_field!r}, observed_shape={raw.shape}"
    )
    if raw.dtype.names is not None:
        if expected_field not in raw.dtype.names:
            raise TypeError(f"Incompatible structured classification labels: {context}.")
        raw = np.asarray(raw[expected_field])
    if raw.ndim == 1:
        labels = raw
    elif raw.ndim == 2 and raw.shape[1] == 1:
        labels = raw[:, 0]
    else:
        raise ValueError(f"Classification labels must contain exactly one value per event: {context}.")
    if labels.size == 0 or not np.issubdtype(labels.dtype, np.number):
        raise TypeError(f"Classification labels must be a non-empty numeric array: {context}.")
    if not np.isfinite(labels).all() or not np.isin(np.unique(labels), [0, 1]).all():
        raise ValueError(f"Classification labels must be finite binary values 0/1: {context}.")
    return labels.astype(np.int8, copy=False)


def read_classification_labels(
    handle: h5py.File,
    *,
    source_h5_path: str | Path,
    expected_field: str = CLASSIFICATION_LABEL_FIELD,
) -> np.ndarray:
    dataset_name = "LABELS/GLOBAL"
    if dataset_name not in handle:
        raise KeyError(f"{Path(source_h5_path).resolve()} has no explicit {dataset_name} event label.")
    return decode_classification_labels(
        handle[dataset_name][:], source_h5_path=source_h5_path,
        dataset_name=dataset_name, expected_field=expected_field,
    )


def _read_num_events_and_labels(h5_path: Path) -> tuple[int, np.ndarray, dict[str, int]]:
    with h5py.File(h5_path, "r") as handle:
        if "INPUTS" not in handle or "GLOBAL" not in handle["INPUTS"]:
            raise KeyError(f"{h5_path} does not contain INPUTS/GLOBAL.")
        n_events = int(len(handle["INPUTS"]["GLOBAL"]))
        labels = read_classification_labels(handle, source_h5_path=h5_path)
        if labels.shape[0] != n_events:
            raise ValueError(
                f"{h5_path}: LABELS/GLOBAL length {labels.shape[0]} does not match "
                f"INPUTS/GLOBAL length {n_events}."
            )
        values, counts = np.unique(labels, return_counts=True)
        label_counts = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return n_events, labels, label_counts


def _default_cache_paths(root: str | Path, name: str, metadata_key: dict[str, Any]) -> tuple[Path, Path]:
    root_path = Path(root).resolve()
    repo_guess = root_path.parent
    digest = hashlib.sha256(json.dumps(metadata_key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem = f"{name}_split_{digest}"
    split_dir = repo_guess / "results" / "splits"
    return split_dir / f"{stem}.npz", split_dir / f"{stem}.json"


def _metadata_key(
    h5_path: Path,
    n_events: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    stratify: bool,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_h5_path": str(h5_path.resolve()),
        "n_events": int(n_events),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "stratify": bool(stratify),
        "seed": int(seed),
    }


def _is_compatible(metadata: dict[str, Any], key: dict[str, Any]) -> bool:
    for name, value in key.items():
        if metadata.get(name) != value:
            return False
    return int(metadata.get("schema_version", -1)) == SCHEMA_VERSION


def _random_partition(indices: np.ndarray, fractions: tuple[float, float, float], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    n_total = int(shuffled.size)
    n_test = int(round(n_total * fractions[2]))
    n_val = int(round(n_total * fractions[1]))
    if n_test + n_val > n_total:
        overflow = n_test + n_val - n_total
        n_val = max(0, n_val - overflow)
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def _stratified_partition(labels: np.ndarray, fractions: tuple[float, float, float], seed: int) -> dict[str, np.ndarray]:
    labels = np.asarray(labels).reshape(-1)
    valid = labels >= 0
    if not valid.all():
        raise ValueError("Cannot stratify with non-finite / invalid labels.")
    rng = np.random.default_rng(int(seed))
    output = {"train": [], "val": [], "test": []}
    for label in sorted(np.unique(labels).tolist()):
        class_indices = np.nonzero(labels == label)[0].astype(np.int64)
        rng.shuffle(class_indices)
        n_class = int(class_indices.size)
        n_test = int(round(n_class * fractions[2]))
        n_val = int(round(n_class * fractions[1]))
        if n_test + n_val > n_class:
            overflow = n_test + n_val - n_class
            n_val = max(0, n_val - overflow)
        output["test"].append(class_indices[:n_test])
        output["val"].append(class_indices[n_test:n_test + n_val])
        output["train"].append(class_indices[n_test + n_val:])
    return {
        name: np.sort(np.concatenate(parts).astype(np.int64)) if parts else np.asarray([], dtype=np.int64)
        for name, parts in output.items()
    }


def _counts_for_indices(labels: np.ndarray | None, indices: np.ndarray) -> dict[str, int]:
    if labels is None:
        return {}
    selected = labels[np.asarray(indices, dtype=np.int64)]
    values, counts = np.unique(selected, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _index_hash(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _load_cached(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as loaded:
        return {
            "train": loaded["train_idx"].astype(np.int64),
            "val": loaded["val_idx"].astype(np.int64),
            "test": loaded["test_idx"].astype(np.int64),
        }


def load_canonical_train_val_only(
    cache_path: str | Path,
    *,
    source_h5_path: str | Path,
) -> dict[str, Any]:
    """Load and validate canonical train/val arrays without reading test_idx.

    Tuning uses this deliberately narrow loader.  The held-out array remains an
    unopened member of the NPZ; its count and hash are retained from the
    canonical JSON metadata for audit provenance only.
    """
    npz_path = Path(cache_path).expanduser().resolve()
    json_path = npz_path.with_suffix(".json")
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Required canonical split pair is missing: {npz_path}, {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    requested_source = str(Path(source_h5_path).expanduser().resolve())
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"Canonical split schema mismatch in {json_path}.")
    if metadata.get("source_h5_path") != requested_source:
        raise RuntimeError(
            f"Canonical split source mismatch: metadata={metadata.get('source_h5_path')}, "
            f"requested={requested_source}."
        )
    with np.load(npz_path, allow_pickle=False) as loaded:
        train = loaded["train_idx"].astype(np.int64)
        val = loaded["val_idx"].astype(np.int64)
    n_events = int(metadata["n_events"])
    for name, values in (("train", train), ("val", val)):
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ValueError(f"Canonical {name} indices must be unique one-dimensional integers.")
        if np.any(values < 0) or np.any(values >= n_events):
            raise ValueError(f"Canonical {name} indices are outside [0, {n_events}).")
        expected_hash = metadata.get("split_index_sha256", {}).get(name)
        if expected_hash != _index_hash(values):
            raise RuntimeError(f"Canonical {name} index hash mismatch in {npz_path}.")
    if np.intersect1d(train, val).size:
        raise ValueError("Canonical train and validation indices overlap.")
    return {
        "indices": {"train": train, "val": val},
        "metadata": metadata,
        "cache_path": str(npz_path),
        "test_array_loaded": False,
    }


def _validate_partition(indices: dict[str, np.ndarray], n_events: int) -> None:
    expected_names = {"train", "val", "test"}
    if set(indices) != expected_names:
        raise ValueError(f"Split must contain exactly {sorted(expected_names)}, got {sorted(indices)}.")
    arrays = {}
    for name in sorted(expected_names):
        values = np.asarray(indices[name])
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Split {name!r} indices must be a one-dimensional integer array.")
        values = values.astype(np.int64, copy=False)
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Split {name!r} contains duplicate source indices.")
        if np.any(values < 0) or np.any(values >= int(n_events)):
            raise ValueError(f"Split {name!r} contains source indices outside [0, {n_events}).")
        arrays[name] = values
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(arrays[left], arrays[right]).size:
            raise ValueError(f"Splits {left!r} and {right!r} overlap.")
    union = np.sort(np.concatenate([arrays[name] for name in ("train", "val", "test")]))
    if not np.array_equal(union, np.arange(int(n_events), dtype=np.int64)):
        raise ValueError("Train/validation/test split union does not equal the full source event set.")


def build_or_load_train_val_test_split(
    root: str | Path,
    name: str,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    stratify: bool = True,
    seed: int = 42,
    cache_path: str | Path | None = None,
    require_existing: bool = False,
    allow_unstratified: bool = False,
    allow_zero_test: bool = False,
    source_h5_path: str | Path | None = None,
) -> dict[str, Any]:
    h5_path = (
        Path(source_h5_path).expanduser()
        if source_h5_path is not None and str(source_h5_path).strip()
        else dataset_h5_path(root, name)
    )
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    train_fraction = _normalise_fraction(train_fraction, "train_fraction")
    val_fraction = _normalise_fraction(val_fraction, "val_fraction")
    test_fraction = _normalise_fraction(test_fraction, "test_fraction")
    total_fraction = train_fraction + val_fraction + test_fraction
    if abs(total_fraction - 1.0) > 1e-8:
        raise ValueError(f"dataset.split fractions must sum to 1.0, got {total_fraction}.")
    if train_fraction <= 0 or val_fraction <= 0:
        raise ValueError("dataset.split train_fraction and val_fraction must be non-zero for training.")
    if test_fraction <= 0 and not allow_zero_test:
        raise ValueError("dataset.split test_fraction can be zero only with allow_zero_test=true.")

    n_h5_events, labels, source_label_counts = _read_num_events_and_labels(h5_path)
    selected_indices = np.arange(n_h5_events, dtype=np.int64)
    n_events = n_h5_events
    selected_labels = labels
    label_counts = _counts_for_indices(labels, selected_indices)

    metadata_key = _metadata_key(
        h5_path=h5_path,
        n_events=n_events,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        stratify=stratify,
        seed=seed,
    )

    if cache_path is None or str(cache_path).strip() == "":
        npz_path, json_path = _default_cache_paths(root, name, metadata_key)
    else:
        npz_path = Path(cache_path)
        json_path = npz_path.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    if npz_path.exists() and json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            cached_metadata = json.load(handle)
        if _is_compatible(cached_metadata, metadata_key):
            indices = _load_cached(npz_path)
            _validate_partition(indices, n_events)
            return {"indices": indices, "metadata": cached_metadata, "cache_path": str(npz_path), "loaded": True}
        raise RuntimeError(
            f"Existing split cache metadata mismatches the requested split: {json_path}. "
            "HyPER will not overwrite it automatically; choose a fresh cache path or rebuild intentionally."
        )

    elif npz_path.exists() or json_path.exists():
        raise RuntimeError(
            f"Incomplete split cache pair ({npz_path}, {json_path}). HyPER will not overwrite it automatically."
        )
    elif require_existing:
        raise FileNotFoundError(f"Required split cache not found: {npz_path}")

    use_stratified = bool(stratify)
    fallback_reason = None
    if use_stratified and len(np.unique(selected_labels)) < 2:
        use_stratified = False
        fallback_reason = "fewer than two label classes"
    if use_stratified and np.any(selected_labels < 0):
        if allow_unstratified:
            use_stratified = False
            fallback_reason = "invalid labels present"
        else:
            raise ValueError("Invalid labels found; set dataset.split.allow_unstratified=true to fall back.")
    if not use_stratified and bool(stratify) and fallback_reason:
        print(f"WARNING: Falling back to non-stratified split: {fallback_reason}")
    if not use_stratified and selected_labels is not None and bool(stratify) and not fallback_reason and not allow_unstratified:
        raise ValueError("Stratified split requested but could not be built.")

    fractions = (train_fraction, val_fraction, test_fraction)
    if use_stratified:
        relative = _stratified_partition(selected_labels, fractions=fractions, seed=seed)
    else:
        relative = _random_partition(
            np.arange(n_events, dtype=np.int64), fractions=fractions, seed=seed
        )
    indices = {
        name: np.sort(selected_indices[np.asarray(value, dtype=np.int64)])
        for name, value in relative.items()
    }
    _validate_partition(indices, n_events)

    metadata = dict(metadata_key)
    metadata.update(
        {
            "created_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_h5_events": int(n_h5_events),
            "source_label_counts": source_label_counts,
            "selected_source_events": "all_events",
            "label_counts": label_counts,
            "effective_stratified": bool(use_stratified),
            "fallback_reason": fallback_reason,
            "split_counts": {name: int(len(value)) for name, value in indices.items()},
            "split_label_counts": {
                name: _counts_for_indices(labels, value) for name, value in indices.items()
            },
            "split_label_fractions": {
                name: {
                    label: count / len(indices[name]) if len(indices[name]) else 0.0
                    for label, count in _counts_for_indices(labels, indices[name]).items()
                }
                for name in ("train", "val", "test")
            },
            "split_index_sha256": {
                name: _index_hash(value) for name, value in indices.items()
            },
            "npz_path": str(npz_path),
        }
    )
    np.savez_compressed(
        npz_path,
        train_idx=indices["train"],
        val_idx=indices["val"],
        test_idx=indices["test"],
    )
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {"indices": indices, "metadata": metadata, "cache_path": str(npz_path), "loaded": False}
