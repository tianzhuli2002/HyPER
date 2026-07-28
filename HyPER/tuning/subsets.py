"""Deterministic fixed-count tuning subsets drawn only from canonical train/val."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

from HyPER.data.splits import load_canonical_train_val_only, read_classification_labels
from HyPER.models.loss import combine_reconstruction_activity


def index_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _cantor(a, b):
    total = a.astype(np.int64) + b.astype(np.int64)
    return (total * (total + 1)) // 2 + b.astype(np.int64)


def _activity(handle: h5py.File, config_path, n_events: int):
    """Exact typed-target activity without using matching/category proxies.

    The graph builder creates every pair/combination of valid nodes. Therefore
    a non-background typed target exists precisely when each source node ID in
    one configured typed connection is present. This is the same condition fed
    to ``reconstruction_active_from_typed_targets`` after candidate targets are
    built; it is evaluated vectorially here for production-scale H5 files.
    """
    if not config_path:
        return None, "unavailable:no_graph_config"
    config = yaml.safe_load(Path(config_path).read_text())
    inputs, targets = config["input"], config["target"]
    nodes, features = inputs["nodes"], inputs["node_features"]
    connections = {"edge": [], "hyperedge": []}
    for kind in ("edge", "hyperedge"):
        for value in (targets.get(kind) or {}).values():
            values = [value] if value and all(isinstance(x, str) for x in value) else value
            connections[kind].extend(values or [])
    parsed = {
        kind: [[tuple(map(int, item.split("-"))) for item in connection] for connection in values]
        for kind, values in connections.items()
    }
    result = {kind: np.zeros(n_events, dtype=bool) for kind in parsed}
    for start in range(0, n_events, 100_000):
        end = min(n_events, start + 100_000)
        present = {}
        for name, type_id in nodes.items():
            raw = np.asarray(handle[f"INPUTS/{name}"][start:end])
            truth = np.asarray(handle[f"LABELS/{name}"][start:end])
            valid = np.ones(raw.shape, dtype=bool)
            for feature in features: valid &= np.isfinite(raw[feature])
            valid &= np.isfinite(truth)
            safe_truth = np.where(np.isfinite(truth), truth, 0)
            configured_pairs = {
                pair for kind_connections in parsed.values()
                for conn in kind_connections for pair in conn if pair[0] == type_id
            }
            for target in configured_pairs:
                node_id = _cantor(np.full(truth.shape, target[0]), safe_truth)
                present[target] = present.get(target, np.zeros(truth.shape, bool)) | (valid & (node_id == _cantor(np.array(target[0]), np.array(target[1]))))
        for kind, kind_connections in parsed.items():
            active = np.zeros(end - start, bool)
            for connection in kind_connections:
                active |= np.logical_and.reduce([
                    present.get(pair, np.zeros((end-start, 1), bool)).any(axis=1)
                    for pair in connection
                ])
            result[kind][start:end] = active
    combined = combine_reconstruction_activity(result["edge"], result["hyperedge"])
    return combined, "any_non_background_typed_edge_or_hyperedge_target"


def _allocate(groups, wanted, total):
    """Largest-remainder proportional allocation, retaining every viable group."""
    exact = {key: wanted * len(values) / total for key, values in groups.items()}
    counts = {key: min(len(groups[key]), int(np.floor(value))) for key, value in exact.items()}
    remaining = wanted - sum(counts.values())
    for key in sorted(groups, key=lambda key: (-(exact[key] - counts[key]), str(key))):
        if not remaining:
            break
        if counts[key] < len(groups[key]):
            counts[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("Could not allocate requested deterministic subset count.")
    return counts


def _subset(parent, labels, activity, count, seed):
    parent = np.asarray(parent, dtype=np.int64)
    wanted = int(count)
    if wanted <= 0 or wanted > len(parent):
        raise ValueError(f"Requested tuning subset count {wanted} exceeds canonical partition size {len(parent)}.")
    # Joint stratification preserves classification and reconstruction activity.
    keys = [(int(labels[index]), int(activity[index]) if activity is not None else None) for index in parent]
    groups = {}
    for index, key in zip(parent, keys):
        groups.setdefault(key, []).append(index)
    groups = {key: np.asarray(values, dtype=np.int64) for key, values in groups.items()}
    rng = np.random.default_rng(int(seed))
    for values in groups.values():
        rng.shuffle(values)
    take = _allocate(groups, wanted, len(parent))
    selected = np.concatenate([groups[key][:take[key]] for key in sorted(groups, key=str)])
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _counts(values, indices):
    keys, counts = np.unique(values[np.asarray(indices, dtype=np.int64)], return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(keys, counts)}


def _source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_subsets(split_cache, source_h5, output_dir, *, seed=42,
                    train_count=None, validation_count=None,
                    train_fraction=None, validation_fraction=None, graph_config=None):
    """Persist subsets; Stage 1 supplies counts, promotion supplies explicit fractions."""
    started = time.perf_counter()
    source = Path(source_h5).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Required tuning source H5 does not exist: {source}")
    canonical = load_canonical_train_val_only(split_cache, source_h5_path=source)
    if train_count is None:
        if train_fraction is None: raise ValueError("A train count is required.")
        train_count = max(1, int(round(len(canonical["indices"]["train"]) * float(train_fraction))))
    if validation_count is None:
        if validation_fraction is None: raise ValueError("A validation count is required.")
        validation_count = max(1, int(round(len(canonical["indices"]["val"]) * float(validation_fraction))))
    labels_started = time.perf_counter()
    with h5py.File(source, "r") as handle:
        labels = read_classification_labels(handle, source_h5_path=source)
        labels_seconds = time.perf_counter() - labels_started
        activity_started = time.perf_counter()
        activity, activity_source = _activity(handle, graph_config, len(labels))
        activity_seconds = time.perf_counter() - activity_started
    if len(labels) != int(canonical["metadata"]["n_events"]):
        raise ValueError(
            f"Classification-label count {len(labels)} in {source} does not match canonical "
            f"split event count {canonical['metadata']['n_events']}."
        )
    sampling_started = time.perf_counter()
    train = _subset(canonical["indices"]["train"], labels, activity, train_count, seed)
    val = _subset(canonical["indices"]["val"], labels, activity, validation_count, seed + 1)
    # This reads test_idx only for an audit assertion; it never contributes to sampling.
    with np.load(Path(split_cache), allow_pickle=False) as loaded:
        test = np.asarray(loaded["test_idx"], dtype=np.int64)
    test_overlap = int(np.intersect1d(np.concatenate((train, val)), test).size)
    if test_overlap:
        raise RuntimeError(f"Stage-1 subset overlaps canonical test indices ({test_overlap}).")
    if len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val):
        raise RuntimeError("Generated tuning subsets contain duplicate source-event indices.")
    if np.intersect1d(train, val).size:
        raise RuntimeError("Generated tuning train and validation subsets overlap.")
    if not np.isin(train, canonical["indices"]["train"]).all():
        raise RuntimeError("Generated tuning train subset escapes canonical train.")
    if not np.isin(val, canonical["indices"]["val"]).all():
        raise RuntimeError("Generated tuning validation subset escapes canonical validation.")
    sampling_seconds = time.perf_counter() - sampling_started
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out / "tuning_train_indices.npy", out / "tuning_validation_indices.npy"
    manifest_path = out / "tuning_subset_manifest.json"
    request = {"topology": str(out.name), "canonical_split_path": str(Path(split_cache).resolve()),
               "source_split_hash": _source_hash(Path(split_cache).resolve()), "canonical_source_h5": str(source),
               "subset_seed": int(seed), "requested_train_count": int(train_count),
               "requested_validation_count": int(validation_count),
               "requested_train_fraction": train_fraction, "requested_validation_fraction": validation_fraction}
    if train_path.exists() or val_path.exists() or manifest_path.exists():
        if not (train_path.is_file() and val_path.is_file() and manifest_path.is_file()):
            raise RuntimeError(f"Incomplete persisted tuning subset at {out}; refusing to overwrite it.")
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(key) != value for key, value in request.items()):
            raise RuntimeError(f"Persisted tuning subset metadata mismatch at {out}; choose a new path.")
        persisted_train = np.load(train_path, allow_pickle=False)
        persisted_val = np.load(val_path, allow_pickle=False)
        checks = {
            "actual_train_count": len(persisted_train),
            "actual_validation_count": len(persisted_val),
            "train_subset_hash": index_hash(persisted_train),
            "validation_subset_hash": index_hash(persisted_val),
        }
        mismatches = {key: (existing.get(key), value) for key, value in checks.items()
                      if existing.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Persisted tuning subset validation failed at {out}: {mismatches}")
        return existing
    writing_started = time.perf_counter()
    np.save(train_path, train, allow_pickle=False); np.save(val_path, val, allow_pickle=False)
    metadata = {**request, "canonical_train_hash": canonical["metadata"]["split_index_sha256"]["train"],
                "canonical_validation_hash": canonical["metadata"]["split_index_sha256"]["val"],
                "canonical_test_hash": canonical["metadata"]["split_index_sha256"]["test"],
                "actual_train_count": len(train), "actual_validation_count": len(val), "test_count": 0,
                "test_indices_used_for_sampling": False, "test_indices_loaded_for_overlap_validation": True,
                "train_subset_hash": index_hash(train), "validation_subset_hash": index_hash(val),
                "test_overlap_count": test_overlap, "train_class_counts": _counts(labels, train),
                "validation_class_counts": _counts(labels, val), "reconstruction_activity_source": activity_source,
                "train_reconstruction_activity_counts": _counts(activity.astype(np.int8), train) if activity is not None else {},
                "validation_reconstruction_activity_counts": _counts(activity.astype(np.int8), val) if activity is not None else {},
                "train_indices_file": str(train_path.resolve()), "validation_indices_file": str(val_path.resolve()),
                "created_timestamp": datetime.now(timezone.utc).isoformat(),
                "timing_seconds": {"read_classification_labels": labels_seconds,
                                   "derive_reconstruction_activity": activity_seconds,
                                   "sample_and_validate": sampling_seconds,
                                   "write_outputs": 0.0, "total": 0.0},
                "peak_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata["timing_seconds"]["write_outputs"] = time.perf_counter() - writing_started
    metadata["timing_seconds"]["total"] = time.perf_counter() - started
    metadata["peak_memory_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
