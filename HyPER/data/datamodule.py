import random
import torch
import numpy as np
import time
from copy import deepcopy
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from .dataset import HyPERDataset
from .preprocessed import HyPEROnDiskDataset
from .splits import (
    build_or_load_train_val_test_split,
    decode_classification_labels,
    load_canonical_train_val_only,
)


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python from PyTorch's worker-aware seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class HyPERDataModule(LightningDataModule):
    r"""HyPER Data Module using VyPER-style on-disk graph preprocessing."""
    
    def __init__(
        self,
        root: str,
        train_set: Optional[str] = None,
        predict_set: Optional[str] = None,
        batch_size: int = 128,
        drop_last: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        graph_config: Optional[dict] = None,
        split_config: Optional[dict] = None,
        predict_split: Optional[str] = None,
        source_indices_file: Optional[str] = None,
        source_h5_path: Optional[str] = None,
        require_two_event_classes: bool = False,
        tuning_mode: bool = False,
        tuning_train_indices_file: Optional[str] = None,
        tuning_val_indices_file: Optional[str] = None,
        seed: int = 42,
        classification_enabled: bool = True,
        reconstruction_enabled: bool = True,
        verify_source_identity_per_event: bool = False,
        source_identity_setup_samples: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.root = root
        self.train_set = train_set
        self.predict_set = predict_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.seed = int(seed)
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        if int(self.num_workers) < 0:
            raise ValueError("dataset.num_workers must be non-negative.")
        if int(self.prefetch_factor) <= 0:
            raise ValueError("dataset.prefetch_factor must be positive.")
        self.graph_config = deepcopy(graph_config) if graph_config is not None else None
        self.split_config = deepcopy(split_config) if split_config is not None else {}
        self.predict_split = self._normalise_split_name(predict_split)
        self.source_indices_file = source_indices_file
        self.source_h5_path = source_h5_path
        self.require_two_event_classes = bool(require_two_event_classes)
        self.tuning_mode = bool(tuning_mode)
        self.tuning_train_indices_file = tuning_train_indices_file
        self.tuning_val_indices_file = tuning_val_indices_file
        self.classification_enabled = bool(classification_enabled)
        self.reconstruction_enabled = bool(reconstruction_enabled)
        if not self.classification_enabled and not self.reconstruction_enabled:
            raise ValueError("At least one of classification or reconstruction must be enabled.")
        self.verify_source_identity_per_event = bool(verify_source_identity_per_event)
        self.source_identity_setup_samples = int(source_identity_setup_samples)
        if self.source_identity_setup_samples < 0:
            raise ValueError("source_identity_setup_samples must be non-negative.")
        if self.tuning_mode and self.predict_split is not None:
            raise ValueError("Tuning data modules cannot configure a prediction/test split.")
        
        # Parse config for channel dimensions
        parsed_inputs = HyPERDataset._resolve_graph_config(root=self.root, config=self.graph_config)
        feature_layout = HyPERDataset.feature_layout_from_config(parsed_inputs)
        self.node_in_channels = len(feature_layout["node"]["resolved_features"]) + 1
        self.edge_in_channels = len(parsed_inputs['input']['edge_features'])
        self.global_in_channels = len(feature_layout["global"]["resolved_features"])
        target_cfg = parsed_inputs.get('target', {})
        self.target_encoding = str(target_cfg.get('encoding', 'typed')).strip().lower()
        if self.target_encoding != 'typed':
            raise ValueError("HyPER supports typed reconstruction targets only.")
        edge_targets = target_cfg.get('edge', {}) or {}
        hyperedge_targets = target_cfg.get('hyperedge', {}) or {}
        self.edge_target_names = list(edge_targets.keys())
        self.hyperedge_target_names = list(hyperedge_targets.keys())
        self.edge_class_names = self.edge_target_names + ['background']
        self.hyperedge_class_names = self.hyperedge_target_names + ['background']
        self.edge_out_channels = len(self.edge_class_names)
        self.hyperedge_out_channels = len(self.hyperedge_class_names)
        self.edge_background_class = self.edge_out_channels - 1
        self.hyperedge_background_class = self.hyperedge_out_channels - 1
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None
        self.split_metadata = None
        self.split_indices = None
        self.split_cache_path = None
        self.resolved_predict_split = None
        self.setup_timings = {}

    @staticmethod
    def _as_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    @staticmethod
    def _normalise_split_name(value):
        if value is None:
            return None
        name = str(value).strip().lower()
        if name in {"", "none", "null", "external"}:
            return None
        if name in {"validation", "valid"}:
            return "val"
        if name not in {"train", "val", "test"}:
            raise ValueError("predict_split must be one of train, val, test, external, null.")
        return name

    def _split_enabled(self) -> bool:
        # Persistent train/validation/test splits are the only supported path.
        return bool(self.split_config)

    def _dataset(self, name: str, training: bool):
        if name is None or not str(name).strip():
            raise ValueError("A graph-database dataset name is required.")
        t0 = time.perf_counter()
        dataset = HyPEROnDiskDataset(
            root=self.root,
            name=name,
            training=training,
            force_reload=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            config=self.graph_config,
        )
        self.setup_timings[f"dataset_{name}_{'train' if training else 'predict'}_s"] = time.perf_counter() - t0
        return dataset

    def _make_indexed_subset(self, dataset, indices, split_name: str):
        verify_each = self.verify_source_identity_per_event or split_name not in {"train", "val"}
        subset = SourceIndexSubset(
            dataset,
            indices,
            split_name=split_name,
            verify_source_identity_per_event=verify_each,
        )
        self._validate_setup_samples(dataset, subset.indices, split_name)
        return subset

    def _make_indexed_full_dataset(self, dataset, split_name: str = "all"):
        dataset = SourceIndexDataset(
            dataset,
            split_name=split_name,
            verify_source_identity_per_event=True,
        )
        indices = np.arange(len(dataset), dtype=np.int64)
        self._validate_setup_samples(dataset.dataset, indices, split_name)
        return dataset

    @staticmethod
    def _sample_positions(size: int, sample_count: int) -> np.ndarray:
        count = min(max(0, int(sample_count)), int(size))
        if count == 0:
            return np.empty((0,), dtype=np.int64)
        return np.unique(np.linspace(0, int(size) - 1, count, dtype=np.int64))

    def _validate_setup_samples(self, dataset, source_indices, split_name: str) -> None:
        """Validate source identity and cached typed targets before a loader starts."""
        source_indices = np.asarray(source_indices, dtype=np.int64)
        for position in self._sample_positions(len(source_indices), self.source_identity_setup_samples):
            source_index = int(source_indices[int(position)])
            graph = dataset[source_index]
            SourceIndexSubset._verify(graph, source_index)
            if self.reconstruction_enabled:
                self._validate_cached_reconstruction_graph(graph, source_index, split_name)

    def _validate_cached_reconstruction_graph(self, graph, source_index: int, split_name: str) -> None:
        context = (
            f"graph_db_root={self.root!r}, dataset={self.train_set!r}, "
            f"split={split_name!r}, source_event_index={int(source_index)}"
        )
        required = (
            "edge_attr_t",
            "hyperedge_attr_t",
            "edge_attr_t_class",
            "hyperedge_attr_t_class",
            "edge_reco_active",
            "hyperedge_reco_active",
        )
        missing = [name for name in required if getattr(graph, name, None) is None]
        if missing:
            raise RuntimeError(
                f"Graph cache is missing required typed reconstruction fields {missing}; {context}. "
                "Rebuild and validate the graph cache."
            )

        checks = (
            (
                "edge",
                graph.edge_attr_t,
                graph.edge_attr_t_class,
                graph.edge_reco_active,
                self.edge_out_channels,
                int(graph.edge_index.size(1)),
            ),
            (
                "hyperedge",
                graph.hyperedge_attr_t,
                graph.hyperedge_attr_t_class,
                graph.hyperedge_reco_active,
                self.hyperedge_out_channels,
                int(graph.hyperedge_index.size(1)),
            ),
        )
        for name, one_hot, cached, active, num_classes, candidate_count in checks:
            if one_hot.ndim != 2 or tuple(one_hot.shape) != (candidate_count, num_classes):
                raise RuntimeError(
                    f"Cached {name} one-hot target shape {tuple(one_hot.shape)} does not match "
                    f"({candidate_count}, {num_classes}); {context}."
                )
            if not torch.isfinite(one_hot).all():
                raise RuntimeError(f"Cached {name} targets are non-finite; {context}.")
            if not torch.all((one_hot == 0) | (one_hot == 1)) or not torch.all(
                one_hot.sum(dim=1) == 1
            ):
                raise RuntimeError(f"Cached {name} targets are not exact one-hot rows; {context}.")
            if cached.ndim != 1 or cached.numel() != candidate_count:
                raise RuntimeError(
                    f"Cached {name} integer targets have shape {tuple(cached.shape)}; "
                    f"expected ({candidate_count},); {context}."
                )
            if cached.dtype == torch.bool or cached.dtype.is_floating_point:
                raise RuntimeError(
                    f"Cached {name} integer targets have invalid dtype {cached.dtype}; {context}."
                )
            expected = one_hot.argmax(dim=1).to(torch.long)
            if not torch.equal(cached.to(torch.long), expected):
                raise RuntimeError(
                    f"Cached {name} integer targets disagree with one-hot targets; {context}."
                )
            if active.dtype != torch.bool or active.numel() != 1:
                raise RuntimeError(
                    f"Cached {name} reconstruction activity must be one boolean; {context}."
                )
            expected_active = bool((expected != num_classes - 1).any())
            if bool(active.reshape(-1)[0]) != expected_active:
                raise RuntimeError(
                    f"Cached {name} reconstruction activity is inconsistent with typed targets; "
                    f"{context}."
                )

    @staticmethod
    def _load_source_indices(filename: str, dataset_size: int) -> np.ndarray:
        path = str(filename).strip()
        if not path:
            raise ValueError("predicting.source_indices_file must be a non-empty .npy path.")
        indices = np.load(path, allow_pickle=False)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Source-index file must contain a one-dimensional integer NumPy array.")
        indices = indices.astype(np.int64, copy=False)
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Source-index file contains duplicate indices.")
        if np.any(indices < 0):
            raise ValueError("Source-index file contains negative indices.")
        if np.any(indices >= int(dataset_size)):
            raise IndexError(
                f"Source-index file contains an index outside dataset range [0, {dataset_size})."
            )
        print(f"Loaded {len(indices)} requested source indices from {path} (order preserved).")
        return indices

    @staticmethod
    def _format_counts(counts):
        if not counts:
            return "{}"
        return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) + "}"

    def _log_split_summary(self, split_result):
        metadata = split_result["metadata"]
        print("================================")
        print("HyPER train/validation/test split")
        print(f"source H5: {metadata.get('source_h5_path')}")
        print(f"n total: {metadata.get('n_events')}")
        print(f"stratify requested: {metadata.get('stratify')}")
        print(f"stratify effective: {metadata.get('effective_stratified')}")
        print(f"seed: {metadata.get('seed')}")
        print(f"split cache path: {split_result.get('cache_path')}")
        print(f"loaded existing cache: {split_result.get('loaded')}")
        print("Split sizes:")
        split_counts = metadata.get("split_counts", {})
        label_counts = metadata.get("split_label_counts", {})
        for name in ("train", "val", "test"):
            print(f"  {name}: {split_counts.get(name, 0)} {self._format_counts(label_counts.get(name, {}))}")
        print("================================")

    def _setup_explicit_split(self, stage: str | None):
        if self.train_set is None or str(self.train_set).strip() == "":
            raise ValueError("A persistent dataset split requires dataset.train_set.")

        if self.tuning_mode:
            cache_path = self.split_config.get("cache_path")
            if not cache_path:
                raise ValueError("Tuning requires an explicit existing canonical split cache path.")
            split_result = load_canonical_train_val_only(
                cache_path,
                source_h5_path=self.source_h5_path,
            )
        else:
            split_result = build_or_load_train_val_test_split(
            root=self.root,
            name=self.train_set,
            train_fraction=self.split_config.get("train_fraction", 0.8),
            val_fraction=self.split_config.get("val_fraction", 0.1),
            test_fraction=self.split_config.get("test_fraction", 0.1),
            stratify=self._as_bool(self.split_config.get("stratify", True), default=True),
            seed=int(self.split_config.get("seed", 42)),
            cache_path=self.split_config.get("cache_path", None),
            require_existing=self._as_bool(self.split_config.get("require_existing", False), default=False),
            allow_unstratified=self._as_bool(self.split_config.get("allow_unstratified", False), default=False),
            allow_zero_test=self._as_bool(self.split_config.get("allow_zero_test", False), default=False),
                source_h5_path=self.source_h5_path,
            )
        self.split_metadata = split_result["metadata"]
        self.split_indices = split_result["indices"]
        self.split_cache_path = split_result["cache_path"]
        self._log_split_summary(split_result)

        if self.tuning_mode:
            if not self.tuning_train_indices_file or not self.tuning_val_indices_file:
                raise ValueError("Tuning requires persisted train and validation subset index files.")
            canonical = split_result["indices"]
            selected = {}
            for name, filename in (
                ("train", self.tuning_train_indices_file),
                ("val", self.tuning_val_indices_file),
            ):
                values = self._load_source_indices(filename, int(self.split_metadata["n_events"]))
                allowed = set(canonical[name].tolist())
                outside = [int(value) for value in values if int(value) not in allowed]
                if outside:
                    raise ValueError(
                        f"Tuning {name} subset contains indices outside canonical {name}: {outside[:10]}."
                    )
                selected[name] = values
            split_result = {**split_result, "indices": selected}
            self.split_indices = selected
            if self.require_two_event_classes:
                import h5py
                with h5py.File(self.source_h5_path, "r") as handle:
                    label_ds = handle["LABELS/GLOBAL"]
                    failures = {}
                    for name, values in selected.items():
                        ordered = np.sort(values)
                        labels = decode_classification_labels(
                            np.asarray(label_ds[ordered]), source_h5_path=self.source_h5_path
                        )
                        unique = set(np.unique(labels).astype(int).tolist())
                        if unique != {0, 1}:
                            failures[name] = sorted(unique)
                if failures:
                    raise ValueError(
                        "Classification tuning requires both explicit event classes in each "
                        f"persisted tuning subset; failures={failures}."
                    )

        if self.require_two_event_classes:
            split_label_counts = self.split_metadata.get("split_label_counts", {})
            failures = {
                name: counts
                for name, counts in split_label_counts.items()
                if name in {"train", "val"} and not ({"0", "1"} <= set(counts))
            }
            if failures:
                raise ValueError(
                    "Classification requires both explicit LABELS/GLOBAL classes in the training "
                    f"and validation splits; insufficient class composition: {failures}."
                )

        full_data = self._dataset(
            self.train_set, training=stage not in {"predict"}
        )
        indices = split_result["indices"]
        needs_train = stage in {None, "fit"}
        needs_val = stage in {None, "fit", "validate"}
        needs_test = not self.tuning_mode and stage in {None, "test"}

        self.train_data = (
            self._make_indexed_subset(full_data, indices["train"], "train")
            if needs_train
            else None
        )
        self.val_data = (
            self._make_indexed_subset(full_data, indices["val"], "val")
            if needs_val
            else None
        )
        self.test_data = (
            self._make_indexed_subset(full_data, indices["test"], "test")
            if needs_test
            else None
        )

        if stage == "predict" and self.source_indices_file is None:
            if self.predict_split not in {"train", "val", "test"}:
                raise ValueError(
                    "Prediction from a persistent split requires predicting.split "
                    "to be train, val, or test."
                )
            self.predict_data = self._make_indexed_subset(
                full_data, indices[self.predict_split], self.predict_split
            )
            self.resolved_predict_split = self.predict_split
    
    def setup(self, stage: str | None):
        """Initialise only the datasets required by the Lightning stage."""
        if stage not in {None, "fit", "validate", "test", "predict"}:
            raise ValueError(f"Unsupported Lightning setup stage: {stage!r}")
        setup_t0 = time.perf_counter()
        if self._split_enabled():
            self._setup_explicit_split(stage)

        elif self.train_set is not None:
            raise ValueError(
                "HyPER requires a persistent dataset.split configuration; "
                "separate ad-hoc train/validation files are no longer supported."
            )

        prediction_requested = stage in {None, "predict"}
        if self.tuning_mode:
            self.predict_data = None
            self.resolved_predict_split = None
        elif prediction_requested and self.source_indices_file is not None:
            if self.predict_set is None:
                raise ValueError("predicting.source_indices_file requires dataset.predict_set.")
            prediction_source = self._dataset(self.predict_set, training=False)
            indices = self._load_source_indices(self.source_indices_file, len(prediction_source))
            requested_split = self.predict_split
            if requested_split is not None:
                if not self._split_enabled():
                    raise ValueError("A named predicting.split with source indices requires persistent splits.")
                allowed = set(np.asarray(self.split_indices[requested_split], dtype=np.int64))
                outside = [int(value) for value in indices if int(value) not in allowed]
                if outside:
                    raise ValueError(
                        f"Source-index file contains {len(outside)} events outside the persistent "
                        f"{requested_split!r} split; first invalid values: {outside[:10]}."
                    )
                split_name = requested_split
            else:
                split_name = "source_indices"
            self.predict_data = self._make_indexed_subset(prediction_source, indices, split_name)
            self.resolved_predict_split = split_name
        elif prediction_requested and self.predict_data is None and self.predict_set is not None:
            self.predict_data = self._make_indexed_full_dataset(
                self._dataset(self.predict_set, training=False), split_name="all"
            )
            self.resolved_predict_split = "all"

        required = {
            "fit": (self.train_data, self.val_data),
            "validate": (self.val_data,),
            "test": (self.test_data,),
            "predict": (self.predict_data,),
            None: (self.train_data, self.val_data, self.test_data, self.predict_data),
        }[stage]
        if any(dataset is None for dataset in required):
            raise RuntimeError(f"Dataset initialisation is incomplete for stage {stage!r}.")
        self.setup_timings["setup_total_s"] = time.perf_counter() - setup_t0

        # Print dataset summary (keep Rich - negligible overhead)
        try:
            from rich import get_console
            from rich.table import Table

            console = get_console()
            table = Table(title="Dataset Status", header_style="orange1")
            table.add_column("Name", justify="left")
            table.add_column("Value", justify="left")
            table.add_row("Drop last batch", str(self.drop_last))
            table.add_row("Batch size", str(self.batch_size))
            table.add_row("Num workers", str(self.num_workers))
            table.add_row("Persistent workers", str(self.persistent_workers))
            table.add_row("Prefetch factor", str(self.prefetch_factor))
            table.add_row("Pin memory", str(self.pin_memory))
            table.add_row("Dataset source", "validated on-disk graph database")
            if self._split_enabled():
                table.add_row("Split cache", str(self.split_cache_path))
                table.add_row("Predict split", str(self.predict_split))
            
            if self.train_data is not None:
                table.add_row("Training set", str(self.train_set))
                table.add_row("Training samples", str(len(self.train_data)))
            if self.val_data is not None:
                table.add_row("Validation set", "persistent validation split")
                table.add_row("Validation samples", str(len(self.val_data)))
            if self.test_data is not None:
                table.add_row("Test samples", str(len(self.test_data)))
            if self.predict_data is not None:
                table.add_row("Prediction set", str(self.predict_set) if self.predict_split is None else f"split:{self.predict_split}")
                table.add_row("Prediction samples", str(len(self.predict_data)))
            
            table.add_row("N node attributes", str(self.node_in_channels))
            table.add_row("N edge attributes", str(self.edge_in_channels))
            table.add_row("N glob attributes", str(self.global_in_channels))
            table.add_row("Target encoding", str(self.target_encoding))
            table.add_row("Edge output channels", str(self.edge_out_channels))
            table.add_row("Hyperedge output channels", str(self.hyperedge_out_channels))
            console.print(table)

        except ImportError:
            print(f"Dataset Status:")
            print(f"  Training samples: {len(self.train_data) if self.train_data else 'N/A'}")
            print(f"  Validation samples: {len(self.val_data) if self.val_data else 'N/A'}")
            print(f"  Test samples: {len(self.test_data) if self.test_data else 'N/A'}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Num workers: {self.num_workers}")

    def _loader_common(self) -> dict:
        kwargs = {
            "batch_size": self.batch_size,
            "pin_memory": self.pin_memory,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
            "worker_init_fn": seed_worker if self.num_workers > 0 else None,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _training_exclude_keys(self) -> list[str]:
        keys = [
            "node_p4",
            "node_ids",
            "node_truth_ids",
            "source_event_index",
        ]
        if self.reconstruction_enabled:
            keys.extend(["edge_attr_t", "hyperedge_attr_t"])
        else:
            keys.extend(
                [
                    "edge_attr_t",
                    "hyperedge_attr_t",
                    "edge_attr_t_class",
                    "hyperedge_attr_t_class",
                    "edge_reco_active",
                    "hyperedge_reco_active",
                ]
            )
        return keys

    def train_dataloader(self):
        """Return a shuffled training loader with only task-required fields."""
        return DataLoader(
            self.train_data,
            drop_last=self.drop_last,
            shuffle=True,
            exclude_keys=self._training_exclude_keys(),
            **self._loader_common(),
        )

    def val_dataloader(self):
        """Return a deterministic validation loader with training-only fields."""
        return DataLoader(
            self.val_data,
            drop_last=False,
            shuffle=False,
            exclude_keys=self._training_exclude_keys(),
            **self._loader_common(),
        )

    def test_dataloader(self):
        """Return held-out loss-evaluation data without prediction-only metadata."""
        if self.tuning_mode:
            raise RuntimeError("The test dataloader is disabled in tuning mode.")
        return DataLoader(
            self.test_data,
            drop_last=False,
            shuffle=False,
            exclude_keys=self._training_exclude_keys(),
            **self._loader_common(),
        )

    def predict_dataloader(self):
        """Return prediction data with complete provenance and output metadata."""
        if self.tuning_mode:
            raise RuntimeError("The prediction dataloader is disabled in tuning mode.")
        return DataLoader(
            self.predict_data,
            drop_last=False,
            shuffle=False,
            **self._loader_common(),
        )


class SourceIndexSubset(Dataset):
    """Map-style subset that preserves original source H5 event identity."""

    def __init__(
        self,
        dataset,
        indices,
        split_name: str,
        verify_source_identity_per_event: bool = False,
    ):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.split_name = str(split_name)
        self.verify_source_identity_per_event = bool(verify_source_identity_per_event)

    def __len__(self):
        return int(self.indices.size)

    @staticmethod
    def _source_identity(data) -> int:
        value = getattr(data, "source_event_index", None)
        if value is None or value.numel() != 1:
            raise RuntimeError("Graph is missing its scalar source_event_index identity.")
        return int(value.reshape(-1)[0])

    @staticmethod
    def _verify(data, source_index: int) -> None:
        actual = SourceIndexSubset._source_identity(data)
        if actual != int(source_index):
            raise RuntimeError(
                f"Dataset source identity mismatch: requested {source_index}, got {actual}."
            )

    def __getitem__(self, index):
        local_index = int(index)
        if local_index < 0 or local_index >= len(self):
            raise IndexError(
                f"Subset index {local_index} lies outside [0, {len(self)}) for split {self.split_name!r}."
            )
        source_index = int(self.indices[local_index])
        data = self.dataset[source_index]
        if self.verify_source_identity_per_event:
            self._verify(data, source_index)
        return data

    def __getitems__(self, indices):
        raw = np.asarray(indices)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("Batched subset indices must be a one-dimensional integer sequence.")
        local = raw.astype(np.int64, copy=False)
        if np.any(local < 0) or np.any(local >= len(self)):
            invalid = local[(local < 0) | (local >= len(self))]
            raise IndexError(
                f"Batched subset indices lie outside [0, {len(self)}): {invalid[:10].tolist()}."
            )
        source = self.indices[local]
        if hasattr(self.dataset, "__getitems__"):
            graphs = self.dataset.__getitems__(source.tolist())
        else:
            graphs = [self.dataset[int(index)] for index in source]
        if len(graphs) != len(source):
            raise RuntimeError(
                f"Batched dataset read returned {len(graphs)} graphs for {len(source)} indices."
            )
        if self.verify_source_identity_per_event:
            for graph, source_index in zip(graphs, source):
                self._verify(graph, int(source_index))
        return graphs

    def validate_source_identity(self, sample_count: int = 32) -> None:
        count = min(max(0, int(sample_count)), len(self))
        if count == 0:
            return
        positions = np.unique(np.linspace(0, len(self) - 1, count, dtype=np.int64))
        for position in positions:
            source_index = int(self.indices[int(position)])
            self._verify(self.dataset[source_index], source_index)


class SourceIndexDataset(Dataset):
    """Prediction wrapper that preserves source H5 row identity for full datasets."""

    def __init__(
        self,
        dataset,
        split_name: str = "all",
        verify_source_identity_per_event: bool = True,
    ):
        self.dataset = dataset
        self.split_name = str(split_name)
        self.verify_source_identity_per_event = bool(verify_source_identity_per_event)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source_index = int(index)
        if source_index < 0 or source_index >= len(self):
            raise IndexError(f"Dataset index {source_index} lies outside [0, {len(self)}).")
        data = self.dataset[source_index]
        if self.verify_source_identity_per_event:
            SourceIndexSubset._verify(data, source_index)
        return data

    def __getitems__(self, indices):
        raw = np.asarray(indices)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("Batched dataset indices must be a one-dimensional integer sequence.")
        source_array = raw.astype(np.int64, copy=False)
        if np.any(source_array < 0) or np.any(source_array >= len(self)):
            invalid = source_array[(source_array < 0) | (source_array >= len(self))]
            raise IndexError(
                f"Batched dataset indices lie outside [0, {len(self)}): {invalid[:10].tolist()}."
            )
        source = source_array.tolist()
        if hasattr(self.dataset, "__getitems__"):
            graphs = self.dataset.__getitems__(source)
        else:
            graphs = [self.dataset[index] for index in source]
        if len(graphs) != len(source):
            raise RuntimeError(
                f"Batched dataset read returned {len(graphs)} graphs for {len(source)} indices."
            )
        if self.verify_source_identity_per_event:
            for graph, source_index in zip(graphs, source):
                SourceIndexSubset._verify(graph, source_index)
        return graphs

    def validate_source_identity(self, sample_count: int = 32) -> None:
        count = min(max(0, int(sample_count)), len(self))
        if count == 0:
            return
        positions = np.unique(np.linspace(0, len(self) - 1, count, dtype=np.int64))
        for source_index in positions:
            data = self.dataset[int(source_index)]
            SourceIndexSubset._verify(data, int(source_index))
