import warnings
import os
import torch
import numpy as np
import time
from copy import deepcopy
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader

from .dataset import HyPERDataset
from .preprocessed import HyPEROnDiskDataset
from .splits import build_or_load_train_val_test_split


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker process with unique random seed for reproducibility."""
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)
    os.environ['PYTHONHASHSEED'] = str(42 + worker_id)


class HyPERDataModule(LightningDataModule):
    r"""HyPER Data Module using VyPER-style on-disk graph preprocessing."""
    
    def __init__(
        self,
        root: str,
        train_set: Optional[str] = None,
        val_set: Optional[str] = None,
        predict_set: Optional[str] = None,
        batch_size: int = 128,
        max_n_events: int = -1,
        percent_valid_samples: float = 0.05,
        drop_last: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
        use_ondisk: bool = True,
        graph_config: Optional[dict] = None,
        split_config: Optional[dict] = None,
        predict_split: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.root = root
        self.train_set = train_set
        self.val_set = val_set
        self.predict_set = predict_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.max_n_events = max_n_events
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.percent_valid_samples = percent_valid_samples
        self.cache_dir = cache_dir
        self.force_reload = force_reload
        self.use_ondisk = use_ondisk
        self.graph_config = deepcopy(graph_config) if graph_config is not None else None
        self.split_config = deepcopy(split_config) if split_config is not None else {}
        self.predict_split = self._normalise_split_name(predict_split)
        
        # Parse config for channel dimensions
        parsed_inputs = HyPERDataset._resolve_graph_config(root=self.root, config=self.graph_config)
        feature_layout = HyPERDataset.feature_layout_from_config(parsed_inputs)
        self.node_in_channels = len(feature_layout["node"]["resolved_features"]) + 1
        self.edge_in_channels = len(parsed_inputs['input']['edge_features'])
        self.global_in_channels = len(feature_layout["global"]["resolved_features"])
        target_cfg = parsed_inputs.get('target', {})
        self.target_encoding = str(target_cfg.get('encoding', 'binary')).strip().lower()
        if self.target_encoding not in {'binary', 'typed'}:
            raise ValueError("target.encoding must be either 'binary' or 'typed'.")
        edge_targets = target_cfg.get('edge', {}) or {}
        hyperedge_targets = target_cfg.get('hyperedge', {}) or {}
        self.edge_target_names = list(edge_targets.keys())
        self.hyperedge_target_names = list(hyperedge_targets.keys())
        self.edge_out_channels = len(self.edge_target_names) + 1 if self.target_encoding == 'typed' else 1
        self.hyperedge_out_channels = len(self.hyperedge_target_names) + 1 if self.target_encoding == 'typed' else 1
        self.edge_background_class = self.edge_out_channels - 1 if self.target_encoding == 'typed' else None
        self.hyperedge_background_class = self.hyperedge_out_channels - 1 if self.target_encoding == 'typed' else None
        
        # Track if hyperedges are used (for conditional follow_batch)
        self._use_hyperedge = len(hyperedge_targets) > 0

        self.index_range = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.predict_data = None
        self.split_metadata = None
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
        return self._as_bool(self.split_config.get("enabled", False), default=False)

    def _dataset(self, name: str, training: bool):
        t0 = time.perf_counter()
        if self.use_ondisk:
            dataset = HyPEROnDiskDataset(
                root=self.root,
                name=name,
                training=training,
                force_reload=self.force_reload,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                config=self.graph_config,
            )
        else:
            dataset = HyPERDataset(
                root=self.root,
                name=name,
                training=training,
                force_reload=self.force_reload,
                config=self.graph_config,
            )
        self.setup_timings[f"dataset_{name}_{'train' if training else 'predict'}_s"] = time.perf_counter() - t0
        return dataset

    def _make_indexed_subset(self, dataset, indices, split_name: str):
        return SourceIndexSubset(dataset, indices, split_name=split_name)

    def _make_indexed_full_dataset(self, dataset, split_name: str = "all"):
        return SourceIndexDataset(dataset, split_name=split_name)

    @staticmethod
    def _format_counts(counts):
        if not counts:
            return "{}"
        return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) + "}"

    def _log_split_summary(self, split_result):
        metadata = split_result["metadata"]
        print("================================")
        print("HyPER train/validation/test split")
        print("split enabled: true")
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

    def _setup_explicit_split(self):
        if self.train_set is None or str(self.train_set).strip() == "":
            raise ValueError("dataset.split.enabled=true requires dataset.train_set to identify the source H5.")

        split_max_events = self.max_n_events if self.max_n_events is not None and int(self.max_n_events) > 0 else None
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
            max_events=split_max_events,
        )
        self.split_metadata = split_result["metadata"]
        self.split_cache_path = split_result["cache_path"]
        self._log_split_summary(split_result)

        full_data = self._dataset(self.train_set, training=True)
        indices = split_result["indices"]
        self.train_data = self._make_indexed_subset(full_data, indices["train"], "train")
        self.val_data = self._make_indexed_subset(full_data, indices["val"], "val")
        self.test_data = self._make_indexed_subset(full_data, indices["test"], "test")

        if self.predict_split == "train":
            self.predict_data = self.train_data
            self.resolved_predict_split = "train"
        elif self.predict_split == "val":
            self.predict_data = self.val_data
            self.resolved_predict_split = "val"
        elif self.predict_split == "test":
            self.predict_data = self.test_data
            self.resolved_predict_split = "test"
    
    def setup(self, stage: str):
        """Setting up datasets."""
        setup_t0 = time.perf_counter()
        if self._split_enabled():
            self._setup_explicit_split()

        elif self.train_set is not None:
            if self.val_set is None or self.val_set == "" or self.train_set == self.val_set:
                # Split training set into train/val
                print(f"Creating validation set using "
                      f"{round(self.percent_valid_samples * 100, 2)}% of the training file.")
                
                full_data = self._dataset(self.train_set, training=True)
                
                n_total = len(full_data)
                n_val = max(1, int(n_total * self.percent_valid_samples))
                n_train = n_total - n_val
                
                self.train_data, self.val_data = random_split(
                    full_data,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(42)
                )
            else:
                # Separate train and val files
                self.train_data = self._dataset(self.train_set, training=True)
                self.val_data = self._dataset(self.val_set, training=True)

            # Limit training dataset size if requested
            if self.max_n_events == -1:
                pass
            elif self.max_n_events >= len(self.train_data):
                warnings.warn(f"`max_n_events` ({self.max_n_events}) larger than dataset "
                              f"({len(self.train_data)}), using all events.")
            elif self.max_n_events > 0:
                self.index_range = list(range(self.max_n_events))
                print(f"Limited training to {self.max_n_events} events.")

        if self.predict_data is None and self.predict_set is not None:
            self.predict_data = self._make_indexed_full_dataset(
                self._dataset(self.predict_set, training=False),
                split_name="all",
            )
            self.resolved_predict_split = "all"

        if self.train_data is None and self.val_data is None and self.predict_data is None:
            raise ValueError("No datasets have been provided. Abort!")
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
            table.add_row("Use on-disk DB", str(self.use_ondisk))
            table.add_row("Force reload", str(self.force_reload))
            table.add_row("Batch size", str(self.batch_size))
            table.add_row("Num workers", str(self.num_workers))
            table.add_row("Persistent workers", str(self.persistent_workers))
            table.add_row("Prefetch factor", str(self.prefetch_factor))
            table.add_row("Pin memory", str(self.pin_memory))
            table.add_row("Dataset mode", "ondisk" if self.use_ondisk else "raw_h5")
            table.add_row("Explicit split enabled", str(self._split_enabled()))
            if self._split_enabled():
                table.add_row("Split cache", str(self.split_cache_path))
                table.add_row("Predict split", str(self.predict_split))
            
            if self.train_data is not None:
                table.add_row("Training set", str(self.train_set))
                table.add_row("Training samples", str(len(self.index_range) if self.index_range else len(self.train_data)))
            if self.val_data is not None:
                table.add_row("Validation set", str(self.val_set) if self.val_set else "split from training set")
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

    def train_dataloader(self):
        """Returns training DataLoader - OPTIMIZED."""
        # === SPEEDUP 1: Conditional follow_batch (like VyPER) ===
        follow_batch = ['edge_attr', 'hyperedge_index'] if self._use_hyperedge else ['edge_attr']
        
        # === SPEEDUP 2: Remove EventSampler overhead if not needed ===
        if self.index_range is not None:
            # Use simple Subset instead of custom sampler
            from torch.utils.data import Subset
            dataset = Subset(self.train_data, self.index_range)
            shuffle = True
        else:
            dataset = self.train_data
            shuffle = True
        
        # === SPEEDUP 3: Optimize prefetch_factor ===
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        
        # === SPEEDUP 4: Use worker_init_fn for reproducible seeding ===
        worker_init_fn_to_use = worker_init_fn if self.num_workers > 0 else None
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            follow_batch=follow_batch,  # ← Conditional!
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=self.drop_last,
            shuffle=shuffle,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn_to_use,
        )

    def val_dataloader(self):
        """Returns validation DataLoader - OPTIMIZED."""
        follow_batch = ['edge_attr', 'hyperedge_index'] if self._use_hyperedge else ['edge_attr']
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        
        # === Use worker_init_fn for reproducible seeding ===
        worker_init_fn_to_use = worker_init_fn if self.num_workers > 0 else None
        
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            follow_batch=follow_batch,  # ← Conditional!
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=self.drop_last,
            shuffle=False,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn_to_use,
        )

    def test_dataloader(self):
        """Returns held-out test DataLoader."""
        follow_batch = ['edge_attr', 'hyperedge_index'] if self._use_hyperedge else ['edge_attr']
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        worker_init_fn_to_use = worker_init_fn if self.num_workers > 0 else None
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            follow_batch=follow_batch,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False,
            shuffle=False,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn_to_use,
        )

    def predict_dataloader(self):
        """Returns prediction DataLoader - OPTIMIZED."""
        follow_batch = ['edge_attr', 'hyperedge_index'] if self._use_hyperedge else ['edge_attr']
        prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        
        # === Use worker_init_fn for reproducible seeding ===
        worker_init_fn_to_use = worker_init_fn if self.num_workers > 0 else None
        
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            follow_batch=follow_batch,  # ← Conditional!
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            drop_last=False,
            shuffle=False,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn_to_use,
        )


class SourceIndexSubset(Dataset):
    """Map-style subset that records the original source H5 event index."""

    def __init__(self, dataset, indices, split_name: str):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.split_name = str(split_name)

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, index):
        source_index = int(self.indices[int(index)])
        data = self.dataset[source_index]
        # Avoid keys containing "index" in PyG Data objects: Batch.__inc__
        # treats them as graph indices and offsets them by node counts.
        data.source_event_id = torch.tensor([source_index], dtype=torch.long)
        return data


class SourceIndexDataset(Dataset):
    """Prediction wrapper that records source H5 row index for full datasets."""

    def __init__(self, dataset, split_name: str = "all"):
        self.dataset = dataset
        self.split_name = str(split_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source_index = int(index)
        data = self.dataset[source_index]
        # Avoid keys containing "index" in PyG Data objects: Batch.__inc__
        # treats them as graph indices and offsets them by node counts.
        data.source_event_id = torch.tensor([source_index], dtype=torch.long)
        return data
