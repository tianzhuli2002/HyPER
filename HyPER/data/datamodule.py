import warnings
import os
import torch
import numpy as np
from copy import deepcopy
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from .dataset import HyPERDataset
from .preprocessed import HyPEROnDiskDataset


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
        cache_dir: Optional[str] = None,
        force_reload: bool = False,
        use_ondisk: bool = True,
        graph_config: Optional[dict] = None,
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
        self.percent_valid_samples = percent_valid_samples
        self.cache_dir = cache_dir
        self.force_reload = force_reload
        self.use_ondisk = use_ondisk
        self.graph_config = deepcopy(graph_config) if graph_config is not None else None
        
        # Parse config for channel dimensions
        parsed_inputs = HyPERDataset._resolve_graph_config(root=self.root, config=self.graph_config)
        self.node_in_channels = len(parsed_inputs['input']['node_features']) + 1
        self.edge_in_channels = len(parsed_inputs['input']['edge_features'])
        self.global_in_channels = len(parsed_inputs['input']['global_features'])
        
        # Track if hyperedges are used (for conditional follow_batch)
        self._use_hyperedge = len(parsed_inputs['target']['hyperedge'].values()) > 0 if 'target' in parsed_inputs else False

        self.index_range = None
        self.train_data = None
        self.val_data = None
        self.predict_data = None
    
    def setup(self, stage: str):
        """Setting up datasets."""
        if self.train_set is not None:
            if self.val_set is None or self.val_set == "" or self.train_set == self.val_set:
                # Split training set into train/val
                print(f"Creating validation set using "
                      f"{round(self.percent_valid_samples * 100, 2)}% of the training file.")
                
                if self.use_ondisk:
                    full_data = HyPEROnDiskDataset(
                        root=self.root,
                        name=self.train_set,
                        training=True,
                        force_reload=self.force_reload,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        config=self.graph_config,
                    )
                else:
                    full_data = HyPERDataset(
                        root=self.root,
                        name=self.train_set,
                        force_reload=self.force_reload,
                        config=self.graph_config,
                    )
                
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
                if self.use_ondisk:
                    self.train_data = HyPEROnDiskDataset(
                        root=self.root,
                        name=self.train_set,
                        training=True,
                        force_reload=self.force_reload,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        config=self.graph_config,
                    )
                    self.val_data = HyPEROnDiskDataset(
                        root=self.root,
                        name=self.val_set,
                        training=True,
                        force_reload=self.force_reload,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        config=self.graph_config,
                    )
                else:
                    self.train_data = HyPERDataset(
                        root=self.root,
                        name=self.train_set,
                        force_reload=self.force_reload,
                        config=self.graph_config,
                    )
                    self.val_data = HyPERDataset(
                        root=self.root,
                        name=self.val_set,
                        training=True,
                        force_reload=self.force_reload,
                        config=self.graph_config,
                    )

            # Limit training dataset size if requested
            if self.max_n_events == -1:
                pass
            elif self.max_n_events >= len(self.train_data):
                warnings.warn(f"`max_n_events` ({self.max_n_events}) larger than dataset "
                              f"({len(self.train_data)}), using all events.")
            elif self.max_n_events > 0:
                self.index_range = list(range(self.max_n_events))
                print(f"Limited training to {self.max_n_events} events.")

        if self.predict_set is not None:
            if self.use_ondisk:
                self.predict_data = HyPEROnDiskDataset(
                    root=self.root,
                    name=self.predict_set,
                    training=False,
                    force_reload=self.force_reload,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    config=self.graph_config,
                )
            else:
                self.predict_data = HyPERDataset(
                    root=self.root,
                    name=self.predict_set,
                    force_reload=self.force_reload,
                    config=self.graph_config,
                )

        if self.train_data is None and self.val_data is None and self.predict_data is None:
            raise ValueError("No datasets have been provided. Abort!")

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
            table.add_row("Pin memory", str(self.pin_memory))
            
            if self.train_data is not None:
                table.add_row("Training set", str(self.train_set))
                table.add_row("Training samples", str(len(self.index_range) if self.index_range else len(self.train_data)))
            if self.val_data is not None:
                table.add_row("Validation set", str(self.val_set) if self.val_set else "split from training set")
                table.add_row("Validation samples", str(len(self.val_data)))
            if self.predict_data is not None:
                table.add_row("Prediction set", str(self.predict_set))
                table.add_row("Prediction samples", str(len(self.predict_data)))
            
            table.add_row("N node attributes", str(self.node_in_channels))
            table.add_row("N edge attributes", str(self.edge_in_channels))
            table.add_row("N glob attributes", str(self.global_in_channels))
            console.print(table)

        except ImportError:
            print(f"Dataset Status:")
            print(f"  Training samples: {len(self.train_data) if self.train_data else 'N/A'}")
            print(f"  Validation samples: {len(self.val_data) if self.val_data else 'N/A'}")
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
        prefetch_factor = 2 if self.num_workers > 0 else None
        
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
        prefetch_factor = 2 if self.num_workers > 0 else None
        
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

    def predict_dataloader(self):
        """Returns prediction DataLoader - OPTIMIZED."""
        follow_batch = ['edge_attr', 'hyperedge_index'] if self._use_hyperedge else ['edge_attr']
        prefetch_factor = 2 if self.num_workers > 0 else None
        
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
