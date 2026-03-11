# HyPER/data/datamodule.py
import yaml
import warnings
import os.path as osp
import torch
from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# ← CHANGE: Import directly from file, not from HyPER.data
from .dataset_lazy import HyPERDatasetLazy
from .sampler import EventSampler  # ← Adjust path to where EventSampler is


class HyPERDataModule(LightningDataModule):
    r"""HyPER Data Module encapsulates all the steps needed to
    process data with lazy loading and optimized data loading.

    Args:
        root (str): dataset path.
        train_set (optional, str): training dataset name.
            (default :obj:`None`)
        val_set (optional, str): validation dataset name.
            (default :obj:`None`)
        predict_set (optional, str): predict dataset name. 
            (default :obj:`None`)
        batch_size (optional, int): number of samples per batch to 
            load. (default :obj:`128`)
        max_n_events (optional, int): maximum number of events used 
            in training. (default :obj:`-1`)
        percent_valid_samples (optional, float): fraction of dataset 
            to use as validation samples. (default :obj:`0.05`)
        drop_last (optional, bool): drop the last incomplete batch.
            (default :obj:`True` for production, `False` for debugging)
        num_workers (optional, int): number of CPU workers for data loading.
            (default :obj:`8` for GPU training)
        pin_memory (optional, bool): use memory pinning for faster CPU→GPU transfer.
            (default :obj:`True`)
        persistent_workers (optional, bool): keep workers alive between epochs.
            (default :obj:`True`)
        cache_dir (optional, str): directory for caching processed events.
            If None, uses default location in dataset root.
            (default :obj:`None`)
        force_reload (optional, bool): force rebuild of cache.
            (default :obj:`False`)
    """
    
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
    ):
        super().__init__()
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
        
        # Parse config for channel dimensions
        parsed_inputs = HyPERDatasetLazy._parse_config_file(f"{self.root}/config.yaml")
        self.node_in_channels = len(parsed_inputs['input']['node_features']) + 1
        self.edge_in_channels = len(parsed_inputs['input']['edge_features'])
        self.global_in_channels = len(parsed_inputs['input']['global_features'])

        self.index_range = None
        self.train_data = None
        self.val_data = None
        self.predict_data = None
    
    @staticmethod
    def parse_config_file(filename):
        with open(filename) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                raise

    def setup(self, stage: str):
        """
        Setting up datasets. Called by Lightning before training/validation/prediction.
        """
        # ============ Training & Validation ============
        if self.train_set is not None:
            if self.val_set is None or self.val_set == "" or self.train_set == self.val_set:
                # Split training set into train/val
                print(f"Creating validation set using "
                      f"{round(self.percent_valid_samples * 100, 2)}% of the training file.")
                
                full_data = HyPERDatasetLazy(
                    root=self.root,
                    name=self.train_set,
                    cache_dir=osp.join(self.cache_dir, self.train_set) if self.cache_dir else None,
                    force_reload=self.force_reload,
                )
                
                # Calculate split sizes
                n_total = len(full_data)
                n_val = max(1, int(n_total * self.percent_valid_samples))
                n_train = n_total - n_val
                
                self.train_data, self.val_data = random_split(
                    full_data,
                    [n_train, n_val],
                    generator=torch.Generator().manual_seed(42)  # Reproducible splits
                )
            else:
                # Separate train and val files
                self.train_data = HyPERDatasetLazy(
                    root=self.root,
                    name=self.train_set,
                    cache_dir=osp.join(self.cache_dir, self.train_set) if self.cache_dir else None,
                    force_reload=self.force_reload,
                )
                self.val_data = HyPERDatasetLazy(
                    root=self.root,
                    name=self.val_set,
                    cache_dir=osp.join(self.cache_dir, self.val_set) if self.cache_dir else None,
                    force_reload=self.force_reload,
                )

            # Limit training dataset size if requested
            if self.max_n_events == -1:
                pass  # Use all events
            elif self.max_n_events >= len(self.train_data):
                warnings.warn(f"`max_n_events` ({self.max_n_events}) larger than the dataset "
                              f"({len(self.train_data)}), using all events.")
            elif self.max_n_events > 0:
                self.index_range = list(range(self.max_n_events))
                print(f"Limited training to {self.max_n_events} events.")

        # ============ Prediction ============
        if self.predict_set is not None:
            self.predict_data = HyPERDatasetLazy(
                root=self.root,
                name=self.predict_set,
                cache_dir=osp.join(self.cache_dir, self.predict_set) if self.cache_dir else None,
                force_reload=self.force_reload,
            )

        # ============ Validation ============
        if self.train_data is None and self.val_data is None and self.predict_data is None:
            raise ValueError("No datasets have been provided. Abort!")

        # ============ Print Dataset Summary ============
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
            table.add_row("Pin memory", str(self.pin_memory))
            
            if self.cache_dir:
                table.add_row("Cache directory", self.cache_dir)
            
            if self.train_data is not None:
                if self.index_range is not None:
                    table.add_row("Training samples", str(len(self.index_range)))
                else:
                    table.add_row("Training samples", str(len(self.train_data)))
            
            if self.val_data is not None:
                table.add_row("Validation samples", str(len(self.val_data)))
            
            if self.predict_data is not None:
                table.add_row("Prediction samples", str(len(self.predict_data)))
            
            table.add_row("N node attributes", str(self.node_in_channels))
            table.add_row("N edge attributes", str(self.edge_in_channels))
            table.add_row("N glob attributes", str(self.global_in_channels))
            console.print(table)

        except ImportError:
            # Fallback if rich is not installed
            print(f"Dataset Status:")
            print(f"  Training samples: {len(self.train_data) if self.train_data else 'N/A'}")
            print(f"  Validation samples: {len(self.val_data) if self.val_data else 'N/A'}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Num workers: {self.num_workers}")

    def train_dataloader(self):
        """Returns the training DataLoader with optimized settings."""
        sampler = EventSampler(self.index_range) if self.index_range is not None else None
        shuffle = False if sampler is not None else True
        pf = 2 if self.num_workers > 0 else 1

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            follow_batch=['edge_attr', 'hyperedge_index'],
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            shuffle=shuffle,
            sampler=sampler,
            prefetch_factor=pf,
        )

    def val_dataloader(self):
        """Returns the validation DataLoader with optimized settings."""
        pf = 2 if self.num_workers > 0 else 1
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            follow_batch=['edge_attr', 'hyperedge_index'],
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            shuffle=False,  # ← IMPORTANT: No shuffle for validation
            prefetch_factor=pf,
        )

    def predict_dataloader(self):
        """Returns the prediction DataLoader with optimized settings."""
        pf = 2 if self.num_workers > 0 else 1
        return DataLoader(
            self.predict_data,
            batch_size=self.batch_size,
            follow_batch=['edge_attr', 'hyperedge_index'],
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            drop_last=False,  # Keep all samples for prediction
            shuffle=False,  # No shuffle for prediction
            prefetch_factor=pf,
        )