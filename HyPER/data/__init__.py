# First, import the dataset classes directly (not through datamodule)
from .dataset import HyPERDataset
from .dataset_lazy import HyPERDatasetLazy

# Then import EventSampler (wherever it's defined)
from .sampler import EventSampler  # ← Adjust path to where EventSampler lives

# Finally import datamodule (now safe, no circular dependency)
from .datamodule import HyPERDataModule


__all__ = [
    'HyPERDataset',
    'HyPERDatasetLazy',
    'EventSampler',
    'HyPERDataModule'
]
