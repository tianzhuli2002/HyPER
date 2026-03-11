"""
Picklable edge feature transforms.
Replaces lambda functions to enable multiprocessing with num_workers > 0.
"""
import torch


def _identity(x):
    """Identity transform (no-op)"""
    return x


def _log_clamped(x):
    """Log transform with clamping to avoid log(0)"""
    return torch.log(torch.clamp(x, min=1e-6))


def _log_positive(x):
    """Log transform assuming positive input"""
    return torch.log(x)


# Registry of all edge feature transforms (module-level = picklable ✓)
EDGE_FEATURE_TRANSFORMS = {
    "delta_eta": _identity,
    "delta_phi": _identity,
    "delta_R": _identity,
    "kT": _log_clamped,      # log(kT), kT can be small
    "Z": _log_clamped,       # log(Z), Z can be small
    "M2": _log_positive,     # log(M^2), mass squared should be positive
}
