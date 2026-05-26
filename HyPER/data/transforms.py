"""
Picklable transform functions for node and global features.
Replaces lambda functions to enable multiprocessing with num_workers > 0.
"""
import torch


def _identity(x):
    return x


def _log(x):
    return torch.log(torch.clamp(x, min=1e-6))


def _log10(x):
    return torch.log10(torch.clamp(x, min=1e-6))


def _sqrt(x):
    return torch.sqrt(torch.clamp(x, min=0))


def _exp(x):
    return torch.exp(torch.clamp(x, max=50))


def _sigmoid(x):
    return torch.sigmoid(x)


def _normalize(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def _abs(x):
    return torch.abs(x)


def _negate(x):
    return -x


def _div6(x):
    return x / 6


def _div8(x):
    return x / 8


def _div4(x):
    return x / 4


# Registry of all picklable transforms
TRANSFORM_REGISTRY = {
    'identity': _identity,
    'x': _identity,
    'log': _log,
    'torch.log(x)': _log,
    'log10': _log10,
    'torch.log10(x)': _log10,
    'sqrt': _sqrt,
    'torch.sqrt(x)': _sqrt,
    'exp': _exp,
    'torch.exp(x)': _exp,
    'sigmoid': _sigmoid,
    'torch.sigmoid(x)': _sigmoid,
    'normalize': _normalize,
    'abs': _abs,
    'torch.abs(x)': _abs,
    'negate': _negate,
    '-x': _negate,
    'x/4': _div4,
    'x/6': _div6,
    'x/8': _div8,
}
