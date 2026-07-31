"""Numerical and source-event alignment operations for HyPER representations."""

from __future__ import annotations

import hashlib

import numpy as np


def event_index_sha256(indices) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _unique_indices(export: dict, description: str) -> np.ndarray:
    if "source_event_index" not in export:
        raise KeyError(f"{description} has no source_event_index array.")
    indices = np.asarray(export["source_event_index"], dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError(f"{description} source_event_index must be unique and one-dimensional.")
    return indices


def align_exports(left: dict, right: dict, *, common_events_only: bool = False):
    """Align two exports by identity and validate evaluation truth fields."""
    left_indices = _unique_indices(left, "left export")
    right_indices = _unique_indices(right, "right export")
    if common_events_only:
        selected = np.intersect1d(left_indices, right_indices)
        if not selected.size:
            raise ValueError("Representation exports have no source events in common.")
    else:
        if len(left_indices) != len(right_indices) or set(left_indices) != set(right_indices):
            only_left = np.setdiff1d(left_indices, right_indices)
            only_right = np.setdiff1d(right_indices, left_indices)
            raise ValueError(
                "Representation source-event sets differ: "
                f"left_only={len(only_left)}, right_only={len(only_right)}."
            )
        selected = left_indices
    left_rows = {int(value): row for row, value in enumerate(left_indices)}
    right_rows = {int(value): row for row, value in enumerate(right_indices)}
    left_order = np.asarray([left_rows[int(value)] for value in selected], dtype=np.int64)
    right_order = np.asarray([right_rows[int(value)] for value in selected], dtype=np.int64)
    for name in ("truth_class", "truth_fully_matched"):
        if name in left and name in right:
            a = np.asarray(left[name])[left_order]
            b = np.asarray(right[name])[right_order]
            if not np.array_equal(a, b):
                raise ValueError(f"Aligned {name} values disagree between exports.")
    return selected.astype(np.int64), left_order, right_order


def linear_cka(x, y) -> float:
    """Linear centred kernel alignment without constructing an event kernel."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError(f"CKA requires 2D features with the same N>=2; got {x.shape}, {y.shape}.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("CKA inputs must be finite.")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    x_norm = np.linalg.norm(x.T @ x, ord="fro") ** 2
    y_norm = np.linalg.norm(y.T @ y, ord="fro") ** 2
    denominator = np.sqrt(x_norm * y_norm)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("CKA is undefined for a zero-variance representation.")
    result = float(cross / denominator)
    if result < -1e-12 or result > 1.0 + 1e-10:
        raise RuntimeError(f"Linear CKA lies outside its numerical range: {result}.")
    return min(1.0, max(0.0, result))


def fit_procrustes(source, target) -> dict[str, np.ndarray | float]:
    """Fit the paired label-free orthogonal map source -> target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise ValueError(f"Procrustes requires equal 2D shapes, got {source.shape}, {target.shape}.")
    if source.shape[0] < 2 or source.shape[1] < 1:
        raise ValueError("Procrustes requires at least two paired events and one feature.")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Procrustes inputs must be finite.")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    if np.linalg.norm(source_centered) == 0 or np.linalg.norm(target_centered) == 0:
        raise ValueError("Procrustes is undefined for a zero-variance representation.")
    u, singular_values, vt = np.linalg.svd(source_centered.T @ target_centered, full_matrices=False)
    rotation = u @ vt
    aligned = source_centered @ rotation + target_mean
    residual = np.linalg.norm(aligned - target) / np.linalg.norm(target_centered)
    cosine_denominator = np.linalg.norm(aligned, axis=1) * np.linalg.norm(target, axis=1)
    valid = cosine_denominator > 0
    mean_cosine = np.mean(np.sum(aligned[valid] * target[valid], axis=1) / cosine_denominator[valid])
    orthogonality = np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0]), ord="fro")
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "rotation": rotation,
        "singular_values": singular_values,
        "normalised_residual": float(residual),
        "mean_aligned_cosine_similarity": float(mean_cosine),
        "orthogonality_error": float(orthogonality),
    }


def apply_procrustes(source, source_mean, target_mean, rotation) -> np.ndarray:
    source = np.asarray(source)
    source_mean = np.asarray(source_mean)
    target_mean = np.asarray(target_mean)
    rotation = np.asarray(rotation)
    if source.ndim != 2 or source.shape[1] != rotation.shape[0] or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("Source and Procrustes rotation dimensions are incompatible.")
    return (source - source_mean) @ rotation + target_mean


def random_orthogonal(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    q, r = np.linalg.qr(rng.normal(size=(int(dimension), int(dimension))))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


class StreamingLinearCKA:
    """Accumulate sufficient statistics for centred linear CKA."""

    def __init__(self, left_dimension: int, right_dimension: int):
        self.left_dimension = int(left_dimension)
        self.right_dimension = int(right_dimension)
        if self.left_dimension <= 0 or self.right_dimension <= 0:
            raise ValueError("Representation dimensions must be positive.")
        self.count = 0
        self.sum_left = np.zeros(self.left_dimension, dtype=np.float64)
        self.sum_right = np.zeros(self.right_dimension, dtype=np.float64)
        self.left_gram = np.zeros((self.left_dimension, self.left_dimension), dtype=np.float64)
        self.right_gram = np.zeros((self.right_dimension, self.right_dimension), dtype=np.float64)
        self.cross = np.zeros((self.left_dimension, self.right_dimension), dtype=np.float64)

    def update(self, left, right) -> None:
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
            raise ValueError(f"Streaming CKA requires paired 2D batches; got {left.shape}, {right.shape}.")
        if left.shape[1] != self.left_dimension or right.shape[1] != self.right_dimension:
            raise ValueError("Streaming CKA batch dimensions differ from the accumulator.")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("Streaming CKA batches must be finite.")
        self.count += int(left.shape[0])
        self.sum_left += left.sum(axis=0)
        self.sum_right += right.sum(axis=0)
        self.left_gram += left.T @ left
        self.right_gram += right.T @ right
        self.cross += left.T @ right

    def centred_products(self):
        if self.count < 2:
            raise ValueError("Streaming CKA requires at least two events.")
        mean_left = self.sum_left / self.count
        mean_right = self.sum_right / self.count
        left = self.left_gram - self.count * np.outer(mean_left, mean_left)
        right = self.right_gram - self.count * np.outer(mean_right, mean_right)
        cross = self.cross - self.count * np.outer(mean_left, mean_right)
        return left, right, cross

    def value(self) -> float:
        left, right, cross = self.centred_products()
        numerator = np.linalg.norm(cross, ord="fro") ** 2
        denominator = np.sqrt(
            (np.linalg.norm(left, ord="fro") ** 2)
            * (np.linalg.norm(right, ord="fro") ** 2)
        )
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("CKA is undefined for a zero-variance representation.")
        result = float(numerator / denominator)
        if result < -1e-12 or result > 1.0 + 1e-10:
            raise RuntimeError(f"Streaming linear CKA lies outside its numerical range: {result}.")
        return min(1.0, max(0.0, result))

    def state_dict(self) -> dict[str, object]:
        return {
            "count": int(self.count),
            "left_dimension": self.left_dimension,
            "right_dimension": self.right_dimension,
            "sum_left": self.sum_left,
            "sum_right": self.sum_right,
            "left_gram": self.left_gram,
            "right_gram": self.right_gram,
            "cross": self.cross,
        }
