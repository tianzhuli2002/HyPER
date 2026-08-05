"""Numerical and source-event alignment operations for HyPER representations."""

from __future__ import annotations

import hashlib
import math

import numpy as np


# The first term is a relative scientific threshold; the second protects the
# exact numerical calculation from declaring round-off noise to be structure.
# Exact repeated rows are always treated as zero variance independently of the
# threshold.  The value is recorded in every representation-health/CKA
# product so that the decision is inspectable.
VARIANCE_RELATIVE_TOLERANCE = 1e-12
VARIANCE_ROUNDOFF_FACTOR = 64.0


def event_index_sha256(indices) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _exactly_constant(values: np.ndarray) -> bool:
    """Return whether every finite row is bitwise equal to the first row."""
    if values.shape[0] == 0:
        return True
    first = values[0]
    for start in range(0, values.shape[0], 8192):
        if not np.all(values[start:start + 8192] == first):
            return False
    return True


def _variance_threshold(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    scale = max(1.0, scale)
    roundoff = (
        np.finfo(np.float64).eps
        * max(values.shape)
        * VARIANCE_ROUNDOFF_FACTOR
        * scale
    )
    return max(VARIANCE_RELATIVE_TOLERANCE * scale, roundoff)


def representation_diagnostics(values, *, unique_row_limit: int = 5000) -> dict[str, object]:
    """Calculate finite, variance, diversity and rank diagnostics for an export."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Representation diagnostics require a 2D array, got {values.shape}.")
    event_count, feature_dimension = map(int, values.shape)
    finite = np.isfinite(values)
    finite_fraction = float(np.mean(finite)) if values.size else 1.0
    result: dict[str, object] = {
        "event_count": event_count,
        "feature_dimension": feature_dimension,
        "finite_fraction": finite_fraction,
        "centred_frobenius_norm": None,
        "active_dimension_count": None,
        "minimum_feature_standard_deviation": None,
        "maximum_feature_standard_deviation": None,
        "unique_row_count": None,
        "numerical_rank": None,
        "effective_rank": None,
        "variance_threshold": None,
        "zero_variance": False,
        "zero_variance_detection": "not_evaluated",
    }
    if event_count == 0 or feature_dimension == 0 or not finite.all():
        return result

    mean = values.mean(axis=0)
    centered = values - mean
    norm = float(np.linalg.norm(centered))
    threshold = _variance_threshold(values)
    exact_constant = _exactly_constant(values)
    # A row-wise norm threshold is derived from the per-feature threshold.
    norm_threshold = threshold * math.sqrt(event_count * feature_dimension)
    zero_variance = exact_constant or norm <= norm_threshold
    if zero_variance:
        norm = 0.0

    standard_deviation = values.std(axis=0)
    active = int(np.count_nonzero(standard_deviation > threshold))
    if zero_variance:
        active = 0
        standard_deviation = np.zeros(feature_dimension, dtype=np.float64)

    result.update(
        {
            "centred_frobenius_norm": norm,
            "active_dimension_count": active,
            "minimum_feature_standard_deviation": float(np.min(standard_deviation)),
            "maximum_feature_standard_deviation": float(np.max(standard_deviation)),
            "variance_threshold": float(threshold),
            "zero_variance": bool(zero_variance),
            "zero_variance_detection": "exact_repeated_rows"
            if exact_constant
            else ("relative_threshold" if zero_variance else "varying"),
        }
    )
    if event_count <= unique_row_limit:
        result["unique_row_count"] = int(np.unique(values, axis=0).shape[0])

    if zero_variance:
        result["numerical_rank"] = 0
        result["effective_rank"] = 0.0
        return result

    # Exact numerical rank of a large 896-wide event representation is an
    # expensive eigendecomposition and adds no protection to the CKA result.
    # Use the feature-variance participation rank for wide arrays and reserve
    # the exact covariance rank for compact intermediate blocks.
    if feature_dimension <= 256:
        covariance = centered.T @ centered
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        largest = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
        rank_threshold = (
            max(1.0, largest)
            * np.finfo(np.float64).eps
            * max(event_count, feature_dimension)
            * VARIANCE_ROUNDOFF_FACTOR
        )
        positive = eigenvalues[eigenvalues > rank_threshold]
        result["numerical_rank"] = int(len(positive))
        power = positive
    else:
        result["numerical_rank"] = None
        power = standard_deviation ** 2
    if np.any(power > 0):
        probabilities = power[power > 0] / power[power > 0].sum()
        result["effective_rank"] = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        result["effective_rank"] = 0.0
    return result


def cka_undefined_reason(left_diagnostics: dict[str, object], right_diagnostics: dict[str, object]) -> str | None:
    """Return the precise supported reason for an undefined matrix cell."""
    left_zero = bool(left_diagnostics.get("zero_variance", False))
    right_zero = bool(right_diagnostics.get("zero_variance", False))
    if left_zero and right_zero:
        return "both_zero_variance"
    if left_zero:
        return "left_zero_variance"
    if right_zero:
        return "right_zero_variance"
    return None


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
    x_exact_constant = _exactly_constant(x)
    y_exact_constant = _exactly_constant(y)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    x_norm_frobenius = float(np.linalg.norm(x))
    y_norm_frobenius = float(np.linalg.norm(y))
    threshold = max(
        VARIANCE_RELATIVE_TOLERANCE,
        np.finfo(np.float64).eps * max(x.shape[0], x.shape[1], y.shape[1]) * VARIANCE_ROUNDOFF_FACTOR,
    )
    if x_exact_constant or y_exact_constant or x_norm_frobenius <= threshold or y_norm_frobenius <= threshold:
        raise ValueError("CKA is undefined for a zero-variance representation.")
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
