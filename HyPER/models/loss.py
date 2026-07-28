"""Stable typed reconstruction and event-classification objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter


def combine_reconstruction_activity(edge_active, hyperedge_active):
    """Canonical edge-or-hyperedge event activity operation."""
    if edge_active is None:
        return hyperedge_active
    if hyperedge_active is None:
        return edge_active
    return edge_active | hyperedge_active


def reconstruction_active_from_typed_targets(edge_target: Tensor | None, hyperedge_target: Tensor | None) -> bool:
    """Canonical event-level activity definition shared by data and tuning."""
    def active(target):
        return target is not None and target.numel() and target.ndim == 2 and bool(
            (target.argmax(dim=1) != target.size(1) - 1).any()
        )
    return bool(combine_reconstruction_activity(active(edge_target), active(hyperedge_target)))


def resolve_class_weights(
    class_names: Sequence[str], configured: Mapping[str, float] | None
) -> Tensor:
    """Resolve fixed semantic class weights in dataset class order."""
    names = [str(name) for name in class_names]
    if not names or names[-1] != "background" or len(set(names)) != len(names):
        raise ValueError(
            "Typed reconstruction class names must be unique and end with 'background'."
        )
    if configured is None:
        configured = {name: 1.0 for name in names}
    configured = dict(configured)
    missing = [name for name in names if name not in configured]
    unknown = [name for name in configured if name not in names]
    if missing or unknown:
        raise ValueError(
            f"Invalid reconstruction class weights: missing={missing}, unknown={unknown}, "
            f"expected={names}."
        )
    values = torch.tensor([float(configured[name]) for name in names], dtype=torch.float32)
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Reconstruction class weights must all be finite and strictly positive.")
    return values


def validate_typed_targets(target: Tensor, num_classes: int) -> Tensor:
    """Validate exact one-hot typed rows and return integer class indices."""
    if target.ndim != 2 or target.size(1) != int(num_classes):
        raise ValueError(
            f"Typed target shape must be [candidates, {num_classes}], got {tuple(target.shape)}."
        )
    if not torch.isfinite(target).all():
        raise ValueError("Typed targets contain NaN or infinity.")
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("Typed target rows must contain exact zeros and ones.")
    if not torch.all(target.sum(dim=1) == 1):
        raise ValueError("Every typed target row must sum exactly to one.")
    return target.argmax(dim=1).to(torch.long)


def validate_typed_target_classes(target_class: Tensor, num_classes: int) -> Tensor:
    """Validate cached integer typed targets without decoding one-hot rows."""
    if target_class.ndim != 1:
        raise ValueError(
            f"Cached typed target classes must be one-dimensional, got {tuple(target_class.shape)}."
        )
    if target_class.dtype == torch.bool or target_class.dtype.is_floating_point:
        raise TypeError(
            f"Cached typed target classes must use an integer dtype, got {target_class.dtype}."
        )
    classes = target_class.to(dtype=torch.long)
    if classes.numel() and (classes.min() < 0 or classes.max() >= int(num_classes)):
        raise ValueError(
            f"Cached typed target class lies outside [0, {int(num_classes)})."
        )
    return classes


def reconstruction_activity_mask(
    target_class: Tensor,
    candidate_batch: Tensor,
    num_events: int,
    background_class: int,
) -> Tensor:
    """Return events with at least one non-background typed target.

    This is the single reconstruction-activity definition.  In particular it
    is deliberately independent of event classification labels and matching
    category diagnostics.
    """
    return scatter(
        (target_class != background_class).to(torch.long),
        candidate_batch,
        dim=0,
        dim_size=num_events,
        reduce="sum",
    ) > 0


def typed_reconstruction_loss(
    logits: Tensor,
    target: Tensor | None,
    candidate_batch: Tensor,
    class_names: Sequence[str],
    class_weights: Mapping[str, float] | Tensor | None,
    *,
    num_events: int,
    target_class: Tensor | None = None,
    active_event_mask: Tensor | None = None,
    validate_cached_targets: bool = True,
    validate_candidate_batch: bool = True,
    validate_class_weights: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return component mean, active-event mask, and per-event typed CE.

    Production graph caches store integer class targets and event-level activity
    masks. Passing those fields avoids validating and decoding the equivalent
    one-hot targets on every batch. The one-hot input remains supported for
    explicit equivalence tests and raw graph construction paths.
    """
    if logits is None or logits.ndim != 2:
        raise ValueError("Typed reconstruction logits must be a 2D tensor.")
    names = [str(name) for name in class_names]
    num_classes = len(names)
    if logits.size(1) != num_classes:
        raise ValueError(
            f"Logit channels {logits.size(1)} do not match class names {names}."
        )

    if target_class is None:
        if target is None:
            raise ValueError("Typed reconstruction requires one-hot or cached integer targets.")
        classes = validate_typed_targets(target, num_classes)
    else:
        cached = target_class.reshape(-1)
        classes = (
            validate_typed_target_classes(cached, num_classes)
            if validate_cached_targets
            else cached.to(dtype=torch.long)
        )
        if classes.numel() != logits.size(0):
            raise ValueError("Cached typed target length does not match reconstruction candidates.")

    batch = candidate_batch.reshape(-1).to(device=logits.device, dtype=torch.long)
    if batch.numel() != logits.size(0):
        raise ValueError("Candidate batch length does not match reconstruction candidates.")
    if (
        validate_candidate_batch
        and batch.numel()
        and (batch.min() < 0 or batch.max() >= int(num_events))
    ):
        raise ValueError("Candidate batch contains an invalid event index.")

    if isinstance(class_weights, Tensor):
        if class_weights.numel() != num_classes:
            raise ValueError("Fixed reconstruction weight tensor has the wrong length.")
        if class_weights.device != logits.device or class_weights.dtype != torch.float32:
            raise RuntimeError(
                "Registered reconstruction class weights must already be float32 on the logits device."
            )
        weights = class_weights
        if validate_class_weights and (
            not torch.isfinite(weights).all() or (weights <= 0).any()
        ):
            raise ValueError("Reconstruction class weights must be finite and positive.")
    else:
        weights = resolve_class_weights(names, class_weights).to(
            device=logits.device, dtype=torch.float32
        )

    if classes.device != logits.device or classes.dtype != torch.long:
        raise RuntimeError(
            "Cached reconstruction target classes must already be int64 on the logits device."
        )
    candidate_loss = F.cross_entropy(logits.float(), classes, reduction="none")

    if active_event_mask is None:
        active = reconstruction_activity_mask(
            classes,
            batch,
            int(num_events),
            num_classes - 1,
        )
    else:
        active = active_event_mask.reshape(-1)
        if active.device != logits.device or active.dtype != torch.bool:
            raise RuntimeError(
                "Cached reconstruction activity must already be boolean on the logits device."
            )
        if active.numel() != int(num_events):
            raise ValueError(
                f"Cached reconstruction activity has {active.numel()} rows; expected {num_events}."
            )

    flat_group = batch * num_classes + classes
    group_size = int(num_events) * num_classes
    sums = scatter(candidate_loss, flat_group, dim=0, dim_size=group_size, reduce="sum")
    counts = scatter(
        torch.ones_like(candidate_loss), flat_group, dim=0, dim_size=group_size, reduce="sum"
    )
    sums = sums.reshape(int(num_events), num_classes)
    counts = counts.reshape(int(num_events), num_classes)
    present = counts > 0
    class_means = sums / counts.clamp_min(1.0)
    present_weights = present.to(weights.dtype) * weights.unsqueeze(0)
    per_event = (class_means * present_weights).sum(dim=1) / present_weights.sum(dim=1).clamp_min(1.0)

    if active.any():
        selected = per_event[active]
        if not torch.isfinite(selected).all():
            raise FloatingPointError("Active typed reconstruction loss contains NaN or infinity.")
        component = selected.mean()
    else:
        component = logits.sum() * 0.0
    return component, active, per_event


def classification_losses(
    logits: Tensor,
    target: Tensor,
    pos_weight: float | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return weighted BCE, unweighted BCE, signal count, and background count."""
    if logits is None or target is None:
        raise ValueError("Classification requires logits and explicit event labels.")
    logits = logits.float().reshape(-1)
    target = target.float().reshape(-1).to(logits.device)
    if logits.numel() != target.numel() or logits.numel() == 0:
        raise ValueError("Classification logits and labels must have the same non-zero length.")
    if not torch.isfinite(target).all() or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Classification labels must be finite and binary.")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Classification logits contain NaN or infinity.")
    unweighted = F.binary_cross_entropy_with_logits(logits, target)
    if pos_weight is None:
        weighted = unweighted
    else:
        value = float(pos_weight)
        if not torch.isfinite(torch.tensor(value)) or value <= 0:
            raise ValueError("loss.classification_pos_weight must be finite and positive.")
        weighted = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=logits.new_tensor(value)
        )
    if not torch.isfinite(weighted):
        raise FloatingPointError("Classification loss contains NaN or infinity.")
    signal = (target == 1).sum()
    background = (target == 0).sum()
    return weighted, unweighted, signal, background


def additive_total_loss(
    *,
    edge_loss: Tensor | None,
    hyperedge_loss: Tensor | None,
    classification_loss: Tensor | None,
    edge_weight: float,
    hyperedge_weight: float,
    classification_weight: float,
) -> Tensor:
    """Combine enabled task losses with fixed additive coefficients."""
    terms = []
    for name, loss, coefficient in (
        ("edge", edge_loss, edge_weight),
        ("hyperedge", hyperedge_loss, hyperedge_weight),
        ("classification", classification_loss, classification_weight),
    ):
        coefficient = float(coefficient)
        if not torch.isfinite(torch.tensor(coefficient)) or coefficient < 0:
            raise ValueError(f"loss.{name}_weight must be finite and non-negative.")
        if coefficient == 0:
            continue
        if loss is None:
            raise ValueError(f"loss.{name}_weight is non-zero but the {name} task loss is disabled.")
        if loss.numel() != 1 or not torch.isfinite(loss).all():
            raise FloatingPointError(f"Active {name} loss must be one finite scalar.")
        terms.append(coefficient * loss)
    if not terms:
        raise ValueError("At least one additive loss coefficient must be non-zero.")
    return torch.stack(terms).sum()
