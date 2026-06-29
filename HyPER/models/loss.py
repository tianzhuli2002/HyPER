import torch

from contextlib import nullcontext
from torch import Tensor
from torch.nn import BCELoss
import torch.nn.functional as F
from torch_scatter import scatter

from typing import Optional


def _autocast_disabled_like(tensor: Tensor):
    """Disable AMP autocast around legacy BCE-on-probabilities helpers."""
    device_type = tensor.device.type
    if device_type in {"cuda", "cpu", "xpu", "mps"}:
        return torch.amp.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _safe_bce(
    pred: Tensor,
    target: Tensor,
    criterion: Optional[callable] = None,
) -> Tensor:
    """Compute BCE in float32 with autocast disabled."""
    if criterion is None:
        criterion = BCELoss(reduction="none")

    with _autocast_disabled_like(pred):
        return criterion(pred.float(), target.float())


def _safe_bce_with_logits(pred: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(pred.float(), target.float(), reduction="none")


def _align_mask_to_loss(loss: Tensor, mask: Tensor | None) -> Tensor | None:
    """Pad/truncate a graph mask so it matches the per-graph loss length."""
    if mask is None:
        return None

    loss = loss.flatten()
    mask = mask.flatten().bool()

    if mask.numel() == loss.numel():
        return mask

    if mask.numel() < loss.numel():
        pad = torch.zeros(
            loss.numel() - mask.numel(),
            dtype=torch.bool,
            device=loss.device,
        )
        return torch.cat([mask.to(loss.device), pad], dim=0)

    return mask[: loss.numel()].to(loss.device)


def _masked_mean(loss: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Return masked mean and a boolean indicating whether anything contributed."""
    loss = loss.flatten()

    if loss.numel() == 0:
        return loss.new_tensor(0.0), torch.tensor(False, device=loss.device)

    if mask is None:
        finite = torch.isfinite(loss)
        if not finite.any():
            return loss.new_tensor(0.0), torch.tensor(False, device=loss.device)
        return loss[finite].mean(), torch.tensor(True, device=loss.device)

    mask = _align_mask_to_loss(loss, mask)
    assert mask is not None

    keep = mask & torch.isfinite(loss)
    if not keep.any():
        return loss.new_tensor(0.0), torch.tensor(False, device=loss.device)

    return loss[keep].mean(), torch.tensor(True, device=loss.device)


def _masked_mean_no_sync(loss: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
    """Return masked mean and an active flag without CPU/GPU sync.

    Semantics:
      - inactive entries do not contribute;
      - inactive NaN/Inf entries are neutralised so they cannot poison the loss;
      - active NaN/Inf entries deliberately propagate NaN instead of being
        silently dropped and converted into a zero loss;
      - the active flag is computed from the unclamped active count.
    """
    loss = loss.flatten()

    if loss.numel() == 0:
        zero = loss.new_tensor(0.0)
        return zero, zero

    if mask is None:
        active_mask = torch.ones_like(loss, dtype=torch.bool, device=loss.device)
    else:
        aligned = _align_mask_to_loss(loss, mask)
        assert aligned is not None
        active_mask = aligned.flatten().bool().to(loss.device)

    active_f = active_mask.to(loss.dtype)
    active_count = active_f.sum()
    denom = active_count.clamp_min(1.0)
    active = (active_count > 0).to(loss.dtype)

    finite = torch.isfinite(loss)
    active_bad = active_mask & (~finite)

    # Neutralise inactive non-finite entries. Do not use torch.where directly
    # on the original loss, because inactive NaN branches can still be awkward
    # in autograd. Active non-finite entries are handled below and deliberately
    # make the returned value NaN.
    safe_loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    value = (safe_loss * active_f).sum() / denom

    # If any active entry was non-finite, propagate NaN. This avoids the
    # previous failure mode where all non-finite active entries could be filtered
    # away and the reconstruction loss became exactly zero.
    has_active_bad = active_bad.to(loss.dtype).sum() > 0
    value = torch.where(
        has_active_bad,
        loss.new_tensor(float("nan")),
        value,
    )

    # If there are no active entries, return a differentiable zero. This should
    # be a rare/no-target batch case, not a way to hide active non-finite losses.
    zero = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    value = torch.where(active_count > 0, value, zero)

    return value, active


def _normalise_weighting_name(weighting: str | None) -> str:
    if weighting is None:
        return "legacy"
    name = str(weighting).strip().lower()
    if name == "":
        return "legacy"
    return name


def _typed_class_weight(
    target_class: Tensor,
    num_classes: int,
    weighting: str | None,
    cap: float | None,
) -> Tensor | None:
    name = _normalise_weighting_name(weighting)
    if name in {"none", "unweighted", "no_class_weight", "no_class_weights"}:
        return None
    if name not in {"legacy", "batch_class", "batch_class_weight", "event_balanced_bce"}:
        raise ValueError(f"Unsupported reconstruction weighting: {weighting!r}")

    counts = torch.bincount(target_class, minlength=num_classes).float()
    total = counts.sum()
    weights = torch.where(
        counts > 0,
        total / (float(num_classes) * counts.clamp(min=1.0)),
        torch.zeros_like(counts),
    )
    if cap is not None and float(cap) > 0:
        weights = weights.clamp(max=float(cap))
    return weights.to(target_class.device)


def _typed_reconstruction_loss(
    logits: Tensor,
    target: Tensor,
    batch: Tensor,
    reduction: str,
    weighting: str | None,
    class_weight_cap: float | None,
    target_class: Tensor | None = None,
    active_events: Tensor | None = None,
    validate_cached_targets: bool = False,
    dim_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    if (
        logits is None
        or logits.numel() == 0
        or target is None
        or target.numel() == 0
        or batch is None
        or batch.numel() == 0
    ):
        if logits is not None:
            device = logits.device
        elif target is not None:
            device = target.device
        else:
            device = torch.device("cpu")
        return torch.zeros(1, device=device), torch.zeros(1, dtype=torch.bool, device=device)

    if target.ndim != 2 or logits.ndim != 2:
        raise ValueError(
            f"Typed reconstruction expects 2D logits/targets, got logits={tuple(logits.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if logits.size(1) != target.size(1):
        raise ValueError(
            f"Typed reconstruction channel mismatch: logits={tuple(logits.shape)}, target={tuple(target.shape)}"
        )

    target = target.float()
    computed_target_class = None
    if target_class is None:
        target_class = target.argmax(dim=1).to(torch.long)
    else:
        target_class = target_class.flatten().to(device=target.device, dtype=torch.long)
        if target_class.numel() != target.size(0):
            raise ValueError(
                "Cached typed target class length mismatch: "
                f"target_class={tuple(target_class.shape)}, target={tuple(target.shape)}"
            )
        if validate_cached_targets:
            computed_target_class = target.argmax(dim=1).to(torch.long)
            if not torch.equal(target_class, computed_target_class):
                raise ValueError("Cached typed target classes do not match target.argmax(dim=1).")
    background_class = target.size(1) - 1
    batch_flat = batch.flatten().to(torch.long)
    if active_events is not None:
        active_events = active_events.flatten().to(device=target.device, dtype=torch.bool)
        dim_size = int(active_events.numel())
    elif dim_size is not None:
        dim_size = int(dim_size)
    else:
        dim_size = int(batch_flat.max().item()) + 1

    class_weight = _typed_class_weight(
        target_class,
        num_classes=target.size(1),
        weighting=weighting,
        cap=class_weight_cap,
    )
    loss = F.cross_entropy(
        logits.float(),
        target_class,
        reduction="none",
        weight=class_weight,
    )
    per_event_loss = scatter(loss, batch_flat, dim=0, dim_size=dim_size, reduce=reduction)
    if active_events is None:
        active_events = scatter(
            (target_class != background_class).float(),
            batch_flat,
            dim=0,
            dim_size=dim_size,
            reduce="sum",
        ) > 0
    return per_event_loss, active_events


def _event_balanced_bce_per_event(
    pred: Tensor,
    target: Tensor,
    batch: Tensor,
    criterion: Optional[callable],
    positive_weight_cap: float | None,
    negative_weight_cap: float | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return per-event BCE where positives and negatives are balanced per event.

    Events with no positive reconstruction targets are marked inactive. They do
    not contribute reconstruction loss, but they can still contribute to the
    event-level S/B classification loss elsewhere.
    """
    target = target.float().view_as(pred)
    loss = _safe_bce_with_logits(pred, target).flatten()
    target_flat = target.flatten()
    batch_flat = batch.flatten().to(torch.long)

    if batch_flat.numel() == 0:
        empty = pred.new_zeros(1)
        return empty, torch.zeros(1, dtype=torch.bool, device=pred.device), empty, empty

    dim_size = int(batch_flat.max().item()) + 1
    positive_mask = target_flat > 0.5
    negative_mask = ~positive_mask
    pos_counts = scatter(positive_mask.float(), batch_flat, dim=0, dim_size=dim_size, reduce="sum")
    neg_counts = scatter(negative_mask.float(), batch_flat, dim=0, dim_size=dim_size, reduce="sum")
    total_counts = pos_counts + neg_counts
    active_events = pos_counts > 0

    eps = loss.new_tensor(1e-8)
    pos_weights = torch.where(
        pos_counts > 0,
        total_counts / (2.0 * pos_counts + eps),
        torch.zeros_like(pos_counts),
    )
    neg_weights = torch.where(
        neg_counts > 0,
        total_counts / (2.0 * neg_counts + eps),
        torch.zeros_like(neg_counts),
    )

    if positive_weight_cap is not None and float(positive_weight_cap) > 0:
        pos_weights = pos_weights.clamp(max=float(positive_weight_cap))
    if negative_weight_cap is not None and float(negative_weight_cap) > 0:
        neg_weights = neg_weights.clamp(max=float(negative_weight_cap))

    candidate_weights = torch.where(
        positive_mask,
        pos_weights[batch_flat],
        neg_weights[batch_flat],
    )
    weighted_loss_sum = scatter(loss * candidate_weights, batch_flat, dim=0, dim_size=dim_size, reduce="sum")
    weight_sum = scatter(candidate_weights, batch_flat, dim=0, dim_size=dim_size, reduce="sum")
    per_event_loss = weighted_loss_sum / (weight_sum + eps)

    return per_event_loss, active_events, pos_counts, neg_counts


def ClassificationLoss(
    cls_out: Tensor,
    cls_t: Tensor,
    criterion: Optional[callable] = None,
) -> Tensor:
    """Per-event S/B classification loss from raw logits.

    This is applied to all events that have class targets.
    """
    if cls_t is None or cls_t.numel() == 0:
        return cls_out.new_zeros(0)

    if cls_out is None or cls_out.numel() == 0:
        return cls_t.new_zeros(0, dtype=torch.float32)

    cls_t = cls_t.float().view(-1, 1)
    cls_out = cls_out.view_as(cls_t)

    return F.binary_cross_entropy_with_logits(
        cls_out.float(),
        cls_t.float(),
        reduction="none",
    ).flatten()


def EdgeLoss(
    edge_attr_out: Tensor,
    edge_attr_t: Tensor,
    edge_attr_batch: Tensor,
    criterion: Optional[callable] = BCELoss(reduction="none"),
    reduction: Optional[str] = "mean",
    weighting: str | None = "legacy",
    positive_weight_cap: float | None = None,
    negative_weight_cap: float | None = None,
    target_encoding: str = "binary",
    target_class: Tensor | None = None,
    active_events: Tensor | None = None,
    validate_cached_targets: bool = False,
    dim_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Calculate per-graph edge loss.

    The returned mask is True for events with at least one positive edge target.
    This means reconstruction loss is applied to any event with reconstructable
    edge truth, regardless of whether its S/B label is signal or background.
    """
    if (
        edge_attr_out is None
        or edge_attr_out.numel() == 0
        or edge_attr_t is None
        or edge_attr_t.numel() == 0
        or edge_attr_batch is None
        or edge_attr_batch.numel() == 0
    ):
        return (
            edge_attr_out.new_zeros(1),
            torch.zeros(1, dtype=torch.bool, device=edge_attr_out.device),
        )

    batch_flat = edge_attr_batch.flatten().to(torch.long)
    if str(target_encoding).strip().lower() == "typed":
        return _typed_reconstruction_loss(
            edge_attr_out,
            edge_attr_t,
            batch_flat,
            reduction=str(reduction or "mean"),
            weighting=weighting,
            class_weight_cap=positive_weight_cap,
            target_class=target_class,
            active_events=active_events,
            validate_cached_targets=validate_cached_targets,
            dim_size=dim_size,
        )

    target = edge_attr_t.float().view_as(edge_attr_out)

    if _normalise_weighting_name(weighting) == "event_balanced_bce":
        per_event_loss, loss_masks, _, _ = _event_balanced_bce_per_event(
            edge_attr_out,
            target,
            batch_flat,
            criterion=criterion,
            positive_weight_cap=positive_weight_cap,
            negative_weight_cap=negative_weight_cap,
        )
        return per_event_loss, loss_masks

    if _normalise_weighting_name(weighting) not in {"legacy", "none", "unweighted"}:
        raise ValueError(f"Unsupported reconstruction weighting: {weighting!r}")

    loss = _safe_bce_with_logits(edge_attr_out, target)
    target_flat = target.flatten()
    loss_flat = loss.flatten()
    loss_masks = scatter(target_flat, batch_flat, reduce="sum") > 0
    per_event_loss = scatter(loss_flat, batch_flat, reduce=reduction)

    return per_event_loss, loss_masks


def HyperedgeLoss(
    x_out: Tensor,
    x_t: Tensor,
    x_t_batch: Tensor,
    criterion: Optional[callable] = BCELoss(reduction="none"),
    reduction: Optional[str] = "mean",
    weighting: str | None = "legacy",
    positive_weight_cap: float | None = None,
    negative_weight_cap: float | None = None,
    target_encoding: str = "binary",
    target_class: Tensor | None = None,
    active_events: Tensor | None = None,
    validate_cached_targets: bool = False,
    dim_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Calculate per-graph hyperedge loss.

    The returned mask is True for events with at least one positive hyperedge
    target. This supports partially reconstructable backgrounds such as
    single-top-like events if they have valid target hyperedges.
    """
    if (
        x_out is None
        or x_out.numel() == 0
        or x_t is None
        or x_t.numel() == 0
        or x_t_batch is None
        or x_t_batch.numel() == 0
    ):
        return (
            x_out.new_zeros(1),
            torch.zeros(1, dtype=torch.bool, device=x_out.device),
        )

    batch_flat = x_t_batch.flatten().to(torch.long)
    if str(target_encoding).strip().lower() == "typed":
        return _typed_reconstruction_loss(
            x_out,
            x_t,
            batch_flat,
            reduction=str(reduction or "mean"),
            weighting=weighting,
            class_weight_cap=positive_weight_cap,
            target_class=target_class,
            active_events=active_events,
            validate_cached_targets=validate_cached_targets,
            dim_size=dim_size,
        )

    target = x_t.float().view_as(x_out)

    if _normalise_weighting_name(weighting) == "event_balanced_bce":
        per_event_loss, loss_masks, _, _ = _event_balanced_bce_per_event(
            x_out,
            target,
            batch_flat,
            criterion=criterion,
            positive_weight_cap=positive_weight_cap,
            negative_weight_cap=negative_weight_cap,
        )
        return per_event_loss, loss_masks

    if _normalise_weighting_name(weighting) not in {"legacy", "none", "unweighted"}:
        raise ValueError(f"Unsupported reconstruction weighting: {weighting!r}")

    loss = _safe_bce_with_logits(x_out, target)
    target_flat = target.flatten()
    loss_flat = loss.flatten()
    loss_masks = scatter(target_flat, batch_flat, reduce="sum") > 0
    per_event_loss = scatter(loss_flat, batch_flat, reduce=reduction)

    return per_event_loss, loss_masks


def CombinedLoss(
    loss_hyperedge: Tensor,
    loss_edge: Tensor,
    loss_class: Optional[Tensor] = None,
    cls_target: Optional[Tensor] = None,
    alpha: Optional[float] = 0.5,
    beta: Optional[float] = 0.5,
    reduction: Optional[str] = "mean",
    loss_hyperedge_masks: Optional[Tensor] = None,
    loss_edge_masks: Optional[Tensor] = None,
) -> Tensor:
    """Combine reconstruction and classification losses.

    Reconstruction policy:
      - use edge loss for events with positive edge targets;
      - use hyperedge loss for events with positive hyperedge targets;
      - do not require cls_target == 1;
      - therefore partially reconstructable backgrounds can contribute.

    Classification policy:
      - if cls_target exists, apply S/B classification to all labelled events;
      - class balancing is applied between signal and background.

    Important:
      The class label no longer decides whether reconstruction loss is used.
      Reconstruction targets decide that.
    """
    alpha = float(alpha)
    beta = float(beta)

    loss_edge = loss_edge.flatten()
    loss_hyperedge = loss_hyperedge.flatten()

    edge_loss, has_edge = _masked_mean_no_sync(loss_edge, loss_edge_masks)
    hyper_loss, has_hyper = _masked_mean_no_sync(loss_hyperedge, loss_hyperedge_masks)

    # Combine only active reconstruction components.
    # If only edge truth exists, edge gets full weight.
    # If only hyperedge truth exists, hyperedge gets full weight.
    # If both exist, use alpha for hyperedge and 1-alpha for edge.
    edge_w = has_edge * loss_edge.new_tensor(1.0 - alpha)
    hyper_w = has_hyper * loss_hyperedge.new_tensor(alpha)
    weight_sum = edge_w + hyper_w
    edge_loss_for_term = torch.where(has_edge > 0, edge_loss, torch.nan_to_num(edge_loss, nan=0.0, posinf=0.0, neginf=0.0))
    hyper_loss_for_term = torch.where(has_hyper > 0, hyper_loss, torch.nan_to_num(hyper_loss, nan=0.0, posinf=0.0, neginf=0.0))
    edge_term = edge_w * edge_loss_for_term
    hyper_term = hyper_w * hyper_loss_for_term

    zero_reco = (
        torch.where(torch.isfinite(loss_hyperedge), loss_hyperedge, loss_hyperedge.new_zeros(())).sum() * 0.0
        + torch.where(torch.isfinite(loss_edge), loss_edge, loss_edge.new_zeros(())).sum() * 0.0
    )
    reco_loss = torch.where(
        weight_sum > 0,
        (edge_term + hyper_term) / weight_sum.clamp_min(1e-8),
        zero_reco,
    )

    has_class_targets = (
        cls_target is not None
        and cls_target.numel() > 0
        and loss_class is not None
        and loss_class.numel() > 0
    )

    if has_class_targets:
        cls_target = cls_target.float().flatten()
        loss_class = loss_class.flatten()

        n = min(cls_target.numel(), loss_class.numel())
        cls_target = cls_target[:n]
        loss_class = loss_class[:n]

        finite = torch.isfinite(loss_class)
        cls_target = cls_target[finite]
        loss_class = loss_class[finite]

        if loss_class.numel() > 0:
            cls_mask = cls_target > 0.5

            n_sig = cls_mask.sum().float()
            n_bkg = (~cls_mask).sum().float()
            eps = loss_class.new_tensor(1e-8)

            w_sig = (n_sig + n_bkg) / (2.0 * n_sig + eps)
            w_bkg = (n_sig + n_bkg) / (2.0 * n_bkg + eps)

            weights = torch.where(cls_mask, w_sig, w_bkg)
            class_loss = (loss_class * weights).sum() / (weights.sum() + eps)
        else:
            class_loss = reco_loss.new_tensor(0.0)
            has_class_targets = False
    else:
        class_loss = reco_loss.new_tensor(0.0)

    beta_eff = beta if has_class_targets else 0.0
    return (1.0 - beta_eff) * reco_loss + beta_eff * class_loss
