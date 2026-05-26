import torch

from torch import Tensor
from torch.nn import BCELoss
from torch_scatter import scatter

from typing import Optional

def ClassificationLoss(cls_out: Tensor, cls_t: Tensor,
                       criterion: Optional[callable] = BCELoss(reduction='none')) -> Tensor:
    """
    Args:
        cls_out (Tensor): output classification scores.
        cls_t (Tensor): classification targets.
        criterion (optional: callable): loss/cost function (default: BCELoss).

    :rtype: :class:`Tensor`
    """
    if cls_t is None or cls_t.numel() == 0:
        return cls_out.new_zeros(0)
    if cls_out.numel() == 0:
        return cls_t.new_zeros(0, dtype=torch.float32)
    l = criterion(cls_out.view_as(cls_t.float()), cls_t.float())
    return l.flatten()

def EdgeLoss(edge_attr_out: Tensor, edge_attr_t: Tensor, edge_attr_batch: Tensor,
             criterion: Optional[callable] = BCELoss(reduction='none'), reduction: Optional[str] = 'mean') -> Tensor:
    r"""Calculate per graph edge loss.

    Args:
        edge_attr_out (Tensor): output edge features.
        edge_attr_t (Tensor): edge targets.
        edge_attr_batch (Tensor): edge batch.
        criterion (optional: callable): loss/cost function (default: BCELoss).
        reduction (optional: str): the reduce operation (default: 'mean').

    :rtype: :class:`Tensor`
    """
    if edge_attr_out.numel() == 0 or edge_attr_t is None or edge_attr_t.numel() == 0:
        return edge_attr_out.new_zeros(1), torch.zeros(1, dtype=torch.bool, device=edge_attr_out.device)

    l = criterion(edge_attr_out, edge_attr_t.float())

    # build loss masks per graph/event: which edges have truth at all
    loss_masks = scatter(edge_attr_t.flatten(), edge_attr_batch, reduce='sum') > 0

    # reduce per-event
    per_event_loss = scatter(l.flatten(), edge_attr_batch, reduce=reduction)

    return per_event_loss, loss_masks


def HyperedgeLoss(x_out: Tensor, x_t: Tensor, x_t_batch: Tensor,
                 criterion: Optional[callable] = BCELoss(reduction='none'), reduction: Optional[str] = 'mean') -> Tensor:
    r"""Calculate per hyperedge loss.

    Args:
        x_out (Tensor): output hyperedge features.
        x_t (Tensor): hyperedge targets.
        x_t_batch (Tensor): hyperedge batch.
        criterion (optional: callable): loss/cost function (default: BCELoss).
        reduction (optional: str): the reduce operation (default: 'mean').
    
    :rtype: :class:`Tensor`
    """
    if x_out.numel() == 0 or x_t is None or x_t.numel() == 0:
        return x_out.new_zeros(1), torch.zeros(1, dtype=torch.bool, device=x_out.device)

    l = criterion(x_out, x_t.float())
    l = scatter(l.flatten(), x_t_batch, reduce=reduction)
    loss_masks = scatter(x_t.flatten(), x_t_batch, reduce='sum') > 0
    return l, loss_masks


    
def CombinedLoss(loss_hyperedge: Tensor, loss_edge: Tensor, loss_class: Optional[Tensor] = None,
                 cls_target: Optional[Tensor] = None, alpha: Optional[float] = 0.5, beta: Optional[float] = 0.5, reduction: Optional[str] = 'mean',
                 loss_hyperedge_masks: Optional[Tensor] = None, loss_edge_masks: Optional[Tensor] = None) -> Tensor:

    eps = 1e-8

    # ----- Flatten everything -----
    loss_edge = loss_edge.flatten()
    loss_hyperedge = loss_hyperedge.flatten()
    loss_class = loss_class.flatten() if loss_class is not None else None
    has_class_targets = (
        cls_target is not None
        and cls_target.numel() > 0
        and loss_class is not None
        and loss_class.numel() > 0
    )
    cls_mask = cls_target.flatten() > 0 if has_class_targets else None


    # ----- Apply masks to drop unmatched truth entries -----
    if loss_edge_masks is not None:
        loss_edge_masks = loss_edge_masks.flatten().float()
        loss_edge = loss_edge * loss_edge_masks
        # normalise ONLY by number of valid edges
        edge_den = loss_edge_masks.sum() + eps
        edge_loss = loss_edge.sum() / edge_den
    else:
        edge_loss = loss_edge.mean()

    if loss_hyperedge_masks is not None:
        loss_hyperedge_masks = loss_hyperedge_masks.flatten().float()
        loss_hyperedge = loss_hyperedge * loss_hyperedge_masks
        hyp_den = loss_hyperedge_masks.sum() + eps
        hyper_loss = loss_hyperedge.sum() / hyp_den
    else:
        hyper_loss = loss_hyperedge.mean()

    # ----- Reconstruction loss only on signal events -----
    # (background = no truth structure → no reconstruction)
    if cls_mask is None or cls_mask.any():
        reco_loss = alpha * hyper_loss + (1 - alpha) * edge_loss
    else:
        reco_loss = loss_hyperedge.new_tensor(0.0)

    # ----- Classification is always applied -----
    if has_class_targets:
        n_sig = cls_mask.sum().float()
        n_bkg = (~cls_mask).sum().float()
        w_sig = (n_sig + n_bkg) / (2.0 * n_sig + eps)
        w_bkg = (n_sig + n_bkg) / (2.0 * n_bkg + eps)
        weights = torch.where(cls_mask, w_sig, w_bkg)
        class_loss = (loss_class * weights).sum() / (weights.sum() + eps)
    else:
        class_loss = reco_loss.new_tensor(0.0)

    # ----- Combine -----
    beta_eff = beta if has_class_targets else 0.0
    return (1 - beta_eff) * reco_loss + beta_eff * class_loss
