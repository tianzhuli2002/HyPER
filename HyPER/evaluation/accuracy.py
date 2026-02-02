import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_accuracy


def Accuracy(
    preds: Tensor,
    target: Tensor,
    batch: Tensor,
    num_patterns: int,
) -> Tensor:
    """
    Compute hyperedge fuzzy accuracy:
    - For each graph in `batch`, select top-k predicted hyperedges
    - Compare against truth hyperedges using binary accuracy

    Args:
        preds (Tensor): shape [N, 1] or [N]
        target (Tensor): shape [N, 1] or [N]
        batch (Tensor): shape [N], graph index per hyperedge
        num_patterns (int): number of true patterns per graph

    Returns:
        Tensor or None
    """
    if preds.numel() == 0:
        return None

    device = preds.device

    preds = preds.view(-1)
    target = target.view(-1)
    batch = batch.view(-1)

    evaluable = torch.zeros_like(preds, device=device)

    # Iterate per graph (safe and correct)
    for g in batch.unique():
        mask = batch == g
        if mask.sum() == 0:
            continue

        p_g = preds[mask]

        # pick top-k predictions inside this graph
        k = min(num_patterns, p_g.numel())
        topk_idx = torch.topk(p_g, k=k, largest=True).indices

        # map back to global indices
        global_idx = torch.nonzero(mask, as_tuple=False).view(-1)[topk_idx]
        evaluable[global_idx] = preds[global_idx]

    # Compute fuzzy accuracy
    return binary_accuracy(
        evaluable,
        target,
        threshold=0.0,
    )
