import torch
import math
import time

from torch.nn import Module, Sequential as Seq, Linear, ReLU, Dropout, Sigmoid, Parameter, init
from torch.nn.functional import relu

from HyPER.utils import softmax


def _profile_now(cuda_sync: bool = False) -> float:
    if cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class HyperedgeModel(Module):
    r"""Hyperedge Model.

    Args:
        n_node_feats (int): number of node features of input graph.
        n_node_feats_out (int): number of node features of the output graph.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)

    :rtype: :class:`Tuple[Tensor,Tensor]`
    """
    def __init__(self, node_in_channels, node_out_channels, global_in_channels, message_feats: int=32, dropout=0.01):
        super().__init__()
        self.node_in_channels = node_in_channels
        self.message_feats    = message_feats
        self.mlp_x  = Seq(Linear(node_in_channels+global_in_channels, message_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats, message_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats, message_feats)
                        )
        self.weight = Parameter(torch.empty((message_feats, message_feats)))
        self.x_hat  = Seq(Linear(message_feats*2, message_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats, message_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats, message_feats)
                        )
                        #   Sigmoid())
        self._profile_sections_enabled = False
        self._profile_cuda_synchronize = False
        self._last_profile = {}
        self._debug_tensors_enabled = False
        self._last_debug_tensors = {}
        self.reset_parameters()

    def set_profile_sections(self, enabled: bool = False, cuda_synchronize: bool = False):
        self._profile_sections_enabled = bool(enabled)
        self._profile_cuda_synchronize = bool(cuda_synchronize)

    def set_debug_tensors(self, enabled: bool = False):
        self._debug_tensors_enabled = bool(enabled)
        self._last_debug_tensors = {}

    def reset_parameters(self):
        for layer in self.mlp_x.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        for layer in self.x_hat.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def __hyperedge_finding__(self, x, hyperedge_index, r):
        idx = hyperedge_index.t().contiguous()
        if idx.numel() == 0:
            empty = idx.new_empty((0,))
            return x.index_select(0, empty)

        r = int(r)
        if r == 2:
            return x.index_select(0, idx[:, 0]) + x.index_select(0, idx[:, 1])

        if r == 3:
            return (
                x.index_select(0, idx[:, 0])
                + x.index_select(0, idx[:, 1])
                + x.index_select(0, idx[:, 2])
            )

        out = x.new_zeros((idx.size(0), x.size(-1)))
        for j in range(r):
            out = out + x.index_select(0, idx[:, j])
        return out

    def weighting(self, x_hyper, batch_hyper, dim_size=None):
        if dim_size is None:
            dim_size = int(batch_hyper.max().item()) + 1 if batch_hyper.numel() else 0
        coefficient = softmax(x_hyper, index=batch_hyper, dim_size=dim_size)
        return coefficient * relu(torch.mm(x_hyper, self.weight), inplace=True)

    def forward(self, x, u, batch, hyperedge_index, batch_hyper, r):
        if self._profile_sections_enabled:
            self._last_profile = {}
            sync = self._profile_cuda_synchronize
        else:
            sync = False
        x_hyper_nodes = self.mlp_x(torch.cat([x, u[batch]], dim=1).float())
        t0 = _profile_now(sync) if self._profile_sections_enabled else None
        x_hyper = self.__hyperedge_finding__(x_hyper_nodes, hyperedge_index, r)
        if self._profile_sections_enabled:
            t1 = _profile_now(sync)
            self._last_profile["hyperedge_finding_seconds"] = t1 - t0
            t0 = t1
        x_hyper_hat = self.weighting(x_hyper, batch_hyper, dim_size=int(u.size(0)))
        if self._profile_sections_enabled:
            t1 = _profile_now(sync)
            self._last_profile["hyperedge_weighting_seconds"] = t1 - t0
        out = torch.cat([x_hyper, x_hyper_hat], dim=1).float()
        x_hat = self.x_hat(out)
        if self._debug_tensors_enabled:
            self._last_debug_tensors = {
                "x_hyper_nodes": x_hyper_nodes,
                "x_hyper": x_hyper,
                "x_hyper_hat": x_hyper_hat,
                "x_hat": x_hat,
            }
        return x_hat, batch_hyper
