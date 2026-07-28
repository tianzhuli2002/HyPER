from torch.nn import Module
from torch_geometric.nn import MetaLayer
from typing import Optional

from .messagePassing import EdgeModel, NodeModel, GlobalModel


class MPNNs(Module):
    r""" The Message Passing Neural Networks.

    Args:
        node_in_channels (int): number of node features of input graph.
        edge_in_channels (int): number of edge features of input graph.
        global_in_channels (int): number of global features of input graph.
        node_out_channels (int, optional): number of node features of output graph.
        edge_out_channels (int, optional): number of edge features of output graph.
        global_out_channels (int, optional): number of global features of output graph.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
    :rtype: :class:`Tuple[torch.Tensor,torch.Tensor]
    """
    def __init__(
            self,
            node_in_channels,
            edge_in_channels,
            global_in_channels,
            node_out_channels: Optional[int] = 1, 
            edge_out_channels: Optional[int] = 1,
            global_out_channels: Optional[int] = 1,
            message_feats: Optional[int] = 32,
            dropout: Optional[float] = 0.01,
        ) -> None:
        super(MPNNs, self).__init__()

        self.MPNNBlock = MetaLayer(
            EdgeModel(node_in_channels=node_in_channels, edge_in_channels=edge_in_channels, global_in_channels=global_in_channels, edge_out_channels=edge_out_channels, message_feats=message_feats, dropout=dropout),
            NodeModel(node_in_channels=node_in_channels, edge_in_channels=edge_out_channels, global_in_channels=global_in_channels, node_out_channels=node_out_channels, message_feats=message_feats, dropout=dropout),
            GlobalModel(node_in_channels=node_out_channels, global_in_channels=global_in_channels, global_out_channels=global_out_channels, message_feats=message_feats, dropout=dropout)
        )

    def reset_parameters(self):
        for layer in self.MPNNBlock.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.MPNNBlock(x, edge_index, edge_attr, u, batch)
