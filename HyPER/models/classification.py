import torch
import math

from torch.nn import Module, Sequential as Seq, Linear, ReLU, Dropout, Sigmoid, Parameter, init
from torch.nn.functional import relu
from torch_geometric.utils import scatter
from HyPER.utils.custom_scatter import custom_scatter

class classificationModel(Module):
    r"""Classification MLP Model.

    Args:
        n_feats_out (int): number of node features of the output graph.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)

    :rtype: :class:`Tuple[Tensor,Tensor]`
    """
    def __init__(self, n_feats_out, contraction_feats: int=64, message_feats: int=32, dropout=0.01):
        super().__init__()
        self.message_feats    = message_feats
        self.contraction_feats = contraction_feats
        self.mlp_class  = Seq(Linear(message_feats*2+contraction_feats*2, message_feats+contraction_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats+contraction_feats, message_feats+contraction_feats),
                          ReLU(),
                          Dropout(p=dropout),
                          Linear(message_feats+contraction_feats, n_feats_out),
                          Sigmoid())
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp_class.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, feat_HE, feat_GE, batch_GE, batch_HE):
        """
        feat_HE:  [N_he, F]    hyperedge features
        feat_GE:  [N_ge, F]    edge features
        batch_HE: [N_he]       graph ids (0..B-1)
        batch_GE: [N_ge]       graph ids (0..B-1)
        returns:  [num_events, n_feats_out]
        """
        num_events = batch_HE.max() + 1 # number of events
        # num_events = batch_HE.max().to(torch.int64) + 1 # number of events

        # Aggregation of hyperedges, edges along the features: mean and max
        he_mean = scatter(feat_HE, batch_HE, dim=0, dim_size=num_events, reduce='mean')  # [num_events, F]
        he_max = scatter(feat_HE, batch_HE, dim=0, dim_size=num_events, reduce='max')   # [num_events, F]
        #he_max = custom_scatter(feat_HE, batch_HE, dim=0, dim_size=num_events.item(), reduce='amax')   # [num_events, F]
        ge_mean = scatter(feat_GE, batch_GE, dim=0, dim_size=num_events, reduce='mean')  # [num_events, F]
        ge_max = scatter(feat_GE, batch_GE, dim=0, dim_size=num_events, reduce='max')   # [num_events, F]
        #ge_max = custom_scatter(feat_GE, batch_GE, dim=0, dim_size=num_events.item(), reduce='amax')   # [num_events, F]

        # Then concatenate hyperedge features with edge features
        x_in = torch.cat([he_mean, he_max, ge_mean, ge_max], dim=1).float()

        # Then a beautiful MLP, return the output
        return self.mlp_class(x_in)