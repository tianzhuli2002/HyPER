import torch
import math

from torch.nn import Module, Sequential as Seq, Linear, ReLU, Dropout, Sigmoid, Parameter, init
from torch.nn.functional import relu
from torch_geometric.utils import scatter
from HyPER.utils.custom_scatter import custom_scatter

# class classificationModel(Module):
#     r"""Classification MLP Model.

#     Args:
#         n_feats_out (int): number of node features of the output graph.
#         message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
#         dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)

#     :rtype: :class:`Tuple[Tensor,Tensor]`
#     """
#     def __init__(self, n_feats_out, contraction_feats: int=64, message_feats: int=32, dropout=0.01):
#         super().__init__()
#         self.message_feats    = message_feats
#         self.contraction_feats = contraction_feats
#         self.mlp_class  = Seq(Linear(message_feats*5+contraction_feats*2, message_feats+contraction_feats),
#                           ReLU(),
#                           Dropout(p=dropout),
#                           Linear(message_feats+contraction_feats, message_feats+contraction_feats),
#                           ReLU(),
#                           Dropout(p=dropout),
#                           Linear(message_feats+contraction_feats, n_feats_out),
#                           Sigmoid())
#         self.reset_parameters()

class classificationModel(Module):
    r"""Event-level S/B classification head using learned HyPER embeddings.

    Inputs:
      - node embeddings after message passing
      - edge embeddings after message passing
      - hyperedge embeddings after hyperedge construction
      - global/event embeddings after message passing
    """

    def __init__(self, n_feats_out, contraction_feats: int = 64, message_feats: int = 32, dropout=0.01):
        super().__init__()

        self.message_feats = message_feats
        self.contraction_feats = contraction_feats

        # Event summary:
        # node mean/max      -> 2 * message_feats
        # edge mean/max      -> 2 * message_feats
        # hyperedge mean/max -> 2 * contraction_feats
        # global embedding   -> 1 * message_feats
        in_feats = 5 * message_feats + 2 * contraction_feats
        hidden_feats = 2 * message_feats + contraction_feats

        self.mlp_class = Seq(
            Linear(in_feats, hidden_feats),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_feats, hidden_feats),
            ReLU(),
            Dropout(p=dropout),
            Linear(hidden_feats, n_feats_out),
            Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp_class.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, feat_HE, feat_GE, feat_N, feat_U, batch_GE, batch_HE, batch_N):
        r"""
        Build an event-level S/B representation from all learned HyPER embeddings.

        feat_HE:  [N_he, contraction_feats] hyperedge embeddings
        feat_GE:  [N_ge, message_feats]     edge embeddings
        feat_N:   [N_nodes, message_feats]  node embeddings
        feat_U:   [N_events, message_feats] global/event embeddings
        batch_GE: [N_ge]                    graph ids for edges
        batch_HE: [N_he]                    graph ids for hyperedges
        batch_N:  [N_nodes]                 graph ids for nodes

        returns:
            [N_events, n_feats_out]
        """
        
        # Prefer feat_U to define the number of events, because global features
        # are one per graph/event and are guaranteed to exist for every event.
        num_events = feat_U.size(0)

        # Hyperedge-level summaries: candidate topology/reconstruction structure.
        he_mean = scatter(feat_HE, batch_HE, dim=0, dim_size=num_events, reduce="mean")
        he_max = scatter(feat_HE, batch_HE, dim=0, dim_size=num_events, reduce="max")

        # Edge-level summaries: pairwise object relationships.
        ge_mean = scatter(feat_GE, batch_GE, dim=0, dim_size=num_events, reduce="mean")
        ge_max = scatter(feat_GE, batch_GE, dim=0, dim_size=num_events, reduce="max")

        # Node-level summaries: updated object representations.
        n_mean = scatter(feat_N, batch_N, dim=0, dim_size=num_events, reduce="mean")
        n_max = scatter(feat_N, batch_N, dim=0, dim_size=num_events, reduce="max")

        # Global/event embedding already has one row per event.
        u_event = feat_U

        x_in = torch.cat(
            [
                n_mean,
                n_max,
                ge_mean,
                ge_max,
                he_mean,
                he_max,
                u_event,
            ],
            dim=1,
        ).float()

#### SOME GUARDRAILS TO CATCH SHAPE MISMATCHES EARLY ####
        if feat_U.dim() != 2:
            raise ValueError(f"Expected feat_U shape [N_events, F], got {tuple(feat_U.shape)}")
        if feat_N.dim() != 2 or feat_GE.dim() != 2 or feat_HE.dim() != 2:
            raise ValueError(
                "Expected 2D embeddings: "
                f"feat_N={tuple(feat_N.shape)}, "
                f"feat_GE={tuple(feat_GE.shape)}, "
                f"feat_HE={tuple(feat_HE.shape)}"
            )
        expected = 5 * self.message_feats + 2 * self.contraction_feats
        if x_in.size(1) != expected:
            raise ValueError(f"Classification input has {x_in.size(1)} features, expected {expected}")


        return self.mlp_class(x_in)