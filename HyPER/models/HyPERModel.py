import torch

from torch.nn import Sequential as Seq, Linear, BCELoss, Sigmoid
from torch.optim import lr_scheduler, Adam
from torch_geometric.utils import unbatch, degree
from lightning import LightningModule
from .MPNNs import MPNNs
from .hyperedge import HyperedgeModel
from .classification import classificationModel
from .loss import ClassificationLoss, HyperedgeLoss, EdgeLoss, CombinedLoss
from HyPER.evaluation import Accuracy
from torchmetrics.classification import BinaryAccuracy
from typing import Optional


class HyPERModel(LightningModule):
    r"""HyPER model.
    :obj:`HyPERModel` is built using the message passing modules :obj:`MPGNN`
    and the hyperedge module :obj:`HyperedgeModel`.

    Args:
        node_in_channels (int): number of node features of input graph.
        edge_in_channels (int): number of edge features of input graph.
        global_in_channels (int): number of global features of input graph.
        message_feats (int, optional): number of intermediate features. (default :obj:`int`=32)
        dropout (float, optional): probability of an element to be zeroed. (default :obj:`float`=0.01)
        message_passing_recurrent (int, optional): number of message passing steps. (default :obj:`int`=3)
        criterion_edge (callable, optional): edge loss function. (default: :obj:`callable`=torch.nn.BCELoss() )
        criterion_hyperedge (callable, optional): hyperedge loss function. (default: :obj:`callable`=torch.nn.BCELoss() )
        optimizer (callable, optional): optimizer. (default: :obj:`callable`=torch.optim.Adam)
        lr (float, optional): learning rate. (default: :obj:`float`=1e-3)
        alpha (float, optional): edge/hyperedge loss balance. (default: :obj:`float`=0.5)
        reduction (str, optional): specifies the reduction to apply to the output loss. (default: 'mean')

    :rtype: :class:`Tuple[torch.Tensor,torch.Tensor,torch.Tensor]
    """
    def __init__(
            self,
            node_in_channels,
            edge_in_channels,
            global_in_channels,
            message_feats: Optional[int] = 32,
            dropout: Optional[float] = 0.01,
            message_passing_recurrent: Optional[int] = 3,
            contraction_feats: Optional[int] = 32,
            hyperedge_order: Optional[int] = 3,
            criterion_edge: Optional[str] = "BCE",
            criterion_hyperedge: Optional[str] = "BCE",
            optimizer: Optional[str] = "Adam",
            lr: Optional[float] = 1e-3,
            alpha: Optional[float] = 0.5,
            beta: Optional[float] = 0.5,
            reduction: Optional[str] = 'mean',
            validation_mode: Optional[str] = 'keep'
        ):
        super().__init__()

        self.save_hyperparameters()

        for i in range(self.hparams.message_passing_recurrent):
            if i == 0:
                setattr(self, 'MessagePassing' + str(i),
                        MPNNs(
                            node_in_channels, edge_in_channels, global_in_channels,
                            node_out_channels = self.hparams.message_feats, edge_out_channels = self.hparams.message_feats, global_out_channels = self.hparams.message_feats,
                            message_feats = self.hparams.message_feats, dropout = self.hparams.dropout
                        )
                )

            # elif i == self.hparams.message_passing_recurrent-1:
            #     setattr(self, 'MessagePassing' + str(i),
            #             MPNNs(
            #                 self.hparams.message_feats, self.hparams.message_feats, self.hparams.message_feats,
            #                 node_out_channels = self.hparams.message_feats, edge_out_channels = 1, global_out_channels = self.hparams.message_feats,
            #                 message_feats = self.hparams.message_feats, dropout = self.hparams.dropout, activation = Sigmoid(), p_out = 'edge'
            #             )
            #     )

            else:
                setattr(self, 'MessagePassing' + str(i),
                        MPNNs(
                            self.hparams.message_feats, self.hparams.message_feats, self.hparams.message_feats,
                            node_out_channels = self.hparams.message_feats, edge_out_channels = self.hparams.message_feats, global_out_channels = self.hparams.message_feats,
                            message_feats = self.hparams.message_feats, dropout = self.hparams.dropout
                        )
                )

        self.Hyperedge = HyperedgeModel(
            node_in_channels = self.hparams.message_feats, node_out_channels = 1, global_in_channels = self.hparams.message_feats,
            message_feats = self.hparams.contraction_feats, dropout = self.hparams.dropout
        )

        self.Classification = classificationModel(
            n_feats_out=1,
            contraction_feats=self.hparams.contraction_feats,
            message_feats=self.hparams.message_feats,
            dropout=self.hparams.dropout
        )

        self.he_head  = Seq(Linear(self.hparams.contraction_feats, 1), Sigmoid()) # last layer for hyperedges
        self.ge_head  = Seq(Linear(self.hparams.message_feats, 1), Sigmoid()) # last layer for edges


    def forward(self, x, edge_index, edge_attr, u, batch, hyperedge_index, hyperedge_index_batch):
        # Message Passing Step
        for i in range(self.hparams.message_passing_recurrent):
            if i == 0:
                x_prime, edge_attr_prime, u_prime = getattr(self, 'MessagePassing' + str(i))(
                    x, edge_index, edge_attr, u, batch
                )
            else:
                x_prime, edge_attr_prime, u_prime = getattr(self, 'MessagePassing' + str(i))(
                    x_prime, edge_index, edge_attr_prime, u_prime, batch
                )

        # Hyperedge Step
        x_hat, batch_hyperedge  = self.Hyperedge(x_prime, u_prime, batch, hyperedge_index, hyperedge_index_batch, self.hparams.hyperedge_order)
        
        # Classification step
        _, col = edge_index
        batch_edges = batch[col] # [N_ge]
        #x_class = self.Classification(x_hat, edge_attr_prime, batch_edges, batch_hyperedge)
        x_class = self.Classification(
            feat_HE=x_hat,
            feat_GE=edge_attr_prime,
            feat_N=x_prime,
            feat_U=u_prime,
            batch_GE=batch_edges,
            batch_HE=batch_hyperedge,
            batch_N=batch,
        )
        # Last layer + sigmoid
        p_hyper = self.he_head(x_hat)             # [N_he, 1]
        p_edge  = self.ge_head(edge_attr_prime)    # [N_ge, 1]

        return p_hyper, batch_hyperedge, p_edge, x_class

    def _shared_step(self, data):
        return self.forward(data.x, data.edge_index, data.edge_attr, data.u, data.batch, data.hyperedge_index, data.hyperedge_index_batch)

    def configure_optimizers(self):
        if str(self.hparams.optimizer).lower() == 'adam':
            optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        # --------- custom optimizers ---------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported optimizers are: `torch.Adam`.")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10),
                "interval": "epoch",
                "monitor": "loss/validation_loss", #_epoch",
                "frequency": 1,
                "strict": True
            },
        }

    def training_step(self, train_batch, batch_idx):
        x_hat, batch_hyperedge, edge_attr_prime, x_class = self._shared_step(train_batch)

        # Train Loss Calculation
        if str(self.hparams.criterion_edge).lower() == 'bce':
            criterion_edge = BCELoss(reduction='none')
        # ------- custom loss functions -------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported edge loss functions are: `torch.BCELoss`.")

        if str(self.hparams.criterion_hyperedge).lower() == 'bce':
            criterion_hyperedge = BCELoss(reduction='none')
        # ------- custom loss functions -------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported edge loss functions are: `torch.BCELoss`.")

        loss_class = ClassificationLoss(x_class, train_batch.cls_t, criterion=BCELoss(reduction='none'))
        loss_edge,loss_edge_masks = EdgeLoss(edge_attr_prime, train_batch.edge_attr_t, train_batch.edge_attr_batch, criterion=criterion_edge, reduction='mean')
        loss_hyperedge, loss_hyperedge_masks = HyperedgeLoss(x_hat, train_batch.hyperedge_attr_t.float(), batch_hyperedge, criterion_hyperedge, reduction='mean')
        loss = CombinedLoss(loss_hyperedge, loss_edge, loss_class, train_batch.cls_t, alpha=self.hparams.alpha, beta=self.hparams.beta, reduction=self.hparams.reduction, loss_hyperedge_masks=loss_hyperedge_masks,loss_edge_masks=loss_edge_masks)

        # Logging
        self.log('loss/train_loss', loss, batch_size=len(train_batch), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x_hat, batch_hyperedge, edge_attr_prime, x_class = self._shared_step(val_batch)
        fast_validation = str(self.hparams.validation_mode).lower() == 'fast'
        val_on_step = not fast_validation

        # Validation Loss Calculation
        if str(self.hparams.criterion_edge).lower() == 'bce':
            criterion_edge = BCELoss(reduction='none')
        # ------- custom loss functions -------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported edge loss functions are: `torch.BCELoss`.")

        if str(self.hparams.criterion_hyperedge).lower() == 'bce':
            criterion_hyperedge = BCELoss(reduction='none')
        # ------- custom loss functions -------
        # elif
        # -------------------------------------
        else:
            raise ValueError("Supported edge loss functions are: `torch.BCELoss`.")

        loss_class = ClassificationLoss(x_class, val_batch.cls_t, criterion=BCELoss(reduction='none'))
        loss_edge,loss_edge_masks = EdgeLoss(edge_attr_prime, val_batch.edge_attr_t, val_batch.edge_attr_batch, criterion=criterion_edge, reduction='mean')
        loss_hyperedge, loss_hyperedge_masks = HyperedgeLoss(x_hat, val_batch.hyperedge_attr_t.float(), batch_hyperedge, criterion_hyperedge, reduction='mean')
        loss = CombinedLoss(loss_hyperedge, loss_edge, loss_class, val_batch.cls_t, alpha=self.hparams.alpha, beta=self.hparams.beta, reduction=self.hparams.reduction, loss_hyperedge_masks=loss_hyperedge_masks,loss_edge_masks=loss_edge_masks)
        
        # Validation Accuracy Calculation
        
        # Not considering bkg events for the reco accuracy
        graph_mask = (val_batch.cls_t.squeeze(-1) == 1)               
        ge_keep  = graph_mask[val_batch.edge_attr_batch]      
        he_keep = graph_mask[batch_hyperedge]

        accuracy_edge  = BinaryAccuracy(ignore_index=0).to(edge_attr_prime)

        # Logging
        self.log('loss/validation_loss', loss, batch_size=len(val_batch), on_step=val_on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        pred_edge = edge_attr_prime.flatten()[ge_keep]
        tgt_edge = val_batch.edge_attr_t.float().flatten()[ge_keep]

        # guard against empty masked tensors
        if pred_edge.numel() > 0 and tgt_edge.numel() > 0:
            self.log(
                'fuzzy_accuracy/validation_accuracy_edge',
                accuracy_edge(pred_edge, tgt_edge),
                batch_size=len(val_batch),
                on_step=val_on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        # --- Compute simple S/B counts for the event-level classifier (x_class) ---
        # x_class shape: [N_events, 1] or [N_events]
        preds = (x_class.view(-1) > 0.5).to(torch.int32)
        labels = val_batch.cls_t.view(-1).to(torch.int32)

        tp = torch.sum(((preds == 1) & (labels == 1)).to(torch.int32)).cpu()
        fp = torch.sum(((preds == 1) & (labels == 0)).to(torch.int32)).cpu()

        # Safely compute S/B: clamp fp to avoid huge ratios when fp~0
        s_over_b_batch = tp.float() / torch.clamp(fp.float(), min=1.0)

        # Log it live to TensorBoard (clamped to reasonable range for stability)
        self.log('metrics/validation_S_over_B_batch', torch.clamp(s_over_b_batch, min=0.0, max=1000.0),
                batch_size=len(val_batch),
            on_step=val_on_step,
            on_epoch=True,
            prog_bar=not fast_validation,
                logger=True,
                sync_dist=True)    
        return {'tp': tp, 'fp': fp, 'batch_size': len(val_batch)}
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # print("PREDICTION STEP INPUTS")
        # print("batch:", batch)
        # #print(batch)
        # print(batch.x)
        # print(batch.edge_index)
        # print(batch.edge_attr)
        # print(batch.u)
        # print(batch.batch)
        #print(x_hat, batch_hyperedge, edge_attr_prime, x_class)
        x_hat, batch_hyperedge, edge_attr_prime, x_class = self._shared_step(batch)

        # Unbatch results
        x_out         = unbatch(x_hat, batch_hyperedge.type(torch.int64), 0)
        edge_attr_out = unbatch(edge_attr_prime, batch.edge_attr_batch, 0)
        N_nodes       = degree(batch.batch).cpu().flatten().tolist()
        encodings     = unbatch(batch.x[:,-1].reshape(-1,1),batch.batch, 0)
        x_class_out       = x_class.squeeze(-1)
        # print("PREDICTION STEP OUTPUTS")
        # print("x_out:", x_out)
        # print("edge_attr_out:", edge_attr_out)
        # print("N_nodes:", N_nodes)
        # print("encodings:", encodings)
        # print("x_class_out:", x_class_out)
        return x_out, edge_attr_out, N_nodes, encodings, x_class_out