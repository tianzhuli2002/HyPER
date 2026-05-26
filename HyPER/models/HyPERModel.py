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
            validation_mode: Optional[str] = 'keep',
            classification_enabled: Optional[bool] = True,
            #classification_input_mode: Optional[str] = "edge_hyperedge",
            classification_loss_weight: Optional[float] = None,
        ):
        super().__init__()

        # if classification_input_mode not in ("edge_hyperedge", "all_embeddings"):
        #     raise ValueError("Supported classification input modes are: edge_hyperedge, all_embeddings.")

        self.save_hyperparameters()
        self.classification_enabled = bool(classification_enabled)
        self.classification_loss_weight = (
            self.hparams.beta if classification_loss_weight is None else float(classification_loss_weight)
        )

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

        self.Classification = None
        if self.classification_enabled:
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
        
        x_class = None
        if self.classification_enabled:
            _, col = edge_index
            batch_edges = batch[col] # [N_ge]
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

    @staticmethod
    def _class_targets(batch):
        cls_t = getattr(batch, 'cls_t', None)
        if cls_t is None or cls_t.numel() == 0:
            return None
        return cls_t.float().view(-1, 1)

    def _require_class_targets(self, batch, stage: str):
        cls_t = self._class_targets(batch)
        if cls_t is None:
            raise RuntimeError(
                f"S/B classification is enabled, but `cls_t` is missing or empty in the {stage} batch."
            )
        return cls_t

    def _combined_loss(self, loss_hyperedge, loss_edge, loss_class, cls_t, loss_hyperedge_masks, loss_edge_masks):
        return CombinedLoss(
            loss_hyperedge,
            loss_edge,
            loss_class,
            cls_t,
            alpha=self.hparams.alpha,
            beta=self.classification_loss_weight if self.classification_enabled else 0.0,
            reduction=self.hparams.reduction,
            loss_hyperedge_masks=loss_hyperedge_masks,
            loss_edge_masks=loss_edge_masks,
        )

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

        loss_edge,loss_edge_masks = EdgeLoss(edge_attr_prime, train_batch.edge_attr_t, train_batch.edge_attr_batch, criterion=criterion_edge, reduction='mean')
        loss_hyperedge, loss_hyperedge_masks = HyperedgeLoss(x_hat, train_batch.hyperedge_attr_t.float(), batch_hyperedge, criterion_hyperedge, reduction='mean')
        cls_t = self._require_class_targets(train_batch, "training") if self.classification_enabled else None
        loss_class = (
            ClassificationLoss(x_class, cls_t, criterion=BCELoss(reduction='none'))
            if self.classification_enabled else None
        )
        loss = self._combined_loss(loss_hyperedge, loss_edge, loss_class, cls_t, loss_hyperedge_masks, loss_edge_masks)

        # Logging
        self.log('loss/train_loss', loss, batch_size=len(train_batch), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if loss_class is not None and loss_class.numel() > 0:
            self.log('loss/train_classification_loss', loss_class.mean(), batch_size=len(train_batch), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

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

        loss_edge,loss_edge_masks = EdgeLoss(edge_attr_prime, val_batch.edge_attr_t, val_batch.edge_attr_batch, criterion=criterion_edge, reduction='mean')
        loss_hyperedge, loss_hyperedge_masks = HyperedgeLoss(x_hat, val_batch.hyperedge_attr_t.float(), batch_hyperedge, criterion_hyperedge, reduction='mean')
        cls_t = self._require_class_targets(val_batch, "validation") if self.classification_enabled else None
        loss_class = (
            ClassificationLoss(x_class, cls_t, criterion=BCELoss(reduction='none'))
            if self.classification_enabled else None
        )
        loss = self._combined_loss(loss_hyperedge, loss_edge, loss_class, cls_t, loss_hyperedge_masks, loss_edge_masks)
        
        # Validation Accuracy Calculation
        
        # Not considering bkg events for the reco accuracy
        if cls_t is not None:
            graph_mask = cls_t.view(-1) == 1
        else:
            graph_mask = torch.ones(int(val_batch.num_graphs), dtype=torch.bool, device=edge_attr_prime.device)
        ge_keep  = graph_mask[val_batch.edge_attr_batch]      

        accuracy_edge  = BinaryAccuracy(ignore_index=0).to(edge_attr_prime)

        # Logging
        self.log('loss/validation_loss', loss, batch_size=len(val_batch), on_step=val_on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if loss_class is not None and loss_class.numel() > 0:
            self.log('loss/validation_classification_loss', loss_class.mean(), batch_size=len(val_batch), on_step=val_on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

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
        if cls_t is None:
            return {'batch_size': len(val_batch)}

        preds = (x_class.view(-1) > 0.5).to(torch.int32)
        labels = cls_t.view(-1).to(torch.int32)

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
        x_class_out = x_class.squeeze(-1) if x_class is not None else None
        # print("PREDICTION STEP OUTPUTS")
        # print("x_out:", x_out)
        # print("edge_attr_out:", edge_attr_out)
        # print("N_nodes:", N_nodes)
        # print("encodings:", encodings)
        # print("x_class_out:", x_class_out)
        return x_out, edge_attr_out, N_nodes, encodings, x_class_out
