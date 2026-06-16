import torch

from torch.nn import Sequential as Seq, Linear, BCELoss
from torch.optim import lr_scheduler, Adam, AdamW
from torch_geometric.utils import unbatch, degree
from lightning import LightningModule
from .MPNNs import MPNNs
from .hyperedge import HyperedgeModel
from .classification import classificationModel
from .loss import ClassificationLoss, HyperedgeLoss, EdgeLoss, CombinedLoss
from HyPER.evaluation import Accuracy
from torchmetrics.classification import BinaryAccuracy
from torch_scatter import scatter
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
            weight_decay: Optional[float] = 0.0,
            lr_scheduler_enabled: Optional[bool] = True,
            lr_scheduler_method: Optional[str] = "reduce_on_plateau",
            lr_scheduler_monitor: Optional[str] = "val_loss",
            lr_scheduler_mode: Optional[str] = "min",
            lr_scheduler_factor: Optional[float] = 0.8,
            lr_scheduler_patience: Optional[int] = 10,
            lr_scheduler_min_lr: Optional[float] = 0.0,
            lr_scheduler_frequency: Optional[int] = 1,
            alpha: Optional[float] = 0.5,
            beta: Optional[float] = 0.5,
            reduction: Optional[str] = 'mean',
            validation_mode: Optional[str] = 'keep',
            classification_enabled: Optional[bool] = True,
            #classification_input_mode: Optional[str] = "edge_hyperedge",
            classification_loss_weight: Optional[float] = None,
            reconstruction_enabled: Optional[bool] = True,
            reconstruction_weighting: Optional[str] = "legacy",
            positive_weight_cap: Optional[float] = 50.0,
            negative_weight_cap: Optional[float] = 5.0,
            edge_out_channels: int = 1,
            hyperedge_out_channels: int = 1,
            target_encoding: str = "binary",
        ):
        super().__init__()

        # if classification_input_mode not in ("edge_hyperedge", "all_embeddings"):
        #     raise ValueError("Supported classification input modes are: edge_hyperedge, all_embeddings.")

        self.save_hyperparameters()
        self.classification_enabled = bool(classification_enabled)
        self.reconstruction_enabled = bool(reconstruction_enabled)
        if not self.reconstruction_enabled and not self.classification_enabled:
            raise ValueError(
                "Invalid HyPERModel configuration: reconstruction.enabled=false "
                "requires classification.enabled=true. Refusing to train a model "
                "with no active loss."
            )
        self.classification_loss_weight = (
            self.hparams.beta if classification_loss_weight is None else float(classification_loss_weight)
        )
        self.reconstruction_weighting = str(reconstruction_weighting or "legacy").lower()
        self.positive_weight_cap = None if positive_weight_cap is None else float(positive_weight_cap)
        self.negative_weight_cap = None if negative_weight_cap is None else float(negative_weight_cap)
        self.target_encoding = str(target_encoding or "binary").strip().lower()
        if self.target_encoding not in {"binary", "typed"}:
            raise ValueError("target_encoding must be either 'binary' or 'typed'.")

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

        self.he_head = Seq(Linear(self.hparams.contraction_feats, int(self.hparams.hyperedge_out_channels)))
        self.ge_head = Seq(Linear(self.hparams.message_feats, int(self.hparams.edge_out_channels)))


    def freeze_for_probe(self, trainable_prefixes=("Classification.",)):
        """Freeze all parameters except the named trainable prefixes."""
        prefixes = tuple(str(prefix) for prefix in trainable_prefixes)
        trainable_names = []
        frozen_names = []

        for name, parameter in self.named_parameters():
            trainable = any(name.startswith(prefix) for prefix in prefixes)
            parameter.requires_grad = trainable
            if trainable:
                trainable_names.append(name)
            else:
                frozen_names.append(name)

        if not trainable_names:
            raise RuntimeError(
                "Probe freezing left no trainable parameters. "
                f"Requested trainable prefixes: {prefixes}"
            )
        if not frozen_names:
            raise RuntimeError("Probe freezing did not freeze any non-classification parameters.")
        bad_trainable = [
            name for name in trainable_names
            if not any(name.startswith(prefix) for prefix in prefixes)
        ]
        if bad_trainable:
            raise RuntimeError(f"Unexpected trainable parameters after probe freeze: {bad_trainable}")

        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        print("================================")
        print("Frozen-probe parameter summary")
        print(f"Trainable prefixes: {prefixes}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Total parameters:     {total_params}")
        print("Trainable parameter names:")
        for name in trainable_names:
            print(f"  {name}")
        print("================================")
        return trainable_names, frozen_names


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
        # Reconstruction heads return logits. Convert to probabilities only for
        # metrics, prediction, or external consumers.
        p_hyper = self.he_head(x_hat)
        p_edge = self.ge_head(edge_attr_prime)

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
        if not self.reconstruction_enabled:
            if loss_class is None or cls_t is None or loss_class.numel() == 0:
                raise RuntimeError(
                    "Classifier-only training requires non-empty S/B classification "
                    "targets and classification loss."
                )

            # Classifier-only mode: preserve the existing class-balanced
            # ClassificationLoss handling inside CombinedLoss, but force the
            # reconstruction contribution to zero.
            dummy_reco = loss_hyperedge.new_zeros(1)
            dummy_mask = torch.zeros(1, dtype=torch.bool, device=loss_hyperedge.device)
            return CombinedLoss(
                dummy_reco,
                dummy_reco,
                loss_class,
                cls_t,
                alpha=0.5,
                beta=1.0,
                reduction=self.hparams.reduction,
                loss_hyperedge_masks=dummy_mask,
                loss_edge_masks=dummy_mask,
            )

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

    def _reco_loss_kwargs(self):
        return {
            "reduction": self.hparams.reduction,
            "weighting": self.reconstruction_weighting,
            "positive_weight_cap": self.positive_weight_cap,
            "negative_weight_cap": self.negative_weight_cap,
            "target_encoding": self.target_encoding,
        }

    def _target_positive_mask(self, target: torch.Tensor) -> torch.Tensor:
        if self.target_encoding == "typed":
            if target.ndim != 2 or target.size(1) == 0:
                return torch.zeros(target.size(0), dtype=torch.bool, device=target.device)
            return target.argmax(dim=1) != (target.size(1) - 1)
        return target.float().flatten() > 0.5

    def _reco_scalar_score(self, logits: torch.Tensor) -> torch.Tensor:
        if self.target_encoding == "typed":
            if logits.ndim != 2 or logits.size(1) == 0:
                return logits.new_zeros((logits.size(0), 1))
            probs = torch.softmax(logits, dim=1)
            return (1.0 - probs[:, -1:]).float()
        return torch.sigmoid(logits.float())

    def _positive_negative_counts(self, target, batch, num_graphs):
        if target is None or target.numel() == 0 or batch is None or batch.numel() == 0:
            device = batch.device if batch is not None and batch.numel() else torch.device("cpu")
            return torch.zeros(num_graphs, device=device), torch.zeros(num_graphs, device=device)
        target_flat = self._target_positive_mask(target)
        batch_flat = batch.flatten().to(torch.long)
        positive = target_flat.bool()
        positive_counts = scatter(positive.float(), batch_flat, dim=0, dim_size=num_graphs, reduce="sum")
        negative_counts = scatter((~positive).float(), batch_flat, dim=0, dim_size=num_graphs, reduce="sum")
        return positive_counts, negative_counts

    def _log_reco_target_stats(self, stage: str, batch, batch_hyperedge, on_step: bool, on_epoch: bool):
        num_graphs = int(getattr(batch, "num_graphs", len(batch)))
        edge_pos, edge_neg = self._positive_negative_counts(
            batch.edge_attr_t,
            batch.edge_attr_batch,
            num_graphs,
        )
        hyper_pos, hyper_neg = self._positive_negative_counts(
            batch.hyperedge_attr_t,
            batch_hyperedge,
            num_graphs,
        )
        active_edge = edge_pos > 0
        active_hyper = hyper_pos > 0
        self.log(f"diagnostics/{stage}_edge_positive_mean", edge_pos.float().mean(), batch_size=len(batch), on_step=on_step, on_epoch=on_epoch, logger=True, sync_dist=True)
        self.log(f"diagnostics/{stage}_edge_negative_mean", edge_neg.float().mean(), batch_size=len(batch), on_step=on_step, on_epoch=on_epoch, logger=True, sync_dist=True)
        self.log(f"diagnostics/{stage}_hyperedge_positive_mean", hyper_pos.float().mean(), batch_size=len(batch), on_step=on_step, on_epoch=on_epoch, logger=True, sync_dist=True)
        self.log(f"diagnostics/{stage}_hyperedge_negative_mean", hyper_neg.float().mean(), batch_size=len(batch), on_step=on_step, on_epoch=on_epoch, logger=True, sync_dist=True)
        self.log(f"diagnostics/{stage}_edge_reco_active_fraction", active_edge.float().mean(), batch_size=len(batch), on_step=on_step, on_epoch=on_epoch, logger=True, sync_dist=True)
        self.log(f"diagnostics/{stage}_hyperedge_reco_active_fraction", active_hyper.float().mean(), batch_size=len(batch), on_step=on_step, on_epoch=on_epoch, logger=True, sync_dist=True)

    @staticmethod
    def _safe_mean(values: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return fallback.new_tensor(0.0)
        return values.mean()

    @staticmethod
    def _align_graph_mask(mask: torch.Tensor, size: int, device: torch.device) -> torch.Tensor:
        mask = mask.flatten().bool().to(device)
        if mask.numel() == size:
            return mask
        if mask.numel() < size:
            pad = torch.zeros(size - mask.numel(), dtype=torch.bool, device=device)
            return torch.cat([mask, pad], dim=0)
        return mask[:size]

    @staticmethod
    def _positive_recall(pred: torch.Tensor, target: torch.Tensor, threshold: float, fallback: torch.Tensor) -> torch.Tensor:
        positive = target > 0.5
        if not positive.any():
            return fallback.new_tensor(0.0)
        return (pred[positive] >= float(threshold)).float().mean()

    @staticmethod
    def _positive_topk_fraction(
        pred: torch.Tensor,
        target: torch.Tensor,
        candidate_batch: torch.Tensor,
        active_graph_mask: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        if pred.numel() == 0 or target.numel() == 0 or candidate_batch.numel() == 0:
            return pred.new_tensor(0.0)

        pred = pred.flatten()
        target = target.flatten()
        candidate_batch = candidate_batch.flatten().to(torch.long)
        active_graph_mask = active_graph_mask.flatten().bool().to(pred.device)
        active_graph_ids = torch.nonzero(active_graph_mask, as_tuple=False).flatten()
        if active_graph_ids.numel() == 0:
            return pred.new_tensor(0.0)

        hits = []
        for graph_id in active_graph_ids.tolist():
            mask = candidate_batch == int(graph_id)
            if not mask.any():
                continue
            event_target = target[mask]
            if not (event_target > 0.5).any():
                continue
            event_pred = pred[mask]
            order = torch.argsort(event_pred, descending=True)
            top_target = event_target[order[: min(int(k), order.numel())]]
            hits.append((top_target > 0.5).any().float())
        if not hits:
            return pred.new_tensor(0.0)
        return torch.stack(hits).mean()

    def _log_reco_score_metrics(
        self,
        stage: str,
        edge_pred: torch.Tensor,
        edge_target: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_active_mask: torch.Tensor,
        hyper_pred: torch.Tensor,
        hyper_target: torch.Tensor,
        hyper_batch: torch.Tensor,
        hyper_active_mask: torch.Tensor,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        metric_specs = (
            ("edge", edge_pred, edge_target, edge_batch, edge_active_mask),
            ("hyperedge", hyper_pred, hyper_target, hyper_batch, hyper_active_mask),
        )
        for name, pred, target, candidate_batch, active_mask in metric_specs:
            pred = pred.flatten()
            target = self._target_positive_mask(target).float().flatten()
            candidate_batch = candidate_batch.flatten().to(torch.long)
            active_size = int(candidate_batch.max().item()) + 1 if candidate_batch.numel() else 0
            active_mask = self._align_graph_mask(active_mask, active_size, pred.device)
            keep = active_mask[candidate_batch] if active_mask.numel() else torch.zeros_like(candidate_batch, dtype=torch.bool)
            pred_keep = pred[keep]
            target_keep = target[keep]
            pos_scores = pred_keep[target_keep > 0.5]
            neg_scores = pred_keep[target_keep <= 0.5]
            pos_mean = self._safe_mean(pos_scores, pred)
            neg_mean = self._safe_mean(neg_scores, pred)

            self.log(f"metrics/{stage}_{name}_positive_mean_score", pos_mean, batch_size=len(candidate_batch), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"metrics/{stage}_{name}_negative_mean_score", neg_mean, batch_size=len(candidate_batch), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"metrics/{stage}_{name}_score_gap", pos_mean - neg_mean, batch_size=len(candidate_batch), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=True)
            for threshold in (0.1, 0.5):
                self.log(
                    f"metrics/{stage}_{name}_positive_recall_at_{str(threshold).replace('.', 'p')}",
                    self._positive_recall(pred_keep, target_keep, threshold, pred),
                    batch_size=len(candidate_batch),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
            if stage == "validation":
                for k in (1, 2, 5):
                    self.log(
                        f"metrics/{stage}_{name}_positive_in_top{k}",
                        self._positive_topk_fraction(pred, target, candidate_batch, active_mask, k),
                        batch_size=len(candidate_batch),
                        on_step=on_step,
                        on_epoch=on_epoch,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
                    )

    def configure_optimizers(self):
        optimizer_name = str(self.hparams.optimizer).lower()
        lr = float(self.hparams.lr)
        weight_decay = float(self.hparams.weight_decay or 0.0)
        print(f"Configuring optimizer: name={optimizer_name}, lr={lr}, weight_decay={weight_decay}")
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters are available for the optimizer.")
        trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
        total_count = sum(parameter.numel() for parameter in self.parameters())
        print(f"Optimizer parameter count: trainable={trainable_count}, total={total_count}")

        if optimizer_name == 'adam':
            optimizer = Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Supported optimizers are: adam, adamw.")

        if not bool(self.hparams.lr_scheduler_enabled):
            return optimizer

        scheduler_method = str(self.hparams.lr_scheduler_method).lower()
        if scheduler_method not in ('reduce_on_plateau', 'reducelronplateau'):
            raise ValueError("Supported lr_scheduler methods are: reduce_on_plateau.")

        monitor = str(self.hparams.lr_scheduler_monitor or "val_loss")
        print(
            "Configuring lr scheduler: "
            f"method=reduce_on_plateau, monitor={monitor}, mode={self.hparams.lr_scheduler_mode}, "
            f"factor={self.hparams.lr_scheduler_factor}, patience={self.hparams.lr_scheduler_patience}, "
            f"min_lr={self.hparams.lr_scheduler_min_lr}"
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=str(self.hparams.lr_scheduler_mode),
                    factor=float(self.hparams.lr_scheduler_factor),
                    patience=int(self.hparams.lr_scheduler_patience),
                    min_lr=float(self.hparams.lr_scheduler_min_lr),
                ),
                "interval": "epoch",
                "monitor": monitor,
                "frequency": int(self.hparams.lr_scheduler_frequency),
                "strict": False,
            },
        }

    def training_step(self, train_batch, batch_idx):
        x_hat, batch_hyperedge, edge_attr_prime, x_class = self._shared_step(train_batch)

        cls_t = self._require_class_targets(train_batch, "training") if self.classification_enabled else None
        loss_class = (
            ClassificationLoss(x_class, cls_t, criterion=BCELoss(reduction='none'))
            if self.classification_enabled else None
        )

        if self.reconstruction_enabled:
            criterion_edge = BCELoss(reduction='none')
            criterion_hyperedge = BCELoss(reduction='none')

            loss_edge,loss_edge_masks = EdgeLoss(edge_attr_prime, train_batch.edge_attr_t, train_batch.edge_attr_batch, criterion=criterion_edge, **self._reco_loss_kwargs())
            loss_hyperedge, loss_hyperedge_masks = HyperedgeLoss(x_hat, train_batch.hyperedge_attr_t.float(), batch_hyperedge, criterion_hyperedge, **self._reco_loss_kwargs())
            self._log_reco_target_stats("train", train_batch, batch_hyperedge, on_step=True, on_epoch=True)
            self._log_reco_score_metrics(
                "train",
                self._reco_scalar_score(edge_attr_prime),
                train_batch.edge_attr_t,
                train_batch.edge_attr_batch,
                loss_edge_masks,
                self._reco_scalar_score(x_hat),
                train_batch.hyperedge_attr_t,
                batch_hyperedge,
                loss_hyperedge_masks,
                on_step=True,
                on_epoch=True,
            )
        else:
            loss_edge = edge_attr_prime.new_zeros(1)
            loss_hyperedge = x_hat.new_zeros(1)
            loss_edge_masks = torch.zeros(1, dtype=torch.bool, device=edge_attr_prime.device)
            loss_hyperedge_masks = torch.zeros(1, dtype=torch.bool, device=x_hat.device)

        loss = self._combined_loss(loss_hyperedge, loss_edge, loss_class, cls_t, loss_hyperedge_masks, loss_edge_masks)
        if not loss.requires_grad:
            loss = loss + (edge_attr_prime.sum() + x_hat.sum()) * 0.0

        # Logging
        self.log('loss/train_loss', loss, batch_size=len(train_batch), on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if loss_class is not None and loss_class.numel() > 0:
            self.log('loss/train_classification_loss', loss_class.mean(), batch_size=len(train_batch), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x_hat, batch_hyperedge, edge_attr_prime, x_class = self._shared_step(val_batch)
        fast_validation = str(self.hparams.validation_mode).lower() == 'fast'
        val_on_step = not fast_validation

        cls_t = self._require_class_targets(val_batch, "validation") if self.classification_enabled else None
        loss_class = (
            ClassificationLoss(x_class, cls_t, criterion=BCELoss(reduction='none'))
            if self.classification_enabled else None
        )

        if self.reconstruction_enabled:
            criterion_edge = BCELoss(reduction='none')
            criterion_hyperedge = BCELoss(reduction='none')

            loss_edge,loss_edge_masks = EdgeLoss(edge_attr_prime, val_batch.edge_attr_t, val_batch.edge_attr_batch, criterion=criterion_edge, **self._reco_loss_kwargs())
            loss_hyperedge, loss_hyperedge_masks = HyperedgeLoss(x_hat, val_batch.hyperedge_attr_t.float(), batch_hyperedge, criterion_hyperedge, **self._reco_loss_kwargs())
            self._log_reco_target_stats("validation", val_batch, batch_hyperedge, on_step=val_on_step, on_epoch=True)
            self._log_reco_score_metrics(
                "validation",
                self._reco_scalar_score(edge_attr_prime),
                val_batch.edge_attr_t,
                val_batch.edge_attr_batch,
                loss_edge_masks,
                self._reco_scalar_score(x_hat),
                val_batch.hyperedge_attr_t,
                batch_hyperedge,
                loss_hyperedge_masks,
                on_step=val_on_step,
                on_epoch=True,
            )
        else:
            loss_edge = edge_attr_prime.new_zeros(1)
            loss_hyperedge = x_hat.new_zeros(1)
            loss_edge_masks = torch.zeros(1, dtype=torch.bool, device=edge_attr_prime.device)
            loss_hyperedge_masks = torch.zeros(1, dtype=torch.bool, device=x_hat.device)

        loss = self._combined_loss(loss_hyperedge, loss_edge, loss_class, cls_t, loss_hyperedge_masks, loss_edge_masks)
        if not loss.requires_grad:
            loss = loss + (edge_attr_prime.sum() + x_hat.sum()) * 0.0

        # Logging
        self.log('val_loss', loss, batch_size=len(val_batch), on_step=val_on_step, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('loss/validation_loss', loss, batch_size=len(val_batch), on_step=val_on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        if loss_class is not None and loss_class.numel() > 0:
            self.log('loss/validation_classification_loss', loss_class.mean(), batch_size=len(val_batch), on_step=val_on_step, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        if self.reconstruction_enabled:
            # Validation Accuracy Calculation

            # Not considering bkg events for the reco accuracy
            # if cls_t is not None:
            #     graph_mask = cls_t.view(-1) == 1
            # else:
            #     graph_mask = torch.ones(int(val_batch.num_graphs), dtype=torch.bool, device=edge_attr_prime.device)
            # ge_keep  = graph_mask[val_batch.edge_attr_batch]
            edge_reco_mask = loss_edge_masks.to(edge_attr_prime.device)
            num_graphs = int(getattr(val_batch, "num_graphs", len(val_batch)))
            edge_reco_mask = self._align_graph_mask(edge_reco_mask, num_graphs, edge_attr_prime.device)
            ge_keep = edge_reco_mask[val_batch.edge_attr_batch]
            accuracy_edge  = BinaryAccuracy(ignore_index=0).to(edge_attr_prime)

            pred_edge = self._reco_scalar_score(edge_attr_prime).flatten()[ge_keep]
            tgt_edge = self._target_positive_mask(val_batch.edge_attr_t).float().flatten()[ge_keep]

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
        x_score = self._reco_scalar_score(x_hat)
        edge_score = self._reco_scalar_score(edge_attr_prime)
        x_probs = torch.softmax(x_hat, dim=1) if self.target_encoding == "typed" else None
        edge_probs = torch.softmax(edge_attr_prime, dim=1) if self.target_encoding == "typed" else None

        # Unbatch results
        x_out = unbatch(x_score, batch_hyperedge.type(torch.int64), 0)
        edge_attr_out = unbatch(edge_score, batch.edge_attr_batch, 0)
        x_probs_out = unbatch(x_probs, batch_hyperedge.type(torch.int64), 0) if x_probs is not None else None
        edge_probs_out = unbatch(edge_probs, batch.edge_attr_batch, 0) if edge_probs is not None else None
        N_nodes       = degree(batch.batch).cpu().flatten().tolist()
        encodings     = unbatch(batch.x[:,-1].reshape(-1,1),batch.batch, 0)
        x_class_out = x_class.squeeze(-1) if x_class is not None else None
        cls_t = self._class_targets(batch)
        cls_t_out = cls_t.squeeze(-1) if cls_t is not None else None
        # print("PREDICTION STEP OUTPUTS")
        # print("x_out:", x_out)
        # print("edge_attr_out:", edge_attr_out)
        # print("N_nodes:", N_nodes)
        # print("encodings:", encodings)
        # print("x_class_out:", x_class_out)
        return x_out, edge_attr_out, N_nodes, encodings, x_class_out, x_probs_out, edge_probs_out, cls_t_out
