"""HyPER shared graph backbone with explicit reconstruction/classification tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch.optim import Adam, AdamW, lr_scheduler
from torch_geometric.utils import degree, unbatch
from torchmetrics.functional.classification import binary_auroc

from .mpnn import MessagePassingBlock
from .classification import ClassificationHead, pool_event_representation
from .hyperedge import HyperedgeModel
from .loss import (
    additive_total_loss,
    classification_losses,
    combine_reconstruction_activity,
    resolve_class_weights,
    typed_reconstruction_loss,
)


class HyPERModel(LightningModule):
    """Shared HyPER embeddings with mode-specific output heads and losses."""

    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        global_in_channels: int,
        edge_out_channels: int,
        hyperedge_out_channels: int,
        edge_class_names: Sequence[str],
        hyperedge_class_names: Sequence[str],
        message_feats: int = 32,
        dropout: float = 0.01,
        num_message_passing_layers: int = 3,
        contraction_feats: int = 32,
        hyperedge_order: int = 3,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_enabled: bool = True,
        lr_scheduler_method: str = "reduce_on_plateau",
        lr_scheduler_monitor: str = "val_loss",
        lr_scheduler_mode: str = "min",
        lr_scheduler_factor: float = 0.8,
        lr_scheduler_patience: int = 10,
        lr_scheduler_min_lr: float = 0.0,
        lr_scheduler_frequency: int = 1,
        classification_enabled: bool = True,
        reconstruction_enabled: bool = True,
        edge_class_weights: Mapping[str, float] | None = None,
        hyperedge_class_weights: Mapping[str, float] | None = None,
        edge_weight: float = 0.475,
        hyperedge_weight: float = 0.475,
        classification_weight: float = 0.05,
        classification_pos_weight: float | None = None,
        log_metrics_to_logger: bool = True,
        validation_subset_path: str | None = None,
        validation_subset_hash: str | None = None,
        validation_role_ranking_enabled: bool = False,
        validation_classification_metrics_enabled: bool = False,
        validation_diagnostics_every_n_epochs: int = 1,
        validation_diagnostics_max_events: int | None = None,
        validate_candidate_event_assignment: bool = False,
        optimizer_foreach: bool | None = None,
        optimizer_fused: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.classification_enabled = bool(classification_enabled)
        self.reconstruction_enabled = bool(reconstruction_enabled)
        self.log_metrics_to_logger = bool(log_metrics_to_logger)
        self.validation_role_ranking_enabled = bool(validation_role_ranking_enabled)
        self.validation_classification_metrics_enabled = bool(
            validation_classification_metrics_enabled
        )
        self.validation_diagnostics_every_n_epochs = int(validation_diagnostics_every_n_epochs)
        if self.validation_diagnostics_every_n_epochs <= 0:
            raise ValueError("validation_diagnostics_every_n_epochs must be positive.")
        self.validation_diagnostics_max_events = (
            None
            if validation_diagnostics_max_events is None
            else int(validation_diagnostics_max_events)
        )
        if (
            self.validation_diagnostics_max_events is not None
            and self.validation_diagnostics_max_events <= 0
        ):
            raise ValueError("validation_diagnostics_max_events must be positive when set.")
        self.validate_candidate_event_assignment = bool(validate_candidate_event_assignment)
        self.runtime_training_shapes = {"events": 0, "nodes": 0, "directed_edges": 0, "hyperedges": 0}
        self.gradients_finite = True
        if not self.classification_enabled and not self.reconstruction_enabled:
            raise ValueError("At least one of classification or reconstruction must be enabled.")
        if int(num_message_passing_layers) <= 0:
            raise ValueError("model.num_message_passing_layers must be positive.")

        self.edge_class_names = [str(name) for name in edge_class_names]
        self.hyperedge_class_names = [str(name) for name in hyperedge_class_names]
        if int(edge_out_channels) != len(self.edge_class_names):
            raise ValueError("edge_out_channels must equal len(edge_class_names).")
        if int(hyperedge_out_channels) != len(self.hyperedge_class_names):
            raise ValueError("hyperedge_out_channels must equal len(hyperedge_class_names).")
        self.register_buffer(
            "edge_class_weight_tensor",
            resolve_class_weights(self.edge_class_names, edge_class_weights),
            persistent=True,
        )
        self.register_buffer(
            "hyperedge_class_weight_tensor",
            resolve_class_weights(self.hyperedge_class_names, hyperedge_class_weights),
            persistent=True,
        )

        layers = []
        for layer_index in range(int(num_message_passing_layers)):
            first = layer_index == 0
            layers.append(
                MessagePassingBlock(
                    node_in_channels if first else message_feats,
                    edge_in_channels if first else message_feats,
                    global_in_channels if first else message_feats,
                    node_out_channels=message_feats,
                    edge_out_channels=message_feats,
                    global_out_channels=message_feats,
                    message_feats=message_feats,
                    dropout=dropout,
                )
            )
        self.message_passing_layers = ModuleList(layers)
        self.Hyperedge = HyperedgeModel(
            node_in_channels=message_feats,
            node_out_channels=1,
            global_in_channels=message_feats,
            message_feats=contraction_feats,
            dropout=dropout,
        )
        self.Classification = (
            ClassificationHead(
                n_feats_out=1,
                contraction_feats=contraction_feats,
                message_feats=message_feats,
                dropout=dropout,
            )
            if self.classification_enabled
            else None
        )
        self.ge_head = Linear(message_feats, edge_out_channels) if self.reconstruction_enabled else None
        self.he_head = Linear(contraction_feats, hyperedge_out_channels) if self.reconstruction_enabled else None

        self._val_logits: list[Tensor] | None = None
        self._val_targets: list[Tensor] | None = None
        self._val_reco: dict | None = None

    @staticmethod
    def _empty_reco_accumulator() -> dict:
        return {
            "edge": {},
            "hyperedge": {},
            "active": 0,
            "active_signal": 0,
            "active_background": 0,
            "signal": 0,
            "background": 0,
            "events": 0,
        }

    def _candidate_event_batch(self, node_batch: Tensor, candidate_index: Tensor, name: str) -> Tensor:
        """Derive candidate-to-event assignments from node memberships."""
        if candidate_index.ndim != 2:
            raise ValueError(f"{name} index must be two-dimensional, got {tuple(candidate_index.shape)}.")
        if candidate_index.size(1) == 0:
            return node_batch.new_empty((0,), dtype=torch.long)
        if name == "edge":
            anchor_row = 1
        elif name == "hyperedge":
            anchor_row = 0
        else:
            raise ValueError(f"Unknown candidate kind {name!r}; expected 'edge' or 'hyperedge'.")
        if candidate_index.size(0) <= anchor_row:
            raise ValueError(
                f"{name} index has {candidate_index.size(0)} member rows; expected more than {anchor_row}."
            )
        candidate_batch = node_batch[candidate_index[anchor_row]].to(dtype=torch.long)
        if self.validate_candidate_event_assignment:
            member_batches = node_batch[candidate_index]
            if not torch.all(member_batches == candidate_batch.unsqueeze(0)):
                raise RuntimeError(f"{name} contains nodes from different events.")
        return candidate_batch

    def _diagnostics_due(self) -> bool:
        return int(self.current_epoch) % self.validation_diagnostics_every_n_epochs == 0

    def _remaining_diagnostic_events(self, num_events: int) -> int:
        if self.validation_diagnostics_max_events is None:
            return int(num_events)
        return max(
            0,
            min(
                int(num_events),
                self.validation_diagnostics_max_events - self._diagnostic_events_seen,
            ),
        )

    def freeze_for_probe(self, trainable_prefixes=("Classification.",)):
        prefixes = tuple(str(prefix) for prefix in trainable_prefixes)
        trainable, frozen = [], []
        for name, parameter in self.named_parameters():
            parameter.requires_grad = any(name.startswith(prefix) for prefix in prefixes)
            (trainable if parameter.requires_grad else frozen).append(name)
        if not trainable or not frozen:
            raise RuntimeError(
                f"Invalid frozen-probe parameter selection for prefixes {prefixes}: "
                f"trainable={len(trainable)}, frozen={len(frozen)}."
            )
        return trainable, frozen

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        u,
        batch,
        hyperedge_index,
        hyperedge_index_batch=None,
        return_representations: bool = False,
    ):
        x_embed, edge_embed, u_embed = x, edge_attr, u
        block_event_representations = []
        for layer in self.message_passing_layers:
            x_embed, edge_embed, u_embed = layer(
                x_embed, edge_index, edge_embed, u_embed, batch
            )
            if return_representations:
                block_event_representations.append(u_embed)
        if hyperedge_index_batch is None:
            hyperedge_batch = self._candidate_event_batch(
                batch, hyperedge_index, "hyperedge"
            )
        else:
            hyperedge_batch = hyperedge_index_batch.reshape(-1).to(
                device=batch.device, dtype=torch.long
            )
            if self.validate_candidate_event_assignment:
                derived_hyper_batch = self._candidate_event_batch(
                    batch, hyperedge_index, "hyperedge"
                )
                if not torch.equal(hyperedge_batch, derived_hyper_batch):
                    raise RuntimeError(
                        "Supplied hyperedge event assignments disagree with node-derived assignments."
                    )
        hyper_embed, hyper_batch = self.Hyperedge(
            x_embed,
            u_embed,
            batch,
            hyperedge_index,
            hyperedge_batch,
            self.hparams.hyperedge_order,
        )
        edge_batch = self._candidate_event_batch(batch, edge_index, "edge")
        event_representation = None
        if self.classification_enabled or return_representations:
            event_representation = pool_event_representation(
                hyper_embed, edge_embed, x_embed, u_embed,
                edge_batch, hyper_batch, batch,
            )
        cls_logits = None
        if self.classification_enabled:
            cls_logits = self.Classification.mlp_class(event_representation)
        edge_logits = self.ge_head(edge_embed) if self.reconstruction_enabled else None
        hyper_logits = self.he_head(hyper_embed) if self.reconstruction_enabled else None
        outputs = (hyper_logits, hyper_batch, edge_logits, cls_logits)
        if not return_representations:
            return outputs
        representations = {
            "final_event": event_representation,
            "classification_head_input": event_representation,
        }
        representations.update(
            {f"block_{index}": value for index, value in enumerate(block_event_representations)}
        )
        return outputs, representations

    def _shared_step(self, data):
        return self.forward(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.u,
            data.batch,
            data.hyperedge_index,
        )

    @staticmethod
    def _class_targets(batch) -> Tensor | None:
        target = getattr(batch, "cls_t", None)
        if target is None or target.numel() == 0:
            return None
        return target.float().reshape(-1)

    def _loss_components(self, batch, outputs):
        hyper_logits, hyper_batch, edge_logits, cls_logits = outputs
        num_events = int(batch.num_graphs)
        cls_target = self._class_targets(batch)
        edge_loss = hyper_loss = cls_loss = cls_unweighted = None
        edge_per_event = hyper_per_event = None
        edge_active = hyper_active = None
        signal_count = background_count = None

        if self.reconstruction_enabled:
            required = (
                "edge_attr_t_class",
                "hyperedge_attr_t_class",
                "edge_reco_active",
                "hyperedge_reco_active",
            )
            missing = [name for name in required if getattr(batch, name, None) is None]
            if missing:
                raise RuntimeError(
                    "Typed reconstruction batches require cached integer targets and "
                    f"activity masks; missing={missing}. Rebuild or validate the graph DB."
                )
            edge_batch = self._candidate_event_batch(batch.batch, batch.edge_index, "edge")
            edge_loss, edge_active, edge_per_event = typed_reconstruction_loss(
                edge_logits,
                None,
                edge_batch,
                self.edge_class_names,
                self.edge_class_weight_tensor,
                num_events=num_events,
                target_class=batch.edge_attr_t_class,
                active_event_mask=batch.edge_reco_active,
                validate_cached_targets=False,
                validate_candidate_batch=False,
                validate_class_weights=False,
            )
            hyper_loss, hyper_active, hyper_per_event = typed_reconstruction_loss(
                hyper_logits,
                None,
                hyper_batch,
                self.hyperedge_class_names,
                self.hyperedge_class_weight_tensor,
                num_events=num_events,
                target_class=batch.hyperedge_attr_t_class,
                active_event_mask=batch.hyperedge_reco_active,
                validate_cached_targets=False,
                validate_candidate_batch=False,
                validate_class_weights=False,
            )
        if self.classification_enabled:
            if cls_target is None:
                raise RuntimeError("Classification is enabled but the batch has no explicit cls_t labels.")
            cls_loss, cls_unweighted, signal_count, background_count = classification_losses(
                cls_logits,
                cls_target,
                self.hparams.classification_pos_weight,
            )
        total = additive_total_loss(
            edge_loss=edge_loss,
            hyperedge_loss=hyper_loss,
            classification_loss=cls_loss,
            edge_weight=self.hparams.edge_weight,
            hyperedge_weight=self.hparams.hyperedge_weight,
            classification_weight=self.hparams.classification_weight,
        )
        return {
            "total": total,
            "edge": edge_loss,
            "hyperedge": hyper_loss,
            "classification": cls_loss,
            "classification_unweighted": cls_unweighted,
            "edge_active": edge_active,
            "hyperedge_active": hyper_active,
            "edge_per_event": edge_per_event,
            "hyperedge_per_event": hyper_per_event,
            "signal_count": signal_count,
            "background_count": background_count,
            "cls_target": cls_target,
        }

    def _log_losses(self, stage: str, losses: dict, num_events: int, on_step: bool) -> None:
        epoch_common = dict(
            batch_size=num_events,
            on_step=False,
            on_epoch=True,
            logger=self.log_metrics_to_logger,
            sync_dist=self._sync_dist(),
        )
        total_name = "loss/validation_total_batch" if stage == "validation" else "loss/train_loss"
        self.log(
            total_name,
            losses["total"],
            batch_size=num_events,
            on_step=bool(on_step),
            on_epoch=True,
            logger=self.log_metrics_to_logger,
            sync_dist=self._sync_dist(),
            prog_bar=True,
        )
        for name in ("edge", "hyperedge", "classification", "classification_unweighted"):
            value = losses[name]
            if value is not None:
                self.log(f"loss/{stage}_{name}", value, prog_bar=False, **epoch_common)
        for name in ("edge_active", "hyperedge_active"):
            value = losses[name]
            if value is not None:
                self.log(
                    f"metrics/{stage}_{name}_fraction",
                    value.float().mean(),
                    prog_bar=False,
                    **epoch_common,
                )
        if losses["edge_active"] is not None and losses["hyperedge_active"] is not None:
            edge_active = losses["edge_active"]
            hyperedge_active = losses["hyperedge_active"]
            reconstruction_active = combine_reconstruction_activity(edge_active, hyperedge_active)
            diagnostics = {
                "edge_active_events": edge_active.sum(),
                "hyperedge_active_events": hyperedge_active.sum(),
                "reconstruction_active_events": reconstruction_active.sum(),
            }
            labels = losses.get("cls_target")
            if labels is not None:
                labels = labels.to(edge_active.device).reshape(-1)
                diagnostics.update(
                    {
                        "edge_active_signal_events": (edge_active & (labels == 1)).sum(),
                        "edge_active_background_events": (edge_active & (labels == 0)).sum(),
                        "hyperedge_active_signal_events": (hyperedge_active & (labels == 1)).sum(),
                        "hyperedge_active_background_events": (hyperedge_active & (labels == 0)).sum(),
                    }
                )
            for metric_name, value in diagnostics.items():
                self.log(
                    metric_name if stage == "validation" else f"{stage}_{metric_name}",
                    value.float(),
                    on_step=False,
                    on_epoch=True,
                    logger=self.log_metrics_to_logger,
                    sync_dist=self._sync_dist(),
                    reduce_fx="sum",
                )
        for name in ("signal_count", "background_count"):
            value = losses[name]
            if value is not None:
                self.log(
                    f"metrics/{stage}_{name}",
                    value.float(),
                    on_step=False,
                    on_epoch=True,
                    logger=self.log_metrics_to_logger,
                    sync_dist=self._sync_dist(),
                    reduce_fx="sum",
                )

    def training_step(self, batch, batch_idx):
        self.runtime_training_shapes["events"] += int(batch.num_graphs)
        self.runtime_training_shapes["nodes"] += int(batch.num_nodes)
        self.runtime_training_shapes["directed_edges"] += int(batch.edge_index.size(1))
        self.runtime_training_shapes["hyperedges"] += int(batch.hyperedge_index.size(1))
        outputs = self._shared_step(batch)
        losses = self._loss_components(batch, outputs)
        self._log_losses("train", losses, int(batch.num_graphs), on_step=True)
        return losses["total"]

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        if gradient_clip_algorithm not in (None, "norm"):
            raise ValueError(
                "HyPER supports norm-based gradient clipping only so the finite-gradient "
                "check and clipping share one parameter traversal."
            )
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not parameters:
            return
        max_norm = float("inf") if gradient_clip_val in (None, 0, 0.0) else float(gradient_clip_val)
        try:
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=max_norm,
                error_if_nonfinite=True,
                foreach=True,
            )
        except RuntimeError as exc:
            self.gradients_finite = False
            raise FloatingPointError(
                "Non-finite gradient detected during HyPER training."
            ) from exc

    def on_validation_epoch_start(self) -> None:
        diagnostics_due = self._diagnostics_due()
        self._collect_classification_diagnostics = bool(
            diagnostics_due
            and self.classification_enabled
            and self.validation_classification_metrics_enabled
        )
        self._collect_role_diagnostics = bool(
            diagnostics_due
            and self.reconstruction_enabled
            and self.validation_role_ranking_enabled
        )
        self._diagnostic_events_seen = 0
        self._val_logits = [] if self._collect_classification_diagnostics else None
        self._val_targets = [] if self._collect_classification_diagnostics else None
        self._val_reco = self._empty_reco_accumulator() if self._collect_role_diagnostics else None
        self._val_loss_accumulator = {
            "edge_sum": None, "edge_count": 0,
            "hyperedge_sum": None, "hyperedge_count": 0,
            "classification_sum": None, "classification_unweighted_sum": None,
            "classification_count": 0,
        }

    @staticmethod
    def _update_role_ranking(store: dict, logits, target_class, candidate_batch, names) -> None:
        probabilities = torch.softmax(logits.detach().float(), dim=1).cpu()
        truth = target_class.detach().long().reshape(-1).cpu()
        batches = candidate_batch.detach().long().cpu()
        for role, name in enumerate(names[:-1]):
            stats = store.setdefault(name, {"count": 0, "top1": 0, "top2": 0, "top5": 0, "rr": 0.0})
            for event in torch.unique(batches).tolist():
                keep = batches == int(event)
                positive = truth[keep] == role
                if not positive.any():
                    continue
                scores = probabilities[keep, role]
                order = torch.argsort(scores, descending=True)
                ranked_truth = positive[order]
                first = int(torch.nonzero(ranked_truth, as_tuple=False)[0]) + 1
                stats["count"] += 1
                stats["top1"] += int(first <= 1)
                stats["top2"] += int(first <= 2)
                stats["top5"] += int(first <= 5)
                stats["rr"] += 1.0 / first

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        losses = self._loss_components(batch, outputs)
        num_events = int(batch.num_graphs)
        self._log_losses("validation", losses, num_events, on_step=False)
        hyper_logits, hyper_batch, edge_logits, cls_logits = outputs

        accumulator = self._val_loss_accumulator
        if losses["edge_active"] is not None:
            active = losses["edge_active"]
            value = losses["edge_per_event"][active].sum().detach()
            accumulator["edge_sum"] = value if accumulator["edge_sum"] is None else accumulator["edge_sum"] + value
            accumulator["edge_count"] += int(active.sum())
        if losses["hyperedge_active"] is not None:
            active = losses["hyperedge_active"]
            value = losses["hyperedge_per_event"][active].sum().detach()
            accumulator["hyperedge_sum"] = value if accumulator["hyperedge_sum"] is None else accumulator["hyperedge_sum"] + value
            accumulator["hyperedge_count"] += int(active.sum())
        if losses["classification"] is not None:
            count = num_events
            weighted = losses["classification"].detach() * count
            raw = losses["classification_unweighted"].detach() * count
            accumulator["classification_sum"] = weighted if accumulator["classification_sum"] is None else accumulator["classification_sum"] + weighted
            accumulator["classification_unweighted_sum"] = raw if accumulator["classification_unweighted_sum"] is None else accumulator["classification_unweighted_sum"] + raw
            accumulator["classification_count"] += count

        cls_target = self._class_targets(batch)
        diagnostic_events = (
            self._remaining_diagnostic_events(num_events)
            if (self._collect_classification_diagnostics or self._collect_role_diagnostics)
            else 0
        )
        if self._collect_classification_diagnostics and diagnostic_events:
            assert self._val_logits is not None and self._val_targets is not None
            self._val_logits.append(
                cls_logits.detach().reshape(-1)[:diagnostic_events].cpu()
            )
            self._val_targets.append(
                cls_target.detach().reshape(-1)[:diagnostic_events].cpu()
            )
        if self._collect_role_diagnostics and diagnostic_events:
            assert self._val_reco is not None
            edge_batch = self._candidate_event_batch(batch.batch, batch.edge_index, "edge")
            edge_keep = edge_batch < diagnostic_events
            hyper_keep = hyper_batch < diagnostic_events
            self._update_role_ranking(
                self._val_reco["edge"],
                edge_logits[edge_keep],
                batch.edge_attr_t_class[edge_keep],
                edge_batch[edge_keep],
                self.edge_class_names,
            )
            self._update_role_ranking(
                self._val_reco["hyperedge"],
                hyper_logits[hyper_keep],
                batch.hyperedge_attr_t_class[hyper_keep],
                hyper_batch[hyper_keep],
                self.hyperedge_class_names,
            )
            active = combine_reconstruction_activity(
                losses["edge_active"], losses["hyperedge_active"]
            )[:diagnostic_events]
            labels = (
                cls_target[:diagnostic_events]
                if cls_target is not None
                else torch.zeros(diagnostic_events, device=active.device)
            )
            active_cpu = active.detach().cpu()
            labels_cpu = labels.detach().cpu()
            self._val_reco["events"] += diagnostic_events
            self._val_reco["active"] += int(active_cpu.sum())
            self._val_reco["signal"] += int((labels_cpu == 1).sum())
            self._val_reco["background"] += int((labels_cpu == 0).sum())
            self._val_reco["active_signal"] += int(
                (active_cpu & (labels_cpu == 1)).sum()
            )
            self._val_reco["active_background"] += int(
                (active_cpu & (labels_cpu == 0)).sum()
            )
        self._diagnostic_events_seen += diagnostic_events
        return losses["total"]

    def test_step(self, batch, batch_idx):
        losses = self._loss_components(batch, self._shared_step(batch))
        common = dict(
            batch_size=int(batch.num_graphs), on_step=False, on_epoch=True,
            logger=self.log_metrics_to_logger, sync_dist=self._sync_dist(),
        )
        self.log("test_loss", losses["total"], **common)
        if losses["edge"] is not None:
            self.log("test_edge_loss", losses["edge"], **common)
        if losses["hyperedge"] is not None:
            self.log("test_hyperedge_loss", losses["hyperedge"], **common)
        if losses["edge"] is not None or losses["hyperedge"] is not None:
            reconstruction = losses["total"].new_zeros(())
            if losses["edge"] is not None:
                reconstruction = reconstruction + self.hparams.edge_weight * losses["edge"]
            if losses["hyperedge"] is not None:
                reconstruction = reconstruction + self.hparams.hyperedge_weight * losses["hyperedge"]
            self.log("test_reconstruction_loss", reconstruction, **common)
        if losses["classification_unweighted"] is not None:
            self.log("test_classification_loss", losses["classification_unweighted"], **common)
        return losses["total"]

    @staticmethod
    def _background_rejection_at_efficiency(labels: Tensor, probabilities: Tensor, efficiency: float) -> Tensor:
        signal_scores = probabilities[labels == 1]
        background_scores = probabilities[labels == 0]
        if signal_scores.numel() == 0 or background_scores.numel() == 0:
            return probabilities.new_tensor(float("nan"))
        threshold = torch.quantile(signal_scores, 1.0 - float(efficiency))
        background_efficiency = (background_scores >= threshold).float().mean()
        return 1.0 / background_efficiency.clamp_min(1.0 / background_scores.numel())

    def on_validation_epoch_end(self) -> None:
        accumulator = self._val_loss_accumulator
        reference = next(self.parameters()).new_zeros(())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for sum_name in (
                "edge_sum",
                "hyperedge_sum",
                "classification_sum",
                "classification_unweighted_sum",
            ):
                value = accumulator[sum_name]
                value = reference.clone() if value is None else value.clone()
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                accumulator[sum_name] = value
            for count_name in ("edge_count", "hyperedge_count", "classification_count"):
                count = reference.new_tensor(float(accumulator[count_name]))
                torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
                accumulator[count_name] = int(count.item())
        if (
            self.reconstruction_enabled
            and not accumulator["edge_count"]
            and not accumulator["hyperedge_count"]
        ):
            raise RuntimeError(
                "Validation data-composition error: reconstruction is enabled but the entire "
                "validation epoch contained no edge-active and no hyperedge-active events. "
                f"edge_active_events={accumulator['edge_count']}, "
                f"hyperedge_active_events={accumulator['hyperedge_count']}, "
                f"validation_subset_path={self.hparams.validation_subset_path!r}, "
                f"validation_subset_hash={self.hparams.validation_subset_hash!r}."
            )
        edge = (accumulator["edge_sum"] / accumulator["edge_count"]
                if accumulator["edge_count"] else reference)
        hyperedge = (accumulator["hyperedge_sum"] / accumulator["hyperedge_count"]
                     if accumulator["hyperedge_count"] else reference)
        classification = (accumulator["classification_sum"] / accumulator["classification_count"]
                          if accumulator["classification_count"] else reference)
        classification_raw = (
            accumulator["classification_unweighted_sum"] / accumulator["classification_count"]
            if accumulator["classification_count"] else reference
        )
        reconstruction = self.hparams.edge_weight * edge + self.hparams.hyperedge_weight * hyperedge
        total = reconstruction + self.hparams.classification_weight * classification_raw
        stable = {"val_loss": total}
        if self.reconstruction_enabled:
            stable.update({"val_edge_loss": edge, "val_hyperedge_loss": hyperedge,
                           "val_reconstruction_loss": reconstruction})
        if self.classification_enabled:
            stable["val_classification_loss"] = classification_raw
        for name, value in stable.items():
            self.log(name, value, on_step=False, on_epoch=True, prog_bar=name == "val_loss",
                     logger=self.log_metrics_to_logger, sync_dist=False)
        if self._collect_classification_diagnostics and self._val_logits:
            logits = torch.cat(self._val_logits).float()
            labels = torch.cat(self._val_targets).long()
            probabilities = torch.sigmoid(logits)
            predictions = probabilities >= 0.5
            tp = ((predictions == 1) & (labels == 1)).sum()
            tn = ((predictions == 0) & (labels == 0)).sum()
            fp = ((predictions == 1) & (labels == 0)).sum()
            fn = ((predictions == 0) & (labels == 1)).sum()
            auc = binary_auroc(probabilities, labels) if torch.unique(labels).numel() == 2 else probabilities.new_tensor(float("nan"))
            metrics = {
                "val_auc": auc,
                "metrics/validation_accuracy": (predictions == labels.bool()).float().mean(),
                "metrics/validation_tp": tp.float(),
                "metrics/validation_tn": tn.float(),
                "metrics/validation_fp": fp.float(),
                "metrics/validation_fn": fn.float(),
                "metrics/validation_signal_efficiency": tp.float() / (tp + fn).clamp_min(1),
                "metrics/validation_background_rejection_at_signal_eff_0p7": self._background_rejection_at_efficiency(labels, probabilities, 0.7),
                "metrics/validation_background_rejection_at_signal_eff_0p8": self._background_rejection_at_efficiency(labels, probabilities, 0.8),
            }
            for name, value in metrics.items():
                self.log(name, value, on_step=False, on_epoch=True, logger=self.log_metrics_to_logger, sync_dist=self._sync_dist())

        if self._collect_classification_diagnostics or self._collect_role_diagnostics:
            self.log("metrics/validation_diagnostic_events", float(self._diagnostic_events_seen),
                     on_step=False, on_epoch=True, logger=self.log_metrics_to_logger,
                     sync_dist=self._sync_dist())

        if self._collect_role_diagnostics:
            assert self._val_reco is not None
            role_top1 = []
            for kind in ("edge", "hyperedge"):
                for role, stats in self._val_reco[kind].items():
                    if stats["count"] == 0:
                        continue
                    denom = float(stats["count"])
                    top1 = stats["top1"] / denom
                    role_top1.append(top1)
                    for metric in ("top1", "top2", "top5"):
                        self.log(
                            f"metrics/validation_{kind}_{role}_{metric}",
                            float(stats[metric]) / denom,
                            on_step=False,
                            on_epoch=True,
                            logger=self.log_metrics_to_logger,
                            sync_dist=self._sync_dist(),
                        )
                    self.log(
                        f"metrics/validation_{kind}_{role}_mrr",
                        float(stats["rr"]) / denom,
                        on_step=False,
                        on_epoch=True,
                        logger=self.log_metrics_to_logger,
                        sync_dist=self._sync_dist(),
                    )
            mean_top1 = sum(role_top1) / len(role_top1) if role_top1 else 0.0
            self.log("val_reco_mean_role_top1", mean_top1, on_step=False, on_epoch=True, logger=self.log_metrics_to_logger, sync_dist=self._sync_dist())
            events = max(1, int(self._val_reco["events"]))
            self.log("metrics/validation_reco_active_fraction", self._val_reco["active"] / events, on_step=False, on_epoch=True, logger=self.log_metrics_to_logger, sync_dist=self._sync_dist())
            for label in ("signal", "background"):
                denom = max(1, int(self._val_reco[label]))
                self.log(
                    f"metrics/validation_reco_active_{label}_fraction",
                    self._val_reco[f"active_{label}"] / denom,
                    on_step=False,
                    on_epoch=True,
                    logger=self.log_metrics_to_logger,
                    sync_dist=self._sync_dist(),
                )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        required = (
            "x",
            "edge_index",
            "edge_attr",
            "u",
            "batch",
            "hyperedge_index",
            "node_p4",
            "node_ids",
            "node_truth_ids",
            "source_event_index",
            "edge_reco_active",
            "hyperedge_reco_active",
        )
        missing = [name for name in required if getattr(batch, name, None) is None]
        if missing:
            raise RuntimeError(
                "Prediction batch is missing required reconstruction/provenance fields: "
                f"{missing}. Rebuild or validate the graph cache."
            )
        hyper_logits, hyper_batch, edge_logits, cls_logits = self._shared_step(batch)
        edge_batch = self._candidate_event_batch(batch.batch, batch.edge_index, "edge")
        hyper_probs = torch.softmax(hyper_logits, dim=1) if hyper_logits is not None else None
        edge_probs = torch.softmax(edge_logits, dim=1) if edge_logits is not None else None
        hyper_logits_out = unbatch(hyper_logits, hyper_batch, 0) if hyper_logits is not None else None
        edge_logits_out = unbatch(edge_logits, edge_batch, 0) if edge_logits is not None else None
        hyper_probs_out = unbatch(hyper_probs, hyper_batch, 0) if hyper_probs is not None else None
        edge_probs_out = unbatch(edge_probs, edge_batch, 0) if edge_probs is not None else None
        node_counts = degree(batch.batch, num_nodes=int(batch.num_graphs)).cpu().long().tolist()
        node_types = unbatch(batch.x[:, -1].reshape(-1, 1), batch.batch, 0)
        node_p4 = unbatch(batch.node_p4, batch.batch, 0)
        node_ids = unbatch(batch.node_ids.reshape(-1, 1), batch.batch, 0)
        node_truth_ids = unbatch(batch.node_truth_ids.reshape(-1, 1), batch.batch, 0)
        cls_logits_out = cls_logits.detach().reshape(-1) if cls_logits is not None else None
        cls_probs_out = torch.sigmoid(cls_logits_out) if cls_logits_out is not None else None
        cls_target = self._class_targets(batch)
        source_index = batch.source_event_index.detach().reshape(-1)
        return {
            "hyperedge_logits": hyper_logits_out,
            "edge_logits": edge_logits_out,
            "hyperedge_probabilities": hyper_probs_out,
            "edge_probabilities": edge_probs_out,
            "node_counts": node_counts,
            "node_types": node_types,
            "node_p4": node_p4,
            "node_ids": node_ids,
            "node_truth_ids": node_truth_ids,
            "classification_logits": cls_logits_out,
            "classification_probabilities": cls_probs_out,
            "classification_target": cls_target,
            "source_event_index": source_index,
            "edge_reco_active": batch.edge_reco_active.detach().reshape(-1),
            "hyperedge_reco_active": batch.hyperedge_reco_active.detach().reshape(-1),
        }

    def _sync_dist(self) -> bool:
        trainer = getattr(self, "_trainer", None)
        return bool(trainer is not None and getattr(trainer, "world_size", 1) > 1)

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        name = str(self.hparams.optimizer).lower()
        optimiser_kwargs = {
            "lr": self.hparams.lr,
            "weight_decay": self.hparams.weight_decay,
        }
        foreach = self.hparams.optimizer_foreach
        fused = bool(self.hparams.optimizer_fused)
        if foreach is not None:
            optimiser_kwargs["foreach"] = bool(foreach)
        if fused:
            if foreach:
                raise ValueError("optimizer.fused and optimizer.foreach=true cannot be enabled together.")
            optimiser_kwargs["fused"] = True
        if name == "adam":
            optimizer = Adam(parameters, **optimiser_kwargs)
        elif name == "adamw":
            optimizer = AdamW(parameters, **optimiser_kwargs)
        else:
            raise ValueError("Supported optimizers are: adam, adamw.")
        if not self.hparams.lr_scheduler_enabled:
            return optimizer
        if str(self.hparams.lr_scheduler_method).lower() not in {"reduce_on_plateau", "reducelronplateau"}:
            raise ValueError("Supported lr_scheduler methods are: reduce_on_plateau.")
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.lr_scheduler_mode,
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            min_lr=self.hparams.lr_scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": self.hparams.lr_scheduler_monitor,
                "frequency": self.hparams.lr_scheduler_frequency,
                "strict": False,
            },
        }
