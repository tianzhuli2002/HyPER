"""Shared construction helpers for train, predict, tuning and analysis."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from HyPER.models import HyPERModel


def plain(value: Any):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def graph_config(cfg: DictConfig) -> dict:
    if "input" not in cfg or "target" not in cfg:
        raise ValueError("HyPER configs must provide input and target sections.")
    graph = plain(OmegaConf.create({"input": cfg.input, "target": cfg.target}))
    graph.setdefault("target", {})["encoding"] = "typed"
    return graph


def build_model(
    cfg: DictConfig,
    datamodule,
    *,
    classification_enabled: bool,
    reconstruction_enabled: bool,
    log_metrics_to_logger: bool = True,
    validation_subset_path: str | None = None,
    validation_subset_hash: str | None = None,
    validation_role_ranking_enabled: bool = False,
    validation_classification_metrics_enabled: bool = False,
    validation_diagnostics_every_n_epochs: int = 1,
    validation_diagnostics_max_events: int | None = None,
    validate_candidate_event_assignment: bool = False,
) -> HyPERModel:
    """Build one model from the same config contract in every workflow."""
    return HyPERModel(
        node_in_channels=datamodule.node_in_channels,
        edge_in_channels=datamodule.edge_in_channels,
        global_in_channels=datamodule.global_in_channels,
        edge_out_channels=datamodule.edge_out_channels,
        hyperedge_out_channels=datamodule.hyperedge_out_channels,
        edge_class_names=datamodule.edge_class_names,
        hyperedge_class_names=datamodule.hyperedge_class_names,
        message_feats=int(cfg.model.message_feats),
        dropout=float(cfg.model.dropout),
        num_message_passing_layers=int(cfg.model.num_message_passing_layers),
        contraction_feats=int(cfg.model.contraction_feats),
        hyperedge_order=int(cfg.model.hyperedge_order),
        optimizer=str(cfg.optimizer.name),
        lr=float(cfg.optimizer.learning_rate),
        weight_decay=float(cfg.optimizer.weight_decay),
        lr_scheduler_enabled=bool(cfg.lr_scheduler.enabled),
        lr_scheduler_monitor=str(cfg.lr_scheduler.monitor),
        lr_scheduler_mode=str(cfg.lr_scheduler.mode),
        lr_scheduler_factor=float(cfg.lr_scheduler.factor),
        lr_scheduler_patience=int(cfg.lr_scheduler.patience),
        lr_scheduler_min_lr=float(cfg.lr_scheduler.min_lr),
        lr_scheduler_frequency=int(cfg.lr_scheduler.frequency),
        classification_enabled=bool(classification_enabled),
        reconstruction_enabled=bool(reconstruction_enabled),
        edge_class_weights=plain(cfg.loss.edge_class_weights),
        hyperedge_class_weights=plain(cfg.loss.hyperedge_class_weights),
        edge_weight=float(cfg.loss.edge_weight),
        hyperedge_weight=float(cfg.loss.hyperedge_weight),
        classification_weight=float(cfg.loss.classification_weight),
        classification_pos_weight=cfg.loss.classification_pos_weight,
        log_metrics_to_logger=bool(log_metrics_to_logger),
        validation_subset_path=validation_subset_path,
        validation_subset_hash=validation_subset_hash,
        validation_role_ranking_enabled=bool(validation_role_ranking_enabled),
        validation_classification_metrics_enabled=bool(validation_classification_metrics_enabled),
        validation_diagnostics_every_n_epochs=int(validation_diagnostics_every_n_epochs),
        validation_diagnostics_max_events=validation_diagnostics_max_events,
        validate_candidate_event_assignment=bool(validate_candidate_event_assignment),
        optimizer_foreach=cfg.optimizer.get("foreach"),
        optimizer_fused=bool(cfg.optimizer.get("fused", False)),
    )
