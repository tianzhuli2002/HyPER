"""HyPER training entry point for the single active typed configuration schema."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import hydra
import lightning.pytorch as pl
import torch
import torch_geometric
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from packaging import version

from HyPER.checkpoints import resolve_checkpoint
from HyPER.configuration import validate_runtime_config
from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel
from HyPER.factories import build_model, graph_config, plain
from HyPER.utils.timing import TrainingTimingCallback
from HyPER.utils.epoch_summary import PersistentEpochSummary


def _file_sha256(path: str | None) -> str | None:
    if path is None or not str(path).strip():
        return None
    candidate = Path(str(path)).expanduser()
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _load_probe_backbone(model: HyPERModel, checkpoint_path: str, skip_prefixes=("Classification.",)):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()
    loadable = {
        name: tensor
        for name, tensor in source_state.items()
        if not any(name.startswith(prefix) for prefix in skip_prefixes)
        and name in target_state
        and tuple(tensor.shape) == tuple(target_state[name].shape)
    }
    if not loadable:
        raise RuntimeError(f"No compatible frozen-probe backbone tensors found in {checkpoint_path}.")
    result = model.load_state_dict(loadable, strict=False)
    return {
        "checkpoint_path": checkpoint_path,
        "loaded_tensor_count": len(loadable),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _checkpoint_callbacks(monitor: str = "val_loss", mode: str = "min"):
    filename_metric = str(monitor)
    return [
        ModelCheckpoint(
            filename="best-total-{epoch:03d}-{" + filename_metric + ":.6f}",
            monitor=str(monitor),
            mode=str(mode),
            save_top_k=1,
            save_last=True,
            save_on_train_epoch_end=False,
        )
    ]


def run_training(cfg: DictConfig, extra_callbacks=None, logger_name="", return_metrics=True):
    print(OmegaConf.to_yaml(cfg))
    seed = int(cfg.general.seed)
    pl.seed_everything(seed, workers=True)
    topology, task = validate_runtime_config(cfg)
    classification_enabled = task.classification_enabled
    reconstruction_enabled = task.reconstruction_enabled
    tuning_cfg = cfg.get("tuning", {})
    tuning_enabled = bool(tuning_cfg.get("train_indices_file")) and bool(
        tuning_cfg.get("validation_indices_file")
    )
    exploratory_tuning = tuning_enabled and not bool(tuning_cfg.get("checkpointing", False))
    performance_cfg = cfg.get("performance", {})
    validation_diagnostics_cfg = cfg.get("validation_diagnostics", {})
    validation_subset_path = (
        tuning_cfg.get("validation_indices_file")
        if tuning_enabled
        else cfg.dataset.split.get("cache_path")
    )

    datamodule = HyPERDataModule(
        root=str(cfg.dataset.root),
        train_set=str(cfg.dataset.train_set),
        predict_set=str(cfg.dataset.predict_set),
        batch_size=int(cfg.dataset.batch_size),
        drop_last=bool(cfg.dataset.drop_last),
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory),
        persistent_workers=bool(cfg.dataset.persistent_workers),
        prefetch_factor=int(cfg.dataset.prefetch_factor),
        graph_config=graph_config(cfg),
        split_config=plain(cfg.dataset.split),
        predict_split=cfg.predicting.split,
        source_indices_file=cfg.predicting.source_indices_file,
        source_h5_path=cfg.dataset.get("source_h5_path"),
        require_two_event_classes=classification_enabled,
        tuning_mode=tuning_enabled,
        tuning_train_indices_file=tuning_cfg.get("train_indices_file"),
        tuning_val_indices_file=tuning_cfg.get("validation_indices_file"),
        seed=seed,
        classification_enabled=classification_enabled,
        reconstruction_enabled=reconstruction_enabled,
        verify_source_identity_per_event=bool(
            performance_cfg.get("verify_source_identity_per_event", False)
        ),
        source_identity_setup_samples=int(
            performance_cfg.get("source_identity_setup_samples", 32)
        ),
    )

    model = build_model(
        cfg,
        datamodule,
        classification_enabled=classification_enabled,
        reconstruction_enabled=reconstruction_enabled,
        log_metrics_to_logger=not exploratory_tuning,
        validation_subset_path=(
            None if validation_subset_path is None else str(validation_subset_path)
        ),
        validation_subset_hash=_file_sha256(validation_subset_path),
        validation_role_ranking_enabled=bool(
            validation_diagnostics_cfg.get("role_ranking_enabled", False)
        ),
        validation_classification_metrics_enabled=bool(
            validation_diagnostics_cfg.get("classification_metrics_enabled", False)
        ),
        validation_diagnostics_every_n_epochs=int(
            validation_diagnostics_cfg.get("every_n_epochs", 1)
        ),
        validation_diagnostics_max_events=validation_diagnostics_cfg.get("max_events"),
        validate_candidate_event_assignment=bool(
            performance_cfg.get("validate_candidate_event_assignment", False)
        ),
    )

    probe_manifest = None
    if task.probe_enabled:
        if not classification_enabled or reconstruction_enabled:
            raise ValueError("Frozen probe requires classification enabled and reconstruction disabled.")
        checkpoint = resolve_checkpoint(
            cfg.probe.get("pretrained_checkpoint"),
            cfg.probe.get("pretrained_model_directory"),
            purpose="Frozen-probe checkpoint",
        )
        prefixes = tuple(str(value) for value in cfg.probe.trainable_parameter_prefixes)
        load_info = _load_probe_backbone(model, checkpoint, skip_prefixes=prefixes)
        trainable, frozen = model.freeze_for_probe(prefixes)
        probe_manifest = {
            "probe_enabled": True,
            "pretrained": load_info,
            "trainable_parameters": trainable,
            "frozen_parameter_tensor_count": len(frozen),
        }

    callbacks = [] if exploratory_tuning else _checkpoint_callbacks(
        str(tuning_cfg.get("monitor", "val_loss")) if tuning_enabled else "val_loss",
        str(tuning_cfg.get("direction", "min")) if tuning_enabled else "min",
    )
    loss_checkpoint = None if exploratory_tuning else callbacks[0]
    if bool(cfg.early_stopping.enabled):
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.early_stopping.monitor),
                mode=str(cfg.early_stopping.mode),
                patience=int(cfg.early_stopping.patience),
                min_delta=float(cfg.early_stopping.get("min_delta", 0.0)),
                strict=True,
                check_on_train_epoch_end=False,
            )
        )
    if not exploratory_tuning:
        callbacks.append(LearningRateMonitor())
    callbacks.append(PersistentEpochSummary())
    progress_bar_enabled = bool(cfg.trainer.enable_progress_bar) and os.isatty(1) and os.isatty(2)
    if progress_bar_enabled:
        callbacks.append(TQDMProgressBar())
    if bool(cfg.profiling.enabled):
        callbacks.append(
            TrainingTimingCallback(
                log_every_n_steps=int(cfg.profiling.log_every_n_steps),
                cuda_synchronize=bool(cfg.profiling.cuda_synchronize),
                output_json=cfg.profiling.output_json,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    trainer_kwargs = {
        "accelerator": str(cfg.trainer.accelerator),
        "devices": cfg.trainer.devices,
        "precision": cfg.trainer.precision,
        "max_epochs": int(cfg.trainer.epochs),
        "callbacks": callbacks,
        "logger": False if exploratory_tuning else TensorBoardLogger(save_dir=str(cfg.paths.savedir), name=logger_name),
        "enable_checkpointing": not exploratory_tuning,
        "log_every_n_steps": int(cfg.trainer.log_every_n_steps),
        "num_sanity_val_steps": int(cfg.trainer.num_sanity_val_steps),
        "check_val_every_n_epoch": int(cfg.trainer.check_val_every_n_epoch),
        "enable_progress_bar": progress_bar_enabled,
        "gradient_clip_val": float(cfg.trainer.gradient_clip_val),
        "limit_val_batches": cfg.trainer.limit_val_batches,
        "limit_test_batches": cfg.trainer.get("limit_test_batches", 1.0),
    }
    if cfg.trainer.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = cfg.trainer.limit_train_batches
    if cfg.trainer.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = cfg.trainer.val_check_interval
    if cfg.trainer.get("max_steps") is not None:
        trainer_kwargs["max_steps"] = int(cfg.trainer.max_steps)
    trainer = pl.Trainer(**trainer_kwargs)

    resume = cfg.training.resume_from_checkpoint
    if resume is not None and str(resume).strip():
        resume = str(Path(str(resume)).expanduser().resolve())
        if not Path(resume).is_file():
            raise FileNotFoundError(resume)
    else:
        resume = None
    use_cuda = str(cfg.trainer.accelerator).lower() in {"gpu", "cuda"} and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    fit_started = time.perf_counter()
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume)
    fit_wall_seconds = time.perf_counter() - fit_started
    validation_metrics = {
        name: float(value.detach().cpu())
        for name, value in trainer.callback_metrics.items()
        if name.startswith("val_") and hasattr(value, "numel") and value.numel() == 1
    }

    best_path = ""
    if loss_checkpoint is not None:
        best_path = str(Path(loss_checkpoint.best_model_path).resolve()) if loss_checkpoint.best_model_path else ""
        if not best_path or not Path(best_path).is_file():
            raise RuntimeError(
                f"Training did not produce a checkpoint for monitored metric "
                f"{loss_checkpoint.monitor!r}; refusing to continue."
            )
    test_metrics = []
    if not tuning_enabled and bool(cfg.trainer.get("run_test_after_fit", True)):
        test_metrics = trainer.test(datamodule=datamodule, ckpt_path=best_path)

    checkpoint_data = torch.load(best_path, map_location="cpu") if best_path else {}
    best_score = None if loss_checkpoint is None else loss_checkpoint.best_model_score
    manifest = {
        "config_name": cfg.get("config_name"),
        "effective_config": OmegaConf.to_container(cfg, resolve=True),
        "log_dir": None if trainer.logger is None else str(Path(trainer.logger.log_dir).resolve()),
        "best_loss_checkpoint": best_path,
        "checkpoint_epoch": checkpoint_data.get("epoch"),
        "checkpoint_global_step": checkpoint_data.get("global_step"),
        "checkpoint_monitor": None if loss_checkpoint is None else str(loss_checkpoint.monitor),
        "checkpoint_mode": None if loss_checkpoint is None else str(loss_checkpoint.mode),
        "checkpoint_score": None if best_score is None else float(best_score.detach().cpu()),
        "split_cache_path": datamodule.split_cache_path,
        "split_metadata": datamodule.split_metadata,
        "loss_weights": {
            "edge_weight": float(cfg.loss.edge_weight),
            "hyperedge_weight": float(cfg.loss.hyperedge_weight),
            "classification_weight": float(cfg.loss.classification_weight),
        },
        "topology": topology,
        "task": task.mode,
        "classification_enabled": classification_enabled,
        "reconstruction_enabled": reconstruction_enabled,
        "performance": {
            "validation_role_ranking_enabled": bool(
                validation_diagnostics_cfg.get("role_ranking_enabled", False)
            ),
            "validation_classification_metrics_enabled": bool(
                validation_diagnostics_cfg.get("classification_metrics_enabled", False)
            ),
            "validation_diagnostics_every_n_epochs": int(
                validation_diagnostics_cfg.get("every_n_epochs", 1)
            ),
            "validation_diagnostics_max_events": validation_diagnostics_cfg.get("max_events"),
            "verify_source_identity_per_event": bool(
                performance_cfg.get("verify_source_identity_per_event", False)
            ),
            "source_identity_setup_samples": int(
                performance_cfg.get("source_identity_setup_samples", 32)
            ),
            "validate_candidate_event_assignment": bool(
                performance_cfg.get("validate_candidate_event_assignment", False)
            ),
            "optimizer_foreach": cfg.optimizer.get("foreach"),
            "optimizer_fused": bool(cfg.optimizer.get("fused", False)),
        },
        "test_metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "runtime": {
            "fit_wall_seconds": fit_wall_seconds,
            "training_events_seen": model.runtime_training_shapes["events"],
            "training_events_per_second": (
                model.runtime_training_shapes["events"] / fit_wall_seconds if fit_wall_seconds else None
            ),
            "training_shapes": model.runtime_training_shapes,
            "batch_size": int(cfg.dataset.batch_size),
            "gradients_finite": bool(model.gradients_finite),
            "peak_host_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if use_cuda else None
            ),
        },
    }
    requested_manifest = cfg.paths.get("training_manifest")
    if exploratory_tuning and not requested_manifest:
        requested_manifest = Path(str(tuning_cfg.output_dir)) / "exploratory_training_manifest.json"
    manifest_path = (
        Path(str(requested_manifest)).expanduser()
        if requested_manifest is not None and str(requested_manifest).strip()
        else Path(trainer.logger.log_dir) / "training_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote training manifest: {manifest_path.resolve()}")

    if probe_manifest is not None and trainer.logger is not None:
        path = Path(trainer.logger.log_dir) / "frozen_probe_manifest.json"
        path.write_text(json.dumps(probe_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not return_metrics:
        return None
    metrics = {
        name: float(value.detach().cpu()) if hasattr(value, "numel") and value.numel() == 1 else value
        for name, value in trainer.callback_metrics.items()
    }
    metrics["checkpoint_paths"] = [best_path]
    metrics["log_dir"] = None if trainer.logger is None else trainer.logger.log_dir
    metrics["test_array_loaded"] = False if tuning_enabled else None
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Train(cfg: DictConfig) -> None:
    run_training(cfg, return_metrics=False)


def setup_torch_runtime() -> None:
    torch.set_float32_matmul_precision("medium")
    if version.parse(torch.__version__) >= version.parse("2.6"):
        torch.serialization.add_safe_globals(
            [
                torch_geometric.data.data.DataEdgeAttr,
                torch_geometric.data.data.DataTensorAttr,
                torch_geometric.data.storage.GlobalStorage,
            ]
        )


if __name__ == "__main__":
    setup_torch_runtime()
    Train()
