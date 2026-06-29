import hydra
import json
import torch
import torch_geometric
import lightning.pytorch as pl

from lightning.pytorch.loggers import TensorBoardLogger
from lightning_utilities.core.imports import RequirementCache
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    DeviceStatsMonitor,
    ModelSummary,
    TQDMProgressBar,
    EarlyStopping
)

from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel
from HyPER.utils.timing import TrainingTimingCallback
from omegaconf import DictConfig, OmegaConf
from packaging import version
import os
import warnings

_RICH_AVAILABLE = RequirementCache("rich>=10.2.2")


def _select(cfg: DictConfig, path: str, legacy: str = None, default=None):
    value = OmegaConf.select(cfg, path, default=None)
    if value is not None:
        return value
    if legacy is not None:
        value = OmegaConf.select(cfg, legacy, default=None)
        if value is not None:
            return value
    return default


def _graph_config_from_cfg(cfg: DictConfig):
    if OmegaConf.select(cfg, 'input', default=None) is None and OmegaConf.select(cfg, 'target', default=None) is None:
        return None
    if OmegaConf.select(cfg, 'input', default=None) is None or OmegaConf.select(cfg, 'target', default=None) is None:
        raise ValueError("Unified HyPER configs must provide both `input` and `target` sections.")
    return OmegaConf.to_container(
        OmegaConf.create({
            'input': OmegaConf.select(cfg, 'input'),
            'target': OmegaConf.select(cfg, 'target'),
        }),
        resolve=True,
    )


def _classification_loss_weight(cfg: DictConfig):
    class_weight = OmegaConf.select(cfg, 'classification.loss_weight', default=None)
    legacy_beta = _select(cfg, 'loss.beta', 'beta', None)
    if class_weight is not None and legacy_beta is not None and abs(float(class_weight) - float(legacy_beta)) > 1e-12:
        raise ValueError(
            "`classification.loss_weight` and legacy `loss.beta`/`beta` are both set "
            f"but disagree ({class_weight} != {legacy_beta}). Use one S/B loss coefficient."
        )
    if class_weight is not None:
        return float(class_weight)
    if legacy_beta is not None:
        warnings.warn(
            "`loss.beta`/`beta` is a deprecated name for the S/B classification loss weight; "
            "prefer `classification.loss_weight` in new configs.",
            DeprecationWarning,
        )
        return float(legacy_beta)
    return 0.5


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if OmegaConf.is_config(value):
        return [str(item) for item in OmegaConf.to_container(value, resolve=True)]
    return [str(value)]


def _as_plain_container(value, default=None):
    if value is None:
        return {} if default is None else default
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _resolve_checkpoint_path(checkpoint_path=None, model_directory=None):
    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path).strip()
        if checkpoint_path:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f"Probe checkpoint not found: {checkpoint_path}")
            return checkpoint_path

    if model_directory is None:
        return None

    model_directory = str(model_directory).strip()
    if not model_directory:
        return None
    checkpoint_dir = os.path.join(model_directory, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Probe model checkpoint directory not found: {checkpoint_dir}")

    candidates = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.endswith(".ckpt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    epoch_candidates = [
        path for path in candidates
        if os.path.basename(path).startswith("epoch")
    ]
    if epoch_candidates:
        return sorted(epoch_candidates)[-1]
    return sorted(candidates)[-1]


def _load_probe_backbone(model: HyPERModel, checkpoint_path: str, skip_prefixes=("Classification.",)):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = model.state_dict()
    skip_prefixes = tuple(str(prefix) for prefix in skip_prefixes)

    loadable = {}
    skipped_prefix = []
    skipped_missing = []
    skipped_shape = []

    for name, tensor in source_state.items():
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            skipped_prefix.append(name)
            continue
        if name not in target_state:
            skipped_missing.append(name)
            continue
        if tuple(tensor.shape) != tuple(target_state[name].shape):
            skipped_shape.append((name, tuple(tensor.shape), tuple(target_state[name].shape)))
            continue
        loadable[name] = tensor

    if not loadable:
        raise RuntimeError(f"No matching non-classification checkpoint weights found in {checkpoint_path}")

    result = model.load_state_dict(loadable, strict=False)
    print("================================")
    print("Loaded frozen-probe backbone checkpoint")
    print(f"Checkpoint:              {checkpoint_path}")
    print(f"Loaded tensors:          {len(loadable)}")
    print(f"Skipped by prefix:       {len(skipped_prefix)}")
    print(f"Skipped missing target:  {len(skipped_missing)}")
    print(f"Skipped shape mismatch:  {len(skipped_shape)}")
    print(f"Missing after load:      {len(result.missing_keys)}")
    print(f"Unexpected after load:   {len(result.unexpected_keys)}")
    if skipped_prefix:
        print("Skipped prefix tensors:")
        for name in skipped_prefix[:20]:
            print(f"  {name}")
        if len(skipped_prefix) > 20:
            print(f"  ... {len(skipped_prefix) - 20} more")
    if skipped_shape:
        print("Shape mismatches:")
        for name, source_shape, target_shape in skipped_shape[:20]:
            print(f"  {name}: checkpoint={source_shape}, model={target_shape}")
        if len(skipped_shape) > 20:
            print(f"  ... {len(skipped_shape) - 20} more")
    print("================================")

    return {
        "checkpoint_path": checkpoint_path,
        "loaded_tensors": sorted(loadable),
        "skipped_prefix": sorted(skipped_prefix),
        "skipped_missing_target": sorted(skipped_missing),
        "skipped_shape_mismatch": [
            {"name": name, "checkpoint_shape": list(source_shape), "model_shape": list(target_shape)}
            for name, source_shape, target_shape in skipped_shape
        ],
        "missing_keys_after_load": list(result.missing_keys),
        "unexpected_keys_after_load": list(result.unexpected_keys),
    }


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker process with unique random seed for reproducibility."""
    import numpy as np
    # Seed: base seed + worker_id to ensure each worker has unique randomness
    np.random.seed(42 + worker_id)
    os.environ['PYTHONHASHSEED'] = str(42 + worker_id)


def _metric_to_float(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value)
        return value.tolist()
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def run_training(cfg: DictConfig, extra_callbacks=None, logger_name="", return_metrics=True):
    r"""Perform network training using parameters defined in 
    `option_file`.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    device = _select(cfg, 'trainer.accelerator', 'device', 'cpu')
    num_devices = _select(cfg, 'trainer.devices', 'num_devices', 1)
    batch_size = _select(cfg, 'dataset.batch_size', 'batch_size', 128)
    num_workers = _select(cfg, 'dataset.num_workers', 'datamodule.num_workers', 8)
    pin_memory = _select(cfg, 'dataset.pin_memory', 'datamodule.pin_memory', True if device == "gpu" else False)
    if device == "cpu" and pin_memory:
        warnings.warn("Disabling `pin_memory` for CPU training.", UserWarning)
        pin_memory = False
    persistent_workers = _select(cfg, 'dataset.persistent_workers', 'datamodule.persistent_workers', True)
    prefetch_factor = _select(cfg, 'dataset.prefetch_factor', 'datamodule.prefetch_factor', 2)
    cache_dir = _select(cfg, 'dataset.cache_dir', 'datamodule.cache_dir', None)
    force_reload = _select(cfg, 'dataset.force_reload', 'datamodule.force_reload', False)
    use_ondisk = _select(cfg, 'dataset.use_ondisk', 'datamodule.use_ondisk', True)
    train_val_split = _select(cfg, 'dataset.train_val_split', 'train_val_split', 0.95)
    graph_config = _graph_config_from_cfg(cfg)
    classification_loss_weight = _classification_loss_weight(cfg)
    
    datamodule = HyPERDataModule(
        root = _select(cfg, 'dataset.root', 'dataset'),
        train_set = _select(cfg, 'dataset.train_set', 'train_set'),
        val_set = _select(cfg, 'dataset.val_set', 'val_set'),
        batch_size = batch_size,
        max_n_events = _select(cfg, 'dataset.max_n_events', 'max_n_events', -1),
        percent_valid_samples = 1 - float(train_val_split),
        drop_last = _select(cfg, 'dataset.drop_last', 'drop_last', True),
        num_workers = num_workers,
        pin_memory = pin_memory,
        persistent_workers = persistent_workers,
        prefetch_factor = prefetch_factor,
        cache_dir = cache_dir,
        force_reload = force_reload,
        use_ondisk = use_ondisk,
        graph_config = graph_config,
        split_config = _as_plain_container(_select(cfg, 'dataset.split', default={})),
        predict_split = _select(cfg, 'predicting.split', 'dataset.split.predict_split', None),
    )

    trainer_cfg = cfg.get('trainer', {})
    check_val_every_n_epoch = trainer_cfg.get('check_val_every_n_epoch', 1)
    lr_scheduler_default_frequency = check_val_every_n_epoch if check_val_every_n_epoch is not None else 1

    model = HyPERModel(
        node_in_channels = datamodule.node_in_channels,
        edge_in_channels = datamodule.edge_in_channels,
        global_in_channels = datamodule.global_in_channels,
        edge_out_channels = datamodule.edge_out_channels,
        hyperedge_out_channels = datamodule.hyperedge_out_channels,
        target_encoding = datamodule.target_encoding,
        message_feats = _select(cfg, 'model.message_feats', 'message_feats', 32),
        dropout = _select(cfg, 'model.dropout', 'dropout', 0.01),
        message_passing_recurrent = _select(cfg, 'model.message_passing_recurrent', 'num_message_layers', 3),
        contraction_feats = _select(cfg, 'model.contraction_feats', 'hyperedge_feats', 32),
        hyperedge_order = _select(cfg, 'model.hyperedge_order', 'hyperedge_order', 3),
        criterion_edge = _select(cfg, 'loss.criterion_edge', 'criterion_edge', 'BCE'),
        criterion_hyperedge = _select(cfg, 'loss.criterion_hyperedge', 'criterion_hyperedge', 'BCE'),
        optimizer = _select(cfg, 'optimizer.name', 'optimizer', 'Adam'),
        lr = _select(cfg, 'optimizer.learning_rate', 'learning_rate', 1e-3),
        weight_decay = _select(cfg, 'optimizer.weight_decay', default=0.0),
        lr_scheduler_enabled = _select(cfg, 'lr_scheduler.enabled', default=True),
        lr_scheduler_method = _select(cfg, 'lr_scheduler.method', default='reduce_on_plateau'),
        lr_scheduler_monitor = _select(cfg, 'lr_scheduler.monitor', default='val_loss'),
        lr_scheduler_mode = _select(cfg, 'lr_scheduler.mode', default='min'),
        lr_scheduler_factor = _select(cfg, 'lr_scheduler.factor', default=0.8),
        lr_scheduler_patience = _select(cfg, 'lr_scheduler.patience', default=10),
        lr_scheduler_min_lr = _select(cfg, 'lr_scheduler.min_lr', default=0.0),
        lr_scheduler_frequency = _select(cfg, 'lr_scheduler.frequency', default=lr_scheduler_default_frequency),
        alpha = _select(cfg, 'loss.alpha', 'alpha', 0.5),
        beta = classification_loss_weight,
        reduction = _select(cfg, 'loss.reduction', 'loss_reduction', 'mean'),
        validation_mode = cfg.get('trainer', {}).get('validation_mode', 'keep'),
        classification_enabled = _select(cfg, 'classification.enabled', default=True),
        # = _select(cfg, 'classification.input_mode', default='edge_hyperedge'),
        classification_loss_weight = classification_loss_weight,
        reconstruction_enabled = _select(cfg, 'reconstruction.enabled', default=True),
        reconstruction_weighting = _select(cfg, 'loss.reconstruction_weighting', default='legacy'),
        positive_weight_cap = _select(cfg, 'loss.positive_weight_cap', default=50.0),
        negative_weight_cap = _select(cfg, 'loss.negative_weight_cap', default=5.0),
        log_reco_diagnostics = _select(cfg, 'metrics.log_reco_diagnostics', default=True),
        log_reco_score_metrics = _select(cfg, 'metrics.log_reco_score_metrics', default=True),
        log_validation_topk = _select(cfg, 'metrics.log_validation_topk', default=True),
        log_classification_batch_sb = _select(cfg, 'metrics.log_classification_batch_sb', default=True),
        classification_debug_finite_checks = _select(cfg, 'classification.debug_finite_checks', default=False),
        classification_debug_range_checks = _select(cfg, 'classification.debug_range_checks', default=False),
        profile_reco_sections_enabled = _select(cfg, 'profiling.reco_sections_enabled', default=False),
        profile_reco_sections_batches = _select(cfg, 'profiling.reco_sections_batches', default=50),
        profile_reco_sections_cuda_synchronize = _select(cfg, 'profiling.reco_sections_cuda_synchronize', default=True),
        profile_reco_sections_log_every_n_steps = _select(cfg, 'profiling.reco_sections_log_every_n_steps', default=1),
        validate_cached_target_classes = _select(cfg, 'loss.validate_cached_target_classes', default=False),
        debug_finite_checks = _select(cfg, 'debug.finite_checks', default=False),
        debug_stop_on_nonfinite = _select(cfg, 'debug.stop_on_nonfinite', default=True),
        debug_log_tensor_ranges = _select(cfg, 'debug.log_tensor_ranges', default=False),
        debug_log_loss_components = _select(cfg, 'debug.log_loss_components', default=False),
        debug_log_reco_target_activity = _select(cfg, 'debug.log_reco_target_activity', default=False),
    )

    probe_enabled = bool(_select(cfg, 'probe.enabled', default=False))
    probe_manifest = None
    probe_trainable_names = []
    probe_frozen_names = []
    if probe_enabled:
        if not bool(_select(cfg, 'classification.enabled', default=True)):
            raise ValueError("probe.enabled=true requires classification.enabled=true.")
        pretrained_checkpoint = _resolve_checkpoint_path(
            checkpoint_path=_select(cfg, 'probe.pretrained_checkpoint_path', default=None),
            model_directory=_select(cfg, 'probe.pretrained_model_directory', default=None),
        )
        if pretrained_checkpoint is None:
            raise ValueError(
                "probe.enabled=true requires probe.pretrained_checkpoint_path "
                "or probe.pretrained_model_directory."
            )
        trainable_prefixes = _as_list(
            _select(cfg, 'probe.trainable_parameter_prefixes', default=["Classification."])
        )
        if not trainable_prefixes:
            trainable_prefixes = ["Classification."]
        load_manifest = _load_probe_backbone(
            model,
            pretrained_checkpoint,
            skip_prefixes=tuple(trainable_prefixes),
        )
        probe_trainable_names, probe_frozen_names = model.freeze_for_probe(
            trainable_prefixes=tuple(trainable_prefixes),
        )
        probe_manifest = {
            "probe_enabled": True,
            "trainable_parameter_prefixes": trainable_prefixes,
            "trainable_parameter_names": probe_trainable_names,
            "n_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "n_total_parameters": int(sum(p.numel() for p in model.parameters())),
            "n_frozen_parameter_tensors": int(len(probe_frozen_names)),
            "pretrained": load_manifest,
        }

    checkpoint_mode = str(trainer_cfg.get('checkpoint_mode', 'keep_best_last')).lower()

    if checkpoint_mode == 'keep_best_only':
        save_top_k = 1
        save_last = False
    elif checkpoint_mode == 'last_only':
        save_top_k = 0
        save_last = True
    else:
        save_top_k = 1
        save_last = True

    checkpoint_callback = ModelCheckpoint(
            verbose=True,
            monitor=_select(cfg, 'checkpoint.monitor', default='val_loss'),
            save_top_k=save_top_k,
            mode=_select(cfg, 'checkpoint.mode', default='min'),
            save_last=save_last,
            save_on_train_epoch_end=False,
        )

    callbacks = [checkpoint_callback]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    early_stopping_enabled = _select(cfg, 'early_stopping.enabled', 'trainer.enable_early_stopping', True)
    early_stopping_monitor = _select(cfg, 'early_stopping.monitor', default='val_loss')
    early_stopping_mode = _select(cfg, 'early_stopping.mode', default='min')
    early_stopping_patience = _select(cfg, 'early_stopping.patience', 'trainer.patience', 30)
    print(
        "Configuring early stopping: "
        f"enabled={early_stopping_enabled}, "
        f"monitor={early_stopping_monitor}, "
        f"mode={early_stopping_mode}, "
        f"patience={early_stopping_patience}",
        flush=True,
    )
    if early_stopping_enabled:
        callbacks.append(EarlyStopping(
            monitor=early_stopping_monitor,
            mode=early_stopping_mode,
            min_delta=_select(cfg, 'early_stopping.min_delta', default=0.00),
            patience=early_stopping_patience,
            verbose=False,
            strict=False,
            check_on_train_epoch_end=False,
        ))

    callbacks.append(LearningRateMonitor())
    if _select(cfg, 'trainer.enable_progress_bar', default=True):
        callbacks.append(RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar())
    callbacks.append(RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1))

    if trainer_cfg.get('enable_device_stats', False):
        callbacks.append(DeviceStatsMonitor())

    profiling_enabled = bool(_select(cfg, 'profiling.enabled', default=False))
    if profiling_enabled:
        callbacks.append(
            TrainingTimingCallback(
                log_every_n_steps=int(_select(cfg, 'profiling.log_every_n_steps', default=50) or 0),
                cuda_synchronize=bool(_select(cfg, 'profiling.cuda_synchronize', default=False)),
                output_json=_select(cfg, 'profiling.output_json', default=None),
            )
        )

    # Extract trainer settings
    precision = trainer_cfg.get("precision", "32-true")
    print(f"Validation cadence: check_val_every_n_epoch={check_val_every_n_epoch}")
    trainer_kwargs = dict(
        accelerator = device,
        devices = num_devices,
        precision = precision,
        max_epochs = _select(cfg, 'trainer.epochs', 'epochs', 1),
        callbacks = callbacks,
        logger = TensorBoardLogger(
            save_dir=_select(cfg, 'paths.savedir', 'savedir', 'HyPER_logs'),
            name=logger_name,
            log_graph=trainer_cfg.get('log_graph', False),
        ),
        log_every_n_steps = trainer_cfg.get('log_every_n_steps', 50),
        num_sanity_val_steps = trainer_cfg.get('num_sanity_val_steps', 0),
        check_val_every_n_epoch = check_val_every_n_epoch,
        enable_progress_bar = bool(_select(cfg, 'trainer.enable_progress_bar', default=True)),
    )
    
    # Add optional trainer controls
    gradient_clip_val = _select(cfg, 'trainer.gradient_clip_val', 'trainer.gradient_clip', None)
    if gradient_clip_val is not None:
        trainer_kwargs['gradient_clip_val'] = gradient_clip_val
    if trainer_cfg.get('limit_val_batches', None) is not None:
        trainer_kwargs['limit_val_batches'] = trainer_cfg['limit_val_batches']
    if trainer_cfg.get('limit_train_batches', None) is not None:
        trainer_kwargs['limit_train_batches'] = trainer_cfg['limit_train_batches']
    if trainer_cfg.get('val_check_interval', None) is not None:
        trainer_kwargs['val_check_interval'] = trainer_cfg['val_check_interval']
    if trainer_cfg.get('max_steps', None) is not None:
        trainer_kwargs['max_steps'] = trainer_cfg['max_steps']
    if trainer_cfg.get('max_time', None) is not None:
        trainer_kwargs['max_time'] = trainer_cfg['max_time']
    
    trainer = pl.Trainer(**trainer_kwargs)

    continue_from_ckpt = _select(cfg, 'training.resume_from_checkpoint', 'paths.checkpoint', None)
    if continue_from_ckpt is None:
        continue_from_ckpt = _select(cfg, 'continue_from_ckpt', default=None)
    if continue_from_ckpt is not None:
        continue_from_ckpt = str(continue_from_ckpt).strip()
        if not continue_from_ckpt:
            continue_from_ckpt = None
    if continue_from_ckpt is not None and not os.path.isfile(continue_from_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {continue_from_ckpt}")

    if continue_from_ckpt is not None and cfg.get('reset_params', False) is True:
        print("Resume training state from %s, using new hyperparameters"%(continue_from_ckpt))

        ckpt = torch.load(continue_from_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)

        trainer.fit(model, datamodule=datamodule)

    elif continue_from_ckpt is not None:
        print("Resume training from %s"%(continue_from_ckpt))

        trainer.fit(model, datamodule=datamodule, ckpt_path=continue_from_ckpt)
    
    else:
        trainer.fit(model, datamodule=datamodule)

    if probe_manifest is not None:
        probe_manifest.update(
            {
                "best_model_path": checkpoint_callback.best_model_path,
                "best_model_score": (
                    float(checkpoint_callback.best_model_score.detach().cpu())
                    if checkpoint_callback.best_model_score is not None
                    else None
                ),
                "last_model_path": getattr(checkpoint_callback, "last_model_path", None),
                "log_dir": getattr(trainer.logger, "log_dir", None),
            }
        )
        manifest_dir = getattr(trainer.logger, "log_dir", None) or os.getcwd()
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(manifest_dir, "frozen_probe_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(probe_manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Wrote frozen-probe manifest: {manifest_path}")

    if not return_metrics:
        return None

    metrics = {
        key: _metric_to_float(value)
        for key, value in trainer.callback_metrics.items()
    }
    metrics.update(
        {
            "best_model_path": checkpoint_callback.best_model_path,
            "best_model_score": _metric_to_float(checkpoint_callback.best_model_score),
            "last_model_path": getattr(checkpoint_callback, "last_model_path", None),
            "log_dir": getattr(trainer.logger, "log_dir", None),
        }
    )
    if "val_loss" not in metrics or metrics["val_loss"] is None:
        raise RuntimeError("Training completed but `val_loss` was not found in trainer.callback_metrics.")
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Train(cfg: DictConfig) -> None:
    run_training(cfg, return_metrics=False)


def setup_torch_runtime() -> None:
    torch.set_float32_matmul_precision('medium')

    # Required for loading PyG Data objects with PyTorch 2.6+ safe unpickling.
    if version.parse(torch.__version__) >= version.parse("2.6"):
        torch.serialization.add_safe_globals([
            torch_geometric.data.data.DataEdgeAttr,
            torch_geometric.data.data.DataTensorAttr,
            torch_geometric.data.storage.GlobalStorage,
        ])

    import tqdm
    import multiprocessing
    tqdm.tqdm.monitor_interval = 0
    tqdm.tqdm.set_lock(multiprocessing.RLock())


if __name__ == '__main__':
    setup_torch_runtime()
    Train()
