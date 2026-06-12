import hydra
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


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker process with unique random seed for reproducibility."""
    import numpy as np
    # Seed: base seed + worker_id to ensure each worker has unique randomness
    np.random.seed(42 + worker_id)
    os.environ['PYTHONHASHSEED'] = str(42 + worker_id)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Train(cfg : DictConfig) -> None:
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
    )

    trainer_cfg = cfg.get('trainer', {})
    check_val_every_n_epoch = trainer_cfg.get('check_val_every_n_epoch', 1)

    model = HyPERModel(
        node_in_channels = datamodule.node_in_channels,
        edge_in_channels = datamodule.edge_in_channels,
        global_in_channels = datamodule.global_in_channels,
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
        lr_scheduler_frequency = _select(cfg, 'lr_scheduler.frequency', default=check_val_every_n_epoch),
        alpha = _select(cfg, 'loss.alpha', 'alpha', 0.5),
        beta = classification_loss_weight,
        reduction = _select(cfg, 'loss.reduction', 'loss_reduction', 'mean'),
        validation_mode = cfg.get('trainer', {}).get('validation_mode', 'keep'),
        classification_enabled = _select(cfg, 'classification.enabled', default=True),
        # = _select(cfg, 'classification.input_mode', default='edge_hyperedge'),
        classification_loss_weight = classification_loss_weight,
        reconstruction_weighting = _select(cfg, 'loss.reconstruction_weighting', default='legacy'),
        positive_weight_cap = _select(cfg, 'loss.positive_weight_cap', default=50.0),
        negative_weight_cap = _select(cfg, 'loss.negative_weight_cap', default=5.0),
    )

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

    callbacks = [
        ModelCheckpoint(
            verbose=True,
            monitor=_select(cfg, 'checkpoint.monitor', default='val_loss'),
            save_top_k=save_top_k,
            mode=_select(cfg, 'checkpoint.mode', default='min'),
            save_last=save_last,
            save_on_train_epoch_end=False,
        ),
    ]

    early_stopping_enabled = _select(cfg, 'early_stopping.enabled', 'trainer.enable_early_stopping', True)
    if early_stopping_enabled:
        callbacks.append(EarlyStopping(
            monitor=_select(cfg, 'early_stopping.monitor', default='val_loss'),
            mode=_select(cfg, 'early_stopping.mode', default='min'),
            min_delta=_select(cfg, 'early_stopping.min_delta', default=0.00),
            patience=_select(cfg, 'early_stopping.patience', 'trainer.patience', 30),
            verbose=False,
            strict=False,
            check_on_train_epoch_end=False,
        ))

    callbacks += [
        LearningRateMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1)
    ]

    if trainer_cfg.get('enable_device_stats', False):
        callbacks.append(DeviceStatsMonitor())

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
            name="",
            log_graph=trainer_cfg.get('log_graph', False),
        ),
        log_every_n_steps = trainer_cfg.get('log_every_n_steps', 50),
        num_sanity_val_steps = trainer_cfg.get('num_sanity_val_steps', 0),
        check_val_every_n_epoch = check_val_every_n_epoch,
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


if __name__ == '__main__':
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

    Train()
