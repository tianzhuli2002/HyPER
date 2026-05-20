import hydra
import torch
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
import os

_RICH_AVAILABLE = RequirementCache("rich>=10.2.2")

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

    # Extract datamodule params from config with sensible defaults
    num_workers = cfg.get('datamodule', {}).get('num_workers', 8)
    pin_memory = cfg.get('datamodule', {}).get('pin_memory', True if cfg['device'] == "gpu" else False)
    cache_dir = cfg.get('datamodule', {}).get('cache_dir', None)
    force_reload = cfg.get('datamodule', {}).get('force_reload', False)
    use_ondisk = cfg.get('datamodule', {}).get('use_ondisk', False)
    
    datamodule = HyPERDataModule(
        root = cfg['dataset'],
        train_set = cfg['train_set'],
        val_set = cfg['val_set'],
        batch_size = cfg['batch_size'],
        max_n_events = cfg['max_n_events'],
        percent_valid_samples = 1 - float(cfg['train_val_split']),
        drop_last = cfg['drop_last'],
        num_workers = num_workers,
        pin_memory = pin_memory,
        cache_dir = cache_dir,
        force_reload = force_reload,
        use_ondisk = use_ondisk,
    )

    model = HyPERModel(
        node_in_channels = datamodule.node_in_channels,
        edge_in_channels = datamodule.edge_in_channels,
        global_in_channels = datamodule.global_in_channels,
        message_feats = cfg['message_feats'],
        dropout = cfg['dropout'],
        message_passing_recurrent = cfg['num_message_layers'],
        contraction_feats = cfg['hyperedge_feats'],
        hyperedge_order = cfg['hyperedge_order'],
        criterion_edge = cfg['criterion_edge'],
        criterion_hyperedge = cfg['criterion_hyperedge'],
        optimizer = cfg['optimizer'],
        lr = cfg['learning_rate'],
        alpha = cfg['alpha'],
        beta = cfg['beta'],
        reduction = cfg['loss_reduction'],
        validation_mode = cfg.get('trainer', {}).get('validation_mode', 'keep'),
    )

    trainer_cfg = cfg.get('trainer', {})
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
            monitor="loss/validation_loss",
            save_top_k=save_top_k,
            mode="min",
            save_last=save_last
        ),
    ]

    if trainer_cfg.get('enable_early_stopping', True):
        callbacks.append(EarlyStopping(
            monitor="loss/validation_loss",
            mode="min",
            min_delta=0.00,
            patience=cfg['patience'],
            verbose=False
        ))

    callbacks += [
        LearningRateMonitor(),
        RichProgressBar() if _RICH_AVAILABLE else TQDMProgressBar(),
        RichModelSummary(max_depth=1) if _RICH_AVAILABLE else ModelSummary(max_depth=1)
    ]

    if trainer_cfg.get('enable_device_stats', False):
        callbacks.append(DeviceStatsMonitor())

    # Extract trainer settings
    trainer_kwargs = dict(
        accelerator = cfg['device'],
        devices = cfg['num_devices'],
        max_epochs = cfg['epochs'],
        callbacks = callbacks,
        logger = TensorBoardLogger(
            save_dir=cfg['savedir'],
            name="",
            log_graph=trainer_cfg.get('log_graph', False),
        ),
        log_every_n_steps = trainer_cfg.get('log_every_n_steps', 50),
        num_sanity_val_steps = trainer_cfg.get('num_sanity_val_steps', 0),
    )
    
    # Add optional trainer controls
    if trainer_cfg.get('gradient_clip', None) is not None:
        trainer_kwargs['gradient_clip_val'] = trainer_cfg['gradient_clip']
    if trainer_cfg.get('check_val_every_n_epoch', None) is not None:
        trainer_kwargs['check_val_every_n_epoch'] = trainer_cfg['check_val_every_n_epoch']
    if trainer_cfg.get('limit_val_batches', None) is not None:
        trainer_kwargs['limit_val_batches'] = trainer_cfg['limit_val_batches']
    if trainer_cfg.get('val_check_interval', None) is not None:
        trainer_kwargs['val_check_interval'] = trainer_cfg['val_check_interval']
    
    trainer = pl.Trainer(**trainer_kwargs)

    if cfg['continue_from_ckpt'] is not None and cfg['reset_params'] is True:
        print("Resume training state from %s, using new hyperparameters"%(cfg['continue_from_ckpt']))

        ckpt = torch.load(cfg['continue_from_ckpt'], map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)

        trainer.fit(model, datamodule=datamodule)

    elif cfg['continue_from_ckpt'] is not None:
        print("Resume training from %s"%(cfg['continue_from_ckpt']))

        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg['continue_from_ckpt'])
    
    else:
        trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.multiprocessing.set_start_method('spawn')
    Train()