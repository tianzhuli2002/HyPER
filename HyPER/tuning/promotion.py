"""Retrain and select top Stage-1 exploratory candidates with checkpointing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna
import numpy as np
import yaml
from omegaconf import OmegaConf

from HyPER.configuration import set_task_mode
from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel
from HyPER.train import _file_sha256, run_training, setup_torch_runtime
from HyPER.factories import build_model, graph_config, plain
from .configuration import configure_effective_graph_dataset, configure_tuning_data_isolation
from .monitor import BestObservedValidation


def top_trials(sqlite_path, study_name, count):
    study = optuna.load_study(study_name=study_name,
                              storage=f"sqlite:///{Path(sqlite_path).resolve()}")
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < int(count):
        raise RuntimeError(f"Promotion requires {count} completed trials; found {len(complete)}.")
    return sorted(complete, key=lambda trial: (float(trial.value), trial.number))[:int(count)]


def _require_file(path, description):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise FileNotFoundError(f"Required {description} is absent or empty: {resolved}")
    return resolved


def _validate_subsets(train_path, validation_path, split_path, cfg=None):
    """Compatibility-sized entry point backed by the one tuning isolation contract."""
    return configure_tuning_data_isolation(
        OmegaConf.create({}) if cfg is None else cfg,
        canonical_split_path=split_path,
        train_indices_path=train_path,
        validation_indices_path=validation_path,
    )


def configure_candidate(args, trial, output):
    cfg = OmegaConf.load(_require_file(args.base_config, "promotion base config"))
    for path, value in trial.params.items():
        OmegaConf.update(cfg, path, value, merge=False, force_add=False)
    set_task_mode(cfg, "reconstruction")
    cfg.loss.edge_weight = .5
    cfg.loss.hyperedge_weight = .5
    cfg.loss.classification_weight = 0.
    if getattr(args, "dataset_root", None) is not None:
        configure_effective_graph_dataset(
            cfg, dataset_root=args.dataset_root, dataset_name=args.dataset_name
        )
    if getattr(args, "num_workers", None) is not None:
        cfg.dataset.num_workers = int(args.num_workers)
    cfg.tuning = {"checkpointing": True, "monitor": "val_reconstruction_loss",
                  "direction": "min", "output_dir": str(output)}
    subset_validation = configure_tuning_data_isolation(
        cfg, canonical_split_path=args.split_cache, train_indices_path=args.train_indices,
        validation_indices_path=args.validation_indices,
    )
    cfg.paths.savedir = str(output / "logs")
    cfg.paths.training_manifest = str(output / "training_manifest.json")
    cfg.trainer.epochs = args.max_epochs
    cfg.trainer.run_test_after_fit = False
    cfg.early_stopping.enabled = True
    cfg.early_stopping.monitor = "val_reconstruction_loss"
    cfg.early_stopping.mode = "min"
    cfg.early_stopping.patience = args.patience
    cfg.lr_scheduler.monitor = "val_reconstruction_loss"
    cfg.lr_scheduler.mode = "min"
    if args.limit_train_batches is not None:
        cfg.trainer.limit_train_batches = args.limit_train_batches
    if args.limit_val_batches is not None:
        cfg.trainer.limit_val_batches = args.limit_val_batches

    # A one-epoch promotion smoke must validate during that epoch.
    # The production configs normally validate every two epochs.
    if (
        args.limit_train_batches is not None
        or args.limit_val_batches is not None
    ):
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.log_every_n_steps = 1

    return cfg, subset_validation


def _instantiate_for_validation(cfg):
    """Construct the production datamodule/model pair without creating a Trainer."""
    tuning_cfg = cfg.tuning
    datamodule = HyPERDataModule(
        root=str(cfg.dataset.root), train_set=str(cfg.dataset.train_set),
        predict_set=str(cfg.dataset.predict_set), batch_size=int(cfg.dataset.batch_size),
        drop_last=bool(cfg.dataset.drop_last), num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory), persistent_workers=bool(cfg.dataset.persistent_workers),
        prefetch_factor=int(cfg.dataset.prefetch_factor), graph_config=graph_config(cfg),
        split_config=plain(cfg.dataset.split), predict_split=cfg.predicting.split,
        source_indices_file=cfg.predicting.source_indices_file,
        source_h5_path=cfg.dataset.get("source_h5_path"), require_two_event_classes=False,
        tuning_mode=True, tuning_train_indices_file=tuning_cfg.train_indices_file,
        tuning_val_indices_file=tuning_cfg.validation_indices_file, seed=int(cfg.general.seed),
    )
    model = build_model(
        cfg,
        datamodule,
        classification_enabled=False,
        reconstruction_enabled=True,
        log_metrics_to_logger=False,
        validation_subset_path=str(tuning_cfg.validation_indices_file),
        validation_subset_hash=_file_sha256(str(tuning_cfg.validation_indices_file)),
    )

    return datamodule, model


def _validate_production_path(cfg):
    import torch
    datamodule, model = _instantiate_for_validation(cfg)
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))
    model.eval()
    with torch.no_grad():
        losses = model._loss_components(batch, model._shared_step(batch))
    edge = losses["edge"]
    hyperedge = losses["hyperedge"]
    reconstruction = model.hparams.edge_weight * edge + model.hparams.hyperedge_weight * hyperedge
    values = {"val_edge_loss": float(edge.detach().cpu()),
              "val_hyperedge_loss": float(hyperedge.detach().cpu()),
              "val_reconstruction_loss": float(reconstruction.detach().cpu())}
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError(f"Promotion validation produced non-finite reconstruction metrics: {values}")
    dataset_root = Path(str(cfg.dataset.root)).expanduser().resolve()
    return {**values, "validation_batch_events": int(batch.num_graphs),
            "edge_output_classes": datamodule.edge_out_channels,
            "hyperedge_output_classes": datamodule.hyperedge_out_channels,
            "dataset_root": str(dataset_root),
            "dataset_name": str(cfg.dataset.train_set),
            "database_path": str(dataset_root / f"{cfg.dataset.train_set}.db")}


def _validate_smoke_checkpoint(cfg, checkpoint_path):
    import torch
    checkpoint = _require_file(checkpoint_path, "promotion smoke checkpoint")
    datamodule, model = _instantiate_for_validation(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False).get("state_dict")
    if state is None:
        raise KeyError(f"Promotion smoke checkpoint lacks state_dict: {checkpoint}")
    model.load_state_dict(state, strict=True)
    datamodule.setup("fit")
    batch = next(iter(datamodule.val_dataloader()))
    model.eval()
    with torch.no_grad():
        losses = model._loss_components(batch, model._shared_step(batch))
    reconstruction = model.hparams.edge_weight * losses["edge"] + model.hparams.hyperedge_weight * losses["hyperedge"]
    value = float(reconstruction.detach().cpu())
    if not np.isfinite(value):
        raise FloatingPointError(f"Reloaded smoke checkpoint produced non-finite val_reconstruction_loss: {value}")
    dataset_root = Path(str(cfg.dataset.root)).expanduser().resolve()
    return {"checkpoint": str(checkpoint), "strict_state_dict": True,
            "edge_output_classes": datamodule.edge_out_channels,
            "hyperedge_output_classes": datamodule.hyperedge_out_channels,
            "val_reconstruction_loss": value,
            "validation_batch_events": int(batch.num_graphs),
            "dataset_root": str(dataset_root),
            "dataset_name": str(cfg.dataset.train_set),
            "database_path": str(dataset_root / f"{cfg.dataset.train_set}.db")}


def _validate_full_checkpoint(cfg, checkpoint_path, validation_indices):
    """Run one final fixed, large validation pass from the retained best checkpoint."""
    import lightning.pytorch as pl
    import torch
    full_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    full_cfg.tuning.validation_indices_file = str(_require_file(
        validation_indices, "full promotion validation indices"
    ))
    datamodule, model = _instantiate_for_validation(full_cfg)
    checkpoint = torch.load(_require_file(checkpoint_path, "promotion checkpoint"),
                            map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict")
    if state is None:
        raise KeyError(f"Promotion checkpoint lacks state_dict: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    trainer = pl.Trainer(
        accelerator=str(full_cfg.trainer.accelerator), devices=full_cfg.trainer.devices,
        precision=full_cfg.trainer.precision, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, num_sanity_val_steps=0,
    )
    rows = trainer.validate(model, datamodule=datamodule, verbose=False)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one full-validation metric row, observed {len(rows)}.")
    metrics = {name: float(value) for name, value in rows[0].items() if name.startswith("val_")}
    required = ("val_reconstruction_loss", "val_edge_loss", "val_hyperedge_loss")
    missing = [name for name in required if name not in metrics or not np.isfinite(metrics[name])]
    if missing:
        raise RuntimeError(f"Full promotion validation lacks finite metrics {missing}: {metrics}")
    return {"checkpoint": str(Path(checkpoint_path).resolve()), "strict_state_dict": True,
            "validation_indices": str(Path(validation_indices).resolve()), "metrics": metrics}


def run_candidate(args):
    if not 0 <= args.candidate_rank < args.top_count:
        raise ValueError(
            f"candidate-rank {args.candidate_rank} is outside [0, {args.top_count - 1}]."
        )
    _require_file(args.sqlite_path, "Stage 1 Optuna SQLite database")
    subset_manifest = _require_file(args.subset_manifest, "promotion subset manifest")
    trial = top_trials(args.sqlite_path, args.study_name, args.top_count)[args.candidate_rank]
    output = Path(args.output_dir) / f"candidate_{args.candidate_rank:02d}_trial_{trial.number:06d}"
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    cfg, subset_validation = configure_candidate(args, trial, output)
    resolved_config = output / "resolved_promotion_config.yaml"
    resolved_config.write_text(
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), sort_keys=False)
    )
    contract = {
        "topology": args.topology,
        "stage": "promotion",
        "candidate_rank": args.candidate_rank,
        "source_stage1_trial": trial.number,
        "source_stage1_objective": float(trial.value),
        "source_study_name": args.study_name,
        "source_sqlite_path": str(Path(args.sqlite_path).resolve()),
        "base_config": str(Path(args.base_config).resolve()),
        "resolved_config": str(resolved_config.resolve()),
        "subset_manifest": str(subset_manifest),
        "canonical_split": subset_validation["canonical_split"],
        "train_indices": subset_validation["training_indices"],
        "validation_indices": subset_validation["validation_indices"],
        "subset_validation": subset_validation,
        "effective_dataset_root": str(Path(str(cfg.dataset.root)).resolve()),
        "effective_dataset_name": str(cfg.dataset.train_set),
        "effective_database_path": str(
            Path(str(cfg.dataset.root)).resolve() / f"{cfg.dataset.train_set}.db"
        ),
        "effective_manifest_path": str(
            Path(str(cfg.dataset.root)).resolve() / f"{cfg.dataset.train_set}.db.manifest.json"
        ),
        "monitor": "val_reconstruction_loss",
        "direction": "min",
        "max_epochs": int(args.max_epochs),
        "early_stopping_patience": int(args.patience),
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "output_directory": str(output.resolve()),
    }
    (output / "promotion_manifest.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    if args.validate_only:
        summary = _validate_production_path(cfg)
        summary.update({"validate_only": True, "source_exploratory_trial": trial.number,
                        "candidate_rank": args.candidate_rank})
        (output / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return summary
    recorder = BestObservedValidation("val_reconstruction_loss", "min")
    metrics = run_training(cfg, extra_callbacks=[recorder], logger_name="promotion", return_metrics=True)
    summary = recorder.summary(); summary.update({"source_exploratory_trial": trial.number,
                                                   "candidate_rank": args.candidate_rank,
                                                   "parameters": trial.params,
                                                   "ordinary_training_metrics": metrics})
    checkpoint_paths = metrics.get("checkpoint_paths", [])
    if len(checkpoint_paths) != 1 or not checkpoint_paths[0]:
        raise RuntimeError(f"Promotion did not return exactly one loadable checkpoint: {checkpoint_paths}")
    if args.final_validation_indices and args.limit_train_batches is None and args.limit_val_batches is None:
        summary["full_validation"] = _validate_full_checkpoint(
            cfg, checkpoint_paths[0], args.final_validation_indices
        )
    (output / "promotion_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str)+"\n")
    if args.limit_train_batches is not None or args.limit_val_batches is not None:
        checkpoint_summary = _validate_smoke_checkpoint(cfg, checkpoint_paths[0])
        (output / "smoke_checkpoint_validation.json").write_text(
            json.dumps(checkpoint_summary, indent=2, sort_keys=True) + "\n"
        )
        hparams = list((output / "logs").glob("**/hparams.yaml"))
        if len(hparams) != 1:
            raise RuntimeError(f"Promotion smoke requires exactly one hparams.yaml below {output / 'logs'}, found {hparams}")
        (output / "hparams.yaml").write_text(hparams[0].read_text(encoding="utf-8"), encoding="utf-8")
    return summary


def select(args):
    paths = sorted(Path(args.output_dir).glob("candidate_*/promotion_metrics.json"))
    if not paths: raise RuntimeError("No completed promotion metrics found.")
    entries = [(json.loads(path.read_text()), path) for path in paths]
    have_full = ["full_validation" in entry for entry, _ in entries]
    if any(have_full) and not all(have_full):
        raise RuntimeError("Promotion candidates mix sampled-only and full-validation results.")
    def rank(item):
        summary, _ = item
        value = (summary["full_validation"]["metrics"]["val_reconstruction_loss"]
                 if all(have_full) else summary["best_observed_monitor_value"])
        return float(value), summary["source_exploratory_trial"]
    winner, path = min(entries, key=rank)
    cfg = OmegaConf.load(args.base_config)
    for dotted, value in winner["parameters"].items(): OmegaConf.update(cfg, dotted, value, merge=False, force_add=False)
    set_task_mode(cfg, "reconstruction")
    cfg.loss.edge_weight = .5; cfg.loss.hyperedge_weight = .5; cfg.loss.classification_weight = 0.
    contract_path = path.parent / "promotion_manifest.json"
    contract = json.loads(_require_file(contract_path, "winning promotion manifest").read_text())
    cfg.dataset.split.cache_path = contract["canonical_split"]
    cfg.dataset.split.require_existing = True
    cfg.dataset.split.predict_split = None
    cfg.predicting.split = None
    cfg.predicting.source_indices_file = None
    if "tuning" in cfg:
        del cfg["tuning"]
    selected_value = (winner["full_validation"]["metrics"]["val_reconstruction_loss"]
                      if all(have_full) else winner["best_observed_monitor_value"])
    cfg.tuning_ancestry = {"stage1_study": args.study_name,
                           "stage1_trial": winner["source_exploratory_trial"],
                           "stage1_promotion_run": str(path.parent.resolve()),
                           "stage1_best_value": selected_value,
                           "stage1_selection_metric": ("full_val_reconstruction_loss"
                                                       if all(have_full) else "val_reconstruction_loss")}
    output = Path(args.selection_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "stage1_best_parameters.yaml").write_text(yaml.safe_dump(winner["parameters"], sort_keys=False))
    (output / "stage1_best_config.yaml").write_text(
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), sort_keys=False))
    (output / "stage1_summary.json").write_text(json.dumps(winner, indent=2, sort_keys=True, default=str)+"\n")


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command", required=True)
    run=sub.add_parser("run")
    for p in (run,):
        p.add_argument("--sqlite-path", required=True); p.add_argument("--study-name", required=True)
        p.add_argument("--base-config", required=True); p.add_argument("--output-dir", required=True)
    run.add_argument("--top-count", type=int, default=3); run.add_argument("--candidate-rank", type=int, required=True)
    run.add_argument("--train-indices", required=True); run.add_argument("--validation-indices", required=True)
    run.add_argument("--topology", choices=("ttbar1L", "ttH"), required=True)
    run.add_argument("--split-cache", required=True); run.add_argument("--subset-manifest", required=True)
    run.add_argument("--dataset-root", required=True); run.add_argument("--dataset-name", required=True)
    run.add_argument("--num-workers", type=int)
    run.add_argument("--max-epochs", type=int, default=100); run.add_argument("--patience", type=int, default=15)
    run.add_argument("--validate-only", action="store_true")
    run.add_argument("--limit-train-batches", type=int)
    run.add_argument("--limit-val-batches", type=int)
    run.add_argument("--final-validation-indices")
    select_parser=sub.add_parser("select")
    select_parser.add_argument("--study-name", required=True); select_parser.add_argument("--base-config", required=True)
    select_parser.add_argument("--output-dir", required=True); select_parser.add_argument("--selection-dir", required=True)
    select_parser.add_argument("--sqlite-path", default="unused")
    args=parser.parse_args(); setup_torch_runtime()
    run_candidate(args) if args.command == "run" else select(args)


if __name__ == "__main__": main()
