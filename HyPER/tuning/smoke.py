"""Standalone, real Stage-1 GPU smoke on deterministic canonical subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import optuna
import torch
from omegaconf import OmegaConf

from .coefficients import loss_coefficients
from .engine import load_configs, run_worker
from HyPER.train import setup_torch_runtime
from .selection import _stage2_row, _stage3_row, choose_stage2, choose_stage3, select_stage
from .subsets import prepare_subsets


SCHEMA_VERSION = 1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured(args, output: Path):
    cfg = load_configs(args.base_config, args.tuning_config)
    cfg.dataset.root = str(Path(args.graph_root).resolve())
    cfg.dataset.source_h5_path = str(Path(args.source_h5).resolve())
    cfg.dataset.train_set = args.dataset_name
    cfg.dataset.predict_set = args.dataset_name
    cfg.dataset.split.cache_path = str(Path(args.split_cache).resolve())
    cfg.dataset.batch_size = args.batch_size
    cfg.dataset.num_workers = args.num_workers
    cfg.dataset.pin_memory = True
    cfg.dataset.persistent_workers = args.num_workers > 0
    cfg.trainer.accelerator = "gpu"
    cfg.trainer.devices = 1
    cfg.trainer.precision = "32-true"
    cfg.trainer.limit_train_batches = args.limit_train_batches
    cfg.trainer.limit_val_batches = args.limit_val_batches
    cfg.trainer.enable_progress_bar = False
    cfg.tuning.train_indices_file = str((output / "subsets/tuning_train_indices.npy").resolve())
    cfg.tuning.validation_indices_file = str((output / "subsets/tuning_validation_indices.npy").resolve())
    cfg.tuning.output_dir = str((output / "stage1").resolve())
    cfg.tuning.sqlite_path = str((output / "stage1/study.sqlite3").resolve())
    cfg.tuning.study_name = f"standalone_smoke_{args.topology}_stage1_{os.getenv('SLURM_JOB_ID', 'manual')}"
    cfg.tuning.max_epochs = args.epochs
    cfg.tuning.early_stopping_patience = 1
    cfg.tuning.startup_trials = 1
    cfg.tuning.pruner = "median"
    cfg.tuning.target_completed_trials = args.required_completed_trials
    cfg.tuning.trials_per_worker = args.max_attempted_trials
    cfg.tuning.search_space["model.message_feats"].values = [32]
    cfg.tuning.search_space["model.contraction_feats"].values = [64]
    return cfg


def main():
    setup_torch_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, choices=("ttbar1L", "ttH"))
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--tuning-config", required=True)
    parser.add_argument("--graph-root", required=True)
    parser.add_argument("--canonical-graph-root", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--split-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit-train-batches", type=int, default=2)
    parser.add_argument("--limit-val-batches", type=int, default=2)
    parser.add_argument("--required-completed-trials", type=int, default=2)
    parser.add_argument("--max-attempted-trials", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        subset = prepare_subsets(
            args.split_cache,
            args.source_h5,
            output / "subsets",
            seed=42,
            train_fraction=0.002,
            validation_fraction=0.005,
        )
        cfg = configured(args, output)
        smoke_base = output / "smoke_base.yaml"
        smoke_base.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        run_worker(cfg, 1)
        selection_dir = output / "stage1/selection"
        select_stage(
            cfg.tuning.sqlite_path,
            cfg.tuning.study_name,
            1,
            smoke_base,
            selection_dir,
            minimum_completed=args.required_completed_trials,
        )
        selected_stage1 = OmegaConf.load(selection_dir / "stage1_best_config.yaml")
        stage23 = []
        previous = selected_stage1
        for stage, value in ((2, 0.20), (3, 0.05)):
            stage_cfg = OmegaConf.merge(previous, OmegaConf.load(args.tuning_config))
            stage_cfg.dataset.root = cfg.dataset.root
            stage_cfg.dataset.source_h5_path = cfg.dataset.source_h5_path
            stage_cfg.dataset.train_set = cfg.dataset.train_set
            stage_cfg.dataset.predict_set = cfg.dataset.predict_set
            stage_cfg.dataset.split.cache_path = cfg.dataset.split.cache_path
            stage_cfg.dataset.batch_size = args.batch_size
            stage_cfg.dataset.num_workers = args.num_workers
            stage_cfg.dataset.pin_memory = True
            stage_cfg.dataset.persistent_workers = args.num_workers > 0
            stage_cfg.trainer.accelerator = "gpu"
            stage_cfg.trainer.devices = 1
            stage_cfg.trainer.precision = "32-true"
            stage_cfg.trainer.limit_train_batches = args.limit_train_batches
            stage_cfg.trainer.limit_val_batches = args.limit_val_batches
            stage_cfg.trainer.enable_progress_bar = False
            stage_cfg.tuning.train_indices_file = cfg.tuning.train_indices_file
            stage_cfg.tuning.validation_indices_file = cfg.tuning.validation_indices_file
            stage_cfg.tuning.output_dir = str((output / f"stage{stage}").resolve())
            stage_cfg.tuning.sqlite_path = str((output / f"stage{stage}/study.sqlite3").resolve())
            stage_cfg.tuning.study_name = f"standalone_smoke_{args.topology}_stage{stage}_{os.getenv('SLURM_JOB_ID', 'manual')}"
            stage_cfg.tuning.max_epochs = args.epochs
            stage_cfg.tuning.early_stopping_patience = 1
            stage_cfg.tuning.pruner = "nop"
            stage_cfg.tuning.target_completed_trials = 1
            stage_cfg.tuning.trials_per_worker = 1
            if stage == 2:
                stage_cfg.tuning.alpha_grid = [value]
                run_worker(stage_cfg, stage)
            else:
                stage_cfg.tuning.beta_grid = [value]
                run_worker(stage_cfg, stage, frozen_alpha=0.20)
            stage_study = optuna.load_study(
                study_name=str(stage_cfg.tuning.study_name),
                storage=f"sqlite:///{Path(stage_cfg.tuning.sqlite_path).resolve()}",
            )
            completed_stage = [trial for trial in stage_study.trials if trial.state.name == "COMPLETE"]
            if len(completed_stage) != 1:
                raise RuntimeError(f"Stage {stage} smoke requires one completed trial, observed {len(completed_stage)}.")
            row = (_stage2_row if stage == 2 else _stage3_row)(completed_stage[0])
            if stage == 2:
                winner = choose_stage2([row])
                next_cfg = OmegaConf.create(OmegaConf.to_container(previous, resolve=True))
                for key, weight in loss_coefficients(value, .05).items(): next_cfg.loss[key] = weight
                next_cfg.classification.enabled = True; next_cfg.reconstruction.enabled = True
                ancestry = dict(next_cfg.get("tuning_ancestry", {})); ancestry["stage2_alpha"] = value
                next_cfg.tuning_ancestry = ancestry
                previous = next_cfg
            else:
                winner, _ = choose_stage3([row])
            stage23.append({"stage": stage, "selected_trial": winner["trial_number"],
                            "checkpoint": winner["checkpoint"], "metrics": winner})
        generated = output / "stage2/generated_stage3_base.yaml"
        generated.write_text(OmegaConf.to_yaml(previous, resolve=True), encoding="utf-8")
        storage = f"sqlite:///{Path(cfg.tuning.sqlite_path).resolve()}"
        study = optuna.load_study(study_name=str(cfg.tuning.study_name), storage=storage)
        counts = {state.name.lower(): 0 for state in optuna.trial.TrialState}
        for trial in study.get_trials(deepcopy=False):
            counts[trial.state.name.lower()] += 1
        if counts["complete"] < args.required_completed_trials:
            raise RuntimeError(
                f"Stage-1 smoke requires {args.required_completed_trials} completed trials; counts={counts}."
            )
        completed = [trial for trial in study.get_trials(deepcopy=False) if trial.state.name == "COMPLETE"]
        best_trial = min(completed, key=lambda trial: float(trial.value))
        trial_dir = Path(best_trial.user_attrs["trial_directory"])
        metrics = json.loads((trial_dir / "trial_metrics.json").read_text(encoding="utf-8"))
        manifests = []
        result_paths = [str(Path(cfg.tuning.sqlite_path).resolve()), str(selection_dir.resolve())]
        for trial in completed:
            path = Path(trial.user_attrs["trial_directory"])
            manifest_path = path / "exploratory_training_manifest.json"
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            result_paths.extend([str((path / "trial_metrics.json").resolve()), str(manifest_path.resolve())])
        graph_manifest = Path(args.graph_root) / f"{args.dataset_name}.db.manifest.json"
        canonical_db = Path(args.canonical_graph_root).resolve() / f"{args.dataset_name}.db"
        marker = {
            "schema_version": SCHEMA_VERSION,
            "topology": args.topology,
            "job_id": os.getenv("SLURM_JOB_ID"),
            "hostname": socket.gethostname(),
            "gpu_name": torch.cuda.get_device_name(0),
            "graph_db_path": str(canonical_db),
            "graph_db_manifest_hash": file_hash(graph_manifest),
            "canonical_split_path": str(Path(args.split_cache).resolve()),
            "train_subset_hash": subset["train_subset_hash"],
            "validation_subset_hash": subset["validation_subset_hash"],
            "test_overlap_count": 0,
            "test_overlap_basis": "subsets validated as members of canonical train/validation; test array unopened",
            "completed_trials": counts["complete"],
            "failed_trials": counts["fail"],
            "pruned_trials": counts["pruned"],
            "monitor": "val_reconstruction_loss",
            "direction": "min",
            "best_objective_value": metrics["best_observed_monitor_value"],
            "best_objective_epoch": metrics["best_observed_monitor_epoch"],
            "final_objective_value": metrics["final_observed_monitor_value"],
            "final_objective_epoch": metrics["final_observed_monitor_epoch"],
            "gradients_finite": all(item["runtime"]["gradients_finite"] for item in manifests),
            "peak_gpu_memory_mb": max(item["runtime"]["peak_gpu_memory_mb"] for item in manifests),
            "fit_wall_seconds": sum(item["runtime"]["fit_wall_seconds"] for item in manifests),
            "events_per_second": sum(item["runtime"]["training_events_seen"] for item in manifests)
            / sum(item["runtime"]["fit_wall_seconds"] for item in manifests),
            "local_graph_stage_path": str(Path(args.graph_root).resolve()),
            "study_name": str(cfg.tuning.study_name),
            "study_sqlite_path": str(Path(cfg.tuning.sqlite_path).resolve()),
            "result_paths": sorted(set(result_paths)),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "stage23_smoke": stage23,
            "generated_stage3_base": str(generated.resolve()),
        }
        (output / "smoke_success.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        (output / "smoke_failure.json").write_text(
            json.dumps({"error": repr(exc)}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
