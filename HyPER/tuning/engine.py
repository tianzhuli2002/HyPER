"""Persistent Optuna workers using HyPER's ordinary training entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
import traceback
import warnings
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import optuna
import yaml
from omegaconf import OmegaConf

from HyPER.configuration import set_task_mode
from HyPER.train import run_training, setup_torch_runtime
from .coefficients import loss_coefficients
from .configuration import configure_tuning_data_isolation
from .monitor import BestObservedValidation
from .search_space import sample_and_apply, validate_search_space
from .subsets import prepare_subsets


STAGE_MONITORS = {
    1: ("val_reconstruction_loss", "min"),
    2: ("val_reco_mean_role_top1", "max"),
    3: ("val_auc", "max"),
}
BETA_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.30, 0.50, 0.70]
ALPHA_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
# Long GPU trials can legitimately spend substantially longer than the normal
# Optuna heartbeat grace period between validation reports.  Keep heartbeats so
# genuinely dead Slurm workers are eventually recovered, but never finalise a
# live Stage-2/3 trial within its expected runtime.
OPTUNA_HEARTBEAT_INTERVAL_SECONDS = 60
OPTUNA_GRACE_PERIOD_SECONDS = 21600
EXPECTED_MAX_SINGLE_TUNING_TRIAL_SECONDS = 7200


def _atomic_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def load_configs(base_config: str, tuning_config: str, overrides=None):
    base = OmegaConf.load(base_config)
    tuning_path = Path(tuning_config).expanduser().resolve()
    common_path = tuning_path.parent / "common.yaml"
    common = OmegaConf.load(common_path) if common_path.is_file() else OmegaConf.create({})
    tune = OmegaConf.load(tuning_path)
    return OmegaConf.merge(
        base, common, tune, OmegaConf.from_dotlist(list(overrides or []))
    )


def configure_stage(cfg, stage: int, trial, frozen_alpha=None):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if stage == 1:
        set_task_mode(cfg, "reconstruction")
        cfg.loss.edge_weight = 0.5
        cfg.loss.hyperedge_weight = 0.5
        cfg.loss.classification_weight = 0.0
        cfg, params = sample_and_apply(trial, cfg, cfg.tuning.search_space)
    elif stage == 2:
        alpha = float(trial.suggest_categorical("alpha", list(cfg.tuning.alpha_grid)))
        params = {"alpha": alpha}
        set_task_mode(cfg, "joint")
        for key, value in loss_coefficients(alpha, 0.05).items():
            cfg.loss[key] = value
    elif stage == 3:
        if frozen_alpha is None:
            raise ValueError("Stage 3 requires the selected Stage 2 alpha.")
        beta = float(trial.suggest_categorical("beta", list(cfg.tuning.beta_grid)))
        params = {"beta": beta, "frozen_alpha": float(frozen_alpha)}
        set_task_mode(cfg, "joint")
        for key, value in loss_coefficients(float(frozen_alpha), beta).items():
            cfg.loss[key] = value
    else:
        raise ValueError(f"Unknown tuning stage {stage}.")
    if "alpha" in cfg.loss or "beta" in cfg.loss:
        raise ValueError("Resolved production loss config must not contain alpha/beta keys.")
    return cfg, params


def build_storage(sqlite_path: str):
    path = Path(sqlite_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(
        url=f"sqlite:///{path}",
        engine_kwargs={"connect_args": {"timeout": 120}},
        heartbeat_interval=OPTUNA_HEARTBEAT_INTERVAL_SECONDS,
        grace_period=OPTUNA_GRACE_PERIOD_SECONDS,
    )


def build_study(cfg, stage: int):
    storage = build_storage(str(cfg.tuning.sqlite_path))
    monitor, mode = STAGE_MONITORS[stage]
    if stage == 1:
        sampler = optuna.samplers.TPESampler(
            seed=int(cfg.tuning.seed), multivariate=True,
            n_startup_trials=int(cfg.tuning.startup_trials),
        )
    else:
        # Explicit WAITING trials below define the exact resumable grid.  A
        # RandomSampler is intentional: Optuna's GridSampler cannot finalize
        # externally enqueued retry trials because they have no private grid_id.
        sampler = optuna.samplers.RandomSampler(seed=int(cfg.tuning.seed))
    pruner_name = str(cfg.tuning.pruner).lower()
    if pruner_name in {"none", "nop"}:
        pruner = optuna.pruners.NopPruner()
    elif pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(cfg.tuning.startup_trials),
            n_warmup_steps=int(cfg.tuning.get("pruning_warmup_epochs", 4)), interval_steps=1,
        )
    else:
        raise ValueError(f"Unsupported pruner {cfg.tuning.pruner!r}.")
    study = optuna.create_study(
        study_name=str(cfg.tuning.study_name), storage=storage,
        load_if_exists=True, direction="minimize" if mode == "min" else "maximize",
        sampler=sampler, pruner=pruner,
    )
    if stage == 1:
        metadata = stage1_budget_metadata(cfg)
        existing = study.user_attrs.get("stage1_budget_metadata")
        if existing is not None and existing != metadata:
            raise RuntimeError("Refusing Stage-1 study with incompatible stored budget metadata; use a fresh campaign root.")
        if existing is None:
            study.set_user_attr("stage1_budget_metadata", metadata)
    return study


def stage1_budget_metadata(cfg):
    manifest_value = cfg.tuning.get("subset_manifest_path")
    if not manifest_value:
        # Unit/integration callers may construct transient index files directly.
        data = {"train_subset_hash": None, "validation_subset_hash": None}
    else:
        manifest = Path(str(manifest_value)).resolve()
        data = json.loads(manifest.read_text())
    search = json.dumps(OmegaConf.to_container(cfg.tuning.search_space, resolve=True), sort_keys=True)
    return {"max_epochs": int(cfg.tuning.max_epochs), "min_epochs": int(cfg.tuning.min_epochs),
            "target_count": int(cfg.tuning.target_completed_trials), "monitor": "val_reconstruction_loss",
            "train_subset_hash": data["train_subset_hash"], "validation_subset_hash": data["validation_subset_hash"],
            "search_space_hash": hashlib.sha256(search.encode()).hexdigest()}


def completed_count(study) -> int:
    return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.get_trials(deepcopy=False))


def grid_status(study, stage: int, configured_values) -> dict:
    """Describe exact grid coverage without treating duplicate rows as coverage."""
    key = "alpha" if stage == 2 else "beta"
    expected = [float(value) for value in configured_values]
    by_state = {name: [] for name in ("complete", "failed", "running", "waiting")}
    complete_trials: dict[float, list[int]] = {value: [] for value in expected}
    for trial in study.get_trials(deepcopy=False):
        parameter = trial.params.get(key)
        if parameter is None:
            parameter = trial.system_attrs.get("fixed_params", {}).get(key)
        if parameter is None:
            continue
        value = float(parameter)
        if value not in complete_trials:
            raise RuntimeError(
                f"Study {study.study_name} contains unexpected {key}={value}; expected {expected}."
            )
        state = trial.state.name.lower()
        mapped = "failed" if state == "fail" else state
        if mapped in by_state:
            by_state[mapped].append({"value": value, "trial": trial.number})
        if state == "complete":
            complete_trials[value].append(trial.number)
    unique = [value for value in expected if complete_trials[value]]
    return {
        "parameter": key,
        "configured_values": expected,
        "complete_rows": sum(len(numbers) for numbers in complete_trials.values()),
        "unique_completed_values": unique,
        "missing_values": [value for value in expected if not complete_trials[value]],
        "duplicates": {str(value): numbers for value, numbers in complete_trials.items() if len(numbers) > 1},
        "failed_values": by_state["failed"],
        "running_values": by_state["running"],
        "waiting_values": by_state["waiting"],
    }


def completed_grid_count(study, stage: int, configured_values) -> int:
    return len(grid_status(study, stage, configured_values)["unique_completed_values"])


def enqueue_missing_grid_values(study, stage: int, configured_values) -> list[float]:
    """Queue only values with no completed, running, or waiting trial."""
    status = grid_status(study, stage, configured_values)
    occupied = {
        float(entry["value"])
        for name in ("running_values", "waiting_values")
        for entry in status[name]
    }
    queued = []
    key = status["parameter"]
    for value in status["missing_values"]:
        if value not in occupied:
            study.enqueue_trial({key: value})
            queued.append(value)
    return queued


def objective_factory(cfg, stage: int, frozen_alpha=None):
    output_root = Path(str(cfg.tuning.output_dir)).expanduser().resolve()

    def objective(trial):
        trial_dir = output_root / "trials" / f"trial_{trial.number:06d}"
        if trial_dir.exists() and any(trial_dir.iterdir()):
            raise FileExistsError(f"Trial output directory already contains data: {trial_dir}")
        trial_dir.mkdir(parents=True, exist_ok=False)
        monitor, mode = STAGE_MONITORS[stage]
        started = time.time()
        try:
            trial_cfg, params = configure_stage(cfg, stage, trial, frozen_alpha=frozen_alpha)
            configure_tuning_data_isolation(
                trial_cfg,
                canonical_split_path=trial_cfg.dataset.split.cache_path,
                train_indices_path=trial_cfg.tuning.train_indices_file,
                validation_indices_path=trial_cfg.tuning.validation_indices_file,
            )
            trial_cfg.tuning.checkpointing = stage in (2, 3)
            trial_cfg.tuning.monitor = monitor
            trial_cfg.tuning.direction = mode
            trial_cfg.tuning.output_dir = str(trial_dir)
            trial_cfg.paths.training_manifest = str(trial_dir / "exploratory_training_manifest.json")
            trial_cfg.paths.savedir = str(trial_dir / "logs")
            trial_cfg.trainer.run_test_after_fit = False
            trial_cfg.trainer.epochs = int(cfg.tuning.max_epochs)
            if stage == 1:
                trial_cfg.trainer.min_epochs = int(cfg.tuning.min_epochs)
            trial_cfg.early_stopping.enabled = True
            trial_cfg.early_stopping.monitor = monitor
            trial_cfg.early_stopping.mode = mode
            trial_cfg.early_stopping.patience = int(cfg.tuning.early_stopping_patience)
            trial_cfg.lr_scheduler.monitor = monitor
            trial_cfg.lr_scheduler.mode = mode
            if stage in (2, 3):
                OmegaConf.update(
                    trial_cfg, "validation_diagnostics.role_ranking_enabled", True,
                    merge=False, force_add=True,
                )
                OmegaConf.update(
                    trial_cfg, "validation_diagnostics.classification_metrics_enabled", True,
                    merge=False, force_add=True,
                )
                OmegaConf.update(
                    trial_cfg, "validation_diagnostics.every_n_epochs", 1,
                    merge=False, force_add=True,
                )
                OmegaConf.update(
                    trial_cfg, "validation_diagnostics.max_events", 10000,
                    merge=False, force_add=True,
                )
            _atomic_yaml(trial_dir / "sampled_parameters.yaml", params)
            _atomic_yaml(
                trial_dir / "resolved_trial_config.yaml",
                OmegaConf.to_container(trial_cfg, resolve=True),
            )
            recorder = BestObservedValidation(monitor, mode, trial=trial)
            budget = {"max_epochs": int(trial_cfg.trainer.epochs),
                      "min_epochs": int(trial_cfg.trainer.get("min_epochs", 0)),
                      "early_stopping_patience": int(trial_cfg.early_stopping.patience),
                      "monitor": monitor, "direction": mode}
            trial.set_user_attr("training_budget", budget)
            metrics = run_training(
                trial_cfg, extra_callbacks=[recorder],
                logger_name=f"trial_{trial.number:06d}", return_metrics=True,
            )
            training_manifest = trial_dir / "exploratory_training_manifest.json"
            if training_manifest.is_file():
                manifest_data = json.loads(training_manifest.read_text())
                manifest_data["stage1_training_budget"] = budget
                _atomic_json(training_manifest, manifest_data)
            summary = recorder.summary()
            summary.update(
                {
                    "trial_number": trial.number,
                    "state": "complete",
                    "sampled_parameters": params,
                    "wall_seconds": time.time() - started,
                    "worker": {"hostname": socket.gethostname(), "pid": os.getpid(),
                               "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                               "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID")},
                    "ordinary_training_metrics": metrics,
                    "training_budget": budget,
                }
            )
            recorder._write_trial_attributes()
            trial.set_user_attr("trial_directory", str(trial_dir))
            _atomic_json(trial_dir / "trial_metrics.json", summary)
            return float(summary["best_observed_monitor_value"])
        except optuna.TrialPruned:
            summary = recorder.summary() if "recorder" in locals() and recorder.observations else {
                "monitor": monitor, "mode": mode, "pruned": True,
            }
            summary.update({"trial_number": trial.number, "state": "pruned", "wall_seconds": time.time() - started})
            if "recorder" in locals() and recorder.observations:
                recorder._write_trial_attributes()
            _atomic_json(trial_dir / "trial_metrics.json", summary)
            raise
        except Exception as exc:
            _atomic_json(
                trial_dir / "trial_metrics.json",
                {"trial_number": trial.number, "state": "failed", "exception": repr(exc),
                 "traceback": traceback.format_exc(), "wall_seconds": time.time() - started},
            )
            raise
    return objective


def _terminal_counts(study):
    return sum(trial.state in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                               optuna.trial.TrialState.FAIL} for trial in study.get_trials(deepcopy=False))


def reserve_stage1_attempt(sqlite_path, study_name, limit):
    """Reserve one attempt while holding SQLite's cross-process write lock."""
    connection = sqlite3.connect(str(Path(sqlite_path).resolve()), timeout=120, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS stage1_attempt_reservations ("
            "study_name TEXT NOT NULL, slot INTEGER NOT NULL, worker TEXT NOT NULL, "
            "status TEXT NOT NULL, trial_number INTEGER, created_unix REAL NOT NULL, "
            "finished_unix REAL, PRIMARY KEY(study_name, slot))"
        )
        columns = [row[1] for row in connection.execute("PRAGMA table_info(stage1_attempt_reservations)")]
        expected = ["study_name", "slot", "worker", "status", "trial_number", "created_unix", "finished_unix"]
        if columns != expected:
            raise RuntimeError(
                "Incompatible Stage-1 attempt reservation table; use a fresh campaign ID. "
                f"expected={expected}, observed={columns}"
            )
        study_row = connection.execute(
            "SELECT study_id FROM studies WHERE study_name=?", (study_name,)
        ).fetchone()
        if study_row is None:
            raise RuntimeError(f"Optuna study {study_name!r} does not exist in {sqlite_path}.")
        attempted = int(connection.execute(
            "SELECT COUNT(*) FROM trials WHERE study_id=?", (study_row[0],)
        ).fetchone()[0])
        reserved = int(connection.execute(
            "SELECT COUNT(*) FROM stage1_attempt_reservations WHERE study_name=? AND status='reserved'",
            (study_name,),
        ).fetchone()[0])
        if attempted + reserved >= int(limit):
            connection.execute("COMMIT"); return None
        next_slot = int(connection.execute(
            "SELECT COALESCE(MAX(slot), -1) + 1 FROM stage1_attempt_reservations WHERE study_name=?",
            (study_name,),
        ).fetchone()[0])
        connection.execute(
            "INSERT INTO stage1_attempt_reservations VALUES (?,?,?,?,?,?,?)",
            (study_name, next_slot, f"{socket.gethostname()}:{os.getpid()}",
             "reserved", None, time.time(), None),
        )
        connection.execute("COMMIT"); return next_slot
    except Exception:
        connection.execute("ROLLBACK"); raise
    finally: connection.close()


def finish_stage1_attempt(sqlite_path, study_name, slot, *, trial_number=None):
    """Mark a reservation consumed by a trial, or explicitly abandoned."""
    status = "consumed" if trial_number is not None else "abandoned"
    connection = sqlite3.connect(str(Path(sqlite_path).resolve()), timeout=120, isolation_level=None)
    try:
        connection.execute("BEGIN IMMEDIATE")
        changed = connection.execute(
            "UPDATE stage1_attempt_reservations SET status=?, trial_number=?, finished_unix=? "
            "WHERE study_name=? AND slot=? AND status='reserved'",
            (status, trial_number, time.time(), study_name, int(slot)),
        ).rowcount
        if changed != 1:
            raise RuntimeError(
                f"Stage-1 reservation {study_name}:{slot} is missing or no longer owned as reserved."
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def remaining_walltime(deadline=None, now=None, worker_start=None):
    now = time.time() if now is None else float(now)
    worker_start = now if worker_start is None else float(worker_start)
    raw = os.getenv("SLURM_JOB_END_TIME")
    if raw:
        if not raw.isdigit():
            raise ValueError(f"SLURM_JOB_END_TIME must be an integer Unix timestamp, got {raw!r}.")
        value = int(raw)
        if value <= worker_start:
            raise ValueError(f"SLURM_JOB_END_TIME={value} is not later than worker start {worker_start}.")
        return float(value) - now, "SLURM_JOB_END_TIME"
    if deadline is not None:
        if isinstance(deadline, bool) or int(deadline) != float(deadline):
            raise ValueError(f"worker deadline must be an integer Unix timestamp, got {deadline!r}.")
        value = int(deadline)
        if value <= worker_start:
            raise ValueError(f"worker deadline {value} is not later than worker start {worker_start}.")
        return float(value) - now, "explicit_deadline"
    if os.getenv("SLURM_JOB_ID"):
        raise RuntimeError(
            "Production Stage-1 Slurm worker has no usable SLURM_JOB_END_TIME or --worker-deadline."
        )
    warnings.warn("No Slurm end time or explicit worker deadline; walltime guard cannot infer remaining time.")
    return None, "unavailable"


def run_worker(cfg, stage: int, frozen_alpha=None, worker_deadline=None, summary_path=None):
    worker_started = time.time()
    study = build_study(cfg, stage)
    target = int(cfg.tuning.target_completed_trials)
    grid_values = None
    if stage in (2, 3):
        grid_values = list(cfg.tuning.alpha_grid if stage == 2 else cfg.tuning.beta_grid)
        if target != len({float(value) for value in grid_values}):
            raise ValueError(
                f"Stage {stage} target_completed_trials={target} does not match unique grid size "
                f"{len(set(map(float, grid_values)))}."
            )
        enqueue_missing_grid_values(study, stage, grid_values)
    max_attempts = int(cfg.tuning.get("trials_per_worker", target))
    campaign_max_attempts = int(cfg.tuning.get("max_attempted_trials", target))
    minimum_remaining = int(cfg.tuning.get("min_remaining_seconds", 0))
    attempts = 0
    local_trials = []
    stop_reason = "worker_attempt_limit"
    # Constructing an objective requires full training configuration; do it only
    # after the pre-trial walltime check has allowed a trial to start.
    objective = None
    coverage = lambda: (completed_count(study) if grid_values is None
                        else completed_grid_count(study, stage, grid_values))
    while coverage() < target and attempts < max_attempts:
        if grid_values is not None:
            enqueue_missing_grid_values(study, stage, grid_values)
            status = grid_status(study, stage, grid_values)
            if not status["waiting_values"] and status["missing_values"]:
                running = {float(entry["value"]) for entry in status["running_values"]}
                if set(status["missing_values"]).issubset(running):
                    stop_reason = "missing_grid_values_already_in_flight"
                    break
        if stage == 1:
            remaining, source = remaining_walltime(
                worker_deadline, worker_start=worker_started
            )
            if remaining is not None and remaining < minimum_remaining:
                stop_reason = f"walltime_guard:{source}"
                break
            reservation = reserve_stage1_attempt(
                cfg.tuning.sqlite_path, study.study_name, campaign_max_attempts
            )
            if reservation is None:
                stop_reason = "maximum_attempted_trials_reserved"
                break
        if objective is None:
            objective = objective_factory(cfg, stage, frozen_alpha=frozen_alpha)
        new_trials = []
        try:
            before = {trial.number for trial in study.get_trials(deepcopy=False)}
            study.optimize(objective, n_trials=1, n_jobs=1, catch=(Exception,))
            new_trials = [trial.number for trial in study.get_trials(deepcopy=False)
                          if trial.number not in before]
            local_trials.extend(new_trials)
        except RuntimeError as exc:
            if "Study.stop" not in str(exc) and "grid" not in str(exc).lower():
                raise
            break
        finally:
            if stage == 1:
                finish_stage1_attempt(
                    cfg.tuning.sqlite_path, study.study_name, reservation,
                    trial_number=new_trials[0] if len(new_trials) == 1 else None,
                )
        attempts += 1
    if coverage() >= target:
        stop_reason = "completed_target_reached"
    output_dir = Path(str(cfg.tuning.get("output_dir", Path(summary_path).parent if summary_path else ".")))
    write_study_summary(study, output_dir, stage, configured_values=grid_values)
    counts = {state.name.lower(): 0 for state in optuna.trial.TrialState}
    for trial in study.get_trials(deepcopy=False): counts[trial.state.name.lower()] += 1
    destination = Path(summary_path) if summary_path else output_dir / "workers" / f"worker_{os.getenv('SLURM_ARRAY_TASK_ID', os.getpid())}.json"
    local = [trial for trial in study.get_trials(deepcopy=False) if trial.number in local_trials]
    _atomic_json(destination, {
        "topology": os.getenv("TOPOLOGY", str(cfg.get("config_name", "unknown"))),
        "stage": stage,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "worker_start_time": datetime.fromtimestamp(worker_started, timezone.utc).isoformat(),
        "worker_end_time": datetime.now(timezone.utc).isoformat(),
        "stop_reason": stop_reason,
        "minimum_remaining_seconds": minimum_remaining,
        "local": {
            "attempted": len(local_trials), "trial_numbers": local_trials,
            "complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in local),
            "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in local),
            "failed": sum(t.state == optuna.trial.TrialState.FAIL for t in local),
        },
        "campaign": {
            "waiting": counts["waiting"], "running": counts["running"],
            "complete": counts["complete"], "pruned": counts["pruned"],
            "failed": counts["fail"],
            "attempted_total": sum(counts[name] for name in ("running", "complete", "pruned", "fail")),
        },
    })


def write_study_summary(study, output_dir: Path, stage: int, configured_values=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in study.get_trials(deepcopy=False):
        rows.append({
            "trial_number": trial.number, "state": trial.state.name.lower(), "value": trial.value,
            **{f"param.{key}": value for key, value in trial.params.items()},
            **{f"user.{key}": value for key, value in trial.user_attrs.items()},
        })
    fields = sorted({key for row in rows for key in row}) if rows else ["trial_number", "state", "value"]
    csv_path = output_dir / f"stage{stage}_trials.csv"
    csv_tmp = csv_path.with_suffix(f".csv.tmp-{os.getpid()}")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    csv_tmp.replace(csv_path)
    counts = {state.name.lower(): 0 for state in optuna.trial.TrialState}
    for trial in study.get_trials(deepcopy=False):
        counts[trial.state.name.lower()] += 1
    payload = {
        "study_name": study.study_name, "stage": stage, "trial_counts": counts,
        "completed_target": completed_count(study),
    }
    if stage in (2, 3):
        values = configured_values or (ALPHA_GRID if stage == 2 else BETA_GRID)
        payload["grid_coverage"] = grid_status(study, stage, values)
        payload["completed_target"] = len(payload["grid_coverage"]["unique_completed_values"])
    _atomic_json(output_dir / f"stage{stage}_study_status.json", payload)


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-subsets")
    prep.add_argument("--split-cache", required=True); prep.add_argument("--source-h5", required=True)
    prep.add_argument("--output-dir", required=True); prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--train-count", type=int)
    prep.add_argument("--validation-count", type=int)
    prep.add_argument("--train-fraction", type=float)
    prep.add_argument("--validation-fraction", type=float)
    prep.add_argument("--graph-config")
    for command in ("worker", "init"):
        worker = sub.add_parser(command)
        worker.add_argument("--base-config", required=True); worker.add_argument("--tuning-config", required=True)
        worker.add_argument("--stage", required=True, type=int, choices=(1, 2, 3))
        worker.add_argument("--frozen-alpha", type=float)
        worker.add_argument("--worker-deadline", type=float)
        worker.add_argument("--worker-summary-path")
        worker.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare-subsets":
        result = prepare_subsets(args.split_cache, args.source_h5, args.output_dir, seed=args.seed,
                                 train_count=args.train_count, validation_count=args.validation_count,
                                 train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
                                 graph_config=args.graph_config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    setup_torch_runtime()
    cfg = load_configs(args.base_config, args.tuning_config, args.override)
    validate_search_space(cfg, cfg.tuning.search_space) if args.stage == 1 else None
    if args.command == "init":
        study = build_study(cfg, args.stage)
        values = None
        if args.stage in (2, 3):
            values = list(cfg.tuning.alpha_grid if args.stage == 2 else cfg.tuning.beta_grid)
        write_study_summary(study, Path(str(cfg.tuning.output_dir)), args.stage, configured_values=values)
        print(f"Initialized persistent study {study.study_name} at {cfg.tuning.sqlite_path}")
        return
    run_worker(cfg, args.stage, frozen_alpha=args.frozen_alpha,
               worker_deadline=args.worker_deadline, summary_path=args.worker_summary_path)


if __name__ == "__main__":
    main()
