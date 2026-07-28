"""Physics-metric stage selection and explicit final-config generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import optuna
import yaml
from omegaconf import OmegaConf

from .coefficients import loss_coefficients
from .engine import ALPHA_GRID, BETA_GRID


def _storage(path):
    return f"sqlite:///{Path(path).expanduser().resolve()}"


def _write_yaml(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _finite(value, context):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing or invalid {context}: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {context}: {result}")
    return result


def _trial_payload(trial, *, require_checkpoint=True):
    directory = trial.user_attrs.get("trial_directory")
    if not directory:
        raise RuntimeError(f"Trial {trial.number} has no trial_directory user attribute.")
    path = Path(directory) / "trial_metrics.json"
    if not path.is_file():
        raise RuntimeError(f"Trial {trial.number} has no metrics file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("best_epoch_metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"Trial {trial.number} lacks best_epoch_metrics in {path}.")
    forbidden = sorted(name for name in metrics if str(name).lower().startswith("test"))
    if forbidden:
        raise RuntimeError(f"Trial {trial.number} exposes forbidden test-set selector metrics: {forbidden}")
    checkpoint = None
    if require_checkpoint:
        paths = payload.get("ordinary_training_metrics", {}).get("checkpoint_paths", [])
        valid = [Path(item).expanduser().resolve() for item in paths if item and Path(item).is_file()]
        if len(valid) != 1:
            raise RuntimeError(
                f"Trial {trial.number} requires exactly one retained checkpoint; observed {paths}."
            )
        checkpoint = str(valid[0])
    return payload, metrics, checkpoint, str(path.resolve())


def _canonical_grid_trials(study, stage):
    key = "alpha" if stage == 2 else "beta"
    expected = [float(value) for value in (ALPHA_GRID if stage == 2 else BETA_GRID)]
    grouped = {value: [] for value in expected}
    complete_rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if key not in trial.params:
            raise RuntimeError(f"Completed trial {trial.number} lacks required parameter {key!r}.")
        value = float(trial.params[key])
        if value not in grouped:
            raise RuntimeError(f"Completed trial {trial.number} has unexpected {key}={value}; expected {expected}.")
        _finite(trial.value, f"trial {trial.number} objective")
        grouped[value].append(trial)
        complete_rows.append(trial)
    missing = [value for value, trials in grouped.items() if not trials]
    if missing:
        raise RuntimeError(
            f"Study {study.study_name} is missing completed {key} values {missing}; "
            f"complete_rows={len(complete_rows)}, unique_complete={len(expected) - len(missing)}."
        )
    canonical = {value: min(trials, key=lambda trial: trial.number) for value, trials in grouped.items()}
    duplicates = {str(value): [trial.number for trial in trials]
                  for value, trials in grouped.items() if len(trials) > 1}
    return key, expected, canonical, duplicates


def _role_count(metrics):
    candidates = (
        metrics.get("metrics/validation_diagnostic_events"),
        metrics.get("reconstruction_active_events"),
        metrics.get("edge_active_events"),
        metrics.get("metrics/validation_tp", 0) + metrics.get("metrics/validation_fn", 0)
        if "metrics/validation_tp" in metrics and "metrics/validation_fn" in metrics else None,
    )
    for candidate in candidates:
        if candidate is not None and _finite(candidate, "validation metric count") > 0:
            return int(float(candidate))
    raise RuntimeError("Cannot determine a positive validation count for reconstruction uncertainty.")


def _stage2_row(trial):
    payload, metrics, checkpoint, metrics_path = _trial_payload(trial)
    edge = _finite(metrics.get("val_edge_loss"), f"trial {trial.number} val_edge_loss")
    hyper = _finite(metrics.get("val_hyperedge_loss"), f"trial {trial.number} val_hyperedge_loss")
    classification = _finite(metrics.get("val_classification_loss"), f"trial {trial.number} val_classification_loss")
    role = _finite(metrics.get("val_reco_mean_role_top1"), f"trial {trial.number} val_reco_mean_role_top1")
    count = _role_count(metrics)
    return {
        "trial_number": trial.number, "alpha": float(trial.params["alpha"]),
        "original_objective": float(trial.value), "val_reco_mean_role_top1": role,
        "role_standard_error": math.sqrt(max(role * (1.0 - role), 0.0) / count),
        "validation_reconstruction_count": count,
        "val_edge_loss": edge, "val_hyperedge_loss": hyper,
        "val_classification_loss": classification,
        "invariant_component_score": 0.475 * edge + 0.475 * hyper + 0.05 * classification,
        "best_epoch": payload.get("best_observed_monitor_epoch"),
        "checkpoint": checkpoint, "metrics_path": metrics_path,
    }


def _stage3_row(trial):
    payload, metrics, checkpoint, metrics_path = _trial_payload(trial)
    auc = _finite(metrics.get("val_auc"), f"trial {trial.number} val_auc")
    role = _finite(metrics.get("val_reco_mean_role_top1"), f"trial {trial.number} val_reco_mean_role_top1")
    edge = _finite(metrics.get("val_edge_loss"), f"trial {trial.number} val_edge_loss")
    hyper = _finite(metrics.get("val_hyperedge_loss"), f"trial {trial.number} val_hyperedge_loss")
    classification = _finite(metrics.get("val_classification_loss"), f"trial {trial.number} val_classification_loss")
    tp = _finite(metrics.get("metrics/validation_tp"), f"trial {trial.number} validation_tp")
    fn = _finite(metrics.get("metrics/validation_fn"), f"trial {trial.number} validation_fn")
    tn = _finite(metrics.get("metrics/validation_tn"), f"trial {trial.number} validation_tn")
    fp = _finite(metrics.get("metrics/validation_fp"), f"trial {trial.number} validation_fp")
    n_signal, n_background = int(tp + fn), int(tn + fp)
    if n_signal <= 0 or n_background <= 0:
        raise RuntimeError(f"Trial {trial.number} has invalid class counts signal={n_signal}, background={n_background}.")
    # Hanley-McNeil large-sample standard error for an AUC.
    q1, q2 = auc / (2.0 - auc), 2.0 * auc * auc / (1.0 + auc)
    auc_var = (auc * (1.0 - auc) + (n_signal - 1) * (q1 - auc * auc)
               + (n_background - 1) * (q2 - auc * auc)) / (n_signal * n_background)
    count = _role_count(metrics)
    return {
        "trial_number": trial.number, "beta": float(trial.params["beta"]),
        "original_objective": float(trial.value), "val_auc": auc,
        "auc_standard_error": math.sqrt(max(auc_var, 0.0)),
        "n_signal": n_signal, "n_background": n_background,
        "val_classification_loss": classification,
        "val_reco_mean_role_top1": role,
        "role_standard_error": math.sqrt(max(role * (1.0 - role), 0.0) / count),
        "validation_reconstruction_count": count,
        "val_edge_loss": edge, "val_hyperedge_loss": hyper,
        "invariant_reconstruction_score": 0.5 * edge + 0.5 * hyper,
        "best_epoch": payload.get("best_observed_monitor_epoch"),
        "checkpoint": checkpoint, "metrics_path": metrics_path,
    }


def choose_stage2(rows):
    if not rows:
        raise RuntimeError("No Stage 2 candidates were supplied.")
    return max(rows, key=lambda row: (row["val_reco_mean_role_top1"], -row["invariant_component_score"], -row["trial_number"]))


def choose_stage3(rows, *, reference_beta=0.05, sigma=2.0, auc_tolerance=None):
    if not rows:
        raise RuntimeError("No Stage 3 candidates were supplied.")
    references = [row for row in rows if math.isclose(row["beta"], reference_beta, abs_tol=1e-12)]
    if len(references) != 1:
        raise RuntimeError(f"Stage 3 requires exactly one canonical beta={reference_beta} reference; found {len(references)}.")
    reference = references[0]
    reconstruction_tolerance = sigma * reference["role_standard_error"]
    threshold = reference["val_reco_mean_role_top1"] - reconstruction_tolerance
    for row in rows:
        row["reconstruction_reference_beta"] = reference_beta
        row["reconstruction_threshold"] = threshold
        row["reconstruction_feasible"] = row["val_reco_mean_role_top1"] >= threshold
    feasible = [row for row in rows if row["reconstruction_feasible"]]
    if not feasible:
        raise RuntimeError(f"No Stage 3 beta satisfies the reconstruction threshold {threshold}.")
    best_auc = max(row["val_auc"] for row in feasible)
    best = min((row for row in feasible if row["val_auc"] == best_auc), key=lambda row: row["trial_number"])
    tolerance = sigma * best["auc_standard_error"] if auc_tolerance is None else float(auc_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"AUC tolerance must be finite and non-negative, got {tolerance}.")
    tied = [row for row in feasible if row["val_auc"] >= best_auc - tolerance]
    selected = max(tied, key=lambda row: (row["val_reco_mean_role_top1"], -row["beta"], -row["trial_number"]))
    return selected, {
        "reference_beta": reference_beta,
        "reference_reconstruction": reference["val_reco_mean_role_top1"],
        "reconstruction_tolerance": reconstruction_tolerance,
        "reconstruction_threshold": threshold,
        "auc_tolerance": tolerance,
        "auc_best_feasible": best_auc,
        "auc_tie_candidates": [row["trial_number"] for row in tied],
        "sigma_multiplier": sigma,
    }


def select_stage(sqlite_path, study_name, stage, base_config, output_dir, tolerance=None,
                 minimum_completed=None, reconstruction_sigma=2.0):
    study = optuna.load_study(study_name=study_name, storage=_storage(sqlite_path))
    states = {state.name: 0 for state in optuna.trial.TrialState}
    for trial in study.trials: states[trial.state.name] += 1
    if stage == 1:
        trials = sorted((trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
                        key=lambda trial: (float(trial.value), trial.number))
        if minimum_completed is not None and len(trials) < minimum_completed:
            raise RuntimeError(f"Study {study_name} has {len(trials)} completed trials; requires {minimum_completed}.")
        selected, rows, duplicates, policy = trials[0], [], {}, "minimum common val_reconstruction_loss"
        selection_details = {}
    else:
        _, _, canonical, duplicates = _canonical_grid_trials(study, stage)
        row_builder = _stage2_row if stage == 2 else _stage3_row
        # Validate every completed scientific row, including duplicates, before
        # deterministically retaining the earliest trial for each grid value.
        validated_rows = {
            trial.number: row_builder(trial) for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        }
        rows = [validated_rows[canonical[value].number] for value in sorted(canonical)]
        if stage == 2:
            winner = choose_stage2(rows); policy = "maximum validation mean typed-role top-1 efficiency"
            selection_details = {"metric": "val_reco_mean_role_top1"}
        else:
            winner, selection_details = choose_stage3(
                rows, sigma=float(reconstruction_sigma), auc_tolerance=tolerance,
            )
            policy = "maximum validation AUC subject to beta=0.05 reconstruction retention; AUC ties prefer reconstruction then lower beta"
        selected = next(trial for trial in canonical.values() if trial.number == winner["trial_number"])
    cfg = OmegaConf.load(base_config)
    if stage == 1:
        for path, value in selected.params.items(): OmegaConf.update(cfg, path, value, merge=False, force_add=False)
        cfg.classification.enabled = False; cfg.reconstruction.enabled = True
        cfg.loss.edge_weight = .5; cfg.loss.hyperedge_weight = .5; cfg.loss.classification_weight = 0.
        parameter_file, parameter = "stage1_best_parameters.yaml", dict(selected.params)
    elif stage == 2:
        alpha = float(selected.params["alpha"]); parameter_file, parameter = "stage2_best_alpha.yaml", {"alpha": alpha}
        cfg.classification.enabled = True; cfg.reconstruction.enabled = True
        for key, value in loss_coefficients(alpha, .05).items(): cfg.loss[key] = value
    else:
        beta = float(selected.params["beta"]); parameter_file, parameter = "stage3_best_beta.yaml", {"beta": beta}
        alpha = float(cfg.get("tuning_ancestry", {}).get("stage2_alpha", -1))
        if alpha < 0: raise ValueError("Stage 3 base config must record tuning_ancestry.stage2_alpha.")
        for key, value in loss_coefficients(alpha, beta).items(): cfg.loss[key] = value
    ancestry = dict(cfg.get("tuning_ancestry", {}))
    ancestry.update({f"stage{stage}_study": study_name, f"stage{stage}_trial": selected.number,
                     f"stage{stage}_best_value": float(selected.value), f"stage{stage}_selection_policy": policy})
    if stage == 2:
        ancestry.update({"stage2_alpha": float(selected.params["alpha"]),
                         "stage2_selection_metric": winner["val_reco_mean_role_top1"]})
    if stage == 3:
        ancestry.update({"stage3_beta": float(selected.params["beta"]),
                         "stage3_selection_auc": winner["val_auc"],
                         "stage3_selection_reconstruction": winner["val_reco_mean_role_top1"]})
    cfg.tuning_ancestry = ancestry
    if "tuning" in cfg: del cfg["tuning"]
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    _write_yaml(output / parameter_file, parameter)
    _write_yaml(output / f"stage{stage}_best_config.yaml", OmegaConf.to_container(cfg, resolve=True))
    summary = {"stage": stage, "study_name": study_name, "selected_trial": selected.number,
               "parameters": parameter, "selection_policy": policy, "selection_details": selection_details,
               "trial_state_counts": states, "duplicate_completed_values": duplicates,
               "source_study_unchanged": True}
    if stage in (2, 3):
        summary["selected_metrics"] = winner
        audit = output / f"stage{stage}_selection_audit.csv"
        with audit.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader(); writer.writerows(rows)
        summary["audit_csv"] = str(audit.resolve())
        _write_json(output / f"stage{stage}_selection_audit.json", {"rows": rows, "selection": summary})
    _write_json(output / f"stage{stage}_summary.json", summary)
    return summary


def final_configs(stage3_config, output_dir, graph_manifest, split_cache, subset_manifest):
    joint = OmegaConf.load(stage3_config)
    required = ((Path(graph_manifest).expanduser().resolve(), "graph build manifest"),
                (Path(subset_manifest).expanduser().resolve(), "tuning subset manifest"),
                (Path(split_cache).expanduser().resolve(), "canonical split cache"))
    for path, description in required:
        if not path.is_file() or path.stat().st_size == 0: raise FileNotFoundError(f"Required {description} is absent or empty: {path}")
    graph_manifest_path, subset_manifest_path, split_path = (item[0] for item in required)
    ancestry = dict(joint.get("tuning_ancestry", {})); alpha = float(ancestry["stage2_alpha"]); beta = float(ancestry["stage3_beta"])
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception: commit, dirty = None, None
    ancestry.update({"graph_db_manifest": str(graph_manifest_path), "canonical_split": str(split_path),
                     "tuning_subset_manifest": str(subset_manifest_path), "code_commit": commit, "dirty_worktree": dirty})
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True); modes = {}
    for mode in ("reconstruction_only", "classification_only", "joint"):
        cfg = OmegaConf.create(OmegaConf.to_container(joint, resolve=True))
        if mode == "reconstruction_only": cfg.classification.enabled=False; cfg.reconstruction.enabled=True; weights=loss_coefficients(alpha, 0.0)
        elif mode == "classification_only": cfg.classification.enabled=True; cfg.reconstruction.enabled=False; weights={"edge_weight":0.0,"hyperedge_weight":0.0,"classification_weight":1.0}
        else: cfg.classification.enabled=True; cfg.reconstruction.enabled=True; weights=loss_coefficients(alpha, beta)
        for key, value in weights.items(): cfg.loss[key] = value
        cfg.dataset.split.enabled=True; cfg.dataset.split.cache_path=str(split_path); cfg.dataset.split.require_existing=True; cfg.dataset.split.predict_split=None
        cfg.dataset.force_reload=False; cfg.tuning_ancestry=ancestry
        path=output/f"{mode}.yaml"; _write_yaml(path, OmegaConf.to_container(cfg, resolve=True)); modes[mode]=str(path.resolve())
    _write_json(output/"final_config_manifest.json", {"alpha":alpha,"beta":beta,"ancestry":ancestry,"configs":modes})
    return modes


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command", required=True)
    select=sub.add_parser("select"); select.add_argument("--sqlite-path", required=True); select.add_argument("--study-name", required=True)
    select.add_argument("--stage", type=int, choices=(1,2,3), required=True); select.add_argument("--base-config", required=True); select.add_argument("--output-dir", required=True)
    select.add_argument("--tolerance", type=float); select.add_argument("--reconstruction-sigma", type=float, default=2.0); select.add_argument("--minimum-completed", type=int)
    final=sub.add_parser("final-configs"); final.add_argument("--stage3-config", required=True); final.add_argument("--output-dir", required=True)
    final.add_argument("--graph-manifest", required=True); final.add_argument("--split-cache", required=True); final.add_argument("--subset-manifest", required=True)
    args=parser.parse_args()
    if args.command == "select":
        print(json.dumps(select_stage(args.sqlite_path,args.study_name,args.stage,args.base_config,args.output_dir,args.tolerance,args.minimum_completed,args.reconstruction_sigma),indent=2,default=str))
    else: print(json.dumps(final_configs(args.stage3_config,args.output_dir,args.graph_manifest,args.split_cache,args.subset_manifest),indent=2))


if __name__ == "__main__": main()
