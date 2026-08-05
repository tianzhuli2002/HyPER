#!/usr/bin/env python3
"""Discover validated final HyPER runs and write a representation-transfer profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any

import numpy as np
import yaml

MODES = ("classification_only", "reconstruction_only", "joint")
MODE_KEYS = {
    "classification_only": "CLASS",
    "reconstruction_only": "RECO",
    "joint": "JOINT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=("ttbar1L", "ttH"), required=True)
    parser.add_argument(
        "--runtime",
        default="/net/scratch/w00238tl/HyPER_24_2_speedup_prod",
        help="Runtime containing final configs, runs, H5 and canonical splits.",
    )
    parser.add_argument(
        "--campaign-id",
        default="stage1_v2_20260721_production_fix2",
    )
    parser.add_argument(
        "--recovery-id",
        default="physics_selection_20260728T111408Z_retry1",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-latest-fallback",
        action="store_true",
        help="Use the newest valid run manifest when the canonical promoted run is absent.",
    )
    return parser.parse_args()


def recursive_values(value: Any, keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in keys and isinstance(item, (str, Path)):
                found.append(str(item))
            found.extend(recursive_values(item, keys))
    elif isinstance(value, list):
        for item in value:
            found.extend(recursive_values(item, keys))
    return found


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_manifest_path(run: Path, manifest: dict[str, Any], keys: set[str]) -> Path | None:
    for raw in recursive_values(manifest, keys):
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = run / candidate
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def resolve_checkpoint(run: Path, manifest: dict[str, Any]) -> Path:
    path = resolve_manifest_path(
        run,
        manifest,
        {"checkpoint", "checkpoint_path", "best_checkpoint", "model_checkpoint"},
    )
    if path is not None and path.is_file():
        return path
    candidates = sorted(
        run.glob("training/version_*/checkpoints/best*.ckpt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            run.glob("**/checkpoints/best*.ckpt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found under {run}.")
    return candidates[0].resolve()


def resolve_prediction(run: Path, manifest: dict[str, Any]) -> Path:
    path = resolve_manifest_path(
        run,
        manifest,
        {"prediction_output", "predictions", "prediction_path"},
    )
    if path is None:
        candidates = sorted(
            [*run.glob("predictions.pkl.parts"), *run.glob("**/predictions.pkl.parts")],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            path = candidates[0].resolve()
    if path is None or not path.exists():
        raise FileNotFoundError(f"No prediction output found for {run}.")
    return path


def valid_run(run: Path, need_predictions: bool) -> tuple[dict[str, Any], Path, Path | None]:
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    checkpoint = resolve_checkpoint(run, manifest)
    prediction = resolve_prediction(run, manifest) if need_predictions else None
    return manifest, checkpoint, prediction


def choose_run(
    runtime: Path,
    topology: str,
    mode: str,
    recovery_id: str,
    allow_latest: bool,
) -> tuple[Path, dict[str, Any], Path, Path | None]:
    base = runtime / "results/sb_transfer/hyper" / topology / mode
    canonical = base / f"promoted_{recovery_id}_{topology}_{mode}"
    try:
        manifest, checkpoint, prediction = valid_run(
            canonical, need_predictions=mode != "classification_only"
        )
        return canonical.resolve(), manifest, checkpoint, prediction
    except (FileNotFoundError, KeyError, ValueError):
        if not allow_latest:
            raise RuntimeError(
                f"Canonical {topology} {mode} run is not complete: {canonical}. "
                "Use --allow-latest-fallback only after checking the candidates."
            )

    candidates = sorted(
        {path.parent for path in base.glob("*/run_manifest.json")},
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    failures: list[str] = []
    for run in candidates:
        try:
            manifest, checkpoint, prediction = valid_run(
                run, need_predictions=mode != "classification_only"
            )
            return run.resolve(), manifest, checkpoint, prediction
        except (FileNotFoundError, KeyError, ValueError) as error:
            failures.append(f"{run}: {error}")
    details = "\n".join(failures[:10])
    raise RuntimeError(f"No valid {topology} {mode} run found under {base}.\n{details}")


def resolve_source_h5(runtime: Path, topology: str, configs: list[Path], manifests: list[dict[str, Any]]) -> Path:
    keys = {"source_h5_path", "h5", "h5_path", "input_h5_path", "dataset_path"}
    for manifest in manifests:
        for raw in recursive_values(manifest, keys):
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = runtime / path
            if path.is_file():
                return path.resolve()
    for config in configs:
        data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        for raw in recursive_values(data, keys):
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = runtime / path
            if path.is_file():
                return path.resolve()
    raw_root = runtime / ("HyPER_ttbarSL_typed/raw" if topology == "ttbar1L" else "HyPER_ttH_SL_typed/raw")
    candidates = sorted(raw_root.glob("*.h5"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Could not resolve one source H5 under {raw_root}; candidates={candidates}."
        )
    return candidates[0].resolve()


def test_count(split: Path) -> int:
    with np.load(split, allow_pickle=False) as data:
        for key in ("test_idx", "test_indices", "test"):
            if key in data.files:
                return int(len(data[key]))
    raise KeyError(f"No test index array found in {split}; keys={np.load(split).files}")


def q(value: object) -> str:
    return shlex.quote(str(value))


def main() -> int:
    args = parse_args()
    runtime = Path(args.runtime).expanduser().resolve()
    topology = args.topology
    campaign = (
        runtime
        / "results/tuning_campaigns"
        / args.campaign_id
        / args.recovery_id
        / topology
    )
    configs = {
        mode: (campaign / "final_configs" / f"{mode}.yaml").resolve()
        for mode in MODES
    }
    for path in configs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    runs: dict[str, Path] = {}
    manifests: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, Path] = {}
    predictions: dict[str, Path | None] = {}
    for mode in MODES:
        run, manifest, checkpoint, prediction = choose_run(
            runtime,
            topology,
            mode,
            args.recovery_id,
            args.allow_latest_fallback,
        )
        runs[mode] = run
        manifests[mode] = manifest
        checkpoints[mode] = checkpoint
        predictions[mode] = prediction

    split = (
        runtime
        / "results/integration_validation/splits"
        / ("ttbar1L_full_schema4.npz" if topology == "ttbar1L" else "ttH_full_schema4.npz")
    ).resolve()
    if not split.is_file():
        raise FileNotFoundError(split)
    h5 = resolve_source_h5(runtime, topology, list(configs.values()), list(manifests.values()))

    graph_root = resolve_manifest_path(
        runs["reconstruction_only"],
        manifests["reconstruction_only"],
        {"graph_db_path", "dataset_root", "graph_database"},
    )
    if graph_root is None:
        raise RuntimeError("Could not resolve graph_db_path from reconstruction run manifest.")
    if graph_root.is_file():
        graph_root = graph_root.parent

    title = (
        r"$t\bar{t}$ single-lepton representation transfer"
        if topology == "ttbar1L"
        else r"$t\bar{t}H$ single-lepton representation transfer"
    )
    score_definition = (
        "p_top1 * p_top2 * p_W1 * p_W2"
        if topology == "ttbar1L"
        else "p_tlep * p_thad * p_Wlep * p_Whad * p_H"
    )

    values: dict[str, object] = {
        "REP_TOPOLOGY": topology,
        "REP_RUN_TAG": args.recovery_id,
        "REP_TITLE": title,
        "REP_CLASS_CONFIG": configs["classification_only"],
        "REP_RECO_CONFIG": configs["reconstruction_only"],
        "REP_JOINT_CONFIG": configs["joint"],
        "REP_CLASS_RUN": runs["classification_only"],
        "REP_RECO_RUN": runs["reconstruction_only"],
        "REP_JOINT_RUN": runs["joint"],
        "REP_CLASS_CHECKPOINT": checkpoints["classification_only"],
        "REP_RECO_CHECKPOINT": checkpoints["reconstruction_only"],
        "REP_JOINT_CHECKPOINT": checkpoints["joint"],
        "REP_RECO_PREDICTIONS": predictions["reconstruction_only"],
        "REP_JOINT_PREDICTIONS": predictions["joint"],
        "REP_H5": h5,
        "REP_SPLIT_CACHE": split,
        "REP_DATASET_ROOT": graph_root,
        "REP_TEST_EVENTS": test_count(split),
        "REP_RECONSTRUCTION_SCORE_DEFINITION": score_definition,
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by tools/create_hyper_representation_profile.py",
        f"# runtime={runtime}",
    ]
    lines.extend(f"{key}={q(value)}" for key, value in values.items())
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"profile={output}")
    for key, value in values.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
