#!/usr/bin/env python3
"""Explicit end-to-end driver for the frozen HyPER representation study."""

from __future__ import annotations

import argparse
import csv
import json
import numpy as np
import subprocess
import sys
from pathlib import Path


MODES = ("classification", "reconstruction", "joint")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument(f"--{mode}-config", required=True)
        parser.add_argument(f"--{mode}-checkpoint", required=True)
        parser.add_argument(f"--{mode}-run-directory", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-cache", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--alignment-event-count", type=int, default=50000)
    parser.add_argument("--cka-event-count", type=int, default=50000)
    parser.add_argument("--evaluation-max-events", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-transfer", action="store_true")
    return parser.parse_args()


def require_file(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def require_directory(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def run(command):
    print("RUN " + " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def complete_export(path: Path, count: int, representations: list[str], split: str) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as loaded:
        if any(name not in loaded for name in ("source_event_index", "prediction_split", *representations)):
            return False
        indices = loaded["source_event_index"]
        return (
            len(indices) == int(count)
            and len(np.unique(indices)) == len(indices)
            and np.all(loaded["prediction_split"] == split)
        )


def complete_alignment(path: Path, count: int) -> bool:
    summary = path.with_suffix(".json")
    if not path.is_file() or not summary.is_file():
        return False
    metadata = json.loads(summary.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as loaded:
        return (
            metadata.get("fit_event_count") == int(count)
            and all(name in loaded for name in ("source_mean", "target_mean", "rotation", "singular_values"))
        )


def complete_cka(directory: Path, count: int) -> bool:
    required = ("cka_summary.json", "cka_matrix.csv", "cka_heatmap.pdf", "cka_heatmap.png")
    if any(not (directory / name).is_file() for name in required):
        return False
    summary = json.loads((directory / "cka_summary.json").read_text(encoding="utf-8"))
    return summary.get("event_count") == int(count)


def complete_transfer(directory: Path, count: int | None) -> bool:
    required = ("transfer_summary.json", "transfer_metrics.csv", "transfer_roc.pdf", "transfer_roc.png")
    if any(not (directory / name).is_file() for name in required):
        return False
    summary = json.loads((directory / "transfer_summary.json").read_text(encoding="utf-8"))
    required_methods = {
        "joint_reconstruction_zero_shot_score",
        "reconstruction_to_joint_aligned_score",
        "joint_to_classification_aligned_score",
    }
    return (
        (count is None or summary.get("event_count") == int(count))
        and required_methods.issubset(summary.get("methods", {}))
    )


def write_scientific_summary(output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cka = json.loads((output / "cka/classification_vs_reconstruction/cka_summary.json").read_text())
    alignment = json.loads((output / "alignments/reconstruction_to_classification.json").read_text())
    with (output / "transfer/transfer_metrics.csv").open(newline="", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    inclusive = {
        row["method"]: float(row["roc_auc"])
        for row in metric_rows if row["subset"] == "inclusive_all_signal"
    }
    rows = [
        ("Final representation CKA", cka["corresponding_layer_cka"]["all"]["final_event"]),
        ("Normalised Procrustes residual", alignment["normalised_alignment_residual"]),
        ("Native classification-only AUC", inclusive["native_classification_only_score"]),
        ("Native joint AUC", inclusive["native_joint_score"]),
        ("Reconstruction zero-shot AUC", inclusive["reconstruction_zero_shot_score"]),
        ("Joint reconstruction zero-shot AUC", inclusive["joint_reconstruction_zero_shot_score"]),
        ("Direct swap AUC", inclusive["direct_head_swap_score"]),
        ("Aligned transfer AUC", inclusive["aligned_head_transfer_score"]),
        ("Shuffled control AUC", inclusive["shuffled_alignment_score"]),
    ]
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    with (plots / "scientific_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(("quantity", "value")); writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=[[name, f"{value:.6g}"] for name, value in rows],
        colLabels=["Quantity", "Value"], loc="center", cellLoc="left", colLoc="left",
    )
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.35)
    ax.set_title("HyPER representation-transfer summary", pad=16)
    fig.savefig(plots / "scientific_summary.pdf", bbox_inches="tight")
    fig.savefig(plots / "scientific_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    tools = Path(__file__).resolve().parent
    inputs = {}
    for mode in MODES:
        inputs[f"{mode}_config"] = require_file(getattr(args, f"{mode}_config"))
        inputs[f"{mode}_checkpoint"] = require_file(getattr(args, f"{mode}_checkpoint"))
        inputs[f"{mode}_run_directory"] = require_directory(getattr(args, f"{mode}_run_directory"))
    inputs["h5"] = require_file(args.h5)
    inputs["split_cache"] = require_file(args.split_cache)
    reco_manifest_path = inputs["reconstruction_run_directory"] / "run_manifest.json"
    reco_manifest = json.loads(require_file(reco_manifest_path).read_text(encoding="utf-8"))
    graph_db = require_file(reco_manifest["graph_db_path"])
    dataset_root = graph_db.parent
    reconstruction_predictions = require_directory(reco_manifest["prediction_output"])
    joint_manifest = json.loads(require_file(inputs["joint_run_directory"] / "run_manifest.json").read_text(encoding="utf-8"))
    joint_predictions = require_directory(joint_manifest["prediction_output"])
    output = Path(args.output_directory).expanduser().resolve()
    for subdir in ("representations", "alignments", "cka", "transfer", "plots"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    print("Resolved study inputs:")
    for name, value in {**inputs, "dataset_root": dataset_root,
                        "reconstruction_predictions": reconstruction_predictions,
                        "joint_predictions": joint_predictions}.items():
        print(f"  {name}: {value}")

    representations = ["final_event", "classification_head_input", "block_0", "block_1", "block_2"]
    exports = {}
    for split, count in (("val", args.alignment_event_count), ("test", args.cka_event_count)):
        for mode in MODES:
            destination = output / "representations" / f"{mode}_{split}.npz"
            exports[(mode, split)] = destination
            if complete_export(destination, count, representations, split):
                print(f"REUSE verified representation export {destination}")
                continue
            run([
                sys.executable, tools / "export_hyper_representations.py",
                "--config", inputs[f"{mode}_config"], "--checkpoint", inputs[f"{mode}_checkpoint"],
                "--h5", inputs["h5"], "--split-cache", inputs["split_cache"],
                "--dataset-root", dataset_root, "--split", split, "--max-events", count,
                "--seed", args.seed, "--batch-size", args.batch_size, "--num-workers", args.num_workers,
                "--accelerator", args.accelerator, "--model-name", mode,
                "--representations", *representations, "--output", destination,
            ])

    pairings = (
        ("reconstruction", "classification"),
        ("reconstruction", "joint"),
        ("joint", "classification"),
    )
    alignments = {}
    for source, target in pairings:
        destination = output / "alignments" / f"{source}_to_{target}.npz"
        summary = destination.with_suffix(".json")
        alignments[(source, target)] = destination
        if complete_alignment(destination, args.alignment_event_count):
            print(f"REUSE verified alignment {destination}")
            continue
        run([
            sys.executable, tools / "fit_hyper_procrustes.py",
            "--source", exports[(source, "val")], "--target", exports[(target, "val")],
            "--source-key", "final_event", "--target-key", "classification_head_input",
            "--output", destination, "--summary", summary,
        ])
    shuffled = output / "alignments" / "reconstruction_to_classification_shuffled.npz"
    if complete_alignment(shuffled, args.alignment_event_count):
        print(f"REUSE verified shuffled alignment {shuffled}")
    else:
        run([
            sys.executable, tools / "fit_hyper_procrustes.py",
            "--source", exports[("reconstruction", "val")],
            "--target", exports[("classification", "val")], "--shuffle-target", "--seed", args.seed,
            "--output", shuffled, "--summary", shuffled.with_suffix(".json"),
        ])

    for left, right in (("classification", "reconstruction"), ("classification", "joint"),
                        ("reconstruction", "joint")):
        cka_directory = output / "cka" / f"{left}_vs_{right}"
        if complete_cka(cka_directory, args.cka_event_count):
            print(f"REUSE verified CKA output {cka_directory}")
            continue
        run([
            sys.executable, tools / "compute_hyper_cka.py",
            "--left", exports[(left, "test")], "--right", exports[(right, "test")],
            "--output-dir", cka_directory,
            "--title", f"ttbar single-lepton: {left} vs {right}",
        ])

    evaluation = [
        sys.executable, tools / "evaluate_hyper_head_transfer.py",
        "--classification-config", inputs["classification_config"],
        "--classification-checkpoint", inputs["classification_checkpoint"],
        "--reconstruction-config", inputs["reconstruction_config"],
        "--reconstruction-checkpoint", inputs["reconstruction_checkpoint"],
        "--joint-config", inputs["joint_config"], "--joint-checkpoint", inputs["joint_checkpoint"],
        "--reconstruction-predictions", reconstruction_predictions,
        "--joint-predictions", joint_predictions,
        "--alignment", alignments[("reconstruction", "classification")],
        "--shuffled-alignment", shuffled, "--h5", inputs["h5"],
        "--reconstruction-to-joint-alignment", alignments[("reconstruction", "joint")],
        "--joint-to-classification-alignment", alignments[("joint", "classification")],
        "--split-cache", inputs["split_cache"], "--dataset-root", dataset_root,
        "--batch-size", args.batch_size, "--num-workers", args.num_workers,
        "--accelerator", args.accelerator, "--seed", args.seed,
        "--output-dir", output / "transfer",
    ]
    if args.evaluation_max_events is not None:
        evaluation.extend(("--max-events", args.evaluation_max_events))
    if args.overwrite_transfer:
        evaluation.append("--overwrite")
    transfer_directory = output / "transfer"
    if complete_transfer(transfer_directory, args.evaluation_max_events):
        print(f"REUSE verified transfer output {transfer_directory}")
    else:
        run(evaluation)
    write_scientific_summary(output)
    summary = {
        "inputs": {name: str(value) for name, value in inputs.items()},
        "dataset_root": str(dataset_root), "alignment_event_count": args.alignment_event_count,
        "cka_event_count": args.cka_event_count, "evaluation_max_events": args.evaluation_max_events,
        "seed": args.seed, "labels_used_for_alignment": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
