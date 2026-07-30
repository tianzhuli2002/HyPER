#!/usr/bin/env python3
"""Stream full-test frozen-head transfer and scientific controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from HyPER.analysis.representations import apply_procrustes, random_orthogonal
from HyPER.analysis.runtime import (
    build_analysis_datamodule, forward_representations, load_analysis_config,
    load_frozen_model, truth_fully_matched,
)
from HyPER.topology.prediction_io import iter_hyper_prediction_parts
from HyPER.topology.reconstruction_score import ttbar_sl_event_reconstruction_score


METHOD_LABELS = {
    "native_classification_only_score": "Native classification-only",
    "native_joint_score": "Native joint",
    "reconstruction_zero_shot_score": "Reconstruction zero-shot",
    "joint_reconstruction_zero_shot_score": "Joint reconstruction zero-shot",
    "direct_head_swap_score": "Direct frozen-head swap",
    "aligned_head_transfer_score": "Procrustes-aligned transfer",
    "shuffled_alignment_score": "Shuffled-pair control",
    "random_orthogonal_score": "Random orthogonal control",
    "reconstruction_to_joint_direct_score": "Reconstruction to joint direct",
    "reconstruction_to_joint_aligned_score": "Reconstruction to joint aligned",
    "joint_to_classification_direct_score": "Joint to classification direct",
    "joint_to_classification_aligned_score": "Joint to classification aligned",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ("classification", "reconstruction", "joint"):
        parser.add_argument(f"--{mode}-config", required=True)
        parser.add_argument(f"--{mode}-checkpoint", required=True)
    parser.add_argument("--reconstruction-predictions", required=True,
                        help="Verified test prediction parts containing selected_reconstruction_scores.")
    parser.add_argument("--joint-predictions", required=True,
                        help="Verified joint-model test prediction parts containing reconstruction scores.")
    parser.add_argument("--alignment", required=True)
    parser.add_argument("--shuffled-alignment", required=True)
    parser.add_argument("--reconstruction-to-joint-alignment", required=True)
    parser.add_argument("--joint-to-classification-alignment", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-cache", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_alignment(path):
    with np.load(path, allow_pickle=False) as loaded:
        required = ("source_mean", "target_mean", "rotation")
        if any(name not in loaded for name in required):
            raise KeyError(f"Alignment {path} is missing one of {required}.")
        return {name: loaded[name] for name in required}


def load_reconstruction_scores(path: str, limit: int | None) -> dict[int, float]:
    scores = {}
    loaded = 0
    for frame in iter_hyper_prediction_parts(Path(path), max_events=None, chunk_size=100000):
        required = {"source_event_index", "selected_reconstruction_scores"}
        if not required.issubset(frame.columns):
            raise KeyError(f"Reconstruction predictions lack {sorted(required - set(frame.columns))}.")
        for index, values in zip(frame["source_event_index"], frame["selected_reconstruction_scores"]):
            index = int(index)
            if index in scores:
                raise ValueError(f"Duplicate reconstruction source_event_index={index}.")
            scores[index] = ttbar_sl_event_reconstruction_score(values)
        loaded += len(frame)
        if limit is not None and loaded >= limit:
            break
    return scores


def background_rejection(labels, scores, efficiency):
    signal = scores[labels == 1]
    background = scores[labels == 0]
    threshold = np.quantile(signal, 1.0 - efficiency)
    background_efficiency = np.mean(background >= threshold)
    return float(np.inf if background_efficiency == 0 else 1.0 / background_efficiency)


def metric_row(method, subset, labels, scores, native):
    finite = np.isfinite(scores) & np.isfinite(native)
    labels, scores, native = labels[finite], scores[finite], native[finite]
    if len(np.unique(labels)) != 2:
        raise ValueError(f"Metric subset {subset}/{method} does not contain both classes.")
    return {
        "method": method, "subset": subset, "event_count": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy_at_0p5": float(accuracy_score(labels, scores >= 0.5)),
        "background_rejection_at_signal_efficiency_0p5": background_rejection(labels, scores, 0.5),
        "background_rejection_at_signal_efficiency_0p7": background_rejection(labels, scores, 0.7),
        "background_rejection_at_signal_efficiency_0p8": background_rejection(labels, scores, 0.8),
        "pearson_with_native_classification_only": float(pearsonr(scores, native).statistic),
        "spearman_with_native_classification_only": float(spearmanr(scores, native).statistic),
    }


def save_figure(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(frame, metrics, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    labels = frame.truth_class.to_numpy()
    main = [
        "native_classification_only_score", "native_joint_score",
        "reconstruction_zero_shot_score", "direct_head_swap_score",
        "aligned_head_transfer_score", "shuffled_alignment_score",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for name in main:
        fpr, tpr, _ = roc_curve(labels, frame[name])
        auc = roc_auc_score(labels, frame[name])
        ax.plot(fpr, tpr, lw=1.8, label=f"{METHOD_LABELS[name]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    ax.set(xlabel="Background efficiency", ylabel="Signal efficiency", title="Frozen representation transfer")
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, output_dir, "transfer_roc")

    score = frame.aligned_head_transfer_score.to_numpy()
    truth, fm = labels, frame.truth_fully_matched.to_numpy().astype(bool)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for mask, label, color in (
        (truth == 0, "Background", "tab:blue"),
        ((truth == 1) & fm, "Signal, fully matched", "tab:orange"),
        ((truth == 1) & ~fm, "Signal, non-fully-matched", "tab:green"),
    ):
        ax.hist(score[mask], bins=60, range=(0, 1), density=True, alpha=0.45, label=label, color=color)
    ax.set(xlabel="Aligned frozen-head score", ylabel="Normalised density", title="Aligned-transfer score")
    ax.legend()
    save_figure(fig, output_dir, "aligned_transfer_score")

    for score_name, stem, title in (
        ("reconstruction_zero_shot_score", "reconstruction_zero_shot_score", "Reconstruction-only zero-shot score"),
        ("joint_reconstruction_zero_shot_score", "joint_reconstruction_zero_shot_score", "Joint reconstruction zero-shot score"),
    ):
        values = frame[score_name].to_numpy()
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for mask, label, color in (
            (truth == 0, "Background", "tab:blue"),
            ((truth == 1) & fm, "Signal, fully matched", "tab:orange"),
            ((truth == 1) & ~fm, "Signal, non-fully-matched", "tab:green"),
        ):
            ax.hist(values[mask], bins=60, range=(0, 1), density=True, alpha=0.45,
                    label=label, color=color)
        ax.set(xlabel="Fixed reconstruction confidence", ylabel="Normalised density", title=title)
        ax.legend()
        save_figure(fig, output_dir, stem)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.hexbin(frame.native_classification_only_score, frame.aligned_head_transfer_score,
              gridsize=70, mincnt=1, cmap="viridis")
    ax.set(xlabel="Native classification-only score", ylabel="Aligned transferred score",
           title="Native versus transferred score")
    save_figure(fig, output_dir, "native_vs_aligned_transfer")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in ("native_classification_only_score", "reconstruction_zero_shot_score",
                 "direct_head_swap_score", "aligned_head_transfer_score"):
        ax.hist(frame[name], bins=60, range=(0, 1), density=True, histtype="step", lw=1.8,
                label=METHOD_LABELS[name])
    ax.set(xlabel="Score", ylabel="Normalised density", title="Method score comparison")
    ax.legend(fontsize=8)
    save_figure(fig, output_dir, "method_score_comparison")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.accelerator == "gpu" and torch.cuda.is_available() else "cpu")
    cfg = load_analysis_config(args.reconstruction_config)
    module = build_analysis_datamodule(
        cfg, h5=args.h5, split_cache=args.split_cache, split="test", dataset_root=args.dataset_root,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    if args.max_events is not None:
        available = module.predict_data.indices
        count = min(int(args.max_events), len(available))
        if count <= 0:
            raise ValueError("--max-events must be positive when supplied.")
        positions = np.sort(np.random.default_rng(args.seed).choice(len(available), count, replace=False))
        module.predict_data.indices = available[positions]
    models = {
        "classification": load_frozen_model(args.classification_checkpoint, device),
        "reconstruction": load_frozen_model(args.reconstruction_checkpoint, device),
        "joint": load_frozen_model(args.joint_checkpoint, device),
    }
    if models["classification"].Classification is None or models["joint"].Classification is None:
        raise RuntimeError("Classification-only and joint checkpoints must contain classification heads.")
    alignment = load_alignment(args.alignment)
    shuffled = load_alignment(args.shuffled_alignment)
    reco_to_joint = load_alignment(args.reconstruction_to_joint_alignment)
    joint_to_class = load_alignment(args.joint_to_classification_alignment)
    dimension = alignment["rotation"].shape[0]
    random_rotation = random_orthogonal(dimension, args.seed)
    reconstruction_scores = load_reconstruction_scores(args.reconstruction_predictions, None)
    joint_reconstruction_scores = load_reconstruction_scores(args.joint_predictions, None)
    parts_dir = output_dir / "transfer_events.pkl.parts"
    if parts_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Transfer event output already exists: {parts_dir}; pass --overwrite explicitly.")
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(exist_ok=False)
    frames, rows, part = [], 0, 0
    with torch.inference_mode():
        for batch in module.predict_dataloader():
            batch = batch.to(device)
            class_outputs, class_reps = forward_representations(models["classification"], batch)
            reco_outputs, reco_reps = forward_representations(models["reconstruction"], batch)
            joint_outputs, joint_reps = forward_representations(models["joint"], batch)
            native_logits = class_outputs[3].reshape(-1)
            reconstructed_logits = models["classification"].Classification.mlp_class(
                class_reps["classification_head_input"]
            ).reshape(-1)
            np.testing.assert_allclose(
                reconstructed_logits.cpu().numpy(), native_logits.cpu().numpy(), rtol=1e-5, atol=1e-6
            )
            source_rep = reco_reps["final_event"]
            direct = models["classification"].Classification.mlp_class(source_rep).reshape(-1)
            source_np = source_rep.float().cpu().numpy()
            aligned_np = apply_procrustes(source_np, **alignment)
            shuffled_np = apply_procrustes(source_np, **shuffled)
            random_np = source_np @ random_rotation
            head = models["classification"].Classification.mlp_class
            aligned = head(torch.as_tensor(aligned_np, dtype=source_rep.dtype, device=device)).reshape(-1)
            shuffled_score = head(torch.as_tensor(shuffled_np, dtype=source_rep.dtype, device=device)).reshape(-1)
            random_score = head(torch.as_tensor(random_np, dtype=source_rep.dtype, device=device)).reshape(-1)
            reco_joint_direct = models["joint"].Classification.mlp_class(source_rep).reshape(-1)
            reco_joint_np = apply_procrustes(source_np, **reco_to_joint)
            reco_joint_aligned = models["joint"].Classification.mlp_class(
                torch.as_tensor(reco_joint_np, dtype=source_rep.dtype, device=device)
            ).reshape(-1)
            joint_rep = joint_reps["final_event"]
            joint_class_direct = head(joint_rep).reshape(-1)
            joint_class_np = apply_procrustes(joint_rep.float().cpu().numpy(), **joint_to_class)
            joint_class_aligned = head(
                torch.as_tensor(joint_class_np, dtype=joint_rep.dtype, device=device)
            ).reshape(-1)
            indices = batch.source_event_index.detach().cpu().reshape(-1).numpy().astype(np.int64)
            missing = [int(index) for index in indices if int(index) not in reconstruction_scores]
            if missing:
                raise KeyError(f"Reconstruction score missing for source_event_index values {missing[:10]}.")
            missing_joint = [int(index) for index in indices if int(index) not in joint_reconstruction_scores]
            if missing_joint:
                raise KeyError(f"Joint reconstruction score missing for source_event_index values {missing_joint[:10]}.")
            frame = pd.DataFrame({
                "source_event_index": indices,
                "truth_class": batch.cls_t.detach().cpu().reshape(-1).numpy().astype(np.int8),
                "truth_fully_matched": truth_fully_matched(batch),
                "native_classification_only_score": torch.sigmoid(native_logits).cpu().numpy(),
                "native_joint_score": torch.sigmoid(joint_outputs[3].reshape(-1)).cpu().numpy(),
                "reconstruction_zero_shot_score": [reconstruction_scores[int(index)] for index in indices],
                "joint_reconstruction_zero_shot_score": [joint_reconstruction_scores[int(index)] for index in indices],
                "direct_head_swap_score": torch.sigmoid(direct).cpu().numpy(),
                "aligned_head_transfer_score": torch.sigmoid(aligned).cpu().numpy(),
                "shuffled_alignment_score": torch.sigmoid(shuffled_score).cpu().numpy(),
                "random_orthogonal_score": torch.sigmoid(random_score).cpu().numpy(),
                "reconstruction_to_joint_direct_score": torch.sigmoid(reco_joint_direct).cpu().numpy(),
                "reconstruction_to_joint_aligned_score": torch.sigmoid(reco_joint_aligned).cpu().numpy(),
                "joint_to_classification_direct_score": torch.sigmoid(joint_class_direct).cpu().numpy(),
                "joint_to_classification_aligned_score": torch.sigmoid(joint_class_aligned).cpu().numpy(),
            })
            if not np.isfinite(frame.select_dtypes(include=[np.number])).all().all():
                raise RuntimeError("Transfer output contains non-finite values.")
            frames.append(frame)
            rows += len(frame)
            if sum(len(value) for value in frames) >= 100000:
                pd.concat(frames, ignore_index=True).to_pickle(parts_dir / f"part_{part:06d}.pkl")
                frames, part = [], part + 1
    if frames:
        pd.concat(frames, ignore_index=True).to_pickle(parts_dir / f"part_{part:06d}.pkl")
    all_frames = [pd.read_pickle(path) for path in sorted(parts_dir.glob("part_*.pkl"))]
    frame = pd.concat(all_frames, ignore_index=True)
    if frame.source_event_index.duplicated().any() or len(frame) != rows:
        raise RuntimeError("Streamed transfer output has duplicate or missing source-event rows.")
    labels, fm = frame.truth_class.to_numpy(), frame.truth_fully_matched.to_numpy().astype(bool)
    subsets = {
        "inclusive_all_signal": np.ones(len(frame), dtype=bool),
        "fully_matched_signal_vs_background": (labels == 0) | ((labels == 1) & fm),
        "non_fully_matched_signal_vs_background": (labels == 0) | ((labels == 1) & ~fm),
    }
    metric_rows = []
    native = frame.native_classification_only_score.to_numpy()
    for subset, mask in subsets.items():
        for method in METHOD_LABELS:
            metric_rows.append(metric_row(method, subset, labels[mask], frame[method].to_numpy()[mask], native[mask]))
    with (output_dir / "transfer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    summary = {
        "event_count": int(len(frame)), "test_split": "test", "alignment_fit_split": "val",
        "methods": METHOD_LABELS, "metrics": metric_rows, "labels_used_for_alignment": False,
        "zero_shot_score_definition": "product of four selected ttbar-SL reconstruction-role probabilities",
        "event_output": str(parts_dir),
    }
    (output_dir / "transfer_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_plots(frame, metric_rows, output_dir)
    print(f"wrote={output_dir} events={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
