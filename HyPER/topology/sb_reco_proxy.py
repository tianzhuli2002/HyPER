#!/usr/bin/env python3
"""Evaluate reconstruction-derived HyPER scores as S/B proxy discriminants."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .prediction_io import iter_hyper_prediction_parts
except ImportError:  # pragma: no cover - direct script execution.
    from prediction_io import iter_hyper_prediction_parts


LABEL_CANDIDATES = (
    "HyPER_CLS_T",
    "truth_label",
    "label",
    "cls_t",
    "class_label",
    "true_label",
    "y",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", help="Prediction .pkl, .pkl.parts, or .h5 input(s).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--h5-key", default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--top-distributions", type=int, default=5)
    parser.add_argument("--formats", nargs="+", default=["pdf"])
    return parser.parse_args()


def _iter_input_chunks(path: str, max_events: int | None, chunk_size: int | None, h5_key: str | None):
    input_path = Path(path)
    if input_path.name.lower().endswith((".h5", ".hdf5")):
        if h5_key is None:
            frame = pd.read_hdf(input_path)
            if max_events is not None:
                frame = frame.iloc[:max_events].copy()
            yield frame
            return
        try:
            iterator = pd.read_hdf(input_path, key=h5_key, chunksize=chunk_size)
        except TypeError:
            frame = pd.read_hdf(input_path, key=h5_key)
            if max_events is not None:
                frame = frame.iloc[:max_events].copy()
            yield frame
            return
        loaded = 0
        for chunk in iterator:
            if max_events is not None and loaded >= max_events:
                break
            if max_events is not None and loaded + len(chunk) > max_events:
                chunk = chunk.iloc[: max_events - loaded].copy()
            loaded += len(chunk)
            yield chunk
        return

    yield from iter_hyper_prediction_parts(input_path, max_events=max_events, chunk_size=chunk_size)


def _resolve_label_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise KeyError(f"Requested label column not found: {requested}")
        print(f"Using requested label column: {requested}")
        return requested

    matches = [name for name in LABEL_CANDIDATES if name in frame.columns]
    if len(matches) == 1:
        print(f"Autodetected label column: {matches[0]}")
        return matches[0]
    if not matches:
        raise KeyError(
            "No S/B label column found. Provide --label-column. "
            f"Checked: {', '.join(LABEL_CANDIDATES)}"
        )
    raise KeyError(
        "Multiple plausible label columns found; provide --label-column explicitly: "
        + ", ".join(matches)
    )


def _binary_labels(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    labels = np.full(arr.shape, -1, dtype=np.int8)
    finite = np.isfinite(arr)
    labels[finite] = (arr[finite] > 0.5).astype(np.int8)
    return labels


def _as_float_array(values) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _topk_mean(values, k: int) -> float:
    arr = _as_float_array(values)
    if arr.size == 0:
        return float("nan")
    k = min(int(k), arr.size)
    if k <= 0:
        return float("nan")
    return float(np.partition(arr, -k)[-k:].mean())


def _raw_family_scores(frame: pd.DataFrame, column: str, prefix: str) -> dict[str, np.ndarray]:
    if column not in frame.columns:
        warnings.warn(f"Missing raw proxy column {column}; skipping {prefix} proxies.", UserWarning)
        return {}

    output = {
        f"{prefix}_max": [],
        f"{prefix}_mean": [],
        f"{prefix}_top3_mean": [],
        f"{prefix}_top5_mean": [],
        f"{prefix}_margin_top1_top2": [],
    }
    for values in frame[column]:
        arr = _as_float_array(values)
        if arr.size == 0:
            max_score = mean_score = top3 = top5 = margin = float("nan")
        else:
            sorted_scores = np.sort(arr)[::-1]
            max_score = float(sorted_scores[0])
            mean_score = float(arr.mean())
            top3 = float(sorted_scores[: min(3, sorted_scores.size)].mean())
            top5 = float(sorted_scores[: min(5, sorted_scores.size)].mean())
            margin = (
                float(sorted_scores[0] - sorted_scores[1])
                if sorted_scores.size >= 2
                else float("nan")
            )
        output[f"{prefix}_max"].append(max_score)
        output[f"{prefix}_mean"].append(mean_score)
        output[f"{prefix}_top3_mean"].append(top3)
        output[f"{prefix}_top5_mean"].append(top5)
        output[f"{prefix}_margin_top1_top2"].append(margin)
    return {name: np.asarray(values, dtype=float) for name, values in output.items()}


def _selected_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    selected_columns = [
        name for name in frame.columns
        if name.startswith("HyPER_best_") and name.endswith("_prob")
    ]
    if not selected_columns:
        warnings.warn("No selected HyPER_best_*_prob columns found; skipping selected proxies.", UserWarning)
        return {}

    selected_frame = frame[selected_columns].apply(pd.to_numeric, errors="coerce")
    output = {
        "selected_mean_prob": selected_frame.mean(axis=1, skipna=True).to_numpy(dtype=float),
        "selected_min_prob": selected_frame.min(axis=1, skipna=True).to_numpy(dtype=float),
        "selected_max_prob": selected_frame.max(axis=1, skipna=True).to_numpy(dtype=float),
    }
    for name in selected_columns:
        output[name] = selected_frame[name].to_numpy(dtype=float)
    return output


def _score_chunk(frame: pd.DataFrame, label_column: str) -> pd.DataFrame:
    if label_column not in frame.columns:
        raise KeyError(
            f"Label column {label_column!r} is missing from prediction chunk. "
            "Run HyPER.predict with a dataset that has LABELS/GLOBAL so "
            "HyPER_CLS_T is written, or pass --label-column for an existing label."
        )
    labels = _binary_labels(frame[label_column])
    scores: dict[str, np.ndarray] = {}
    scores.update(_selected_scores(frame))
    scores.update(_raw_family_scores(frame, "HyPER_HE_RAW", "he"))
    scores.update(_raw_family_scores(frame, "HyPER_GE_RAW", "ge"))

    if {"he_max", "ge_max"}.issubset(scores):
        scores["combined_max_mean"] = 0.5 * scores["he_max"] + 0.5 * scores["ge_max"]
    else:
        warnings.warn("Cannot build combined_max_mean without he_max and ge_max.", UserWarning)
    if {"he_top3_mean", "ge_top3_mean"}.issubset(scores):
        scores["combined_top3_mean"] = 0.5 * scores["he_top3_mean"] + 0.5 * scores["ge_top3_mean"]
    else:
        warnings.warn("Cannot build combined_top3_mean without he_top3_mean and ge_top3_mean.", UserWarning)

    out = pd.DataFrame({"label": labels})
    for name in sorted(scores):
        out[name] = scores[name]
    return out


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    keep = np.isfinite(scores) & np.isin(labels, [0, 1])
    scores = scores[keep]
    labels = labels[keep]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.asarray([]), np.asarray([]), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    tpr = np.concatenate([[0.0], tp / float(n_pos), [1.0]])
    fpr = np.concatenate([[0.0], fp / float(n_neg), [1.0]])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def _save_fig(fig, output_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def _write_summary_and_plots(score_store, label_store, output_dir: Path, formats: list[str], top_distributions: int):
    labels = np.concatenate(label_store).astype(np.int8) if label_store else np.asarray([], dtype=np.int8)
    summary_rows = []
    roc_data = {}

    for name in sorted(score_store):
        scores = np.concatenate(score_store[name]).astype(float)
        keep = np.isfinite(scores) & np.isin(labels, [0, 1])
        y = labels[keep]
        s = scores[keep]
        if y.size == 0:
            auc = float("nan")
            fpr = tpr = np.asarray([])
            n_signal = n_background = 0
        else:
            fpr, tpr, auc = _roc_auc(scores, labels)
            n_signal = int((y == 1).sum())
            n_background = int((y == 0).sum())
        if n_signal == 0 or n_background == 0:
            auc = float("nan")
        auc_abs = max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan")
        preferred = "score_high_signal" if math.isfinite(auc) and auc >= 0.5 else "score_low_signal"
        summary_rows.append(
            {
                "score": name,
                "n_valid": int(keep.sum()),
                "n_signal": n_signal,
                "n_background": n_background,
                "auc": auc,
                "auc_abs": auc_abs,
                "preferred_direction": preferred,
                "score_min": float(np.nanmin(s)) if s.size else float("nan"),
                "score_max": float(np.nanmax(s)) if s.size else float("nan"),
                "score_mean": float(np.nanmean(s)) if s.size else float("nan"),
                "score_std": float(np.nanstd(s)) if s.size else float("nan"),
            }
        )
        if fpr.size and tpr.size and math.isfinite(auc):
            roc_data[name] = (fpr, tpr, auc, auc_abs)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["auc_abs", "n_valid"], ascending=[False, False]
    )
    summary.to_csv(output_dir / "reco_proxy_summary.csv", index=False)
    with (output_dir / "reco_proxy_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_rows": int(len(labels)),
                "n_scores": int(len(summary)),
                "scores": summary.to_dict(orient="records"),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    if not summary.empty and roc_data:
        fig, ax = plt.subplots(figsize=(7, 7))
        for _, row in summary.head(12).iterrows():
            name = row["score"]
            if name not in roc_data:
                continue
            fpr, tpr, auc, auc_abs = roc_data[name]
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f}, abs={auc_abs:.3f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Reconstruction proxy ROC")
        ax.legend(loc="lower right", fontsize=8)
        _save_fig(fig, output_dir, "roc_all_proxy_scores", formats)

        best_name = str(summary.iloc[0]["score"])
        if best_name in roc_data:
            fpr, tpr, auc, auc_abs = roc_data[best_name]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, label=f"{best_name} (AUC={auc:.4f}, abs={auc_abs:.4f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="0.5")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("Best reconstruction proxy ROC")
            ax.legend(loc="lower right")
            _save_fig(fig, output_dir, "roc_best_proxy_score", formats)

    for idx, row in enumerate(summary.head(max(1, int(top_distributions))).itertuples(index=False)):
        name = row.score
        scores = np.concatenate(score_store[name]).astype(float)
        keep = np.isfinite(scores) & np.isin(labels, [0, 1])
        y = labels[keep]
        s = scores[keep]
        if not s.size:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(s[y == 0], bins=60, alpha=0.55, density=True, label="background")
        ax.hist(s[y == 1], bins=60, alpha=0.55, density=True, label="signal")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"Proxy score distribution: {name}")
        ax.legend()
        stem = "score_distribution_best_proxy" if idx == 0 else f"score_distribution_{name}"
        _save_fig(fig, output_dir, stem, formats)

    return summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / "reco_proxy_scores.csv"

    score_store: dict[str, list[np.ndarray]] = {}
    label_store: list[np.ndarray] = []
    label_column = args.label_column
    wrote_header = False
    total_rows = 0
    remaining = args.max_events

    for input_path in args.input:
        for chunk in _iter_input_chunks(
            input_path,
            max_events=remaining,
            chunk_size=args.chunk_size,
            h5_key=args.h5_key,
        ):
            if len(chunk) == 0:
                continue
            if label_column is None:
                label_column = _resolve_label_column(chunk, args.label_column)
            scores = _score_chunk(chunk, label_column)
            valid_chunk_labels = scores.loc[scores["label"].isin([0, 1]), "label"]
            if valid_chunk_labels.nunique() < 2:
                warnings.warn(
                    "Current chunk does not contain both labels; global AUC will be checked at the end.",
                    UserWarning,
                )

            mode = "a" if wrote_header else "w"
            scores.to_csv(scores_csv, mode=mode, header=not wrote_header, index=False)
            wrote_header = True

            labels = scores["label"].to_numpy(dtype=np.int8)
            label_store.append(labels)
            for name in scores.columns:
                if name == "label":
                    continue
                score_store.setdefault(name, []).append(scores[name].to_numpy(dtype=float))
            total_rows += len(scores)
            if remaining is not None:
                remaining -= len(scores)
                if remaining <= 0:
                    break
        if remaining is not None and remaining <= 0:
            break

    if total_rows == 0:
        raise RuntimeError("No prediction rows were processed.")
    labels_all = np.concatenate(label_store) if label_store else np.asarray([], dtype=np.int8)
    if np.isin(labels_all, [0, 1]).sum() == 0 or len(np.unique(labels_all[np.isin(labels_all, [0, 1])])) < 2:
        raise RuntimeError("AUC requires both S/B classes; labels did not contain both classes.")
    if not score_store:
        raise RuntimeError("No reconstruction proxy scores could be built from the input columns.")

    summary = _write_summary_and_plots(
        score_store=score_store,
        label_store=label_store,
        output_dir=output_dir,
        formats=args.formats,
        top_distributions=args.top_distributions,
    )
    print(f"Wrote {scores_csv}")
    print(f"Wrote {output_dir / 'reco_proxy_summary.csv'}")
    print(summary.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
