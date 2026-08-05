#!/usr/bin/env python3
"""Run one-pass full-test HyPER transfer evaluation and dense alignment controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from HyPER.analysis.representations import random_orthogonal
from HyPER.analysis.runtime import (
    build_analysis_datamodule,
    forward_representations,
    load_analysis_config,
    load_frozen_model,
    truth_fully_matched,
    resource_diagnostics,
    write_resource_diagnostics,
)
from HyPER.topology.prediction_io import iter_hyper_prediction_parts
from HyPER.topology.reconstruction_score import (
    event_reconstruction_score,
    normalise_reconstruction_topology,
)

DIRECTIONS = {
    "reconstruction_to_classification": ("reconstruction", "classification"),
    "reconstruction_to_joint": ("reconstruction", "joint"),
    "joint_to_classification": ("joint", "classification"),
}

PRINCIPAL_SCORE_FIELDS = (
    "native_classification_only_score",
    "native_joint_score",
    "reconstruction_zero_shot_score",
    "joint_reconstruction_zero_shot_score",
    "reconstruction_to_classification_direct_score",
    "reconstruction_to_classification_paired_score",
    "reconstruction_to_joint_direct_score",
    "reconstruction_to_joint_paired_score",
    "joint_to_classification_direct_score",
    "joint_to_classification_paired_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in ("classification", "reconstruction", "joint"):
        parser.add_argument(f"--{mode}-config", required=True)
        parser.add_argument(f"--{mode}-checkpoint", required=True)
    parser.add_argument("--topology", choices=("ttbar1L", "ttH"), required=True)
    parser.add_argument("--reconstruction-predictions", required=True)
    parser.add_argument("--joint-predictions", required=True)
    for direction in DIRECTIONS:
        parser.add_argument(f"--{direction.replace('_', '-')}-alignments", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split-cache", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--expected-event-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-random-controls", type=int, default=20)
    parser.add_argument("--expected-shuffles", type=int, default=50)
    parser.add_argument("--expected-alignment-event-count", type=int, default=100000)
    parser.add_argument("--random-seed-start", type=int, default=1000)
    parser.add_argument("--control-chunk-size", type=int, default=2)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()



def validate_alignment_root(
    root: str,
    direction: str,
    expected_shuffles: int,
    expected_event_count: int,
) -> Path:
    directory = Path(root).expanduser().resolve()
    summary_path = directory / "alignment_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("direction") != direction:
        raise RuntimeError(
            f"Alignment root {directory} is for {summary.get('direction')!r}, expected {direction!r}."
        )
    if summary.get("fit_event_count") != expected_event_count:
        raise RuntimeError(
            f"{direction} alignment used {summary.get('fit_event_count')} events; "
            f"expected {expected_event_count}."
        )
    if summary.get("num_shuffled_alignments") != expected_shuffles:
        raise RuntimeError(
            f"{direction} has {summary.get('num_shuffled_alignments')} shuffled alignments; "
            f"expected {expected_shuffles}."
        )
    if summary.get("labels_loaded_for_fitting") is not False:
        raise RuntimeError(f"{direction} alignment is not marked label-free.")
    seeds = summary.get("shuffle_seeds", [])
    if len(seeds) != expected_shuffles or len(set(seeds)) != expected_shuffles:
        raise RuntimeError(f"{direction} shuffled seeds are incomplete or duplicated.")
    return directory

def load_alignment(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        required = ("source_mean", "target_mean", "rotation")
        missing = [name for name in required if name not in loaded]
        if missing:
            raise KeyError(f"Alignment {path} is missing {missing}.")
        result = {name: np.asarray(loaded[name], dtype=np.float32) for name in required}
    rotation = result["rotation"]
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"Alignment {path} rotation is not square: {rotation.shape}.")
    return result


def load_alignment_ensemble(root: str) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    directory = Path(root).expanduser().resolve()
    paired_path = directory / "paired.npz"
    if not paired_path.is_file():
        raise FileNotFoundError(paired_path)
    shuffled_paths = sorted((directory / "shuffled").glob("seed_*.npz"))
    if not shuffled_paths:
        raise FileNotFoundError(f"No seed_*.npz shuffled alignments under {directory / 'shuffled'}.")
    controls = []
    for path in shuffled_paths:
        seed = int(path.stem.split("_")[-1])
        controls.append({"seed": seed, "path": str(path), **load_alignment(path)})
    seeds = [int(item["seed"]) for item in controls]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"Duplicate shuffled seeds under {directory}.")
    return load_alignment(paired_path), controls


def load_reconstruction_scores(path: str, topology: str) -> dict[int, float]:
    topology = normalise_reconstruction_topology(topology)
    scores: dict[int, float] = {}
    for frame in iter_hyper_prediction_parts(Path(path), max_events=None, chunk_size=100000):
        required = {"source_event_index", "selected_reconstruction_scores"}
        if not required.issubset(frame.columns):
            raise KeyError(f"Reconstruction predictions lack {sorted(required - set(frame.columns))}.")
        for index, values in zip(frame["source_event_index"], frame["selected_reconstruction_scores"]):
            index = int(index)
            if index in scores:
                raise ValueError(f"Duplicate reconstruction source_event_index={index}.")
            scores[index] = event_reconstruction_score(values, topology)
    if not scores:
        raise RuntimeError(f"No reconstruction scores were loaded from {path}.")
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        invalid = int(np.sum(~np.isfinite(values)))
        raise RuntimeError(
            f"{invalid} {topology} reconstruction scores are non-finite; "
            "check selected_reconstruction_scores and topology compatibility."
        )
    return scores


def deterministic_limit(indices: np.ndarray, count: int | None, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if count is None or count >= len(indices):
        return indices
    if count <= 0:
        raise ValueError("--max-events must be positive.")
    positions = np.sort(np.random.default_rng(seed).choice(len(indices), count, replace=False))
    return indices[positions]


def alignment_to_device(
    alignment: dict[str, np.ndarray],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(alignment[name], dtype=dtype, device=device)
        for name in ("source_mean", "target_mean", "rotation")
    }


def controls_to_device(
    controls: list[dict[str, object]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if not controls:
        return {
            "source_mean": torch.empty((0, 0), dtype=dtype, device=device),
            "target_mean": torch.empty((0, 0), dtype=dtype, device=device),
            "rotation": torch.empty((0, 0, 0), dtype=dtype, device=device),
        }
    return {
        "source_mean": torch.as_tensor(
            np.stack([item["source_mean"] for item in controls]), dtype=dtype, device=device
        ),
        "target_mean": torch.as_tensor(
            np.stack([item["target_mean"] for item in controls]), dtype=dtype, device=device
        ),
        "rotation": torch.as_tensor(
            np.stack([item["rotation"] for item in controls]), dtype=dtype, device=device
        ),
    }


def evaluate_alignment(
    source: torch.Tensor,
    alignment: dict[str, torch.Tensor],
    head,
) -> torch.Tensor:
    aligned = (source - alignment["source_mean"]) @ alignment["rotation"]
    aligned = aligned + alignment["target_mean"]
    return torch.sigmoid(head(aligned).reshape(-1))


def evaluate_control_chunk(
    source: torch.Tensor,
    controls: dict[str, torch.Tensor],
    head,
    start: int,
    stop: int,
) -> np.ndarray:
    source_mean = controls["source_mean"][start:stop]
    target_mean = controls["target_mean"][start:stop]
    rotation = controls["rotation"][start:stop]
    centred = source.unsqueeze(0) - source_mean.unsqueeze(1)
    aligned = torch.einsum("cbd,cde->cbe", centred, rotation) + target_mean.unsqueeze(1)
    flat = aligned.reshape(-1, aligned.shape[-1])
    score = torch.sigmoid(head(flat).reshape(stop - start, len(source)))
    return score.detach().float().cpu().numpy()


def random_control_ensemble(
    paired: dict[str, np.ndarray], count: int, seed_start: int
) -> list[dict[str, object]]:
    dimension = paired["rotation"].shape[0]
    return [
        {
            "seed": seed,
            "source_mean": paired["source_mean"],
            "target_mean": paired["target_mean"],
            "rotation": random_orthogonal(dimension, seed).astype(np.float32),
        }
        for seed in range(seed_start, seed_start + count)
    ]


def create_memmap(path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    if args.control_chunk_size <= 0:
        raise ValueError("--control-chunk-size must be positive.")
    if args.num_random_controls <= 0:
        raise ValueError("--num-random-controls must be positive.")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Production score output already exists and is non-empty: {output}")
    scores_dir = output / "scores"
    controls_dir = output / "controls"
    parts_dir = output / "transfer_events.pkl.parts"
    scores_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU acceleration was requested but PyTorch cannot access CUDA.")
    device = torch.device("cuda:0" if args.accelerator == "gpu" else "cpu")
    cfg = load_analysis_config(args.reconstruction_config)
    module = build_analysis_datamodule(
        cfg,
        h5=args.h5,
        split_cache=args.split_cache,
        split="test",
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    selected = deterministic_limit(module.predict_data.indices, args.max_events, args.seed)
    module.predict_data.indices = selected
    event_count = len(selected)
    if args.expected_event_count is not None and event_count != args.expected_event_count:
        raise RuntimeError(f"Selected {event_count} events, expected {args.expected_event_count}.")

    models = {
        mode: load_frozen_model(getattr(args, f"{mode}_checkpoint"), device)
        for mode in ("classification", "reconstruction", "joint")
    }
    if models["classification"].Classification is None or models["joint"].Classification is None:
        raise RuntimeError("Target checkpoints must contain classification heads.")

    alignment_roots = {
        direction: validate_alignment_root(
            getattr(args, f"{direction}_alignments"),
            direction,
            args.expected_shuffles,
            args.expected_alignment_event_count,
        )
        for direction in DIRECTIONS
    }
    paired = {}
    shuffled = {}
    random_controls = {}
    for direction, root in alignment_roots.items():
        paired[direction], shuffled[direction] = load_alignment_ensemble(str(root))
        random_controls[direction] = random_control_ensemble(
            paired[direction], args.num_random_controls, args.random_seed_start
        )
    model_dtype = next(models["classification"].parameters()).dtype
    paired_device = {
        direction: alignment_to_device(alignment, device, model_dtype)
        for direction, alignment in paired.items()
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        print(
            f"paired_alignment_controls_on_gpu={torch.cuda.memory_allocated(device) / 1024**3:.3f} GiB",
            flush=True,
        )

    topology = normalise_reconstruction_topology(args.topology)
    reconstruction_scores = load_reconstruction_scores(args.reconstruction_predictions, topology)
    joint_reconstruction_scores = load_reconstruction_scores(args.joint_predictions, topology)

    canonical = {
        "source_event_index": create_memmap(scores_dir / "test_source_event_index.npy", (event_count,), np.int64),
        "truth_class": create_memmap(scores_dir / "test_truth_class.npy", (event_count,), np.int8),
        "truth_fully_matched": create_memmap(scores_dir / "test_truth_fully_matched.npy", (event_count,), np.int8),
    }
    for field in PRINCIPAL_SCORE_FIELDS:
        canonical[field] = create_memmap(scores_dir / f"{field}.npy", (event_count,), np.float32)

    control_maps = {}
    control_metadata = {}
    for direction in DIRECTIONS:
        control_maps[(direction, "shuffled")] = create_memmap(
            controls_dir / f"{direction}_shuffled_scores.npy",
            (len(shuffled[direction]), event_count),
            np.float32,
        )
        control_maps[(direction, "random")] = create_memmap(
            controls_dir / f"{direction}_random_scores.npy",
            (len(random_controls[direction]), event_count),
            np.float32,
        )
        control_metadata[direction] = {
            "shuffled": [
                {"row": row, "seed": int(item["seed"]), "alignment_path": str(item["path"])}
                for row, item in enumerate(shuffled[direction])
            ],
            "random": [
                {"row": row, "seed": int(item["seed"])}
                for row, item in enumerate(random_controls[direction])
            ],
        }

    row_start = 0
    part_number = 0
    buffered_frames = []
    buffered_rows = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(module.predict_dataloader(), start=1):
            batch = batch.to(device)
            class_outputs, class_reps = forward_representations(models["classification"], batch)
            reco_outputs, reco_reps = forward_representations(models["reconstruction"], batch)
            joint_outputs, joint_reps = forward_representations(models["joint"], batch)
            native_class_logit = class_outputs[3].reshape(-1)
            reproduced = models["classification"].Classification.mlp_class(
                class_reps["classification_head_input"]
            ).reshape(-1)
            np.testing.assert_allclose(
                reproduced.detach().cpu().numpy(),
                native_class_logit.detach().cpu().numpy(),
                rtol=1e-5,
                atol=1e-6,
            )
            reco_rep = reco_reps["final_event"]
            joint_rep = joint_reps["final_event"]
            class_head = models["classification"].Classification.mlp_class
            joint_head = models["joint"].Classification.mlp_class
            indices = batch.source_event_index.detach().cpu().reshape(-1).numpy().astype(np.int64)
            row_stop = row_start + len(indices)
            missing_reco = [int(index) for index in indices if int(index) not in reconstruction_scores]
            missing_joint = [int(index) for index in indices if int(index) not in joint_reconstruction_scores]
            if missing_reco or missing_joint:
                raise KeyError(
                    f"Missing reconstruction scores: reco={missing_reco[:5]}, joint={missing_joint[:5]}."
                )
            frame_data = {
                "source_event_index": indices,
                "truth_class": batch.cls_t.detach().cpu().reshape(-1).numpy().astype(np.int8),
                "truth_fully_matched": truth_fully_matched(batch, topology).astype(np.int8),
                "native_classification_only_score": torch.sigmoid(native_class_logit).cpu().numpy(),
                "native_joint_score": torch.sigmoid(joint_outputs[3].reshape(-1)).cpu().numpy(),
                "reconstruction_zero_shot_score": np.asarray(
                    [reconstruction_scores[int(index)] for index in indices], dtype=np.float32
                ),
                "joint_reconstruction_zero_shot_score": np.asarray(
                    [joint_reconstruction_scores[int(index)] for index in indices], dtype=np.float32
                ),
                "reconstruction_to_classification_direct_score": torch.sigmoid(
                    class_head(reco_rep).reshape(-1)
                ).cpu().numpy(),
                "reconstruction_to_classification_paired_score": evaluate_alignment(
                    reco_rep, paired_device["reconstruction_to_classification"], class_head
                ).cpu().numpy(),
                "reconstruction_to_joint_direct_score": torch.sigmoid(
                    joint_head(reco_rep).reshape(-1)
                ).cpu().numpy(),
                "reconstruction_to_joint_paired_score": evaluate_alignment(
                    reco_rep, paired_device["reconstruction_to_joint"], joint_head
                ).cpu().numpy(),
                "joint_to_classification_direct_score": torch.sigmoid(
                    class_head(joint_rep).reshape(-1)
                ).cpu().numpy(),
                "joint_to_classification_paired_score": evaluate_alignment(
                    joint_rep, paired_device["joint_to_classification"], class_head
                ).cpu().numpy(),
            }
            for field, values in frame_data.items():
                values = np.asarray(values)
                canonical[field][row_start:row_stop] = values
                if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                    raise RuntimeError(f"Non-finite values in {field} for batch {batch_number}.")

            source_by_direction = {
                "reconstruction_to_classification": reco_rep,
                "reconstruction_to_joint": reco_rep,
                "joint_to_classification": joint_rep,
            }
            head_by_direction = {
                "reconstruction_to_classification": class_head,
                "reconstruction_to_joint": joint_head,
                "joint_to_classification": class_head,
            }
            for direction in DIRECTIONS:
                for control_type, control_list in (
                    ("shuffled", shuffled[direction]),
                    ("random", random_controls[direction]),
                ):
                    destination = control_maps[(direction, control_type)]
                    for control_start in range(0, len(control_list), args.control_chunk_size):
                        control_stop = min(control_start + args.control_chunk_size, len(control_list))
                        # Keep only the bounded control chunk on the accelerator;
                        # production-sized ensembles remain resident on host RAM.
                        device_controls = controls_to_device(
                            control_list[control_start:control_stop], device, model_dtype
                        )
                        scores = evaluate_control_chunk(
                            source_by_direction[direction],
                            device_controls,
                            head_by_direction[direction],
                            0,
                            control_stop - control_start,
                        )
                        destination[control_start:control_stop, row_start:row_stop] = scores
                        del device_controls, scores

            frame = pd.DataFrame(frame_data)
            buffered_frames.append(frame)
            buffered_rows += len(frame)
            if buffered_rows >= 100000:
                pd.concat(buffered_frames, ignore_index=True).to_pickle(
                    parts_dir / f"part_{part_number:06d}.pkl"
                )
                buffered_frames = []
                buffered_rows = 0
                part_number += 1
            row_start = row_stop
            if batch_number % 20 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"batch={batch_number} events={row_start}/{event_count} "
                    f"events_per_second={row_start / elapsed:.2f}",
                    flush=True,
                )
    if buffered_frames:
        pd.concat(buffered_frames, ignore_index=True).to_pickle(
            parts_dir / f"part_{part_number:06d}.pkl"
        )
    if row_start != event_count:
        raise RuntimeError(f"Wrote {row_start} events but expected {event_count}.")
    if not np.array_equal(np.asarray(canonical["source_event_index"]), selected):
        raise RuntimeError("Canonical event order differs from the selected test indices.")
    for value in canonical.values():
        value.flush()
    for value in control_maps.values():
        value.flush()
    (controls_dir / "control_metadata.json").write_text(
        json.dumps(control_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    summary = {
        "event_count": event_count,
        "split": "test",
        "principal_score_fields": list(PRINCIPAL_SCORE_FIELDS),
        "canonical_event_order": str(scores_dir / "test_source_event_index.npy"),
        "parts_output": str(parts_dir),
        "alignment_fit_split": "val",
        "labels_used_for_alignment": False,
        "topology": topology,
        "zero_shot_score_definition": (
            "p_top1 * p_top2 * p_W1 * p_W2"
            if topology == "ttbar1L"
            else "p_tlep * p_thad * p_Wlep * p_Whad * p_H"
        ),
        "alignment_directions": control_metadata,
        "num_random_controls": args.num_random_controls,
        "control_chunk_size": args.control_chunk_size,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "events_per_second": float(event_count / elapsed) if elapsed > 0 else None,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    diagnostics = resource_diagnostics(stage="evaluate", started=started, events_processed=event_count, output_root=output)
    diagnostics.update({"topology": topology, "control_chunk_size": args.control_chunk_size})
    write_resource_diagnostics(output, diagnostics)
    print(f"wrote={output} events={event_count} elapsed_seconds={elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
