"""Streaming, source-aligned HyPER prediction."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
from datetime import datetime, timezone
from itertools import combinations, permutations
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from HyPER.checkpoints import checkpoint_metadata, resolve_checkpoint
from HyPER.configuration import TaskSpec, validate_runtime_config
from HyPER.data import HyPERDataModule
from HyPER.factories import graph_config, plain
from HyPER.models import HyPERModel
from HyPER.topology.ttbar import reconstruct_ttbar1l
from HyPER.topology.tth import reconstruct_tth


TOPOLOGY_REGISTRY = {
    "ttbar1L": reconstruct_ttbar1l,
    "ttH": reconstruct_tth,
}


def plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _graph_config(cfg: DictConfig) -> dict:
    graph = plain(OmegaConf.create({"input": cfg.input, "target": cfg.target}))
    graph.setdefault("target", {})["encoding"] = "typed"
    return graph



def symmetrise_directed_edge_logits(logits: torch.Tensor, num_nodes: int):
    """Average both directed logit vectors for each canonical physical pair."""
    directed_pairs = list(permutations(range(int(num_nodes)), 2))
    if logits.ndim != 2 or logits.size(0) != len(directed_pairs):
        raise ValueError(
            f"Expected {len(directed_pairs)} directed edge-logit rows for {num_nodes} nodes, "
            f"got {tuple(logits.shape)}."
        )
    row_by_pair = {pair: row for row, pair in enumerate(directed_pairs)}
    if len(row_by_pair) != len(directed_pairs):
        raise RuntimeError("Directed edge list contains duplicates.")
    physical_pairs = list(combinations(range(int(num_nodes)), 2))
    averaged = []
    for i, j in physical_pairs:
        if (i, j) not in row_by_pair or (j, i) not in row_by_pair:
            raise RuntimeError(f"Physical pair {(i, j)} does not have exactly both directed arcs.")
        averaged.append((logits[row_by_pair[(i, j)]] + logits[row_by_pair[(j, i)]]) / 2.0)
    if not averaged:
        return physical_pairs, logits.new_empty((0, logits.size(1)))
    return physical_pairs, torch.stack(averaged)


def _nonbackground_score(probabilities: torch.Tensor) -> list[float]:
    if probabilities.ndim != 2 or probabilities.size(1) < 2:
        raise ValueError("Typed probabilities must include at least one role and background.")
    return probabilities[:, :-1].max(dim=1).values.cpu().tolist()


def _raw_frame(predictions: list[dict], hyperedge_order: int, edge_names, hyperedge_names) -> pd.DataFrame:
    rows = []
    for prediction in predictions:
        count = len(prediction["node_counts"])
        for event in range(count):
            n_nodes = int(prediction["node_counts"][event])
            node_types = prediction["node_types"][event].detach().cpu().reshape(-1).long().tolist()
            node_p4 = prediction["node_p4"][event].detach().cpu().tolist()
            node_ids = prediction["node_ids"][event].detach().cpu().reshape(-1).long().tolist()
            node_truth = prediction["node_truth_ids"][event].detach().cpu().reshape(-1).long().tolist()
            if len(node_types) != n_nodes or len(node_p4) != n_nodes:
                raise RuntimeError(
                    f"Prediction node feature rows do not match number_of_nodes={n_nodes} "
                    f"for source_event_index={int(prediction['source_event_index'][event])}."
                )
            if len(node_ids) != n_nodes or len(node_truth) != n_nodes:
                raise RuntimeError(
                    f"Prediction node identity rows do not match number_of_nodes={n_nodes} "
                    f"for source_event_index={int(prediction['source_event_index'][event])}."
                )
            cls_logit = prediction["classification_logits"]
            cls_prob = prediction["classification_probabilities"]
            cls_target = prediction["classification_target"]
            row = {
                "source_event_index": int(prediction["source_event_index"][event]),
                "number_of_nodes": n_nodes,
                "node_types": node_types,
                "node_p4": node_p4,
                "node_ids": node_ids,
                "node_truth_ids": node_truth,
                "edge_reconstruction_active": bool(prediction["edge_reco_active"][event]),
                "hyperedge_reconstruction_active": bool(prediction["hyperedge_reco_active"][event]),
            }
            if prediction["hyperedge_logits"] is not None and prediction["edge_logits"] is not None:
                hyper_logits = prediction["hyperedge_logits"][event].detach().cpu()
                hyper_probs = torch.softmax(hyper_logits, dim=1)
                physical_pairs, edge_logits = symmetrise_directed_edge_logits(
                    prediction["edge_logits"][event].detach().cpu(), n_nodes
                )
                edge_probs = torch.softmax(edge_logits, dim=1)
                hyperedges = list(combinations(range(n_nodes), int(hyperedge_order)))
                if hyper_logits.size(0) != len(hyperedges):
                    raise ValueError("Hyperedge logits do not align with unordered hyperedge combinations.")
                row.update({
                    "HyPER_HE_IDX": [list(value) for value in hyperedges],
                    "HyPER_GE_IDX": [list(value) for value in physical_pairs],
                    "HyPER_HE_VCT": [[node_types[index] for index in value] for value in hyperedges],
                    "HyPER_GE_VCT": [[node_types[index] for index in value] for value in physical_pairs],
                    "HyPER_HE_LOGITS": hyper_logits.tolist(),
                    "HyPER_GE_LOGITS": edge_logits.tolist(),
                    "HyPER_HE_CLASS_PROBS": hyper_probs.tolist(),
                    "HyPER_GE_CLASS_PROBS": edge_probs.tolist(),
                    "HyPER_HE_CLASS_NAMES": list(hyperedge_names),
                    "HyPER_GE_CLASS_NAMES": list(edge_names),
                    "HyPER_HE_RAW": _nonbackground_score(hyper_probs),
                    "HyPER_GE_RAW": _nonbackground_score(edge_probs),
                })
            if cls_logit is not None:
                row["HyPER_CLS_LOGIT"] = float(cls_logit[event])
                row["HyPER_CLS_PROB"] = float(cls_prob[event])
            if cls_target is not None:
                row["HyPER_CLS_T"] = int(cls_target[event])
            rows.append(row)
    return pd.DataFrame(rows)


def _role_truth(node_truth_ids, node_types, topology_name: str) -> dict:
    if topology_name == "ttbar1L":
        mapping = {"b_lep": 1, "b_had": 2, "W_j1": 3, "W_j2": 4}
    elif topology_name == "ttH":
        mapping = {"b_had": 1, "W_j1": 2, "W_j2": 3, "b_lep": 4, "H_b1": 5, "H_b2": 6}
    else:
        raise ValueError(f"Unsupported topology {topology_name!r}.")
    output = {}
    matched = 0
    for role, truth_id in mapping.items():
        matches = [
            index for index, (value, node_type) in enumerate(zip(node_truth_ids, node_types))
            if int(node_type) == 1 and int(value) == truth_id
        ]
        index = matches[0] if len(matches) == 1 else -1
        output[f"truth_{role}_local_index"] = int(index)
        output[f"truth_{role}_valid"] = bool(index >= 0)
        matched += int(index >= 0)
    output["truth_fully_matched"] = matched == len(mapping)
    output["truth_partially_matched"] = 0 < matched < len(mapping)
    output["truth_unmatched"] = matched == 0
    return output


def _selected_fields(row: pd.Series, topology_name: str) -> tuple[list[int], list[float]]:
    if topology_name == "ttbar1L":
        index_fields = ("HyPER_best_top1", "HyPER_best_top2", "HyPER_best_w1", "HyPER_best_w2")
        score_fields = ("HyPER_best_top1_prob", "HyPER_best_top2_prob", "HyPER_best_w1_prob", "HyPER_best_w2_prob")
    else:
        index_fields = (
            "HyPER_best_tlep", "HyPER_best_thad", "HyPER_best_wlep", "HyPER_best_whad", "HyPER_best_higgs"
        )
        score_fields = (
            "HyPER_best_tlep_prob", "HyPER_best_thad_prob", "HyPER_best_wlep_prob",
            "HyPER_best_whad_prob", "HyPER_best_higgs_prob",
        )
    indices = []
    for field in index_fields:
        value = row.get(field, [])
        indices.extend(int(item) for item in (value if isinstance(value, (list, tuple)) else [value]))
    return indices, [float(row.get(field, 0.0)) for field in score_fields]


def _format_output(
    raw: pd.DataFrame, output_mode: str, topology_name: str, split: str,
    dataset_name: str, start: int, topology_strategy: str | None = None,
):
    mode = str(output_mode).lower()
    if mode not in {"selected", "raw", "both", "classifier"}:
        raise ValueError("Internal prediction product must be selected, raw, both, or classifier.")
    common = [
        "source_event_index", "number_of_nodes", "node_types", "node_p4", "node_ids",
        "node_truth_ids",
        "edge_reconstruction_active", "hyperedge_reconstruction_active",
    ]
    for optional in ("HyPER_CLS_LOGIT", "HyPER_CLS_PROB", "HyPER_CLS_T"):
        if optional in raw.columns:
            common.append(optional)
    if mode == "classifier":
        if "HyPER_CLS_LOGIT" not in raw.columns:
            raise RuntimeError("Classifier output requested from a model without a classification head.")
        output = raw[common].copy()
    elif mode == "raw":
        output = raw.copy()
    else:
        topology = TOPOLOGY_REGISTRY[topology_name]
        topology_kwargs = {"classification": "HyPER_CLS_PROB" in raw.columns}
        if topology_name == "ttH" and topology_strategy is not None:
            topology_kwargs["strategy"] = topology_strategy
        selected = topology(raw, **topology_kwargs)
        selected = selected.drop(columns=[column for column in common if column in selected.columns])
        output = pd.concat([raw[common].reset_index(drop=True), selected.reset_index(drop=True)], axis=1)
        if mode == "both":
            extras = raw.drop(columns=[column for column in common if column in raw.columns])
            output = pd.concat([output, extras.reset_index(drop=True)], axis=1)
        selected_indices, selected_scores = [], []
        for _, row in output.iterrows():
            indices, scores = _selected_fields(row, topology_name)
            selected_indices.append(indices)
            selected_scores.append(scores)
        output["selected_reconstruction_local_indices"] = selected_indices
        output["selected_reconstruction_scores"] = selected_scores

    truth_rows = [
        _role_truth(truth_ids, node_types, topology_name)
        for truth_ids, node_types in zip(output["node_truth_ids"], output["node_types"])
    ]
    output = pd.concat([output.reset_index(drop=True), pd.DataFrame(truth_rows)], axis=1)
    output.insert(0, "prediction_row_index", range(start, start + len(output)))
    output.insert(1, "prediction_split", split)
    output.insert(2, "source_dataset_name", dataset_name)
    if output["source_event_index"].duplicated().any():
        raise RuntimeError("Prediction output contains duplicate source_event_index values.")
    return output


class ChunkedPickleWriter:
    def __init__(self, path: Path, overwrite: bool):
        self.path = path
        if path.exists():
            if not overwrite:
                raise FileExistsError(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        path.mkdir(parents=True)
        self.parts, self.rows = [], 0

    def write(self, frame: pd.DataFrame):
        filename = f"part_{len(self.parts):06d}.pkl"
        temporary = self.path / f"{filename}.tmp"
        frame.to_pickle(temporary)
        os.replace(temporary, self.path / filename)
        self.parts.append(filename)
        self.rows += len(frame)


def _manifest(
    cfg,
    checkpoint: Path,
    datamodule,
    output: Path,
    rows: int,
    topology: str,
    task: TaskSpec,
) -> dict:
    configured_source = cfg.dataset.get("source_h5_path")
    source_h5 = (
        Path(str(configured_source)).expanduser().resolve()
        if configured_source is not None and str(configured_source).strip()
        else (Path(str(cfg.dataset.root)) / "raw" / f"{cfg.dataset.predict_set}.h5").resolve()
    )
    index_file = cfg.predicting.source_indices_file
    index_hash = None
    if index_file is not None:
        indices = np.load(str(index_file), allow_pickle=False).astype(np.int64, copy=False)
        index_hash = hashlib.sha256(indices.tobytes()).hexdigest()[:16]
    manifest = {
        **checkpoint_metadata(checkpoint),
        "source_h5_path": str(source_h5),
        "prediction_split": str(datamodule.resolved_predict_split),
        "source_indices_file": None if index_file is None else str(Path(str(index_file)).resolve()),
        "source_indices_hash": index_hash,
        "number_of_prediction_rows": int(rows),
        "classification_score_representation": (
            "HyPER_CLS_LOGIT and HyPER_CLS_PROB" if task.classification_enabled else None
        ),
        "reconstruction_score_representation": (
            "reverse-directed logits averaged per physical pair, then softmax"
            if task.reconstruction_enabled else None
        ),
        "task": task.mode,
        "topology": topology,
        "prediction_product": task.prediction_product,
        "split_cache_path": datamodule.split_cache_path,
        "split_metadata": datamodule.split_metadata,
        "target_encoding": "typed",
        "edge_class_names": datamodule.edge_class_names,
        "hyperedge_class_names": datamodule.hyperedge_class_names,
        "prediction_output": str(output.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return manifest


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Predict(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    topology_name, task = validate_runtime_config(cfg)
    checkpoint = resolve_checkpoint(
        cfg.predicting.checkpoint,
        cfg.predicting.model_directory,
        purpose="Prediction checkpoint",
    )
    output = cfg.predicting.save_as
    if output is None or not str(output).strip():
        raise ValueError("predicting.save_as must be supplied explicitly.")
    output = Path(str(output)).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    overwrite = bool(cfg.predicting.overwrite)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if output.suffix not in {".pkl", ".parts"} and not str(output).endswith(".pkl.parts"):
        raise ValueError("Production prediction output must be .pkl or .pkl.parts; H5 output was removed.")

    accelerator = str(cfg.predicting.accelerator).lower()
    device = torch.device("cuda:0" if accelerator in {"gpu", "cuda"} and torch.cuda.is_available() else "cpu")
    datamodule = HyPERDataModule(
        root=str(cfg.dataset.root),
        train_set=str(cfg.dataset.train_set) if cfg.predicting.split is not None else None,
        predict_set=str(cfg.dataset.predict_set),
        batch_size=int(cfg.predicting.batch_size),
        drop_last=False,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=bool(cfg.dataset.pin_memory) and device.type == "cuda",
        persistent_workers=bool(cfg.dataset.persistent_workers),
        prefetch_factor=int(cfg.dataset.prefetch_factor),
        graph_config=graph_config(cfg),
        split_config=plain(cfg.dataset.split),
        predict_split=cfg.predicting.split,
        source_indices_file=cfg.predicting.source_indices_file,
        source_h5_path=cfg.dataset.get("source_h5_path"),
        require_two_event_classes=task.classification_enabled,
        seed=int(cfg.general.seed),
    )
    datamodule.setup("predict")
    model = HyPERModel.load_from_checkpoint(str(checkpoint), map_location=device).to(device).eval()
    if model.edge_class_names != datamodule.edge_class_names or model.hyperedge_class_names != datamodule.hyperedge_class_names:
        raise RuntimeError("Checkpoint reconstruction class names disagree with the prediction dataset.")

    if topology_name not in TOPOLOGY_REGISTRY:
        raise ValueError(f"Unsupported topology {topology_name!r}.")
    maximum = cfg.predicting.max_events
    maximum = None if maximum is None else int(maximum)
    chunk_size = int(cfg.predicting.chunk_size_events)
    frames, pending, pending_events, written = [], [], 0, 0
    writer = ChunkedPickleWriter(output, overwrite) if str(output).endswith(".pkl.parts") else None

    def flush():
        nonlocal pending, pending_events, written
        if not pending:
            return
        raw = _raw_frame(pending, int(cfg.model.hyperedge_order), datamodule.edge_class_names, datamodule.hyperedge_class_names)
        formatted = _format_output(
            raw, ("both" if bool(cfg.predicting.get("include_raw", False)) else task.prediction_product), topology_name,
            str(datamodule.resolved_predict_split), str(cfg.dataset.predict_set), written,
            cfg.predicting.get("strategy"),
        )
        (writer.write(formatted) if writer is not None else frames.append(formatted))
        written += len(formatted)
        pending, pending_events = [], 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(datamodule.predict_dataloader()):
            if maximum is not None and written + pending_events >= maximum:
                break
            prediction = model.predict_step(batch.to(device), batch_index)
            batch_events = len(prediction["node_counts"])
            if maximum is not None and written + pending_events + batch_events > maximum:
                keep = maximum - written - pending_events
                for key, value in prediction.items():
                    if isinstance(value, list):
                        prediction[key] = value[:keep]
                    elif value is not None and hasattr(value, "__len__"):
                        prediction[key] = value[:keep]
                batch_events = keep
            pending.append(prediction)
            pending_events += batch_events
            if pending_events >= chunk_size:
                flush()
            if cfg.predicting.memory_log_every_batches and (batch_index + 1) % int(cfg.predicting.memory_log_every_batches) == 0:
                print(f"[predict] batches={batch_index + 1} rows={written + pending_events} rss_mb={_rss_mb():.1f}")
        flush()

    if writer is None:
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        temporary = output.with_name(output.name + ".tmp")
        result.to_pickle(temporary)
        os.replace(temporary, output)
    manifest = _manifest(cfg, checkpoint, datamodule, output, written, topology_name, task)
    if writer is not None:
        manifest.update({"format": "chunked_pickle_parts", "part_files": writer.parts})
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    Path(str(output) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {written} prediction rows to {output}")


if __name__ == "__main__":
    Predict()
