import json
import os
import shutil
import resource
import socket
import subprocess
import hydra
import torch
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import lightning.pytorch as pl

from tqdm import tqdm
from itertools import combinations
from omegaconf import DictConfig, OmegaConf

from HyPER.data import HyPERDataModule
from HyPER.models import HyPERModel
from HyPER.utils import ResultWriter
from HyPER.topology import *


TOPOLOGY_REGISTRY = {
    "ttbar_single_lep": ttbar_single_lep,
    "ttbar_singlelep": ttbar_single_lep,
    "ttbar_sl": ttbar_single_lep,
    "tth": ttH_single_lep,
    "ttH": ttH_single_lep,
    "tth_single_lep": ttH_single_lep,
    "tth_singlelep": ttH_single_lep,
    "ttH_single_lep": ttH_single_lep,
    "ttbar_dilep": ttbar_dilep,
    "ttbar_dilepton": ttbar_dilep,
    "ttbar_allhad": ttbar_allhad,
    "ttbar_allhadronic": ttbar_allhad,
    "fourtop_allhad": FourTop_allhad,
    "fourtop_allhadronic": FourTop_allhad,
}


def _select(cfg: DictConfig, path: str, legacy: str = None, default=None):
    value = OmegaConf.select(cfg, path, default=None)
    if value is not None:
        return value
    if legacy is not None:
        value = OmegaConf.select(cfg, legacy, default=None)
        if value is not None:
            return value
    return default


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalise_split_name(value):
    if value is None:
        return None
    name = str(value).strip().lower()
    if name in {"", "none", "null", "external"}:
        return None
    if name in {"validation", "valid"}:
        return "val"
    if name not in {"train", "val", "test"}:
        raise ValueError("predicting.split / dataset.split.predict_split must be one of train, val, test, external, null.")
    return name


def _as_plain_container(value, default=None):
    if value is None:
        return {} if default is None else default
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _graph_config_from_cfg(cfg: DictConfig):
    if OmegaConf.select(cfg, 'input', default=None) is None and OmegaConf.select(cfg, 'target', default=None) is None:
        return None
    if OmegaConf.select(cfg, 'input', default=None) is None or OmegaConf.select(cfg, 'target', default=None) is None:
        raise ValueError("Unified HyPER configs must provide both `input` and `target` sections.")
    return OmegaConf.to_container(
        OmegaConf.create({
            'input': OmegaConf.select(cfg, 'input'),
            'target': OmegaConf.select(cfg, 'target'),
        }),
        resolve=True,
    )


def _topology_func(name: str):
    if name is None:
        raise ValueError("No topology configured.")
    key = str(name).strip()
    normalized = key.lower().replace("-", "_")
    if key in TOPOLOGY_REGISTRY:
        return TOPOLOGY_REGISTRY[key]
    if normalized in TOPOLOGY_REGISTRY:
        return TOPOLOGY_REGISTRY[normalized]
    raise ValueError(
        f"Unsupported topology {name!r}. Available topologies: "
        + ", ".join(sorted(TOPOLOGY_REGISTRY))
    )


class ChunkedPickleWriter:
    """
    Writes output as:
      output.pkl.parts/
        part_000000.pkl
        part_000001.pkl
        ...
        manifest.json

    This avoids creating one huge final pickle in memory.
    """

    def __init__(self, output_path: str, overwrite: bool = True, metadata: dict | None = None):
        self.output_path = output_path
        if str(output_path).endswith(".pkl.parts"):
            self.parts_dir = output_path
        else:
            self.parts_dir = output_path + ".parts"
        self.overwrite = overwrite
        self.part_files = []
        self.n_parts = 0
        self.n_events = 0
        self.metadata = dict(metadata or {})

        if os.path.exists(self.parts_dir):
            if not overwrite:
                raise FileExistsError(f"Output parts directory already exists: {self.parts_dir}")
            shutil.rmtree(self.parts_dir)

        os.makedirs(self.parts_dir, exist_ok=True)

    def write(self, df: pd.DataFrame):
        if len(df) == 0:
            return

        part_name = f"part_{self.n_parts:06d}.pkl"
        part_path = os.path.join(self.parts_dir, part_name)
        tmp_path = part_path + ".tmp"

        df.to_pickle(tmp_path)
        os.replace(tmp_path, part_path)

        self.part_files.append(part_name)
        self.n_parts += 1
        self.n_events += len(df)

        print(f"Wrote {part_path} with {len(df)} events; total events written = {self.n_events}")

    def close(self):
        manifest = {
            "format": "chunked_pickle_parts",
            "requested_output": self.output_path,
            "parts_dir": self.parts_dir,
            "part_files": self.part_files,
            "n_parts": self.n_parts,
            "n_events": self.n_events,
        }
        manifest.update(self.metadata)

        manifest_path = os.path.join(self.parts_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print("================================")
        print("Chunked output complete")
        print(f"Requested output: {self.output_path}")
        print(f"Parts directory:   {self.parts_dir}")
        print(f"Number of parts:   {self.n_parts}")
        print(f"Number of events:  {self.n_events}")
        print(f"Manifest:          {manifest_path}")
        print("================================")
        return manifest


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _prediction_metadata(
    cfg: DictConfig,
    ckpt_file: str,
    predict_output: str,
    output_mode: str,
    topology_name: str | None,
    chunk_size_events,
    n_events=None,
    datamodule=None,
    predict_split: str | None = None,
    max_events=None,
) -> dict:
    dataset_root = str(_select(cfg, "dataset.root", "dataset", ""))
    train_set = str(_select(cfg, "dataset.train_set", "train_set", ""))
    predict_set = str(_select(cfg, "dataset.predict_set", "predict_set", ""))
    source_h5 = None
    if datamodule is not None and getattr(datamodule, "split_metadata", None):
        source_h5 = datamodule.split_metadata.get("source_h5_path")
    if source_h5 is None and dataset_root and predict_set:
        source_h5 = str(Path(dataset_root) / "raw" / f"{predict_set}.h5")
    return {
        "config_name": str(OmegaConf.select(cfg, "hydra.job.config_name", default="unknown")),
        "model_dir": str(_select(cfg, "predicting.model_directory", "predict_model", "")),
        "checkpoint": str(ckpt_file),
        "source_h5": source_h5,
        "dataset_root": dataset_root,
        "train_set": train_set,
        "predict_set": predict_set,
        "predict_split": None if predict_split is None else str(predict_split),
        "predicting.split": None if predict_split is None else str(predict_split),
        "dataset.split.predict_split": str(_select(cfg, "dataset.split.predict_split", default="")),
        "topology": None if topology_name is None else str(topology_name),
        "output_mode": str(output_mode),
        "prediction_output_mode": str(output_mode),
        "prediction_output": str(predict_output),
        "predicting.max_events": None if max_events is None else max_events,
        "dataset.max_n_events": _select(cfg, "dataset.max_n_events", "max_n_events", None),
        "split_cache_path": None if datamodule is None else getattr(datamodule, "split_cache_path", None),
        "n_events": None if n_events is None else int(n_events),
        "n_prediction_rows": None if n_events is None else int(n_events),
        "has_HYPER_SOURCE_INDEX": None,
        "min_HYPER_SOURCE_INDEX": None,
        "max_HYPER_SOURCE_INDEX": None,
        "chunk_size_events": None if chunk_size_events is None else int(chunk_size_events),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "hostname": socket.gethostname(),
    }


def _write_prediction_manifest(predict_output: str, metadata: dict, n_events: int | None = None) -> None:
    manifest = dict(metadata)
    if n_events is not None:
        manifest["n_events"] = int(n_events)
        manifest["n_prediction_rows"] = int(n_events)
    output_path = os.path.abspath(str(predict_output))
    if output_path.endswith(".pkl.parts"):
        manifest_path = os.path.join(output_path, "prediction_manifest.json")
    elif output_path.endswith(".parts") and os.path.isdir(output_path):
        manifest_path = os.path.join(output_path, "prediction_manifest.json")
    else:
        manifest_path = os.path.join(os.path.dirname(output_path) or ".", "prediction_manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote prediction manifest: {manifest_path}")


def _update_source_index_metadata(metadata: dict, frame: pd.DataFrame) -> None:
    if "HYPER_SOURCE_INDEX" not in frame.columns or len(frame) == 0:
        if metadata.get("has_HYPER_SOURCE_INDEX") is None:
            metadata["has_HYPER_SOURCE_INDEX"] = False
        return
    values = pd.to_numeric(frame["HYPER_SOURCE_INDEX"], errors="coerce")
    metadata["has_HYPER_SOURCE_INDEX"] = True
    if values.notna().any():
        current_min = int(values.min())
        current_max = int(values.max())
        old_min = metadata.get("min_HYPER_SOURCE_INDEX")
        old_max = metadata.get("max_HYPER_SOURCE_INDEX")
        metadata["min_HYPER_SOURCE_INDEX"] = current_min if old_min is None else min(int(old_min), current_min)
        metadata["max_HYPER_SOURCE_INDEX"] = current_max if old_max is None else max(int(old_max), current_max)


def _memory_rss_mb() -> float:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 10_000_000:
            return float(rss) / (1024.0 * 1024.0)
        return float(rss) / 1024.0
    except Exception:
        return float("nan")


def _resolve_torch_device(accelerator, devices=1):
    accelerator = str(accelerator).lower()
    if accelerator in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            warnings.warn("GPU accelerator requested but CUDA is unavailable; using CPU.", UserWarning)
            return torch.device("cpu")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _build_raw_chunk_from_batches(
    prediction_batches,
    hyperedge_order: int,
    desc: str = "Evaluating prediction chunk",
) -> pd.DataFrame:
    hyperedge_out = []
    graphedge_out = []
    cls_scores = []
    hyperedge_vct = []
    graphedge_vct = []
    hyperedges = []
    graphedges = []
    hyperedge_probs = []
    graphedge_probs = []
    cls_targets = []
    source_event_indices = []

    have_cls = False
    have_typed_probs = False
    have_cls_targets = False
    have_source_event_index = False

    for prediction in tqdm(
        prediction_batches,
        desc=desc,
        unit="batch",
    ):
        if len(prediction) == 5:
            x_out, edge_attr_out, N_nodes, encodings, x_class_out = prediction
            x_probs_out = None
            edge_probs_out = None
            cls_t_out = None
            source_event_index_out = None
        elif len(prediction) == 7:
            x_out, edge_attr_out, N_nodes, encodings, x_class_out, x_probs_out, edge_probs_out = prediction
            cls_t_out = None
            source_event_index_out = None
        elif len(prediction) == 8:
            x_out, edge_attr_out, N_nodes, encodings, x_class_out, x_probs_out, edge_probs_out, cls_t_out = prediction
            source_event_index_out = None
        else:
            x_out, edge_attr_out, N_nodes, encodings, x_class_out, x_probs_out, edge_probs_out, cls_t_out, source_event_index_out = prediction

        for j in range(len(x_out)):
            n_nodes = int(N_nodes[j])
            enc = encodings[j].cpu().flatten().tolist()

            hyperedges.append([
                list(x) for x in combinations(range(n_nodes), r=hyperedge_order)
            ])
            hyperedge_out.append(
                x_out[j].cpu().flatten().tolist()
            )
            hyperedge_vct.append([
                list(x) for x in combinations(enc, r=hyperedge_order)
            ])

            graphedges.append([
                list(x) for x in combinations(range(n_nodes), r=2)
            ])
            graphedge_out.append(
                edge_attr_out[j].cpu().flatten().tolist()
            )
            graphedge_vct.append([
                list(x) for x in combinations(enc, r=2)
            ])

        if x_class_out is not None:
            have_cls = True
            cls_scores.extend(x_class_out.cpu().flatten().tolist())

        if cls_t_out is not None:
            have_cls_targets = True
            cls_targets.extend(cls_t_out.cpu().flatten().tolist())

        if source_event_index_out is not None:
            have_source_event_index = True
            source_event_indices.extend(source_event_index_out.cpu().flatten().tolist())

        if x_probs_out is not None and edge_probs_out is not None:
            have_typed_probs = True
            for j in range(len(x_probs_out)):
                hyperedge_probs.append(x_probs_out[j].cpu().tolist())
                graphedge_probs.append(edge_probs_out[j].cpu().tolist())

    output_columns = {
        "HyPER_HE_RAW": hyperedge_out,
        "HyPER_GE_RAW": graphedge_out,
        "HyPER_HE_VCT": hyperedge_vct,
        "HyPER_GE_VCT": graphedge_vct,
        "HyPER_HE_IDX": hyperedges,
        "HyPER_GE_IDX": graphedges,
    }

    if have_cls:
        output_columns["HyPER_CLS_RAW"] = cls_scores

    if have_cls_targets:
        output_columns["HyPER_CLS_T"] = cls_targets

    if have_source_event_index:
        output_columns["HYPER_SOURCE_INDEX"] = source_event_indices
        output_columns["source_event_index"] = source_event_indices

    if have_typed_probs:
        output_columns["HyPER_HE_CLASS_PROBS"] = hyperedge_probs
        output_columns["HyPER_GE_CLASS_PROBS"] = graphedge_probs

    return pd.DataFrame(output_columns)


def _build_classifier_chunk(raw_chunk: pd.DataFrame) -> pd.DataFrame:
    if "HyPER_CLS_RAW" not in raw_chunk.columns:
        raise RuntimeError(
            "predicting.output_mode=classifier requires a checkpoint with an "
            "event-level classification head."
        )
    columns = ["HyPER_CLS_RAW"]
    if "HYPER_SOURCE_INDEX" in raw_chunk.columns:
        columns.insert(0, "HYPER_SOURCE_INDEX")
    if "source_event_index" in raw_chunk.columns:
        columns.insert(1 if "HYPER_SOURCE_INDEX" in columns else 0, "source_event_index")
    if "HyPER_CLS_T" in raw_chunk.columns:
        columns.append("HyPER_CLS_T")
    return raw_chunk.loc[:, columns].copy()


def _build_raw_chunk_from_out(
    out,
    batch_start: int,
    batch_stop: int,
    hyperedge_order: int,
) -> pd.DataFrame:
    return _build_raw_chunk_from_batches(
        out[batch_start:batch_stop],
        hyperedge_order=hyperedge_order,
        desc=f"Evaluating batches {batch_start}:{batch_stop}",
    )


def _write_h5_single_chunk_or_warn(results: pd.DataFrame, predict_output: str):
    warnings.warn(
        "Saving `.h5` still uses ResultWriter and may not be chunked. "
        "For large prediction jobs, prefer `.pkl` so chunked parts are used.",
        UserWarning,
    )
    ResultWriter(results, predict_output)


def _prediction_batch_events(batch) -> int:
    x_out = batch[0]
    return int(len(x_out))


def _iter_prediction_chunks(out, chunk_size=None, chunk_batches=None):
    if chunk_size is not None:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("predicting.chunk_size / predict_chunk_size must be positive.")
        batch_start = 0
        events_in_chunk = 0
        for batch_idx, batch in enumerate(out):
            events_in_chunk += _prediction_batch_events(batch)
            if events_in_chunk >= chunk_size:
                yield batch_start, batch_idx + 1
                batch_start = batch_idx + 1
                events_in_chunk = 0
        if batch_start < len(out):
            yield batch_start, len(out)
        return

    if chunk_batches is not None:
        chunk_batches = int(chunk_batches)
        if chunk_batches <= 0:
            raise ValueError("predicting.chunk_batches must be positive.")
        for batch_start in range(0, len(out), chunk_batches):
            yield batch_start, min(batch_start + chunk_batches, len(out))
        return

    yield 0, len(out)


def _build_reco_chunks(
    out,
    hyperedge_order: int,
    topology,
    classification_enabled: bool,
    chunk_size=None,
    chunk_batches=None,
    include_event_index: bool = False,
):
    global_event_index = 0
    for batch_start, batch_stop in _iter_prediction_chunks(
        out,
        chunk_size=chunk_size,
        chunk_batches=chunk_batches,
    ):
        raw_chunk = _build_raw_chunk_from_out(
            out=out,
            batch_start=batch_start,
            batch_stop=batch_stop,
            hyperedge_order=hyperedge_order,
        )

        results_chunk = topology(
            raw_chunk,
            classification=classification_enabled,
        )

        if include_event_index:
            results_chunk.insert(
                0,
                "event_index",
                range(global_event_index, global_event_index + len(results_chunk)),
            )
        global_event_index += len(results_chunk)

        yield results_chunk

        del raw_chunk


def _format_output_chunk(
    raw_chunk: pd.DataFrame,
    topology,
    classification_enabled: bool,
    output_mode: str,
    global_event_index: int,
    predict_split_label: str,
) -> pd.DataFrame:
    output_mode = str(output_mode).lower()
    if output_mode not in {"selected", "raw", "both", "classifier"}:
        raise ValueError("predicting.output_mode must be one of: selected, raw, both, classifier")

    if output_mode == "classifier":
        output = _build_classifier_chunk(raw_chunk)
    elif output_mode == "raw":
        output = raw_chunk.copy()
    else:
        if topology is None:
            raise ValueError("A topology is required unless predicting.output_mode=classifier.")
        selected = topology(raw_chunk, classification=classification_enabled)
        if "HYPER_SOURCE_INDEX" in raw_chunk.columns and "HYPER_SOURCE_INDEX" not in selected.columns:
            selected.insert(0, "HYPER_SOURCE_INDEX", raw_chunk["HYPER_SOURCE_INDEX"].to_numpy())
        if "source_event_index" in raw_chunk.columns and "source_event_index" not in selected.columns:
            insert_at = 1 if "HYPER_SOURCE_INDEX" in selected.columns else 0
            selected.insert(insert_at, "source_event_index", raw_chunk["source_event_index"].to_numpy())
        if classification_enabled and "HyPER_CLS_T" in raw_chunk.columns and "HyPER_CLS_T" not in selected.columns:
            selected["HyPER_CLS_T"] = raw_chunk["HyPER_CLS_T"].to_numpy()
        if output_mode == "selected":
            output = selected
        else:
            raw_extra = raw_chunk.drop(columns=[c for c in raw_chunk.columns if c in selected.columns])
            output = pd.concat(
                [
                    selected.reset_index(drop=True),
                    raw_extra.reset_index(drop=True),
                ],
                axis=1,
            )

    if "event_index" in output.columns:
        output = output.drop(columns=["event_index"])
    if "HYPER_PREDICT_ORDER" in output.columns:
        output = output.drop(columns=["HYPER_PREDICT_ORDER"])
    if "HYPER_PREDICT_SPLIT" in output.columns:
        output = output.drop(columns=["HYPER_PREDICT_SPLIT"])
    output.insert(0, "event_index", range(global_event_index, global_event_index + len(output)))
    output.insert(1, "HYPER_PREDICT_ORDER", range(global_event_index, global_event_index + len(output)))
    output.insert(2, "HYPER_PREDICT_SPLIT", str(predict_split_label))
    if "HYPER_SOURCE_INDEX" not in output.columns:
        if "source_event_index" not in output.columns:
            raise RuntimeError(
                "Prediction output is missing source-event metadata. Refusing to "
                "write output without HYPER_SOURCE_INDEX; split/subset plots would "
                "not be self-aligning."
            )
        output.insert(3, "HYPER_SOURCE_INDEX", output["source_event_index"].astype("int64").to_numpy())
    if "source_event_index" not in output.columns:
        output.insert(4, "source_event_index", output["HYPER_SOURCE_INDEX"].astype("int64").to_numpy())
    return output


def _stream_prediction_chunks(
    model,
    datamodule,
    device,
    hyperedge_order: int,
    topology,
    classification_enabled: bool,
    chunk_size_events: int | None,
    chunk_batches: int | None,
    output_mode: str,
    memory_log_every_batches: int,
    max_events: int | None = None,
):
    if getattr(datamodule, "predict_data", None) is None:
        datamodule.setup("predict")
    loader = datamodule.predict_dataloader()
    model.to(device)
    model.eval()

    pending_batches = []
    pending_events = 0
    processed_batches = 0
    processed_events = 0
    global_event_index = 0
    predict_split_label = str(getattr(datamodule, "resolved_predict_split", None) or "all")

    def flush_pending():
        nonlocal pending_batches, pending_events, global_event_index
        if not pending_batches:
            return None
        raw_chunk = _build_raw_chunk_from_batches(
            pending_batches,
            hyperedge_order=hyperedge_order,
            desc=f"Post-processing {len(pending_batches)} streamed batches",
        )
        if getattr(datamodule, "target_encoding", "binary") == "typed":
            edge_names = list(getattr(datamodule, "edge_target_names", [])) + ["background"]
            hyperedge_names = list(getattr(datamodule, "hyperedge_target_names", [])) + ["background"]
            raw_chunk["HyPER_GE_CLASS_NAMES"] = [edge_names for _ in range(len(raw_chunk))]
            raw_chunk["HyPER_HE_CLASS_NAMES"] = [hyperedge_names for _ in range(len(raw_chunk))]
        output_chunk = _format_output_chunk(
            raw_chunk=raw_chunk,
            topology=topology,
            classification_enabled=classification_enabled,
            output_mode=output_mode,
            global_event_index=global_event_index,
            predict_split_label=predict_split_label,
        )
        global_event_index += len(output_chunk)
        pending_batches = []
        pending_events = 0
        del raw_chunk
        return output_chunk

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_events is not None and processed_events >= int(max_events):
                break
            batch = batch.to(device)
            prediction = model.predict_step(batch, batch_idx)
            if max_events is not None and processed_events + _prediction_batch_events(prediction) > int(max_events):
                keep = int(max_events) - processed_events
                if len(prediction) == 5:
                    x_out, edge_attr_out, N_nodes, encodings, x_class_out = prediction
                    prediction = (
                        x_out[:keep],
                        edge_attr_out[:keep],
                        N_nodes[:keep],
                        encodings[:keep],
                        x_class_out[:keep] if x_class_out is not None else None,
                    )
                elif len(prediction) == 7:
                    x_out, edge_attr_out, N_nodes, encodings, x_class_out, x_probs_out, edge_probs_out = prediction
                    prediction = (
                        x_out[:keep],
                        edge_attr_out[:keep],
                        N_nodes[:keep],
                        encodings[:keep],
                        x_class_out[:keep] if x_class_out is not None else None,
                        x_probs_out[:keep] if x_probs_out is not None else None,
                        edge_probs_out[:keep] if edge_probs_out is not None else None,
                    )
                elif len(prediction) == 8:
                    x_out, edge_attr_out, N_nodes, encodings, x_class_out, x_probs_out, edge_probs_out, cls_t_out = prediction
                    prediction = (
                        x_out[:keep],
                        edge_attr_out[:keep],
                        N_nodes[:keep],
                        encodings[:keep],
                        x_class_out[:keep] if x_class_out is not None else None,
                        x_probs_out[:keep] if x_probs_out is not None else None,
                        edge_probs_out[:keep] if edge_probs_out is not None else None,
                        cls_t_out[:keep] if cls_t_out is not None else None,
                    )
                else:
                    x_out, edge_attr_out, N_nodes, encodings, x_class_out, x_probs_out, edge_probs_out, cls_t_out, source_event_index_out = prediction
                    prediction = (
                        x_out[:keep],
                        edge_attr_out[:keep],
                        N_nodes[:keep],
                        encodings[:keep],
                        x_class_out[:keep] if x_class_out is not None else None,
                        x_probs_out[:keep] if x_probs_out is not None else None,
                        edge_probs_out[:keep] if edge_probs_out is not None else None,
                        cls_t_out[:keep] if cls_t_out is not None else None,
                        source_event_index_out[:keep] if source_event_index_out is not None else None,
                    )
            pending_batches.append(prediction)
            n_events = _prediction_batch_events(prediction)
            pending_events += n_events
            processed_batches += 1
            processed_events += n_events

            should_flush = False
            if chunk_size_events is not None and pending_events >= int(chunk_size_events):
                should_flush = True
            if chunk_batches is not None and len(pending_batches) >= int(chunk_batches):
                should_flush = True

            if memory_log_every_batches and processed_batches % int(memory_log_every_batches) == 0:
                print(
                    f"[predict] batch={processed_batches} events={processed_events} "
                    f"pending_events={pending_events} rss_mb={_memory_rss_mb():.1f}",
                    flush=True,
                )

            if should_flush:
                output_chunk = flush_pending()
                if output_chunk is not None:
                    yield output_chunk

        output_chunk = flush_pending()
        if output_chunk is not None:
            yield output_chunk


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Predict(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = _select(
        cfg,
        'predicting.accelerator',
        'predict_with',
        _select(cfg, 'trainer.accelerator', 'device', 'cpu'),
    )

    batch_size = _select(cfg, 'predicting.batch_size', 'batch_size', 128)
    predict_split = _normalise_split_name(
        _select(cfg, "predicting.split", "dataset.split.predict_split", None)
    )

    # Keep old default behaviour unless explicitly overridden.
    pin_memory = _select(cfg, "dataset.pin_memory", default=None)
    if pin_memory is None:
        pin_memory = True if device == "gpu" else False
    else:
        pin_memory = _as_bool(pin_memory)
    if str(device).lower() == "cpu" and pin_memory:
        warnings.warn("Disabling `pin_memory` for CPU prediction.", UserWarning)
        pin_memory = False

    datamodule = HyPERDataModule(
        root=_select(cfg, 'dataset.root', 'dataset'),
        train_set=_select(cfg, 'dataset.train_set', 'train_set', None) if predict_split is not None else None,
        val_set=None,
        predict_set=_select(cfg, 'dataset.predict_set', 'predict_set'),
        batch_size=batch_size,
        max_n_events=_select(cfg, 'dataset.max_n_events', 'max_n_events', -1),
        percent_valid_samples=1 - float(_select(cfg, 'dataset.train_val_split', 'train_val_split', 0.95)),
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=_select(cfg, 'dataset.num_workers', 'datamodule.num_workers', 0),
        force_reload=_select(cfg, 'dataset.force_reload', 'datamodule.force_reload', False),
        use_ondisk=_select(cfg, 'dataset.use_ondisk', 'datamodule.use_ondisk', True),
        graph_config=_graph_config_from_cfg(cfg),
        split_config=_as_plain_container(_select(cfg, 'dataset.split', default={})),
        predict_split=predict_split,
    )

    map_location = torch.device('cuda') if str(device).lower() == "gpu" else torch.device('cpu')

    predict_model = _select(cfg, 'predicting.model_directory', 'predict_model', None)
    assert predict_model is not None, "No model directory is provided in `predict_model`/`predicting.model_directory`. Abort!"

    checkpoint_dir = os.path.join(predict_model, "checkpoints")
    ckpt_files = sorted(
        filename
        for filename in os.listdir(checkpoint_dir)
        if filename.startswith("epoch") and filename.endswith(".ckpt")
    )
    if len(ckpt_files) == 0 and os.path.exists(os.path.join(checkpoint_dir, "last.ckpt")):
        ckpt_files = ["last.ckpt"]

    if len(ckpt_files) > 1:
        warnings.warn(
            f"There are multiple .ckpt files listed in {predict_model}; using last sorted checkpoint: {ckpt_files[-1]}",
            UserWarning,
        )

    if len(ckpt_files) == 0:
        raise RuntimeError(f"No checkpoint files have been found in {predict_model}.")

    ckpt_file = os.path.join(checkpoint_dir, ckpt_files[-1])

    hparams_file = os.path.join(predict_model, "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.yaml` is not found in {predict_model}."

    model = HyPERModel.load_from_checkpoint(
        checkpoint_path=ckpt_file,
        hparams_file=hparams_file,
        map_location=map_location,
    )

    print("================================")
    print("Launching streaming prediction")
    print(f"Device:        {device}")
    print(f"Batch size:    {batch_size}")
    print(f"Pin memory:    {pin_memory}")
    print(f"Checkpoint:    {ckpt_file}")
    print("================================")

    output_mode = str(_select(cfg, "predicting.output_mode", default="selected")).lower()
    # ``classifier`` is an S/B-only prediction mode. It writes event-level
    # classifier scores and truth labels when available, and intentionally
    # bypasses topology reconstruction/post-processing.
    hyperedge_order = int(_select(cfg, 'model.hyperedge_order', 'hyperedge_order', 3))
    topology_name = _select(cfg, 'predicting.topology', 'topology')
    topology = None if output_mode == "classifier" else _topology_func(topology_name)

    requested_classification = _as_bool(
        _select(
            cfg,
            "classification.enabled",
            default=getattr(model, "classification_enabled", False),
        )
    )
    model_has_classification = bool(getattr(model, "classification_enabled", False))
    if requested_classification and not model_has_classification:
        warnings.warn(
            "classification.enabled is true in the prediction config, but the loaded "
            "checkpoint has no classification head. Disabling classification output "
            "for this prediction run.",
            UserWarning,
        )
    classification_enabled = requested_classification and model_has_classification
    if output_mode == "classifier" and not classification_enabled:
        raise RuntimeError(
            "predicting.output_mode=classifier requires classification.enabled=true "
            "and a checkpoint with a classification head."
        )

    predict_output = _select(cfg, 'predicting.save_as', 'predict_output', None)
    if predict_output is None:
        warnings.warn("No output path is provided in `predict_output`, using default: `output.pkl`.")
        predict_output = "output.pkl"

    chunk_size = _select(cfg, "predicting.chunk_size_events", "predict_chunk_size", None)
    if chunk_size is None:
        chunk_size = _select(cfg, "predicting.chunk_size", default=None)
    chunk_batches = _select(cfg, "predicting.chunk_batches", default=None)
    memory_log_every_batches = int(_select(cfg, "predicting.memory_log_every_batches", default=500) or 0)
    max_events = _select(cfg, "predicting.max_events", default=None)
    overwrite = _as_bool(_select(cfg, "predicting.overwrite", default=True))
    output_str = str(predict_output)
    force_chunked_parts = output_str.endswith(".pkl.parts")
    chunking_enabled = force_chunked_parts or chunk_size is not None or chunk_batches is not None
    if force_chunked_parts and chunk_size is None and chunk_batches is None:
        chunk_size = 100000

    print("================================")
    print("Starting post-processing")
    print(f"Topology:                {topology_name}")
    print(f"Classification enabled:  {classification_enabled}")
    print(f"Hyperedge order:         {hyperedge_order}")
    print(f"Output mode:             {output_mode}")
    print(f"Chunking enabled:        {chunking_enabled}")
    print(f"Chunk size events:       {chunk_size}")
    print(f"Chunk batches:           {chunk_batches}")
    print(f"Max events:              {max_events}")
    print(f"Output:                  {predict_output}")
    print("================================")
    datamodule.setup("predict")
    prediction_metadata = _prediction_metadata(
        cfg=cfg,
        ckpt_file=ckpt_file,
        predict_output=predict_output,
        output_mode=output_mode,
        topology_name=topology_name,
        chunk_size_events=chunk_size if chunking_enabled else None,
        datamodule=datamodule,
        predict_split=predict_split,
        max_events=max_events,
    )

    torch_device = _resolve_torch_device(device, _select(cfg, 'trainer.devices', 'num_devices', 1))
    prediction_chunks = _stream_prediction_chunks(
        model=model,
        datamodule=datamodule,
        device=torch_device,
        hyperedge_order=hyperedge_order,
        topology=topology,
        classification_enabled=classification_enabled,
        chunk_size_events=chunk_size if chunking_enabled else None,
        chunk_batches=chunk_batches if chunking_enabled else None,
        output_mode=output_mode,
        memory_log_every_batches=memory_log_every_batches,
        max_events=max_events,
    )

    if output_str.endswith(".pkl.parts"):
        writer = ChunkedPickleWriter(predict_output, overwrite=overwrite, metadata=prediction_metadata)

        for results_chunk in prediction_chunks:
            _update_source_index_metadata(writer.metadata, results_chunk)
            writer.write(results_chunk)
            print(f"[predict] wrote_part rows={len(results_chunk)} total_rows={writer.n_events}", flush=True)
            del results_chunk

        manifest = writer.close()
        _write_prediction_manifest(predict_output, manifest, n_events=writer.n_events)

    elif output_str.endswith(".pkl"):
        if chunking_enabled:
            warnings.warn(
                "Chunked `.pkl` output writes a `.parts` directory. Use an explicit "
                "`.pkl.parts` output path for production so event_index alignment is obvious.",
                UserWarning,
            )
            writer = ChunkedPickleWriter(predict_output, overwrite=overwrite, metadata=prediction_metadata)
            for results_chunk in prediction_chunks:
                _update_source_index_metadata(writer.metadata, results_chunk)
                writer.write(results_chunk)
                print(f"[predict] wrote_part rows={len(results_chunk)} total_rows={writer.n_events}", flush=True)
                del results_chunk
            manifest = writer.close()
            _write_prediction_manifest(writer.parts_dir, manifest, n_events=writer.n_events)
        else:
            chunks = list(prediction_chunks)
            results = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            _update_source_index_metadata(prediction_metadata, results)
            results.to_pickle(predict_output)
            _write_prediction_manifest(predict_output, prediction_metadata, n_events=len(results))
            print(f"Wrote {predict_output} with {len(results)} events")

    elif output_str.endswith(".h5"):
        warnings.warn(
            "Chunked `.h5` writing is not appendable here. This path post-processes "
            "chunks when requested, concatenates selected reco outputs, and writes "
            "once with ResultWriter. Use `.pkl.parts` for the memory-safe large path.",
            UserWarning,
        )

        chunks = list(prediction_chunks)
        results = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        _update_source_index_metadata(prediction_metadata, results)
        _write_h5_single_chunk_or_warn(results, predict_output)
        _write_prediction_manifest(predict_output, prediction_metadata, n_events=len(results))

    else:
        raise ValueError("You must provide a file extension: `.h5`, `.pkl`, or `.pkl.parts`.")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    Predict()
