import json
import os
import shutil
import resource
import hydra
import torch
import warnings

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

    def __init__(self, output_path: str, overwrite: bool = True):
        self.output_path = output_path
        if str(output_path).endswith(".pkl.parts"):
            self.parts_dir = output_path
        else:
            self.parts_dir = output_path + ".parts"
        self.overwrite = overwrite
        self.part_files = []
        self.n_parts = 0
        self.n_events = 0

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

    have_cls = False

    for x_out, edge_attr_out, N_nodes, encodings, x_class_out in tqdm(
        prediction_batches,
        desc=desc,
        unit="batch",
    ):

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

    return pd.DataFrame(output_columns)


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
) -> pd.DataFrame:
    output_mode = str(output_mode).lower()
    if output_mode not in {"selected", "raw", "both"}:
        raise ValueError("predicting.output_mode must be one of: selected, raw, both")

    if output_mode == "raw":
        output = raw_chunk.copy()
    else:
        selected = topology(raw_chunk, classification=classification_enabled)
        if output_mode == "selected":
            output = selected
        else:
            output = pd.concat(
                [
                    selected.reset_index(drop=True),
                    raw_chunk.reset_index(drop=True),
                ],
                axis=1,
            )

    output.insert(
        0,
        "event_index",
        range(global_event_index, global_event_index + len(output)),
    )
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
    datamodule.setup("predict")
    loader = datamodule.predict_dataloader()
    model.to(device)
    model.eval()

    pending_batches = []
    pending_events = 0
    processed_batches = 0
    processed_events = 0
    global_event_index = 0

    def flush_pending():
        nonlocal pending_batches, pending_events, global_event_index
        if not pending_batches:
            return None
        raw_chunk = _build_raw_chunk_from_batches(
            pending_batches,
            hyperedge_order=hyperedge_order,
            desc=f"Post-processing {len(pending_batches)} streamed batches",
        )
        output_chunk = _format_output_chunk(
            raw_chunk=raw_chunk,
            topology=topology,
            classification_enabled=classification_enabled,
            output_mode=output_mode,
            global_event_index=global_event_index,
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
                x_out, edge_attr_out, N_nodes, encodings, x_class_out = prediction
                prediction = (
                    x_out[:keep],
                    edge_attr_out[:keep],
                    N_nodes[:keep],
                    encodings[:keep],
                    x_class_out[:keep] if x_class_out is not None else None,
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
        train_set=None,
        val_set=None,
        predict_set=_select(cfg, 'dataset.predict_set', 'predict_set'),
        batch_size=batch_size,
        percent_valid_samples=1 - float(_select(cfg, 'dataset.train_val_split', 'train_val_split', 0.95)),
        pin_memory=pin_memory,
        drop_last=False,
        num_workers=_select(cfg, 'dataset.num_workers', 'datamodule.num_workers', 0),
        force_reload=_select(cfg, 'dataset.force_reload', 'datamodule.force_reload', False),
        use_ondisk=_select(cfg, 'dataset.use_ondisk', 'datamodule.use_ondisk', True),
        graph_config=_graph_config_from_cfg(cfg),
    )

    map_location = torch.device('cuda') if str(device).lower() == "gpu" else torch.device('cpu')

    predict_model = _select(cfg, 'predicting.model_directory', 'predict_model', None)
    assert predict_model is not None, "No model directory is provided in `predict_model`/`predicting.model_directory`. Abort!"

    ckpt_files = sorted(
        filename
        for filename in os.listdir(os.path.join(predict_model, "checkpoints"))
        if filename.startswith("epoch") and filename.endswith(".ckpt")
    )

    if len(ckpt_files) > 1:
        warnings.warn(
            f"There are multiple .ckpt files listed in {predict_model}; using last sorted checkpoint: {ckpt_files[-1]}",
            UserWarning,
        )

    if len(ckpt_files) == 0:
        raise RuntimeError(f"No checkpoint files have been found in {predict_model}.")

    ckpt_file = os.path.join(predict_model, "checkpoints", ckpt_files[-1])

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

    hyperedge_order = int(_select(cfg, 'model.hyperedge_order', 'hyperedge_order', 3))
    topology_name = _select(cfg, 'predicting.topology', 'topology')
    topology = _topology_func(topology_name)

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

    predict_output = _select(cfg, 'predicting.save_as', 'predict_output', None)
    if predict_output is None:
        warnings.warn("No output path is provided in `predict_output`, using default: `output.pkl`.")
        predict_output = "output.pkl"

    chunk_size = _select(cfg, "predicting.chunk_size_events", "predict_chunk_size", None)
    if chunk_size is None:
        chunk_size = _select(cfg, "predicting.chunk_size", default=None)
    chunk_batches = _select(cfg, "predicting.chunk_batches", default=None)
    output_mode = _select(cfg, "predicting.output_mode", default="selected")
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
        writer = ChunkedPickleWriter(predict_output, overwrite=overwrite)

        for results_chunk in prediction_chunks:
            writer.write(results_chunk)
            print(f"[predict] wrote_part rows={len(results_chunk)} total_rows={writer.n_events}", flush=True)
            del results_chunk

        writer.close()

    elif output_str.endswith(".pkl"):
        if chunking_enabled:
            writer = ChunkedPickleWriter(predict_output, overwrite=overwrite)
            for results_chunk in prediction_chunks:
                writer.write(results_chunk)
                print(f"[predict] wrote_part rows={len(results_chunk)} total_rows={writer.n_events}", flush=True)
                del results_chunk
            writer.close()
        else:
            chunks = list(prediction_chunks)
            results = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            if "event_index" in results.columns:
                results = results.drop(columns=["event_index"])
            results.to_pickle(predict_output)
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
        if "event_index" in results.columns:
            results = results.drop(columns=["event_index"])
        _write_h5_single_chunk_or_warn(results, predict_output)

    else:
        raise ValueError("You must provide a file extension: `.h5`, `.pkl`, or `.pkl.parts`.")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    Predict()
