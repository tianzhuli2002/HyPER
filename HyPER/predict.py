import os
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


def _select(cfg: DictConfig, path: str, legacy: str = None, default=None):
    value = OmegaConf.select(cfg, path, default=None)
    if value is not None:
        return value
    if legacy is not None:
        value = OmegaConf.select(cfg, legacy, default=None)
        if value is not None:
            return value
    return default


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


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Predict(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = _select(cfg, 'predicting.accelerator', 'predict_with', _select(cfg, 'trainer.accelerator', 'device', 'cpu'))
    batch_size = _select(cfg, 'predicting.batch_size', 'batch_size', 128)
    datamodule = HyPERDataModule(
        root = _select(cfg, 'dataset.root', 'dataset'),
        train_set = None,
        val_set = None,
        predict_set = _select(cfg, 'dataset.predict_set', 'predict_set'),
        batch_size = batch_size,
        percent_valid_samples = 1 - float(_select(cfg, 'dataset.train_val_split', 'train_val_split', 0.95)),
        pin_memory = True if device == "gpu" else False,
        drop_last = False,
        num_workers = _select(cfg, 'dataset.num_workers', 'datamodule.num_workers', 0),
        force_reload = _select(cfg, 'dataset.force_reload', 'datamodule.force_reload', False),
        use_ondisk = _select(cfg, 'dataset.use_ondisk', 'datamodule.use_ondisk', True),
        graph_config = _graph_config_from_cfg(cfg),
    )

    # Map location
    map_location = torch.device('cuda') if str(device).lower() == "gpu" else torch.device('cpu')

    # Load checkpoints
    predict_model = _select(cfg, 'predicting.model_directory', 'predict_model', None)
    assert predict_model is not None, "No model directory is provided in `predict_model`/`predicting.model_directory`. Abort!"
    ckpt_file = [filename for filename in os.listdir(os.path.join(predict_model, "checkpoints")) if filename.startswith("epoch")]
    if len(ckpt_file) > 1:
        warnings.warn(f"There are multiple .ckpt files listed in {predict_model}, using the last checkpoint.")
        ckpt_file = os.path.join(predict_model, "checkpoints", ckpt_file[-1])
    if len(ckpt_file) == 0:
        raise RuntimeError(f"No checkpoint files have been found in {predict_model}.")
    ckpt_file = os.path.join(predict_model, "checkpoints", ckpt_file[0])

    # Load hyperparameters
    hparams_file = os.path.join(predict_model, "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.ymal` is not found in {predict_model}."

    model = HyPERModel.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = map_location,
    )

    trainer = pl.Trainer(
        accelerator = device,
        devices = _select(cfg, 'trainer.devices', 'num_devices', 1)
    )

    out = trainer.predict(model, datamodule=datamodule)

    hyperedge_out = []
    graphedge_out = []
    cls_scores = []
    hyperedge_vct = []
    graphedge_vct = []
    hyperedges = []
    graphedges = []
    hyperedge_order = _select(cfg, 'model.hyperedge_order', 'hyperedge_order', 3)
    
    for i in tqdm(range(len(out)), desc="Evaluating", unit='batch'):
        x_out, edge_attr_out, N_nodes, encodings, x_class_out = out[i]

        for j in range(len(x_out)):
            hyperedges.append([list(x) for x in combinations(range(int(N_nodes[j])),r=hyperedge_order)])
            hyperedge_out.append(x_out[j].cpu().flatten().tolist())
            hyperedge_vct.append([list(x) for x in combinations(encodings[j].cpu().flatten().tolist(),r=hyperedge_order)])

            graphedges.append([list(x) for x in combinations(range(int(N_nodes[j])),r=2)])
            graphedge_out.append(edge_attr_out[j].cpu().flatten().tolist())
            graphedge_vct.append([list(x) for x in combinations(encodings[j].cpu().flatten().tolist(),r=2)])
        
        if x_class_out is not None:
            cls_scores.extend(x_class_out.cpu().flatten().tolist())

    output_columns = {
        "HyPER_HE_RAW"  : hyperedge_out,
        "HyPER_GE_RAW"  : graphedge_out,
        "HyPER_HE_VCT"  : hyperedge_vct,
        "HyPER_GE_VCT"  : graphedge_vct,
        "HyPER_HE_IDX"  : hyperedges,
        "HyPER_GE_IDX"  : graphedges,
    }
    if cls_scores:
        output_columns["HyPER_CLS_RAW"] = cls_scores
    results = pd.DataFrame(output_columns)

    results = eval(_select(cfg, 'predicting.topology', 'topology'))(results)

    predict_output = _select(cfg, 'predicting.save_as', 'predict_output', None)
    if predict_output is None:
        warnings.warn("No output path is provided in `predict_output`, use default: `output.h5`.")
        ResultWriter(results, "output.h5")
    else:
        if str(predict_output)[-3:] == '.h5':
            warnings.warn("Saving results to a `.h5` file, RAW outputs will not be saved. If you want to save all output, use `.pkl` extension.", UserWarning)
            ResultWriter(results, str(predict_output))
        elif str(predict_output)[-4:] == '.pkl':
            warnings.warn("Pickling all results (including RAW network outputs), your performance may suffer.", UserWarning)
            results.to_pickle(str(predict_output))
        else:
            raise ValueError("You must provide a file extension: `.h5` or `.pkl`.")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    Predict()
