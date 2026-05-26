import os
import yaml
import hydra
import torch
import warnings

from omegaconf import DictConfig, OmegaConf
from torch.export.dynamic_shapes import Dim

from HyPER.models import HyPERModel


def _select(cfg: DictConfig, path: str, legacy: str = None, default=None):
    value = OmegaConf.select(cfg, path, default=None)
    if value is not None:
        return value
    if legacy is not None:
        value = OmegaConf.select(cfg, legacy, default=None)
        if value is not None:
            return value
    return default


class _ReconstructionOnlyONNX(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr, u, batch, hyperedge_index, hyperedge_index_batch):
        p_hyper, batch_hyperedge, p_edge, _ = self.model(
            x, edge_index, edge_attr, u, batch, hyperedge_index, hyperedge_index_batch
        )
        return p_hyper, batch_hyperedge, p_edge


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Onnx(cfg : DictConfig) -> None:
    r"""Convert a trained network model to Onnx.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    # Map location
    predict_with = _select(cfg, 'onnx_export.accelerator', 'predict_with', 'cpu')
    map_location = torch.device('cuda') if str(predict_with).lower() == "gpu" else torch.device('cpu')

    # Load checkpoints
    convert_model = _select(cfg, 'onnx_export.model_directory', 'convert_model', None)
    assert convert_model is not None, "No model directory is provided in `convert_model`/`onnx_export.model_directory`. Abort!"
    ckpt_file = [filename for filename in os.listdir(os.path.join(convert_model, "checkpoints")) if filename.startswith("epoch")]
    if len(ckpt_file) > 1:
        warnings.warn(f"There are multiple .ckpt files listed in {convert_model}, using the last checkpoint.")
        ckpt_file = os.path.join(convert_model, "checkpoints", ckpt_file[-1])
    if len(ckpt_file) == 0:
        raise RuntimeError(f"No checkpoint files have been found in {convert_model}.")
    ckpt_file = os.path.join(convert_model, "checkpoints", ckpt_file[0])

    hparams_file = os.path.join(convert_model, "hparams.yaml")
    assert os.path.isfile(hparams_file), f"`hparams.ymal` is not found in {convert_model}."
    with open(hparams_file) as stream:
        hparams = yaml.safe_load(stream)

    model = HyPERModel.load_from_checkpoint(
        checkpoint_path = ckpt_file,
        hparams_file = hparams_file,
        map_location = map_location,
    )

    model.eval()
    export_model = model
    classification_enabled = bool(getattr(model, 'classification_enabled', True))
    output_names = ['hyperedge_prime','batch_hyperedge','edge_prime','classification_score']
    if not classification_enabled:
        export_model = _ReconstructionOnlyONNX(model)
        output_names = ['hyperedge_prime','batch_hyperedge','edge_prime']

    onnx_program = torch.onnx.export(
        export_model,
        (
            torch.randn((13,hparams['node_in_channels'])),
            torch.randint(0,12,(2,72)),
            torch.randn((72,hparams['edge_in_channels'])),
            torch.randn((2,hparams['global_in_channels'])),
            torch.LongTensor([0,0,0,0,0,0,1,1,1,1,1,1,1]),
            torch.randint(0,12,(hparams['hyperedge_order'],55)),
            torch.cat([torch.full([20],0, dtype=torch.int64),torch.full([35],1, dtype=torch.int64)],dim=0)
        ),
        dynamo=True,
        opset_version=18,
        input_names=['x_s', 'edge_index', 'edge_attr_s', 'u_s', 'batch', 'edge_index_h', 'batch_hyperedge'],
        dynamic_shapes={'x_s'               : {0 : Dim.DYNAMIC},
                        'edge_index'        : {1 : Dim.DYNAMIC},
                        'edge_attr_s'       : {0 : Dim.DYNAMIC},
                        'u_s'               : {0 : Dim.DYNAMIC},
                        'batch'             : {0 : Dim.DYNAMIC},
                        'edge_index_h'      : {1 : Dim.DYNAMIC},
                        'batch_hyperedge': {0 : Dim.DYNAMIC}},
        output_names = output_names,
    )

    onnx_program.optimize()
    onnx_program.save(_select(cfg, 'onnx_export.save_as', 'onnx_output', 'HyPER.onnx'))

if __name__ == "__main__":
    Onnx()
