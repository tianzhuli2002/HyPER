from pathlib import Path
import hydra
import torch

from omegaconf import DictConfig, OmegaConf
from torch.export.dynamic_shapes import Dim

from HyPER.models import HyPERModel


def _resolve_checkpoint(selector: str | None, model_directory: str | None) -> Path:
    if selector is None or not str(selector).strip():
        raise ValueError("onnx_export.checkpoint must be an explicit path or selector 'best'/'last'.")
    direct = Path(str(selector)).expanduser()
    if direct.is_file():
        return direct.resolve()
    if selector not in {"best", "last"} or not model_directory:
        raise ValueError("ONNX checkpoint must be an existing path, or best/last with model_directory.")
    directory = Path(str(model_directory)).expanduser() / "checkpoints"
    if selector == "last":
        path = directory / "last.ckpt"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates = list(directory.glob("best-total*.ckpt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one best-total checkpoint in {directory}, found {len(candidates)}.")
    return candidates[0].resolve()


class _ONNXOutputAdapter(torch.nn.Module):
    def __init__(self, model, include_classification: bool = True):
        super().__init__()
        self.model = model
        self.include_classification = bool(include_classification)

    @staticmethod
    def _reco_score(logits):
        return 1.0 - torch.softmax(logits, dim=1)[:, -1:]

    def forward(self, x, edge_index, edge_attr, u, batch, hyperedge_index, hyperedge_index_batch):
        p_hyper, batch_hyperedge, p_edge, cls_out = self.model(
            x, edge_index, edge_attr, u, batch, hyperedge_index, hyperedge_index_batch
        )
        if self.model.reconstruction_enabled:
            p_hyper = self._reco_score(p_hyper)
            p_edge = self._reco_score(p_edge)
        if self.model.reconstruction_enabled and self.include_classification:
            return p_hyper, batch_hyperedge, p_edge, cls_out
        if self.model.reconstruction_enabled:
            return p_hyper, batch_hyperedge, p_edge
        return cls_out


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def Onnx(cfg : DictConfig) -> None:
    r"""Convert a trained network model to Onnx.

    Args:
        cfg (str): a `.yaml` file, stores training related parameters. (default: :obj:`str`=None).
    """
    print(OmegaConf.to_yaml(cfg))

    # Map location
    predict_with = cfg.onnx_export.accelerator
    map_location = torch.device('cuda') if str(predict_with).lower() == "gpu" else torch.device('cpu')

    # Load checkpoints
    ckpt_file = _resolve_checkpoint(cfg.onnx_export.checkpoint, cfg.onnx_export.model_directory)
    model = HyPERModel.load_from_checkpoint(str(ckpt_file), map_location=map_location)

    model.eval()
    classification_enabled = bool(getattr(model, 'classification_enabled', True))
    export_model = _ONNXOutputAdapter(model, include_classification=classification_enabled)
    if model.reconstruction_enabled and classification_enabled:
        output_names = ['hyperedge_score', 'batch_hyperedge', 'edge_score', 'classification_logit']
    elif model.reconstruction_enabled:
        output_names = ['hyperedge_prime','batch_hyperedge','edge_prime']
    else:
        output_names = ['classification_logit']

    hparams = model.hparams

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
    onnx_program.save(str(cfg.onnx_export.save_as))

if __name__ == "__main__":
    Onnx()
