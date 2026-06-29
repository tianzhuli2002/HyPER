from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Sequence
from warnings import warn
from copy import deepcopy
from os import path as osp
from itertools import combinations, permutations
import time
import sys
import os

import math
import h5py
import yaml
import torch
import numpy as np
import awkward as ak
import numpy.lib.recfunctions as rf

from torch import Tensor
from torch_geometric.data import Data, Dataset

from .transform import TransformFeatures
from .filter import TargetConnectivityFilter
from .edge_features import EDGE_FEATURE_TRANSFORMS
from .transforms import TRANSFORM_REGISTRY


@dataclass(frozen=True)
class FeatureOutput:
    name: str
    source: str
    transform: str = "identity"


@dataclass(frozen=True)
class FeaturePlan:
    raw_features: List[str]
    resolved_features: List[str]
    outputs: List[FeatureOutput]
    builders: dict


class HyPERDataset(Dataset):
    """Build HyPER PyG graphs on demand from HDF5 input.

    This is the canonical event-level graph builder. The training path normally
    wraps it with :class:`HyPEROnDiskDataset`, which materialises these graphs
    into a VyPER-style HDF5 ``.db`` file.
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        training: bool = True,
        force_reload: bool = False,
        cache_dir: Optional[str] = None,
        config: Optional[dict] = None,
        vectorized_chunks: Optional[bool] = None,
    ) -> None:

        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._train_mode = training
        if vectorized_chunks is None:
            vectorized_chunks = self._env_bool("HYPER_VECTORIZE_CHUNKS", True)
        self.use_vectorized_chunks = bool(vectorized_chunks)
        self._edge_template_cache: dict[int, torch.Tensor] = {}
        self._hyperedge_template_cache: dict[tuple[int, int], torch.Tensor] = {}

        if cache_dir is not None:
            warn("`cache_dir` is deprecated; HyPERDataset no longer writes per-event .pt caches.", DeprecationWarning)
        if force_reload:
            warn("`force_reload` is handled by HyPEROnDiskDataset for .db preprocessing.", UserWarning)

        # Parse graph-construction config. New unified Hydra configs can pass
        # embedded input/target sections; old workflows still use root/config.yaml.
        parsed_inputs = self._resolve_graph_config(root=self.root, config=config)
        self.node_input_names = list(parsed_inputs['input']['nodes'].keys())
        self.input_id = parsed_inputs['input']['nodes']
        self.input_pad_size = parsed_inputs['input']['padding']
        self.node_feature_plan = self._resolve_feature_plan(
            parsed_inputs['input'].get('node_features', []),
            parsed_inputs['input'].get('node_feature_transforms', {}),
            "input.node_feature_transforms",
        )
        self.global_feature_plan = self._resolve_feature_plan(
            parsed_inputs['input'].get('global_features', []),
            parsed_inputs['input'].get('global_feature_transforms', {}),
            "input.global_feature_transforms",
        )
        self.reco_feature_plan = self._resolve_feature_plan(
            parsed_inputs['input'].get('reco_features', []),
            parsed_inputs['input'].get('reco_feature_transforms', {}),
            "input.reco_feature_transforms",
        )
        self.raw_node_feature_names = list(self.node_feature_plan.raw_features)
        self.node_feature_names = list(self.node_feature_plan.resolved_features)
        self.raw_global_feature_names = list(self.global_feature_plan.raw_features)
        self.global_feature_names = list(self.global_feature_plan.resolved_features)
        self._node_feature_index = {
            name: idx for idx, name in enumerate(self.node_feature_names)
        }
        self._node_raw_feature_index = {
            name: idx for idx, name in enumerate(self.raw_node_feature_names)
        }
        self._graph_config_for_debug = {
            'input': deepcopy(parsed_inputs['input']),
            'target': deepcopy(parsed_inputs['target']),
        }
        self._invalid_truth_label_warning_count = 0

        target_cfg = parsed_inputs.get('target', {})
        self.target_encoding = str(target_cfg.get("encoding", "binary")).strip().lower()
        if self.target_encoding not in {"binary", "typed"}:
            raise ValueError("target.encoding must be either 'binary' or 'typed'.")

        edge_target_cfg = target_cfg.get('edge', {}) or {}
        hyperedge_target_cfg = target_cfg.get('hyperedge', {}) or {}
        self.edge_target_names = list(edge_target_cfg.keys())
        self.hyperedge_target_names = list(hyperedge_target_cfg.keys())
        self.edge_targets = [self._normalise_target_group(v) for v in edge_target_cfg.values()]
        self.hyperedge_targets = [self._normalise_target_group(v) for v in hyperedge_target_cfg.values()]
        self.hyperedge_order = len(self.hyperedge_targets[0][0]) if self.hyperedge_targets and self.hyperedge_targets[0] else 2
        self.hyperedge_ordered = bool(target_cfg.get("hyperedge_ordered", False))
        self.target_edge_ids, self.target_hyperedge_ids = self.assign_target_ids()
        self.edge_out_channels = len(self.edge_target_names) + 1 if self.target_encoding == "typed" else 1
        self.hyperedge_out_channels = len(self.hyperedge_target_names) + 1 if self.target_encoding == "typed" else 1
        self.edge_background_class = self.edge_out_channels - 1 if self.target_encoding == "typed" else None
        self.hyperedge_background_class = self.hyperedge_out_channels - 1 if self.target_encoding == "typed" else None

        # 4-vector setup. The model-facing node feature list may use periodic
        # sin/cos phi inputs, but graph construction still needs an internal phi.
        self._use_raw_EEtaPhiPt = self._has_raw_node_features('e', 'eta', 'phi', 'pt')
        self._use_EEtaPhiPt = self._has_node_features('e', 'eta', 'phi', 'pt')
        self._use_EEtaSinCosPhiPt = self._has_node_features('e', 'eta', 'sin_phi', 'cos_phi', 'pt')
        self._use_EPxPyPz = self._has_node_features('e', 'px', 'py', 'pz')
        if self._use_raw_EEtaPhiPt:
            self._phi_source = 'phi'
        elif self._use_EEtaSinCosPhiPt:
            self._phi_source = 'atan2(sin_phi, cos_phi)'
        elif self._use_EEtaPhiPt:
            self._phi_source = 'phi'
        elif self._use_EPxPyPz:
            self._phi_source = 'px/py'
        else:
            raise ValueError(
                "Node features must contain one supported 4-vector basis: "
                "['e', 'eta', 'phi', 'pt'], ['e', 'eta', 'sin_phi', 'cos_phi', 'pt'], "
                "or ['e', 'px', 'py', 'pz']."
            )

        # Edge features
        self.edge_features_to_use = self.parse_edge_features(parsed_inputs)
        self.edge_feature_names = list(self.edge_features_to_use.keys())
        self.edge_directionality = str(
            parsed_inputs["input"].get("edge_directionality", "undirected")
        ).strip().lower()
        if self.edge_directionality not in {"undirected", "directed"}:
            raise ValueError(
                "input.edge_directionality must be either 'undirected' or 'directed'."
            )

        # Node/global transforms
        node_transform_names = parsed_inputs['input'].get('node_transforms', [])
        global_transform_names = parsed_inputs['input'].get('global_transforms', [])

        if len(node_transform_names) != len(self.node_feature_names):
            raise ValueError(
                "`input.node_transforms` must match the resolved model-facing node features. "
                f"raw features={self.raw_node_feature_names}, "
                f"resolved features={self.node_feature_names}, "
                f"scalar transform list length={len(node_transform_names)}, "
                f"expected length={len(self.node_feature_names)}."
            )
        if len(global_transform_names) != len(self.global_feature_names):
            raise ValueError(
                "`input.global_transforms` must match the resolved model-facing global features. "
                f"raw features={self.raw_global_feature_names}, "
                f"resolved features={self.global_feature_names}, "
                f"scalar transform list length={len(global_transform_names)}, "
                f"expected length={len(self.global_feature_names)}."
            )

        self._node_transforms = self._resolve_transforms(node_transform_names, "input.node_transforms")
        self._global_transforms = self._resolve_transforms(global_transform_names, "input.global_transforms")
        self.node_in_channels = len(self.node_feature_names) + 1
        self.edge_in_channels = len(self.edge_feature_names)
        self.global_in_channels = len(self.global_feature_names)

        # Build transform pipeline
        self._transforms = TransformFeatures(
            ["x", "u", "edge_attr"],
            transforms=[
                self._node_transforms,
                self._global_transforms,
                list(self.edge_features_to_use.values())
            ]
        )
        if self.transform is None:
            self.transform = self._transforms

        # Filter
        if 'filter' in parsed_inputs:
            self._pre_filter = TargetConnectivityFilter(
                num_edge_targets=parsed_inputs['filter']['num_edges'],
                num_hyperedge_targets=parsed_inputs['filter']['num_hyperedge']
            )
        else:
            self._pre_filter = None

        # HDF5 path (do NOT keep a global open handle - open per-worker)
        self._h5_path = osp.join(root, "raw", f"{name}.h5")

        # Get number of events by opening/closing briefly (safe at init)
        with h5py.File(self._h5_path, 'r') as _f:
            self._validate_h5_feature_fields(_f["INPUTS"])
            self._num_events = len(_f["INPUTS"]["GLOBAL"])
        self.file = None

        # Precompute target sets
        self._target_edge_set = set(tuple(t) for t in self._flatten_target_ids(self.target_edge_ids)) if self.target_edge_ids else set()
        self._target_hyperedge_set = set(tuple(t) for t in self._flatten_target_ids(self.target_hyperedge_ids)) if self.target_hyperedge_ids else set()

        # === DEBUG: Log dataset creation ===
        if os.getenv("HYPER_DEBUG", "0") == "1":
            print(f"[DEBUG] HyPERDataset created: {self.name}, {self._num_events} events, PID={os.getpid()}",
                  file=sys.stderr, flush=True)


    def __len__(self) -> int:
        return self._num_events


    def __getitem__(self, idx: int) -> Data:
        """Load and process a single event on demand."""

        # === DEBUG: Progress logging (every 1000 events) ===
        if os.getenv("HYPER_DEBUG", "0") == "1" and idx % 1000 == 0:
            t0 = time.time()
            print(f"[DATA] Event {idx}/{len(self)} - PID={os.getpid()} - Time={t0:.0f}",
                  file=sys.stderr, flush=True)

        t0 = time.time()
        data = self.processing(idx)
        t_process = time.time() - t0

        # Apply filters/transforms
        if self._pre_filter and not self._pre_filter(data):
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            data = self.transform(data)

        if os.getenv("HYPER_DEBUG", "0") == "1" and idx % 1000 == 0:
            t_total = time.time() - t0
            print(f"[DATA] Processed {idx}: process={t_process:.2f}s, total={t_total:.2f}s",
                  file=sys.stderr, flush=True)

        return data


    def processing_chunk(self, start: int, end: int) -> List[Data]:
        """Build a contiguous event chunk.

        By default this uses a chunk-aware path that performs contiguous HDF5
        reads once per input/label collection, then assembles per-event PyG Data
        objects. Set ``HYPER_VECTORIZE_CHUNKS=0`` to force the legacy loop.
        """
        if self.use_vectorized_chunks and self._pre_filter is None:
            try:
                return self.processing_chunk_vectorized(start, end)
            except Exception:
                if self._env_bool("HYPER_VECTORIZE_CHUNKS_FALLBACK", False):
                    warn(
                        "Vectorized chunk graph construction failed; falling back "
                        "to per-event processing because HYPER_VECTORIZE_CHUNKS_FALLBACK=1.",
                        UserWarning,
                    )
                else:
                    raise
        return [self[idx] for idx in range(start, end)]


    def processing_chunk_vectorized(self, start: int, end: int) -> List[Data]:
        """Build graphs for a contiguous HDF5 slice with amortised I/O."""
        if end <= start:
            return []

        self._ensure_file_open()
        inputs = self.file["INPUTS"]
        labels = self.file["LABELS"] if "LABELS" in self.file else None
        # Build labels and reconstruction targets whenever the H5 provides
        # them. This keeps typed graph caches reusable between reconstruction,
        # classifier-only training, and prediction/evaluation modes.
        has_targets = labels is not None

        x_all, counts, node_p4_all, valid_input_mask = self.build_node_attributes_with_mask(
            inputs, start, end
        )
        counts_list = [int(v) for v in counts.tolist()]
        offsets = np.concatenate([[0], np.cumsum(counts_list, dtype=np.int64)])

        if has_targets:
            cantor_nodes_all = self.assign_node_ids_from_chunk(
                x_all,
                labels,
                start,
                end,
                counts,
                valid_input_mask,
            ).to(torch.long)
        else:
            cantor_nodes_all = None

        u_all = self._build_global_model_array(inputs, start, end)
        if u_all.ndim == 3 and u_all.shape[1] == 1:
            u_all = u_all[:, 0, :]
        if u_all.ndim == 1:
            u_all = u_all.unsqueeze(0)

        if has_targets and "GLOBAL" in labels:
            cls_all = self.build_global_targets(labels, start, end)
            if cls_all.ndim == 1:
                cls_all = cls_all.unsqueeze(1)
        else:
            cls_all = None

        graphs: List[Data] = []
        for local_idx, n_nodes in enumerate(counts_list):
            lo = int(offsets[local_idx])
            hi = int(offsets[local_idx + 1])
            x = x_all[lo:hi].clone()
            node_p4 = node_p4_all[lo:hi].clone()

            edge_index = self._edge_index_template(n_nodes)
            edge_attr = self.build_edge_attributes_from_pairs(edge_index, node_p4)

            if has_targets and cantor_nodes_all is not None:
                cantor_nodes = cantor_nodes_all[lo:hi]
                cantor_edge_index = cantor_nodes[edge_index] if edge_index.numel() else torch.empty((2, 0), dtype=torch.long)
                if self.target_encoding == "typed":
                    edge_attr_t = self.build_typed_edge_target(cantor_edge_index)
                else:
                    edge_attr_t = self.find_matched_connections(
                        cantor_edge_index,
                        self._flatten_target_ids(self.target_edge_ids),
                    )
                node_ids = cantor_nodes.float().clone()
            else:
                cantor_nodes = None
                edge_attr_t = torch.empty((edge_index.shape[1], 0), dtype=torch.float32)
                node_ids = torch.empty((0,), dtype=torch.float32)

            if self.hyperedge_targets:
                hyperedge_index = self._hyperedge_index_template(n_nodes, self.hyperedge_order)
                if has_targets and cantor_nodes is not None:
                    cantor_hyperedge_index = (
                        cantor_nodes[hyperedge_index]
                        if hyperedge_index.numel()
                        else torch.empty((self.hyperedge_order, 0), dtype=torch.long)
                    )
                    if self.target_encoding == "typed":
                        hyperedge_attr_t = self.build_typed_hyperedge_target(cantor_hyperedge_index)
                    else:
                        hyperedge_attr_t = self.find_matched_connections(
                            cantor_hyperedge_index,
                            self._flatten_target_ids(self.target_hyperedge_ids),
                        )
                else:
                    hyperedge_attr_t = torch.empty((hyperedge_index.shape[1], 0), dtype=torch.float32)
            else:
                hyperedge_index = torch.empty((2, 0), dtype=torch.long)
                hyperedge_attr_t = torch.empty((0, 1), dtype=torch.float32)

            u = u_all[local_idx].unsqueeze(0).clone()
            cls_t = (
                cls_all[local_idx].unsqueeze(0).clone()
                if cls_all is not None
                else torch.empty((1, 0), dtype=torch.float32)
            )

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_p4=node_p4,
                u=u,
                cls_t=cls_t,
                node_ids=node_ids,
                edge_attr_t=edge_attr_t,
                hyperedge_index=hyperedge_index,
                hyperedge_attr_t=hyperedge_attr_t,
                **self._cached_reco_target_fields(edge_attr_t, hyperedge_attr_t),
            )

            if self.transform:
                data = self.transform(data)
            graphs.append(data)

        return graphs


    def processing(self, idx: int) -> Data:
        """
        Extract and build one event from HDF5 using self.file.
        The HDF5 file handle is opened lazily per process/worker.
        """
        # Ensure file handle is opened in this process/worker
        self._ensure_file_open()
        inputs = self.file["INPUTS"]
        labels = self.file["LABELS"] if "LABELS" in self.file else None

        # Node attributes (also returns precomputed per-node 4-vectors)
        x, Nobjects, node_p4 = self.build_node_attributes(inputs, idx, idx+1)

        # Edge indices and attributes. Build candidate pairs once and derive
        # edge features from the already-materialised node four-vectors.
        # Build labels and reconstruction targets whenever the H5 provides
        # them. This keeps typed graph caches reusable between reconstruction,
        # classifier-only training, and prediction/evaluation modes.
        has_targets = labels is not None
        if has_targets:
            cantor_node_ids, local_node_ids = self.assign_node_ids(
                x, labels, idx, idx+1, Nobjects
            )
            edge_index, cantor_edge_index = self.build_edge_indices(
                local_node_ids, cantor_node_ids
            )
        else:
            cantor_node_ids = None
            local_node_ids = self.build_local_node_ids(Nobjects)
            edge_index = self.build_edge_indices(local_node_ids, None)[0]
            cantor_edge_index = None
        edge_attr = self.build_edge_attributes_from_pairs(edge_index, node_p4)

        if edge_index.shape[1] != edge_attr.shape[0]:
            raise RuntimeError(
                f"edge_index/edge_attr mismatch for event {idx}: "
                f"{edge_index.shape[1]} edges vs {edge_attr.shape[0]} edge rows"
            )
        if edge_index.numel() > 0 and (edge_index[0] == edge_index[1]).any():
            raise RuntimeError(f"Self-edge found while building event {idx}.")

        # Hyperedges (if used)
        if self.hyperedge_targets:
            hyperedge_index, cantor_hyperedge_index = self.build_hyperedge_indices(
                local_node_ids, cantor_node_ids, self.hyperedge_order
            )
            if has_targets:
                if self.target_encoding == "typed":
                    hyperedge_attr_t = self.build_typed_hyperedge_target(cantor_hyperedge_index)
                else:
                    hyperedge_attr_t = self.find_matched_connections(
                        cantor_hyperedge_index, self._flatten_target_ids(self.target_hyperedge_ids)
                    )
            else:
                hyperedge_attr_t = torch.empty((hyperedge_index.shape[1], 0), dtype=torch.float32)
        else:
            hyperedge_index = torch.empty((2, 0), dtype=torch.long)
            hyperedge_attr_t = torch.empty((0, 1), dtype=torch.float32)

        # Global attributes & targets
        u = self.build_global_attributes(inputs, idx, idx+1)
        cls_t = self.build_global_targets(labels, idx, idx+1) if has_targets and "GLOBAL" in labels else torch.empty((1, 0), dtype=torch.float32)

        # Edge targets
        if has_targets:
            if self.target_encoding == "typed":
                edge_attr_t = self.build_typed_edge_target(cantor_edge_index)
            else:
                edge_attr_t = self.find_matched_connections(
                    cantor_edge_index, self._flatten_target_ids(self.target_edge_ids)
                )
        else:
            edge_attr_t = torch.empty((edge_index.shape[1], 0), dtype=torch.float32)

        # Flatten node IDs
        if cantor_node_ids is not None:
            cantor_node_ids_flat = torch.tensor(
                np.asarray(ak.flatten(cantor_node_ids)), dtype=torch.float32
            )
        else:
            cantor_node_ids_flat = torch.empty((0,), dtype=torch.float32)

        # Ensure proper shapes for transforms
        if u.ndim == 1:
            u = u.unsqueeze(0)
        if cls_t.ndim == 1:
            cls_t = cls_t.unsqueeze(0)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_p4=node_p4,
            u=u,
            cls_t=cls_t,
            node_ids=cantor_node_ids_flat,
            edge_attr_t=edge_attr_t,
            hyperedge_index=hyperedge_index,
            hyperedge_attr_t=hyperedge_attr_t,
            **self._cached_reco_target_fields(edge_attr_t, hyperedge_attr_t),
        )


    # === Static helper methods (all fork-compatible) ===

    def _cached_reco_target_fields(self, edge_attr_t: torch.Tensor, hyperedge_attr_t: torch.Tensor) -> dict:
        """Optional future cache fields; old graph DBs are still valid without them."""
        fields = {}
        if self.target_encoding == "typed":
            if edge_attr_t is not None and edge_attr_t.numel() > 0 and edge_attr_t.ndim == 2:
                edge_class = edge_attr_t.argmax(dim=1).to(torch.long)
                fields["edge_attr_t_class"] = edge_class
                fields["edge_reco_active"] = (edge_class != (edge_attr_t.size(1) - 1)).any().reshape(1)
            else:
                fields["edge_attr_t_class"] = torch.empty((0,), dtype=torch.long)
                fields["edge_reco_active"] = torch.zeros((1,), dtype=torch.bool)

            if hyperedge_attr_t is not None and hyperedge_attr_t.numel() > 0 and hyperedge_attr_t.ndim == 2:
                hyper_class = hyperedge_attr_t.argmax(dim=1).to(torch.long)
                fields["hyperedge_attr_t_class"] = hyper_class
                fields["hyperedge_reco_active"] = (hyper_class != (hyperedge_attr_t.size(1) - 1)).any().reshape(1)
            else:
                fields["hyperedge_attr_t_class"] = torch.empty((0,), dtype=torch.long)
                fields["hyperedge_reco_active"] = torch.zeros((1,), dtype=torch.bool)
        else:
            if edge_attr_t is not None and edge_attr_t.numel() > 0:
                fields["edge_reco_active"] = (edge_attr_t.float().flatten() > 0.5).any().reshape(1)
            else:
                fields["edge_reco_active"] = torch.zeros((1,), dtype=torch.bool)
            if hyperedge_attr_t is not None and hyperedge_attr_t.numel() > 0:
                fields["hyperedge_reco_active"] = (hyperedge_attr_t.float().flatten() > 0.5).any().reshape(1)
            else:
                fields["hyperedge_reco_active"] = torch.zeros((1,), dtype=torch.bool)
        return fields

    @staticmethod
    def _parse_config_file(filename):
        with open(filename) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return bool(default)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _resolve_graph_config(root: str, config: Optional[dict] = None):
        if config is not None:
            parsed = deepcopy(config)
            if 'input' in parsed and 'target' in parsed:
                return parsed
            raise ValueError("Embedded HyPER graph config must contain both `input` and `target` sections.")

        config_path = f"{root}/config.yaml"
        if not osp.exists(config_path):
            raise FileNotFoundError(
                "No embedded `input`/`target` config was provided and no external "
                f"graph config was found at {config_path}."
            )
        return HyPERDataset._parse_config_file(config_path)

    @staticmethod
    def _normalise_target_group(value) -> List[List[str]]:
        """Return a list of same-type target connections.

        HyPER legacy configs use ``name: ['1-1', '1-2']`` for one connection.
        VyPER-style configs may use ``name: [['1-1', '1-2'], ...]``.  Internally
        both are represented as a list of connection definitions so one typed
        output channel can match multiple same-semantic targets.
        """
        if value is None:
            return []
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError(f"Target definition must be a list, got {type(value)}")
        if len(value) == 0:
            return []
        if all(isinstance(item, str) for item in value):
            return [list(value)]
        groups = []
        for item in value:
            if isinstance(item, tuple):
                item = list(item)
            if not isinstance(item, list) or not all(isinstance(label, str) for label in item):
                raise TypeError(f"Unsupported nested target definition: {item!r}")
            groups.append(list(item))
        return groups

    @staticmethod
    def _flatten_target_ids(target_ids: Sequence[Sequence[Sequence[int]]]) -> List[List[int]]:
        return [list(connection) for group in target_ids for connection in group]

    @staticmethod
    def _awkward_nondiag_cartesian(arr: ak.Array) -> ak.Array:
        tmp = ak.cartesian((arr,arr), nested=True)
        tmp_index = ak.argcartesian((arr,arr), nested=True)
        return tmp[tmp_index["0"] != tmp_index["1"]]

    @staticmethod
    def _map_nested_awkward_to_torch(arr: ak.Array) -> torch.Tensor:
        """
        Takes an ak.Array and converts to torch tensor.
        Handles both:
        - Structured arrays (from cartesian products of node indices) → [F, N]
        - Regular float arrays (from vector operations) → [N, 1]
        """
        # Flatten the array into 1D
        flat = ak.flatten(arr)

        # Convert to numpy via awkward (returns ndarray)
        np_array = ak.to_numpy(flat)

        # Check if it's a structured array (has named fields)
        if np_array.dtype.names is not None:
            # Structured array: convert to unstructured, then transpose
            unstruct = rf.structured_to_unstructured(np_array)
            # Use torch.from_numpy to avoid extra copies where possible
            return torch.from_numpy(unstruct).t().to(torch.long)
        else:
            # Regular float array: convert via from_numpy and ensure float32
            return torch.from_numpy(np_array.astype(np.float32)).reshape(-1, 1)

    @staticmethod
    def _remove_self_pairing(arr, mask):
        return ak.drop_none(ak.nan_to_none(ak.where(mask, np.nan, arr)))

    @staticmethod
    def _cantor_pairing(a, b):
        # Ensure tensor inputs, compute with integer arithmetic then cast to float32
        a_t = torch.as_tensor(a)
        b_t = torch.as_tensor(b)
        s = (a_t.to(torch.long) + b_t.to(torch.long))
        paired = (s * (s + 1)) // 2 + b_t.to(torch.long)
        return paired.to(torch.float32)


    def parse_edge_features(self, parsed_inputs):
        if "edge_features" in parsed_inputs["input"]:
            requested = parsed_inputs["input"]["edge_features"]
            return {k: EDGE_FEATURE_TRANSFORMS[k] for k in requested}
        return EDGE_FEATURE_TRANSFORMS.copy()

    def _has_node_features(self, *names: str) -> bool:
        return all(name in self._node_feature_index for name in names)

    def _has_raw_node_features(self, *names: str) -> bool:
        return all(name in self._node_raw_feature_index for name in names)

    @staticmethod
    def _normalise_feature_builder_name(name: str, config_key: str, feature: str) -> str:
        normalised = str(name).strip().replace("_", "").lower()
        if normalised == "sincos":
            return "sincos"
        raise ValueError(
            f"Unknown feature-builder transform `{name}` for `{feature}` in `{config_key}`. "
            "Known feature-builder transforms are: ['sincos', 'sin_cos', 'SinCos']."
        )

    @staticmethod
    def _sincos_output_names(feature: str) -> Tuple[str, str]:
        if feature == "phi":
            return "sin_phi", "cos_phi"
        return f"{feature}_sin", f"{feature}_cos"

    @classmethod
    def _resolve_feature_plan(
        cls,
        raw_features: Sequence[str],
        feature_transforms,
        config_key: str,
    ) -> FeaturePlan:
        raw_features = [str(feature) for feature in (raw_features or [])]
        if feature_transforms is None:
            feature_transforms = {}
        if not isinstance(feature_transforms, dict):
            raise TypeError(f"`{config_key}` must be a mapping from raw feature name to transform name.")

        builders = {
            str(feature): cls._normalise_feature_builder_name(transform, config_key, str(feature))
            for feature, transform in feature_transforms.items()
        }
        unknown = [feature for feature in builders if feature not in raw_features]
        if unknown:
            raise ValueError(
                f"`{config_key}` refers to raw feature(s) not present in the configured feature list: {unknown}. "
                f"Raw feature list: {raw_features}."
            )

        outputs: List[FeatureOutput] = []
        resolved_features: List[str] = []
        for feature in raw_features:
            builder = builders.get(feature)
            if builder is None:
                outputs.append(FeatureOutput(name=feature, source=feature))
                resolved_features.append(feature)
            elif builder == "sincos":
                sin_name, cos_name = cls._sincos_output_names(feature)
                outputs.append(FeatureOutput(name=sin_name, source=feature, transform="sin"))
                outputs.append(FeatureOutput(name=cos_name, source=feature, transform="cos"))
                resolved_features.extend([sin_name, cos_name])
            else:
                raise AssertionError(f"Unhandled feature-builder transform: {builder}")

        return FeaturePlan(
            raw_features=raw_features,
            resolved_features=resolved_features,
            outputs=outputs,
            builders=builders,
        )

    @classmethod
    def feature_layout_from_config(cls, config: Optional[dict]) -> dict:
        parsed = deepcopy(config or {})
        input_cfg = parsed.get("input", {})
        node_plan = cls._resolve_feature_plan(
            input_cfg.get("node_features", []),
            input_cfg.get("node_feature_transforms", {}),
            "input.node_feature_transforms",
        )
        global_plan = cls._resolve_feature_plan(
            input_cfg.get("global_features", []),
            input_cfg.get("global_feature_transforms", {}),
            "input.global_feature_transforms",
        )
        reco_plan = cls._resolve_feature_plan(
            input_cfg.get("reco_features", []),
            input_cfg.get("reco_feature_transforms", {}),
            "input.reco_feature_transforms",
        )
        return {
            "node": {
                "raw_features": node_plan.raw_features,
                "resolved_features": node_plan.resolved_features,
                "feature_builders": node_plan.builders,
                "scalar_transforms": list(input_cfg.get("node_transforms", []) or []),
            },
            "global": {
                "raw_features": global_plan.raw_features,
                "resolved_features": global_plan.resolved_features,
                "feature_builders": global_plan.builders,
                "scalar_transforms": list(input_cfg.get("global_transforms", []) or []),
            },
            "reco": {
                "raw_features": reco_plan.raw_features,
                "resolved_features": reco_plan.resolved_features,
                "feature_builders": reco_plan.builders,
                "scalar_transforms": list(input_cfg.get("reco_transforms", []) or []),
            },
        }

    @staticmethod
    def _resolve_transforms(names: Sequence[str], config_key: str) -> List[Callable]:
        unknown = [name for name in names if name not in TRANSFORM_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown transform name(s) in `{config_key}`: {unknown}. "
                f"Known transforms are: {sorted(TRANSFORM_REGISTRY)}"
            )
        return [TRANSFORM_REGISTRY[name] for name in names]

    @staticmethod
    def _format_missing_features_error(group_name: str, missing: Sequence[str], available: Sequence[str]) -> str:
        return (
            f"H5 INPUTS/{group_name} is missing raw feature field(s) required by the resolved feature plan: "
            f"{list(missing)}. Available fields: {list(available)}."
        )

    def _validate_h5_feature_fields(self, inputs: h5py._hl.group.Group) -> None:
        for obj in self.input_id.keys():
            available = list(inputs[obj].dtype.names or [])
            missing = [feature for feature in self.raw_node_feature_names if feature not in available]
            if missing:
                raise ValueError(self._format_missing_features_error(obj, missing, available))

        available_global = list(inputs["GLOBAL"].dtype.names or [])
        missing_global = [feature for feature in self.raw_global_feature_names if feature not in available_global]
        if missing_global:
            raise ValueError(self._format_missing_features_error("GLOBAL", missing_global, available_global))

    @staticmethod
    def _structured_raw_feature_array(arr: np.ndarray, raw_features: Sequence[str], group_name: str) -> np.ndarray:
        available = list(arr.dtype.names or [])
        missing = [feature for feature in raw_features if feature not in available]
        if missing:
            raise ValueError(HyPERDataset._format_missing_features_error(group_name, missing, available))
        if not raw_features:
            return np.empty(arr.shape + (0,), dtype=np.float32)
        return np.stack([np.asarray(arr[feature]) for feature in raw_features], axis=-1).astype(np.float32, copy=False)

    @staticmethod
    def _apply_feature_plan(raw_values: np.ndarray, plan: FeaturePlan) -> np.ndarray:
        if len(plan.outputs) == 0:
            return np.empty(raw_values.shape[:-1] + (0,), dtype=np.float32)
        raw_index = {name: idx for idx, name in enumerate(plan.raw_features)}
        columns = []
        for output in plan.outputs:
            source = raw_values[..., raw_index[output.source]]
            if output.transform == "identity":
                columns.append(source)
            elif output.transform == "sin":
                columns.append(np.sin(source))
            elif output.transform == "cos":
                columns.append(np.cos(source))
            else:
                raise AssertionError(f"Unhandled feature-plan output transform: {output.transform}")
        return np.stack(columns, axis=-1).astype(np.float32, copy=False)

    def _build_node_raw_and_model_arrays(
        self,
        input_h5: h5py._hl.group.Group,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_arrays = []
        model_arrays = []
        for name, uid in self.input_id.items():
            arr = input_h5[name][start:end]
            raw_np = self._structured_raw_feature_array(arr, self.raw_node_feature_names, name)
            model_np = self._apply_feature_plan(raw_np, self.node_feature_plan)
            raw_t = torch.tensor(raw_np, dtype=torch.float32)
            model_t = torch.tensor(model_np, dtype=torch.float32)
            uids = torch.full((model_t.shape[0], model_t.shape[1], 1), uid, dtype=torch.float32)
            raw_arrays.append(raw_t)
            model_arrays.append(torch.cat((model_t, uids), dim=2))
        return torch.cat(raw_arrays, dim=1), torch.cat(model_arrays, dim=1)

    def _build_global_model_array(
        self,
        input_h5: h5py._hl.group.Group,
        start: int,
        end: int,
    ) -> torch.Tensor:
        arr = input_h5["GLOBAL"][start:end]
        raw_np = self._structured_raw_feature_array(arr, self.raw_global_feature_names, "GLOBAL")
        model_np = self._apply_feature_plan(raw_np, self.global_feature_plan)
        return torch.tensor(model_np, dtype=torch.float32)


    def assign_target_ids(self) -> Tuple[List, List]:
        def compute_target_id(label: str) -> int:
            k1, k2 = map(int, label.split('-'))
            return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

        target_edge_ids = [
            [[compute_target_id(label) for label in target] for target in target_group]
            for target_group in self.edge_targets
        ] if self.edge_targets else []

        target_hyperedge_ids = [
            [[compute_target_id(label) for label in target] for target in target_group]
            for target_group in self.hyperedge_targets
        ] if self.hyperedge_targets else []

        return target_edge_ids, target_hyperedge_ids


    def build_node_attributes(self, input_h5: h5py._hl.group.Group, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_chunk, Nobjects_chunk, p4, _ = self.build_node_attributes_with_mask(input_h5, start, end)
        return x_chunk, Nobjects_chunk, p4


    def build_node_attributes_with_mask(
        self,
        input_h5: h5py._hl.group.Group,
        start: int,
        end: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        raw_combined, combined = self._build_node_raw_and_model_arrays(input_h5, start, end)
        remove_nan_mask = torch.isfinite(raw_combined).all(dim=2)
        Nobjects_chunk = torch.count_nonzero(remove_nan_mask, dim=1)
        x_chunk = combined[remove_nan_mask]
        raw_x_chunk = raw_combined[remove_nan_mask]

        # Compute per-node 4-vector (e, px, py, pz) from named features rather
        # than positional assumptions, so sin_phi/cos_phi can remain model inputs.
        if self._use_raw_EEtaPhiPt:
            e_col = raw_x_chunk[:, self._node_raw_feature_index['e']]
            eta = raw_x_chunk[:, self._node_raw_feature_index['eta']]
            phi = raw_x_chunk[:, self._node_raw_feature_index['phi']]
            pt = raw_x_chunk[:, self._node_raw_feature_index['pt']]
            px = pt * torch.cos(phi)
            py = pt * torch.sin(phi)
            pz = pt * torch.sinh(eta)
        else:
            e_col = x_chunk[:, self._node_feature_index['e']]
        if (not self._use_raw_EEtaPhiPt) and self._use_EEtaSinCosPhiPt:
            eta = x_chunk[:, self._node_feature_index['eta']]
            sin_phi = x_chunk[:, self._node_feature_index['sin_phi']]
            cos_phi = x_chunk[:, self._node_feature_index['cos_phi']]
            pt = x_chunk[:, self._node_feature_index['pt']]
            px = pt * cos_phi
            py = pt * sin_phi
            pz = pt * torch.sinh(eta)
        elif (not self._use_raw_EEtaPhiPt) and self._use_EEtaPhiPt:
            eta = x_chunk[:, self._node_feature_index['eta']]
            phi = x_chunk[:, self._node_feature_index['phi']]
            pt = x_chunk[:, self._node_feature_index['pt']]
            px = pt * torch.cos(phi)
            py = pt * torch.sin(phi)
            pz = pt * torch.sinh(eta)
        elif not self._use_raw_EEtaPhiPt:
            px = x_chunk[:, self._node_feature_index['px']]
            py = x_chunk[:, self._node_feature_index['py']]
            pz = x_chunk[:, self._node_feature_index['pz']]

        # Stack into (N,4): (e, px, py, pz)
        p4 = torch.stack([e_col, px, py, pz], dim=1)

        return x_chunk, Nobjects_chunk, p4, remove_nan_mask.numpy()

    def build_global_attributes(self, input_h5: h5py._hl.group.Group, start: int, end: int) -> torch.Tensor:
        u_chunk = self._build_global_model_array(input_h5, start, end)
        return u_chunk.squeeze(0)


    def build_global_targets(self, labels_h5: h5py._hl.group.Group, start: int, end: int) -> torch.Tensor:
        arr = np.asarray(labels_h5["GLOBAL"][start:end])
        if arr.ndim == 1:
            arr = arr[:, None]
        return torch.from_numpy(arr).float()


    def assign_node_ids(self, node_feature_array: torch.Tensor, labels_h5: h5py._hl.group.Group,
                       start: int, end: int, Nobjects_chunk: torch.Tensor) -> Tuple[ak.Array, ak.Array]:
        valid_input_mask_np = self._valid_input_object_mask(start, end)
        cantor_node_ids_tensor = self.assign_node_ids_from_chunk(
            node_feature_array,
            labels_h5,
            start,
            end,
            Nobjects_chunk,
            valid_input_mask_np,
        )

        cantor_node_ids_chunk = ak.unflatten(
            ak.Array(np.asarray(cantor_node_ids_tensor)),
            ak.Array(np.asarray(Nobjects_chunk)),
        )
        local_node_ids_chunk = ak.local_index(cantor_node_ids_chunk)

        return cantor_node_ids_chunk, local_node_ids_chunk


    def assign_node_ids_from_chunk(
        self,
        node_feature_array: torch.Tensor,
        labels_h5: h5py._hl.group.Group,
        start: int,
        end: int,
        Nobjects_chunk: torch.Tensor,
        valid_input_mask_np: np.ndarray,
    ) -> torch.Tensor:
        k1 = node_feature_array[:, -1]

        truth_label_imported = [labels_h5[obj][start:end] for obj in self.input_id.keys()]
        truthlabels_np = np.concatenate(truth_label_imported, axis=1)
        truthlabels = torch.tensor(truthlabels_np, dtype=torch.float32)

        valid_input_mask = torch.from_numpy(valid_input_mask_np)
        valid_truth_mask = ~torch.isnan(truthlabels)

        ignored_padded_truth = valid_truth_mask & ~valid_input_mask
        if ignored_padded_truth.any():
            self._warn_ignored_padded_truth_labels(
                start=start,
                end=end,
                truthlabels=truthlabels,
                valid_input_mask=valid_input_mask,
                ignored_mask=ignored_padded_truth,
            )

        selected_truth_mask = valid_truth_mask & valid_input_mask
        k2 = truthlabels[selected_truth_mask]

        if k1.numel() != k2.numel():
            raise RuntimeError(
                self._format_node_id_mismatch(
                    start=start,
                    end=end,
                    k1=k1,
                    k2=k2,
                    truthlabels=truthlabels,
                    valid_input_mask=valid_input_mask,
                    valid_truth_mask=valid_truth_mask,
                    node_feature_array=node_feature_array,
                    Nobjects_chunk=Nobjects_chunk,
                )
            )

        return self._cantor_pairing(k1, k2)

    def _valid_input_object_mask(self, start: int, end: int) -> np.ndarray:
        inputs = self.file["INPUTS"]
        masks = []
        for obj in self.input_id.keys():
            arr = inputs[obj][start:end]
            obj_mask = np.ones(arr.shape, dtype=bool)
            for field in self.raw_node_feature_names:
                obj_mask &= np.isfinite(arr[field])
            masks.append(obj_mask)
        return np.concatenate(masks, axis=1)

    @staticmethod
    def _truncate_debug(value, max_items: int = 24):
        arr = np.asarray(value).reshape(-1)
        out = arr[:max_items].tolist()
        if arr.size > max_items:
            out.append(f"... ({arr.size - max_items} more)")
        return out

    def _warn_ignored_padded_truth_labels(
        self,
        start: int,
        end: int,
        truthlabels: torch.Tensor,
        valid_input_mask: torch.Tensor,
        ignored_mask: torch.Tensor,
    ) -> None:
        if self._invalid_truth_label_warning_count >= 5:
            return

        ignored_positions = torch.nonzero(ignored_mask, as_tuple=False)
        ignored_values = truthlabels[ignored_mask]
        warn(
            "Ignoring truth labels attached to padded/invalid input objects while "
            f"building event range [{start}, {end}). This keeps node IDs aligned "
            "with graph nodes; reconstruction targets that reference absent objects "
            "will be unmatched for that event. "
            f"ignored_positions={self._truncate_debug(ignored_positions.numpy())}, "
            f"ignored_values={self._truncate_debug(ignored_values.numpy())}, "
            f"valid_input_count={int(valid_input_mask.sum())}, "
            f"valid_truth_count={int((~torch.isnan(truthlabels)).sum())}.",
            UserWarning,
        )
        self._invalid_truth_label_warning_count += 1

    def _format_node_id_mismatch(
        self,
        start: int,
        end: int,
        k1: torch.Tensor,
        k2: torch.Tensor,
        truthlabels: torch.Tensor,
        valid_input_mask: torch.Tensor,
        valid_truth_mask: torch.Tensor,
        node_feature_array: torch.Tensor,
        Nobjects_chunk: torch.Tensor,
    ) -> str:
        missing_truth_mask = valid_input_mask & ~valid_truth_mask
        extra_truth_mask = valid_truth_mask & ~valid_input_mask
        return (
            "Could not align graph nodes with truth labels while assigning node IDs. "
            f"event_range=[{start}, {end}), "
            f"k1_shape={tuple(k1.shape)}, k2_shape={tuple(k2.shape)}, "
            f"k1_values={self._truncate_debug(k1.detach().cpu().numpy())}, "
            f"k2_values={self._truncate_debug(k2.detach().cpu().numpy())}, "
            f"node_feature_shape={tuple(node_feature_array.shape)}, "
            f"Nobjects_chunk={self._truncate_debug(Nobjects_chunk.detach().cpu().numpy())}, "
            f"valid_input_count={int(valid_input_mask.sum())}, "
            f"valid_truth_count={int(valid_truth_mask.sum())}, "
            f"missing_truth_positions={self._truncate_debug(torch.nonzero(missing_truth_mask, as_tuple=False).numpy())}, "
            f"extra_truth_positions={self._truncate_debug(torch.nonzero(extra_truth_mask, as_tuple=False).numpy())}, "
            f"truthlabels={self._truncate_debug(truthlabels.detach().cpu().numpy())}, "
            f"input.nodes={self._graph_config_for_debug['input'].get('nodes')}, "
            f"target={self._graph_config_for_debug['target']}."
        )


    def build_edge_indices(self, local_node_ids_chunk: ak.Array, cantor_node_ids_chunk: ak.Array) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph edge indices.

        The default is ``input.edge_directionality: undirected`` for backwards
        compatibility with existing HyPER caches and reconstruction outputs.
        ``directed`` is available for explicit audits/training tests against the
        VyPER-style convention.
        """

        # === Local node indices ===
        edge_pairs = (
            ak.combinations(local_node_ids_chunk, 2)
            if self.edge_directionality == "undirected"
            else ak.cartesian((local_node_ids_chunk, local_node_ids_chunk), nested=True)
        )
        if self.edge_directionality == "directed":
            edge_pair_indices = ak.argcartesian((local_node_ids_chunk, local_node_ids_chunk), nested=True)
            edge_pairs = edge_pairs[edge_pair_indices["0"] != edge_pair_indices["1"]]
        edge_index_0 = ak.flatten(edge_pairs["0"])
        edge_index_1 = ak.flatten(edge_pairs["1"])
        edge_index_0_torch = torch.tensor(edge_index_0.to_numpy(), dtype=torch.long)
        edge_index_1_torch = torch.tensor(edge_index_1.to_numpy(), dtype=torch.long)
        edge_index_chunk = torch.stack([edge_index_0_torch, edge_index_1_torch], dim=0)

        if cantor_node_ids_chunk is None:
            cantor_edge_index_chunk = torch.empty((2, 0), dtype=torch.long)
        else:
            # === Cantor node indices ===
            cantor_edge_pairs = (
                ak.combinations(cantor_node_ids_chunk, 2)
                if self.edge_directionality == "undirected"
                else ak.cartesian((cantor_node_ids_chunk, cantor_node_ids_chunk), nested=True)
            )
            if self.edge_directionality == "directed":
                cantor_edge_pair_indices = ak.argcartesian((cantor_node_ids_chunk, cantor_node_ids_chunk), nested=True)
                cantor_edge_pairs = cantor_edge_pairs[
                    cantor_edge_pair_indices["0"] != cantor_edge_pair_indices["1"]
                ]
            cantor_edge_index_0 = ak.flatten(cantor_edge_pairs["0"])
            cantor_edge_index_1 = ak.flatten(cantor_edge_pairs["1"])
            cantor_edge_index_0_torch = torch.tensor(cantor_edge_index_0.to_numpy(), dtype=torch.long)
            cantor_edge_index_1_torch = torch.tensor(cantor_edge_index_1.to_numpy(), dtype=torch.long)
            cantor_edge_index_chunk = torch.stack([cantor_edge_index_0_torch, cantor_edge_index_1_torch], dim=0)

        return edge_index_chunk, cantor_edge_index_chunk


    def build_edge_attributes_from_pairs(self, edge_index: torch.Tensor, node_p4: torch.Tensor) -> torch.Tensor:
        """Construct edge attributes from precomputed candidate pairs and p4."""
        num_edges = int(edge_index.shape[1])
        if num_edges == 0 or not self.edge_feature_names:
            return torch.empty((num_edges, len(self.edge_feature_names)), dtype=torch.float32)

        src, dst = edge_index
        p4_1 = node_p4[src]
        p4_2 = node_p4[dst]
        e1, px1, py1, pz1 = p4_1.unbind(dim=1)
        e2, px2, py2, pz2 = p4_2.unbind(dim=1)

        eps = node_p4.new_tensor(1e-8)
        pt1 = torch.sqrt(torch.clamp(px1 * px1 + py1 * py1, min=0.0))
        pt2 = torch.sqrt(torch.clamp(px2 * px2 + py2 * py2, min=0.0))
        eta1 = torch.asinh(pz1 / torch.clamp(pt1, min=float(eps)))
        eta2 = torch.asinh(pz2 / torch.clamp(pt2, min=float(eps)))
        phi1 = torch.atan2(py1, px1)
        phi2 = torch.atan2(py2, px2)

        d_eta_signed = eta1 - eta2
        d_phi = torch.atan2(torch.sin(phi1 - phi2), torch.cos(phi1 - phi2))
        d_r = torch.sqrt(torch.clamp(d_eta_signed * d_eta_signed + d_phi * d_phi, min=0.0))
        e_sum = e1 + e2
        px_sum = px1 + px2
        py_sum = py1 + py2
        pz_sum = pz1 + pz2
        m2 = e_sum * e_sum - px_sum * px_sum - py_sum * py_sum - pz_sum * pz_sum
        min_pt = torch.minimum(pt1, pt2)

        raw_features = {
            "delta_eta": torch.abs(d_eta_signed),
            "delta_phi": d_phi,
            "delta_R": d_r,
            "M": torch.sqrt(torch.clamp(m2, min=0.0)),
            "M2": m2,
            "kT": min_pt * d_r,
            "Z": min_pt / torch.clamp(pt1 + pt2, min=float(eps)),
        }
        edge_attr_chunk = torch.stack([raw_features[name] for name in self.edge_feature_names], dim=1).float()
        for col, name in enumerate(self.edge_feature_names):
            if name in {"M", "M2", "kT", "Z"}:
                edge_attr_chunk[:, col] = torch.clamp(edge_attr_chunk[:, col], min=0.001)
        return edge_attr_chunk


    def _edge_index_template(self, n_nodes: int) -> torch.Tensor:
        n_nodes = int(n_nodes)
        cached = self._edge_template_cache.get(n_nodes)
        if cached is not None:
            return cached
        if n_nodes < 2:
            template = torch.empty((2, 0), dtype=torch.long)
        elif self.edge_directionality == "undirected":
            template = torch.tensor(list(combinations(range(n_nodes), 2)), dtype=torch.long).t().contiguous()
        else:
            template = torch.tensor(list(permutations(range(n_nodes), 2)), dtype=torch.long).t().contiguous()
        self._edge_template_cache[n_nodes] = template
        return template


    def _hyperedge_index_template(self, n_nodes: int, order: int) -> torch.Tensor:
        n_nodes = int(n_nodes)
        order = int(order)
        key = (n_nodes, order)
        cached = self._hyperedge_template_cache.get(key)
        if cached is not None:
            return cached
        if n_nodes < order or order <= 0:
            template = torch.empty((order, 0), dtype=torch.long)
        else:
            template = torch.tensor(list(combinations(range(n_nodes), order)), dtype=torch.long).t().contiguous()
        self._hyperedge_template_cache[key] = template
        return template

    def build_hyperedge_indices(self, local_node_ids_chunk: ak.Array, cantor_node_ids_chunk: ak.Array,
                                hyperedge_cardinality: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hyperedge_node_index_combinations = ak.combinations(local_node_ids_chunk, hyperedge_cardinality)
        hyperedge_index_chunk = self._map_nested_awkward_to_torch(hyperedge_node_index_combinations)

        if cantor_node_ids_chunk is None:
            cantor_hyperedge_index_chunk = torch.empty((hyperedge_cardinality, 0), dtype=torch.long)
        else:
            hyperedge_cantor_index_combinations = ak.combinations(cantor_node_ids_chunk, hyperedge_cardinality)
            cantor_hyperedge_index_chunk = self._map_nested_awkward_to_torch(hyperedge_cantor_index_combinations)

        return hyperedge_index_chunk, cantor_hyperedge_index_chunk

    @staticmethod
    def build_local_node_ids(Nobjects_chunk: torch.Tensor) -> ak.Array:
        counts = np.asarray(Nobjects_chunk.cpu(), dtype=np.int64)
        return ak.unflatten(ak.Array(np.arange(int(counts.sum()), dtype=np.int64)), ak.Array(counts))


    def find_matched_connections(self, connection_input_cantor_tensor: torch.Tensor,
                                target_connection_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        """Assign target 1 or 0 to all connection objects.

        Candidate edges/hyperedges are unordered combinations of graph nodes.
        Truth target definitions are semantic sets, so matching must be
        permutation-invariant.  Sorting both the candidate Cantor IDs and target
        Cantor IDs avoids dropping valid Whad/thad targets when, for example,
        W_HAD_J2 appears before W_HAD_J1 in the input jet order.
        """
        if len(target_connection_ids) == 0:
            n_conn = connection_input_cantor_tensor.shape[1]
            return torch.zeros(n_conn, 1, dtype=torch.float32)

        targets = torch.stack([torch.tensor(t, dtype=connection_input_cantor_tensor.dtype)
                              for t in target_connection_ids])

        connections = connection_input_cantor_tensor.t()
        connections = torch.sort(connections, dim=1).values
        targets = torch.sort(targets, dim=1).values
        matches = (connections.unsqueeze(1) == targets.unsqueeze(0))
        full_matches = matches.all(dim=2)
        output_labels = full_matches.any(dim=1, keepdim=True).float()

        return output_labels

    def _typed_connection_target(
        self,
        connection_input_cantor_tensor: torch.Tensor,
        target_connection_ids: Sequence[Sequence[Sequence[int]]],
        out_channels: int,
        ordered: bool,
    ) -> torch.Tensor:
        n_conn = int(connection_input_cantor_tensor.shape[1])
        out = torch.zeros((n_conn, out_channels), dtype=torch.float32)
        if n_conn == 0:
            return out

        connections = connection_input_cantor_tensor.t()
        if not ordered:
            connections = torch.sort(connections, dim=1).values

        for class_idx, target_group in enumerate(target_connection_ids):
            if not target_group:
                continue
            targets = torch.stack([
                torch.tensor(target, dtype=connections.dtype, device=connections.device)
                for target in target_group
            ])
            if not ordered:
                targets = torch.sort(targets, dim=1).values
            matches = (connections.unsqueeze(1) == targets.unsqueeze(0)).all(dim=2)
            out[matches.any(dim=1), class_idx] = 1.0

        matched = out[:, :-1].sum(dim=1) > 0
        out[~matched, -1] = 1.0

        row_sums = out.sum(dim=1)
        if not torch.all(row_sums == 1):
            bad = torch.nonzero(row_sums != 1, as_tuple=False).flatten()[:10].tolist()
            raise RuntimeError(f"Typed target rows must sum to 1; ambiguous rows={bad}")
        return out

    def build_typed_edge_target(self, cantor_edge_index: torch.Tensor) -> torch.Tensor:
        return self._typed_connection_target(
            cantor_edge_index,
            self.target_edge_ids,
            self.edge_out_channels,
            ordered=self.edge_directionality == "directed",
        )

    def build_typed_hyperedge_target(self, cantor_hyperedge_index: torch.Tensor) -> torch.Tensor:
        return self._typed_connection_target(
            cantor_hyperedge_index,
            self.target_hyperedge_ids,
            self.hyperedge_out_channels,
            ordered=self.hyperedge_ordered,
        )


    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'file') and self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass

    def _ensure_file_open(self):
        """Open HDF5 file handle on-demand (safe for multiprocessing workers)."""
        if getattr(self, 'file', None) is None:
            self.file = h5py.File(self._h5_path, 'r')

    def __getstate__(self):
        """Ensure the HDF5 file handle is not pickled when dataset is sent to workers."""
        state = self.__dict__.copy()
        # Remove file handle from state so workers can open their own
        if 'file' in state:
            state['file'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Ensure file handle starts closed in new process
        self.file = None
