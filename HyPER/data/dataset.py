from typing import Callable, List, Optional, Tuple, Sequence
from warnings import warn
from copy import deepcopy
from os import path as osp
import time
import sys
import os

import math
import h5py
import yaml
import torch
import numpy as np
import awkward as ak
import vector
import numpy.lib.recfunctions as rf

from torch import Tensor
from torch_geometric.data import Data, Dataset

from .transform import TransformFeatures
from .filter import TargetConnectivityFilter
from .edge_features import EDGE_FEATURE_TRANSFORMS
from .transforms import TRANSFORM_REGISTRY


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
    ) -> None:

        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self._train_mode = training

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
        self.node_feature_names = list(parsed_inputs['input']['node_features'])
        self._node_feature_index = {
            name: idx for idx, name in enumerate(self.node_feature_names)
        }
        self._graph_config_for_debug = {
            'input': deepcopy(parsed_inputs['input']),
            'target': deepcopy(parsed_inputs['target']),
        }
        self._invalid_truth_label_warning_count = 0

        self.edge_targets = list(parsed_inputs['target']['edge'].values())
        self.hyperedge_targets = list(parsed_inputs['target']['hyperedge'].values())
        self.hyperedge_order = len(self.hyperedge_targets[0]) if self.hyperedge_targets else 2
        self.target_edge_ids, self.target_hyperedge_ids = self.assign_target_ids()

        # 4-vector setup. The model-facing node feature list may use periodic
        # sin/cos phi inputs, but graph construction still needs an internal phi.
        self._use_EEtaPhiPt = self._has_node_features('e', 'eta', 'phi', 'pt')
        self._use_EEtaSinCosPhiPt = self._has_node_features('e', 'eta', 'sin_phi', 'cos_phi', 'pt')
        self._use_EPxPyPz = self._has_node_features('e', 'px', 'py', 'pz')
        if self._use_EEtaSinCosPhiPt:
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

        # Node/global transforms
        node_transform_names = parsed_inputs['input'].get('node_transforms', [])
        global_transform_names = parsed_inputs['input'].get('global_transforms', [])

        self._node_transforms = [
            TRANSFORM_REGISTRY.get(name, TRANSFORM_REGISTRY['identity'])
            for name in node_transform_names
        ]
        self._global_transforms = [
            TRANSFORM_REGISTRY.get(name, TRANSFORM_REGISTRY['identity'])
            for name in global_transform_names
        ]

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
            self._num_events = len(_f["INPUTS"]["GLOBAL"])
        self.file = None

        # Precompute target sets
        self._target_edge_set = set(tuple(t) for t in self.target_edge_ids) if self.target_edge_ids else set()
        self._target_hyperedge_set = set(tuple(t) for t in self.target_hyperedge_ids) if self.target_hyperedge_ids else set()

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

        # Edge attributes & indices
        edge_attr = self.build_edge_attributes(inputs, idx, idx+1)
        has_targets = self._train_mode and labels is not None
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
                hyperedge_attr_t = self.find_matched_connections(
                    cantor_hyperedge_index, self.target_hyperedge_ids
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
            edge_attr_t = self.find_matched_connections(
                cantor_edge_index, self.target_edge_ids
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
        )


    # === Static helper methods (all fork-compatible) ===

    @staticmethod
    def _parse_config_file(filename):
        with open(filename) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

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


    def assign_target_ids(self) -> Tuple[List, List]:
        def compute_target_id(label: str) -> int:
            k1, k2 = map(int, label.split('-'))
            return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

        target_edge_ids = [
            [compute_target_id(label) for label in target]
            for target in self.edge_targets
        ] if self.edge_targets else []

        target_hyperedge_ids = [
            [compute_target_id(label) for label in target]
            for target in self.hyperedge_targets
        ] if self.hyperedge_targets else []

        return target_edge_ids, target_hyperedge_ids


    def build_node_attributes(self, input_h5: h5py._hl.group.Group, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        object_arrays = []

        for name, uid in self.input_id.items():
            arr = input_h5[name][start:end]
            t = torch.tensor(
                rf.structured_to_unstructured(arr),
                dtype=torch.float32,
            )
            uids = torch.full((t.shape[0], t.shape[1], 1), uid, dtype=torch.float32)
            object_arrays.append(torch.cat((t, uids), dim=2))

        combined = torch.cat(object_arrays, dim=1)
        remove_nan_mask = ~torch.any(combined.isnan(), dim=2)
        Nobjects_chunk = torch.count_nonzero(remove_nan_mask, dim=1)
        x_chunk = combined[remove_nan_mask]

        # Compute per-node 4-vector (e, px, py, pz) from named features rather
        # than positional assumptions, so sin_phi/cos_phi can remain model inputs.
        e_col = x_chunk[:, self._node_feature_index['e']]
        if self._use_EEtaSinCosPhiPt:
            eta = x_chunk[:, self._node_feature_index['eta']]
            sin_phi = x_chunk[:, self._node_feature_index['sin_phi']]
            cos_phi = x_chunk[:, self._node_feature_index['cos_phi']]
            pt = x_chunk[:, self._node_feature_index['pt']]
            px = pt * cos_phi
            py = pt * sin_phi
            pz = pt * torch.sinh(eta)
        elif self._use_EEtaPhiPt:
            eta = x_chunk[:, self._node_feature_index['eta']]
            phi = x_chunk[:, self._node_feature_index['phi']]
            pt = x_chunk[:, self._node_feature_index['pt']]
            px = pt * torch.cos(phi)
            py = pt * torch.sin(phi)
            pz = pt * torch.sinh(eta)
        else:
            px = x_chunk[:, self._node_feature_index['px']]
            py = x_chunk[:, self._node_feature_index['py']]
            pz = x_chunk[:, self._node_feature_index['pz']]

        # Stack into (N,4): (e, px, py, pz)
        p4 = torch.stack([e_col, px, py, pz], dim=1)

        return x_chunk, Nobjects_chunk, p4

    def build_global_attributes(self, input_h5: h5py._hl.group.Group, start: int, end: int) -> torch.Tensor:
        arr = rf.structured_to_unstructured(input_h5["GLOBAL"][start:end])
        u_chunk = torch.tensor(arr, dtype=torch.float32)
        return u_chunk.squeeze(0)


    def build_global_targets(self, labels_h5: h5py._hl.group.Group, start: int, end: int) -> torch.Tensor:
        arr = np.asarray(labels_h5["GLOBAL"][start:end])
        if arr.ndim == 1:
            arr = arr[:, None]
        return torch.from_numpy(arr).float()


    def assign_node_ids(self, node_feature_array: torch.Tensor, labels_h5: h5py._hl.group.Group,
                       start: int, end: int, Nobjects_chunk: torch.Tensor) -> Tuple[ak.Array, ak.Array]:
        k1 = node_feature_array[:, -1]

        truth_label_imported = [labels_h5[obj][start:end] for obj in self.input_id.keys()]
        truthlabels_np = np.concatenate(truth_label_imported, axis=1)
        truthlabels = torch.tensor(truthlabels_np, dtype=torch.float32)

        valid_input_mask_np = self._valid_input_object_mask(start, end)
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

        # Use static method (no lambda!)
        cantor_node_ids_tensor = self._cantor_pairing(k1, k2)

        cantor_node_ids_chunk = ak.unflatten(
            ak.Array(np.asarray(cantor_node_ids_tensor)),
            ak.Array(np.asarray(Nobjects_chunk)),
        )
        local_node_ids_chunk = ak.local_index(cantor_node_ids_chunk)

        return cantor_node_ids_chunk, local_node_ids_chunk

    def _valid_input_object_mask(self, start: int, end: int) -> np.ndarray:
        inputs = self.file["INPUTS"]
        masks = []
        for obj in self.input_id.keys():
            arr = inputs[obj][start:end]
            obj_mask = np.ones(arr.shape, dtype=bool)
            for field in arr.dtype.names:
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
        """Build edge indices for UNDIRECTED graph (i < j only)."""

        # === Local node indices ===
        edge_pairs = ak.combinations(local_node_ids_chunk, 2)
        edge_index_0 = ak.flatten(edge_pairs["0"])
        edge_index_1 = ak.flatten(edge_pairs["1"])
        edge_index_0_torch = torch.tensor(edge_index_0.to_numpy(), dtype=torch.long)
        edge_index_1_torch = torch.tensor(edge_index_1.to_numpy(), dtype=torch.long)
        edge_index_chunk = torch.stack([edge_index_0_torch, edge_index_1_torch], dim=0)

        if cantor_node_ids_chunk is None:
            cantor_edge_index_chunk = torch.empty((2, 0), dtype=torch.long)
        else:
            # === Cantor node indices ===
            cantor_edge_pairs = ak.combinations(cantor_node_ids_chunk, 2)
            cantor_edge_index_0 = ak.flatten(cantor_edge_pairs["0"])
            cantor_edge_index_1 = ak.flatten(cantor_edge_pairs["1"])
            cantor_edge_index_0_torch = torch.tensor(cantor_edge_index_0.to_numpy(), dtype=torch.long)
            cantor_edge_index_1_torch = torch.tensor(cantor_edge_index_1.to_numpy(), dtype=torch.long)
            cantor_edge_index_chunk = torch.stack([cantor_edge_index_0_torch, cantor_edge_index_1_torch], dim=0)

        return edge_index_chunk, cantor_edge_index_chunk


    def build_edge_attributes(self, input_h5: h5py._hl.group.Group, start: int, end: int) -> torch.Tensor:
        """
        Construct edge attributes using awkward arrays (matches build_edge_indices).
        Creates UNDIRECTED edges only (i < j).
        """
        object_vectors_list = []

        for obj in self.input_id.keys():
            obj_arr = input_h5[obj][start:end]

            if self._use_EEtaPhiPt:
                obj_e   = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["e"])))
                obj_pt  = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["pt"])))
                obj_eta = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["eta"])))
                obj_phi = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["phi"])))
                obj_vectors = vector.zip({"pt": obj_pt, "eta": obj_eta, "phi": obj_phi, "e": obj_e})
            elif self._use_EEtaSinCosPhiPt:
                obj_e       = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["e"])))
                obj_pt      = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["pt"])))
                obj_eta     = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["eta"])))
                obj_sin_phi = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["sin_phi"])))
                obj_cos_phi = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["cos_phi"])))
                obj_phi     = np.arctan2(obj_sin_phi, obj_cos_phi)
                obj_vectors = vector.zip({"pt": obj_pt, "eta": obj_eta, "phi": obj_phi, "e": obj_e})
            elif self._use_EPxPyPz:
                obj_e   = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["e"])))
                obj_px  = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["px"])))
                obj_py  = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["py"])))
                obj_pz  = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["pz"])))
                obj_vectors = vector.zip({"px": obj_px, "py": obj_py, "pz": obj_pz, "e": obj_e})
            else:
                raise ValueError("Either _use_EEtaPhiPt or _use_EPxPyPz must be True.")

            object_vectors_list.append(obj_vectors)

        vectors = ak.concatenate(object_vectors_list, axis=1)

        # === Use ak.combinations for undirected edges (i < j), NOT cartesian! ===
        pairs = ak.combinations(vectors, 2)

        # Extract components from pairs
        v1 = pairs["0"]
        v2 = pairs["1"]

        # Compute pairwise features
        # Note: deltaR/deltaeta/deltaphi already return positive values, no abs needed
        # Also: use deltaphi (no underscore), not delta_phi
        DR   = v1.deltaR(v2)
        Deta = v1.eta - v2.eta
        Dphi = v1.deltaphi(v2)  # ← No underscore!
        M    = (v1 + v2).m

        # Remove any NaN/None values
        DR   = ak.drop_none(ak.nan_to_none(DR))
        Deta = ak.drop_none(ak.nan_to_none(Deta))
        Dphi = ak.drop_none(ak.nan_to_none(Dphi))
        M    = ak.drop_none(ak.nan_to_none(M))

        # Convert to numpy via awkward and perform a single conversion to torch
        np_DR = ak.to_numpy(DR)
        np_Deta = ak.to_numpy(Deta)
        np_Dphi = ak.to_numpy(Dphi)
        np_M = ak.to_numpy(M)

        # ak.to_numpy may return arrays with a leading event dimension (1, N).
        # Flatten to 1D per-event arrays for stacking (N,).
        np_DR = np.asarray(np_DR).ravel()
        np_Deta = np.asarray(np_Deta).ravel()
        np_Dphi = np.asarray(np_Dphi).ravel()
        np_M = np.asarray(np_M).ravel()

        # Stack into a single (N,4) numpy array to perform one torch.from_numpy call
        if np_DR.size == 0:
            # No edges in this event
            return torch.empty((0, 4), dtype=torch.float32)

        stacked = np.stack([
            np.abs(np_Deta).astype(np.float32),
            np_Dphi.astype(np.float32),
            np.abs(np_DR).astype(np.float32),
            np_M.astype(np.float32),
        ], axis=1)

        edge_attr_chunk = torch.from_numpy(stacked)
        # Clamp mass column to avoid zeros/very small values
        edge_attr_chunk[:, 3] = torch.clamp(edge_attr_chunk[:, 3], min=0.001)

        return edge_attr_chunk

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
        """Vectorized version - assign target 1 or 0 to all connection objects."""
        if len(target_connection_ids) == 0:
            n_conn = connection_input_cantor_tensor.shape[1]
            return torch.zeros(n_conn, 1, dtype=torch.float32)

        targets = torch.stack([torch.tensor(t, dtype=connection_input_cantor_tensor.dtype)
                              for t in target_connection_ids])

        connections = connection_input_cantor_tensor.t()
        matches = (connections.unsqueeze(1) == targets.unsqueeze(0))
        full_matches = matches.all(dim=2)
        output_labels = full_matches.any(dim=1, keepdim=True).float()

        return output_labels


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
