# HyPER/data/dataset_lazy.py
from typing import Callable, List, Optional, Tuple, Dict, Sequence
from warnings import warn
from os import listdir, path as osp, makedirs
import shutil

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


class HyPERDatasetLazy(Dataset):
    """Lazy-loading HyPERDataset with picklable everything."""
    
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        
        # Caching setup
        self._cache_dir = cache_dir or osp.join(root, ".lazy_cache", name)
        if force_reload and osp.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir)
        makedirs(self._cache_dir, exist_ok=True)
        
        # Parse config
        parsed_inputs = self._parse_config_file(f"{self.root}/config.yaml")
        self.node_input_names = list(parsed_inputs['input']['nodes'].keys())
        self.input_id = parsed_inputs['input']['nodes']
        self.input_pad_size = parsed_inputs['input']['padding']
        
        self.edge_targets = list(parsed_inputs['target']['edge'].values())
        self.hyperedge_targets = list(parsed_inputs['target']['hyperedge'].values())
        self.hyperedge_order = len(self.hyperedge_targets[0]) if self.hyperedge_targets else 2
        self.target_edge_ids, self.target_hyperedge_ids = self.assign_target_ids()
        
        # 4-vector setup
        self._use_EEtaPhiPt = parsed_inputs['input']['node_features'][:4] == ['e','eta','phi','pt']
        self._use_EPxPyPz = parsed_inputs['input']['node_features'][:4] == ['e','px','py','pz']
        if not self._use_EEtaPhiPt and not self._use_EPxPyPz:
            warn("Non-standard 4-vector features", UserWarning)
            self._use_EPxPyPz = True

        # Edge features (picklable)
        self.edge_features_to_use = self.parse_edge_features(parsed_inputs)
        
        # Node/global transforms (picklable)
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
        
        # Filter
        if 'filter' in parsed_inputs:
            self._pre_filter = TargetConnectivityFilter(
                num_edge_targets=parsed_inputs['filter']['num_edges'],
                num_hyperedge_targets=parsed_inputs['filter']['num_hyperedge']
            )
        else:
            self._pre_filter = None

        # STORE PATH ONLY (not file handle!)
        self._h5_path = osp.join(root, "raw", f"{name}.h5")
        self._num_events = self._get_num_events()
        
        # Precompute target sets
        self._target_edge_set = set(tuple(t) for t in self.target_edge_ids) if self.target_edge_ids else set()
        self._target_hyperedge_set = set(tuple(t) for t in self.target_hyperedge_ids) if self.target_hyperedge_ids else set()
    
    
    def _get_num_events(self) -> int:
        """Get number of events without keeping file handle open."""
        with h5py.File(self._h5_path, "r") as f:
            return len(f["INPUTS"]["GLOBAL"])
    
    
    def __len__(self) -> int:
        return self._num_events
    
    
    def __getitem__(self, idx: int) -> Data:
        """Load/process single event on-demand"""
        
        # Check cache first
        cache_path = osp.join(self._cache_dir, f"event_{idx}.pt")
        if osp.exists(cache_path):
            try:
                data = torch.load(cache_path)
                if self.transform:
                    data = self.transform(data)
                return data
            except:
                pass
        
        # OPEN HDF5 HERE (each worker gets its own handle)
        with h5py.File(self._h5_path, "r") as h5_file:
            data = self._process_single_event(idx, h5_file)
        
        # Apply filters/transforms
        if self._pre_filter and not self._pre_filter(data):
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            data = self.transform(data)
        
        # Cache for next time
        torch.save(data, cache_path)
        
        return data
    
    
    def _process_single_event(self, idx: int, h5_file: h5py.File) -> Data:
        """Extract and build one event from HDF5."""
        
        inputs = h5_file["INPUTS"]
        labels = h5_file["LABELS"]
        
        # Node attributes
        x, Nobjects = self.build_node_attributes(inputs, idx, idx+1)
        
        # Edge attributes & indices
        edge_attr = self.build_edge_attributes(inputs, idx, idx+1)
        cantor_node_ids, local_node_ids = self.assign_node_ids(
            x, labels, idx, idx+1, Nobjects
        )
        edge_index, cantor_edge_index = self.build_edge_indices(
            local_node_ids, cantor_node_ids
        )
        
        # Hyperedges (if used)
        if self.hyperedge_targets:
            hyperedge_index, cantor_hyperedge_index = self.build_hyperedge_indices(
                local_node_ids, cantor_node_ids, self.hyperedge_order
            )
            hyperedge_attr_t = self.find_matched_connections(
                cantor_hyperedge_index, self.target_hyperedge_ids
            )
        else:
            hyperedge_index = torch.empty((2, 0), dtype=torch.long)
            hyperedge_attr_t = torch.empty((0, 1), dtype=torch.float32)
        
        # Global attributes & targets
        u = self.build_global_attributes(inputs, idx, idx+1)
        cls_t = self.build_global_targets(labels, idx, idx+1)
        
        # Edge targets
        edge_attr_t = self.find_matched_connections(
            cantor_edge_index, self.target_edge_ids
        )
        
        # Flatten node IDs
        cantor_node_ids_flat = torch.tensor(
            np.asarray(ak.flatten(cantor_node_ids)), dtype=torch.float32
        )
        
        # Ensure proper shapes for transforms
        if u.ndim == 1:
            u = u.unsqueeze(0)
        if cls_t.ndim == 1:
            cls_t = cls_t.unsqueeze(0)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u=u,
            cls_t=cls_t,
            node_ids=cantor_node_ids_flat,
            edge_attr_t=edge_attr_t,
            hyperedge_index=hyperedge_index,
            hyperedge_attr_t=hyperedge_attr_t,
        )
    
    
    # === Static helper methods (all picklable) ===
    
    @staticmethod
    def _parse_config_file(filename):
        with open(filename) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
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

        # Convert to numpy
        np_array = flat.to_numpy()

        # Check if it's a structured array (has named fields)
        if np_array.dtype.names is not None:
            # Structured array: convert to unstructured, then transpose
            unstruct = rf.structured_to_unstructured(np_array)
            return torch.as_tensor(unstruct, dtype=torch.long).t()  # [F, N] -> indices as long
        else:
            # Regular float array: just reshape to column vector
            return torch.as_tensor(np_array, dtype=torch.float32).reshape(-1, 1)  # [N, 1]
    
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
    

    def build_node_attributes(self, input_h5: h5py._hl.group.Group, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return x_chunk, Nobjects_chunk


    def build_edge_attributes(self, input_h5: h5py._hl.group.Group, start: int, end: int) -> torch.Tensor:
        object_vectors_list = []

        for obj in self.input_id.keys():
            obj_arr = input_h5[obj][start:end]

            if self._use_EEtaPhiPt:
                obj_e   = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["e"])))
                obj_pt  = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["pt"])))
                obj_eta = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["eta"])))
                obj_phi = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["phi"])))
                obj_vectors = vector.zip({"pt": obj_pt, "eta": obj_eta, "phi": obj_phi, "e": obj_e})
            elif self._use_EPxPyPz:
                obj_e   = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["e"])))
                obj_px  = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["px"])))
                obj_py  = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["py"])))
                obj_pz = ak.drop_none(ak.nan_to_none(ak.Array(obj_arr["pz"])))
                obj_vectors = vector.zip({"px": obj_px, "py": obj_py, "pz": obj_pz, "e": obj_e})
            else:
                raise ValueError("Either _use_EEtaPhiPt or _use_EPxPyPz must be True.")

            object_vectors_list.append(obj_vectors)

        vectors = ak.concatenate(object_vectors_list, axis=1)

        DR   = vectors[:, None].deltaR(vectors)
        Deta = vectors[:, None].deltaeta(vectors)
        Dphi = vectors[:, None].deltaphi(vectors)
        M    = (vectors[:, None] + vectors).m

        itself_mask = DR == 0.0

        # Use static methods (no lambdas!)
        torch_DR   = self._map_nested_awkward_to_torch(self._remove_self_pairing(DR, itself_mask))
        torch_Deta = self._map_nested_awkward_to_torch(self._remove_self_pairing(Deta, itself_mask))
        torch_Dphi = self._map_nested_awkward_to_torch(self._remove_self_pairing(Dphi, itself_mask))
        torch_M    = self._map_nested_awkward_to_torch(self._remove_self_pairing(M, itself_mask))

        torch_M = torch.clamp(torch_M, 0.001)

        edge_attr_chunk = torch.cat((torch_Deta, torch_Dphi, torch_DR, torch_M), dim=1)
        return edge_attr_chunk


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

        remove_nan_mask_2 = ~torch.isnan(truthlabels)
        k2 = truthlabels[remove_nan_mask_2]

        # Use static method (no lambda!)
        cantor_node_ids_tensor = self._cantor_pairing(k1, k2)

        cantor_node_ids_chunk = ak.unflatten(
            ak.Array(np.asarray(cantor_node_ids_tensor)),
            ak.Array(np.asarray(Nobjects_chunk)),
        )
        local_node_ids_chunk = ak.local_index(cantor_node_ids_chunk)

        return cantor_node_ids_chunk, local_node_ids_chunk


    def build_edge_indices(self, local_node_ids_chunk: ak.Array, cantor_node_ids_chunk: ak.Array) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_pairs = self._awkward_nondiag_cartesian(local_node_ids_chunk)
        edge_pairs_1flat = ak.flatten(edge_pairs)
        edge_index_chunk = self._map_nested_awkward_to_torch(edge_pairs_1flat)

        cantor_edge_pairs = self._awkward_nondiag_cartesian(cantor_node_ids_chunk)
        cantor_edge_pairs_1flat = ak.flatten(cantor_edge_pairs)
        cantor_edge_index_chunk = self._map_nested_awkward_to_torch(cantor_edge_pairs_1flat)

        return edge_index_chunk, cantor_edge_index_chunk


    def build_hyperedge_indices(self, local_node_ids_chunk: ak.Array, cantor_node_ids_chunk: ak.Array,
                                hyperedge_cardinality: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hyperedge_node_index_combinations = ak.combinations(local_node_ids_chunk, hyperedge_cardinality)
        hyperedge_index_chunk = self._map_nested_awkward_to_torch(hyperedge_node_index_combinations)

        hyperedge_cantor_index_combinations = ak.combinations(cantor_node_ids_chunk, hyperedge_cardinality)
        cantor_hyperedge_index_chunk = self._map_nested_awkward_to_torch(hyperedge_cantor_index_combinations)

        return hyperedge_index_chunk, cantor_hyperedge_index_chunk


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