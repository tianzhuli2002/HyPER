# HyPER/data/dataset_lazy.py
from typing import Callable, List, Optional, Tuple, Dict, Sequence
from warnings import warn
from os import listdir, path as osp, makedirs
import shutil
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


class HyPERDatasetLazy(Dataset):
    """Lazy-loading HyPERDataset in VyPER style (fork-compatible)."""
    
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
            print(f"[DEBUG] HyPERDatasetLazy created: {self.name}, {self._num_events} events, PID={os.getpid()}", 
                  file=sys.stderr, flush=True)
    
    
    def __len__(self) -> int:
        return self._num_events
    
    
    def __getitem__(self, idx: int) -> Data:
        """Load/process single event on-demand - VYPER STYLE"""
        
        # === DEBUG: Progress logging (every 1000 events) ===
        if os.getenv("HYPER_DEBUG", "0") == "1" and idx % 1000 == 0:
            t0 = time.time()
            print(f"[CACHE] Event {idx}/{len(self)} - PID={os.getpid()} - Time={t0:.0f}", 
                  file=sys.stderr, flush=True)
        # ================================================
        
        # Check cache first (VyPER pattern) - cache stores already-processed Data
        cache_path = osp.join(self._cache_dir, f"processed_{idx}.pt")
        if osp.exists(cache_path):
            try:
                # Load and return cached processed Data directly (do NOT re-apply transforms)
                data = torch.load(cache_path)
                return data
            except Exception as e:
                # === DEBUG: Log cache load failures and continue processing ===
                if os.getenv("HYPER_DEBUG", "0") == "1":
                    print(f"[CACHE] Load failed for {idx}: {e}", file=sys.stderr, flush=True)
                # fall through to reprocess
                pass
        
        # === VYPER-STYLE: Call processing() which uses self.file ===
        t0 = time.time()
        data = self.processing(idx)
        t_process = time.time() - t0
        
        # Apply filters/transforms
        if self._pre_filter and not self._pre_filter(data):
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            data = self.transform(data)
        
        # Cache for next time (save processed/transformed Data)
        t_save = time.time()
        try:
            torch.save(data, cache_path)
        except Exception:
            # Best-effort caching; ignore failures
            if os.getenv("HYPER_DEBUG", "0") == "1":
                print(f"[CACHE] Save failed for {idx}", file=sys.stderr, flush=True)
        t_save = time.time() - t_save
        
        # === DEBUG: Log processing time ===
        if os.getenv("HYPER_DEBUG", "0") == "1" and idx % 1000 == 0:
            t_total = time.time() - t0
            print(f"[CACHE] Processed {idx}: process={t_process:.2f}s, save={t_save:.2f}s, total={t_total:.2f}s", 
                  file=sys.stderr, flush=True)
        
        return data
    
    
    def processing(self, idx: int) -> Data:
        """
        Extract and build one event from HDF5 using self.file.
        VyPER-style: uses the file handle opened in __init__.
        """
        # Ensure file handle is opened in this process/worker
        self._ensure_file_open()
        inputs = self.file["INPUTS"]
        labels = self.file["LABELS"]
        
        # Node attributes (also returns precomputed per-node 4-vectors)
        x, Nobjects, node_p4 = self.build_node_attributes(inputs, idx, idx+1)
        
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

        # Compute per-node 4-vector (e, px, py, pz) and mass to cache/precompute
        # Assume first four entries of features correspond to either
        # [e, eta, phi, pt] or [e, px, py, pz] depending on config flags.
        if x_chunk.size(1) >= 4:
            e_col = x_chunk[:, 0]
            a1 = x_chunk[:, 1]
            a2 = x_chunk[:, 2]
            a3 = x_chunk[:, 3]
            if self._use_EEtaPhiPt:
                eta = a1
                phi = a2
                pt = a3
                px = pt * torch.cos(phi)
                py = pt * torch.sin(phi)
                pz = pt * torch.sinh(eta)
            else:
                # EPxPyPz
                px = a1
                py = a2
                pz = a3

            # Stack into (N,4): (e, px, py, pz)
            p4 = torch.stack([e_col, px, py, pz], dim=1)
            # mass = sqrt(max(e^2 - p^2, eps))
            p_sq = px * px + py * py + pz * pz
            mass = torch.sqrt(torch.clamp(e_col * e_col - p_sq, min=1e-6))
            # Optionally attach mass as a separate column if needed downstream
            # Return node features, counts, and p4
        else:
            p4 = torch.empty((x_chunk.shape[0], 4), dtype=torch.float32)

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
        """Build edge indices for UNDIRECTED graph (i < j only)."""
        
        # === Local node indices ===
        edge_pairs = ak.combinations(local_node_ids_chunk, 2)
        edge_index_0 = ak.flatten(edge_pairs["0"])
        edge_index_1 = ak.flatten(edge_pairs["1"])
        edge_index_0_torch = torch.tensor(edge_index_0.to_numpy(), dtype=torch.long)
        edge_index_1_torch = torch.tensor(edge_index_1.to_numpy(), dtype=torch.long)
        edge_index_chunk = torch.stack([edge_index_0_torch, edge_index_1_torch], dim=0)

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
            np.abs(np_Dphi).astype(np.float32),
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