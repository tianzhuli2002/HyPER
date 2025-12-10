from typing import Callable, List, Optional, Tuple, Dict, Sequence
from warnings import warn
from os import listdir, path as osp

import math
import h5py
import yaml
import torch
import numpy as np
import awkward as ak
import vector
import numpy.lib.recfunctions as rf 

from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs

from .transform import TransformFeatures
from .filter import TargetConnectivityFilter

class HyPERDataset(InMemoryDataset):
    
    """
    Builds the graph dataset. Inherits from the PyTorchGeometric InMemoryDataset,
        which comes with __init__ and process methods.
        
    Process is called when the data is loaded. 
    It calls the various methods for building the nodes, edge and global parameters
    
    """
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        
        self.root = root
        self.name = name # This is the name of the file to be loaded
        # self.names is a list of all names in the directory
        
        self.names = [
            osp.splitext(file)[0] for file in 
            listdir(osp.join(self.root, "raw"))]
        # Filter self.names to only include the files that match input_name
        self.names = [name for name in self.names if name == self.name]
        # Throw an error if no file matching name exists
        if len(self.names) == 0:
            raise FileNotFoundError(f"No file matching '{self.name}' found in the 'raw' directory.")
        
        file_index = [i for i in range(len(self.names))
                            if self.names[i] == self.name]
        assert len(file_index) == 1
        self.file_index = file_index[0]

        # Parse database config file, setting instance attributes
        parsed_inputs = HyPERDataset._parse_config_file(f"{self.root}/config.yaml")
        
        self.node_input_names   = list(parsed_inputs['input']['nodes'].keys())
        self.input_id           = parsed_inputs['input']['nodes']
        self.input_pad_size     = parsed_inputs['input']['padding']
        
        self.edge_targets       = list(parsed_inputs['target']['edge'].values())
        self.hyperedge_targets  = list(parsed_inputs['target']['hyperedge'].values())
        self.hyperedge_order = len(self.hyperedge_targets[0])
        self.target_edge_ids, self.target_hyperedge_ids = self.assign_target_ids()
    
        # Check 4-momentum inputs
        self._use_EEtaPhiPt = False
        self._use_EPxPyPz = False
        if parsed_inputs['input']['node_features'][:4] == ['e','eta','phi','pt']: # Make this more flexible
            self._use_EEtaPhiPt = True
        elif parsed_inputs['input']['node_features'][:4] == ['e','px','py','pz']:
            self._use_EPxPyPz = True
        else:
            warn("You are not using the standard feature ordering " \
                "or the naming scheme: ['e', 'eta', 'phi', 'pt'] or "\
                "['e', 'px', 'py', 'pz'] (for the first 4 features). "\
                "This might cause problems in the edge construction stage.", 
                UserWarning)
            self._use_EPxPyPz = True

        # Parse edge features, returning dict
        self.edge_features_to_use = self.parse_edge_features(parsed_inputs)
        
        node_transforms   = [eval(f"lambda x: {f}") for f in parsed_inputs['input']['node_transforms']]
        global_transforms = [eval(f"lambda x: {f}") for f in parsed_inputs['input']['global_transforms']]

        transforms = TransformFeatures(["x", "u", "edge_attr"],
            transforms=[
                node_transforms,
                global_transforms,
                list(self.edge_features_to_use.values())
            ])
        
        if parsed_inputs['input']['pre_transform']:
            print("`pre_transform` is turned on.")
            pre_transform = transforms
        else:
            transform = transforms 

        # Check if filter is requested
        if 'filter' in parsed_inputs.keys():
            print("`pre_filter` is turned on.")
            pre_filter = TargetConnectivityFilter(
                num_edge_targets=parsed_inputs['filter']['num_edges'],
                num_hyperedge_targets=parsed_inputs['filter']['num_hyperedge'])

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)
        self.load(self.processed_paths[self.file_index])
    
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{name}.h5' for name in self.names]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{name}.pt' for name in self.names]
    
    
    @staticmethod
    def _parse_config_file(filename):
        
        """Parses YAML config"""
        with open(filename) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    @staticmethod
    def _awkward_nondiag_cartesian(arr: ak.Array) -> ak.Array:
        
        """
        Performs cartesian product on an awkward.Array with itself (arr x arr), 
            dropping the diagonal components
        Parameters:
        - ak.Array
        Returns:
        - ak.Array
        """
        tmp       = ak.cartesian((arr,arr),nested=True)     # Compute the standard cartesian product
        tmp_index = ak.argcartesian((arr,arr),nested=True)  # Compute the cartesian product of the indices
        return tmp[tmp_index["0"]!=tmp_index["1"]]          # Return a filtered object where matching indices are dropped
    
    @staticmethod
    def _map_nested_awkward_to_torch(arr:ak.Array) -> torch.tensor:
        
        """
        Takes an ak.Array which is singly-ragged and converts to torch.tensor column required
        [Four instances of this use within the build_edge_indices and build_hyperedge_indices methods]
        """
        # Flatten the array into 1D
        flat = ak.flatten(arr)
        # Convert the result to an unstructured numpy array of shape 
        unstruct_numpy_array = rf.structured_to_unstructured(flat.to_numpy())
        # Convert the numpt array to a tensor of shape Nx1, where .t() takes the transpose
        return torch.as_tensor(unstruct_numpy_array).t()      
    
    
    def parse_edge_features(self,parsed_inputs):
        
        """
        User selects from set of pre-defined edge features
        Must select the appropriate transforms too
        """
        all_edge_feature_names = {
            "delta_eta": lambda x: x,
            "delta_phi": lambda x: x,
            "delta_R"  : lambda x: x,
            "kT"       : lambda x: torch.log(x),
            "Z"        : lambda x: torch.log(x),
            "M2"       : lambda x: torch.log(x)
        }

        if "edge_features" in parsed_inputs["input"]:
            requested_features  = parsed_inputs["input"]["edge_features"]
            HyPER_edge_features = list(all_edge_feature_names.keys())
            
            # Check if there are any requested features not in the known edge features
            if len(set(requested_features) - set(HyPER_edge_features)) != 0:
                # warn("Edge feature specified which is not in known HyPER edge features", UserWarning)
                raise KeyError("Edge features have been specified which do not match the pre-defined attributes")
            # Create a dictionary containing only the requested edge features
            edge_features_to_use = {k: all_edge_feature_names[k] for k in requested_features}
        else:
            # If no specific edge features are requested, use all available edge features
            edge_features_to_use = all_edge_feature_names
            
        return edge_features_to_use 

    def assign_target_ids(self) -> Tuple[List, List]:
        """
        Assign each edge/hyperedge target with a unique ID.
        
        Uses the Cantor function to define each ID, 
        then parses the input edge and hyperedge targets to define target_edge_ids and target_hyperedge_ids
        """
        def compute_target_id(label: str) -> int:
            k1, k2 = map(int, label.split('-'))
            return (k1 + k2) * (k1 + k2 + 1) // 2 + k2

        # Compute target edge IDs
        target_edge_ids = [
            [compute_target_id(label) for label in target]
            for target in self.edge_targets
        ] if self.edge_targets else []

        # Compute target hyperedge IDs
        target_hyperedge_ids = [
            [compute_target_id(label) for label in target]
            for target in self.hyperedge_targets
        ] if self.hyperedge_targets else []

        return target_edge_ids, target_hyperedge_ids
    
    def build_node_attributes(self, input_h5: h5py._hl.group.Group, start: int, end: int,) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs node attribute tensor 'x' for events [start:end] from INPUTS h5 group.

        Returns:
            x_chunk      : (N_nodes_chunk, F+1) features (last column = object uid)
            Nobjects_chunk : (B_chunk,) number of nodes per event in this chunk
        """
        object_arrays = []

        for name, uid in self.input_id.items():
            # Slice only events [start:end]
            arr = input_h5[name][start:end]
            t = torch.tensor(
                rf.structured_to_unstructured(arr),
                dtype=torch.float32,
            )  # shape: [B_chunk, Nmax_obj, F_obj]
            uids = torch.full((t.shape[0], t.shape[1], 1), uid, dtype=torch.float32)
            object_arrays.append(torch.cat((t, uids), dim=2))

        # Concatenate different object types along node axis
        combined = torch.cat(object_arrays, dim=1)  # [B_chunk, Nnodes_max_total, F+1]

        # Remove padded entries (NaNs) per event
        remove_nan_mask = ~torch.any(combined.isnan(), dim=2)  # [B_chunk, Nnodes_max_total]
        Nobjects_chunk = torch.count_nonzero(remove_nan_mask, dim=1)  # [B_chunk]

        # Flatten over events/nodes, keep features
        x_chunk = combined[remove_nan_mask]  # [N_nodes_chunk, F+1]

        return x_chunk, Nobjects_chunk
    
    def build_edge_attributes(
        self,
        input_h5: h5py._hl.group.Group,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """
        Constructs edge attribute tensor for events [start:end] from INPUTS group.

        Returns:
            edge_attr_chunk: (E_chunk, 4) with columns [Δη, Δφ, ΔR, M]
        """

        object_vectors_list = []

        for obj in self.input_id.keys():
            # Slice events [start:end]
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

        # Concatenate different object types along the node axis
        vectors = ak.concatenate(object_vectors_list, axis=1)

        remove_self_pairing = lambda arr, mask: ak.drop_none(
            ak.nan_to_none(ak.where(mask, np.nan, arr))
        )
        map_awk_to_torch = lambda arr: torch.tensor(
            ak.flatten(ak.flatten(arr)).to_numpy(), dtype=torch.float32
        ).reshape(-1, 1)

        DR   = vectors[:, None].deltaR(vectors)
        Deta = vectors[:, None].deltaeta(vectors)
        Dphi = vectors[:, None].deltaphi(vectors)
        M    = (vectors[:, None] + vectors).m

        # Mask out self-pairings (jet1 with jet1, etc.)
        itself_mask = DR == 0.0

        torch_DR   = map_awk_to_torch(remove_self_pairing(DR, itself_mask))
        torch_Deta = map_awk_to_torch(remove_self_pairing(Deta, itself_mask))
        torch_Dphi = map_awk_to_torch(remove_self_pairing(Dphi, itself_mask))
        torch_M    = map_awk_to_torch(remove_self_pairing(M, itself_mask))

        torch_M = torch.clamp(torch_M, 0.001)  # below zero masses are unphysical anyway

        edge_attr_chunk = torch.cat((torch_Deta, torch_Dphi, torch_DR, torch_M), dim=1)
        return edge_attr_chunk

    def build_global_attributes(
        self,
        input_h5: h5py._hl.group.Group,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """
        Global per-event attributes for events [start:end].
        """
        arr = rf.structured_to_unstructured(input_h5["GLOBAL"][start:end])
        u_chunk = torch.tensor(arr, dtype=torch.float32)  # shape [B_chunk, 1, F] or [B_chunk, F]
        return u_chunk.squeeze(1)

    def build_global_targets(
        self,
        labels_h5: h5py._hl.group.Group,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """
        Global targets (labels) for events [start:end].
        """
        arr = np.asarray(labels_h5["GLOBAL"][start:end])  # e.g. shape [B_chunk] or [B_chunk, 1]
        if arr.ndim == 1:
            arr = arr[:, None]  # -> [B_chunk, 1]
        return torch.from_numpy(arr).float()

    def assign_node_ids(
        self,
        node_feature_array: torch.Tensor,
        labels_h5: h5py._hl.group.Group,
        start: int,
        end: int,
        Nobjects_chunk: torch.Tensor,
    ) -> Tuple[ak.Array, ak.Array]:
        """
        Create awkward arrays of local and Cantor node IDs for events [start:end].

        Returns:
            cantor_node_ids_chunk : ak.Array of shape [B_chunk, N_i]
            local_node_ids_chunk  : ak.Array of same shape, local indices 0..N_i-1
        """

        # k1: object type from last feature column
        k1 = node_feature_array[:, -1]

        # Extract the matching index (truth label) for each object type, for [start:end]
        truth_label_imported = [labels_h5[obj][start:end] for obj in self.input_id.keys()]
        truthlabels_np = np.concatenate(truth_label_imported, axis=1)  # [B_chunk, Nmax_nodes]
        truthlabels = torch.tensor(truthlabels_np, dtype=torch.float32)

        # Remove padded nodes (NaNs)
        remove_nan_mask_2 = ~torch.isnan(truthlabels)  # [B_chunk, Nmax_nodes]
        k2 = truthlabels[remove_nan_mask_2]            # -> [N_nodes_chunk]

        # Cantor pairing function
        cantor_pairing = lambda a, b: (a + b) * (a + b + 1) / 2 + b
        cantor_node_ids_tensor = cantor_pairing(k1, k2)  # [N_nodes_chunk]

        # Re-cast as awkward array split by event using Nobjects_chunk
        cantor_node_ids_chunk = ak.unflatten(
            ak.Array(np.asarray(cantor_node_ids_tensor)),
            ak.Array(np.asarray(Nobjects_chunk)),
        )

        # Local integer index per node in each event
        local_node_ids_chunk = ak.local_index(cantor_node_ids_chunk)

        return cantor_node_ids_chunk, local_node_ids_chunk

    def build_edge_indices(
        self,
        local_node_ids_chunk: ak.Array,
        cantor_node_ids_chunk: ak.Array,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute all edge (2-node) combinations for this chunk.

        Returns:
            edge_index_chunk        : torch.Tensor of shape (2, E_chunk) local node indices
            cantor_edge_index_chunk : torch.Tensor of shape (2, E_chunk) Cantor node IDs
        """

        # Local node indices
        edge_pairs = HyPERDataset._awkward_nondiag_cartesian(local_node_ids_chunk)
        edge_pairs_1flat = ak.flatten(edge_pairs)
        edge_index_chunk = HyPERDataset._map_nested_awkward_to_torch(edge_pairs_1flat)

        # Cantor node IDs
        cantor_edge_pairs = HyPERDataset._awkward_nondiag_cartesian(cantor_node_ids_chunk)
        cantor_edge_pairs_1flat = ak.flatten(cantor_edge_pairs)
        cantor_edge_index_chunk = HyPERDataset._map_nested_awkward_to_torch(cantor_edge_pairs_1flat)

        return edge_index_chunk, cantor_edge_index_chunk


    def build_hyperedge_indices(
        self,
        local_node_ids_chunk: ak.Array,
        cantor_node_ids_chunk: ak.Array,
        hyperedge_cardinality: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hyperedges (H-node combinations) for this chunk.

        Returns:
            hyperedge_index_chunk        : torch.Tensor of shape (H, E_H_chunk) local node indices
            cantor_hyperedge_index_chunk : torch.Tensor of shape (H, E_H_chunk) Cantor node IDs
        """

        # Local indices
        hyperedge_node_index_combinations = ak.combinations(
            local_node_ids_chunk, hyperedge_cardinality
        )
        hyperedge_index_chunk = HyPERDataset._map_nested_awkward_to_torch(
            hyperedge_node_index_combinations
        )

        # Cantor IDs
        hyperedge_cantor_index_combinations = ak.combinations(
            cantor_node_ids_chunk, hyperedge_cardinality
        )
        cantor_hyperedge_index_chunk = HyPERDataset._map_nested_awkward_to_torch(
            hyperedge_cantor_index_combinations
        )

        return hyperedge_index_chunk, cantor_hyperedge_index_chunk


    def find_matched_connections(
        self,
        connection_input_cantor_tensor: torch.Tensor,
        target_connection_ids: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """
        Assign target 1 or 0 to all connection objects (edges / hyperedges separately).

        Parameters:
            connection_input_cantor_tensor: shape (K, N_conn_chunk) with K=2 for edges, K=H for hyperedges
            target_connection_ids         : iterable of lists of Cantor IDs (for matched connections)

        Returns:
            torch.Tensor of shape (N_conn_chunk, 1), each element 0 or 1
        """

        # N_conn_chunk is second dimension
        n_conn = connection_input_cantor_tensor.shape[1]
        output_labels = torch.zeros(n_conn, 1, dtype=torch.float32)

        for target in target_connection_ids:
            # target is a list of Cantor IDs for this connection
            target_tensor = torch.tensor(target, dtype=connection_input_cantor_tensor.dtype)
            eid = torch.isin(connection_input_cantor_tensor, target_tensor)
            output_labels += 1.0 * torch.all(eid, dim=0).reshape(-1, 1)

        return output_labels


    def nk_comb(self, n: torch.Tensor, k: int) -> torch.Tensor:
        """
        Elementwise binomial coefficient C(n, k) for integer tensor n and scalar k.
        Works for any integer k >= 0, on CPU or GPU.
        """

        if k == 0:
            return torch.ones_like(n)

        result = torch.ones_like(n)
        for i in range(1, k + 1):
            result = result * (n - (k - i))
            result = result // i  # exact integer division

        return result

    def generate_slices(self, Nobjects: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes indices that demarcate separate events, based on Nobjects.

        Nobjects: tensor of shape [B_total] with per-event multiplicities.
        """

        slice_x_index         = torch.cat((torch.tensor([0]), torch.cumsum(Nobjects, dim=0)))
        slice_u_index         = torch.arange(0, len(Nobjects) + 1)
        slice_edge_index      = torch.cat((torch.tensor([0]), torch.cumsum(2 * self.nk_comb(Nobjects,2), dim=0)))
        slice_hyperedge_index = torch.cat((torch.tensor([0]), torch.cumsum(self.nk_comb(Nobjects,self.hyperedge_order), dim=0)))

        return {
            "x"                : slice_x_index,
            "edge_index"       : slice_edge_index,
            "edge_attr"        : slice_edge_index,
            "edge_attr_t"      : slice_edge_index,
            "u"                : slice_u_index,
            "hyperedge_index"  : slice_hyperedge_index,
            "hyperedge_attr_t" : slice_hyperedge_index,
            "cls_t"            : slice_u_index,
        }

           
        
    def process(self) -> None:
        
        """
        Built-in PyG-InMemoryDataset method to generate dataset        
        """

        filename = osp.join(self.raw_dir, self.raw_file_names[0])
        print(f"Parsing {filename}")

        data_list = []

        # You can make this configurable in __init__
        chunk_size = getattr(self, "chunk_size", 1000000)

        with h5py.File(filename, "r") as f:
            inputs = f["INPUTS"]
            labels = f["LABELS"]

            num_events = len(inputs["GLOBAL"])
            print(f"Building HyPERDataset with {num_events} events, chunk_size = {chunk_size}")

            # Precompute target Cantor IDs (global, not per-chunk)
            self.target_edge_ids, self.target_hyperedge_ids = self.assign_target_ids()

            for start in range(0, num_events, chunk_size):
                end = min(start + chunk_size, num_events)
                print(f"  -> Processing events [{start}:{end})")

                # --------- build chunk-level tensors ----------
                print("    Building node attributes")
                x_chunk, Nobjects_chunk = self.build_node_attributes(inputs, start, end)

                print("    Building edge attributes")
                edge_attr_chunk = self.build_edge_attributes(inputs, start, end)

                print("    Building global attributes")
                u_chunk = self.build_global_attributes(inputs, start, end)

                print("    Assigning node IDs")
                cantor_node_ids_chunk, local_node_ids_chunk = self.assign_node_ids(
                    x_chunk, labels, start, end, Nobjects_chunk
                )

                print("    Building global target labels")
                cls_t_chunk = self.build_global_targets(labels, start, end)

                print("    Building edge indices")
                edge_index_chunk, cantor_edge_index_chunk = self.build_edge_indices(
                    local_node_ids_chunk, cantor_node_ids_chunk
                )

                print("    Building hyperedge indices")
                hyperedge_index_chunk, cantor_hyperedge_index_chunk = self.build_hyperedge_indices(
                    local_node_ids_chunk,
                    cantor_node_ids_chunk,
                    hyperedge_cardinality=self.hyperedge_order,
                )

                print("    Building edge target labels")
                edge_attr_t_chunk = self.find_matched_connections(
                    cantor_edge_index_chunk, self.target_edge_ids
                )

                print("    Building hyperedge target labels")
                hyperedge_attr_t_chunk = self.find_matched_connections(
                    cantor_hyperedge_index_chunk, self.target_hyperedge_ids
                )

                # Flatten Cantor node IDs to align with x_chunk
                cantor_node_ids_flat = torch.tensor(
                    np.asarray(ak.flatten(cantor_node_ids_chunk)),
                    dtype=torch.float32,
                )

                # --------- per-event slicing inside this chunk ----------
                slices_chunk = self.generate_slices(Nobjects_chunk)
                sx = slices_chunk["x"]
                su = slices_chunk["u"]
                se = slices_chunk["edge_index"]
                sh = slices_chunk["hyperedge_index"]

                for i_evt in range(len(Nobjects_chunk)):
                    # node range for this event
                    x_start = int(sx[i_evt].item())
                    x_end   = int(sx[i_evt + 1].item())

                    # nodes
                    x_evt = x_chunk[x_start:x_end]
                    node_ids_evt = cantor_node_ids_flat[x_start:x_end]

                    # globals
                    # make sure u is 2D: [1, F], so value.size(1) is valid in transform
                    if u_chunk.ndim == 2:
                        u_evt = u_chunk[i_evt].unsqueeze(0)   # [1, F]
                    else:
                        u_evt = u_chunk[i_evt]

                    # if your transform also touches labels (e.g. you had a "cls_t" attr before),
                    # you may want to keep y 2D as well:
                    if cls_t_chunk.ndim == 2:
                        cls_t_evt = cls_t_chunk[i_evt].unsqueeze(0)  # [1, Fy]
                    else:
                        cls_t_evt = cls_t_chunk[i_evt]


                    # edges
                    e_start = int(se[i_evt].item())
                    e_end   = int(se[i_evt + 1].item())
                    edge_index_evt   = edge_index_chunk[:, e_start:e_end]
                    edge_attr_evt    = edge_attr_chunk[e_start:e_end]
                    edge_attr_t_evt  = edge_attr_t_chunk[e_start:e_end]

                    # hyperedges
                    h_start = int(sh[i_evt].item())
                    h_end   = int(sh[i_evt + 1].item())
                    hyperedge_index_evt  = hyperedge_index_chunk[:, h_start:h_end]
                    hyperedge_attr_t_evt = hyperedge_attr_t_chunk[h_start:h_end]

                    data = Data(
                        x=x_evt,
                        edge_index=edge_index_evt,
                        edge_attr=edge_attr_evt,
                        u=u_evt,
                        cls_t=cls_t_evt,
                        node_ids=node_ids_evt,
                        edge_attr_t=edge_attr_t_evt,
                        hyperedge_index=hyperedge_index_evt,
                        hyperedge_attr_t=hyperedge_attr_t_evt,
                    )

                    # honour pre_filter / pre_transform, as usual in InMemoryDataset
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

        # --------- collate & save once at the end ----------
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

