#!/usr/bin/env python3
"""Decorate an S/B HDF5 dataset with HyPER reconstruction observables."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prediction_io import load_hyper_prediction_output  # noqa: E402
from ttbar import ttbar_single_lep  # noqa: E402
from tth import ttH_single_lep  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


OBJECT_DTYPE = np.dtype(
    [("e", "f4"), ("eta", "f4"), ("phi", "f4"), ("pt", "f4")]
)
TTBAR_SL_CONSTITUENT_DTYPE = np.dtype(
    [
        ("e", "f4"),
        ("eta", "f4"),
        ("phi", "f4"),
        ("pt", "f4"),
        ("btag", "f4"),
        ("id", "f4"),
        ("reco_score", "f4"),
        ("is_b_lep", "f4"),
        ("is_b_had", "f4"),
        ("is_Whad_j1", "f4"),
        ("is_Whad_j2", "f4"),
    ]
)
TTH_CONSTITUENT_DTYPE = np.dtype(
    [
        ("e", "f4"),
        ("eta", "f4"),
        ("phi", "f4"),
        ("pt", "f4"),
        ("btag", "f4"),
        ("id", "f4"),
        ("reco_score", "f4"),
        ("is_b_lep", "f4"),
        ("is_b_had", "f4"),
        ("is_Whad_j1", "f4"),
        ("is_Whad_j2", "f4"),
        ("is_Higgs_b1", "f4"),
        ("is_Higgs_b2", "f4"),
    ]
)
TTBAR_SL_RECO_GLOBAL_DTYPE = np.dtype(
    [
        ("m_thad", "f4"),
        ("m_Whad", "f4"),
        ("m_tlep_visible", "f4"),
        ("m_ttbar_visible", "f4"),
        ("top_had_score", "f4"),
        ("top_lep_score", "f4"),
        ("w_had_score", "f4"),
        ("w_lep_score", "f4"),
        ("reco_valid", "f4"),
        ("thad_valid", "f4"),
        ("whad_valid", "f4"),
        ("b_lep_valid", "f4"),
    ]
)
TTH_RECO_GLOBAL_DTYPE = np.dtype(
    [
        ("m_thad", "f4"),
        ("m_Whad", "f4"),
        ("m_tlep_visible", "f4"),
        ("m_H", "f4"),
        ("m_ttH_visible", "f4"),
        ("top_had_score", "f4"),
        ("top_lep_score", "f4"),
        ("w_had_score", "f4"),
        ("w_lep_score", "f4"),
        ("higgs_score", "f4"),
        ("reco_valid", "f4"),
        ("thad_valid", "f4"),
        ("whad_valid", "f4"),
        ("b_lep_valid", "f4"),
        ("higgs_valid", "f4"),
    ]
)

# Backwards-compatible local aliases for helper code that still refers to the
# original names internally.
RECO_GLOBAL_DTYPE = TTBAR_SL_RECO_GLOBAL_DTYPE

PRIMARY_INPUT_PATHS = [
    "INPUTS/TOP_HAD",
    "INPUTS/W_HAD",
    "INPUTS/TOP_HAD_B",
    "INPUTS/W_HAD_J1",
    "INPUTS/W_HAD_J2",
    "INPUTS/B_LEP",
    "INPUTS/RECO_GLOBAL",
]
DEBUG_INDEX_PATHS = [
    "DEBUG_RECO_INDICES/b_lep_idx",
    "DEBUG_RECO_INDICES/b_had_idx",
    "DEBUG_RECO_INDICES/whad_j1_idx",
    "DEBUG_RECO_INDICES/whad_j2_idx",
    "DEBUG_RECO_INDICES/top_had_nodes",
    "DEBUG_RECO_INDICES/top_lep_nodes",
    "DEBUG_RECO_INDICES/whad_nodes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an S/B H5 file and append HyPER reconstruction decorations."
    )
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--hyper-output", required=True)
    parser.add_argument("--output-h5", required=True)
    parser.add_argument("--topology", required=True, choices=["ttbar_singlelep", "ttH"])
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--debug-mirror", action="store_true")
    parser.add_argument(
        "--write-debug-indices",
        action="store_true",
        help="Write selected slot indices under DEBUG_RECO_INDICES. Off by default.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of events to copy/decorate per H5 chunk.",
    )
    return parser.parse_args()


def load_hyper_dataframe(path: str, max_events: int | None = None) -> pd.DataFrame:
    return load_hyper_prediction_output(path, max_events=max_events)


def valid_structured_rows(arr: np.ndarray) -> np.ndarray:
    """Return non-NaN rows of a structured array using the first field as padding tag."""
    arr = np.asarray(arr)
    if arr.dtype.names is None:
        if arr.ndim == 1:
            mask = np.isfinite(arr)
        else:
            mask = np.isfinite(arr[:, 0])
        return arr[mask]
    first_field = arr.dtype.names[0]
    return arr[np.isfinite(arr[first_field])]


def get_field(row: np.void, candidates: tuple[str, ...]) -> float:
    if getattr(row.dtype, "names", None) is None:
        return float("nan")
    names = row.dtype.names or ()
    lower_to_name = {name.lower(): name for name in names}
    for candidate in candidates:
        name = lower_to_name.get(candidate.lower())
        if name is not None:
            value = row[name]
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")
def phi_from_row(
    row: np.void,
    phi_candidates: tuple[str, ...] = ("phi",),
    sin_candidates: tuple[str, ...] = ("sin_phi", "sinphi"),
    cos_candidates: tuple[str, ...] = ("cos_phi", "cosphi"),
) -> float:
    phi = get_field(row, phi_candidates)
    if math.isfinite(phi):
        return phi

    sin_phi = get_field(row, sin_candidates)
    cos_phi = get_field(row, cos_candidates)
    if math.isfinite(sin_phi) and math.isfinite(cos_phi):
        return math.atan2(sin_phi, cos_phi)

    return float("nan")


def row_to_candidate_features(row: np.void) -> tuple[float, float, float, float, float, float]:
    """Extract e, eta, phi, pt, btag, id from a structured row."""
    return (
        get_field(row, ("e", "energy", "E")),
        get_field(row, ("eta",)),
        phi_from_row(row),
        get_field(row, ("pt", "pT")),
        get_field(row, ("btag", "btag_score", "btagging", "btagged")),
        get_field(row, ("id", "pid", "type")),
    )

def p4_from_e_eta_phi_pt(e: float, eta: float, phi: float, pt: float) -> np.ndarray:
    """Return [E, px, py, pz]."""
    values = np.asarray([e, eta, phi, pt], dtype=float)
    if not np.all(np.isfinite(values)):
        return np.full(4, np.nan, dtype=float)
    return np.asarray(
        [e, pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)],
        dtype=float,
    )


def e_eta_phi_pt_from_p4(vec: np.ndarray) -> tuple[float, float, float, float]:
    """Convert [E, px, py, pz] to e, eta, phi, pt."""
    if not np.all(np.isfinite(vec)):
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    e, px, py, pz = [float(x) for x in vec]
    pt = math.hypot(px, py)
    phi = math.atan2(py, px)
    p = math.sqrt(px * px + py * py + pz * pz)
    if p == abs(pz):
        eta = math.copysign(float("inf"), pz) if pz != 0 else 0.0
    else:
        eta = 0.5 * math.log((p + pz) / (p - pz))
    if not math.isfinite(eta):
        eta = float("nan")
    return (e, eta, phi, pt)


def invariant_mass(vec: np.ndarray) -> float:
    """sqrt(max(E^2 - |p|^2, 0))."""
    if not np.all(np.isfinite(vec)):
        return float("nan")
    e, px, py, pz = [float(x) for x in vec]
    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


def met_proxy_p4(row: np.void) -> np.ndarray:
    """Build a massless transverse MET proxy, ignoring MET eta if present."""
    pt = get_field(row, ("pt", "met", "met_pt"))
    phi = phi_from_row(
        row,
        phi_candidates=("phi", "met_phi"),
        sin_candidates=("sin_phi", "sinphi", "met_sin_phi", "met_sinphi"),
        cos_candidates=("cos_phi", "cosphi", "met_cos_phi", "met_cosphi"),
    )
    if not (math.isfinite(pt) and math.isfinite(phi)):
        return np.full(4, np.nan, dtype=float)
    return np.asarray([pt, pt * math.cos(phi), pt * math.sin(phi), 0.0], dtype=float)


def decode_sl_node(
    node_idx: int, n_jets: int, n_leps: int, n_met: int = 1
) -> tuple[str | None, int | None]:
    """Decode single-lepton event graph nodes: jets first, then leptons, then MET."""
    node_idx = int(node_idx)
    if 0 <= node_idx < n_jets:
        return "jet", node_idx
    if n_jets <= node_idx < n_jets + n_leps:
        return "lepton", node_idx - n_jets
    if n_jets + n_leps <= node_idx < n_jets + n_leps + n_met:
        return "met", node_idx - n_jets - n_leps
    return None, None


def as_node_list(value: Any, expected_len: int) -> list[int] | None:
    if value is None:
        return None
    arr = np.asarray(value).astype(int).flatten()
    if len(arr) != expected_len or np.any(arr < 0):
        return None
    return [int(x) for x in arr]


def finite_float(value: Any) -> float:
    try:
        scalar = float(np.asarray(value).flatten()[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")
    return scalar if math.isfinite(scalar) else float("nan")


def fill_nan_structured(shape: tuple[int, int], dtype: np.dtype) -> np.ndarray:
    arr = np.empty(shape, dtype=dtype)
    for name in dtype.names or ():
        arr[name] = np.nan
    return arr


def fill_object_row(arr: np.ndarray, event_idx: int, vec: np.ndarray) -> None:
    e, eta, phi, pt = e_eta_phi_pt_from_p4(vec)
    arr[event_idx, 0] = (e, eta, phi, pt)


ROLE_FLAG_FIELDS = (
    "is_b_lep",
    "is_b_had",
    "is_Whad_j1",
    "is_Whad_j2",
    "is_Higgs_b1",
    "is_Higgs_b2",
)


def fill_constituent_row(
    arr: np.ndarray,
    event_idx: int,
    row: np.void,
    reco_score: float,
    role: str,
) -> None:
    """Fill one selected constituent and encode its semantic reco role.

    Missing rows stay NaN from fill_nan_structured. Filled rows get kinematics,
    one component score, and one-hot role flags directly in the constituent row.
    """
    e, eta, phi, pt, btag, obj_id = row_to_candidate_features(row)
    values = {
        "e": e,
        "eta": eta,
        "phi": phi,
        "pt": pt,
        "btag": btag,
        "id": obj_id,
        "reco_score": reco_score,
    }
    names = arr.dtype.names or ()
    for name, value in values.items():
        if name in names:
            arr[name][event_idx, 0] = value
    for name in ROLE_FLAG_FIELDS:
        if name in names:
            arr[name][event_idx, 0] = 1.0 if name == role else 0.0


def collect_node_roles(
    nodes: list[int], n_jets: int, n_leps: int, n_met: int
) -> dict[str, list[int]]:
    roles: dict[str, list[int]] = {"jet": [], "lepton": [], "met": [], "invalid": []}
    for node in nodes:
        kind, slot = decode_sl_node(node, n_jets, n_leps, n_met)
        if kind is None or slot is None:
            roles["invalid"].append(node)
        else:
            roles[kind].append(slot)
    return roles


def ensure_writable_paths(
    h5: h5py.File, paths: list[str], overwrite: bool, include_debug_mirror: bool
) -> None:
    paths_to_check = list(paths)
    if include_debug_mirror:
        paths_to_check.extend(path.replace("INPUTS/", "INPUTS_RECO/", 1) for path in paths)
    existing = [path for path in paths_to_check if path in h5]
    if existing and not overwrite:
        raise FileExistsError(
            "Decoration paths already exist; rerun with --overwrite to replace them: "
            + ", ".join(existing)
        )
    for path in existing:
        del h5[path]


def write_dataset(h5: h5py.File, path: str, data: np.ndarray) -> None:
    parent = str(Path(path).parent).replace("\\", "/")
    if parent and parent != ".":
        h5.require_group(parent)
    h5.create_dataset(path, data=data)


def write_outputs(
    h5: h5py.File,
    outputs: dict[str, np.ndarray],
    overwrite: bool,
    debug_mirror: bool,
    debug_indices: dict[str, np.ndarray] | None,
) -> list[str]:
    primary_paths = [f"INPUTS/{name}" for name in outputs]
    all_paths = list(primary_paths)
    if debug_indices is not None:
        all_paths.extend(f"DEBUG_RECO_INDICES/{name}" for name in debug_indices)
    ensure_writable_paths(h5, all_paths, overwrite, debug_mirror)

    written = []
    for name, data in outputs.items():
        path = f"INPUTS/{name}"
        write_dataset(h5, path, data)
        written.append(path)
        if debug_mirror:
            mirror_path = f"INPUTS_RECO/{name}"
            write_dataset(h5, mirror_path, data)
            written.append(mirror_path)
    if debug_indices is not None:
        for name, data in debug_indices.items():
            path = f"DEBUG_RECO_INDICES/{name}"
            write_dataset(h5, path, data)
            written.append(path)
    return written


def create_decoration_datasets(
    h5: h5py.File,
    n_events: int,
    chunk_size: int,
    overwrite: bool,
    debug_mirror: bool,
    write_debug_indices: bool,
) -> tuple[dict[str, h5py.Dataset], dict[str, h5py.Dataset] | None, list[str]]:
    output_dtypes = {
        "TOP_HAD": OBJECT_DTYPE,
        "W_HAD": OBJECT_DTYPE,
        "TOP_HAD_B": TTBAR_SL_CONSTITUENT_DTYPE,
        "W_HAD_J1": TTBAR_SL_CONSTITUENT_DTYPE,
        "W_HAD_J2": TTBAR_SL_CONSTITUENT_DTYPE,
        "B_LEP": TTBAR_SL_CONSTITUENT_DTYPE,
        "RECO_GLOBAL": TTBAR_SL_RECO_GLOBAL_DTYPE,
    }
    debug_shapes = {
        "b_lep_idx": (n_events,),
        "b_had_idx": (n_events,),
        "whad_j1_idx": (n_events,),
        "whad_j2_idx": (n_events,),
        "top_had_nodes": (n_events, 3),
        "top_lep_nodes": (n_events, 3),
        "whad_nodes": (n_events, 2),
    }
    paths = [f"INPUTS/{name}" for name in output_dtypes]
    if write_debug_indices:
        paths.extend(f"DEBUG_RECO_INDICES/{name}" for name in debug_shapes)
    ensure_writable_paths(h5, paths, overwrite, debug_mirror)

    written: list[str] = []
    datasets: dict[str, h5py.Dataset] = {}
    for name, dtype in output_dtypes.items():
        h5.require_group("INPUTS")
        ds = h5.create_dataset(
            f"INPUTS/{name}",
            shape=(n_events, 1),
            dtype=dtype,
            chunks=(min(int(chunk_size), max(n_events, 1)), 1),
        )
        datasets[name] = ds
        written.append(f"INPUTS/{name}")
        if debug_mirror:
            h5.require_group("INPUTS_RECO")
            mirror = h5.create_dataset(
                f"INPUTS_RECO/{name}",
                shape=(n_events, 1),
                dtype=dtype,
                chunks=(min(int(chunk_size), max(n_events, 1)), 1),
            )
            datasets[f"INPUTS_RECO/{name}"] = mirror
            written.append(f"INPUTS_RECO/{name}")

    debug_datasets = None
    if write_debug_indices:
        h5.require_group("DEBUG_RECO_INDICES")
        debug_datasets = {}
        for name, shape in debug_shapes.items():
            ds = h5.create_dataset(
                f"DEBUG_RECO_INDICES/{name}",
                shape=shape,
                dtype=np.int32,
                chunks=(min(int(chunk_size), max(n_events, 1)),) + tuple(shape[1:]),
            )
            ds[...] = -1
            debug_datasets[name] = ds
            written.append(f"DEBUG_RECO_INDICES/{name}")
    return datasets, debug_datasets, written


def create_tth_decoration_datasets(
    h5: h5py.File,
    n_events: int,
    chunk_size: int,
    overwrite: bool,
    debug_mirror: bool,
    write_debug_indices: bool,
) -> tuple[dict[str, h5py.Dataset], dict[str, h5py.Dataset] | None, list[str]]:
    output_dtypes = {
        "TOP_HAD": OBJECT_DTYPE,
        "W_HAD": OBJECT_DTYPE,
        "HIGGS": OBJECT_DTYPE,
        "TOP_HAD_B": TTH_CONSTITUENT_DTYPE,
        "W_HAD_J1": TTH_CONSTITUENT_DTYPE,
        "W_HAD_J2": TTH_CONSTITUENT_DTYPE,
        "B_LEP": TTH_CONSTITUENT_DTYPE,
        "HIGGS_B1": TTH_CONSTITUENT_DTYPE,
        "HIGGS_B2": TTH_CONSTITUENT_DTYPE,
        "RECO_GLOBAL": TTH_RECO_GLOBAL_DTYPE,
    }
    debug_shapes = {
        "b_lep_idx": (n_events,),
        "b_had_idx": (n_events,),
        "whad_j1_idx": (n_events,),
        "whad_j2_idx": (n_events,),
        "higgs_b1_idx": (n_events,),
        "higgs_b2_idx": (n_events,),
        "top_had_nodes": (n_events, 3),
        "top_lep_nodes": (n_events, 3),
        "whad_nodes": (n_events, 2),
        "higgs_nodes": (n_events, 2),
    }
    paths = [f"INPUTS/{name}" for name in output_dtypes]
    if write_debug_indices:
        paths.extend(f"DEBUG_RECO_INDICES/{name}" for name in debug_shapes)
    ensure_writable_paths(h5, paths, overwrite, debug_mirror)

    written: list[str] = []
    datasets: dict[str, h5py.Dataset] = {}
    for name, dtype in output_dtypes.items():
        h5.require_group("INPUTS")
        ds = h5.create_dataset(
            f"INPUTS/{name}",
            shape=(n_events, 1),
            dtype=dtype,
            chunks=(min(int(chunk_size), max(n_events, 1)), 1),
        )
        datasets[name] = ds
        written.append(f"INPUTS/{name}")
        if debug_mirror:
            h5.require_group("INPUTS_RECO")
            mirror = h5.create_dataset(
                f"INPUTS_RECO/{name}",
                shape=(n_events, 1),
                dtype=dtype,
                chunks=(min(int(chunk_size), max(n_events, 1)), 1),
            )
            datasets[f"INPUTS_RECO/{name}"] = mirror
            written.append(f"INPUTS_RECO/{name}")

    debug_datasets = None
    if write_debug_indices:
        h5.require_group("DEBUG_RECO_INDICES")
        debug_datasets = {}
        for name, shape in debug_shapes.items():
            ds = h5.create_dataset(
                f"DEBUG_RECO_INDICES/{name}",
                shape=shape,
                dtype=np.int32,
                chunks=(min(int(chunk_size), max(n_events, 1)),) + tuple(shape[1:]),
            )
            ds[...] = -1
            debug_datasets[name] = ds
            written.append(f"DEBUG_RECO_INDICES/{name}")
    return datasets, debug_datasets, written


def semantic_ttbar_singlelep_reco(hyper_df: pd.DataFrame) -> pd.DataFrame:
    required_selected = {
        "HyPER_best_top1",
        "HyPER_best_top2",
        "HyPER_best_w1",
        "HyPER_best_w2",
        "HyPER_best_top1_prob",
        "HyPER_best_top2_prob",
        "HyPER_best_w1_prob",
        "HyPER_best_w2_prob",
    }
    if not required_selected.issubset(hyper_df.columns):
        raw_required = {
            "HyPER_HE_IDX",
            "HyPER_HE_RAW",
            "HyPER_HE_VCT",
            "HyPER_GE_IDX",
            "HyPER_GE_RAW",
            "HyPER_GE_VCT",
        }
        missing = sorted(raw_required - set(hyper_df.columns))
        if missing:
            raise KeyError(
                "HyPER output is missing selected reco columns and raw columns: "
                + ", ".join(missing)
            )
        # Current production models are trained without a classification head.
        # Ignore any legacy HyPER_CLS_RAW column and reconstruct without requiring it.
        hyper_df = ttbar_single_lep(hyper_df, classification=False)
    else:
        hyper_df = hyper_df.copy()

    # ttbar_single_lep swaps candidates after finding tlep_position, so the old
    # names below are semantic after that swap: top1/w1 are leptonic, top2/w2
    # are hadronic.
    semantic = pd.DataFrame(index=hyper_df.index)
    semantic["HyPER_best_tlep"] = hyper_df["HyPER_best_top1"]
    semantic["HyPER_best_thad"] = hyper_df["HyPER_best_top2"]
    semantic["HyPER_best_wlep"] = hyper_df["HyPER_best_w1"]
    semantic["HyPER_best_whad"] = hyper_df["HyPER_best_w2"]
    semantic["HyPER_best_tlep_score"] = hyper_df["HyPER_best_top1_prob"]
    semantic["HyPER_best_thad_score"] = hyper_df["HyPER_best_top2_prob"]
    semantic["HyPER_best_wlep_score"] = hyper_df["HyPER_best_w1_prob"]
    semantic["HyPER_best_whad_score"] = hyper_df["HyPER_best_w2_prob"]
    return semantic


def semantic_tth_singlelep_reco(hyper_df: pd.DataFrame) -> pd.DataFrame:
    required_selected = {
        "HyPER_best_tlep",
        "HyPER_best_thad",
        "HyPER_best_wlep",
        "HyPER_best_whad",
        "HyPER_best_higgs",
        "HyPER_best_blep",
        "HyPER_best_bhad",
        "HyPER_best_whad_j1",
        "HyPER_best_whad_j2",
        "HyPER_best_higgs_b1",
        "HyPER_best_higgs_b2",
    }
    if not required_selected.issubset(hyper_df.columns):
        raw_required = {
            "HyPER_HE_IDX",
            "HyPER_HE_RAW",
            "HyPER_HE_VCT",
            "HyPER_GE_IDX",
            "HyPER_GE_RAW",
            "HyPER_GE_VCT",
        }
        missing = sorted(raw_required - set(hyper_df.columns))
        if missing:
            raise KeyError(
                "HyPER output is missing selected ttH reco columns and raw columns: "
                + ", ".join(missing)
            )
        # Current production models are no-class; old classification outputs are ignored.
        hyper_df = ttH_single_lep(hyper_df, classification=False)
    else:
        hyper_df = hyper_df.copy()
    for score_column in (
        "HyPER_best_thad_prob",
        "HyPER_best_tlep_prob",
        "HyPER_best_whad_prob",
        "HyPER_best_wlep_prob",
        "HyPER_best_higgs_prob",
    ):
        if score_column not in hyper_df.columns:
            hyper_df[score_column] = np.nan
    return hyper_df.reset_index(drop=True)


def align_reco_to_h5_prefix(
    raw_hyper_df: pd.DataFrame,
    reco: pd.DataFrame,
    n_processed: int,
    strict: bool,
    warnings_list: list[str],
) -> tuple[pd.DataFrame, str]:
    """Align prediction rows to the copied H5 prefix when event_index is available.

    The decorator writes a prefix of the input H5.  Selected prediction outputs may
    carry an ``event_index`` column; when that index is the same prefix, or a
    permutation of it, we verify/reorder explicitly.  Arbitrary sparse subsets need
    a matching sampled H5 and are reported loudly instead of being silently treated
    as aligned.
    """
    reco = reco.iloc[:n_processed].reset_index(drop=True)
    if "event_index" not in raw_hyper_df.columns:
        warning = (
            "HyPER output has no event_index column; assuming prediction row order "
            "matches the input H5 prefix."
        )
        LOGGER.warning(warning)
        warnings_list.append(warning)
        return reco, "row_order_prefix_assumed"

    event_index = pd.to_numeric(
        raw_hyper_df["event_index"].iloc[:n_processed],
        errors="coerce",
    ).to_numpy(dtype=float)
    if len(event_index) != n_processed or not np.all(np.isfinite(event_index)):
        warning = (
            "HyPER output event_index is missing or non-finite for some processed rows; "
            "assuming prediction row order matches the input H5 prefix."
        )
        if strict:
            raise ValueError(warning)
        LOGGER.warning(warning)
        warnings_list.append(warning)
        return reco, "row_order_prefix_assumed_event_index_invalid"

    event_index_int = event_index.astype(np.int64)
    if not np.allclose(event_index, event_index_int):
        warning = (
            "HyPER output event_index contains non-integer values; assuming prediction "
            "row order matches the input H5 prefix."
        )
        if strict:
            raise ValueError(warning)
        LOGGER.warning(warning)
        warnings_list.append(warning)
        return reco, "row_order_prefix_assumed_event_index_invalid"

    expected = np.arange(n_processed, dtype=np.int64)
    if np.array_equal(event_index_int, expected):
        return reco, "event_index_prefix_verified"

    if np.array_equal(np.sort(event_index_int), expected):
        order = np.argsort(event_index_int)
        return reco.iloc[order].reset_index(drop=True), "event_index_prefix_reordered"

    warning = (
        "HyPER output event_index is not the copied H5 prefix 0..n_processed-1. "
        "This decorator currently writes prefix H5 files, so arbitrary sampled "
        "prediction subsets need a matching sampled H5. Assuming row-order alignment "
        "for this run."
    )
    if strict:
        raise ValueError(warning)
    LOGGER.warning(warning)
    warnings_list.append(warning)
    return reco, "row_order_prefix_assumed_event_index_nonprefix"


def decorate_ttbar_singlelep(
    jets_chunk: np.ndarray,
    leptons_chunk: np.ndarray,
    met_chunk: np.ndarray,
    reco: pd.DataFrame,
    n_input_events: int,
    n_processed: int,
    write_debug_indices: bool,
    output_events: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, Counter]:
    n_output_events = n_input_events if output_events is None else int(output_events)

    outputs = {
        "TOP_HAD": fill_nan_structured((n_output_events, 1), OBJECT_DTYPE),
        "W_HAD": fill_nan_structured((n_output_events, 1), OBJECT_DTYPE),
        "TOP_HAD_B": fill_nan_structured((n_output_events, 1), TTBAR_SL_CONSTITUENT_DTYPE),
        "W_HAD_J1": fill_nan_structured((n_output_events, 1), TTBAR_SL_CONSTITUENT_DTYPE),
        "W_HAD_J2": fill_nan_structured((n_output_events, 1), TTBAR_SL_CONSTITUENT_DTYPE),
        "B_LEP": fill_nan_structured((n_output_events, 1), TTBAR_SL_CONSTITUENT_DTYPE),
        "RECO_GLOBAL": fill_nan_structured((n_output_events, 1), TTBAR_SL_RECO_GLOBAL_DTYPE),
    }

    debug_indices = None
    if write_debug_indices:
        debug_indices = {
            "b_lep_idx": np.full(n_output_events, -1, dtype=np.int32),
            "b_had_idx": np.full(n_output_events, -1, dtype=np.int32),
            "whad_j1_idx": np.full(n_output_events, -1, dtype=np.int32),
            "whad_j2_idx": np.full(n_output_events, -1, dtype=np.int32),
            "top_had_nodes": np.full((n_output_events, 3), -1, dtype=np.int32),
            "top_lep_nodes": np.full((n_output_events, 3), -1, dtype=np.int32),
            "whad_nodes": np.full((n_output_events, 2), -1, dtype=np.int32),
        }

    invalid_reasons: Counter = Counter()
    global_arr = outputs["RECO_GLOBAL"]
    for name in ("reco_valid", "thad_valid", "whad_valid", "b_lep_valid"):
        global_arr[name][:] = 0.0

    for event_idx in range(n_processed):
        jet_rows = valid_structured_rows(jets_chunk[event_idx])
        lepton_rows = valid_structured_rows(leptons_chunk[event_idx])
        met_rows = valid_structured_rows(met_chunk[event_idx])
        n_jets = len(jet_rows)
        n_leps = len(lepton_rows)
        n_met = min(len(met_rows), 1)

        if n_jets < 4:
            invalid_reasons["not_enough_jets"] += 1
        if n_leps < 1:
            invalid_reasons["not_enough_leptons"] += 1
        if n_met < 1:
            invalid_reasons["missing_met"] += 1

        row = reco.iloc[event_idx]
        top_had_score = finite_float(row["HyPER_best_thad_score"])
        top_lep_score = finite_float(row["HyPER_best_tlep_score"])
        w_had_score = finite_float(row["HyPER_best_whad_score"])
        w_lep_score = finite_float(row["HyPER_best_wlep_score"])

        tlep_nodes = as_node_list(row["HyPER_best_tlep"], 3)
        thad_nodes = as_node_list(row["HyPER_best_thad"], 3)
        whad_nodes = as_node_list(row["HyPER_best_whad"], 2)
        if tlep_nodes is None:
            invalid_reasons["missing_tlep"] += 1
        if thad_nodes is None:
            invalid_reasons["missing_thad"] += 1
        if whad_nodes is None:
            invalid_reasons["missing_whad"] += 1

        if debug_indices is not None:
            if tlep_nodes is not None:
                debug_indices["top_lep_nodes"][event_idx] = tlep_nodes
            if thad_nodes is not None:
                debug_indices["top_had_nodes"][event_idx] = thad_nodes
            if whad_nodes is not None:
                debug_indices["whad_nodes"][event_idx] = whad_nodes

        tlep_roles = (
            collect_node_roles(tlep_nodes, n_jets, n_leps, n_met)
            if tlep_nodes is not None
            else None
        )
        thad_roles = (
            collect_node_roles(thad_nodes, n_jets, n_leps, n_met)
            if thad_nodes is not None
            else None
        )
        whad_roles = (
            collect_node_roles(whad_nodes, n_jets, n_leps, n_met)
            if whad_nodes is not None
            else None
        )
        if any(
            roles is not None and roles["invalid"]
            for roles in (tlep_roles, thad_roles, whad_roles)
        ):
            invalid_reasons["decode_failed"] += 1

        b_lep_idx = None
        lep_idx = None
        met_idx = None
        if tlep_roles is not None:
            if tlep_roles["jet"]:
                b_lep_idx = tlep_roles["jet"][0]
            if tlep_roles["lepton"]:
                lep_idx = tlep_roles["lepton"][0]
            if tlep_roles["met"]:
                met_idx = tlep_roles["met"][0]

        whad_jet_indices: list[int] = []
        if whad_roles is not None:
            whad_jet_indices = whad_roles["jet"]
        b_had_idx = None
        if thad_roles is not None:
            b_had_candidates = [
                idx for idx in thad_roles["jet"] if idx not in set(whad_jet_indices)
            ]
            if b_had_candidates:
                b_had_idx = b_had_candidates[0]

        if debug_indices is not None:
            if b_lep_idx is not None:
                debug_indices["b_lep_idx"][event_idx] = b_lep_idx
            if b_had_idx is not None:
                debug_indices["b_had_idx"][event_idx] = b_had_idx
            if len(whad_jet_indices) > 0:
                debug_indices["whad_j1_idx"][event_idx] = whad_jet_indices[0]
            if len(whad_jet_indices) > 1:
                debug_indices["whad_j2_idx"][event_idx] = whad_jet_indices[1]

        thad_p4 = np.full(4, np.nan, dtype=float)
        whad_p4 = np.full(4, np.nan, dtype=float)
        b_lep_p4 = np.full(4, np.nan, dtype=float)
        lep_p4 = np.full(4, np.nan, dtype=float)
        met_p4 = np.full(4, np.nan, dtype=float)

        whad_valid = len(whad_jet_indices) == 2 and all(i < n_jets for i in whad_jet_indices)
        if whad_valid:
            j1, j2 = whad_jet_indices[:2]
            fill_constituent_row(
                outputs["W_HAD_J1"],
                event_idx,
                jet_rows[j1],
                reco_score=w_had_score,
                role="is_Whad_j1",
            )
            fill_constituent_row(
                outputs["W_HAD_J2"],
                event_idx,
                jet_rows[j2],
                reco_score=w_had_score,
                role="is_Whad_j2",
            )
            whad_p4 = sum(
                (
                    p4_from_e_eta_phi_pt(*row_to_candidate_features(jet_rows[idx])[:4])
                    for idx in (j1, j2)
                ),
                np.zeros(4, dtype=float),
            )
            fill_object_row(outputs["W_HAD"], event_idx, whad_p4)
            global_arr["m_Whad"][event_idx, 0] = invariant_mass(whad_p4)
            global_arr["whad_valid"][event_idx, 0] = 1.0
        elif whad_nodes is not None:
            invalid_reasons["decode_failed"] += 1

        thad_valid = whad_valid and b_had_idx is not None and b_had_idx < n_jets
        if thad_valid:
            fill_constituent_row(
                outputs["TOP_HAD_B"],
                event_idx,
                jet_rows[b_had_idx],
                reco_score=top_had_score,
                role="is_b_had",
            )
            b_had_p4 = p4_from_e_eta_phi_pt(*row_to_candidate_features(jet_rows[b_had_idx])[:4])
            thad_p4 = whad_p4 + b_had_p4
            fill_object_row(outputs["TOP_HAD"], event_idx, thad_p4)
            global_arr["m_thad"][event_idx, 0] = invariant_mass(thad_p4)
            global_arr["thad_valid"][event_idx, 0] = 1.0
        elif thad_nodes is not None:
            invalid_reasons["decode_failed"] += 1

        b_lep_valid = b_lep_idx is not None and b_lep_idx < n_jets
        if b_lep_valid:
            fill_constituent_row(
                outputs["B_LEP"],
                event_idx,
                jet_rows[b_lep_idx],
                reco_score=top_lep_score,
                role="is_b_lep",
            )
            b_lep_p4 = p4_from_e_eta_phi_pt(*row_to_candidate_features(jet_rows[b_lep_idx])[:4])
            global_arr["b_lep_valid"][event_idx, 0] = 1.0
        elif tlep_nodes is not None:
            invalid_reasons["decode_failed"] += 1

        lep_valid = lep_idx is not None and lep_idx < n_leps
        if lep_valid:
            lep_p4 = p4_from_e_eta_phi_pt(*row_to_candidate_features(lepton_rows[lep_idx])[:4])
        met_valid = met_idx is not None and met_idx < n_met
        if met_valid:
            met_p4 = met_proxy_p4(met_rows[met_idx])

        if b_lep_valid and lep_valid and met_valid:
            tlep_visible = b_lep_p4 + lep_p4 + met_p4
            global_arr["m_tlep_visible"][event_idx, 0] = invariant_mass(tlep_visible)
            if thad_valid:
                global_arr["m_ttbar_visible"][event_idx, 0] = invariant_mass(
                    thad_p4 + tlep_visible
                )

        global_arr["top_had_score"][event_idx, 0] = top_had_score
        global_arr["top_lep_score"][event_idx, 0] = top_lep_score
        global_arr["w_had_score"][event_idx, 0] = w_had_score
        global_arr["w_lep_score"][event_idx, 0] = w_lep_score

        reco_valid = thad_valid and whad_valid and b_lep_valid
        if reco_valid:
            global_arr["reco_valid"][event_idx, 0] = 1.0

    return outputs, debug_indices, invalid_reasons


def decorate_ttH(
    jets_chunk: np.ndarray,
    leptons_chunk: np.ndarray,
    met_chunk: np.ndarray,
    reco: pd.DataFrame,
    n_input_events: int,
    n_processed: int,
    write_debug_indices: bool,
    output_events: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, Counter]:
    n_output_events = n_input_events if output_events is None else int(output_events)

    outputs = {
        "TOP_HAD": fill_nan_structured((n_output_events, 1), OBJECT_DTYPE),
        "W_HAD": fill_nan_structured((n_output_events, 1), OBJECT_DTYPE),
        "HIGGS": fill_nan_structured((n_output_events, 1), OBJECT_DTYPE),
        "TOP_HAD_B": fill_nan_structured((n_output_events, 1), TTH_CONSTITUENT_DTYPE),
        "W_HAD_J1": fill_nan_structured((n_output_events, 1), TTH_CONSTITUENT_DTYPE),
        "W_HAD_J2": fill_nan_structured((n_output_events, 1), TTH_CONSTITUENT_DTYPE),
        "B_LEP": fill_nan_structured((n_output_events, 1), TTH_CONSTITUENT_DTYPE),
        "HIGGS_B1": fill_nan_structured((n_output_events, 1), TTH_CONSTITUENT_DTYPE),
        "HIGGS_B2": fill_nan_structured((n_output_events, 1), TTH_CONSTITUENT_DTYPE),
        "RECO_GLOBAL": fill_nan_structured((n_output_events, 1), TTH_RECO_GLOBAL_DTYPE),
    }

    debug_indices = None
    if write_debug_indices:
        debug_indices = {
            "b_lep_idx": np.full(n_output_events, -1, dtype=np.int32),
            "b_had_idx": np.full(n_output_events, -1, dtype=np.int32),
            "whad_j1_idx": np.full(n_output_events, -1, dtype=np.int32),
            "whad_j2_idx": np.full(n_output_events, -1, dtype=np.int32),
            "higgs_b1_idx": np.full(n_output_events, -1, dtype=np.int32),
            "higgs_b2_idx": np.full(n_output_events, -1, dtype=np.int32),
            "top_had_nodes": np.full((n_output_events, 3), -1, dtype=np.int32),
            "top_lep_nodes": np.full((n_output_events, 3), -1, dtype=np.int32),
            "whad_nodes": np.full((n_output_events, 2), -1, dtype=np.int32),
            "higgs_nodes": np.full((n_output_events, 2), -1, dtype=np.int32),
        }

    invalid_reasons: Counter = Counter()
    global_arr = outputs["RECO_GLOBAL"]
    for name in ("reco_valid", "thad_valid", "whad_valid", "b_lep_valid", "higgs_valid"):
        global_arr[name][:] = 0.0

    for event_idx in range(n_processed):
        jet_rows = valid_structured_rows(jets_chunk[event_idx])
        lepton_rows = valid_structured_rows(leptons_chunk[event_idx])
        met_rows = valid_structured_rows(met_chunk[event_idx])
        n_jets = len(jet_rows)
        n_leps = len(lepton_rows)
        n_met = min(len(met_rows), 1)

        if n_jets < 6:
            invalid_reasons["not_enough_jets"] += 1
        if n_leps < 1:
            invalid_reasons["not_enough_leptons"] += 1
        if n_met < 1:
            invalid_reasons["missing_met"] += 1

        row = reco.iloc[event_idx]
        top_had_score = finite_float(row.get("HyPER_best_thad_prob", np.nan))
        top_lep_score = finite_float(row.get("HyPER_best_tlep_prob", np.nan))
        w_had_score = finite_float(row.get("HyPER_best_whad_prob", np.nan))
        w_lep_score = finite_float(row.get("HyPER_best_wlep_prob", np.nan))
        higgs_score = finite_float(row.get("HyPER_best_higgs_prob", np.nan))

        tlep_nodes = as_node_list(row["HyPER_best_tlep"], 3)
        thad_nodes = as_node_list(row["HyPER_best_thad"], 3)
        whad_nodes = as_node_list(row["HyPER_best_whad"], 2)
        higgs_nodes = as_node_list(row["HyPER_best_higgs"], 2)
        if tlep_nodes is None:
            invalid_reasons["missing_tlep"] += 1
        if thad_nodes is None:
            invalid_reasons["missing_thad"] += 1
        if whad_nodes is None:
            invalid_reasons["missing_whad"] += 1
        if higgs_nodes is None:
            invalid_reasons["missing_higgs"] += 1

        if debug_indices is not None:
            if tlep_nodes is not None:
                debug_indices["top_lep_nodes"][event_idx] = tlep_nodes
            if thad_nodes is not None:
                debug_indices["top_had_nodes"][event_idx] = thad_nodes
            if whad_nodes is not None:
                debug_indices["whad_nodes"][event_idx] = whad_nodes
            if higgs_nodes is not None:
                debug_indices["higgs_nodes"][event_idx] = higgs_nodes

        b_lep_idx = int(row.get("HyPER_best_blep", -1))
        b_had_idx = int(row.get("HyPER_best_bhad", -1))
        whad_jet_indices = [
            int(row.get("HyPER_best_whad_j1", -1)),
            int(row.get("HyPER_best_whad_j2", -1)),
        ]
        higgs_jet_indices = [
            int(row.get("HyPER_best_higgs_b1", -1)),
            int(row.get("HyPER_best_higgs_b2", -1)),
        ]
        whad_jet_indices = [idx for idx in whad_jet_indices if idx >= 0]
        higgs_jet_indices = [idx for idx in higgs_jet_indices if idx >= 0]

        if debug_indices is not None:
            debug_indices["b_lep_idx"][event_idx] = b_lep_idx
            debug_indices["b_had_idx"][event_idx] = b_had_idx
            if len(whad_jet_indices) > 0:
                debug_indices["whad_j1_idx"][event_idx] = whad_jet_indices[0]
            if len(whad_jet_indices) > 1:
                debug_indices["whad_j2_idx"][event_idx] = whad_jet_indices[1]
            if len(higgs_jet_indices) > 0:
                debug_indices["higgs_b1_idx"][event_idx] = higgs_jet_indices[0]
            if len(higgs_jet_indices) > 1:
                debug_indices["higgs_b2_idx"][event_idx] = higgs_jet_indices[1]

        whad_p4 = np.full(4, np.nan, dtype=float)
        thad_p4 = np.full(4, np.nan, dtype=float)
        higgs_p4 = np.full(4, np.nan, dtype=float)
        b_lep_p4 = np.full(4, np.nan, dtype=float)
        lep_p4 = np.full(4, np.nan, dtype=float)
        met_p4 = np.full(4, np.nan, dtype=float)

        whad_valid = len(whad_jet_indices) == 2 and all(i < n_jets for i in whad_jet_indices)
        if whad_valid:
            j1, j2 = whad_jet_indices[:2]
            fill_constituent_row(
                outputs["W_HAD_J1"],
                event_idx,
                jet_rows[j1],
                reco_score=w_had_score,
                role="is_Whad_j1",
            )
            fill_constituent_row(
                outputs["W_HAD_J2"],
                event_idx,
                jet_rows[j2],
                reco_score=w_had_score,
                role="is_Whad_j2",
            )
            whad_p4 = sum(
                (p4_from_e_eta_phi_pt(*row_to_candidate_features(jet_rows[idx])[:4]) for idx in (j1, j2)),
                np.zeros(4, dtype=float),
            )
            fill_object_row(outputs["W_HAD"], event_idx, whad_p4)
            global_arr["m_Whad"][event_idx, 0] = invariant_mass(whad_p4)
            global_arr["whad_valid"][event_idx, 0] = 1.0
        elif whad_nodes is not None:
            invalid_reasons["decode_failed"] += 1

        thad_valid = whad_valid and b_had_idx >= 0 and b_had_idx < n_jets
        if thad_valid:
            fill_constituent_row(
                outputs["TOP_HAD_B"],
                event_idx,
                jet_rows[b_had_idx],
                reco_score=top_had_score,
                role="is_b_had",
            )
            b_had_p4 = p4_from_e_eta_phi_pt(*row_to_candidate_features(jet_rows[b_had_idx])[:4])
            thad_p4 = whad_p4 + b_had_p4
            fill_object_row(outputs["TOP_HAD"], event_idx, thad_p4)
            global_arr["m_thad"][event_idx, 0] = invariant_mass(thad_p4)
            global_arr["thad_valid"][event_idx, 0] = 1.0
        elif thad_nodes is not None:
            invalid_reasons["decode_failed"] += 1

        higgs_valid = len(higgs_jet_indices) == 2 and all(i < n_jets for i in higgs_jet_indices)
        if higgs_valid:
            h1, h2 = higgs_jet_indices[:2]
            fill_constituent_row(
                outputs["HIGGS_B1"],
                event_idx,
                jet_rows[h1],
                reco_score=higgs_score,
                role="is_Higgs_b1",
            )
            fill_constituent_row(
                outputs["HIGGS_B2"],
                event_idx,
                jet_rows[h2],
                reco_score=higgs_score,
                role="is_Higgs_b2",
            )
            higgs_p4 = sum(
                (p4_from_e_eta_phi_pt(*row_to_candidate_features(jet_rows[idx])[:4]) for idx in (h1, h2)),
                np.zeros(4, dtype=float),
            )
            fill_object_row(outputs["HIGGS"], event_idx, higgs_p4)
            global_arr["m_H"][event_idx, 0] = invariant_mass(higgs_p4)
            global_arr["higgs_valid"][event_idx, 0] = 1.0
        elif higgs_nodes is not None:
            invalid_reasons["decode_failed"] += 1

        b_lep_valid = b_lep_idx >= 0 and b_lep_idx < n_jets
        if b_lep_valid:
            fill_constituent_row(
                outputs["B_LEP"],
                event_idx,
                jet_rows[b_lep_idx],
                reco_score=top_lep_score,
                role="is_b_lep",
            )
            b_lep_p4 = p4_from_e_eta_phi_pt(*row_to_candidate_features(jet_rows[b_lep_idx])[:4])
            global_arr["b_lep_valid"][event_idx, 0] = 1.0
        elif tlep_nodes is not None:
            invalid_reasons["decode_failed"] += 1

        lep_idx = None
        met_idx = None
        if tlep_nodes is not None:
            tlep_roles = collect_node_roles(tlep_nodes, n_jets, n_leps, n_met)
            if tlep_roles["lepton"]:
                lep_idx = tlep_roles["lepton"][0]
            if tlep_roles["met"]:
                met_idx = tlep_roles["met"][0]
        if lep_idx is not None and lep_idx < n_leps:
            lep_p4 = p4_from_e_eta_phi_pt(*row_to_candidate_features(lepton_rows[lep_idx])[:4])
        if met_idx is not None and met_idx < n_met:
            met_p4 = met_proxy_p4(met_rows[met_idx])

        if b_lep_valid and np.all(np.isfinite(lep_p4)) and np.all(np.isfinite(met_p4)):
            tlep_visible = b_lep_p4 + lep_p4 + met_p4
            global_arr["m_tlep_visible"][event_idx, 0] = invariant_mass(tlep_visible)
            if thad_valid and higgs_valid:
                global_arr["m_ttH_visible"][event_idx, 0] = invariant_mass(
                    thad_p4 + tlep_visible + higgs_p4
                )

        global_arr["top_had_score"][event_idx, 0] = top_had_score
        global_arr["top_lep_score"][event_idx, 0] = top_lep_score
        global_arr["w_had_score"][event_idx, 0] = w_had_score
        global_arr["w_lep_score"][event_idx, 0] = w_lep_score
        global_arr["higgs_score"][event_idx, 0] = higgs_score

        if thad_valid and whad_valid and b_lep_valid and higgs_valid:
            global_arr["reco_valid"][event_idx, 0] = 1.0

    return outputs, debug_indices, invalid_reasons


def _copy_attrs(source: h5py.AttributeManager, dest: h5py.AttributeManager) -> None:
    for key, value in source.items():
        dest[key] = value


def _copy_group_with_event_limit(
    source_group: h5py.Group,
    dest_group: h5py.Group,
    n_input_events: int,
    event_limit: int,
) -> None:
    _copy_attrs(source_group.attrs, dest_group.attrs)
    for name, item in source_group.items():
        if isinstance(item, h5py.Group):
            child = dest_group.create_group(name)
            _copy_group_with_event_limit(item, child, n_input_events, event_limit)
            continue

        if not isinstance(item, h5py.Dataset):
            continue

        kwargs: dict[str, Any] = {}
        if item.compression is not None:
            kwargs["compression"] = item.compression
            kwargs["compression_opts"] = item.compression_opts
        if item.shuffle:
            kwargs["shuffle"] = item.shuffle
        if item.fletcher32:
            kwargs["fletcher32"] = item.fletcher32

        if item.shape and item.shape[0] == n_input_events:
            data = item[:event_limit]
        else:
            data = item[()]
        dataset = dest_group.create_dataset(name, data=data, **kwargs)
        _copy_attrs(item.attrs, dataset.attrs)


def _dataset_create_kwargs(item: h5py.Dataset) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if item.compression is not None:
        kwargs["compression"] = item.compression
        kwargs["compression_opts"] = item.compression_opts
    if item.shuffle:
        kwargs["shuffle"] = item.shuffle
    if item.fletcher32:
        kwargs["fletcher32"] = item.fletcher32
    return kwargs


def copy_dataset_chunked(
    src_ds: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    n_input_events: int,
    event_limit: int,
    chunk_size: int,
) -> None:
    kwargs = _dataset_create_kwargs(src_ds)
    if src_ds.shape and src_ds.shape[0] == n_input_events:
        shape = (event_limit,) + tuple(src_ds.shape[1:])
        dst = dst_group.create_dataset(name, shape=shape, dtype=src_ds.dtype, **kwargs)
        _copy_attrs(src_ds.attrs, dst.attrs)
        for start in range(0, event_limit, chunk_size):
            stop = min(start + chunk_size, event_limit)
            dst[start:stop] = src_ds[start:stop]
        return

    if src_ds.shape and src_ds.size > max(chunk_size, 1):
        dst = dst_group.create_dataset(name, shape=src_ds.shape, dtype=src_ds.dtype, **kwargs)
        _copy_attrs(src_ds.attrs, dst.attrs)
        rows = src_ds.shape[0] if src_ds.shape else 1
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            dst[start:stop] = src_ds[start:stop]
        return

    dst = dst_group.create_dataset(name, data=src_ds[()], **kwargs)
    _copy_attrs(src_ds.attrs, dst.attrs)


def copy_group_chunked(
    source_group: h5py.Group,
    dest_group: h5py.Group,
    n_input_events: int,
    event_limit: int,
    chunk_size: int,
) -> None:
    _copy_attrs(source_group.attrs, dest_group.attrs)
    for name, item in source_group.items():
        if isinstance(item, h5py.Group):
            child = dest_group.create_group(name)
            copy_group_chunked(item, child, n_input_events, event_limit, chunk_size)
        elif isinstance(item, h5py.Dataset):
            copy_dataset_chunked(item, dest_group, name, n_input_events, event_limit, chunk_size)


def copy_input_h5(
    input_h5: str,
    output_h5: str,
    overwrite: bool,
    n_input_events: int,
    event_limit: int | None = None,
    chunk_size: int = 100000,
) -> None:
    if Path(input_h5).resolve() == Path(output_h5).resolve():
        raise ValueError("--output-h5 must be different from --input-h5.")
    output_path = Path(output_h5)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_h5} exists; rerun with --overwrite to replace it.")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = n_input_events if event_limit is None else min(int(event_limit), n_input_events)
    with h5py.File(input_h5, "r") as source, h5py.File(output_h5, "w") as dest:
        _copy_attrs(source.attrs, dest.attrs)
        copy_group_chunked(source, dest, n_input_events, limit, int(chunk_size))


def build_summary(
    args: argparse.Namespace,
    n_input_events: int,
    n_reco_events: int,
    n_loaded_reco_events: int,
    n_output_events: int,
    n_processed: int,
    invalid_reasons: Counter,
    fields_written: list[str],
    warnings_list: list[str],
    alignment_mode: str,
    reco_counters: Counter | None = None,
) -> dict[str, Any]:
    reco_counters = reco_counters or Counter()
    return {
        "input_h5": str(args.input_h5),
        "hyper_output": str(args.hyper_output),
        "output_h5": str(args.output_h5),
        "topology": args.topology,
        "n_input_events": int(n_input_events),
        "n_reco_events": int(n_reco_events),
        "n_loaded_reco_events": int(n_loaded_reco_events),
        "n_output_events": int(n_output_events),
        "n_processed": int(n_processed),
        "n_reco_valid": int(reco_counters.get("reco_valid", 0)),
        "n_thad_valid": int(reco_counters.get("thad_valid", 0)),
        "n_whad_valid": int(reco_counters.get("whad_valid", 0)),
        "n_b_lep_valid": int(reco_counters.get("b_lep_valid", 0)),
        "length_mismatch": bool(n_input_events != n_reco_events),
        "event_alignment": alignment_mode,
        "event_alignment_policy": (
            "event_index is verified/reordered when it describes the copied H5 prefix; "
            "otherwise row-order prefix alignment is assumed unless --strict raises."
        ),
        "invalid_reasons": {
            key: int(invalid_reasons.get(key, 0))
            for key in [
                "missing_tlep",
                "missing_thad",
                "missing_whad",
                "missing_higgs",
                "decode_failed",
                "not_enough_jets",
                "not_enough_leptons",
                "missing_met",
            ]
        },
        "reco_counters": {key: int(value) for key, value in reco_counters.items()},
        "fields_written": fields_written,
        "warnings": warnings_list,
    }


def main() -> None:
    args = parse_args()

    input_h5 = Path(args.input_h5)
    hyper_output = Path(args.hyper_output)
    if not input_h5.exists():
        raise FileNotFoundError(input_h5)
    if not hyper_output.exists():
        raise FileNotFoundError(hyper_output)

    hyper_df = load_hyper_dataframe(str(hyper_output), max_events=args.max_events)
    n_loaded_reco_events = len(hyper_df)
    n_reco_events = int(hyper_df.attrs.get("hyper_total_rows", n_loaded_reco_events))
    if args.topology == "ttbar_singlelep":
        reco = semantic_ttbar_singlelep_reco(hyper_df)
    elif args.topology == "ttH":
        reco = semantic_tth_singlelep_reco(hyper_df)
    else:
        raise ValueError(f"Unsupported topology {args.topology}")

    with h5py.File(input_h5, "r") as source:
        for required in ("INPUTS/JET", "INPUTS/LEPTON", "INPUTS/MET"):
            if required not in source:
                raise KeyError(f"Missing required input dataset {required}")
        n_input_events = len(source["INPUTS/JET"])

    if args.strict and n_input_events != n_reco_events:
        raise ValueError(
            f"Length mismatch: input H5 has {n_input_events} events, "
            f"HyPER output has {n_reco_events} rows."
        )

    warnings_list: list[str] = []
    if n_input_events != n_reco_events:
        warning = (
            f"Length mismatch: input H5 has {n_input_events} events, "
            f"HyPER output has {n_reco_events} rows; processing the available prefix."
        )
        LOGGER.warning(warning)
        warnings_list.append(warning)

    n_processed = min(n_input_events, n_loaded_reco_events, n_reco_events)
    if args.max_events is not None:
        n_processed = min(n_processed, int(args.max_events))

    reco_for_processing, alignment_mode = align_reco_to_h5_prefix(
        hyper_df,
        reco,
        n_processed,
        args.strict,
        warnings_list,
    )

    copy_limit = n_processed if args.max_events is not None else None
    copy_input_h5(
        str(input_h5),
        args.output_h5,
        args.overwrite,
        n_input_events=n_input_events,
        event_limit=copy_limit,
        chunk_size=args.chunk_size,
    )
    with h5py.File(args.output_h5, "a") as output:
        n_output_events = len(output["INPUTS/JET"])
        decorate_start_time = time.monotonic()
        if args.topology == "ttbar_singlelep":
            datasets, debug_datasets, fields_written = create_decoration_datasets(
                output,
                n_output_events,
                args.chunk_size,
                args.overwrite,
                args.debug_mirror,
                args.write_debug_indices,
            )
            invalid_reasons: Counter = Counter()
            reco_counters: Counter = Counter()
            for start in range(0, n_processed, int(args.chunk_size)):
                stop = min(start + int(args.chunk_size), n_processed)
                reco_chunk = reco_for_processing.iloc[start:stop].reset_index(drop=True)
                jets_chunk = output["INPUTS/JET"][start:stop]
                leptons_chunk = output["INPUTS/LEPTON"][start:stop]
                met_chunk = output["INPUTS/MET"][start:stop]
                outputs, debug_indices, invalid_chunk = decorate_ttbar_singlelep(
                    jets_chunk,
                    leptons_chunk,
                    met_chunk,
                    reco_chunk,
                    n_output_events,
                    stop - start,
                    args.write_debug_indices,
                    output_events=stop - start,
                )
                invalid_reasons.update(invalid_chunk)
                reco_global = outputs["RECO_GLOBAL"]
                reco_counters["reco_valid"] += int(np.sum(reco_global["reco_valid"] == 1))
                reco_counters["thad_valid"] += int(np.sum(reco_global["thad_valid"] == 1))
                reco_counters["whad_valid"] += int(np.sum(reco_global["whad_valid"] == 1))
                reco_counters["b_lep_valid"] += int(np.sum(reco_global["b_lep_valid"] == 1))
                for name, arr in outputs.items():
                    datasets[name][start:stop] = arr
                    mirror = datasets.get(f"INPUTS_RECO/{name}")
                    if mirror is not None:
                        mirror[start:stop] = arr
                if debug_datasets is not None and debug_indices is not None:
                    for name, arr in debug_indices.items():
                        debug_datasets[name][start:stop] = arr
                elapsed = max(time.monotonic() - decorate_start_time, 1e-9)
                LOGGER.info(
                    "Decorated events %d:%d (%d/%d, %.1f events/s)",
                    start,
                    stop,
                    stop,
                    n_processed,
                    stop / elapsed,
                )
        elif args.topology == "ttH":
            datasets, debug_datasets, fields_written = create_tth_decoration_datasets(
                output,
                n_output_events,
                args.chunk_size,
                args.overwrite,
                args.debug_mirror,
                args.write_debug_indices,
            )
            invalid_reasons = Counter()
            reco_counters = Counter()
            for start in range(0, n_processed, int(args.chunk_size)):
                stop = min(start + int(args.chunk_size), n_processed)
                reco_chunk = reco_for_processing.iloc[start:stop].reset_index(drop=True)
                jets_chunk = output["INPUTS/JET"][start:stop]
                leptons_chunk = output["INPUTS/LEPTON"][start:stop]
                met_chunk = output["INPUTS/MET"][start:stop]
                outputs, debug_indices, invalid_chunk = decorate_ttH(
                    jets_chunk,
                    leptons_chunk,
                    met_chunk,
                    reco_chunk,
                    n_output_events,
                    stop - start,
                    args.write_debug_indices,
                    output_events=stop - start,
                )
                invalid_reasons.update(invalid_chunk)
                reco_global = outputs["RECO_GLOBAL"]
                reco_counters["reco_valid"] += int(np.sum(reco_global["reco_valid"] == 1))
                reco_counters["thad_valid"] += int(np.sum(reco_global["thad_valid"] == 1))
                reco_counters["whad_valid"] += int(np.sum(reco_global["whad_valid"] == 1))
                reco_counters["b_lep_valid"] += int(np.sum(reco_global["b_lep_valid"] == 1))
                reco_counters["higgs_valid"] += int(np.sum(reco_global["higgs_valid"] == 1))
                for name, arr in outputs.items():
                    datasets[name][start:stop] = arr
                    mirror = datasets.get(f"INPUTS_RECO/{name}")
                    if mirror is not None:
                        mirror[start:stop] = arr
                if debug_datasets is not None and debug_indices is not None:
                    for name, arr in debug_indices.items():
                        debug_datasets[name][start:stop] = arr
                elapsed = max(time.monotonic() - decorate_start_time, 1e-9)
                LOGGER.info(
                    "Decorated ttH events %d:%d (%d/%d, %.1f events/s)",
                    start,
                    stop,
                    stop,
                    n_processed,
                    stop / elapsed,
                )
        else:
            raise ValueError(f"Unsupported topology {args.topology}")

    summary = build_summary(
        args,
        n_input_events,
        n_reco_events,
        n_loaded_reco_events,
        n_output_events,
        n_processed,
        invalid_reasons,
        fields_written,
        warnings_list,
        alignment_mode,
        reco_counters=reco_counters,
    )
    summary_json = args.summary_json or f"{args.output_h5}.summary.json"
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
