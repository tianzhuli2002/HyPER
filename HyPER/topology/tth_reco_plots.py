#!/usr/bin/env python3
"""Standalone ttH single-lepton HyPER reconstruction evaluation plots.

ttH single-lepton role convention:

    1 = t_had_b
    2 = t_had_j1
    3 = t_had_j2
    4 = t_lep_b
    5 = h_b1
    6 = h_b2

Semantic reconstruction contract after ttH_single_lep():

    HyPER_best_tlep      = leptonic top candidate
    HyPER_best_thad      = hadronic top candidate
    HyPER_best_wlep      = leptonic W candidate
    HyPER_best_whad      = hadronic W candidate
    HyPER_best_higgs     = Higgs candidate
    HyPER_best_blep      = leptonic b-jet index
    HyPER_best_bhad      = hadronic b-jet index
    HyPER_best_whad_j1   = hadronic W jet index
    HyPER_best_whad_j2   = hadronic W jet index
    HyPER_best_higgs_b1  = Higgs b-jet index
    HyPER_best_higgs_b2  = Higgs b-jet index

Evaluation policy:

    S/B plots:
        all events with binary labels and finite classification score.

    Reconstruction efficiency:
        signal events only, fully matched, n_jets >= 6.

    Observable plots:
        signal events only, fully matched, n_jets >= 6, finite observable values.

    Background events:
        still reconstructed by ttH_single_lep() if raw outputs are provided and
        still written to observables.csv for diagnostics, but excluded from
        reconstruction efficiencies and observable plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prediction_io import coerce_source_indices, iter_hyper_prediction_parts, read_h5_rows, source_index_column  # noqa: E402
from joint_reco_plotting import (  # noqa: E402
    H5_LABEL_CANDIDATES,
    binary_labels_from_h5_data,
    binary_labels_from_h5_handle,
    category_masks,
    category_summary_from_counts,
    plot_joint_sb,
    plot_observable_pair,
    resolve_event_labels,
    write_category_diagnostics,
)
from tth import ttH_single_lep  # noqa: E402


LOGGER = logging.getLogger(__name__)


TTH_ROLE_MAP = {
    # From configs/ttH.yaml target labels:
    # tlep=(1-4,2-1,3-1), thad=(1-1,1-2,1-3), H=(1-5,1-6).
    "t_had_b": 1,
    "t_had_j1": 2,
    "t_had_j2": 3,
    "t_lep_b": 4,
    "h_b1": 5,
    "h_b2": 6,
}
TTH_ROLE_VALUES = set(TTH_ROLE_MAP.values())

NJET_BINS = ("overall", "6", "7", "8", "9", "10+")

SELECTED_COLUMNS = {
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

RAW_COLUMNS = {
    "HyPER_HE_IDX",
    "HyPER_HE_RAW",
    "HyPER_HE_VCT",
    "HyPER_GE_IDX",
    "HyPER_GE_RAW",
    "HyPER_GE_VCT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, help="Original ttH H5 file.")
    parser.add_argument("--prediction-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--score-field", default="HyPER_CLS_RAW")
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf"])
    parser.add_argument("--no-sb", action="store_true")
    parser.add_argument("--strict-length", action="store_true")
    parser.add_argument("--alignment", choices=["auto", "row_order", "source_index"], default="auto")
    parser.add_argument("--allow-duplicate-source-index", action="store_true")
    parser.add_argument(
        "--classification",
        action="store_true",
        help="Keep HyPER_CLS_RAW when reconstructing raw outputs.",
    )
    parser.add_argument(
        "--strategy",
        choices=["thad_first", "higgs_first", "score_global"],
        default="thad_first",
        help="ttH reconstruction assignment strategy.",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Evaluate thad_first, higgs_first, and score_global on the same events.",
    )
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--write-event-csv", action="store_true")
    parser.add_argument("--skip-event-csv", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=1000000)
    parser.add_argument(
        "--plot-scope",
        choices=["fully_matched", "signal_split", "all_split", "all"],
        default="fully_matched",
        help=(
            "Observable plotting scope. fully_matched preserves the existing "
            "signal fully matched selection; split modes add suffixed inclusive plots."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def valid_structured_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.names is None:
        if arr.ndim == 1:
            return arr[np.isfinite(arr)]
        return arr[np.isfinite(arr[:, 0])]
    first_field = arr.dtype.names[0]
    return arr[np.isfinite(arr[first_field])]


def get_field(row: np.void, candidates: tuple[str, ...]) -> float:
    if getattr(row.dtype, "names", None) is None:
        return float("nan")

    names = row.dtype.names or ()
    lower_to_name = {name.lower(): name for name in names}

    for candidate in candidates:
        name = lower_to_name.get(candidate.lower())
        if name is None:
            continue
        try:
            return float(row[name])
        except (TypeError, ValueError):
            return float("nan")

    return float("nan")


def row_to_e_eta_phi_pt(row: np.void) -> tuple[float, float, float, float]:
    e = get_field(row, ("e", "energy", "E"))
    eta = get_field(row, ("eta",))
    phi = get_field(row, ("phi",))

    if not math.isfinite(phi):
        sin_phi = get_field(row, ("sin_phi", "sinphi"))
        cos_phi = get_field(row, ("cos_phi", "cosphi"))
        if math.isfinite(sin_phi) and math.isfinite(cos_phi):
            phi = math.atan2(sin_phi, cos_phi)

    pt = get_field(row, ("pt", "pT"))
    return e, eta, phi, pt


def p4_from_row(row: np.void) -> np.ndarray:
    e, eta, phi, pt = row_to_e_eta_phi_pt(row)
    values = np.asarray([e, eta, phi, pt], dtype=float)

    if not np.all(np.isfinite(values)):
        return np.full(4, np.nan, dtype=float)

    return np.asarray(
        [e, pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)],
        dtype=float,
    )


def met_proxy_p4(row: np.void) -> np.ndarray:
    _, _, phi, pt = row_to_e_eta_phi_pt(row)

    if not (math.isfinite(pt) and math.isfinite(phi)):
        return np.full(4, np.nan, dtype=float)

    return np.asarray(
        [pt, pt * math.cos(phi), pt * math.sin(phi), 0.0],
        dtype=float,
    )


def invariant_mass(vec: np.ndarray) -> float:
    if not np.all(np.isfinite(vec)):
        return float("nan")

    e, px, py, pz = [float(x) for x in vec]
    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


def as_node_list(value: Any, expected_len: int | None = None) -> list[int] | None:
    if value is None:
        return None

    try:
        arr = np.asarray(value).astype(int).reshape(-1)
    except (TypeError, ValueError):
        return None

    if expected_len is not None and len(arr) != expected_len:
        return None

    if np.any(arr < 0):
        return None

    return [int(x) for x in arr]


def finite_float(value: Any) -> float:
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")

    return scalar if math.isfinite(scalar) else float("nan")


def normalise_predictions(
    predictions: pd.DataFrame,
    classification: bool,
    strategy: str,
) -> pd.DataFrame:
    """Return semantically ordered ttH SL selected columns."""
    if RAW_COLUMNS.issubset(predictions.columns):
        reco = ttH_single_lep(
            predictions,
            classification=classification or ("HyPER_CLS_RAW" in predictions.columns),
            strategy=strategy,
        )
    elif SELECTED_COLUMNS.issubset(predictions.columns):
        if strategy != "thad_first":
            raise KeyError(
                "Alternative ttH reconstruction strategies require raw HyPER columns. "
                "This prediction output only contains selected ttH columns."
            )
        reco = predictions.copy()
        if "reco_strategy" not in reco.columns:
            reco["reco_strategy"] = "selected"
    else:
        missing = sorted(RAW_COLUMNS - set(predictions.columns))
        if missing:
            raise KeyError(
                "Prediction output is neither selected ttH output nor raw HyPER output. "
                "Missing raw columns: " + ", ".join(missing)
            )

    return reco.reset_index(drop=True)


def align_prediction_rows_to_h5_prefix(
    predictions: pd.DataFrame,
    n_processed: int,
    start_event: int,
    strict: bool,
    warnings_list: list[str],
) -> tuple[pd.DataFrame, str]:
    """Verify or reorder prediction rows when a prefix event_index is available."""
    predictions = predictions.iloc[:n_processed].reset_index(drop=True)

    if "event_index" not in predictions.columns:
        warning = (
            "Prediction output has no event_index column; assuming row-order alignment "
            "with the H5 prefix."
        )
        LOGGER.warning(warning)
        warnings_list.append(warning)
        return predictions, "row_order_prefix_assumed"

    event_index = pd.to_numeric(predictions["event_index"], errors="coerce").to_numpy(dtype=float)

    if len(event_index) != n_processed or not np.all(np.isfinite(event_index)):
        warning = (
            "Prediction event_index is invalid for some rows; assuming row-order "
            "alignment with the H5 prefix."
        )
        if strict:
            raise ValueError(warning)
        LOGGER.warning(warning)
        warnings_list.append(warning)
        return predictions, "row_order_prefix_assumed_event_index_invalid"

    event_index_int = event_index.astype(np.int64)

    if not np.allclose(event_index, event_index_int):
        warning = (
            "Prediction event_index contains non-integer values; assuming row-order "
            "alignment with the H5 prefix."
        )
        if strict:
            raise ValueError(warning)
        LOGGER.warning(warning)
        warnings_list.append(warning)
        return predictions, "row_order_prefix_assumed_event_index_invalid"

    expected = np.arange(start_event, start_event + n_processed, dtype=np.int64)

    if np.array_equal(event_index_int, expected):
        return predictions, "event_index_prefix_verified"

    if np.array_equal(np.sort(event_index_int), expected):
        order = np.argsort(event_index_int)
        return predictions.iloc[order].reset_index(drop=True), "event_index_prefix_reordered"

    warning = (
        "Prediction event_index is not the copied H5 prefix for this chunk; assuming "
        "row-order alignment."
    )
    if strict:
        raise ValueError(warning)

    LOGGER.warning(warning)
    warnings_list.append(warning)
    return predictions, "row_order_prefix_assumed_event_index_nonprefix"


def read_h5_indexed(handle: h5py.File, source_indices: np.ndarray, chunk_size: int) -> dict[str, np.ndarray]:
    dataset_paths = {
        "jets": "INPUTS/JET",
        "leptons": "INPUTS/LEPTON",
        "met": "INPUTS/MET",
        "jet_labels": "LABELS/JET",
    }
    if "INPUTS/GLOBAL" in handle:
        dataset_paths["global_inputs"] = "INPUTS/GLOBAL"
    for candidate in H5_LABEL_CANDIDATES:
        if candidate in handle:
            dataset_paths[candidate] = candidate
    return read_h5_rows(handle, dataset_paths, source_indices, chunk_size=chunk_size)


def jet_multiplicity(global_row: np.ndarray | None, jet_rows: np.ndarray) -> int:
    if global_row is not None and getattr(global_row.dtype, "names", None):
        names = global_row.dtype.names or ()
        lower_to_name = {name.lower(): name for name in names}
        name = None
        for candidate in ("njet", "njets", "n_jets"):
            name = lower_to_name.get(candidate)
            if name is not None:
                break

        if name is not None:
            value = finite_float(global_row[name])
            if math.isfinite(value):
                return int(value)

    return int(len(jet_rows))


def njet_bin(n_jets: int) -> str:
    if n_jets >= 10:
        return "10+"
    if n_jets in {6, 7, 8, 9}:
        return str(n_jets)
    return "under6"


def label_set_for_indices(indices: list[int], labels: np.ndarray) -> set[int]:
    out: set[int] = set()

    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(labels):
            value = labels[idx]
            if np.isfinite(value):
                out.add(int(value))

    return out


def label_tuple_for_indices(indices: list[int], labels: np.ndarray) -> str:
    out = []

    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(labels):
            value = labels[idx]
            out.append(int(value) if np.isfinite(value) else -999)

    return str(tuple(out))


def label_at(index: int, labels: np.ndarray) -> int:
    index = int(index)
    if 0 <= index < len(labels) and np.isfinite(labels[index]):
        return int(labels[index])
    return -999


def sum_p4(rows: np.ndarray, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.full(4, np.nan, dtype=float)

    total = np.zeros(4, dtype=float)

    for idx in indices:
        idx = int(idx)
        if idx >= len(rows):
            return np.full(4, np.nan, dtype=float)

        p4 = p4_from_row(rows[idx])
        if not np.all(np.isfinite(p4)):
            return np.full(4, np.nan, dtype=float)

        total += p4

    return total


def truth_indices(jet_labels: np.ndarray) -> dict[str, list[int]]:
    labels = np.asarray(jet_labels).reshape(-1)
    return {
        role: [int(idx) for idx in np.where(labels == value)[0]]
        for role, value in TTH_ROLE_MAP.items()
    }


def evaluate_event(
    event_idx: int,
    row: pd.Series,
    jets_event: np.ndarray,
    leptons_event: np.ndarray,
    met_event: np.ndarray,
    jet_labels_event: np.ndarray,
    global_input_event: np.ndarray | None,
    is_signal: int | None,
) -> dict[str, Any]:
    jets = valid_structured_rows(jets_event)
    leptons = valid_structured_rows(leptons_event)
    met = valid_structured_rows(met_event)

    labels = np.asarray(jet_labels_event, dtype=float).reshape(-1)
    labels = labels[: len(jets)]

    njet = jet_multiplicity(global_input_event, jets)
    n_leps = len(leptons)
    n_met = min(len(met), 1)

    finite_truth_labels = {
        int(value)
        for value in labels
        if np.isfinite(value) and int(value) > 0
    }

    n_truth_roles_matched = len(finite_truth_labels & TTH_ROLE_VALUES)
    fully_matched = TTH_ROLE_VALUES.issubset(finite_truth_labels)
    if fully_matched:
        truth_match_category = "fully_matched"
    elif n_truth_roles_matched > 0:
        truth_match_category = "partially_matched"
    else:
        truth_match_category = "unmatched"

    if is_signal is None:
        is_signal_eval = int(fully_matched)
        signal_label_source = "fallback_fully_matched"
    else:
        is_signal_eval = int(is_signal == 1)
        signal_label_source = "event_label"

    reco_eval_event = int(is_signal_eval == 1 and fully_matched and njet >= 6)

    tlep_nodes = as_node_list(row.get("HyPER_best_tlep"), 3)
    thad_nodes = as_node_list(row.get("HyPER_best_thad"), 3)
    whad_nodes = as_node_list(row.get("HyPER_best_whad"), 2)
    higgs_nodes = as_node_list(row.get("HyPER_best_higgs"), 2)

    b_lep_idx = int(row.get("HyPER_best_blep", -1))
    b_had_idx = int(row.get("HyPER_best_bhad", -1))

    whad_indices = [
        int(row.get("HyPER_best_whad_j1", -1)),
        int(row.get("HyPER_best_whad_j2", -1)),
    ]
    higgs_indices = [
        int(row.get("HyPER_best_higgs_b1", -1)),
        int(row.get("HyPER_best_higgs_b2", -1)),
    ]

    whad_indices = [idx for idx in whad_indices if idx >= 0]
    higgs_indices = [idx for idx in higgs_indices if idx >= 0]

    thad_indices = [idx for idx in [b_had_idx] + whad_indices if idx >= 0]
    tlep_jet_indices = [b_lep_idx] if b_lep_idx >= 0 else []

    tlep_labels = label_set_for_indices(tlep_jet_indices, labels)
    thad_labels = label_set_for_indices(thad_indices, labels)
    whad_labels = label_set_for_indices(whad_indices, labels)
    higgs_labels = label_set_for_indices(higgs_indices, labels)

    b_lep_label = label_at(b_lep_idx, labels)
    b_had_label = label_at(b_had_idx, labels)
    whad_label_values = [label_at(idx, labels) for idx in whad_indices]
    higgs_label_values = [label_at(idx, labels) for idx in higgs_indices]
    thad_label_values = [label_at(idx, labels) for idx in thad_indices]

    b_lep_correct = b_lep_label == TTH_ROLE_MAP["t_lep_b"]

    b_had_correct = b_had_label == TTH_ROLE_MAP["t_had_b"]

    whad_correct = (
        len(whad_indices) == 2
        and set(whad_label_values)
        == {TTH_ROLE_MAP["t_had_j1"], TTH_ROLE_MAP["t_had_j2"]}
    )

    higgs_correct = (
        len(higgs_indices) == 2
        and set(higgs_label_values)
        == {TTH_ROLE_MAP["h_b1"], TTH_ROLE_MAP["h_b2"]}
    )

    thad_correct = b_had_correct and whad_correct
    event_correct = b_lep_correct and b_had_correct and whad_correct and higgs_correct

    whad_valid = len(whad_indices) == 2 and all(idx < len(jets) for idx in whad_indices)
    thad_valid = b_had_idx >= 0 and b_had_idx < len(jets) and whad_valid
    b_lep_valid = b_lep_idx >= 0 and b_lep_idx < len(jets)
    higgs_valid = len(higgs_indices) == 2 and all(idx < len(jets) for idx in higgs_indices)
    reco_valid = thad_valid and b_lep_valid and higgs_valid

    whad_p4 = sum_p4(jets, whad_indices) if whad_valid else np.full(4, np.nan)
    thad_p4 = (
        p4_from_row(jets[b_had_idx]) + whad_p4
        if thad_valid
        else np.full(4, np.nan)
    )
    higgs_p4 = (
        sum_p4(jets, higgs_indices)
        if higgs_valid
        else np.full(4, np.nan)
    )

    lep_idx = None
    met_idx = None

    if tlep_nodes is not None:
        for node in tlep_nodes:
            node = int(node)

            if len(jets) <= node < len(jets) + n_leps:
                lep_idx = node - len(jets)

            elif len(jets) + n_leps <= node < len(jets) + n_leps + n_met:
                met_idx = node - len(jets) - n_leps

    tlep_visible_valid = (
        b_lep_valid
        and lep_idx is not None
        and 0 <= lep_idx < n_leps
        and met_idx is not None
        and 0 <= met_idx < n_met
    )

    tlep_p4 = (
        p4_from_row(jets[b_lep_idx])
        + p4_from_row(leptons[lep_idx])
        + met_proxy_p4(met[met_idx])
        if tlep_visible_valid
        else np.full(4, np.nan)
    )

    tth_visible_p4 = (
        thad_p4 + tlep_p4 + higgs_p4
        if thad_valid and tlep_visible_valid and higgs_valid
        else np.full(4, np.nan)
    )

    truth = truth_indices(labels)

    truth_whad_indices = truth["t_had_j1"][:1] + truth["t_had_j2"][:1]
    truth_thad_indices = truth["t_had_b"][:1] + truth_whad_indices
    truth_higgs_indices = truth["h_b1"][:1] + truth["h_b2"][:1]

    truth_whad_p4 = (
        sum_p4(jets, truth_whad_indices)
        if len(truth_whad_indices) == 2
        else np.full(4, np.nan)
    )

    truth_thad_p4 = (
        sum_p4(jets, truth_thad_indices)
        if len(truth_thad_indices) == 3
        else np.full(4, np.nan)
    )

    truth_higgs_p4 = (
        sum_p4(jets, truth_higgs_indices)
        if len(truth_higgs_indices) == 2
        else np.full(4, np.nan)
    )

    truth_tlep_visible_valid = (
        len(truth["t_lep_b"]) >= 1
        and truth["t_lep_b"][0] < len(jets)
        and n_leps >= 1
        and n_met >= 1
    )

    truth_tlep_visible_p4 = (
        p4_from_row(jets[truth["t_lep_b"][0]])
        + p4_from_row(leptons[0])
        + met_proxy_p4(met[0])
        if truth_tlep_visible_valid
        else np.full(4, np.nan)
    )

    truth_tth_visible_p4 = (
        truth_thad_p4 + truth_tlep_visible_p4 + truth_higgs_p4
        if (
            np.all(np.isfinite(truth_thad_p4))
            and np.all(np.isfinite(truth_tlep_visible_p4))
            and np.all(np.isfinite(truth_higgs_p4))
        )
        else np.full(4, np.nan)
    )

    top_had_score = finite_float(row.get("HyPER_best_thad_prob", np.nan))
    top_lep_score = finite_float(row.get("HyPER_best_tlep_prob", np.nan))
    w_had_score = finite_float(row.get("HyPER_best_whad_prob", np.nan))
    w_lep_score = finite_float(row.get("HyPER_best_wlep_prob", np.nan))
    higgs_score = finite_float(row.get("HyPER_best_higgs_prob", np.nan))

    event_reco_score = finite_float(row.get("HyPER_best_event_score", np.nan))

    if not math.isfinite(event_reco_score):
        scores = [top_had_score, top_lep_score, w_had_score, w_lep_score, higgs_score]
        event_reco_score = (
            float(np.prod(scores))
            if all(math.isfinite(score) for score in scores)
            else float("nan")
        )

    return {
        "event_index": int(event_idx),
        "n_jets": int(njet),
        "njet_bin": njet_bin(njet),
        "is_signal": int(is_signal_eval),
        "signal_label_source": signal_label_source,
        "fully_matched": int(fully_matched),
        "n_truth_roles_matched": int(n_truth_roles_matched),
        "truth_match_category": truth_match_category,
        "reco_eval_event": int(reco_eval_event),
        "reco_valid": int(reco_valid),
        "thad_valid": int(thad_valid),
        "b_lep_valid": int(b_lep_valid),
        "tlep_valid": int(b_lep_valid),
        "whad_valid": int(whad_valid),
        "higgs_valid": int(higgs_valid),
        "tlep_visible_valid": int(tlep_visible_valid),
        "b_lep_correct": int(b_lep_correct),
        "b_had_correct": int(b_had_correct),
        "whad_correct": int(whad_correct),
        "thad_correct": int(thad_correct),
        "higgs_correct": int(higgs_correct),
        "event_correct": int(event_correct),
        "reco_strategy": str(row.get("reco_strategy", "")),
        "b_lep_idx": int(b_lep_idx),
        "b_had_idx": int(b_had_idx),
        "whad_j1_idx": whad_indices[0] if len(whad_indices) > 0 else -1,
        "whad_j2_idx": whad_indices[1] if len(whad_indices) > 1 else -1,
        "higgs_b1_idx": higgs_indices[0] if len(higgs_indices) > 0 else -1,
        "higgs_b2_idx": higgs_indices[1] if len(higgs_indices) > 1 else -1,
        "b_lep_label": int(b_lep_label),
        "b_had_label": int(b_had_label),
        "tlep_label_tuple": label_tuple_for_indices(tlep_jet_indices, labels),
        "thad_label_tuple": label_tuple_for_indices(thad_indices, labels),
        "whad_label_tuple": label_tuple_for_indices(whad_indices, labels),
        "higgs_label_tuple": label_tuple_for_indices(higgs_indices, labels),
        "higgs_stolen_by_thad": int(bool({5, 6}.intersection(thad_label_values))),
        "higgs_stolen_by_whad": int(bool({5, 6}.intersection(whad_label_values))),
        "higgs_candidate_contains_thad_truth": int(bool({1, 2, 3}.intersection(higgs_label_values))),
        "whad_candidate_contains_higgs_truth": int(bool({5, 6}.intersection(whad_label_values))),
        "bhad_candidate_is_higgs_truth": int(b_had_label in {5, 6}),
        "blep_candidate_is_higgs_truth": int(b_lep_label in {5, 6}),
        "m_Whad": invariant_mass(whad_p4),
        "m_thad": invariant_mass(thad_p4),
        "m_H": invariant_mass(higgs_p4),
        "m_tlep_visible": invariant_mass(tlep_p4),
        "m_ttH_visible": invariant_mass(tth_visible_p4),
        "m_Whad_truth": invariant_mass(truth_whad_p4),
        "m_thad_truth": invariant_mass(truth_thad_p4),
        "m_H_truth": invariant_mass(truth_higgs_p4),
        "m_tlep_visible_truth": invariant_mass(truth_tlep_visible_p4),
        "m_ttH_visible_truth": invariant_mass(truth_tth_visible_p4),
        "top_had_score": top_had_score,
        "top_lep_score": top_lep_score,
        "w_had_score": w_had_score,
        "w_lep_score": w_lep_score,
        "higgs_score": higgs_score,
        "event_reco_score": event_reco_score,
        "HyPER_CLS_RAW": finite_float(row.get("HyPER_CLS_RAW", np.nan)),
    }


def efficiency_rows(evaluation: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    base_mask = evaluation["reco_eval_event"].to_numpy(dtype=bool)

    metrics = (
        ("event_eff", "event_correct"),
        ("b_lep_eff", "b_lep_correct"),
        ("b_had_eff", "b_had_correct"),
        ("whad_eff", "whad_correct"),
        ("thad_eff", "thad_correct"),
        ("higgs_eff", "higgs_correct"),
        ("reco_valid_fraction", "reco_valid"),
    )

    for bin_name in NJET_BINS:
        if bin_name == "overall":
            mask = base_mask
        else:
            mask = base_mask & (evaluation["njet_bin"].to_numpy() == bin_name)

        subset = evaluation.loc[mask]
        n_events = int(len(subset))

        row: dict[str, Any] = {
            "njet_bin": bin_name,
            "n_events": n_events,
        }

        for name, column in metrics:
            row[name] = float(subset[column].mean()) if n_events else float("nan")

        rows.append(row)

    return rows


def update_efficiency_counters(
    counters: dict[str, Counter],
    evaluation: pd.DataFrame,
) -> None:
    base = evaluation.loc[evaluation["reco_eval_event"] == 1]
    metrics = (
        ("event_eff", "event_correct"),
        ("b_lep_eff", "b_lep_correct"),
        ("b_had_eff", "b_had_correct"),
        ("whad_eff", "whad_correct"),
        ("thad_eff", "thad_correct"),
        ("higgs_eff", "higgs_correct"),
        ("reco_valid_fraction", "reco_valid"),
    )
    for bin_name in NJET_BINS:
        subset = base if bin_name == "overall" else base.loc[base["njet_bin"] == bin_name]
        counter = counters.setdefault(bin_name, Counter())
        counter["n_events"] += int(len(subset))
        for name, column in metrics:
            counter[name] += int(np.sum(subset[column].to_numpy(dtype=int)))


def efficiency_rows_from_counters(
    counters: dict[str, Counter],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names = (
        "event_eff",
        "b_lep_eff",
        "b_had_eff",
        "whad_eff",
        "thad_eff",
        "higgs_eff",
        "reco_valid_fraction",
    )
    for bin_name in NJET_BINS:
        counter = counters.get(bin_name, Counter())
        n_events = int(counter.get("n_events", 0))
        row: dict[str, Any] = {"njet_bin": bin_name, "n_events": n_events}
        for name in metric_names:
            row[name] = (
                float(counter.get(name, 0) / n_events)
                if n_events
                else float("nan")
            )
        rows.append(row)
    return rows


def pattern_counts(evaluation: pd.DataFrame) -> dict[str, dict[str, int]]:
    base = evaluation.loc[evaluation["reco_eval_event"] == 1].copy()

    out: dict[str, dict[str, int]] = {}

    for column in (
        "b_lep_label",
        "b_had_label",
        "tlep_label_tuple",
        "thad_label_tuple",
        "whad_label_tuple",
        "higgs_label_tuple",
    ):
        out[column] = {
            str(key): int(value)
            for key, value in Counter(base[column].astype(str)).most_common(30)
        }

    out["whad_wrong_only"] = {
        str(key): int(value)
        for key, value in Counter(
            base.loc[base["whad_correct"] == 0, "whad_label_tuple"].astype(str)
        ).most_common(30)
    }

    out["thad_wrong_only"] = {
        str(key): int(value)
        for key, value in Counter(
            base.loc[base["thad_correct"] == 0, "thad_label_tuple"].astype(str)
        ).most_common(30)
    }

    out["higgs_wrong_only"] = {
        str(key): int(value)
        for key, value in Counter(
            base.loc[base["higgs_correct"] == 0, "higgs_label_tuple"].astype(str)
        ).most_common(30)
    }

    return out


def clean_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fields = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: clean_json_value(value) for key, value in row.items()})


def save_fig(output_dir: Path, name: str, formats: list[str]) -> None:
    for fmt in formats:
        plt.savefig(output_dir / f"{name}.{fmt}", bbox_inches="tight")
    plt.close()


def finite(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def binomial_error_fraction(eff: float, n: int) -> float:
    if n <= 0 or not math.isfinite(eff):
        return float("nan")
    return math.sqrt(max(eff * (1.0 - eff), 0.0) / n)


def plot_efficiency_bar(
    eff_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    rows = [row for row in eff_rows if row["njet_bin"] != "overall"]

    labels = [f"{row['njet_bin']} jets" for row in rows]
    values = np.asarray([100.0 * row["event_eff"] for row in rows], dtype=float)
    counts = np.asarray([row["n_events"] for row in rows], dtype=int)

    errors = np.asarray(
        [100.0 * binomial_error_fraction(row["event_eff"], row["n_events"]) for row in rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rows))

    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        edgecolor="black",
        alpha=0.9,
    )

    for bar, value, count in zip(bars, values, counts):
        if math.isfinite(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 1.5,
                f"{value:.1f}%\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reconstruction efficiency [%]")
    ax.set_ylim(0.0, 110.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_title("ttH single-lepton event reconstruction efficiency")

    fig.tight_layout()
    save_fig(output_dir, "efficiency_by_njets", formats)


def plot_component_efficiency_bars(
    eff_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    rows = [row for row in eff_rows if row["njet_bin"] != "overall"]

    components = [
        ("event_eff", "Event"),
        ("b_had_eff", r"$b_\mathrm{had}$"),
        ("b_lep_eff", r"$b_\mathrm{lep}$"),
        ("whad_eff", r"$W_\mathrm{had}$"),
        ("thad_eff", r"$t_\mathrm{had}$"),
        ("higgs_eff", r"$H\rightarrow b\bar{b}$"),
    ]

    x = np.arange(len(rows))
    width = 0.13

    fig, ax = plt.subplots(figsize=(11.5, 6))

    for i, (key, label) in enumerate(components):
        offset = (i - (len(components) - 1) / 2.0) * width
        values = np.asarray([100.0 * row[key] for row in rows], dtype=float)

        ax.bar(
            x + offset,
            values,
            width=width,
            label=label,
            edgecolor="black",
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['njet_bin']} jets" for row in rows])
    ax.set_ylabel("Efficiency [%]")
    ax.set_ylim(0.0, 110.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=3)
    ax.set_title("ttH single-lepton reconstruction component efficiencies")

    fig.tight_layout()
    save_fig(output_dir, "efficiency_components_by_njets", formats)


def plot_efficiency_per_jet_detailed(
    eff_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    rows = [row for row in eff_rows if row["njet_bin"] != "overall"]

    for row in rows:
        categories = [
            "overall",
            r"$b_\mathrm{had}$",
            r"$b_\mathrm{lep}$",
            r"$W_\mathrm{had}$",
            r"$t_\mathrm{had}$",
            r"$H\rightarrow b\bar{b}$",
        ]

        values = np.asarray(
            [
                row["event_eff"],
                row["b_had_eff"],
                row["b_lep_eff"],
                row["whad_eff"],
                row["thad_eff"],
                row["higgs_eff"],
            ],
            dtype=float,
        ) * 100.0

        n_events = int(row["n_events"])

        errors = np.asarray(
            [
                100.0 * binomial_error_fraction(value / 100.0, n_events)
                for value in values
            ],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(8.2, 5.8))
        x = np.arange(len(categories))

        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=4,
            edgecolor="black",
            alpha=0.9,
        )

        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 2.0,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel("Reconstruction efficiency [%]")
        ax.set_ylim(0.0, 110.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_title(f"ttH single-lepton, {row['njet_bin']} jets (n={n_events})")

        fig.tight_layout()

        safe_bin = str(row["njet_bin"]).replace("+", "plus")
        save_fig(output_dir, f"efficiency_components_{safe_bin}_jets", formats)


def plot_efficiencies(
    eff_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    plot_efficiency_bar(eff_rows, output_dir, formats)
    plot_component_efficiency_bars(eff_rows, output_dir, formats)
    plot_efficiency_per_jet_detailed(eff_rows, output_dir, formats)


def plot_hist(
    values: np.ndarray,
    xlabel: str,
    title: str,
    output_dir: Path,
    name: str,
    formats: list[str],
) -> None:
    values = finite(values)

    if len(values) == 0:
        return

    plt.figure(figsize=(6.0, 4.2))
    plt.hist(values, bins=50, histtype="step", linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.title(title)
    plt.tight_layout()
    save_fig(output_dir, name, formats)


def plot_scope_masks(evaluation: pd.DataFrame, plot_scope: str) -> dict[str, np.ndarray]:
    signal = evaluation["is_signal"].to_numpy(dtype=int) == 1
    background = evaluation["is_signal"].to_numpy(dtype=int) == 0
    fully_matched_signal = evaluation["reco_eval_event"].to_numpy(dtype=bool)
    unmatched_signal = signal & ~fully_matched_signal

    if plot_scope == "fully_matched":
        return {"fully_matched": fully_matched_signal}
    if plot_scope == "signal_split":
        return {
            "signal_fully_matched": fully_matched_signal,
            "signal_not_fully_matched": unmatched_signal,
        }
    if plot_scope == "all_split":
        return {
            "signal_fully_matched": fully_matched_signal,
            "signal_not_fully_matched": unmatched_signal,
            "background": background,
        }
    if plot_scope == "all":
        return {"all": np.ones(len(evaluation), dtype=bool)}
    raise ValueError(f"Unsupported plot scope: {plot_scope}")


def print_plot_scope_counts(evaluation: pd.DataFrame, plot_scope: str) -> dict[str, int | str]:
    masks = category_masks(evaluation)
    signal = masks["signal_fm"] | masks["signal_nonfm"]
    background = masks["background"]
    counts = {
        "plot_scope": plot_scope,
        "total_events": int(len(evaluation)),
        "signal_events": int(np.sum(signal)),
        "background_events": int(np.sum(background)),
        "fully_matched_signal_events": int(np.sum(masks["signal_fm"])),
        "unmatched_signal_events": int(np.sum(masks["signal_nonfm"])),
    }
    print("Plot category counts:", counts)
    LOGGER.info("Plot category counts: %s", counts)
    return counts


def plot_hist_by_scope(
    evaluation: pd.DataFrame,
    values: np.ndarray,
    xlabel: str,
    title: str,
    output_dir: Path,
    name: str,
    formats: list[str],
    plot_scope: str,
) -> None:
    masks = plot_scope_masks(evaluation, plot_scope)
    any_values = False
    plt.figure(figsize=(6.0, 4.2))
    for label, mask in masks.items():
        scoped = np.asarray(values, dtype=float)[mask]
        scoped = scoped[np.isfinite(scoped)]
        if len(scoped) == 0:
            continue
        any_values = True
        plt.hist(scoped, bins=50, histtype="step", linewidth=1.5, label=label.replace("_", " "))
    if not any_values:
        plt.close()
        return
    plt.xlabel(xlabel)
    plt.ylabel("Events")
    plt.title(title)
    if len(masks) > 1:
        plt.legend()
    plt.tight_layout()
    save_fig(output_dir, f"{name}_{plot_scope}", formats)


def plot_observables(
    evaluation: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    max_points: int,
    plot_scope: str = "fully_matched",
) -> None:
    if plot_scope == "fully_matched":
        evaluation = evaluation.loc[evaluation["reco_eval_event"] == 1].copy()
    else:
        evaluation = evaluation.copy()

    if len(evaluation) > max_points:
        evaluation = evaluation.sample(max_points, random_state=42)

    for name in ("m_Whad", "m_thad", "m_H", "m_tlep_visible", "m_ttH_visible"):
        if name not in evaluation.columns:
            continue

        values = evaluation[name].to_numpy(dtype=float)
        if plot_scope == "fully_matched":
            plot_hist(values, name, name, output_dir, name, formats)
        else:
            plot_hist_by_scope(evaluation, values, name, name, output_dir, name, formats, plot_scope)

        truth_name = f"{name}_truth"
        if truth_name in evaluation.columns:
            pred = evaluation[name].to_numpy(dtype=float)
            truth = evaluation[truth_name].to_numpy(dtype=float)

            mask = np.isfinite(pred) & np.isfinite(truth)

            if np.any(mask):
                resolution = pred[mask] - truth[mask]

                if plot_scope == "fully_matched":
                    plot_hist(
                        resolution,
                        f"{name} - truth [GeV]",
                        f"{name} resolution",
                        output_dir,
                        f"{name}_resolution",
                        formats,
                    )
                else:
                    resolution_values = np.full(len(evaluation), np.nan, dtype=float)
                    resolution_values[mask] = resolution
                    plot_hist_by_scope(
                        evaluation,
                        resolution_values,
                        f"{name} - truth [GeV]",
                        f"{name} resolution",
                        output_dir,
                        f"{name}_resolution",
                        formats,
                        plot_scope,
                    )

    for name in (
        "event_reco_score",
        "top_had_score",
        "top_lep_score",
        "w_had_score",
        "w_lep_score",
        "higgs_score",
    ):
        if name not in evaluation.columns:
            continue

        output_name = name if name != "event_reco_score" else "reco_score_distribution"

        values = evaluation[name].to_numpy(dtype=float)
        if plot_scope == "fully_matched":
            plot_hist(values, name, name, output_dir, output_name, formats)
        else:
            plot_hist_by_scope(evaluation, values, name, name, output_dir, output_name, formats, plot_scope)


def plot_standard_observable_families(
    evaluation: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> dict[str, dict[str, int]]:
    rows_used: dict[str, dict[str, int]] = {}
    for column in ("m_Whad", "m_thad", "m_H", "m_tlep_visible", "m_ttH_visible"):
        if column in evaluation.columns:
            rows_used[column] = plot_observable_pair(
                evaluation, column, column, column, column, 6, output_dir, formats
            )
    for column in (
        "event_reco_score",
        "top_had_score",
        "top_lep_score",
        "w_had_score",
        "w_lep_score",
        "higgs_score",
    ):
        if column in evaluation.columns:
            rows_used[column] = plot_observable_pair(
                evaluation, column, column, column, column, 6, output_dir, formats
            )
    return rows_used


def binary_roc(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    finite_mask = np.isfinite(scores)

    labels = labels[finite_mask].astype(int)
    scores = scores[finite_mask].astype(float)

    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))

    if positives == 0 or negatives == 0:
        return np.asarray([]), np.asarray([]), float("nan")

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]

    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    tpr = np.concatenate(([0.0], tps / positives, [1.0]))
    fpr = np.concatenate(([0.0], fps / negatives, [1.0]))

    return fpr, tpr, float(np.trapz(tpr, fpr))


def plot_sb(
    evaluation: pd.DataFrame,
    score_field: str,
    output_dir: Path,
    formats: list[str],
) -> dict[str, Any]:
    if score_field not in evaluation.columns:
        return {
            "available": False,
            "reason": f"missing score field {score_field}",
        }

    scores = evaluation[score_field].to_numpy(dtype=float)
    labels = evaluation["is_signal"].to_numpy(dtype=int)

    finite_mask = np.isfinite(scores)

    if not np.any(finite_mask):
        return {
            "available": False,
            "reason": f"score field {score_field} is all NaN",
        }

    unique_labels = sorted(set(labels[finite_mask].astype(int).tolist()))

    if unique_labels != [0, 1]:
        return {
            "available": False,
            "reason": f"binary labels do not contain both classes: {unique_labels}",
        }

    plt.figure(figsize=(6.0, 4.2))
    bins = np.linspace(0.0, 1.0, 41)

    for label, title in ((0, "Background"), (1, "Signal")):
        values = scores[(labels == label) & np.isfinite(scores)]

        if len(values):
            plt.hist(
                values,
                bins=bins,
                histtype="step",
                density=True,
                label=title,
            )

    plt.xlabel(score_field)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    save_fig(output_dir, "sb_score_distribution", formats)

    fpr, tpr, auc = binary_roc(labels, scores)

    if len(fpr):
        plt.figure(figsize=(5.2, 5.0))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("Background efficiency")
        plt.ylabel("Signal efficiency")
        plt.legend()
        plt.tight_layout()
        save_fig(output_dir, "roc_curve", formats)

    return {
        "available": True,
        "score_field": score_field,
        "n_events": int(np.sum(finite_mask)),
        "n_signal": int(np.sum(labels == 1)),
        "n_background": int(np.sum(labels == 0)),
        "auc": auc,
        "mean_score_signal": float(np.nanmean(scores[labels == 1])),
        "mean_score_background": float(np.nanmean(scores[labels == 0])),
    }


def append_csv(path: Path, rows: pd.DataFrame, write_header: bool) -> None:
    rows.to_csv(path, mode="a", index=False, header=write_header)


def run_strategy_comparison(args: argparse.Namespace, output_dir: Path) -> None:
    rows = []
    for strategy in ("thad_first", "higgs_first", "score_global"):
        strategy_dir = output_dir / strategy
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--h5",
            str(args.h5),
            "--prediction-output",
            str(args.prediction_output),
            "--output-dir",
            str(strategy_dir),
            "--strategy",
            strategy,
            "--chunk-size",
            str(args.chunk_size),
            "--max-plot-points",
            str(args.max_plot_points),
            "--log-level",
            str(args.log_level),
            "--score-field",
            str(args.score_field),
            "--alignment",
            str(args.alignment),
            "--formats",
            *[str(fmt) for fmt in args.formats],
        ]
        if args.max_events is not None:
            cmd.extend(["--max-events", str(args.max_events)])
        if args.label_field is not None:
            cmd.extend(["--label-field", str(args.label_field)])
        if args.no_sb:
            cmd.append("--no-sb")
        if args.strict_length:
            cmd.append("--strict-length")
        if args.allow_duplicate_source_index:
            cmd.append("--allow-duplicate-source-index")
        if args.classification:
            cmd.append("--classification")
        if args.write_event_csv:
            cmd.append("--write-event-csv")
        if args.skip_event_csv:
            cmd.append("--skip-event-csv")

        subprocess.run(cmd, check=True)

        with (strategy_dir / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        overall = next(
            row for row in summary["efficiency_by_njets"] if row["njet_bin"] == "overall"
        )
        rows.append(
            {
                "strategy": strategy,
                "n_processed": int(summary["n_processed"]),
                "reco_eval_events": int(summary["reco_eval_events"]),
                "event_eff": overall["event_eff"],
                "b_lep_eff": overall["b_lep_eff"],
                "b_had_eff": overall["b_had_eff"],
                "whad_eff": overall["whad_eff"],
                "thad_eff": overall["thad_eff"],
                "higgs_eff": overall["higgs_eff"],
                "reco_valid_fraction": overall["reco_valid_fraction"],
                "higgs_stolen_by_thad_count": summary["diagnostics"].get("higgs_stolen_by_thad_count"),
                "higgs_stolen_by_whad_count": summary["diagnostics"].get("higgs_stolen_by_whad_count"),
                "higgs_candidate_contains_thad_truth_count": summary["diagnostics"].get("higgs_candidate_contains_thad_truth_count"),
                "whad_candidate_contains_higgs_truth_count": summary["diagnostics"].get("whad_candidate_contains_higgs_truth_count"),
                "bhad_candidate_is_higgs_truth_count": summary["diagnostics"].get("bhad_candidate_is_higgs_truth_count"),
                "blep_candidate_is_higgs_truth_count": summary["diagnostics"].get("blep_candidate_is_higgs_truth_count"),
            }
        )

    write_csv(output_dir / "strategy_comparison.csv", rows)
    with (output_dir / "strategy_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True, default=clean_json_value)
        handle.write("\n")


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_strategies:
        run_strategy_comparison(args, output_dir)
        LOGGER.info("Wrote strategy comparison to %s", output_dir)
        return

    warnings_list: list[str] = []
    rows: list[dict[str, Any]] = []

    n_reco_events = 0
    n_loaded_events = 0
    n_processed = 0
    n_signal_total = 0
    n_background_total = 0
    n_signal_fm_total = 0
    n_signal_nonfm_total = 0
    n_signal_partial_total = 0
    n_signal_unmatched_total = 0
    n_reco_eval_total = 0
    efficiency_counters: dict[str, Counter] = {}

    alignment_modes = Counter()
    alignment_mode = None
    label_source = None

    event_csv_path = output_dir / "observables.csv"
    event_csv_written = False
    write_event_csv = bool(args.write_event_csv and not args.skip_event_csv)

    with h5py.File(args.h5, "r") as handle:
        for required in ("INPUTS/JET", "INPUTS/LEPTON", "INPUTS/MET", "LABELS/JET"):
            if required not in handle:
                raise KeyError(f"Missing required dataset {required}")

        n_h5_events = len(handle["INPUTS/JET"])

        start = 0

        for prediction_chunk in iter_hyper_prediction_parts(
            args.prediction_output,
            max_events=args.max_events,
            chunk_size=args.chunk_size,
        ):
            if len(prediction_chunk) == 0:
                continue

            if args.max_events is not None and n_processed >= int(args.max_events):
                break

            if n_reco_events == 0:
                n_reco_events = int(
                    prediction_chunk.attrs.get("hyper_total_rows", len(prediction_chunk))
                )
                if args.alignment == "auto":
                    if n_h5_events == n_reco_events:
                        alignment_mode = "row_order"
                    elif source_index_column(prediction_chunk) is not None:
                        alignment_mode = "source_index"
                    else:
                        raise ValueError(
                            "Length mismatch and no HYPER_SOURCE_INDEX available. Re-run prediction "
                            "with source-index export enabled, or predict the full H5."
                        )
                elif args.alignment == "row_order":
                    alignment_mode = "row_order"
                else:
                    alignment_mode = "source_index"
                if alignment_mode == "row_order" and n_h5_events != n_reco_events:
                    raise ValueError(
                        f"Length mismatch: H5 has {n_h5_events} events, prediction has {n_reco_events} rows"
                    )

            reco_chunk = normalise_predictions(
                prediction_chunk,
                classification=args.classification,
                strategy=args.strategy,
            )

            n_loaded_events += len(reco_chunk)

            assert alignment_mode is not None
            if alignment_mode == "row_order":
                stop = min(start + len(reco_chunk), n_h5_events)
                if args.max_events is not None:
                    stop = min(stop, int(args.max_events))
                n_process = stop - start
            else:
                n_process = len(reco_chunk)

            if n_process <= 0:
                break

            if alignment_mode == "row_order":
                aligned, mode = align_prediction_rows_to_h5_prefix(
                    reco_chunk,
                    n_process,
                    start,
                    args.strict_length,
                    warnings_list,
                )
                source_indices = np.arange(start, start + n_process, dtype=np.int64)
                jets = handle["INPUTS/JET"][start:stop]
                leptons = handle["INPUTS/LEPTON"][start:stop]
                met = handle["INPUTS/MET"][start:stop]
                jet_labels = handle["LABELS/JET"][start:stop]

                global_inputs = (
                    handle["INPUTS/GLOBAL"][start:stop]
                    if "INPUTS/GLOBAL" in handle
                    else None
                )

                h5_labels, h5_label_source = binary_labels_from_h5_handle(
                    handle,
                    start,
                    stop,
                    args.label_field,
                )
            else:
                aligned = reco_chunk.iloc[:n_process].reset_index(drop=True)
                mode = "source_index"
                source_indices, index_warnings = coerce_source_indices(
                    prediction_chunk.iloc[:n_process],
                    n_h5_events=n_h5_events,
                    allow_duplicates=bool(args.allow_duplicate_source_index),
                )
                for warning in index_warnings:
                    if warning not in warnings_list:
                        warnings_list.append(warning)
                        LOGGER.warning(warning)
                data = read_h5_indexed(handle, source_indices, chunk_size=args.chunk_size)
                jets = data["jets"]
                leptons = data["leptons"]
                met = data["met"]
                jet_labels = data["jet_labels"]
                global_inputs = data.get("global_inputs")
                h5_labels, h5_label_source = binary_labels_from_h5_data(
                    data,
                    args.label_field,
                )
            signal_labels, this_label_source = resolve_event_labels(
                prediction_chunk.iloc[:n_process],
                h5_labels,
                h5_label_source,
                args.label_field,
                warnings_list,
                context=f"prediction rows {n_processed}:{n_processed + n_process}",
            )
            alignment_modes[mode] += 1

            label_source = label_source or this_label_source

            if signal_labels is None and not any("No event-level binary label found" in w for w in warnings_list):
                warning = (
                    "No event-level binary label found. Reconstruction efficiency will "
                    "fall back to fully_matched == 1 as the signal mask."
                )
                LOGGER.warning(warning)
                warnings_list.append(warning)

            chunk_rows = []

            for offset in range(n_process):
                global_row = (
                    global_inputs[offset, 0]
                    if global_inputs is not None and getattr(global_inputs, "ndim", 0) > 1
                    else None
                )

                is_signal = None if signal_labels is None else int(signal_labels[offset])

                chunk_rows.append(
                    evaluate_event(
                        event_idx=int(source_indices[offset]),
                        row=aligned.iloc[offset],
                        jets_event=jets[offset],
                        leptons_event=leptons[offset],
                        met_event=met[offset],
                        jet_labels_event=jet_labels[offset],
                        global_input_event=global_row,
                        is_signal=is_signal,
                    )
                )

            evaluation_chunk = pd.DataFrame(chunk_rows)
            chunk_masks = category_masks(evaluation_chunk)
            n_signal_fm_total += int(np.sum(chunk_masks["signal_fm"]))
            n_signal_nonfm_total += int(np.sum(chunk_masks["signal_nonfm"]))
            n_signal_partial_total += int(np.sum(chunk_masks["signal_partial"]))
            n_signal_unmatched_total += int(np.sum(chunk_masks["signal_unmatched"]))
            n_background_total += int(np.sum(chunk_masks["background"]))
            n_signal_total += int(
                np.sum(chunk_masks["signal_fm"] | chunk_masks["signal_nonfm"])
            )
            n_reco_eval_total += int(
                np.sum(evaluation_chunk["reco_eval_event"].to_numpy(dtype=bool))
            )
            update_efficiency_counters(efficiency_counters, evaluation_chunk)

            if write_event_csv:
                append_csv(
                    event_csv_path,
                    evaluation_chunk,
                    write_header=not event_csv_written,
                )
                event_csv_written = True

            if len(rows) < int(args.max_plot_points):
                keep = min(int(args.max_plot_points) - len(rows), len(chunk_rows))
                rows.extend(chunk_rows[:keep])

            n_processed += n_process
            if alignment_mode == "row_order":
                start = stop
                LOGGER.info("Processed plotting events %d:%d", start - n_process, stop)
            else:
                LOGGER.info(
                    "Processed plotting prediction rows %d:%d using source_index range %d:%d",
                    n_processed - n_process,
                    n_processed,
                    int(source_indices.min()) if len(source_indices) else -1,
                    int(source_indices.max()) if len(source_indices) else -1,
                )

            if args.max_events is not None and n_processed >= int(args.max_events):
                break

    if n_reco_events == 0:
        n_reco_events = n_loaded_events

    if args.strict_length and alignment_mode == "row_order" and n_h5_events != n_reco_events:
        raise ValueError(
            f"Length mismatch: H5 has {n_h5_events} events, prediction has {n_reco_events} rows"
        )
    if n_h5_events != n_reco_events:
        warning = (
            f"Source H5 has {n_h5_events} events; prediction has {n_reco_events} rows; "
            f"processed {n_processed} rows using {alignment_mode} alignment with chunk "
            f"size {args.chunk_size}."
        )
        LOGGER.warning(warning)
        warnings_list.append(warning)

    evaluation = pd.DataFrame(rows)

    if evaluation.empty:
        raise RuntimeError("No events were evaluated.")

    eff_rows = efficiency_rows_from_counters(efficiency_counters)
    eff = pd.DataFrame(eff_rows)

    eff.to_csv(output_dir / "efficiency_by_njets.csv", index=False)

    if not event_csv_written and not args.skip_event_csv:
        if args.write_event_csv or len(evaluation) <= min(args.max_plot_points, 100000):
            evaluation.to_csv(output_dir / "observables.csv", index=False)
            event_csv_written = True

    plot_counts = print_plot_scope_counts(evaluation, args.plot_scope)
    plot_efficiencies(eff_rows, output_dir, args.formats)
    observable_rows_used = plot_standard_observable_families(
        evaluation, output_dir, args.formats
    )

    sb_summary = {"available": False, "reason": "disabled"}
    if not args.no_sb:
        sb_summary = plot_joint_sb(
            evaluation, args.score_field, output_dir, args.formats
        )

    category_summary = category_summary_from_counts(
        n_processed,
        n_signal_total,
        n_background_total,
        n_signal_fm_total,
        n_signal_nonfm_total,
        n_reco_eval_total,
        n_signal_partial_total,
        n_signal_unmatched_total,
        label_source or "fallback_fully_matched",
        label_source is None,
    )
    write_category_diagnostics(category_summary, output_dir, args.formats)

    overall = next(row for row in eff_rows if row["njet_bin"] == "overall")

    write_csv(
        output_dir / "summary.csv",
        [
            {
                "topology": "ttH_singlelep",
                "strategy": args.strategy,
                "efficiency_denominator": (
                    "signal or fully matched fallback, fully matched six ttH jet roles, n_jets >= 6"
                ),
                "processed_events": int(n_processed),
                "sampled_plot_events": int(len(evaluation)),
                "signal_events": int(np.sum(evaluation["is_signal"] == 1)),
                "background_events": int(np.sum(evaluation["is_signal"] == 0)),
                "fully_matched_events": int(np.sum(evaluation["fully_matched"] == 1)),
                "reco_eval_events": int(np.sum(evaluation["reco_eval_event"] == 1)),
                "reco_eval_fraction": float(np.mean(evaluation["reco_eval_event"] == 1)),
                "event_eff": overall["event_eff"],
                "b_lep_eff": overall["b_lep_eff"],
                "b_had_eff": overall["b_had_eff"],
                "whad_eff": overall["whad_eff"],
                "thad_eff": overall["thad_eff"],
                "higgs_eff": overall["higgs_eff"],
                "reco_valid_fraction": overall["reco_valid_fraction"],
                "plot_scope": args.plot_scope,
                "sb_status": (
                    "written"
                    if sb_summary.get("available")
                    else sb_summary.get("reason")
                ),
            }
        ],
    )

    summary = {
        "topology": "ttH_singlelep",
        "strategy": args.strategy,
        "input_h5": str(args.h5),
        "prediction_output": str(args.prediction_output),
        "output_dir": str(args.output_dir),
        "n_h5_events": int(n_h5_events),
        "n_reco_events": int(n_reco_events),
        "n_loaded_events": int(n_loaded_events),
        "n_processed": int(n_processed),
        "plot_sample_events": int(len(evaluation)),
        "event_csv": "written" if event_csv_written else "skipped",
        "max_events": args.max_events,
        "max_plot_points": int(args.max_plot_points),
        "plot_scope": args.plot_scope,
        "observable_scope": [
            "signal_fm_with_n_jets_gte_6",
            "signal_fm_vs_signal_nonfm_vs_background_with_n_jets_gte_6",
        ],
        "score_scope": sb_summary.get("score_scope", []),
        "plot_category_counts": plot_counts,
        "strict_length": bool(args.strict_length),
        "chunk_size": int(args.chunk_size),
        "event_alignment_modes": {key: int(value) for key, value in alignment_modes.items()},
        "event_alignment_policy": (
            "auto uses row_order only when prediction length equals H5 length; "
            "otherwise HYPER_SOURCE_INDEX is required and source-index H5 rows are read."
        ),
        "label_source": label_source or "fallback_fully_matched",
        "fallback_fully_matched_used": label_source is None,
        "signal_events": int(np.sum(evaluation["is_signal"] == 1)),
        "background_events": int(np.sum(evaluation["is_signal"] == 0)),
        "n_total_rows": category_summary["n_total_rows"],
        "n_signal": category_summary["n_signal"],
        "n_background": category_summary["n_background"],
        "n_signal_fm": category_summary["n_signal_fm"],
        "n_signal_nonfm": category_summary["n_signal_nonfm"],
        "n_signal_partial": category_summary["n_signal_partial"],
        "n_signal_unmatched": category_summary["n_signal_unmatched"],
        "n_non_fully_matched_signal": category_summary["n_non_fully_matched_signal"],
        "n_partially_matched_signal": category_summary["n_partially_matched_signal"],
        "n_unmatched_signal": category_summary["n_unmatched_signal"],
        "n_reco_eval_event": category_summary["n_reco_eval_event"],
        "category_fractions": category_summary,
        "fully_matched_events": int(np.sum(evaluation["fully_matched"] == 1)),
        "fully_matched_fraction": float(np.mean(evaluation["fully_matched"] == 1)),
        "reco_eval_events": int(np.sum(evaluation["reco_eval_event"] == 1)),
        "reco_eval_fraction": float(np.mean(evaluation["reco_eval_event"] == 1)),
        "evaluation_policy": {
            "sb_plots": "all events with binary labels and finite classification score",
            "reconstruction_efficiency": (
                "event-level signal label == 1, fully_matched == 1, and n_jets >= 6; "
                "if no event label is available, signal label falls back to fully_matched"
            ),
            "observable_plots": (
                "same as reconstruction_efficiency, with finite pred/truth observable "
                "values where relevant"
            ),
            "background_events": (
                "can be streamed to observables.csv with --write-event-csv but are "
                "excluded from reconstruction efficiencies and observable plots"
            ),
        },
        "semantic_contract": {
            "truth_role_map": TTH_ROLE_MAP,
            "node_order": "jets first, then leptons, then MET",
            "HyPER_best_tlep": "leptonic top candidate after ttH_single_lep()",
            "HyPER_best_thad": "hadronic top candidate after ttH_single_lep()",
            "HyPER_best_wlep": "leptonic W candidate after ttH_single_lep()",
            "HyPER_best_whad": "hadronic W candidate after ttH_single_lep()",
            "HyPER_best_higgs": "Higgs candidate after ttH_single_lep()",
            "selected_columns": sorted(SELECTED_COLUMNS),
            "raw_columns": sorted(RAW_COLUMNS),
        },
        "efficiency_logic": {
            "efficiency_denominator": (
                "signal or fully matched fallback, fully matched six ttH jet roles, n_jets >= 6"
            ),
            "b_lep_eff": "selected HyPER_best_blep has truth label 4",
            "b_had_eff": "selected HyPER_best_bhad has truth label 1",
            "whad_eff": "selected Whad jets have labels {2, 3}, order-insensitive",
            "higgs_eff": "selected Higgs jets have labels {5, 6}, order-insensitive",
            "thad_eff": "b_had_eff and whad_eff",
            "event_eff": "b_lep_eff and b_had_eff and whad_eff and higgs_eff",
        },
        "diagnostics": {
            "higgs_stolen_by_thad_count": int(evaluation.loc[evaluation["reco_eval_event"] == 1, "higgs_stolen_by_thad"].sum()),
            "higgs_stolen_by_whad_count": int(evaluation.loc[evaluation["reco_eval_event"] == 1, "higgs_stolen_by_whad"].sum()),
            "higgs_candidate_contains_thad_truth_count": int(evaluation.loc[evaluation["reco_eval_event"] == 1, "higgs_candidate_contains_thad_truth"].sum()),
            "whad_candidate_contains_higgs_truth_count": int(evaluation.loc[evaluation["reco_eval_event"] == 1, "whad_candidate_contains_higgs_truth"].sum()),
            "bhad_candidate_is_higgs_truth_count": int(evaluation.loc[evaluation["reco_eval_event"] == 1, "bhad_candidate_is_higgs_truth"].sum()),
            "blep_candidate_is_higgs_truth_count": int(evaluation.loc[evaluation["reco_eval_event"] == 1, "blep_candidate_is_higgs_truth"].sum()),
            "label_pattern_counts": pattern_counts(evaluation),
            "interpretation": {
                "whad_label_tuple == (2, 3) or (3, 2)": "correct hadronic W",
                "higgs_label_tuple == (5, 6) or (6, 5)": "correct Higgs candidate",
                "thad_label_tuple contains 1, 2, 3": "correct hadronic top candidate",
                "tlep_label_tuple contains 4": "correct leptonic b candidate",
            },
        },
        "role_map": TTH_ROLE_MAP,
        "jet_bins": list(NJET_BINS),
        "event_reco_score_definition": (
            "product(top_had_score, top_lep_score, w_had_score, w_lep_score, "
            "higgs_score) when all five are finite"
        ),
        "truth_visible_lepton_met_note": (
            "Visible leptonic masses use truth t_lep_b plus first valid lepton and MET "
            "proxy; no neutrino is inferred."
        ),
        "efficiency_by_njets": [
            {key: clean_json_value(value) for key, value in row.items()}
            for row in eff_rows
        ],
        "component_counts": {
            column: int(evaluation[column].sum())
            for column in (
                "event_correct",
                "thad_correct",
                "higgs_correct",
                "b_lep_correct",
                "b_had_correct",
                "whad_correct",
                "reco_valid",
            )
        },
        "n_rows_used_per_plot_family": {
            "score": int(sb_summary.get("n_rows_with_finite_score", 0)),
            "observables": observable_rows_used,
            "truth_correct_efficiency": int(n_reco_eval_total),
        },
        "sb_summary": {
            key: clean_json_value(value)
            for key, value in sb_summary.items()
        },
        "sb_status": (
            "written"
            if sb_summary.get("available")
            else sb_summary.get("reason")
        ),
        "warnings": warnings_list,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=clean_json_value)
        handle.write("\n")

    LOGGER.info("Wrote ttH single-lepton reconstruction evaluation to %s", output_dir)

    print(json.dumps(summary, indent=2, sort_keys=True, default=clean_json_value))


if __name__ == "__main__":
    main()
