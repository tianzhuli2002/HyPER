#!/usr/bin/env python3
"""Standalone ttbar single-lepton HyPER reconstruction evaluation plots.

Legacy ttbar SL role convention:

    1 = b_lep
    2 = b_had
    3 = W_HAD_J1
    4 = W_HAD_J2

Semantic reconstruction contract after ttbar_single_lep():

    HyPER_best_top1 = leptonic top
    HyPER_best_top2 = hadronic top
    HyPER_best_w1   = leptonic W edge
    HyPER_best_w2   = hadronic W edge

If raw HyPER outputs are provided, this script first calls ttbar_single_lep().

Evaluation policy:

    S/B plots:
        all events with binary labels and finite classification score.

    Reconstruction efficiency:
        signal events only, fully matched, n_jets >= 4.

    Observable plots:
        signal events only, fully matched, n_jets >= 4, finite observable values.

    Background events:
        still reconstructed by ttbar_single_lep() if raw outputs are provided and
        still written to observables.csv for diagnostics, but excluded from
        reconstruction efficiencies and observable plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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

from prediction_io import iter_hyper_prediction_parts, load_hyper_prediction_output  # noqa: E402
from ttbar import ttbar_single_lep  # noqa: E402


LOGGER = logging.getLogger(__name__)

TTBAR_SL_ROLE_MAP = {
    "b_lep": 1,
    "b_had": 2,
    "whad_j1": 3,
    "whad_j2": 4,
}

NJET_BINS = ("overall", "4", "5", "6", "7", "8", "9+")

LABEL_CANDIDATES = (
    "LABELS/GLOBAL",
    "LABELS/SIGNAL",
    "LABELS/Y",
    "LABELS/CLASS",
    "LABELS/label",
)

SELECTED_COLUMNS = {
    "HyPER_best_top1",
    "HyPER_best_top2",
    "HyPER_best_w1",
    "HyPER_best_w2",
    "HyPER_best_top1_prob",
    "HyPER_best_top2_prob",
    "HyPER_best_w1_prob",
    "HyPER_best_w2_prob",
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
    parser.add_argument("--h5", required=True, help="Original ttbar SL H5 file.")
    parser.add_argument("--prediction-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--score-field", default="HyPER_CLS_RAW")
    parser.add_argument("--label-field", default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf"])
    parser.add_argument("--no-sb", action="store_true")
    parser.add_argument("--strict-length", action="store_true")
    parser.add_argument(
        "--classification",
        action="store_true",
        help="Keep HyPER_CLS_RAW when reconstructing raw outputs.",
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
    return np.asarray([pt, pt * math.cos(phi), pt * math.sin(phi), 0.0], dtype=float)


def invariant_mass(vec: np.ndarray) -> float:
    if not np.all(np.isfinite(vec)):
        return float("nan")
    e, px, py, pz = [float(x) for x in vec]
    return math.sqrt(max(e * e - px * px - py * py - pz * pz, 0.0))


def as_node_list(value: Any, expected_len: int) -> list[int] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value).astype(int).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(arr) != expected_len or np.any(arr < 0):
        return None
    return [int(x) for x in arr]


def finite_float(value: Any) -> float:
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float("nan")
    return scalar if math.isfinite(scalar) else float("nan")


def normalise_predictions(predictions: pd.DataFrame, classification: bool) -> pd.DataFrame:
    """Return semantically ordered ttbar SL selected columns."""
    if SELECTED_COLUMNS.issubset(predictions.columns):
        reco = predictions.copy()
    else:
        missing = sorted(RAW_COLUMNS - set(predictions.columns))
        if missing:
            raise KeyError(
                "Prediction output is neither selected ttbar SL output nor raw HyPER output. "
                "Missing raw columns: " + ", ".join(missing)
            )

        reco = ttbar_single_lep(
            predictions,
            classification=classification or ("HyPER_CLS_RAW" in predictions.columns),
        )

    if "thad_first" not in reco.columns:
        reco["thad_first"] = np.nan
    if "HyPER_CLS_RAW" not in reco.columns:
        reco["HyPER_CLS_RAW"] = np.nan

    return reco.reset_index(drop=True)


def align_prediction_rows_to_h5_prefix(
    predictions: pd.DataFrame,
    n_processed: int,
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
            "Prediction event_index is missing or non-finite for some processed rows; "
            "assuming row-order alignment with the H5 prefix."
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

    expected = np.arange(n_processed, dtype=np.int64)
    if np.array_equal(event_index_int, expected):
        return predictions, "event_index_prefix_verified"

    if np.array_equal(np.sort(event_index_int), expected):
        order = np.argsort(event_index_int)
        return predictions.iloc[order].reset_index(drop=True), "event_index_prefix_reordered"

    warning = (
        "Prediction event_index is not the H5 prefix 0..n_processed-1. This plotting "
        "tool reads the H5 prefix, so arbitrary sampled predictions need the matching "
        "sampled H5. Assuming row-order alignment for this run."
    )
    if strict:
        raise ValueError(warning)
    LOGGER.warning(warning)
    warnings_list.append(warning)
    return predictions, "row_order_prefix_assumed_event_index_nonprefix"


def read_h5_prefix(path: str | Path, n_events: int) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        for required in ("INPUTS/JET", "INPUTS/LEPTON", "INPUTS/MET", "LABELS/JET"):
            if required not in handle:
                raise KeyError(f"Missing required dataset {required}")

        data = {
            "jets": handle["INPUTS/JET"][:n_events],
            "leptons": handle["INPUTS/LEPTON"][:n_events],
            "met": handle["INPUTS/MET"][:n_events],
            "jet_labels": handle["LABELS/JET"][:n_events],
        }

        if "INPUTS/GLOBAL" in handle:
            data["global_inputs"] = handle["INPUTS/GLOBAL"][:n_events]

        for candidate in LABEL_CANDIDATES:
            if candidate in handle:
                data[candidate] = handle[candidate][:n_events]

    return data


def flatten_labels(values: np.ndarray) -> np.ndarray:
    if getattr(values.dtype, "names", None):
        values = values[values.dtype.names[0]]
    return np.asarray(values).reshape(-1)


def binary_labels_from_h5(
    data: dict[str, np.ndarray],
    label_field: str | None,
) -> tuple[np.ndarray | None, str | None]:
    candidates = (label_field,) if label_field else LABEL_CANDIDATES
    for candidate in candidates:
        if candidate and candidate in data:
            labels = flatten_labels(data[candidate]).astype(float)
            return np.where(labels > 0.5, 1, 0).astype(int), candidate
    return None, None


def jet_multiplicity(global_inputs: np.ndarray | None, jet_rows: np.ndarray) -> int:
    if global_inputs is not None and getattr(global_inputs.dtype, "names", None):
        names = global_inputs.dtype.names or ()
        lower_to_name = {name.lower(): name for name in names}
        name = lower_to_name.get("njet")
        if name is not None:
            value = finite_float(global_inputs[name])
            if math.isfinite(value):
                return int(value)
    return int(len(jet_rows))


def njet_bin(n_jets: int) -> str:
    if n_jets >= 9:
        return "9+"
    if n_jets in {4, 5, 6, 7, 8}:
        return str(n_jets)
    return "under4"


def label_set_for_nodes(nodes: list[int] | None, labels: np.ndarray, valid_njets: int) -> set[int]:
    if nodes is None:
        return set()

    out: set[int] = set()
    for node in nodes:
        node = int(node)
        if 0 <= node < valid_njets and node < len(labels):
            value = labels[node]
            if np.isfinite(value):
                out.add(int(value))
    return out


def label_list_for_nodes(nodes: list[int] | None, labels: np.ndarray, valid_njets: int) -> list[int]:
    if nodes is None:
        return []

    out: list[int] = []
    for node in nodes:
        node = int(node)
        if 0 <= node < valid_njets and node < len(labels):
            value = labels[node]
            if np.isfinite(value):
                out.append(int(value))
    return out


def jet_indices_for_nodes(nodes: list[int] | None, valid_njets: int) -> list[int]:
    if nodes is None:
        return []
    return [int(node) for node in nodes if 0 <= int(node) < valid_njets]


def truth_indices(jet_labels: np.ndarray) -> dict[str, list[int]]:
    labels = np.asarray(jet_labels).reshape(-1)
    return {
        role: [int(idx) for idx in np.where(labels == value)[0]]
        for role, value in TTBAR_SL_ROLE_MAP.items()
    }


def sum_p4(rows: np.ndarray, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.full(4, np.nan, dtype=float)

    total = np.zeros(4, dtype=float)
    for idx in indices:
        if idx >= len(rows):
            return np.full(4, np.nan, dtype=float)

        p4 = p4_from_row(rows[idx])
        if not np.all(np.isfinite(p4)):
            return np.full(4, np.nan, dtype=float)

        total += p4

    return total


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

    valid_njets = len(jets)
    n_leps = len(leptons)
    n_met = min(len(met), 1)
    njet = jet_multiplicity(global_input_event, jets)

    finite_truth_labels = {
        int(value)
        for value in labels
        if np.isfinite(value) and int(value) > 0
    }
    fully_matched = {1, 2, 3, 4}.issubset(finite_truth_labels)

    # If no class label exists, fall back to fully_matched as the signal-like
    # reconstruction mask. If labels exist, use them explicitly.
    if is_signal is None:
        is_signal_eval = int(fully_matched)
        signal_label_source = "fallback_fully_matched"
    else:
        is_signal_eval = int(is_signal == 1)
        signal_label_source = "event_label"

    reco_eval_event = int(is_signal_eval == 1 and fully_matched and njet >= 4)

    top1_nodes = as_node_list(row.get("HyPER_best_top1"), 3)
    top2_nodes = as_node_list(row.get("HyPER_best_top2"), 3)
    w1_nodes = as_node_list(row.get("HyPER_best_w1"), 2)
    w2_nodes = as_node_list(row.get("HyPER_best_w2"), 2)

    top1_labels = label_set_for_nodes(top1_nodes, labels, valid_njets)
    top2_labels = label_set_for_nodes(top2_nodes, labels, valid_njets)
    w1_labels = label_set_for_nodes(w1_nodes, labels, valid_njets)
    w2_labels = label_set_for_nodes(w2_nodes, labels, valid_njets)

    top1_label_list = label_list_for_nodes(top1_nodes, labels, valid_njets)
    top2_label_list = label_list_for_nodes(top2_nodes, labels, valid_njets)
    w1_label_list = label_list_for_nodes(w1_nodes, labels, valid_njets)
    w2_label_list = label_list_for_nodes(w2_nodes, labels, valid_njets)

    # Exact known-working SL efficiency logic.
    whad_correct = {3, 4}.issubset(w2_labels)
    b_had_correct = {2}.issubset(top2_labels) and not {2}.issubset(w2_labels)
    thad_correct = {2, 3, 4}.issubset(top2_labels) and whad_correct
    b_lep_correct = {1}.issubset(top1_labels)
    event_correct = b_lep_correct and thad_correct

    top1_jet_indices = jet_indices_for_nodes(top1_nodes, valid_njets)
    top2_jet_indices = jet_indices_for_nodes(top2_nodes, valid_njets)
    whad_jet_indices = jet_indices_for_nodes(w2_nodes, valid_njets)

    b_lep_idx = top1_jet_indices[0] if top1_jet_indices else None
    b_had_candidates = [idx for idx in top2_jet_indices if idx not in set(whad_jet_indices)]
    b_had_idx = b_had_candidates[0] if b_had_candidates else None

    whad_valid = len(whad_jet_indices) == 2
    b_had_valid = b_had_idx is not None
    thad_valid = whad_valid and b_had_valid
    b_lep_valid = b_lep_idx is not None
    reco_valid = thad_valid and b_lep_valid

    whad_p4 = sum_p4(jets, whad_jet_indices[:2]) if whad_valid else np.full(4, np.nan)
    thad_p4 = (
        p4_from_row(jets[b_had_idx]) + whad_p4
        if thad_valid
        else np.full(4, np.nan)
    )

    lep_idx = None
    met_idx = None
    if top1_nodes is not None:
        for node in top1_nodes:
            node = int(node)
            if valid_njets <= node < valid_njets + n_leps:
                lep_idx = node - valid_njets
            elif valid_njets + n_leps <= node < valid_njets + n_leps + n_met:
                met_idx = node - valid_njets - n_leps

    tlep_visible_valid = (
        b_lep_valid
        and lep_idx is not None
        and 0 <= lep_idx < n_leps
        and met_idx is not None
        and 0 <= met_idx < n_met
    )

    tlep_visible_p4 = (
        p4_from_row(jets[b_lep_idx])
        + p4_from_row(leptons[lep_idx])
        + met_proxy_p4(met[met_idx])
        if tlep_visible_valid
        else np.full(4, np.nan)
    )

    ttbar_visible_p4 = (
        thad_p4 + tlep_visible_p4
        if thad_valid and tlep_visible_valid
        else np.full(4, np.nan)
    )

    truth = truth_indices(labels)
    truth_whad_indices = truth["whad_j1"][:1] + truth["whad_j2"][:1]
    truth_thad_indices = truth["b_had"][:1] + truth_whad_indices

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

    truth_tlep_visible_valid = (
        len(truth["b_lep"]) >= 1
        and truth["b_lep"][0] < valid_njets
        and n_leps >= 1
        and n_met >= 1
    )

    truth_tlep_visible_p4 = (
        p4_from_row(jets[truth["b_lep"][0]])
        + p4_from_row(leptons[0])
        + met_proxy_p4(met[0])
        if truth_tlep_visible_valid
        else np.full(4, np.nan)
    )

    truth_ttbar_visible_p4 = (
        truth_thad_p4 + truth_tlep_visible_p4
        if np.all(np.isfinite(truth_thad_p4)) and np.all(np.isfinite(truth_tlep_visible_p4))
        else np.full(4, np.nan)
    )

    top_lep_score = finite_float(row.get("HyPER_best_top1_prob", np.nan))
    top_had_score = finite_float(row.get("HyPER_best_top2_prob", np.nan))
    w_lep_score = finite_float(row.get("HyPER_best_w1_prob", np.nan))
    w_had_score = finite_float(row.get("HyPER_best_w2_prob", np.nan))

    finite_component_scores = [
        score
        for score in (top_had_score, top_lep_score, w_had_score, w_lep_score)
        if math.isfinite(score)
    ]
    event_reco_score = (
        float(np.prod(finite_component_scores))
        if len(finite_component_scores) == 4
        else float("nan")
    )

    return {
        "event_index": int(event_idx),
        "n_jets": int(njet),
        "njet_bin": njet_bin(njet),
        "is_signal": int(is_signal_eval),
        "signal_label_source": signal_label_source,
        "fully_matched": int(fully_matched),
        "reco_eval_event": int(reco_eval_event),
        "reco_valid": int(reco_valid),
        "thad_valid": int(thad_valid),
        "whad_valid": int(whad_valid),
        "b_lep_valid": int(b_lep_valid),
        "tlep_visible_valid": int(tlep_visible_valid),
        "b_lep_correct": int(b_lep_correct),
        "b_had_correct": int(b_had_correct),
        "whad_correct": int(whad_correct),
        "thad_correct": int(thad_correct),
        "event_correct": int(event_correct),
        "b_lep_idx": -1 if b_lep_idx is None else int(b_lep_idx),
        "b_had_idx": -1 if b_had_idx is None else int(b_had_idx),
        "whad_j1_idx": int(whad_jet_indices[0]) if len(whad_jet_indices) > 0 else -1,
        "whad_j2_idx": int(whad_jet_indices[1]) if len(whad_jet_indices) > 1 else -1,
        "top1_labels": sorted(top1_labels),
        "top2_labels": sorted(top2_labels),
        "w1_labels": sorted(w1_labels),
        "w2_labels": sorted(w2_labels),
        "top1_label_tuple": str(tuple(top1_label_list)),
        "top2_label_tuple": str(tuple(top2_label_list)),
        "w1_label_tuple": str(tuple(w1_label_list)),
        "w2_label_tuple": str(tuple(w2_label_list)),
        "m_Whad": invariant_mass(whad_p4),
        "m_thad": invariant_mass(thad_p4),
        "m_tlep_visible": invariant_mass(tlep_visible_p4),
        "m_ttbar_visible": invariant_mass(ttbar_visible_p4),
        "m_Whad_truth": invariant_mass(truth_whad_p4),
        "m_thad_truth": invariant_mass(truth_thad_p4),
        "m_tlep_visible_truth": invariant_mass(truth_tlep_visible_p4),
        "m_ttbar_visible_truth": invariant_mass(truth_ttbar_visible_p4),
        "top_had_score": top_had_score,
        "top_lep_score": top_lep_score,
        "w_had_score": w_had_score,
        "w_lep_score": w_lep_score,
        "event_reco_score": event_reco_score,
        "thad_first": finite_float(row.get("thad_first", np.nan)),
        "HyPER_CLS_RAW": finite_float(row.get("HyPER_CLS_RAW", np.nan)),
    }


def evaluate_events(
    data: dict[str, np.ndarray],
    reco: pd.DataFrame,
    n_processed: int,
    signal_labels: np.ndarray | None,
) -> pd.DataFrame:
    global_inputs = data.get("global_inputs")
    rows = []

    for event_idx in range(n_processed):
        global_row = (
            global_inputs[event_idx, 0]
            if global_inputs is not None and global_inputs.ndim > 1
            else None
        )
        is_signal = None if signal_labels is None else int(signal_labels[event_idx])

        rows.append(
            evaluate_event(
                event_idx=event_idx,
                row=reco.iloc[event_idx],
                jets_event=data["jets"][event_idx],
                leptons_event=data["leptons"][event_idx],
                met_event=data["met"][event_idx],
                jet_labels_event=data["jet_labels"][event_idx],
                global_input_event=global_row,
                is_signal=is_signal,
            )
        )

    return pd.DataFrame(rows)


def efficiency_rows(evaluation: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    base_mask = evaluation["reco_eval_event"].to_numpy(dtype=bool)

    for bin_name in NJET_BINS:
        if bin_name == "overall":
            mask = base_mask
        else:
            mask = base_mask & (evaluation["njet_bin"].to_numpy() == bin_name)

        subset = evaluation.loc[mask]
        n_events = int(len(subset))
        row: dict[str, Any] = {"njet_bin": bin_name, "n_events": n_events}

        for name, column in (
            ("event_eff", "event_correct"),
            ("b_lep_eff", "b_lep_correct"),
            ("b_had_eff", "b_had_correct"),
            ("whad_eff", "whad_correct"),
            ("thad_eff", "thad_correct"),
            ("reco_valid_fraction", "reco_valid"),
        ):
            row[name] = float(subset[column].mean()) if n_events else float("nan")

        rows.append(row)

    return rows


def pattern_counts(evaluation: pd.DataFrame) -> dict[str, dict[str, int]]:
    base = evaluation.loc[evaluation["reco_eval_event"] == 1].copy()

    out: dict[str, dict[str, int]] = {}
    for column in ("top1_label_tuple", "top2_label_tuple", "w1_label_tuple", "w2_label_tuple"):
        out[column] = {
            str(key): int(value)
            for key, value in Counter(base[column].astype(str)).most_common(30)
        }

    out["w2_wrong_only"] = {
        str(key): int(value)
        for key, value in Counter(
            base.loc[base["whad_correct"] == 0, "w2_label_tuple"].astype(str)
        ).most_common(30)
    }

    out["top2_wrong_only"] = {
        str(key): int(value)
        for key, value in Counter(
            base.loc[base["thad_correct"] == 0, "top2_label_tuple"].astype(str)
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
    bars = ax.bar(x, values, yerr=errors, capsize=4, edgecolor="black", alpha=0.9)

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
    ax.set_title("ttbar SL event reconstruction efficiency")
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
    ]

    x = np.arange(len(rows))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (key, label) in enumerate(components):
        offset = (i - (len(components) - 1) / 2.0) * width
        values = np.asarray([100.0 * row[key] for row in rows], dtype=float)
        ax.bar(x + offset, values, width=width, label=label, edgecolor="black", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['njet_bin']} jets" for row in rows])
    ax.set_ylabel("Efficiency [%]")
    ax.set_ylim(0.0, 110.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=3)
    ax.set_title("ttbar SL reconstruction component efficiencies")
    fig.tight_layout()
    save_fig(output_dir, "efficiency_components_by_njets", formats)


def plot_efficiency_per_jet_detailed(
    eff_rows: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
) -> None:
    rows = [row for row in eff_rows if row["njet_bin"] != "overall"]

    for row in rows:
        categories = ["overall", r"$b_\mathrm{had}$", r"$b_\mathrm{lep}$", r"$W_\mathrm{had}$"]
        values = np.asarray(
            [
                row["event_eff"],
                row["b_had_eff"],
                row["b_lep_eff"],
                row["whad_eff"],
            ],
            dtype=float,
        ) * 100.0

        n_events = int(row["n_events"])
        errors = np.asarray(
            [100.0 * binomial_error_fraction(value / 100.0, n_events) for value in values],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        x = np.arange(len(categories))
        bars = ax.bar(x, values, yerr=errors, capsize=4, edgecolor="black", alpha=0.9)

        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 2.0,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylabel("Reconstruction efficiency [%]")
        ax.set_ylim(0.0, 110.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        ax.set_title(f"ttbar SL, {row['njet_bin']} jets (n={n_events})")
        fig.tight_layout()

        safe_bin = str(row["njet_bin"]).replace("+", "plus")
        save_fig(output_dir, f"efficiency_components_{safe_bin}_jets", formats)


def plot_efficiencies(eff_rows: list[dict[str, Any]], output_dir: Path, formats: list[str]) -> None:
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
    signal = evaluation["is_signal"].to_numpy(dtype=int) == 1
    background = evaluation["is_signal"].to_numpy(dtype=int) == 0
    fully_matched = evaluation["fully_matched"].to_numpy(dtype=int) == 1
    fully_matched_signal = evaluation["reco_eval_event"].to_numpy(dtype=bool)
    counts = {
        "plot_scope": plot_scope,
        "total_events": int(len(evaluation)),
        "signal_events": int(np.sum(signal)),
        "background_events": int(np.sum(background)),
        "fully_matched_signal_events": int(np.sum(fully_matched_signal)),
        "unmatched_signal_events": int(np.sum(signal & ~fully_matched)),
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
    plot_scope: str = "fully_matched",
) -> None:
    if plot_scope == "fully_matched":
        evaluation = evaluation.loc[evaluation["reco_eval_event"] == 1].copy()
    else:
        evaluation = evaluation.copy()

    for name in ("m_Whad", "m_thad", "m_tlep_visible", "m_ttbar_visible"):
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

    for name in ("event_reco_score", "top_had_score", "top_lep_score", "w_had_score", "w_lep_score"):
        output_name = name if name != "event_reco_score" else "reco_score_distribution"
        values = evaluation[name].to_numpy(dtype=float)
        if plot_scope == "fully_matched":
            plot_hist(values, name, name, output_dir, output_name, formats)
        else:
            plot_hist_by_scope(evaluation, values, name, name, output_dir, output_name, formats, plot_scope)


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


def make_sb_plots(
    evaluation: pd.DataFrame,
    labels: np.ndarray | None,
    label_source: str | None,
    score_field: str,
    no_sb: bool,
    output_dir: Path,
    formats: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    if no_sb:
        return None, "S/B plots skipped by --no-sb."

    if score_field not in evaluation.columns or not np.any(np.isfinite(evaluation[score_field].to_numpy(dtype=float))):
        return None, f"S/B plots skipped because score field {score_field!r} is absent or all NaN."

    if labels is None:
        return None, "S/B plots skipped because no binary event labels were found."

    labels = labels[: len(evaluation)]
    scores = evaluation[score_field].to_numpy(dtype=float)

    unique_labels = sorted(set(labels[np.isfinite(scores)].astype(int).tolist()))
    if unique_labels != [0, 1]:
        return None, f"S/B plots skipped because labels do not contain both classes: {unique_labels}."

    plt.figure(figsize=(6.0, 4.2))
    bins = np.linspace(0.0, 1.0, 41)
    for label, title in ((0, "Background"), (1, "Signal")):
        values = scores[(labels == label) & np.isfinite(scores)]
        if len(values):
            plt.hist(values, bins=bins, histtype="step", density=True, label=title)
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

    summary = {
        "score_field": score_field,
        "label_source": label_source,
        "n_events": int(np.sum(np.isfinite(scores))),
        "n_signal": int(np.sum(labels == 1)),
        "n_background": int(np.sum(labels == 0)),
        "auc": auc,
        "mean_score_signal": float(np.nanmean(scores[labels == 1])),
        "mean_score_background": float(np.nanmean(scores[labels == 0])),
    }

    with (output_dir / "sb_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({key: clean_json_value(value) for key, value in summary.items()}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary, None


def write_outputs(
    output_dir: Path,
    evaluation: pd.DataFrame,
    eff_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation.to_csv(output_dir / "observables.csv", index=False)
    write_csv(output_dir / "efficiency_by_njets.csv", eff_rows)

    overall = next(row for row in eff_rows if row["njet_bin"] == "overall")
    summary_csv_row = {
        "processed_events": summary["processed_events"],
        "signal_events": summary["signal_events"],
        "background_events": summary["background_events"],
        "fully_matched_events": summary["fully_matched_events"],
        "reco_eval_events": summary["reco_eval_events"],
        "reco_eval_fraction": summary["reco_eval_fraction"],
        "event_eff": overall["event_eff"],
        "b_lep_eff": overall["b_lep_eff"],
        "b_had_eff": overall["b_had_eff"],
        "whad_eff": overall["whad_eff"],
        "thad_eff": overall["thad_eff"],
        "reco_valid_fraction": overall["reco_valid_fraction"],
        "sb_status": summary.get("sb_status"),
    }
    write_csv(output_dir / "summary.csv", [summary_csv_row])

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=clean_json_value)
        handle.write("\n")


def read_h5_slice(path: str | Path, start: int, stop: int) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        for required in ("INPUTS/JET", "INPUTS/LEPTON", "INPUTS/MET", "LABELS/JET"):
            if required not in handle:
                raise KeyError(f"Missing required dataset {required}")
        data = {
            "jets": handle["INPUTS/JET"][start:stop],
            "leptons": handle["INPUTS/LEPTON"][start:stop],
            "met": handle["INPUTS/MET"][start:stop],
            "jet_labels": handle["LABELS/JET"][start:stop],
        }
        if "INPUTS/GLOBAL" in handle:
            data["global_inputs"] = handle["INPUTS/GLOBAL"][start:stop]
        for candidate in LABEL_CANDIDATES:
            if candidate in handle:
                data[candidate] = handle[candidate][start:stop]
    return data


def update_efficiency_counters(counters: dict[str, Counter], rows: pd.DataFrame) -> None:
    base = rows.loc[rows["reco_eval_event"] == 1]
    for bin_name in NJET_BINS:
        subset = base if bin_name == "overall" else base.loc[base["njet_bin"] == bin_name]
        c = counters.setdefault(bin_name, Counter())
        c["n_events"] += int(len(subset))
        for key, col in (
            ("event_eff", "event_correct"),
            ("b_lep_eff", "b_lep_correct"),
            ("b_had_eff", "b_had_correct"),
            ("whad_eff", "whad_correct"),
            ("thad_eff", "thad_correct"),
            ("reco_valid_fraction", "reco_valid"),
        ):
            c[key] += int(np.sum(subset[col].to_numpy(dtype=int))) if len(subset) else 0


def counters_to_eff_rows(counters: dict[str, Counter]) -> list[dict[str, Any]]:
    rows = []
    for bin_name in NJET_BINS:
        c = counters.get(bin_name, Counter())
        n = int(c.get("n_events", 0))
        row: dict[str, Any] = {"njet_bin": bin_name, "n_events": n}
        for key in ("event_eff", "b_lep_eff", "b_had_eff", "whad_eff", "thad_eff", "reco_valid_fraction"):
            row[key] = float(c.get(key, 0) / n) if n else float("nan")
        rows.append(row)
    return rows


def update_pattern_counters(patterns: dict[str, Counter], rows: pd.DataFrame) -> None:
    base = rows.loc[rows["reco_eval_event"] == 1]
    for column in ("top1_label_tuple", "top2_label_tuple", "w1_label_tuple", "w2_label_tuple"):
        patterns[column].update(base[column].astype(str).tolist())
    patterns["w2_wrong_only"].update(
        base.loc[base["whad_correct"] == 0, "w2_label_tuple"].astype(str).tolist()
    )
    patterns["top2_wrong_only"].update(
        base.loc[base["thad_correct"] == 0, "top2_label_tuple"].astype(str).tolist()
    )


def pattern_counters_to_summary(patterns: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {
        key: {str(k): int(v) for k, v in counter.most_common(30)}
        for key, counter in patterns.items()
    }


def append_csv(path: Path, rows: pd.DataFrame, write_header: bool) -> None:
    rows.to_csv(path, mode="a", index=False, header=write_header)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as handle:
        n_h5_events = len(handle["INPUTS/JET"])

    warnings_list: list[str] = []
    eff_counters: dict[str, Counter] = {}
    patterns: dict[str, Counter] = {
        "top1_label_tuple": Counter(),
        "top2_label_tuple": Counter(),
        "w1_label_tuple": Counter(),
        "w2_label_tuple": Counter(),
        "w2_wrong_only": Counter(),
        "top2_wrong_only": Counter(),
    }
    observable_counts: Counter = Counter()
    sample_frames: list[pd.DataFrame] = []
    sample_rows = 0
    signal_events = 0
    background_events = 0
    fully_matched_events = 0
    reco_eval_events = 0
    n_processed = 0
    n_prediction_loaded = 0
    n_prediction_events: int | None = None
    label_source: str | None = None
    alignment_mode = "streaming_row_order_prefix_assumed"
    event_csv_written = False
    event_csv_path = output_dir / "observables.csv"
    write_event_csv = bool(args.write_event_csv and not args.skip_event_csv)

    for prediction_chunk in iter_hyper_prediction_parts(
        args.prediction_output,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
    ):
        if len(prediction_chunk) == 0:
            continue
        if n_prediction_events is None:
            n_prediction_events = int(prediction_chunk.attrs.get("hyper_total_rows", len(prediction_chunk)))
            if args.strict_length and n_h5_events != n_prediction_events:
                raise ValueError(
                    f"Length mismatch: H5 has {n_h5_events} events, "
                    f"prediction has {n_prediction_events} rows."
                )

        chunk_len = min(len(prediction_chunk), n_h5_events - n_processed)
        if chunk_len <= 0:
            break
        prediction_chunk = prediction_chunk.iloc[:chunk_len].reset_index(drop=True)
        start = n_processed
        stop = start + chunk_len

        if "event_index" in prediction_chunk.columns:
            event_index = pd.to_numeric(prediction_chunk["event_index"], errors="coerce").to_numpy(dtype=float)
            expected = np.arange(start, stop, dtype=np.int64)
            if len(event_index) == chunk_len and np.all(np.isfinite(event_index)):
                event_index_int = event_index.astype(np.int64)
                if np.array_equal(event_index_int, expected):
                    alignment_mode = "event_index_streaming_prefix_verified"
                elif np.array_equal(np.sort(event_index_int), expected):
                    prediction_chunk = prediction_chunk.iloc[np.argsort(event_index_int)].reset_index(drop=True)
                    alignment_mode = "event_index_streaming_prefix_reordered"
                else:
                    warning = "Prediction event_index is not the expected streaming H5 prefix; assuming row-order alignment."
                    if args.strict_length:
                        raise ValueError(warning)
                    if warning not in warnings_list:
                        warnings_list.append(warning)
                        LOGGER.warning(warning)
            else:
                warning = "Prediction event_index is invalid in a streaming chunk; assuming row-order alignment."
                if args.strict_length:
                    raise ValueError(warning)
                if warning not in warnings_list:
                    warnings_list.append(warning)
                    LOGGER.warning(warning)

        reco = normalise_predictions(prediction_chunk, classification=args.classification).reset_index(drop=True)
        data = read_h5_slice(args.h5, start, stop)
        labels, chunk_label_source = binary_labels_from_h5(data, args.label_field)
        if label_source is None and chunk_label_source is not None:
            label_source = chunk_label_source
        signal_labels = labels[:chunk_len] if labels is not None else None
        if signal_labels is None and not any("No event-level binary label found" in w for w in warnings_list):
            warning = (
                "No event-level binary label found. Reconstruction efficiency will fall back to "
                "fully_matched == 1 as the signal mask."
            )
            LOGGER.warning(warning)
            warnings_list.append(warning)

        global_inputs = data.get("global_inputs")
        chunk_rows = []
        for local_idx in range(chunk_len):
            global_row = (
                global_inputs[local_idx, 0]
                if global_inputs is not None and global_inputs.ndim > 1
                else None
            )
            is_signal = None if signal_labels is None else int(signal_labels[local_idx])
            chunk_rows.append(
                evaluate_event(
                    event_idx=start + local_idx,
                    row=reco.iloc[local_idx],
                    jets_event=data["jets"][local_idx],
                    leptons_event=data["leptons"][local_idx],
                    met_event=data["met"][local_idx],
                    jet_labels_event=data["jet_labels"][local_idx],
                    global_input_event=global_row,
                    is_signal=is_signal,
                )
            )
        evaluation_chunk = pd.DataFrame(chunk_rows)

        update_efficiency_counters(eff_counters, evaluation_chunk)
        update_pattern_counters(patterns, evaluation_chunk)
        signal_events += int(np.sum(evaluation_chunk["is_signal"] == 1))
        background_events += int(np.sum(evaluation_chunk["is_signal"] == 0))
        fully_matched_events += int(np.sum(evaluation_chunk["fully_matched"] == 1))
        reco_eval_events += int(np.sum(evaluation_chunk["reco_eval_event"] == 1))
        reco_mask = evaluation_chunk["reco_eval_event"] == 1
        for name in (
            "m_Whad",
            "m_thad",
            "m_tlep_visible",
            "m_ttbar_visible",
            "m_Whad_truth",
            "m_thad_truth",
            "m_tlep_visible_truth",
            "m_ttbar_visible_truth",
        ):
            observable_counts[name] += int(np.sum(np.isfinite(evaluation_chunk.loc[reco_mask, name].to_numpy(dtype=float))))

        if write_event_csv:
            append_csv(event_csv_path, evaluation_chunk, write_header=not event_csv_written)
            event_csv_written = True
        if sample_rows < int(args.max_plot_points):
            keep = min(int(args.max_plot_points) - sample_rows, len(evaluation_chunk))
            sample_frames.append(evaluation_chunk.iloc[:keep].copy())
            sample_rows += keep

        n_processed += chunk_len
        n_prediction_loaded += chunk_len
        LOGGER.info("Processed plotting events %d:%d", start, stop)
        if args.max_events is not None and n_processed >= int(args.max_events):
            break

    if n_processed <= 0:
        raise ValueError("No events to process.")
    if n_prediction_events is None:
        n_prediction_events = n_prediction_loaded
    if n_h5_events != n_prediction_events:
        warning = (
            f"Length mismatch: H5 has {n_h5_events} events, prediction has "
            f"{n_prediction_events} rows; processed {n_processed} events."
        )
        LOGGER.warning(warning)
        warnings_list.append(warning)

    evaluation_sample = (
        pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    )
    eff_rows = counters_to_eff_rows(eff_counters)

    plot_counts = print_plot_scope_counts(evaluation_sample, args.plot_scope) if len(evaluation_sample) else {
        "plot_scope": args.plot_scope,
        "total_events": 0,
        "signal_events": 0,
        "background_events": 0,
        "fully_matched_signal_events": 0,
        "unmatched_signal_events": 0,
    }
    plot_efficiencies(eff_rows, output_dir, args.formats)
    if len(evaluation_sample):
        plot_observables(evaluation_sample, output_dir, args.formats, args.plot_scope)

    sb_summary, sb_skip_reason = make_sb_plots(
        evaluation=evaluation_sample,
        labels=(
            evaluation_sample["is_signal"].to_numpy(dtype=int)
            if len(evaluation_sample) and "is_signal" in evaluation_sample.columns
            else None
        ),
        label_source=label_source,
        score_field=args.score_field,
        no_sb=args.no_sb,
        output_dir=output_dir,
        formats=args.formats,
    )
    if sb_skip_reason:
        LOGGER.warning(sb_skip_reason)
        warnings_list.append(sb_skip_reason)

    summary = {
        "h5": str(args.h5),
        "prediction_output": str(args.prediction_output),
        "output_dir": str(args.output_dir),
        "topology": "ttbar_singlelep",
        "h5_events": int(n_h5_events),
        "prediction_events": int(n_prediction_events),
        "prediction_rows_loaded": int(n_prediction_loaded),
        "processed_events": int(n_processed),
        "max_events": args.max_events,
        "prediction_truncated_by_loader": bool(args.max_events is not None and n_processed >= int(args.max_events)),
        "strict_length": bool(args.strict_length),
        "chunk_size": int(args.chunk_size),
        "event_csv": "written" if event_csv_written else "skipped",
        "max_plot_points": int(args.max_plot_points),
        "plot_scope": args.plot_scope,
        "plot_category_counts": plot_counts,
        "plot_sample_events": int(len(evaluation_sample)),
        "event_alignment": alignment_mode,
        "event_alignment_policy": (
            "event_index is verified/reordered when it describes the H5 prefix; "
            "otherwise row-order prefix alignment is assumed unless --strict-length raises."
        ),
        "label_source": label_source,
        "signal_events": signal_events,
        "background_events": background_events,
        "fully_matched_events": fully_matched_events,
        "fully_matched_fraction": float(fully_matched_events / n_processed) if n_processed else None,
        "reco_eval_events": reco_eval_events,
        "reco_eval_fraction": float(reco_eval_events / n_processed) if n_processed else None,
        "evaluation_policy": {
            "sb_plots": "all events with binary labels and finite classification score",
            "reconstruction_efficiency": (
                "event-level signal label == 1, fully_matched == 1, and n_jets >= 4; "
                "if no event label is available, signal label falls back to fully_matched"
            ),
            "observable_plots": (
                "same as reconstruction_efficiency, with finite pred/truth observable values where relevant"
            ),
            "background_events": (
                "can be streamed to observables.csv with --write-event-csv but are excluded from "
                "reconstruction efficiencies and observable plots"
            ),
        },
        "semantic_contract": {
            "HyPER_best_top1": "leptonic top after ttbar_single_lep()",
            "HyPER_best_top2": "hadronic top after ttbar_single_lep()",
            "HyPER_best_w1": "leptonic W edge after ttbar_single_lep()",
            "HyPER_best_w2": "hadronic W edge after ttbar_single_lep()",
        },
        "efficiency_logic": {
            "Whad_is_correct": "{3, 4}.issubset(w2)",
            "bhad_is_correct": "{2}.issubset(top2) and not {2}.issubset(w2)",
            "thad_is_correct": "{2, 3, 4}.issubset(top2) and Whad_is_correct",
            "blep_is_correct": "{1}.issubset(top1)",
            "event_is_correct": "blep_is_correct and thad_is_correct",
        },
        "diagnostics": {
            "label_pattern_counts": pattern_counters_to_summary(patterns),
            "interpretation": {
                "w2_label_tuple == (3, 4)": "correct hadronic W",
                "w2_label_tuple == (2, 3) or (2, 4)": "edge picked b_had plus one W jet",
                "w2_label_tuple == ()": "selected W edge contains no valid jet labels",
            },
        },
        "role_map": TTBAR_SL_ROLE_MAP,
        "jet_bins": list(NJET_BINS),
        "event_reco_score_definition": (
            "product(top_had_score, top_lep_score, w_had_score, w_lep_score) "
            "when all four are finite"
        ),
        "truth_visible_lepton_met_note": (
            "Visible leptonic masses use truth b_lep plus first valid lepton and MET proxy; "
            "no neutrino is inferred."
        ),
        "efficiency_by_njets": [
            {key: clean_json_value(value) for key, value in row.items()}
            for row in eff_rows
        ],
        "observable_available_counts": {key: int(value) for key, value in observable_counts.items()},
        "sb_summary": (
            {key: clean_json_value(value) for key, value in sb_summary.items()}
            if sb_summary
            else None
        ),
        "sb_status": "written" if sb_summary else sb_skip_reason,
        "warnings": warnings_list,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if not event_csv_written and not args.skip_event_csv and n_processed <= 1_000_000 and len(evaluation_sample) == n_processed:
        evaluation_sample.to_csv(output_dir / "observables.csv", index=False)
        event_csv_written = True
        summary["event_csv"] = "written_sample_complete"
    write_csv(output_dir / "efficiency_by_njets.csv", eff_rows)
    overall = next(row for row in eff_rows if row["njet_bin"] == "overall")
    write_csv(
        output_dir / "summary.csv",
        [
            {
                "processed_events": summary["processed_events"],
                "signal_events": signal_events,
                "background_events": background_events,
                "fully_matched_events": fully_matched_events,
                "reco_eval_events": reco_eval_events,
                "reco_eval_fraction": summary["reco_eval_fraction"],
                "event_eff": overall["event_eff"],
                "b_lep_eff": overall["b_lep_eff"],
                "b_had_eff": overall["b_had_eff"],
                "whad_eff": overall["whad_eff"],
                "thad_eff": overall["thad_eff"],
                "reco_valid_fraction": overall["reco_valid_fraction"],
                "sb_status": summary.get("sb_status"),
            }
        ],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=clean_json_value)
        handle.write("\n")
    LOGGER.info("Wrote ttbar SL reconstruction evaluation to %s", output_dir)


if __name__ == "__main__":
    main()
