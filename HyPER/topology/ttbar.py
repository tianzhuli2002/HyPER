from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.rich import tqdm

try:
    from .prediction_io import load_hyper_prediction_output
except ImportError:  # pragma: no cover - allows direct script-style imports.
    from prediction_io import load_hyper_prediction_output


def _load_results(HyPER_outputs: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(HyPER_outputs, pd.DataFrame):
        return HyPER_outputs.copy()

    if isinstance(HyPER_outputs, str):
        return load_hyper_prediction_output(HyPER_outputs)

    raise ValueError(
        f"Unrecognised HyPER output type {type(HyPER_outputs)}. "
        "Expected `str` or `pandas.DataFrame`."
    )


def _load_config(config: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}

    if isinstance(config, dict):
        return config

    config_path = Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        loaded = yaml.safe_load(f)

    return loaded or {}


def _classification_enabled(
    config: str | Path | dict[str, Any] | None = None,
    classification: bool | None = None,
) -> bool:
    """
    Resolve whether the classification head is enabled.

    Priority:
      1. Explicit `classification` argument if provided.
      2. `classification.enabled` from config.
      3. Default True for backwards compatibility.
    """
    if classification is not None:
        return bool(classification)

    cfg = _load_config(config)

    return bool(
        cfg.get("classification", {}).get("enabled", True)
    )


def _add_cls_column_if_enabled(
    columns_to_return: list[str],
    results: pd.DataFrame,
    classification_enabled: bool,
) -> list[str]:
    if not classification_enabled:
        return columns_to_return

    if "HyPER_CLS_RAW" not in results.columns:
        raise KeyError(
            "classification.enabled is true, but 'HyPER_CLS_RAW' is missing from "
            "the HyPER outputs. Either run prediction with the classification head "
            "enabled or set classification.enabled: false in the config."
        )

    return columns_to_return + ["HyPER_CLS_RAW"]


def _get_best_edge_in_pattern(
    pattern: list[int] | np.ndarray,
    GE_IDX,
    GE_RAW,
    GE_VCT=None,
    require_leptonic_edge: bool = False,
) -> tuple[list[int], float]:
    """
    Find the best graph edge inside a selected hyperedge pattern.

    If require_leptonic_edge=True, require the edge VCT to contain both
    lepton and MET labels, i.e. {2, 3}.
    """
    best_edge = [-1, -1]
    best_score = 0.0

    ge_idx_array = np.array(GE_IDX)

    for possible_edge in combinations(pattern, r=2):
        edge_match = np.argwhere(
            np.sum(
                np.where(ge_idx_array == possible_edge[0], 1, 0)
                + np.where(ge_idx_array == possible_edge[1], 1, 0),
                axis=1,
            )
            == 2
        ).flatten()

        if len(edge_match) == 0:
            continue

        edge_i = edge_match[0]

        if require_leptonic_edge:
            if GE_VCT is None:
                continue
            if len(set(GE_VCT[edge_i]).intersection({2, 3})) != 2:
                continue

        if GE_RAW[edge_i] > best_score:
            best_score = float(GE_RAW[edge_i])
            best_edge = list(possible_edge)

    return best_edge, best_score


def ttbar_dilep(
    HyPER_outputs: str | pd.DataFrame,
    config: str | Path | dict[str, Any] | None = None,
    classification: bool | None = None,
) -> pd.DataFrame:
    r"""Reconstruct ttbar events with dileptonic final states."""

    results = _load_results(HyPER_outputs)

    use_classification = _classification_enabled(
        config=config,
        classification=classification,
    )

    HyPER_best_top1 = []
    HyPER_best_top2 = []

    HyPER_best_top1_score = []
    HyPER_best_top2_score = []
    HyPER_best_event_score = []

    for i in tqdm(range(len(results)), desc="Reconstructing", unit="event"):
        HE_IDX = results["HyPER_HE_IDX"][i]
        HE_RAW = results["HyPER_HE_RAW"][i]
        GE_IDX = results["HyPER_GE_IDX"][i]
        GE_RAW = results["HyPER_GE_RAW"][i]

        selected_GE = []
        softProb_GE = []

        completed_patterns = 0
        rank = np.argsort(GE_RAW)
        p = -1
        skip_event = False

        while completed_patterns < 2:
            if completed_patterns > 0:
                for pattern in selected_GE:
                    while True:
                        if abs(p) > len(rank):
                            skip_event = True
                            break

                        if len(set(pattern).intersection(GE_IDX[rank[p]])) == 0:
                            break

                        p -= 1

            if skip_event:
                break

            selected_GE.append(GE_IDX[rank[p]])
            softProb_GE.append(float(GE_RAW[rank[p]]))

            p -= 1
            completed_patterns += 1

        if skip_event or len(selected_GE) < 2:
            HyPER_best_top1.append(None)
            HyPER_best_top2.append(None)
            HyPER_best_top1_score.append(0.0)
            HyPER_best_top2_score.append(0.0)
            HyPER_best_event_score.append(0.0)
            continue

        hyperedge_to_find = set(selected_GE[0]) | set(selected_GE[1])

        matched_HE_index = None
        for j, he in enumerate(HE_IDX):
            if hyperedge_to_find.issubset(set(he)):
                matched_HE_index = j
                break

        HyPER_best_top1.append(selected_GE[0])
        HyPER_best_top2.append(selected_GE[1])

        HyPER_best_top1_score.append(softProb_GE[0])
        HyPER_best_top2_score.append(softProb_GE[1])

        if matched_HE_index is None or matched_HE_index >= len(HE_RAW):
            HyPER_best_event_score.append(0.0)
        else:
            HyPER_best_event_score.append(float(HE_RAW[matched_HE_index]))

    results["HyPER_best_top1"] = HyPER_best_top1
    results["HyPER_best_top2"] = HyPER_best_top2
    results["HyPER_best_top1_score"] = HyPER_best_top1_score
    results["HyPER_best_top2_score"] = HyPER_best_top2_score
    results["HyPER_best_event_score"] = HyPER_best_event_score

    columns_to_return = [
        "HyPER_best_top1",
        "HyPER_best_top2",
        "HyPER_best_top1_score",
        "HyPER_best_top2_score",
        "HyPER_best_event_score",
    ]

    columns_to_return = _add_cls_column_if_enabled(
        columns_to_return,
        results,
        use_classification,
    )

    return results[columns_to_return]


def ttbar_single_lep(
    HyPER_outputs: str | pd.DataFrame,
    config: str | Path | dict[str, Any] | None = None,
    classification: bool | None = None,
) -> pd.DataFrame:
    r"""Reconstruct ttbar events with lepton+jets final states."""

    results = _load_results(HyPER_outputs)

    use_classification = _classification_enabled(
        config=config,
        classification=classification,
    )

    HyPER_best_top1 = []
    HyPER_best_top2 = []
    HyPER_best_w1 = []
    HyPER_best_w2 = []

    HyPER_best_top1_prob = []
    HyPER_best_top2_prob = []
    HyPER_best_w1_prob = []
    HyPER_best_w2_prob = []

    thad_first = []

    for i in tqdm(range(len(results)), desc="Reconstructing", unit="event"):
        HE_IDX = results["HyPER_HE_IDX"][i]
        HE_RAW = results["HyPER_HE_RAW"][i]
        GE_IDX = results["HyPER_GE_IDX"][i]
        GE_RAW = results["HyPER_GE_RAW"][i]

        HE_VCT = results["HyPER_HE_VCT"][i]
        GE_VCT = results["HyPER_GE_VCT"][i]

        selected_HE = []
        softProb_HE = []
        selected_GE = []
        softProb_GE = []

        completed_patterns = 0
        tlep_position = 0
        rank = np.argsort(HE_RAW)
        p = -1
        skip_event = False

        while completed_patterns < 2:
            if abs(p) > len(rank):
                skip_event = True
                break

            if completed_patterns == 0:
                # Avoid selecting an invalid top containing only one of lepton/MET.
                while len(set(HE_VCT[rank[p]]).intersection({2, 3})) == 1:
                    p -= 1
                    if abs(p) > len(rank):
                        skip_event = True
                        break

                if skip_event:
                    break

                # If the best valid top is not leptonic, then hadronic was found first.
                if len(set(HE_VCT[rank[p]]).intersection({2, 3})) != 2:
                    tlep_position = 1

            if completed_patterns > 0:
                for pattern in selected_HE:
                    while (
                        len(set(pattern).intersection(set(HE_IDX[rank[p]]))) != 0
                        or (
                            tlep_position == 1
                            and len(set(HE_VCT[rank[p]]).intersection({2, 3})) != 2
                        )
                    ):
                        p -= 1
                        if abs(p) > len(rank):
                            skip_event = True
                            break

                    if skip_event:
                        break

            if skip_event:
                break

            if HE_RAW[rank[p]] != 0:
                selected_HE.append(HE_IDX[rank[p]])
                softProb_HE.append(float(HE_RAW[rank[p]]))
            else:
                selected_HE.append([-1, -1, -1])
                softProb_HE.append(0.0)

            require_leptonic_edge = completed_patterns == tlep_position

            best_edge, best_edge_score = _get_best_edge_in_pattern(
                pattern=HE_IDX[rank[p]],
                GE_IDX=GE_IDX,
                GE_RAW=GE_RAW,
                GE_VCT=GE_VCT,
                require_leptonic_edge=require_leptonic_edge,
            )

            selected_GE.append(best_edge)
            softProb_GE.append(best_edge_score)

            p -= 1
            completed_patterns += 1

        if skip_event or len(selected_HE) < 2 or len(selected_GE) < 2:
            thad_first.append(-1)

            HyPER_best_top1.append([-1, -1, -1])
            HyPER_best_top2.append([-1, -1, -1])
            HyPER_best_w1.append([-1, -1])
            HyPER_best_w2.append([-1, -1])

            HyPER_best_top1_prob.append(0.0)
            HyPER_best_top2_prob.append(0.0)
            HyPER_best_w1_prob.append(0.0)
            HyPER_best_w2_prob.append(0.0)
            continue

        if tlep_position == 0:
            HyPER_best_top1.append(selected_HE[0])
            HyPER_best_top2.append(selected_HE[1])
            HyPER_best_w1.append(selected_GE[0])
            HyPER_best_w2.append(selected_GE[1])

            HyPER_best_top1_prob.append(softProb_HE[0])
            HyPER_best_top2_prob.append(softProb_HE[1])
            HyPER_best_w1_prob.append(softProb_GE[0])
            HyPER_best_w2_prob.append(softProb_GE[1])
        else:
            HyPER_best_top1.append(selected_HE[1])
            HyPER_best_top2.append(selected_HE[0])
            HyPER_best_w1.append(selected_GE[1])
            HyPER_best_w2.append(selected_GE[0])

            HyPER_best_top1_prob.append(softProb_HE[1])
            HyPER_best_top2_prob.append(softProb_HE[0])
            HyPER_best_w1_prob.append(softProb_GE[1])
            HyPER_best_w2_prob.append(softProb_GE[0])

        thad_first.append(tlep_position)

    results["thad_first"] = thad_first

    results["HyPER_best_top1"] = HyPER_best_top1
    results["HyPER_best_top2"] = HyPER_best_top2
    results["HyPER_best_w1"] = HyPER_best_w1
    results["HyPER_best_w2"] = HyPER_best_w2

    results["HyPER_best_top1_prob"] = HyPER_best_top1_prob
    results["HyPER_best_top2_prob"] = HyPER_best_top2_prob
    results["HyPER_best_w1_prob"] = HyPER_best_w1_prob
    results["HyPER_best_w2_prob"] = HyPER_best_w2_prob

    columns_to_return = [
        "thad_first",
        "HyPER_best_top1",
        "HyPER_best_top2",
        "HyPER_best_w1",
        "HyPER_best_w2",
        "HyPER_best_top1_prob",
        "HyPER_best_top2_prob",
        "HyPER_best_w1_prob",
        "HyPER_best_w2_prob",
    ]

    columns_to_return = _add_cls_column_if_enabled(
        columns_to_return,
        results,
        use_classification,
    )

    return results[columns_to_return]


def ttbar_allhad(
    HyPER_outputs: str | pd.DataFrame,
    config: str | Path | dict[str, Any] | None = None,
    classification: bool | None = None,
) -> pd.DataFrame:
    r"""Reconstruct ttbar events with all-hadronic final states."""

    results = _load_results(HyPER_outputs)

    use_classification = _classification_enabled(
        config=config,
        classification=classification,
    )

    HyPER_best_top1 = []
    HyPER_best_top2 = []
    HyPER_best_w1 = []
    HyPER_best_w2 = []

    HyPER_best_top1_prob = []
    HyPER_best_top2_prob = []
    HyPER_best_w1_prob = []
    HyPER_best_w2_prob = []

    for i in tqdm(range(len(results)), desc="Reconstructing", unit="event"):
        HE_IDX = results["HyPER_HE_IDX"][i]
        HE_RAW = results["HyPER_HE_RAW"][i]
        GE_IDX = results["HyPER_GE_IDX"][i]
        GE_RAW = results["HyPER_GE_RAW"][i]

        selected_HE = []
        softProb_HE = []
        selected_GE = []
        softProb_GE = []

        completed_patterns = 0
        rank = np.argsort(HE_RAW)
        p = -1
        skip_event = False

        while completed_patterns < 2:
            if abs(p) > len(rank):
                skip_event = True
                break

            if completed_patterns > 0:
                for pattern in selected_HE:
                    while len(set(pattern).intersection(set(HE_IDX[rank[p]]))) != 0:
                        p -= 1
                        if abs(p) > len(rank):
                            skip_event = True
                            break

                    if skip_event:
                        break

            if skip_event:
                break

            selected_HE.append(HE_IDX[rank[p]])
            softProb_HE.append(float(HE_RAW[rank[p]]))

            best_edge, best_edge_score = _get_best_edge_in_pattern(
                pattern=HE_IDX[rank[p]],
                GE_IDX=GE_IDX,
                GE_RAW=GE_RAW,
            )

            selected_GE.append(best_edge)
            softProb_GE.append(best_edge_score)

            p -= 1
            completed_patterns += 1

        if skip_event or len(selected_HE) < 2 or len(selected_GE) < 2:
            HyPER_best_top1.append([-1, -1, -1])
            HyPER_best_top2.append([-1, -1, -1])
            HyPER_best_w1.append([-1, -1])
            HyPER_best_w2.append([-1, -1])

            HyPER_best_top1_prob.append(0.0)
            HyPER_best_top2_prob.append(0.0)
            HyPER_best_w1_prob.append(0.0)
            HyPER_best_w2_prob.append(0.0)
            continue

        HyPER_best_top1.append(selected_HE[0])
        HyPER_best_top2.append(selected_HE[1])
        HyPER_best_w1.append(selected_GE[0])
        HyPER_best_w2.append(selected_GE[1])

        HyPER_best_top1_prob.append(softProb_HE[0])
        HyPER_best_top2_prob.append(softProb_HE[1])
        HyPER_best_w1_prob.append(softProb_GE[0])
        HyPER_best_w2_prob.append(softProb_GE[1])

    results["HyPER_best_top1"] = HyPER_best_top1
    results["HyPER_best_top2"] = HyPER_best_top2
    results["HyPER_best_w1"] = HyPER_best_w1
    results["HyPER_best_w2"] = HyPER_best_w2

    results["HyPER_best_top1_prob"] = HyPER_best_top1_prob
    results["HyPER_best_top2_prob"] = HyPER_best_top2_prob
    results["HyPER_best_w1_prob"] = HyPER_best_w1_prob
    results["HyPER_best_w2_prob"] = HyPER_best_w2_prob

    columns_to_return = [
        "HyPER_best_top1",
        "HyPER_best_top2",
        "HyPER_best_w1",
        "HyPER_best_w2",
        "HyPER_best_top1_prob",
        "HyPER_best_top2_prob",
        "HyPER_best_w1_prob",
        "HyPER_best_w2_prob",
    ]

    columns_to_return = _add_cls_column_if_enabled(
        columns_to_return,
        results,
        use_classification,
    )

    return results[columns_to_return]
