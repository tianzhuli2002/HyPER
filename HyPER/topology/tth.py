import numpy as np
import pandas as pd

from tqdm.rich import tqdm


def _load_hyper_outputs(HyPER_outputs: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(HyPER_outputs, pd.DataFrame):
        return HyPER_outputs.copy()
    if isinstance(HyPER_outputs, str):
        return pd.read_pickle(HyPER_outputs)
    raise ValueError(
        f"Unrecognised HyPER output type {type(HyPER_outputs)}, "
        "it must be `str` or `pandas.DataFrame`."
    )


def _as_list(x):
    return list(np.asarray(x, dtype=int).reshape(-1))


def _score_or_zero(scores, idx):
    if idx is None:
        return 0.0
    return float(scores[idx])


def _vct_counts(vct):
    values = _as_list(vct)
    return {
        "jet": sum(v == 1 for v in values),
        "lepton": sum(v == 2 for v in values),
        "met": sum(v == 3 for v in values),
    }


def _is_tlep_hyperedge(vct):
    counts = _vct_counts(vct)
    return counts["jet"] == 1 and counts["lepton"] == 1 and counts["met"] == 1


def _is_thad_hyperedge(vct):
    counts = _vct_counts(vct)
    return counts["jet"] == 3 and counts["lepton"] == 0 and counts["met"] == 0


def _is_whad_or_higgs_edge(vct):
    counts = _vct_counts(vct)
    return counts["jet"] == 2 and counts["lepton"] == 0 and counts["met"] == 0


def _is_wlep_edge(vct):
    counts = _vct_counts(vct)
    return counts["jet"] == 0 and counts["lepton"] == 1 and counts["met"] == 1


def _select_best_hyperedge(HE_IDX, HE_RAW, HE_VCT, predicate, forbidden_nodes=None):
    forbidden_nodes = set() if forbidden_nodes is None else set(forbidden_nodes)
    order = np.argsort(np.asarray(HE_RAW, dtype=float))[::-1]

    for idx in order:
        nodes = _as_list(HE_IDX[idx])
        if forbidden_nodes.intersection(nodes):
            continue
        if predicate(HE_VCT[idx]):
            return idx, nodes, float(HE_RAW[idx])

    return None, [-1, -1, -1], 0.0


def _select_best_edge(GE_IDX, GE_RAW, GE_VCT, predicate, required_subset=None, forbidden_nodes=None):
    required_subset = None if required_subset is None else set(required_subset)
    forbidden_nodes = set() if forbidden_nodes is None else set(forbidden_nodes)
    order = np.argsort(np.asarray(GE_RAW, dtype=float))[::-1]

    for idx in order:
        nodes = _as_list(GE_IDX[idx])
        node_set = set(nodes)

        if forbidden_nodes.intersection(node_set):
            continue
        if required_subset is not None and not node_set.issubset(required_subset):
            continue
        if predicate(GE_VCT[idx]):
            return idx, nodes, float(GE_RAW[idx])

    return None, [-1, -1], 0.0


def _jet_nodes_from_vct(nodes, vct):
    return [node for node, kind in zip(_as_list(nodes), _as_list(vct)) if kind == 1]


def _nonjet_nodes_from_vct(nodes, vct):
    return [node for node, kind in zip(_as_list(nodes), _as_list(vct)) if kind != 1]


def ttH_single_lep(
    HyPER_outputs: str | pd.DataFrame,
    classification: bool | None = None,
):
    r"""Reconstruct ttH events with single-lepton ttH topology.

    Assumed training-label convention:

        1 = b_had
        2 = W_HAD_J1
        3 = W_HAD_J2
        4 = b_lep
        5 = HIGGS_B1
        6 = HIGGS_B2

    The predicted topology contract is:

        edge:
          Wlep = lepton + MET
          Whad = W_HAD_J1 + W_HAD_J2
          H    = HIGGS_B1 + HIGGS_B2

        hyperedge:
          tlep = b_lep + lepton + MET
          thad = b_had + W_HAD_J1 + W_HAD_J2

    This function reconstructs candidate node indices and scores only. It does
    not build four-vectors or decorate an H5. That should be done by the H5
    decorator using these selected node indices.
    """

    results = _load_hyper_outputs(HyPER_outputs)

    HyPER_best_tlep = []
    HyPER_best_thad = []
    HyPER_best_wlep = []
    HyPER_best_whad = []
    HyPER_best_higgs = []

    HyPER_best_blep = []
    HyPER_best_bhad = []
    HyPER_best_whad_j1 = []
    HyPER_best_whad_j2 = []
    HyPER_best_higgs_b1 = []
    HyPER_best_higgs_b2 = []

    HyPER_best_tlep_prob = []
    HyPER_best_thad_prob = []
    HyPER_best_wlep_prob = []
    HyPER_best_whad_prob = []
    HyPER_best_higgs_prob = []
    HyPER_best_event_score = []

    reco_valid = []
    tlep_valid = []
    thad_valid = []
    wlep_valid = []
    whad_valid = []
    higgs_valid = []

    for i in tqdm(range(len(results)), desc="Reconstructing ttH", unit="event"):
        HE_IDX = results["HyPER_HE_IDX"][i]
        HE_RAW = results["HyPER_HE_RAW"][i]
        GE_IDX = results["HyPER_GE_IDX"][i]
        GE_RAW = results["HyPER_GE_RAW"][i]
        HE_VCT = results["HyPER_HE_VCT"][i]
        GE_VCT = results["HyPER_GE_VCT"][i]

        # 1. Select leptonic top: b_lep + lepton + MET.
        tlep_idx, tlep_nodes, tlep_score = _select_best_hyperedge(
            HE_IDX,
            HE_RAW,
            HE_VCT,
            predicate=_is_tlep_hyperedge,
        )

        if tlep_idx is not None:
            tlep_jet_nodes = _jet_nodes_from_vct(tlep_nodes, HE_VCT[tlep_idx])
            blep_node = tlep_jet_nodes[0] if len(tlep_jet_nodes) == 1 else -1
        else:
            blep_node = -1

        # 2. Select hadronic top: three jets, disjoint from b_lep.
        thad_forbidden = {blep_node} if blep_node >= 0 else set()
        thad_idx, thad_nodes, thad_score = _select_best_hyperedge(
            HE_IDX,
            HE_RAW,
            HE_VCT,
            predicate=_is_thad_hyperedge,
            forbidden_nodes=thad_forbidden,
        )

        thad_node_set = set(thad_nodes) if thad_idx is not None else set()

        # 3. Select Whad: best two-jet edge inside THAD.
        whad_idx, whad_nodes, whad_score = _select_best_edge(
            GE_IDX,
            GE_RAW,
            GE_VCT,
            predicate=_is_whad_or_higgs_edge,
            required_subset=thad_node_set if thad_idx is not None else None,
        )

        if thad_idx is not None and whad_idx is not None:
            whad_set = set(whad_nodes)
            bhad_candidates = [node for node in thad_nodes if node not in whad_set]
            bhad_node = bhad_candidates[0] if len(bhad_candidates) == 1 else -1
            whad_j1 = whad_nodes[0]
            whad_j2 = whad_nodes[1]
        else:
            bhad_node = -1
            whad_j1 = -1
            whad_j2 = -1

        # 4. Select H: best two-jet edge disjoint from THAD and b_lep.
        higgs_forbidden = set()
        if thad_idx is not None:
            higgs_forbidden.update(thad_nodes)
        if blep_node >= 0:
            higgs_forbidden.add(blep_node)

        higgs_idx, higgs_nodes, higgs_score = _select_best_edge(
            GE_IDX,
            GE_RAW,
            GE_VCT,
            predicate=_is_whad_or_higgs_edge,
            forbidden_nodes=higgs_forbidden,
        )

        if higgs_idx is not None:
            higgs_b1 = higgs_nodes[0]
            higgs_b2 = higgs_nodes[1]
        else:
            higgs_b1 = -1
            higgs_b2 = -1

        # 5. Select Wlep: best lepton-MET edge.
        wlep_idx, wlep_nodes, wlep_score = _select_best_edge(
            GE_IDX,
            GE_RAW,
            GE_VCT,
            predicate=_is_wlep_edge,
        )

        this_tlep_valid = int(tlep_idx is not None and blep_node >= 0)
        this_thad_valid = int(thad_idx is not None)
        this_whad_valid = int(whad_idx is not None and bhad_node >= 0)
        this_higgs_valid = int(higgs_idx is not None)
        this_wlep_valid = int(wlep_idx is not None)

        this_reco_valid = int(
            this_tlep_valid
            and this_thad_valid
            and this_whad_valid
            and this_higgs_valid
        )

        if this_reco_valid:
            event_score = float(tlep_score * thad_score * whad_score * higgs_score)
        else:
            event_score = 0.0

        HyPER_best_tlep.append(tlep_nodes)
        HyPER_best_thad.append(thad_nodes)
        HyPER_best_wlep.append(wlep_nodes)
        HyPER_best_whad.append(whad_nodes)
        HyPER_best_higgs.append(higgs_nodes)

        HyPER_best_blep.append(blep_node)
        HyPER_best_bhad.append(bhad_node)
        HyPER_best_whad_j1.append(whad_j1)
        HyPER_best_whad_j2.append(whad_j2)
        HyPER_best_higgs_b1.append(higgs_b1)
        HyPER_best_higgs_b2.append(higgs_b2)

        HyPER_best_tlep_prob.append(tlep_score)
        HyPER_best_thad_prob.append(thad_score)
        HyPER_best_wlep_prob.append(wlep_score)
        HyPER_best_whad_prob.append(whad_score)
        HyPER_best_higgs_prob.append(higgs_score)
        HyPER_best_event_score.append(event_score)

        reco_valid.append(this_reco_valid)
        tlep_valid.append(this_tlep_valid)
        thad_valid.append(this_thad_valid)
        wlep_valid.append(this_wlep_valid)
        whad_valid.append(this_whad_valid)
        higgs_valid.append(this_higgs_valid)

    results["HyPER_best_tlep"] = HyPER_best_tlep
    results["HyPER_best_thad"] = HyPER_best_thad
    results["HyPER_best_wlep"] = HyPER_best_wlep
    results["HyPER_best_whad"] = HyPER_best_whad
    results["HyPER_best_higgs"] = HyPER_best_higgs

    results["HyPER_best_blep"] = HyPER_best_blep
    results["HyPER_best_bhad"] = HyPER_best_bhad
    results["HyPER_best_whad_j1"] = HyPER_best_whad_j1
    results["HyPER_best_whad_j2"] = HyPER_best_whad_j2
    results["HyPER_best_higgs_b1"] = HyPER_best_higgs_b1
    results["HyPER_best_higgs_b2"] = HyPER_best_higgs_b2

    results["HyPER_best_tlep_prob"] = HyPER_best_tlep_prob
    results["HyPER_best_thad_prob"] = HyPER_best_thad_prob
    results["HyPER_best_wlep_prob"] = HyPER_best_wlep_prob
    results["HyPER_best_whad_prob"] = HyPER_best_whad_prob
    results["HyPER_best_higgs_prob"] = HyPER_best_higgs_prob
    results["HyPER_best_event_score"] = HyPER_best_event_score

    results["reco_valid"] = reco_valid
    results["tlep_valid"] = tlep_valid
    results["thad_valid"] = thad_valid
    results["wlep_valid"] = wlep_valid
    results["whad_valid"] = whad_valid
    results["higgs_valid"] = higgs_valid

    columns_to_return = [
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
        "HyPER_best_tlep_prob",
        "HyPER_best_thad_prob",
        "HyPER_best_wlep_prob",
        "HyPER_best_whad_prob",
        "HyPER_best_higgs_prob",
        "HyPER_best_event_score",
        "reco_valid",
        "tlep_valid",
        "thad_valid",
        "wlep_valid",
        "whad_valid",
        "higgs_valid",
    ]

    if (classification is None or bool(classification)) and "HyPER_CLS_RAW" in results.columns:
        columns_to_return.append("HyPER_CLS_RAW")

    return results[columns_to_return]


# Alias with shorter naming for config/CLI dispatch.
def tth_single_lep(
    HyPER_outputs: str | pd.DataFrame,
    classification: bool | None = None,
):
    return ttH_single_lep(HyPER_outputs, classification=classification)
