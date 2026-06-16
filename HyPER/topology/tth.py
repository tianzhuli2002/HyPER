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


def _normalise_class_name(name):
    return str(name).strip().lower()


def _class_index(names, class_name):
    if names is None:
        return None
    wanted = _normalise_class_name(class_name)
    for idx, name in enumerate(list(names)):
        if _normalise_class_name(name) == wanted:
            return idx
    return None


def _class_scores(raw_scores, class_probs, class_names, class_name):
    class_idx = _class_index(class_names, class_name)
    if class_probs is None or class_idx is None:
        return np.asarray(raw_scores, dtype=float), False
    probs = np.asarray(class_probs, dtype=float)
    if probs.ndim != 2 or class_idx >= probs.shape[1]:
        return np.asarray(raw_scores, dtype=float), False
    return probs[:, class_idx], True


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


def _candidate_hyperedges(HE_IDX, HE_RAW, HE_VCT, predicate, max_candidates):
    candidates = []
    order = np.argsort(np.asarray(HE_RAW, dtype=float))[::-1]
    for idx in order:
        if not predicate(HE_VCT[idx]):
            continue
        candidates.append((int(idx), _as_list(HE_IDX[idx]), float(HE_RAW[idx])))
        if len(candidates) >= int(max_candidates):
            break
    return candidates


def _candidate_edges(GE_IDX, GE_RAW, GE_VCT, predicate, max_candidates):
    candidates = []
    order = np.argsort(np.asarray(GE_RAW, dtype=float))[::-1]
    for idx in order:
        if not predicate(GE_VCT[idx]):
            continue
        candidates.append((int(idx), _as_list(GE_IDX[idx]), float(GE_RAW[idx])))
        if len(candidates) >= int(max_candidates):
            break
    return candidates


def _default_assignment():
    return {
        "tlep_idx": None,
        "tlep_nodes": [-1, -1, -1],
        "tlep_score": 0.0,
        "thad_idx": None,
        "thad_nodes": [-1, -1, -1],
        "thad_score": 0.0,
        "whad_idx": None,
        "whad_nodes": [-1, -1],
        "whad_score": 0.0,
        "higgs_idx": None,
        "higgs_nodes": [-1, -1],
        "higgs_score": 0.0,
        "wlep_idx": None,
        "wlep_nodes": [-1, -1],
        "wlep_score": 0.0,
        "wlep_derived": False,
        "blep_node": -1,
        "bhad_node": -1,
        "whad_j1": -1,
        "whad_j2": -1,
        "higgs_b1": -1,
        "higgs_b2": -1,
    }


def _finish_thad_assignment(assignment):
    if assignment["thad_idx"] is None or assignment["whad_idx"] is None:
        return assignment
    whad_set = set(assignment["whad_nodes"])
    bhad_candidates = [
        node for node in assignment["thad_nodes"] if node not in whad_set
    ]
    assignment["bhad_node"] = bhad_candidates[0] if len(bhad_candidates) == 1 else -1
    assignment["whad_j1"] = assignment["whad_nodes"][0]
    assignment["whad_j2"] = assignment["whad_nodes"][1]
    return assignment


def _fill_wlep(assignment, GE_IDX, GE_RAW, GE_VCT):
    wlep_idx, wlep_nodes, wlep_score = _select_best_edge(
        GE_IDX,
        GE_RAW,
        GE_VCT,
        predicate=_is_wlep_edge,
    )
    assignment["wlep_idx"] = wlep_idx
    assignment["wlep_nodes"] = wlep_nodes
    assignment["wlep_score"] = wlep_score
    assignment["wlep_derived"] = False
    return assignment


def _derive_wlep_from_tlep(assignment, HE_VCT):
    tlep_idx = assignment.get("tlep_idx")
    if tlep_idx is None:
        return assignment
    nonjet_nodes = _nonjet_nodes_from_vct(assignment["tlep_nodes"], HE_VCT[tlep_idx])
    if len(nonjet_nodes) != 2:
        return assignment
    assignment["wlep_idx"] = None
    assignment["wlep_nodes"] = nonjet_nodes
    assignment["wlep_score"] = assignment["tlep_score"]
    assignment["wlep_derived"] = True
    return assignment


def _score_global_assignment(
    HE_IDX,
    HE_RAW,
    HE_VCT,
    GE_IDX,
    GE_RAW,
    GE_VCT,
    max_tlep_candidates,
    max_thad_candidates,
    max_pair_candidates,
    max_wlep_candidates,
    HE_THAD_RAW=None,
    GE_WHAD_RAW=None,
    GE_HIGGS_RAW=None,
    GE_WLEP_RAW=None,
):
    HE_THAD_RAW = HE_RAW if HE_THAD_RAW is None else HE_THAD_RAW
    GE_WHAD_RAW = GE_RAW if GE_WHAD_RAW is None else GE_WHAD_RAW
    GE_HIGGS_RAW = GE_RAW if GE_HIGGS_RAW is None else GE_HIGGS_RAW
    GE_WLEP_RAW = GE_RAW if GE_WLEP_RAW is None else GE_WLEP_RAW
    tlep_candidates = _candidate_hyperedges(
        HE_IDX, HE_RAW, HE_VCT, _is_tlep_hyperedge, max_tlep_candidates
    )
    thad_candidates = _candidate_hyperedges(
        HE_IDX, HE_THAD_RAW, HE_VCT, _is_thad_hyperedge, max_thad_candidates
    )
    whad_candidates = _candidate_edges(
        GE_IDX, GE_WHAD_RAW, GE_VCT, _is_whad_or_higgs_edge, max_pair_candidates
    )
    higgs_candidates = _candidate_edges(
        GE_IDX, GE_HIGGS_RAW, GE_VCT, _is_whad_or_higgs_edge, max_pair_candidates
    )
    wlep_candidates = _candidate_edges(
        GE_IDX, GE_WLEP_RAW, GE_VCT, _is_wlep_edge, max_wlep_candidates
    )

    best = None
    best_log_score = -np.inf
    eps = 1e-12

    for tlep_idx, tlep_nodes, tlep_score in tlep_candidates:
        tlep_jet_nodes = _jet_nodes_from_vct(tlep_nodes, HE_VCT[tlep_idx])
        if len(tlep_jet_nodes) != 1:
            continue
        blep_node = tlep_jet_nodes[0]

        for thad_idx, thad_nodes, thad_score in thad_candidates:
            thad_set = set(thad_nodes)
            if blep_node in thad_set:
                continue

            whad_options = [
                (idx, nodes, score)
                for idx, nodes, score in whad_candidates
                if set(nodes).issubset(thad_set)
            ]
            if not whad_options:
                continue

            higgs_options = [
                (idx, nodes, score)
                for idx, nodes, score in higgs_candidates
                if blep_node not in set(nodes) and not thad_set.intersection(nodes)
            ]
            if not higgs_options:
                continue

            for whad_idx, whad_nodes, whad_score in whad_options:
                whad_set = set(whad_nodes)
                bhad_candidates = [node for node in thad_nodes if node not in whad_set]
                if len(bhad_candidates) != 1:
                    continue

                for higgs_idx, higgs_nodes, higgs_score in higgs_options:
                    log_score = (
                        np.log(eps + max(float(tlep_score), 0.0))
                        + np.log(eps + max(float(thad_score), 0.0))
                        + np.log(eps + max(float(whad_score), 0.0))
                        + np.log(eps + max(float(higgs_score), 0.0))
                    )
                    if log_score <= best_log_score:
                        continue

                    assignment = _default_assignment()
                    assignment.update(
                        {
                            "tlep_idx": tlep_idx,
                            "tlep_nodes": tlep_nodes,
                            "tlep_score": tlep_score,
                            "thad_idx": thad_idx,
                            "thad_nodes": thad_nodes,
                            "thad_score": thad_score,
                            "whad_idx": whad_idx,
                            "whad_nodes": whad_nodes,
                            "whad_score": whad_score,
                            "higgs_idx": higgs_idx,
                            "higgs_nodes": higgs_nodes,
                            "higgs_score": higgs_score,
                            "blep_node": blep_node,
                            "bhad_node": bhad_candidates[0],
                            "whad_j1": whad_nodes[0],
                            "whad_j2": whad_nodes[1],
                            "higgs_b1": higgs_nodes[0],
                            "higgs_b2": higgs_nodes[1],
                        }
                    )
                    best = assignment
                    best_log_score = log_score

    if best is None:
        best = _default_assignment()

    if wlep_candidates:
        wlep_idx, wlep_nodes, wlep_score = wlep_candidates[0]
        best["wlep_idx"] = wlep_idx
        best["wlep_nodes"] = wlep_nodes
        best["wlep_score"] = wlep_score

    return best


def _jet_nodes_from_vct(nodes, vct):
    return [node for node, kind in zip(_as_list(nodes), _as_list(vct)) if kind == 1]


def _nonjet_nodes_from_vct(nodes, vct):
    return [node for node, kind in zip(_as_list(nodes), _as_list(vct)) if kind != 1]


def ttH_single_lep(
    HyPER_outputs: str | pd.DataFrame,
    classification: bool | None = None,
    strategy: str = "thad_first",
    max_tlep_candidates: int = 20,
    max_thad_candidates: int = 30,
    max_pair_candidates: int = 60,
    max_wlep_candidates: int = 10,
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

    valid_strategies = {"thad_first", "higgs_first", "score_global"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown ttH reconstruction strategy {strategy!r}; "
            f"expected one of {sorted(valid_strategies)}"
        )

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
    HyPER_best_wlep_derived = []

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
        HE_PROBS = results["HyPER_HE_CLASS_PROBS"][i] if "HyPER_HE_CLASS_PROBS" in results.columns else None
        GE_PROBS = results["HyPER_GE_CLASS_PROBS"][i] if "HyPER_GE_CLASS_PROBS" in results.columns else None
        HE_CLASS_NAMES = results["HyPER_HE_CLASS_NAMES"][i] if "HyPER_HE_CLASS_NAMES" in results.columns else None
        GE_CLASS_NAMES = results["HyPER_GE_CLASS_NAMES"][i] if "HyPER_GE_CLASS_NAMES" in results.columns else None

        HE_TLEP_RAW, _ = _class_scores(HE_RAW, HE_PROBS, HE_CLASS_NAMES, "tlep")
        HE_THAD_RAW, _ = _class_scores(HE_RAW, HE_PROBS, HE_CLASS_NAMES, "thad")
        GE_WHAD_RAW, _ = _class_scores(GE_RAW, GE_PROBS, GE_CLASS_NAMES, "Whad")
        GE_HIGGS_RAW, _ = _class_scores(GE_RAW, GE_PROBS, GE_CLASS_NAMES, "H")
        GE_WLEP_RAW, has_wlep_class = _class_scores(GE_RAW, GE_PROBS, GE_CLASS_NAMES, "Wlep")

        assignment = _default_assignment()

        if strategy == "score_global":
            assignment = _score_global_assignment(
                HE_IDX,
                HE_TLEP_RAW,
                HE_VCT,
                GE_IDX,
                GE_RAW,
                GE_VCT,
                max_tlep_candidates=max_tlep_candidates,
                max_thad_candidates=max_thad_candidates,
                max_pair_candidates=max_pair_candidates,
                max_wlep_candidates=max_wlep_candidates,
                HE_THAD_RAW=HE_THAD_RAW,
                GE_WHAD_RAW=GE_WHAD_RAW,
                GE_HIGGS_RAW=GE_HIGGS_RAW,
                GE_WLEP_RAW=GE_WLEP_RAW,
            )
        else:
            # 1. Select leptonic top: b_lep + lepton + MET.
            tlep_idx, tlep_nodes, tlep_score = _select_best_hyperedge(
                HE_IDX,
                HE_TLEP_RAW,
                HE_VCT,
                predicate=_is_tlep_hyperedge,
            )
            assignment.update(
                {
                    "tlep_idx": tlep_idx,
                    "tlep_nodes": tlep_nodes,
                    "tlep_score": tlep_score,
                }
            )

            if tlep_idx is not None:
                tlep_jet_nodes = _jet_nodes_from_vct(tlep_nodes, HE_VCT[tlep_idx])
                assignment["blep_node"] = (
                    tlep_jet_nodes[0] if len(tlep_jet_nodes) == 1 else -1
                )

        if strategy == "thad_first":
            # 2. Select hadronic top: three jets, disjoint from b_lep.
            blep_node = assignment["blep_node"]
            thad_forbidden = {blep_node} if blep_node >= 0 else set()
            thad_idx, thad_nodes, thad_score = _select_best_hyperedge(
                HE_IDX,
                HE_THAD_RAW,
                HE_VCT,
                predicate=_is_thad_hyperedge,
                forbidden_nodes=thad_forbidden,
            )
            assignment.update(
                {
                    "thad_idx": thad_idx,
                    "thad_nodes": thad_nodes,
                    "thad_score": thad_score,
                }
            )

            thad_node_set = set(thad_nodes) if thad_idx is not None else set()

            # 3. Select Whad: best two-jet edge inside THAD.
            whad_idx, whad_nodes, whad_score = _select_best_edge(
                GE_IDX,
                GE_WHAD_RAW,
                GE_VCT,
                predicate=_is_whad_or_higgs_edge,
                required_subset=thad_node_set if thad_idx is not None else None,
            )
            assignment.update(
                {
                    "whad_idx": whad_idx,
                    "whad_nodes": whad_nodes,
                    "whad_score": whad_score,
                }
            )
            assignment = _finish_thad_assignment(assignment)

            # 4. Select H: best two-jet edge disjoint from THAD and b_lep.
            higgs_forbidden = set()
            if thad_idx is not None:
                higgs_forbidden.update(thad_nodes)
            if blep_node >= 0:
                higgs_forbidden.add(blep_node)

            higgs_idx, higgs_nodes, higgs_score = _select_best_edge(
                GE_IDX,
                GE_HIGGS_RAW,
                GE_VCT,
                predicate=_is_whad_or_higgs_edge,
                forbidden_nodes=higgs_forbidden,
            )

            if higgs_idx is not None:
                assignment["higgs_b1"] = higgs_nodes[0]
                assignment["higgs_b2"] = higgs_nodes[1]
            assignment.update(
                {
                    "higgs_idx": higgs_idx,
                    "higgs_nodes": higgs_nodes,
                    "higgs_score": higgs_score,
                }
            )

            # 5. Select Wlep if it is a trained edge class. For ttH_SL typed
            # models it is derived from the selected tlep hyperedge.
            assignment = (
                _fill_wlep(assignment, GE_IDX, GE_WLEP_RAW, GE_VCT)
                if has_wlep_class or GE_PROBS is None
                else _derive_wlep_from_tlep(assignment, HE_VCT)
            )
        elif strategy == "higgs_first":
            # 2. Select H first: best two-jet edge disjoint from b_lep.
            blep_node = assignment["blep_node"]
            higgs_forbidden = {blep_node} if blep_node >= 0 else set()

            higgs_idx, higgs_nodes, higgs_score = _select_best_edge(
                GE_IDX,
                GE_HIGGS_RAW,
                GE_VCT,
                predicate=_is_whad_or_higgs_edge,
                forbidden_nodes=higgs_forbidden,
            )

            if higgs_idx is not None:
                assignment["higgs_b1"] = higgs_nodes[0]
                assignment["higgs_b2"] = higgs_nodes[1]
            assignment.update(
                {
                    "higgs_idx": higgs_idx,
                    "higgs_nodes": higgs_nodes,
                    "higgs_score": higgs_score,
                }
            )

            # 3. Select thad disjoint from b_lep and Higgs.
            thad_forbidden = set()
            if blep_node >= 0:
                thad_forbidden.add(blep_node)
            if higgs_idx is not None:
                thad_forbidden.update(higgs_nodes)

            thad_idx, thad_nodes, thad_score = _select_best_hyperedge(
                HE_IDX,
                HE_THAD_RAW,
                HE_VCT,
                predicate=_is_thad_hyperedge,
                forbidden_nodes=thad_forbidden,
            )
            assignment.update(
                {
                    "thad_idx": thad_idx,
                    "thad_nodes": thad_nodes,
                    "thad_score": thad_score,
                }
            )

            thad_node_set = set(thad_nodes) if thad_idx is not None else set()

            # 4. Select Whad inside thad.
            whad_idx, whad_nodes, whad_score = _select_best_edge(
                GE_IDX,
                GE_WHAD_RAW,
                GE_VCT,
                predicate=_is_whad_or_higgs_edge,
                required_subset=thad_node_set if thad_idx is not None else None,
            )
            assignment.update(
                {
                    "whad_idx": whad_idx,
                    "whad_nodes": whad_nodes,
                    "whad_score": whad_score,
                }
            )
            assignment = _finish_thad_assignment(assignment)
            assignment = (
                _fill_wlep(assignment, GE_IDX, GE_WLEP_RAW, GE_VCT)
                if has_wlep_class or GE_PROBS is None
                else _derive_wlep_from_tlep(assignment, HE_VCT)
            )
        if strategy == "score_global" and GE_PROBS is not None and not has_wlep_class:
            assignment = _derive_wlep_from_tlep(assignment, HE_VCT)

        tlep_idx = assignment["tlep_idx"]
        tlep_nodes = assignment["tlep_nodes"]
        tlep_score = assignment["tlep_score"]
        thad_idx = assignment["thad_idx"]
        thad_nodes = assignment["thad_nodes"]
        thad_score = assignment["thad_score"]
        whad_idx = assignment["whad_idx"]
        whad_nodes = assignment["whad_nodes"]
        whad_score = assignment["whad_score"]
        higgs_idx = assignment["higgs_idx"]
        higgs_nodes = assignment["higgs_nodes"]
        higgs_score = assignment["higgs_score"]
        wlep_idx = assignment["wlep_idx"]
        wlep_nodes = assignment["wlep_nodes"]
        wlep_score = assignment["wlep_score"]
        wlep_derived = assignment["wlep_derived"]
        blep_node = assignment["blep_node"]
        bhad_node = assignment["bhad_node"]
        whad_j1 = assignment["whad_j1"]
        whad_j2 = assignment["whad_j2"]
        higgs_b1 = assignment["higgs_b1"]
        higgs_b2 = assignment["higgs_b2"]

        this_tlep_valid = int(tlep_idx is not None and blep_node >= 0)
        this_thad_valid = int(thad_idx is not None)
        this_whad_valid = int(whad_idx is not None and bhad_node >= 0)
        this_higgs_valid = int(higgs_idx is not None)
        this_wlep_valid = int((wlep_idx is not None) or bool(wlep_derived))

        this_reco_valid = int(
            this_tlep_valid
            and this_thad_valid
            and this_whad_valid
            and this_higgs_valid
        )

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
        HyPER_best_wlep_derived.append(bool(wlep_derived))

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
    results["HyPER_best_wlep_derived"] = HyPER_best_wlep_derived

    results["reco_valid"] = reco_valid
    results["tlep_valid"] = tlep_valid
    results["thad_valid"] = thad_valid
    results["wlep_valid"] = wlep_valid
    results["whad_valid"] = whad_valid
    results["higgs_valid"] = higgs_valid
    results["reco_strategy"] = strategy

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
        "HyPER_best_wlep_derived",
        "reco_valid",
        "tlep_valid",
        "thad_valid",
        "wlep_valid",
        "whad_valid",
        "higgs_valid",
        "reco_strategy",
    ]

    if (classification is None or bool(classification)) and "HyPER_CLS_RAW" in results.columns:
        columns_to_return.append("HyPER_CLS_RAW")

    return results[columns_to_return]


# Alias with shorter naming for config/CLI dispatch.
def tth_single_lep(
    HyPER_outputs: str | pd.DataFrame,
    classification: bool | None = None,
    strategy: str = "thad_first",
    **kwargs,
):
    return ttH_single_lep(
        HyPER_outputs,
        classification=classification,
        strategy=strategy,
        **kwargs,
    )
