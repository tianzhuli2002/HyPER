import numpy as np
import pandas as pd

from itertools import combinations
from tqdm.rich import tqdm


def ttbar_dilep(HyPER_outputs: str | pd.DataFrame):
    r"""Reconstruct ttbar events with di-leptonic final states.

    """
    if   type(HyPER_outputs) is pd.DataFrame:
        results = HyPER_outputs
    elif type(HyPER_outputs) is str:
        results = pd.read_pickle(HyPER_outputs)
    else:
        raise ValueError(f"Unrecognised HyPER output type {type(HyPER_outputs)}, it must be `str` or `pandas.DataFrame`.")
    
    #results["event_idx"] = np.arange(len(results))

    # Current strategy is to not consider the hyperedge (still keeping the score in outputs, expect huge correlation)
    HyPER_best_top1 = []
    HyPER_best_top2 = []
    #HyPER_best_HE = []
    
    HyPER_best_top1_score   = []
    HyPER_best_top2_score   = []
    HyPER_best_event_score  = [] # hyperedge of order 4

   
    # TO DO: save hyperedge score corresponding to the selected graph edges
    for i in tqdm(range(len(results)), desc="Reconstructing", unit='event'):
        HE_IDX = results['HyPER_HE_IDX'][i]
        HE_RAW = results['HyPER_HE_RAW'][i]
        GE_IDX = results['HyPER_GE_IDX'][i]
        GE_RAW = results['HyPER_GE_RAW'][i]

        HE_VCT = results['HyPER_HE_VCT'][i]
        skip_event = False
        selected_HE = []    # Selected HyperEdge
        softProb_HE = []    # Soft probability of the selected HyperEdge
        selected_GE = []    # Selected GraphEdge
        softProb_GE = []    # Soft probability of the selected GraphEdge

        completed_patterns = 0
        rank = np.argsort(GE_RAW)
        p    = -1           # current position

        # We need 2 top quarks
        while completed_patterns < 2:
            if completed_patterns == 0:
                pass
            
            if completed_patterns > 0:
                # for pattern in selected_GE:
                #     while len(set(pattern).intersection(set(GE_IDX[rank[p]]))) != 0:
                #         p -= 1
                        
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

            # print(f"Selected pattern {GE_IDX[rank[p]]} with rank {rank[p]} and score {GE_RAW[rank[p]]}")

            selected_GE.append(GE_IDX[rank[p]])
            softProb_GE.append(GE_RAW[rank[p]])

            p -= 1
            completed_patterns += 1

        if skip_event or len(selected_GE) < 2:
            HyPER_best_top1.append(None)
            HyPER_best_top2.append(None)
            HyPER_best_top1_score.append(0.0)
            HyPER_best_top2_score.append(0.0)
            HyPER_best_event_score.append(0.0)
            continue
        
        hyperedge_to_find = set(selected_GE[0]) | set(selected_GE[1]) # Union of the edges

        # Find the hyperedge
        matched_HE_index = None
        for j, he in enumerate(HE_IDX):
            if hyperedge_to_find.issubset(set(he)):
                matched_HE_index = j
                break

        HyPER_best_top1.append(selected_GE[0])
        HyPER_best_top2.append(selected_GE[1])

        HyPER_best_top1_score.append(softProb_GE[0])
        HyPER_best_top2_score.append(softProb_GE[1])
        # print("Matched_HE_index:", matched_HE_index)
        # print("len(HE_RAW):", len(HE_RAW))
        #print(f"Matched HE index {matched_HE_index} with score {HE_RAW[matched_HE_index]}")
        if matched_HE_index is None or matched_HE_index >= len(HE_RAW):
            HyPER_best_event_score.append(0.0)
        else:
            HyPER_best_event_score.append(HE_RAW[matched_HE_index])

        #HyPER_best_event_score.append(HE_RAW[matched_HE_index])
        

    results['HyPER_best_top1'] = HyPER_best_top1
    results['HyPER_best_top2'] = HyPER_best_top2

    results['HyPER_best_top1_score']    = HyPER_best_top1_score
    results['HyPER_best_top2_score']    = HyPER_best_top2_score
    results['HyPER_best_event_score']   = HyPER_best_event_score
    #results['HyPER_CLS_RAW']            = results['HyPER_CLS_RAW']  # just to be sure it's there
    columns_to_return = [
        'HyPER_best_top1',
        'HyPER_best_top2',
        'HyPER_best_top1_score',
        'HyPER_best_top2_score',
        'HyPER_CLS_RAW'
    ]

    # Filter results to include only the selected columns (else pickle file might be too heavy)
    #TO DO: add this as an option in the config file
    light_results = results[columns_to_return]
    
    return light_results #light_results

def ttbar_single_lep_cluster(HyPER_outputs: str | pd.DataFrame):
    r"""Reconstruct ttbar events with lepton+jets final states.

    """
    if   type(HyPER_outputs) is pd.DataFrame:
        results = HyPER_outputs
    elif type(HyPER_outputs) is str:
        results = pd.read_pickle(HyPER_outputs)
    else:
        raise ValueError(f"Unrecognised HyPER output type {type(HyPER_outputs)}, it must be `str` or `pandas.DataFrame`.")

    HyPER_best_top1 = []
    HyPER_best_top2 = []
    HyPER_best_w1 = []
    HyPER_best_w2 = []

    HyPER_best_top1_prob = []
    HyPER_best_top2_prob = []
    HyPER_best_w1_prob = []
    HyPER_best_w2_prob = []
    
    thad_first  = []    #to remember if thad or tlep was reconstructed first (i.e. had the best score)

    def is_leptonic(vct):
            return len(set(vct).intersection(set([2, 3]))) == 2 

    def is_hadronic(vct):
        return all(v == 1 for v in vct)

    for i in tqdm(range(len(results)), desc="Reconstructing", unit='event'):
        if i % 1000 == 0:
            print(f"Processing event {i}/{len(results)}")

        HE_IDX = results['HyPER_HE_IDX'][i]
        HE_RAW = results['HyPER_HE_RAW'][i]
        GE_IDX = results['HyPER_GE_IDX'][i]
        GE_RAW = results['HyPER_GE_RAW'][i]

        HE_VCT = results['HyPER_HE_VCT'][i]
        GE_VCT = results['HyPER_GE_VCT'][i]

        selected_HE = []    # Selected HyperEdge
        softProb_HE = []    # Soft probability of the selected HyperEdge
        selected_GE = []    # Selected GraphEdge
        softProb_GE = []    # Soft probability of the selected GraphEdge

        completed_patterns = 0
        tlep_position = 0 #was leptonic top reconstructed first (0) or not (1)
        rank = np.argsort(HE_RAW) #array of sorted indices
        p    = -1           # current position

        # We need 2 top quarks
        while completed_patterns < 2:
            if completed_patterns == 0: 
                #Make sure its not reconstructing MET and lepton in different tops
                while (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) == 1): 
                    p -=1 

                #check if best top is tlep (!!! Here we assume there is only one charged lepton per event in the tuples !!!)
                if (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) != 2):
                    tlep_position = 1
                
            
            if completed_patterns > 0:
                for pattern in selected_HE:
                    #Check that the other reconstructed top does not contain the same nodes and is/isnt leptonic
                    while (len(set(pattern).intersection(set(HE_IDX[rank[p]]))) != 0) or ((tlep_position == 1) and (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) != 2)): 
                        p -= 1

            #Check for clusters of hadronic tops 
            if completed_patterns != tlep_position: #check out for "clusters of hadronic tops", i.e. bunch of hadronic tops with similar scores. In those cases, we want to consider the WHad score.
                cluster_candidates_idx = [rank[p]]
                
                p_cluster = p-1 
                jet_already_in_top = False

                while abs(p_cluster) < len(rank): #fill cluster candidates
                    if is_hadronic(HE_VCT[rank[p_cluster]]) and (HE_RAW[rank[p_cluster]] > 0.8*HE_RAW[rank[p]]): #check if its a hadronic top and if its score is similar to best one 
                        if completed_patterns > 0:
                            for pattern in selected_HE:
                                if len(set(pattern).intersection(set(HE_IDX[rank[p_cluster]]))) != 0:
                                    jet_already_in_top = True
                        if not jet_already_in_top:
                            cluster_candidates_idx.append(rank[p_cluster])
                    p_cluster -= 1
                    jet_already_in_top = False

                nominal_top_idx = rank[p]
                nominal_edge_idx = 0
                nominal_score = 0

                for idx in cluster_candidates_idx: #chose best candidate in the cluster
                    best_edge_score_in_pattern = 0
                    for possible_edge in list(combinations(HE_IDX[idx],r=2)):
                        # this looks complated but it is faster during computing
                        edge_in_pattern = np.argwhere(np.sum(np.where(np.array(GE_IDX)==possible_edge[0],1,0) + np.where(np.array(GE_IDX)==possible_edge[1],1,0),axis=1)==2).flatten()[0]

                        if GE_RAW[edge_in_pattern] > best_edge_score_in_pattern:
                            if (completed_patterns != tlep_position) or ((completed_patterns == tlep_position) and (len(set(GE_VCT[edge_in_pattern]).intersection(set([2,3]))) == 2)): #Wlep automatically found as constituted of lepton+MET
                                best_edge_score_in_pattern = GE_RAW[edge_in_pattern]
                                best_edge_in_pattern = edge_in_pattern
                    candidateScore = HE_RAW[idx]*0.01 + GE_RAW[best_edge_in_pattern]*0.99
                    if candidateScore > nominal_score:                  
                        nominal_top_idx = idx
                        nominal_edge_idx = best_edge_in_pattern
                        nominal_score = candidateScore
                        if nominal_top_idx != rank[p]:
                            print(f"Incredible, the cluster algo actually made smth happen, topLep in position {tlep_position}")
                            print(f"Scores for cluster candidates: {HE_RAW[cluster_candidates_idx[0]]}, {HE_RAW[cluster_candidates_idx[1]]}")

                if HE_RAW[nominal_top_idx] != 0:
                    selected_HE.append(HE_IDX[nominal_top_idx])
                    softProb_HE.append(HE_RAW[nominal_top_idx])
                else:
                    selected_HE.append([-1, -1, -1])
                    softProb_HE.append(0)

                if GE_RAW[nominal_edge_idx] != 0:
                    selected_GE.append(GE_IDX[nominal_edge_idx])
                    softProb_GE.append(GE_RAW[nominal_edge_idx])
                else:
                    selected_GE.append([-1,-1])
                    softProb_GE.append(0)
            else:
                if HE_RAW[rank[p]] != 0:
                    selected_HE.append(HE_IDX[rank[p]])
                    softProb_HE.append(HE_RAW[rank[p]])
                else:
                    selected_HE.append([-1, -1, -1])
                    softProb_HE.append(0)
                
                # Find best edge in pattern
                best_edge_score_in_pattern = 0
                for possible_edge in list(combinations(HE_IDX[rank[p]],r=2)):
                    # this looks complated but it is faster during computing
                    edge_in_pattern = np.argwhere(np.sum(np.where(np.array(GE_IDX)==possible_edge[0],1,0) + np.where(np.array(GE_IDX)==possible_edge[1],1,0),axis=1)==2).flatten()[0]

                    if GE_RAW[edge_in_pattern] > best_edge_score_in_pattern:
                        if (completed_patterns != tlep_position) or ((completed_patterns == tlep_position) and (len(set(GE_VCT[edge_in_pattern]).intersection(set([2,3]))) == 2)): #Wlep automatically found as constituted of lepton+MET
                            best_edge_score_in_pattern = GE_RAW[edge_in_pattern]
                            best_edge_in_pattern = list(possible_edge)


                if best_edge_score_in_pattern != 0:
                    selected_GE.append(best_edge_in_pattern)
                    softProb_GE.append(best_edge_score_in_pattern)
                else:
                    selected_GE.append([-1,-1])
                    softProb_GE.append(0)

            p -= 1
            completed_patterns += 1

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

    results['thad_first'] = thad_first

    results['HyPER_best_top1'] = HyPER_best_top1
    results['HyPER_best_top2'] = HyPER_best_top2
    results['HyPER_best_w1'] = HyPER_best_w1
    results['HyPER_best_w2'] = HyPER_best_w2

    results['HyPER_best_top1_prob'] = HyPER_best_top1_prob
    results['HyPER_best_top2_prob'] = HyPER_best_top2_prob
    results['HyPER_best_w1_prob'] = HyPER_best_w1_prob
    results['HyPER_best_w2_prob'] = HyPER_best_w2_prob

    columns_to_return = [
        'thad_first',
        'HyPER_best_top1',
        'HyPER_best_top2',
        'HyPER_best_w1',
        'HyPER_best_w2'
    ]

    # Filter results to include only the selected columns (else pickle file might be too heavy)
    #TO DO: add this as an option in the config file
    light_results = results[columns_to_return]

    return results

    



def ttbar_single_lep_alt(HyPER_outputs: str | pd.DataFrame):
    r"""Reconstruct ttbar events with lepton+jets final states.
        Here we take hyperedge with best (edge + hyperedge)/2 score.
    """
    if   type(HyPER_outputs) is pd.DataFrame:
        results = HyPER_outputs
    elif type(HyPER_outputs) is str:
        results = pd.read_pickle(HyPER_outputs)
    else:
        raise ValueError(f"Unrecognised HyPER output type {type(HyPER_outputs)}, it must be `str` or `pandas.DataFrame`.")
    
    HyPER_best_top1 = []
    HyPER_best_top2 = []
    HyPER_best_w1 = []
    HyPER_best_w2 = []

    HyPER_best_top1_prob = []
    HyPER_best_top2_prob = []
    HyPER_best_w1_prob = []
    HyPER_best_w2_prob = []
    
    thad_first  = []    #to remember if thad or tlep was reconstructed first (i.e. had the best score)


    for i in tqdm(range(len(results)), desc="Reconstructing", unit='event'):
        HE_IDX = results['HyPER_HE_IDX'][i]
        HE_RAW = results['HyPER_HE_RAW'][i]
        GE_IDX = results['HyPER_GE_IDX'][i]
        GE_RAW = results['HyPER_GE_RAW'][i]

        HE_VCT = results['HyPER_HE_VCT'][i]
        GE_VCT = results['HyPER_GE_VCT'][i]

        selected_HE = []    # Selected HyperEdge
        softProb_HE = []    # Soft probability of the selected HyperEdge
        selected_GE = []    # Selected GraphEdge
        softProb_GE = []    # Soft probability of the selected GraphEdge

        raw_scores = []
        he_scores = []
        ge_scores = []
        index_pairs = []
        vct_array = []

        def is_leptonic(vct):
            return len(set(vct).intersection(set([2, 3]))) == 2 

        def is_hadronic(vct):
            return all(v == 1 for v in vct)

        for i in range(len(HE_RAW)):
            he_vct = HE_VCT[i]
            he_idx = HE_IDX[i]
            he_valid = False

            if is_leptonic(he_vct):
                he_valid = 'leptonic'
            elif is_hadronic(he_vct):
                he_valid = 'hadronic'


            if not he_valid: #Do not consider unphysical combinations
                continue 

            for j in range(len(GE_RAW)):
                ge_vct = GE_VCT[j]
                ge_idx = GE_IDX[j]

                # Leptonic top need leptonic edge
                if he_valid == 'leptonic':
                    if not is_leptonic(ge_vct):
                        continue

                    score = HE_RAW[i]

                # Hadronic top => consider the 3 possibles edges
                elif he_valid == 'hadronic':
                    if not set(ge_idx).issubset(set(he_idx)):
                        continue

                    score = (HE_RAW[i] + GE_RAW[j])/2
                
                raw_scores.append(score)
                he_scores.append(HE_RAW[i])
                ge_scores.append(GE_RAW[j])
                index_pairs.append((he_idx, ge_idx))
                vct_array.append(he_vct)


        completed_patterns = 0
        tlep_position = 0 #was leptonic top reconstructed first (0) or not (1)
        rank = np.argsort(raw_scores) #array of sorted indices
        p    = -1           # current position

        # We need 2 top quarks
        while completed_patterns < 2:
            if completed_patterns == 0: 

                #check if best top is tlep (!!! Here we assume there is only one charged lepton per event in the tuples !!!)
                if is_hadronic(vct_array[rank[p]]):
                    tlep_position = 1
            
            if completed_patterns > 0:
                for pattern in selected_HE:
                    #Check that the other reconstructed top does not contain the same nodes and is/isnt leptonic
                    while (len(set(pattern).intersection(set(index_pairs[rank[p]][0]))) != 0) or ((tlep_position == 1) and (len(set(vct_array[rank[p]]).intersection(set([2,3]))) != 2)): 
                        p -= 1

                        

            if he_scores[rank[p]] != 0:
                selected_HE.append(index_pairs[rank[p]][0])
                softProb_HE.append(he_scores[rank[p]])
            else:
                selected_HE.append([-1, -1, -1])
                softProb_HE.append(0)

            if ge_scores[rank[p]] != 0:
                selected_GE.append(index_pairs[rank[p]][1])
                softProb_GE.append(ge_scores[rank[p]])
            else:
                selected_GE.append([-1,-1])
                softProb_GE.append(0)

            p -= 1
            completed_patterns += 1

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

    results['thad_first'] = thad_first

    results['HyPER_best_top1'] = HyPER_best_top1
    results['HyPER_best_top2'] = HyPER_best_top2
    results['HyPER_best_w1'] = HyPER_best_w1
    results['HyPER_best_w2'] = HyPER_best_w2

    results['HyPER_best_top1_prob'] = HyPER_best_top1_prob
    results['HyPER_best_top2_prob'] = HyPER_best_top2_prob
    results['HyPER_best_w1_prob'] = HyPER_best_w1_prob
    results['HyPER_best_w2_prob'] = HyPER_best_w2_prob

    columns_to_return = [
        'thad_first',
        'HyPER_best_top1',
        'HyPER_best_top2',
        'HyPER_best_w1',
        'HyPER_best_w2',
        'HyPER_best_top1_prob',
        'HyPER_best_top2_prob',
        'HyPER_best_w1_prob',
        'HyPER_best_w2_prob',
        'HyPER_CLS_RAW'
    ]

    # Filter results to include only the selected columns (else pickle file might be too heavy)
    #TO DO: add this as an option in the config file
    light_results = results[columns_to_return]

    return light_results

def ttbar_single_lep(HyPER_outputs: str | pd.DataFrame):
    r"""Reconstruct ttbar events with lepton+jets final states.

    """
    if   type(HyPER_outputs) is pd.DataFrame:
        results = HyPER_outputs
    elif type(HyPER_outputs) is str:
        results = pd.read_pickle(HyPER_outputs)
    else:
        raise ValueError(f"Unrecognised HyPER output type {type(HyPER_outputs)}, it must be `str` or `pandas.DataFrame`.")
    
    HyPER_best_top1 = []
    HyPER_best_top2 = []
    HyPER_best_w1 = []
    HyPER_best_w2 = []

    HyPER_best_top1_prob = []
    HyPER_best_top2_prob = []
    HyPER_best_w1_prob = []
    HyPER_best_w2_prob = []
    
    thad_first  = []    #to remember if thad or tlep was reconstructed first (i.e. had the best score)


    for i in tqdm(range(len(results)), desc="Reconstructing", unit='event'):
        HE_IDX = results['HyPER_HE_IDX'][i]
        HE_RAW = results['HyPER_HE_RAW'][i]
        GE_IDX = results['HyPER_GE_IDX'][i]
        GE_RAW = results['HyPER_GE_RAW'][i]

        HE_VCT = results['HyPER_HE_VCT'][i]
        GE_VCT = results['HyPER_GE_VCT'][i]

        selected_HE = []    # Selected HyperEdge
        softProb_HE = []    # Soft probability of the selected HyperEdge
        selected_GE = []    # Selected GraphEdge
        softProb_GE = []    # Soft probability of the selected GraphEdge

        completed_patterns = 0
        tlep_position = 0 #was leptonic top reconstructed first (0) or not (1)
        rank = np.argsort(HE_RAW) #array of sorted indices
        p    = -1           # current position

        # We need 2 top quarks
        while completed_patterns < 2:
            if completed_patterns == 0: 
                #Make sure its not reconstructing MET and lepton in different tops
                while (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) == 1): 
                    p -=1 

                #check if best top is tlep (!!! Here we assume there is only one charged lepton per event in the tuples !!!)
                if (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) != 2):
                    tlep_position = 1
            
            if completed_patterns > 0:
                for pattern in selected_HE:
                    #Check that the other reconstructed top does not contain the same nodes and is/isnt leptonic
                    while (len(set(pattern).intersection(set(HE_IDX[rank[p]]))) != 0) or ((tlep_position == 1) and (len(set(HE_VCT[rank[p]]).intersection(set([2,3]))) != 2)): 
                        p -= 1

            if HE_RAW[rank[p]] != 0:
                selected_HE.append(HE_IDX[rank[p]])
                softProb_HE.append(HE_RAW[rank[p]])
            else:
                selected_HE.append([-1, -1, -1])
                softProb_HE.append(0)
            
            # Find best edge in pattern
            best_edge_score_in_pattern = 0
            for possible_edge in list(combinations(HE_IDX[rank[p]],r=2)):
                # this looks complated but it is faster during computing
                edge_in_pattern = np.argwhere(np.sum(np.where(np.array(GE_IDX)==possible_edge[0],1,0) + np.where(np.array(GE_IDX)==possible_edge[1],1,0),axis=1)==2).flatten()[0]

                if GE_RAW[edge_in_pattern] > best_edge_score_in_pattern:
                    if (completed_patterns != tlep_position) or ((completed_patterns == tlep_position) and (len(set(GE_VCT[edge_in_pattern]).intersection(set([2,3]))) == 2)): #Wlep automatically found as constituted of lepton+MET
                        best_edge_score_in_pattern = GE_RAW[edge_in_pattern]
                        best_edge_in_pattern = list(possible_edge)


            if best_edge_score_in_pattern != 0:
                selected_GE.append(best_edge_in_pattern)
                softProb_GE.append(best_edge_score_in_pattern)
            else:
                selected_GE.append([-1,-1])
                softProb_GE.append(0)

            p -= 1
            completed_patterns += 1

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

    results['thad_first'] = thad_first

    results['HyPER_best_top1'] = HyPER_best_top1
    results['HyPER_best_top2'] = HyPER_best_top2
    results['HyPER_best_w1'] = HyPER_best_w1
    results['HyPER_best_w2'] = HyPER_best_w2

    results['HyPER_best_top1_prob'] = HyPER_best_top1_prob
    results['HyPER_best_top2_prob'] = HyPER_best_top2_prob
    results['HyPER_best_w1_prob'] = HyPER_best_w1_prob
    results['HyPER_best_w2_prob'] = HyPER_best_w2_prob

    columns_to_return = [
        'thad_first',
        'HyPER_best_top1',
        'HyPER_best_top2',
        'HyPER_best_w1',
        'HyPER_best_w2',
        'HyPER_best_top1_prob',
        'HyPER_best_top2_prob',
        'HyPER_best_w1_prob',
        'HyPER_best_w2_prob',
        'HyPER_CLS_RAW'
    ]

    # Filter results to include only the selected columns (else pickle file might be too heavy)
    #TO DO: add this as an option in the config file
    light_results = results[columns_to_return]

    return light_results

    

def ttbar_allhad(HyPER_outputs: str | pd.DataFrame):
    r"""Reconstruct ttbar events with all-hadronic final states.

    """
    if   type(HyPER_outputs) is pd.DataFrame:
        results = HyPER_outputs
    elif type(HyPER_outputs) is str:
        results = pd.read_pickle(HyPER_outputs)
    else:
        raise ValueError(f"Unrecognised HyPER output type {type(HyPER_outputs)}, it must be `str` or `pandas.DataFrame`.")
    
    HyPER_best_top1 = []
    HyPER_best_top2 = []
    HyPER_best_w1 = []
    HyPER_best_w2 = []

    for i in tqdm(range(len(results)), desc="Reconstructing", unit='event'):
        HE_IDX = results['HyPER_HE_IDX'][i]
        HE_RAW = results['HyPER_HE_RAW'][i]
        GE_IDX = results['HyPER_GE_IDX'][i]
        GE_RAW = results['HyPER_GE_RAW'][i]

        selected_HE = []    # Selected HyperEdge
        softProb_HE = []    # Soft probability of the selected HyperEdge
        selected_GE = []    # Selected GraphEdge
        softProb_GE = []    # Soft probability of the selected GraphEdge

        completed_patterns = 0
        rank = np.argsort(HE_RAW)
        p    = -1           # current position

        # We need 2 top quarks
        while completed_patterns < 2:
            if completed_patterns == 0:
                pass
            
            if completed_patterns > 0:
                for pattern in selected_HE:
                    while len(set(pattern).intersection(set(HE_IDX[rank[p]]))) != 0:
                        p -= 1

            selected_HE.append(HE_IDX[rank[p]])
            softProb_HE.append(HE_RAW[rank[p]])
            
            # Find best edge in pattern
            best_edge_score_in_pattern = 0
            for possible_edge in list(combinations(HE_IDX[rank[p]],r=2)):
                # this looks complated but it is faster during computing
                edge_in_pattern = np.argwhere(np.sum(np.where(np.array(GE_IDX)==possible_edge[0],1,0) + np.where(np.array(GE_IDX)==possible_edge[1],1,0),axis=1)==2).flatten()[0]

                if GE_RAW[edge_in_pattern] > best_edge_score_in_pattern:
                    best_edge_score_in_pattern = GE_RAW[edge_in_pattern]
                    best_edge_in_pattern = list(possible_edge)

            selected_GE.append(best_edge_in_pattern)
            softProb_GE.append(best_edge_score_in_pattern)

            p -= 1
            completed_patterns += 1

        HyPER_best_top1.append(selected_HE[0])
        HyPER_best_top2.append(selected_HE[1])
        HyPER_best_w1.append(selected_GE[0])
        HyPER_best_w2.append(selected_GE[1])

    results['HyPER_best_top1'] = HyPER_best_top1
    results['HyPER_best_top2'] = HyPER_best_top2
    results['HyPER_best_w1'] = HyPER_best_w1
    results['HyPER_best_w2'] = HyPER_best_w2

    columns_to_return = [
        'HyPER_best_top1',
        'HyPER_best_top2',
        'HyPER_best_w1',
        'HyPER_best_w2',
    ]

    # Filter results to include only the selected columns (else pickle file might be too heavy)
    #TO DO: add this as an option in the config file
    light_results = results[columns_to_return]

    return light_results

    # return results