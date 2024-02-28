import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from collections import defaultdict


from .feature_extractor import FeatureExtractor
from .distance import compute_distance_matrix
from .re_ranking import re_ranking


def retrieve_candidates(
    query_features,
    gallery_features,
    query_IDs,
    gallery_IDs,
    max_rank=1,
    return_distmat=False,
    reduce_avg=False,
):

    num_gallery = len(gallery_IDs)

    max_rank = min(max_rank, num_gallery)

    distmat = compute_distance_matrix(query_features, gallery_features).numpy()

    indices = np.argsort(distmat, axis=1)
    
    candidates = {query_IDs[i]: gallery_IDs[candidates_rank] for i, candidates_rank in enumerate(indices[:, :max_rank])}


    if return_distmat:
        sorted_candidates_IDs = gallery_IDs[indices[:, :max_rank]]
        sorted_distmat = np.take_along_axis(distmat, indices[:, :max_rank], axis=1)

        if reduce_avg:
            sorted_distmat, sorted_candidates_IDs = reduce_distmat_avg(sorted_distmat, sorted_candidates_IDs)
        
        return sorted_candidates_IDs, sorted_distmat
    else:
        return candidates


def reduce_distmat_avg(distmat, candmat):
    num_q, num_g = distmat.shape[0], distmat.shape[1]
    
    reduce_distmat = np.empty((num_q, num_g))
    reduce_candmat = np.empty((num_q, num_g), dtype=int)
    
    for i in range(num_q):
        distances, candidate_IDs = distmat[i], candmat[i]
        avg = pd.Series(distances, index=candidate_IDs).groupby(level=0).mean()
        avg.sort_values(inplace=True)
        
        padded_dists = np.pad(avg.values, (0, num_g - avg.values.size), 'constant', constant_values=(np.inf, np.inf))
        padded_cands = np.pad(avg.index.values, (0, num_g - avg.index.values.size), 'constant')
        
        reduce_distmat[i] = padded_dists
        reduce_candmat[i] = padded_cands

    return reduce_distmat, reduce_candmat


def get_best_candidates(distmat, candmat, query_IDs):
    best_candidates = {}
    used_querys = set()
    used_candidates = set()
    
    num_querys = candmat.shape[0]
    num_cands = candmat.shape[1]
    
    for col in range(num_cands):
        repeated = defaultdict(list)
        for i, cand in enumerate(candmat[:, col]):
            repeated[cand].append(i)
        
        for cand, querys in repeated.items():
            if cand in used_candidates:
                continue
            
            dists = np.copy(distmat[:, col])
            
            # only get the current query and candidate
            dist_mask = np.ones(num_querys, dtype=bool)
            dist_mask[querys] = False

            # remove the distance for used 
            dist_mask[list(used_querys)] = True
            dists[dist_mask] = np.inf
            
            best = np.argmin(dists)
            if (dists[best] != np.inf):
                best_candidates[query_IDs[best]] = (cand, dists[best])

                used_querys.add(best)
                used_candidates.add(cand)
    
    return best_candidates

