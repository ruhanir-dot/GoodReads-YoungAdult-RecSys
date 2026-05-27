"""
Candidate-generation utilities that we share for each by recommender.

Problem this aims to solve: 
Don't want to have one teammate to forget to mask out items the
user has already interacted with in training. To standardize we have
`filter_seen` method for this step. 
`popularity_fallback` will cover cold users. 
`merge_topk` unions candidate lists from multiple recall channels for the hybrid notebook
"""

# imports
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .data import Bundle


def filter_seen( item_ids: Sequence[int] | np.ndarray,
                 seen: Iterable[int] | np.ndarray | set, 
                 k: Optional[int] = None,
) -> np.ndarray:
    """
    Drop item ids that the user has already interacted with.
    Parameters
    ----------
    item_ids : ranked candidate items with the best first
    seen : item ids to exclude, items the user has already seen
    k : if this arguement is given we reduce to top-k candidates after filtering.
    """
    seen_set = seen if isinstance(seen, set) else set(np.asarray(list(seen)).tolist()) # converting seen to a set
    arr = np.asarray(item_ids)
    
    # vectorized filtering
    mask = np.isin(arr, list(seen_set), invert=True)
    out = arr[mask]

    if k is not None:
        out = out[:k]
    return out


_POP_CACHE: dict[int, np.ndarray] = {} # global cache! stiring precomputed popularity ranking

def popularity_fallback( bundle: Bundle, 
                        k: int = 20, 
                        exclude: Optional[Iterable[int] | set] = None, 
                        by: str = "n_positive",
) -> np.ndarray:
    """
    Top-k item ids by popularity,  fallback for old users with no train history
    `by` can be 'n_positive', 'n_interactions', or 'avg_rating'.
    """
    # id(bundle) is the memory adress of bundle object
    cache_key = hash((id(bundle), by)) #combine with by paramater which tells us what popularity metric to use
    
    if cache_key not in _POP_CACHE:
        pop = bundle.popularity.sort_values(by, ascending=False) # sort by metric highest first
        _POP_CACHE[cache_key] = pop["item_id"].to_numpy() # store the item ids as array
    
    ranked = _POP_CACHE[cache_key] # get cached popular items
    if exclude is not None:
        return filter_seen(ranked, exclude, k=k) # filter out the seen items and take the top k
    return ranked[:k] # return top k


def merge_topk(
        score_dicts: List[Dict[int, float]] | List[Tuple[Dict[int, float], float]],
        k: int = 20,
        normalize: bool = True,
) -> List[Tuple[int, float]]:
    """
    Goal is that we want to combine reccomendations from multiple sources into a singe ranked list
    Can accept score dictionaries with weight defaults to 1, custom weights and mixed formats

    Each entry can be either `{iid: score}` (weight=1) or
    `({iid: score}, weight)`. If `normalize=True`, each channel is min-max scaled to [0, 1] before weighting so scales are comparable
    ex. ALS (approx. 10) and cosine (approx. 1) 

    Returns a list of (iid, merged_score) sorted desc, length <= k.
    """
    merged: Dict[int, float] = {} # initialized the merged cores dictionary
    for entry in score_dicts: # for each entry
        if isinstance(entry, tuple): 
            scores, weight = entry # extract dictionary and weight
        else:
            scores, weight = entry, 1.0 # else give default weight 1
        if not scores:
            continue
        if normalize: # if normalize true, to compare different algorithms
            values = np.array(list(scores.values()), dtype=np.float64)
            # min max scale
            low, high = float(values.min()), float(values.max())
            range = high - low if high > low else 1.0
            for item_id, score in scores.items():
                # add normalized weighted score to the accumulated total for each item
                merged[item_id] = merged.get(item_id, 0.0) + weight * (score - low) / range 
        else:
            for item_id, score in scores.items(): # no normalization version
                merged[item_id] = merged.get(item_id, 0.0) + weight * score
    ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True) # sort return top k
    return ranked[:k]


def topk_dataframe(
        userid_to_ranked: Dict[int, Sequence[int] | List[Tuple[int, float]]],
        k: int = 20,
) -> pd.DataFrame:
    """
    Convert {uid: [iid, ...]} or {uid: [(iid, score), ...]} into the
    standard predictions schema in df (uid, iid, rank, score).
    """
    rows = [] # create empty list that will store each reccomendation as tuple
    
    for user_id, ranked in userid_to_ranked.items():# loop thorugh dict and get ranked items
        for rank, item in enumerate(list(ranked)[:k]): # take first k items cinvert to list slice get index and rank,item
            if isinstance(item, tuple): # handle two diff formats
                item_id, score = item # item id and score in tuple form
            else:
                item_id, score = item, float(k - rank)  # no score provided in integer synthesisze scor eon rank
            rows.append((int(user_id), int(item_id), int(rank), float(score)))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])
