"""
retrieval/metrics.py
Ranking metrics for abstract retrieval and sentence selection.

All rank computations use pessimistic tie-breaking: when multiple candidates
share the best score, the rank is the *last* position among them. This
prevents all-zero score vectors from looking artificially good.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def rank_of(scores: np.ndarray, cols: Sequence[int]) -> int:
    """1-based pessimistic rank of the best-placed column in `cols`.

    Parameters
    ----------
    scores : array-like
        Score vector over all candidates.
    cols : sequence of int
        Column indices of the gold candidates.

    Returns
    -------
    int
        Rank (1 = best). Ties are broken by position (stable sort).
    """
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    return int(pos[list(cols)].min()) + 1


def eval_abstracts(
    index,
    claims: List[dict],
    q_terms: Dict[int, List[str]],
    scorer: str = "tfidf",
    k1: float = 1.2,
    b: float = 0.75,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Score all claims against all abstracts.

    Returns
    -------
    S : np.ndarray
        Score matrix of shape (n_claims, n_docs).
    ranks : dict
        Mapping from claim_id to the rank of the best gold abstract.
        Only claims with at least one gold document in the index are included.
    """
    S = index.score_abstracts([q_terms[c["id"]] for c in claims], scorer, k1, b)
    ranks: Dict[int, int] = {}
    for i, c in enumerate(claims):
        cols = [index.doc_pos[d] for d in c.get("gold_docs", []) if d in index.doc_pos]
        if cols:
            ranks[c["id"]] = rank_of(S[i], cols)
    return S, ranks


def eval_sentences(
    index,
    claims: List[dict],
    q_terms: Dict[int, List[str]],
    scorer: str = "tfidf",
    k1: float = 1.2,
    b: float = 0.75,
    target: str = "first",
) -> Dict[Tuple[int, int], int]:
    """Oracle-abstract sentence ranking.

    For each (claim, evidence abstract) pair, score every sentence of the
    abstract against the claim and return the rank of the best gold sentence.

    Parameters
    ----------
    target : {"first", "any"}
        "first": the lowest-index gold sentence is the single target.
        "any"  : the best-ranked gold sentence counts.

    Returns
    -------
    dict
        Mapping from (claim_id, doc_id) to rank.
    """
    ranks: Dict[Tuple[int, int], int] = {}
    for c in claims:
        qt = q_terms[c["id"]]
        for doc_id, gold in c.get("evidence", {}).items():
            doc_id = int(doc_id)
            sents = index.sent_terms.get(doc_id)
            if not sents:
                continue
            g = [i for i in gold if i < len(sents)]
            if not g:
                continue
            sc = index.score_sentences(qt, doc_id, scorer, k1, b)
            cols = [min(g)] if target == "first" else g
            ranks[(c["id"], doc_id)] = rank_of(sc, cols)
    return ranks


def summarize_ranks(
    ranks: Dict,
    ks: Tuple[int, ...] = (1, 3, 10),
) -> Dict[str, float]:
    """Compute Top-k, MRR and median rank from a dict of ranks.

    Returns
    -------
    dict
        Keys: "n", "Top-1", "Top-3", ..., "MRR", "median rank".
        Values are floats; Top-k and MRR are NaN when there are no ranks.
    """
    r = np.asarray(list(ranks.values()), dtype=float)
    out: Dict[str, float] = {"n": int(len(r))}
    for k in ks:
        out[f"Top-{k}"] = float((r <= k).mean()) if len(r) else float("nan")
    out["MRR"] = float((1.0 / r).mean()) if len(r) else float("nan")
    out["median rank"] = float(np.median(r)) if len(r) else float("nan")
    return out