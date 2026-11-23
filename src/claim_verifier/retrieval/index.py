"""
retrieval/index.py
Sparse retrieval index supporting TF-IDF and BM25 scoring.

The index is built over abstract-level term lists. Sentence scoring reuses the
same vocabulary and idf weights, with BM25 sentence-length normalisation
computed from the mean sentence length of the corpus.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def _identity(x):
    """Pass-through analyzer: terms are already tokenised."""
    return x


class RetrievalIndex:
    """TF-IDF / BM25 scoring over abstracts (documents) and their sentences.

    Parameters
    ----------
    doc_terms : dict
        Mapping from doc_id to a list of term lists, one entry per abstract
        sentence. The abstract-level term list is the concatenation of all
        sentence term lists.
    """

    def __init__(self, doc_terms: Dict[int, List[List[str]]]) -> None:
        self.doc_ids: List[int] = sorted(doc_terms)
        self.doc_pos: Dict[int, int] = {d: i for i, d in enumerate(self.doc_ids)}
        self.sent_terms: Dict[int, List[List[str]]] = {d: doc_terms[d] for d in self.doc_ids}

        # Build the abstract-level term lists (title + sentences concatenated).
        abstract_terms = [
            [t for s in self.sent_terms[d] for t in s]
            for d in self.doc_ids
        ]

        # Vocabulary and document frequencies are computed on the abstracts.
        self.vec = CountVectorizer(analyzer=_identity)
        self.X = self.vec.fit_transform(abstract_terms).tocsr().astype(np.float64)

        self.tfidf = TfidfTransformer()
        self.Xt = self.tfidf.fit_transform(self.X)

        self.N = len(self.doc_ids)
        self.df = np.asarray((self.X > 0).sum(axis=0)).ravel()
        self.doc_len = np.asarray(self.X.sum(axis=1)).ravel()
        self.avgdl = float(self.doc_len.mean()) if self.N else 1.0

        # Mean sentence length for BM25 sentence-length normalisation.
        sent_lens = [len(s) for d in self.doc_ids for s in self.sent_terms[d]]
        self.avg_sent_len = float(np.mean(sent_lens)) if sent_lens else 1.0

        # BM25 idf: log(1 + (N - df + 0.5) / (df + 0.5))
        self.idf_bm25 = np.log(1.0 + (self.N - self.df + 0.5) / (self.df + 0.5))

        self._bm25_cache: Dict[tuple, sparse.csr_matrix] = {}

    # ------------------------------------------------------------------ abstracts

    def _bm25_matrix(self, k1: float, b: float) -> sparse.csr_matrix:
        """Return the BM25-weighted document-term matrix, cached by (k1, b)."""
        key = (round(float(k1), 6), round(float(b), 6))
        if key not in self._bm25_cache:
            if len(self._bm25_cache) >= 8:
                self._bm25_cache.clear()
            X = self.X
            rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
            tf = X.data
            norm = k1 * (1.0 - b + b * self.doc_len[rows] / self.avgdl)
            data = self.idf_bm25[X.indices] * tf * (k1 + 1.0) / (tf + norm)
            self._bm25_cache[key] = sparse.csr_matrix(
                (data, X.indices.copy(), X.indptr.copy()), shape=X.shape
            )
        return self._bm25_cache[key]

    def score_abstracts(
        self,
        q_terms_list: Sequence[List[str]],
        scorer: str = "tfidf",
        k1: float = 1.2,
        b: float = 0.75,
    ) -> np.ndarray:
        """Score every query against every abstract. Returns a dense matrix
        of shape (n_queries, n_docs)."""
        Q = self.vec.transform(q_terms_list).astype(np.float64)
        if scorer == "tfidf":
            Qt = self.tfidf.transform(Q)
            return (Qt @ self.Xt.T).toarray()
        if scorer == "bm25":
            W = self._bm25_matrix(k1, b)
            return (Q @ W.T).toarray()
        raise ValueError(f"Unknown scorer '{scorer}'")

    # ------------------------------------------------------------------ sentences

    def score_sentences(
        self,
        q_terms: List[str],
        doc_id: int,
        scorer: str = "tfidf",
        k1: float = 1.2,
        b: float = 0.75,
    ) -> np.ndarray:
        """Score every sentence of one abstract against a single query.
        Returns a 1-D array of length n_sentences."""
        sents = self.sent_terms[doc_id]
        if not sents:
            return np.zeros(0)

        S = self.vec.transform(sents).astype(np.float64).tocsr()
        q = self.vec.transform([q_terms]).astype(np.float64)

        if scorer == "tfidf":
            St = self.tfidf.transform(S)
            qt = self.tfidf.transform(q)
            return (St @ qt.T).toarray().ravel()

        if scorer == "bm25":
            lens = np.asarray(S.sum(axis=1)).ravel()
            rows = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
            tf = S.data
            norm = k1 * (1.0 - b + b * lens[rows] / self.avg_sent_len)
            data = self.idf_bm25[S.indices] * tf * (k1 + 1.0) / (tf + norm)
            W = sparse.csr_matrix(
                (data, S.indices.copy(), S.indptr.copy()), shape=S.shape
            )
            return (W @ q.T).toarray().ravel()

        raise ValueError(f"Unknown scorer '{scorer}'")