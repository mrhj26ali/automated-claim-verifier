"""
labeling/pipeline.py
Pipeline orchestration: retrieval -> sentence ranking -> feature extraction -> classification.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import DEFAULT_CFG
from ..constants import LABELS, NO_INFO
from ..preprocessing.recipes import unit_terms
from ..retrieval.index import RetrievalIndex
from ..retrieval.metrics import eval_abstracts, eval_sentences, rank_of
from .features import columns_for, lexical_features, neg_dep_features, svo_features, ent_features, dir_dep_features


def make_model(C: float = 1.0, seed: int = 0):
    """Create a logistic regression pipeline with standardization."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=seed))


@dataclass
class Prepared:
    """Result of running the pipeline on a set of claims."""
    claims: List[dict]
    S: np.ndarray              # abstract score matrix (n_claims x n_docs)
    top1: Dict[int, int]       # claim_id -> retrieved abstract id
    abs_ranks: Dict[int, int]  # claim_id -> rank of best gold abstract
    Xg: pd.DataFrame           # features computed on the gold (or cited) abstract
    Xr: pd.DataFrame           # features computed on the retrieved abstract
    y: np.ndarray              # true labels (object array of str)
    sent_info: pd.DataFrame    # per claim: abs_correct, sent_rank, sent_ok (retrieved abstract)


class Context:
    """Corpus + annotations + memoised retrieval indexes and claim terms."""

    def __init__(self, corpus: Dict[int, dict], claim_ann: Optional[Dict] = None,
                 doc_ann: Optional[Dict] = None, has_ann: bool = False):
        self.corpus = corpus
        self.claim_ann = claim_ann or {}
        self.doc_ann = doc_ann or {}
        self.has_ann = bool(has_ann)
        self._indexes: Dict[str, RetrievalIndex] = {}
        self._cterms: Dict = {}

    def index(self, recipe_name: str) -> RetrievalIndex:
        """Get or build a RetrievalIndex for the given recipe."""
        if recipe_name not in self._indexes:
            from ..preprocessing.recipes import build_doc_terms, RECIPES
            r = RECIPES[recipe_name]
            terms = build_doc_terms(r, self.corpus, self.doc_ann)
            self._indexes[recipe_name] = RetrievalIndex(terms)
        return self._indexes[recipe_name]

    def claim_terms(self, recipe_name: str, claims: List[dict]) -> Dict[int, List[str]]:
        """Get or compute claim terms for the given recipe."""
        from ..preprocessing.recipes import RECIPES
        r = RECIPES[recipe_name]
        out = {}
        for c in claims:
            key = (recipe_name, c["id"])
            if key not in self._cterms:
                ann = self.claim_ann.get(c["id"]) if r.source == "stanza" else None
                self._cterms[key] = unit_terms(r, text=c["claim"], ann=ann)
            out[c["id"]] = self._cterms[key]
        return out


class Pipeline:
    """Retrieval -> sentence ranking -> feature extraction pipeline."""

    def __init__(self, ctx: Context, cfg: dict):
        self.ctx = ctx
        self.cfg = dict(DEFAULT_CFG)
        self.cfg.update(cfg)
        self.abs_index = ctx.index(self.cfg["abs_recipe"])
        self.sent_index = ctx.index(self.cfg["sent_recipe"])

    def _sentence_scores(self, claim: dict, doc_id: int) -> np.ndarray:
        """Score every sentence of an abstract against a claim."""
        cfg = self.cfg
        q = self.ctx.claim_terms(cfg["sent_recipe"], [claim])[claim["id"]]
        return self.sent_index.score_sentences(q, doc_id, cfg["sent_scorer"], cfg["sent_k1"], cfg["sent_b"])

    def top_sentences(self, claim: dict, doc_id: int, n: Optional[int] = None) -> List[int]:
        """Get the top-n sentence indices for a claim-abstract pair."""
        sc = self._sentence_scores(claim, doc_id)
        n = n or self.cfg["n_evidence"]
        return [int(j) for j in np.argsort(-sc, kind="stable")[:n]]

    def features(self, claims: List[dict], doc_for_claim: Dict[int, int], S: np.ndarray) -> pd.DataFrame:
        """Extract features for each claim given its assigned abstract."""
        cfg, ctx = self.cfg, self.ctx
        qs = ctx.claim_terms(cfg["sent_recipe"], claims)
        rows = []

        for i, c in enumerate(claims):
            doc_id = doc_for_claim[c["id"]]
            sc = self._sentence_scores(c, doc_id)
            order = np.argsort(-sc, kind="stable")[:cfg["n_evidence"]] if len(sc) else np.array([], dtype=int)
            ev_idx = [int(j) for j in order]
            top = sc[ev_idx] if ev_idx else np.array([0.0])
            srt = np.sort(sc)[::-1] if len(sc) else np.array([0.0])
            col = self.abs_index.doc_pos.get(doc_id)

            row = {"claim_id": c["id"], "doc_id": doc_id}
            row["sim_top1"] = float(top[0])
            row["sim_mean"] = float(top.mean())
            row["sim_gap"] = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])
            row["abs_score_norm"] = float(S[i, col] / (S[i].max() + 1e-12)) if col is not None else 0.0

            q_uni = {t for t in qs[c["id"]] if " " not in t and "::" not in t}
            e_uni = {t for j in ev_idx for t in self.sent_index.sent_terms[doc_id][j]
                     if " " not in t and "::" not in t}
            row["overlap_frac"] = len(q_uni & e_uni) / len(q_uni) if q_uni else 0.0

            ev_texts = [ctx.corpus[doc_id]["sents"][j] for j in ev_idx]
            row.update(lexical_features(c["claim"], ev_texts))

            if ctx.has_ann:
                c_ann = ctx.claim_ann[c["id"]]
                c_anns = [c_ann]
                e_anns = [ctx.doc_ann[doc_id][j] for j in ev_idx]
                row.update(neg_dep_features(c_anns, e_anns))
                row.update(svo_features(c_anns, e_anns))
                row.update(ent_features(c_ann, e_anns))
                row.update(dir_dep_features(c_anns, e_anns))

            rows.append(row)

        return pd.DataFrame(rows).set_index("claim_id")

    @staticmethod
    def _gold_doc(c: dict, top1: Dict[int, int]) -> int:
        """Get the gold evidence document, or the first cited document, or the retrieved one."""
        if c.get("evidence"):
            return list(c["evidence"].keys())[0]
        if c.get("cited_doc_ids"):
            return c["cited_doc_ids"][0]
        return top1[c["id"]]

    def prepare(self, claims: List[dict]) -> Prepared:
        """Run the full pipeline on a list of claims."""
        cfg, ctx = self.cfg, self.ctx
        q_abs = ctx.claim_terms(cfg["abs_recipe"], claims)
        S, abs_ranks = eval_abstracts(self.abs_index, claims, q_abs,
                                      cfg["abs_scorer"], cfg["abs_k1"], cfg["abs_b"])

        top_cols = S.argmax(axis=1)
        top1 = {c["id"]: self.abs_index.doc_ids[int(j)] for c, j in zip(claims, top_cols)}
        gold = {c["id"]: self._gold_doc(c, top1) for c in claims}

        Xg = self.features(claims, gold, S)
        Xr = self.features(claims, top1, S)

        info = []
        for c in claims:
            d = top1[c["id"]]
            abs_ok = d in c.get("evidence", {})
            rank, ok = np.nan, False
            if abs_ok:
                sc = self._sentence_scores(c, d)
                g = [i for i in c["evidence"][d] if i < len(sc)]
                if g:
                    cols = [min(g)] if cfg["sent_target"] == "first" else g
                    rank = rank_of(sc, cols)
                    ok = rank <= 3
            info.append({"claim_id": c["id"], "abs_correct": abs_ok, "sent_rank": rank, "sent_ok": ok})

        sent_info = pd.DataFrame(info).set_index("claim_id")
        y = np.array([c.get("label", NO_INFO) for c in claims], dtype=object)

        return Prepared(list(claims), S, top1, abs_ranks, Xg, Xr, y, sent_info)


def cv_predict(prep: Prepared, groups: List[str], C: float = 1.0,
               n_splits: int = 5, seed: int = 0) -> tuple:
    """Stratified CV on one Prepared set. Returns (predictions A, predictions B)."""
    cols = columns_for(groups)
    y = prep.y.astype(str)
    Xg, Xr = prep.Xg[cols].values, prep.Xr[cols].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pa = np.empty(len(y), dtype=object)
    pb = np.empty(len(y), dtype=object)

    for tr, te in skf.split(np.zeros(len(y)), y):
        model = make_model(C, seed).fit(Xg[tr], y[tr])
        pa[te] = model.predict(Xg[te])
        pb[te] = model.predict(Xr[te])

    return pa.astype(str), pb.astype(str)


def predict_eval(prep_train: Prepared, prep_eval: Optional[Prepared],
                 groups: List[str], C: float = 1.0,
                 n_splits: int = 5, seed: int = 0) -> tuple:
    """Train on prep_train, predict prep_eval. If prep_eval is None, use CV on prep_train."""
    if prep_eval is None:
        return cv_predict(prep_train, groups, C, n_splits, seed)

    cols = columns_for(groups)
    model = make_model(C, seed).fit(prep_train.Xg[cols].values, prep_train.y.astype(str))
    pa = model.predict(prep_eval.Xg[cols].values)
    pb = model.predict(prep_eval.Xr[cols].values)
    return np.asarray(pa).astype(str), np.asarray(pb).astype(str)