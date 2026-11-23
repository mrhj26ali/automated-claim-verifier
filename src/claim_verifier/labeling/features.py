"""
labeling/features.py
Feature extraction for the label classifier.

Phase 1 features: lexical (similarity, negation, direction)
Phase 2 features: dependency-based (neg_dep, svo, ent, dir_dep)
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from nltk.stem import PorterStemmer

from ..constants import NEG_WORDS
from ..preprocessing.tokens import nltk_tokenize, stem

# Feature group definitions
FEATURE_GROUPS = {
    "sim": ["sim_top1", "sim_mean", "sim_gap", "abs_score_norm", "overlap_frac"],
    "neg_lex": ["lx_neg_claim", "lx_neg_ev", "lx_neg_xor", "lx_neg_both", "lx_imp_claim", "lx_imp_ev"],
    "dir_lex": ["lx_dir_claim", "lx_dir_ev", "lx_dir_conflict", "lx_dir_agree"],
    "neg_dep": ["nd_neg_claim", "nd_neg_ev", "nd_head_mismatch", "nd_head_shared", "nd_xor"],
    "svo": ["sv_verb_match", "sv_subj_match", "sv_obj_match", "sv_full_match",
            "sv_neg_conflict", "sv_role_swap", "sv_n_claim"],
    "ent": ["en_shared", "en_tok_cov", "en_claim"],
    "dir_dep": ["dd_claim", "dd_ev", "dd_conflict_any", "dd_conflict_linked", "dd_agree_linked"],
}

SYNTACTIC_GROUPS = ("neg_dep", "svo", "ent", "dir_dep")
PHASE1_GROUPS = ["sim", "neg_lex", "dir_lex"]

# Direction words
_UP = ("increase increased increases increasing rise rises rose risen rising elevate elevated "
       "elevates enhance enhanced enhances raise raised raises high higher greater more promote "
       "promotes promoted induce induces induced stimulate stimulates stimulated activate "
       "activates activated upregulate upregulated upregulation improve improved improves boost "
       "boosted expand expanded accelerate accelerated amplify amplified gain gains grow growth "
       "overexpress overexpressed overexpression").split()

_DOWN = ("decrease decreased decreases decreasing reduce reduced reduces reducing reduction lower "
         "lowered lowers low decline declined declines impair impaired impairs block blocked blocks "
         "diminish diminished less fewer few downregulate downregulated downregulation deplete "
         "depleted depletion prevent prevents prevented attenuate attenuated abolish abolished loss "
         "lost delay delayed slow slowed limit limited").split()

DIR_LEX = {w: 1 for w in _UP}
DIR_LEX.update({w: -1 for w in _DOWN})

DIR_STEM = {}
for _w, _d in DIR_LEX.items():
    DIR_STEM.setdefault(stem(_w), _d)

# Implicit negation words
IMPLICIT_NEG = frozenset({"fail", "lack", "absent", "absence", "unable", "insufficient"})
IMPLICIT_STEMS = frozenset(stem(w) for w in IMPLICIT_NEG)

# Generic words (not useful as direction targets)
GENERIC = frozenset({"level", "rate", "effect", "risk", "amount", "number", "change", "patient",
                     "study", "group", "use", "case", "increase", "decrease", "be", "have", "do"})

CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ"}


def columns_for(groups: List[str]) -> List[str]:
    """Get feature column names for a list of feature groups."""
    cols = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group '{g}'. Known: {list(FEATURE_GROUPS)}")
        cols += FEATURE_GROUPS[g]
    return cols


def _clip(x: float, lim: float = 2) -> float:
    return float(max(-lim, min(lim, x)))


def direction_of(text: str, lemma: Optional[str] = None) -> int:
    """+1 / -1 / 0 for increase-like / decrease-like / neither."""
    low = str(text).lower()
    lem = str(lemma or "").lower()
    if lem in DIR_LEX:
        return DIR_LEX[lem]
    if low in DIR_LEX:
        return DIR_LEX[low]
    return DIR_STEM.get(stem(low), 0)


# ============================================================================
# Lexical features (Phase 1 style)
# ============================================================================

def lexical_features(claim_text: str, ev_texts: List[str]) -> Dict[str, float]:
    """Lexical features: negation cues, implicit negation, direction words."""
    ct = [t.lower() for t in nltk_tokenize(claim_text)]
    et = [t.lower() for s in ev_texts for t in nltk_tokenize(s)]

    def neg(toks):
        return sum(1 for t in toks if t in NEG_WORDS)

    def imp(toks):
        return sum(1 for t in toks if stem(t) in IMPLICIT_STEMS)

    def dsum(toks):
        return sum(direction_of(t) for t in toks)

    nc, ne = min(neg(ct), 3), min(neg(et), 3)
    dc, de = _clip(dsum(ct)), _clip(dsum(et))

    return {
        "lx_neg_claim": float(nc), "lx_neg_ev": float(ne),
        "lx_neg_xor": float((nc > 0) != (ne > 0)), "lx_neg_both": float(nc > 0 and ne > 0),
        "lx_imp_claim": float(min(imp(ct), 3)), "lx_imp_ev": float(min(imp(et), 3)),
        "lx_dir_claim": dc, "lx_dir_ev": de,
        "lx_dir_conflict": float(dc * de < 0), "lx_dir_agree": float(dc * de > 0),
    }


# ============================================================================
# Dependency helpers (Phase 2)
# ============================================================================

def _lem(sent: List, idx: Optional[int]) -> str:
    """Get lemma from a stored word [text, lemma, upos, xpos, head, deprel, feats]."""
    if idx is None:
        return ""
    w = sent[idx]
    return str(w[1] or w[0]).lower()


def all_sents(anns: List[Dict]) -> List[List]:
    """Flatten all sentences from a list of annotations."""
    return [s for a in anns for s in a.get("sents", [])]


def children_map(sent: List) -> Dict[int, List[int]]:
    """Map head index -> list of child indices (1-based head, 0 = root)."""
    ch = {}
    for i, w in enumerate(sent):
        h = int(w[4]) - 1  # head is 1-based
        if h >= 0:
            ch.setdefault(h, []).append(i)
    return ch


def content_lemmas(anns: List[Dict]) -> set:
    """All content word lemmas (NOUN, PROPN, VERB, ADJ) from annotations."""
    return {_lem(s, i) for s in all_sents(anns) for i, w in enumerate(s) if w[2] in CONTENT_UPOS}


def verb_lemmas(anns: List[Dict]) -> set:
    """All verb lemmas from annotations."""
    return {_lem(s, i) for s in all_sents(anns) for i, w in enumerate(s) if w[2] == "VERB"}


def is_negation_word(w: List) -> bool:
    """Check if a stored word is a negation cue."""
    text = str(w[0]).lower()
    lem = str(w[1] or "").lower()
    feats = w[6] or ""
    return text in NEG_WORDS or lem in NEG_WORDS or "Polarity=Neg" in feats


def negated_heads(anns: List[Dict]) -> Tuple[set, int]:
    """(set of lemmas that carry a negation, number of negation cues)."""
    heads, count = set(), 0
    for sent in all_sents(anns):
        for i, w in enumerate(sent):
            if is_negation_word(w):
                count += 1
                h = int(w[4]) - 1
                if h >= 0:
                    heads.add(_lem(sent, h))
    return heads, count


def neg_dep_features(c_anns: List[Dict], e_anns: List[Dict]) -> Dict[str, float]:
    """Dependency-based negation features."""
    hc, nc = negated_heads(c_anns)
    he, ne = negated_heads(e_anns)
    lc, le = content_lemmas(c_anns), content_lemmas(e_anns)
    mismatch = len((hc & le) - he) + len((he & lc) - hc)

    return {
        "nd_neg_claim": float(min(nc, 3)), "nd_neg_ev": float(min(ne, 3)),
        "nd_head_mismatch": float(min(mismatch, 3)),
        "nd_head_shared": float(bool(hc & he)),
        "nd_xor": float((nc > 0) != (ne > 0)),
    }


def svo_tuples(anns: List[Dict]) -> List[Tuple[str, str, str, bool]]:
    """Extract (subject, verb, object, negated) tuples. Passives are normalized."""
    out = []
    for sent in all_sents(anns):
        ch = children_map(sent)
        for i, w in enumerate(sent):
            if w[2] != "VERB":
                continue
            subs, objs, neg = [], [], False
            for c in ch.get(i, []):
                dep = sent[c][5]  # deprel
                if dep in ("nsubj", "csubj", "obl:agent"):
                    subs.append(c)
                elif dep in ("obj", "iobj", "nsubj:pass", "csubj:pass"):
                    objs.append(c)
                if is_negation_word(sent[c]):
                    neg = True
            if not subs and not objs:
                continue
            verb = _lem(sent, i)
            for s in (subs or [None]):
                for o in (objs or [None]):
                    out.append((_lem(sent, s), verb, _lem(sent, o), neg))
    return out


def svo_features(c_anns: List[Dict], e_anns: List[Dict]) -> Dict[str, float]:
    """SVO alignment features."""
    feats = {k: 0.0 for k in FEATURE_GROUPS["svo"]}
    ct, et = svo_tuples(c_anns), svo_tuples(e_anns)
    feats["sv_n_claim"] = float(min(len(ct), 3))
    if not ct:
        return feats

    e_lemmas = content_lemmas(e_anns)
    e_verbs = {t[1] for t in et} | verb_lemmas(e_anns)
    feats["sv_verb_match"] = sum(t[1] in e_verbs for t in ct) / len(ct)
    feats["sv_subj_match"] = float(any(t[0] and t[0] in e_lemmas for t in ct))
    feats["sv_obj_match"] = float(any(t[2] and t[2] in e_lemmas for t in ct))

    full = conflict = swap = False
    for a in ct:
        for b in et:
            if a[1] == b[1]:
                if (a[0] and a[0] == b[0]) or (a[2] and a[2] == b[2]):
                    full = True
                if a[3] != b[3]:
                    conflict = True
                if a[0] and a[2] and a[0] == b[2] and a[2] == b[0]:
                    swap = True

    feats["sv_full_match"] = float(full)
    feats["sv_neg_conflict"] = float(conflict)
    feats["sv_role_swap"] = float(swap)
    return feats


def ent_features(c_ann: Dict, e_anns: List[Dict]) -> Dict[str, float]:
    """Named entity overlap features."""
    ce = {str(e[0]).lower() for e in c_ann.get("ents", [])}
    ee = {str(e[0]).lower() for a in e_anns for e in a.get("ents", [])}
    c_tok = {t for e in ce for t in e.split()}
    e_tok = {t for e in ee for t in e.split()}

    return {
        "en_shared": float(min(len(ce & ee), 3)),
        "en_tok_cov": (len(c_tok & e_tok) / len(c_tok)) if c_tok else 0.0,
        "en_claim": float(min(len(ce), 4)),
    }


def direction_cues(anns: List[Dict]) -> List[Tuple[int, set]]:
    """Extract (signed direction, nearby content lemmas) for every direction cue."""
    cues = []
    for sent in all_sents(anns):
        ch = children_map(sent)
        adj = {}
        for i, w in enumerate(sent):
            h = int(w[4]) - 1
            if h >= 0:
                adj.setdefault(i, set()).add(h)
                adj.setdefault(h, set()).add(i)

        for i, w in enumerate(sent):
            d = direction_of(w[0], w[1])
            if d == 0:
                continue
            flip = any(is_negation_word(sent[c]) for c in ch.get(i, []))
            h = int(w[4]) - 1
            if h >= 0:
                if any(is_negation_word(sent[c]) for c in ch.get(h, []) if c != i):
                    flip = True
                if _lem(sent, h) in IMPLICIT_NEG:
                    flip = True

            seen, frontier = {i}, {i}
            for _ in range(2):
                frontier = {n for f in frontier for n in adj.get(f, set())} - seen
                seen |= frontier
            neigh = {_lem(sent, j) for j in seen if j != i and sent[j][2] in CONTENT_UPOS}
            neigh = {t for t in neigh if t not in GENERIC and t not in DIR_LEX}
            cues.append((-d if flip else d, neigh))
    return cues


def dir_dep_features(c_anns: List[Dict], e_anns: List[Dict]) -> Dict[str, float]:
    """Dependency-based direction features."""
    cc, ec = direction_cues(c_anns), direction_cues(e_anns)
    conflict_any = conflict_linked = agree_linked = 0.0
    for dc, nc in cc:
        for de, ne in ec:
            linked = bool(nc & ne)
            if dc * de < 0:
                conflict_any = 1.0
                if linked:
                    conflict_linked = 1.0
            elif dc * de > 0 and linked:
                agree_linked = 1.0

    return {
        "dd_claim": _clip(sum(d for d, _ in cc)),
        "dd_ev": _clip(sum(d for d, _ in ec)),
        "dd_conflict_any": conflict_any,
        "dd_conflict_linked": conflict_linked,
        "dd_agree_linked": agree_linked,
    }