"""
constants.py
Global constants, labels, color palettes, and Phase 1 reference numbers.
"""

# Label definitions
LABELS = ["SUPPORT", "CONTRADICT", "NO INFO"]
NO_INFO = "NO INFO"

# Color palette for plots
PALETTE = {
    "baseline": "#8C8C8C",
    "p2": "#1F77B4",
    "alt": "#FF7F0E",
    "bm25": "#2CA02C",
    "bad": "#D62728",
}

# Phase 1 reference numbers (dev set) used for comparison and regression checks.
PHASE1 = {
    "abstract": {  # 188 dev claims with evidence
        "p1_raw": {"Top-1": 0.516, "Top-3": 0.702, "Top-10": 0.862, "MRR": 0.634},
        "p1_tok": {"Top-1": 0.564, "Top-3": 0.793, "Top-10": 0.883, "MRR": 0.679},
        "p1_lower": {"Top-1": 0.622, "Top-3": 0.809, "Top-10": 0.910, "MRR": 0.726},
        "p1_stem": {"Top-1": 0.644, "Top-3": 0.835, "Top-10": 0.931, "MRR": 0.748},
        "p1_planned": {"Top-1": 0.628, "Top-3": 0.798, "Top-10": 0.915, "MRR": 0.726},
    },
    "sentence": {  # 209 claim-abstract pairs, oracle abstract
        "p1_raw": {"Top-1": 0.493, "Top-3": 0.742, "MRR": 0.648},
        "p1_tok": {"Top-1": 0.450, "Top-3": 0.775, "MRR": 0.632},
        "p1_lower": {"Top-1": 0.455, "Top-3": 0.775, "MRR": 0.632},
        "p1_stem": {"Top-1": 0.450, "Top-3": 0.766, "MRR": 0.629},
        "p1_planned": {"Top-1": 0.498, "Top-3": 0.804, "MRR": 0.670},
    },
    "label": {  # (accuracy, macro-F1) on the 300 dev claims
        "majority": (0.413, 0.195),
        "A_sim": (0.530, 0.491),
        "A_neg": (0.567, 0.512),
        "A_all": (0.613, 0.575),
        "B_full": (0.553, 0.521),
    },
    "funnel": {
        "claims with evidence": 188,
        "abstract ranked first": 117,
        "+ sentence in top-3": 101,
        "+ correct label": 74,
    },
}

# Negation words used in lexical features and recipe filtering
NEG_WORDS = frozenset({
    "no", "not", "nor", "n't", "never", "neither", "none", "cannot",
    "without", "nothing", "nobody",
})