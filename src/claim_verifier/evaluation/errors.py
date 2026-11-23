"""
evaluation/errors.py
Error taxonomy and funnel analysis.
"""
from typing import Dict
import pandas as pd
import numpy as np
from ..constants import NO_INFO


def error_taxonomy(prep, pred: np.ndarray) -> pd.DataFrame:
    """Classify each prediction by the first stage that failed."""
    info = prep.sent_info
    rows = []

    for c, p in zip(prep.claims, pred):
        true = c.get("label", NO_INFO)
        abs_ok = bool(info.at[c["id"], "abs_correct"])
        sent_ok = bool(info.at[c["id"], "sent_ok"])

        if true == NO_INFO:
            cat = "Correct label" if p == NO_INFO else "Verdict given on NO INFO"
        elif p == true:
            cat = "Correct label"
        elif not abs_ok:
            cat = "Retrieval error"
        elif not sent_ok:
            cat = "Sentence error"
        else:
            cat = "Interpretation error"

        rows.append({
            "claim_id": c["id"], "label": true, "pred": p, "category": cat,
            "abs_correct": abs_ok, "sent_ok": sent_ok
        })

    return pd.DataFrame(rows)


def funnel_counts(tax: pd.DataFrame) -> Dict[str, int]:
    """Count how many claims survive each stage of the pipeline."""
    ev = tax[tax["label"] != NO_INFO]
    a = ev["abs_correct"].astype(bool)
    s = a & ev["sent_ok"].astype(bool)
    lab = s & (ev["pred"] == ev["label"])

    return {
        "claims with evidence": int(len(ev)),
        "abstract ranked first": int(a.sum()),
        "+ sentence in top-3": int(s.sum()),
        "+ correct label": int(lab.sum()),
    }