"""
evaluation/statistics.py
Bootstrap confidence intervals, ladder comparison, recipe selection.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..constants import LABELS


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean of values."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                     seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Bootstrap CI of mean(a) - mean(b) for paired per-item values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired_bootstrap needs arrays of equal length")
    return bootstrap_ci(a - b, n_boot, seed, alpha)


def bootstrap_stat(fn, n: int, n_boot: int = 1000, seed: int = 0,
                   alpha: float = 0.05) -> Tuple[float, float, float]:
    """CI for an arbitrary statistic fn(index_array) -> float."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        vals[i] = fn(rng.integers(0, n, n))
    full = float(fn(np.arange(n)))
    return full, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_bootstrap_stat(fn_a, fn_b, n: int, n_boot: int = 1000,
                          seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """CI for fn_a - fn_b evaluated on the same resampled indices."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = fn_a(idx) - fn_b(idx)
    full_idx = np.arange(n)
    point = float(fn_a(full_idx) - fn_b(full_idx))
    return point, float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def compare_ladder(rank_dicts: Dict[str, Dict[int, int]], baseline: str,
                   ks: Tuple[int, ...] = (1, 3, 10), n_boot: int = 2000,
                   seed: int = 0, label: str = "recipe") -> pd.DataFrame:
    """Compare multiple configurations against a baseline using paired bootstrap on MRR."""
    keys = sorted(set.intersection(*[set(d) for d in rank_dicts.values()]))
    if not keys:
        raise ValueError("No evaluation items are shared by all configurations")

    base_rr = 1.0 / np.array([rank_dicts[baseline][k] for k in keys], dtype=float)
    rows = []

    for name, d in rank_dicts.items():
        r = np.array([d[k] for k in keys], dtype=float)
        rr = 1.0 / r
        m, lo, hi = bootstrap_ci(rr, n_boot, seed)

        row = {label: name, "n": len(keys)}
        for k in ks:
            row[f"Top-{k}"] = float((r <= k).mean())
        row.update({"MRR": m, "MRR_lo": lo, "MRR_hi": hi})

        if name == baseline:
            row.update({"dMRR": 0.0, "dMRR_lo": 0.0, "dMRR_hi": 0.0, "verdict": "baseline"})
        else:
            dm, dlo, dhi = paired_bootstrap(rr, base_rr, n_boot, seed)
            verdict = "keep" if dlo > 0 else ("worse" if dhi < 0 else "no effect")
            row.update({"dMRR": dm, "dMRR_lo": dlo, "dMRR_hi": dhi, "verdict": verdict})

        rows.append(row)

    return pd.DataFrame(rows)


def format_ladder(df: pd.DataFrame, ks: Tuple[int, ...] = (1, 3, 10)) -> pd.DataFrame:
    """Format a ladder DataFrame for display."""
    label = df.columns[0]
    out = pd.DataFrame({label: df[label], "n": df["n"]})

    for k in ks:
        if f"Top-{k}" in df:
            out[f"Top-{k}"] = df[f"Top-{k}"].map(lambda v: f"{100 * v:.1f}%")

    out["MRR [95% CI]"] = [f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
                           for m, lo, hi in zip(df["MRR"], df["MRR_lo"], df["MRR_hi"])]
    out["dMRR vs baseline [95% CI]"] = [f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                                        for m, lo, hi in zip(df["dMRR"], df["dMRR_lo"], df["dMRR_hi"])]
    out["verdict"] = df["verdict"]

    return out.set_index(label)


def select_recipe(df: pd.DataFrame, baseline: str) -> str:
    """Select the highest-MRR configuration whose paired CI excludes zero."""
    label = df.columns[0]
    keep = df[df["verdict"] == "keep"]
    if keep.empty:
        return baseline
    return str(keep.sort_values("MRR", ascending=False).iloc[0][label])


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute accuracy and per-class F1 scores."""
    from sklearn.metrics import accuracy_score, f1_score
    f1s = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    out = {"acc": float(accuracy_score(y_true, y_pred)), "macro_f1": float(np.mean(f1s))}
    for lab, f in zip(LABELS, f1s):
        out[f"f1_{lab}"] = float(f)
    return out