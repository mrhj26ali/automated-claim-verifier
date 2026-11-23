"""
plotting.py
Visualization helpers for results.
"""
from typing import Dict, List, Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt


def pct(x: float, digits: int = 1) -> str:
    """Format a proportion as a percentage string."""
    return f"{100 * x:.{digits}f}%"


def plot_bars(labels: List[str], values: Sequence[float],
              lo: Optional[Sequence[float]] = None,
              hi: Optional[Sequence[float]] = None,
              title: str = "", ylabel: str = "",
              figsize: tuple = (8, 4),
              highlight: Optional[Sequence[int]] = None) -> plt.Figure:
    """Bar chart with optional error bars and highlighting."""
    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))

    colors = ["#8C8C8C"] + ["#1F77B4"] * (len(labels) - 1)
    if highlight is not None:
        for i in np.atleast_1d(highlight):
            colors[int(i)] = "#2CA02C"

    yerr = None
    if lo is not None and hi is not None:
        yerr = np.vstack([values - np.asarray(lo, dtype=float),
                          np.asarray(hi, dtype=float) - values])

    ax.bar(x, values, color=colors, yerr=yerr, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_grouped(categories: List[str], series: Dict[str, Sequence[float]],
                 title: str = "", ylabel: str = "",
                 figsize: tuple = (9, 4)) -> plt.Figure:
    """Grouped bar chart for comparing multiple series."""
    fig, ax = plt.subplots(figsize=figsize)
    n = len(series)
    width = 0.8 / max(n, 1)
    x = np.arange(len(categories))

    cols = ["#8C8C8C", "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=name, color=cols[i % len(cols)])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
