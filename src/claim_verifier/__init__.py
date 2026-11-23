"""
claim_verifier
A modular package for scientific claim verification on SciFact.
"""
from . import constants, config, data, preprocessing, retrieval, labeling, evaluation
from .plotting import plot_bars, plot_grouped, pct

__version__ = "0.1.0"
__all__ = [
    "constants", "config", "data", "preprocessing", "retrieval", "labeling", "evaluation",
    "plot_bars", "plot_grouped", "pct",
]