"""
evaluation/__init__.py
Public API for the evaluation subpackage.
"""
from .statistics import (
    bootstrap_ci, paired_bootstrap, bootstrap_stat, paired_bootstrap_stat,
    compare_ladder, format_ladder, select_recipe,
    macro_f1, label_metrics
)
from .errors import error_taxonomy, funnel_counts

__all__ = [
    "bootstrap_ci", "paired_bootstrap", "bootstrap_stat", "paired_bootstrap_stat",
    "compare_ladder", "format_ladder", "select_recipe",
    "macro_f1", "label_metrics",
    "error_taxonomy", "funnel_counts",
]