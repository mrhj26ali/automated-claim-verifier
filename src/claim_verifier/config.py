"""
config.py
Pipeline configuration and factory functions.
"""

# Default configuration for the claim verification pipeline.
# This acts as the single source of truth for default hyperparameters and recipe choices.
DEFAULT_CFG = dict(
    name="",
    # Abstract retrieval
    abs_recipe="p1_stem",
    abs_scorer="tfidf",
    abs_k1=1.2,
    abs_b=0.75,
    # Sentence ranking
    sent_recipe="p1_planned",
    sent_scorer="tfidf",
    sent_k1=1.2,
    sent_b=0.75,
    # Evidence selection
    n_evidence=2,
    sent_target="first",
)

def make_cfg(**kw):
    """
    Create a pipeline configuration dictionary by overriding defaults.
    
    Parameters
    ----------
    **kw : dict
        Key-value pairs to override in DEFAULT_CFG.
        
    Returns
    -------
    dict
        The complete configuration dictionary.
        
    Raises
    ------
    KeyError
        If an unknown configuration key is provided.
    """
    unknown = set(kw) - set(DEFAULT_CFG)
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    
    cfg = dict(DEFAULT_CFG)
    cfg.update(kw)
    return cfg