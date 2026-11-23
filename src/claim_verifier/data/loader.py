"""
data/loader.py
Handles locating, loading, and normalizing the SciFact dataset.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Possible file names for each split (handles variations like val.jsonl vs dev.jsonl)
FILE_NAMES = {
    "corpus": ["corpus.jsonl"],
    "train": ["claims_train.jsonl", "train.jsonl"],
    "dev": ["claims_dev.jsonl", "claims_val.jsonl", "dev.jsonl", "val.jsonl"],
    "test": ["claims_test.jsonl", "test.jsonl"],
}

def read_jsonl(path: Path) -> List[dict]:
    """Read a JSON-lines file, skipping empty lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def find_file(data_dir: Path, names: List[str]) -> Path:
    """Look for one of `names` directly in data_dir, then recursively below it."""
    for name in names:
        p = data_dir / name
        if p.is_file():
            return p
    for name in names:
        hits = sorted(h for h in data_dir.rglob(name) if h.is_file())
        if hits:
            return hits[0]
    raise FileNotFoundError(f"None of {names} found under {data_dir.resolve()}")

def load_corpus(path: Path) -> Dict[int, dict]:
    """Load corpus.jsonl into a dict mapping doc_id -> {title, abstract (list of sentences)}."""
    corpus = {}
    for rec in read_jsonl(path):
        sents = [str(s).strip() for s in rec.get("abstract", [])]
        corpus[int(rec["doc_id"])] = {"title": rec.get("title", ""), "abstract": sents}
    return corpus

def normalize_claim(rec: dict) -> dict:
    """
    Normalize a SciFact claim record into a consistent internal format.
    
    Returns a dict with:
        id: claim id
        claim: claim text
        label: SUPPORT / CONTRADICT / NO INFO
        evidence: {doc_id: sorted list of gold sentence indices}
        cited_doc_ids: list of cited doc ids
        has_evidence: bool
    """
    raw_ev = rec.get("evidence")
    evidence = {}
    label = None
    
    if raw_ev is not None:
        for doc_key, sets in raw_ev.items():
            sent_ids = sorted({int(i) for s in sets for i in s.get("sentences", [])})
            evidence[int(doc_key)] = sent_ids
            if label is None and sets:
                label = sets[0].get("label")
        # If evidence dict is empty, it's a NO INFO claim
        if not evidence:
            label = "NO INFO"
            
    return {
        "id": int(rec["id"]),
        "claim": rec["claim"],
        "label": label,
        "evidence": evidence,
        "cited_doc_ids": [int(x) for x in rec.get("cited_doc_ids", [])],
        "has_evidence": bool(evidence),
    }

def load_scifact(data_dir: str = "data/scifact") -> Tuple[Dict[int, dict], Dict[str, List[dict]]]:
    """
    Load the SciFact dataset.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory containing corpus.jsonl and claims_*.jsonl.
        
    Returns
    -------
    corpus : dict
        Mapping from doc_id to {"title": str, "abstract": list[str]}.
    splits : dict
        Mapping from split name ("train", "dev", "test") to list of normalized claims.
    """
    data_path = Path(data_dir)
    
    # Load corpus
    corpus_path = find_file(data_path, FILE_NAMES["corpus"])
    corpus = load_corpus(corpus_path)
    
    # Load splits
    splits = {}
    for split in ("train", "dev", "test"):
        try:
            split_path = find_file(data_path, FILE_NAMES[split])
            splits[split] = [normalize_claim(r) for r in read_jsonl(split_path)]
        except FileNotFoundError:
            if split == "test":
                splits[split] = []  # Test set might not be present or might lack labels
            else:
                raise
                
    return corpus, splits