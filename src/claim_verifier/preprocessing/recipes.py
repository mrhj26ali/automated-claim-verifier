"""
preprocessing/recipes.py
Preprocessing recipe definitions and term extraction.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import re

import nltk
from nltk import RegexpParser

from .tokens import (
    nltk_tokenize, stem, lemma, wn_pos, is_negation_word,
    TOKENIZERS, STOP_WORDS
)
from ..constants import NEG_WORDS

@dataclass(frozen=True)
class Recipe:
    """Preprocessing recipe configuration.
    
    Attributes:
        name: Recipe identifier
        source: "text" (raw strings) or "stanza" (cached annotations)
        tokenizer: "nltk" or "space" (text source only)
        lower: Whether to lowercase tokens
        stem: "porter" (text source), "lemma" (stanza source), or None
        bigrams: Whether to add adjacent-token bigrams
        pos_keep: Stanza only - keep only these UPOS tags (plus negation words)
        drop_punct: Stanza only - drop PUNCT tokens
        entities: Stanza only - add one term per named entity
        chunks: Stanza only - add one term per multiword NP chunk
        ent_repeat: How many times each entity term is repeated
    """
    name: str
    source: str = "text"
    tokenizer: str = "nltk"
    lower: bool = True
    stem: Optional[str] = None
    bigrams: bool = False
    pos_keep: Optional[Tuple[str, ...]] = None
    drop_punct: bool = True
    entities: bool = False
    chunks: bool = False
    ent_repeat: int = 1
    
    CONTENT_POS = ("NOUN", "PROPN", "VERB", "ADJ", "NUM", "X", "SYM")
    
    def __str__(self):
        return self.name

# Define all recipes
RECIPES = {
    r.name: r
    for r in [
        # Phase 1 ladder (text based)
        Recipe("p1_raw", tokenizer="space", lower=False),
        Recipe("p1_tok", lower=False),
        Recipe("p1_lower"),
        Recipe("p1_stem", stem="porter"),
        Recipe("p1_planned", stem="porter", bigrams=True),
        
        # Phase 2 syntactic recipes (need Stanza cache)
        Recipe("syn_lemma", source="stanza", stem="lemma"),
        Recipe("syn_lemma_pos", source="stanza", stem="lemma", pos_keep=Recipe.CONTENT_POS),
        Recipe("syn_lemma_pos_ent", source="stanza", stem="lemma", pos_keep=Recipe.CONTENT_POS, entities=True),
        Recipe("syn_lemma_pos_np", source="stanza", stem="lemma", pos_keep=Recipe.CONTENT_POS, chunks=True),
        Recipe("syn_lemma_bi", source="stanza", stem="lemma", bigrams=True),
        Recipe("syn_lemma_pos_bi", source="stanza", stem="lemma", pos_keep=Recipe.CONTENT_POS, bigrams=True),
        Recipe("syn_lemma_bi_np", source="stanza", stem="lemma", bigrams=True, chunks=True),
        Recipe("syn_lemma_bi_ent", source="stanza", stem="lemma", bigrams=True, entities=True),
        Recipe("syn_lemma_bi_np_ent", source="stanza", stem="lemma", bigrams=True, chunks=True, entities=True),
    ]
}

# NP chunking pattern
_NP_PATTERN = "NP: {<DET>?<NUM>(<ADV>?<ADJ>)<NOUN|PROPN>+}"
_CHUNKER = None

def _get_chunker():
    """Lazy initialization of the NP chunker."""
    global _CHUNKER
    if _CHUNKER is None:
        _CHUNKER = RegexpParser(_NP_PATTERN)
    return _CHUNKER

def _text_tokens(text: str, recipe: Recipe) -> List[str]:
    """Extract tokens from raw text according to recipe."""
    toks = text.split() if recipe.tokenizer == "space" else nltk_tokenize(text)
    if recipe.lower:
        toks = [t.lower() for t in toks]
    if recipe.stem == "porter":
        toks = [stem(t) for t in toks]
    return toks

def _stanza_tokens(annotation: dict, recipe: Recipe) -> List[str]:
    """Extract tokens from Stanza annotation according to recipe."""
    toks = []
    for sent in annotation.get("sents", []):
        for word in sent:
            text, lemma_text, upos = word[0], word[1], word[2]
            if recipe.drop_punct and upos == "PUNCT":
                continue
            if recipe.pos_keep is not None and upos not in recipe.pos_keep and not is_negation_word(word):
                continue
            tok = (lemma_text or text) if recipe.stem == "lemma" else text
            toks.append(tok.lower() if recipe.lower else tok)
    return toks

def np_chunks(sentence: list) -> List[List[str]]:
    """Extract multiword NP chunks from a Stanza-annotated sentence.
    
    Returns list of chunks, each chunk is a list of lower-cased lemmas (DET removed).
    """
    chunker = _get_chunker()
    tagged = [(str(word[1] or word[0]).lower(), word[2]) for word in sentence]
    if not tagged:
        return []
    tree = chunker.parse(tagged)
    chunks = []
    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        leaves = [word for word, tag in subtree.leaves() if tag != "DET"]
        if len(leaves) >= 2:
            chunks.append(leaves)
    return chunks

def unit_terms(recipe: Recipe, text: Optional[str] = None, annotation: Optional[dict] = None) -> List[str]:
    """Extract index terms for one text unit (claim or abstract sentence).
    
    Args:
        recipe: Preprocessing recipe
        text: Raw text (for text-based recipes)
        annotation: Stanza annotation (for stanza-based recipes)
    
    Returns:
        List of terms (unigrams, bigrams, entities, chunks as configured)
    """
    if recipe.source == "text":
        uni = _text_tokens(text or "", recipe)
        extras = []
    else:
        if annotation is None:
            raise RuntimeError(f"Recipe '{recipe.name}' needs a Stanza annotation but none was found. "
                             "Run annotation pipeline first.")
        uni = _stanza_tokens(annotation, recipe)
        extras = []
        if recipe.chunks:
            for sent in annotation.get("sents", []):
                extras += ["np::" + "_".join(chunk) for chunk in np_chunks(sent)]
        if recipe.entities:
            for entity in annotation.get("ents", []):
                term = "ent::" + re.sub(r"\s+", " ", str(entity[0]).strip().lower())
                extras += [term] * max(1, recipe.ent_repeat)
    
    terms = list(uni)
    if recipe.bigrams:
        terms += [a + " " + b for a, b in zip(uni, uni[1:])]
    
    return terms + extras

def build_doc_terms(recipe: Recipe, corpus: Dict[int, dict], doc_annotations: Optional[Dict[int, list]] = None) -> Dict[int, List[List[str]]]:
    """Build term lists for all documents in corpus.
    
    Args:
        recipe: Preprocessing recipe
        corpus: {doc_id: {"title": str, "sents": [str]}}
        doc_annotations: {doc_id: [annotation per sentence]} (for stanza recipes)
    
    Returns:
        {doc_id: [list of terms per sentence]}
    """
    out = {}
    for doc_id, doc in corpus.items():
        if recipe.source == "text":
            out[doc_id] = [unit_terms(recipe, text=sent) for sent in doc["sents"]]
        else:
            annotations = (doc_annotations or {}).get(doc_id)
            if annotations is None or len(annotations) != len(doc["sents"]):
                raise RuntimeError(f"Missing or misaligned annotations for abstract {doc_id} "
                                 f"(recipe '{recipe.name}'). Run annotation pipeline first.")
            out[doc_id] = [unit_terms(recipe, annotation=ann) for ann in annotations]
    return out