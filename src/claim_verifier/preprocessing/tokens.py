"""
preprocessing/tokens.py
Tokenization, stemming, lemmatization, and related utilities.
"""
import re
from functools import lru_cache

import nltk
from nltk.tokenize import NLTKWordTokenizer, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from ..constants import NEG_WORDS

# Ensure NLTK resources are downloaded
def ensure_nltk(resource: str, path: str) -> None:
    """Download an NLTK resource only if it is missing."""
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, quiet=True, raise_on_error=True)

# Download required NLTK resources
for res, path in [
    ("punkt_tab", "tokenizers/punkt_tab/english/"),
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet.zip"),
    ("omw-1.4", "corpora/omw-1.4.zip"),
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng/"),
]:
    ensure_nltk(res, path)

# Initialize tokenizers and stemmers/lemmatizers
TOKENIZER = NLTKWordTokenizer()
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

# Regex tokenizer (simple word/punctuation split)
RX = re.compile(r"\w+|[^\w\s]")

# Tokenizer registry
TOKENIZERS = {
    "whitespace": str.split,
    "regex": RX.findall,
    "nltk": word_tokenize,
}

@lru_cache(maxsize=None)
def stem(word: str) -> str:
    """Porter stemmer with caching."""
    return STEMMER.stem(word, to_lowercase=False)

def wn_pos(tag: str) -> str:
    """Convert Penn Treebank POS tag to WordNet POS hint."""
    return {"J": "a", "V": "v", "R": "r"}.get(tag[0], "n")

@lru_cache(maxsize=None)
def lemma(word: str, pos: str) -> str:
    """WordNet lemmatizer with caching."""
    return LEMMATIZER.lemmatize(word, pos)

def is_negation_word(word_data: list) -> bool:
    """Check if a Stanza-annotated word is a negation cue.
    
    word_data: [text, lemma, upos, xpos, head, deprel, feats]
    """
    text = str(word_data[0]).lower()
    lem = str(word_data[1] or "").lower()
    feats = word_data[6] or ""
    return text in NEG_WORDS or lem in NEG_WORDS or "Polarity=Neg" in feats

def nltk_tokenize(text: str) -> list:
    """Tokenize text using NLTK's word tokenizer."""
    return TOKENIZER.tokenize(text)