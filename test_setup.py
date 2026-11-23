"""test_setup.py - Quick validation that the modularized package works."""
import sys
from pathlib import Path

# Add the 'src' directory to the Python path so we can import claim_verifier
sys.path.insert(0, str(Path.cwd() / "src"))

def test_imports():
    print("1. Testing imports...")
    from claim_verifier.config import make_cfg, DEFAULT_CFG
    from claim_verifier.constants import LABELS, RECIPES
    from claim_verifier.retrieval.index import RetrievalIndex
    from claim_verifier.retrieval.metrics import eval_abstracts, summarize_ranks
    from claim_verifier.labeling.pipeline import Context, Pipeline
    print("   ✅ All imports successful!")
    return True

def test_basic_logic():
    print("2. Testing basic logic with mock data...")
    from claim_verifier.preprocessing.recipes import build_doc_terms, RECIPES
    from claim_verifier.retrieval.index import RetrievalIndex

    # Mock a tiny corpus to test the pipeline without needing the full dataset
    mock_corpus = {
        1: {"title": "Test title", "sents": ["This is a test sentence.", "Another test sentence."]},
        2: {"title": "Another document", "sents": ["Some other text here.", "More text."]}
    }
    
    # 1. Build terms using a recipe
    recipe = RECIPES["p1_stem"]
    doc_terms = build_doc_terms(recipe, mock_corpus)
    
    # 2. Build the retrieval index
    idx = RetrievalIndex(doc_terms)
    assert idx.N == 2, f"Index should have 2 documents, but has {idx.N}"
    
    # 3. Score a mock query
    q_terms = [["test", "sentence"]]
    scores = idx.score_abstracts(q_terms, scorer="tfidf")
    assert scores.shape == (1, 2), f"Expected score shape (1, 2), got {scores.shape}"
    
    # The first document should have a higher score because it contains "test" and "sentence"
    assert scores[0, 0] > scores[0, 1], "First document should score higher than the second"
    
    print("   ✅ Basic logic (RetrievalIndex & scoring) works correctly!")
    return True

if __name__ == "__main__":
    try:
        test_imports()
        test_basic_logic()
        print("\n🎉 All tests passed! Your modular package is correctly structured and ready to use.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()