from triage_agent.embeddings import Embedder


def test_embed_is_deterministic():
    e = Embedder()
    v1 = e.embed("memory leak in async middleware")
    v2 = e.embed("memory leak in async middleware")
    assert v1 == v2


def test_embed_different_text_differs():
    e = Embedder()
    v1 = e.embed("memory leak in async middleware")
    v2 = e.embed("typo in the README installation instructions")
    assert v1 != v2


def test_similar_text_has_higher_cosine_similarity_than_dissimilar():
    import numpy as np

    e = Embedder()
    query = e.embed("memory leak when chaining async middleware")
    related = e.embed("middleware chain causes unbounded memory growth")
    unrelated = e.embed("typo in README installation section")

    def cos(a, b):
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    assert cos(query, related) > cos(query, unrelated)
