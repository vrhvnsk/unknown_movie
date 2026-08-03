"""
Embedding backend.

Tries sentence-transformers first (true semantic embeddings, 768-dim with
the default model). If the model can't be downloaded (no internet / no
HuggingFace access), falls back automatically to a TF-IDF + LSA vectorizer
so the app still runs offline. Everything downstream (FAISS, search,
explanations) is agnostic to which backend produced the vectors.
"""
import os
import pickle
import numpy as np

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
BACKEND_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embedding_backend.pkl")


class SentenceTransformerBackend:
    name = "sentence-transformers"

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def encode_one(self, text):
        return self.encode([text])[0]


class TfidfLsaBackend:
    """
    Offline fallback: TF-IDF over movie text, compressed with truncated SVD
    (LSA) into a dense vector space, then L2-normalized so cosine similarity
    behaves the same way it would with real sentence embeddings. Not true
    semantic understanding, but captures shared vocabulary/theme overlap
    reasonably well for a demo without any network access.
    """
    name = "tfidf-lsa"

    def __init__(self, dim: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.dim = dim
        self.vectorizer = TfidfVectorizer(
            max_features=20000, ngram_range=(1, 2), stop_words="english"
        )
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False

    def fit(self, corpus):
        tfidf = self.vectorizer.fit_transform(corpus)
        n_comp = min(self.dim, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if n_comp < self.svd.n_components:
            from sklearn.decomposition import TruncatedSVD
            self.svd = TruncatedSVD(n_components=max(2, n_comp), random_state=42)
        self.svd.fit(tfidf)
        self.dim = self.svd.n_components
        self._fitted = True

    def _normalize(self, mat):
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return (mat / norms).astype("float32")

    def encode(self, texts):
        if not self._fitted:
            self.fit(texts)
        tfidf = self.vectorizer.transform(texts)
        reduced = self.svd.transform(tfidf)
        return self._normalize(reduced)

    def encode_one(self, text):
        tfidf = self.vectorizer.transform([text])
        reduced = self.svd.transform(tfidf)
        return self._normalize(reduced)[0]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "svd": self.svd, "dim": self.dim}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(dim=state["dim"])
        obj.vectorizer = state["vectorizer"]
        obj.svd = state["svd"]
        obj._fitted = True
        return obj


def get_backend(corpus_for_fallback_fit=None, prefer_offline: bool = False):
    """
    Returns an embedding backend instance. Tries sentence-transformers
    unless prefer_offline is set or the model download fails, in which
    case it falls back to TF-IDF/LSA (fit on corpus_for_fallback_fit if
    the fallback needs fitting and hasn't been saved yet).
    """
    if not prefer_offline:
        try:
            return SentenceTransformerBackend()
        except Exception as e:
            print(f"[embeddings] sentence-transformers unavailable ({e}); "
                  f"falling back to offline TF-IDF/LSA backend.")

    if os.path.exists(BACKEND_PATH):
        return TfidfLsaBackend.load(BACKEND_PATH)

    backend = TfidfLsaBackend(dim=min(EMBEDDING_DIM, 128))
    if corpus_for_fallback_fit is not None:
        backend.fit(corpus_for_fallback_fit)
        backend.save(BACKEND_PATH)
    return backend
