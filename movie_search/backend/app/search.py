import json
import os
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.embeddings import get_backend

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "movies_meta.json")

# Mood sliders map to short phrases that get embedded and blended with the
# index's centroid direction for that mood, then used to bias/re-rank
# results. Keeping this simple and transparent rather than a trained model.
MOOD_PHRASES = {
    "happy": "an uplifting, joyful, warm-hearted, feel-good story",
    "dark": "a bleak, disturbing, grim, unsettling story",
    "romantic": "a tender, passionate, love-focused story",
    "fast_paced": "a fast-paced, high-energy, tense, propulsive story",
}


class SearchEngine:
    def __init__(self):
        with open(META_PATH) as f:
            meta = json.load(f)
        self.movies = meta["movies"]
        self.backend_name = meta["backend"]
        self.dim = meta["dim"]
        self.index = faiss.read_index(INDEX_PATH)

        prefer_offline = self.backend_name != "sentence-transformers"
        self.backend = get_backend(prefer_offline=prefer_offline)
        # If we're using the tfidf-lsa fallback it must be the *fitted*
        # instance saved during build_index.py, which get_backend already
        # loads from disk when it exists.

        self._mood_vecs = None  # lazy-computed

    @property
    def mood_vecs(self):
        if self._mood_vecs is None:
            phrases = list(MOOD_PHRASES.values())
            vecs = self.backend.encode(phrases)
            self._mood_vecs = dict(zip(MOOD_PHRASES.keys(), vecs))
        return self._mood_vecs

    def _search_vec(self, query_vec: np.ndarray, k: int):
        query_vec = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        scores, idxs = self.index.search(query_vec, k)
        return scores[0], idxs[0]

    def explain(self, query_text: str, movie: dict, score: float) -> dict:
        """
        Heuristic, human-readable explanation: which of the movie's own
        genres/keywords textually overlap with the query, plus the raw
        cosine similarity as a percentage. This is intentionally simple
        and inspectable rather than another model call.
        """
        query_lower = query_text.lower()
        matched_keywords = [k for k in movie.get("keywords", []) if k.lower() in query_lower or
                             any(w in query_lower for w in k.lower().split())]
        matched_genres = [g for g in movie.get("genres", []) if g.lower() in query_lower]
        return {
            "similarity_pct": round(float(score) * 100, 1),
            "matched_genres": matched_genres[:4],
            "matched_themes": list(dict.fromkeys(matched_keywords))[:5],
        }

    def search(self, query_text: str, k: int = 20):
        query_vec = self.backend.encode_one(query_text)
        scores, idxs = self._search_vec(query_vec, k)
        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            movie = self.movies[idx]
            results.append({
                **movie,
                "explanation": self.explain(query_text, movie, score),
            })
        return results

    def mood_search(self, sliders: dict, k: int = 20, base_query: str = ""):
        """
        sliders: dict of mood_name -> 0..1 weight, e.g.
          {"happy": 0.1, "dark": 0.9, "romantic": 0.2, "fast_paced": 0.0}
        Builds a composite query vector: weighted sum of mood direction
        vectors (plus an optional free-text base query), re-normalized.
        """
        vec = np.zeros(self.dim if self.backend_name == "sentence-transformers" else self.backend.dim,
                        dtype="float32")
        total_weight = 0.0
        for mood, weight in sliders.items():
            if mood not in MOOD_PHRASES or weight <= 0:
                continue
            vec += weight * self.mood_vecs[mood]
            total_weight += weight

        if base_query.strip():
            vec += self.backend.encode_one(base_query)
            total_weight += 1.0

        if total_weight == 0:
            # no signal at all; just return a neutral sample
            vec = self.mood_vecs["happy"] * 0.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        scores, idxs = self._search_vec(vec, k)
        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            movie = self.movies[idx]
            results.append({
                **movie,
                "explanation": {
                    "similarity_pct": round(float(score) * 100, 1),
                    "matched_genres": movie.get("genres", [])[:3],
                    "matched_themes": movie.get("keywords", [])[:4],
                },
            })
        return results

    def get_movie(self, movie_id: int):
        for m in self.movies:
            if m["id"] == movie_id:
                return m
        return None

    def neighbors(self, movie_id: int, k: int = 6):
        """Returns the k nearest other movies to movie_id, for the graph view."""
        movie = self.get_movie(movie_id)
        if movie is None:
            return []
        text = None
        # Re-derive the same text used at index time isn't stored, so we
        # reconstruct it consistently from the movie's own fields.
        from build_index import movie_to_text  # local import avoids circulars at module load
        text = movie_to_text(movie)
        vec = self.backend.encode_one(text)
        scores, idxs = self._search_vec(vec, k + 1)  # +1 since the movie matches itself
        out = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or self.movies[idx]["id"] == movie_id:
                continue
            out.append({"movie": self.movies[idx], "similarity_pct": round(float(score) * 100, 1)})
        return out[:k]

    def graph(self, movie_id: int, depth: int = 1, k: int = 5):
        """
        BFS-style small graph around a movie: nodes + edges, for the
        Plotly/D3 exploration view. depth=1 keeps it small and readable.
        """
        nodes = {}
        edges = []
        frontier = [movie_id]
        seen = set()
        for _ in range(depth + 1):
            next_frontier = []
            for mid in frontier:
                if mid in seen:
                    continue
                seen.add(mid)
                m = self.get_movie(mid)
                if m is None:
                    continue
                nodes[mid] = {"id": mid, "title": m["title"], "year": m.get("year"),
                               "genres": m.get("genres", [])}
                for n in self.neighbors(mid, k=k):
                    nid = n["movie"]["id"]
                    nodes.setdefault(nid, {"id": nid, "title": n["movie"]["title"],
                                            "year": n["movie"].get("year"),
                                            "genres": n["movie"].get("genres", [])})
                    edges.append({"source": mid, "target": nid, "weight": n["similarity_pct"]})
                    next_frontier.append(nid)
            frontier = next_frontier
        return {"nodes": list(nodes.values()), "edges": edges}


_engine = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine
