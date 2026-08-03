""" THIS is supposed to be ran once (and again any time the movie dataset changes) to build the FAISS index

would produce in backend/data/:
  - faiss.index, i.e. the FAISS file with the vector indexes itself, and
  - movies_meta.json, i.e. movie metadata in the same order as index vectors"""


import json
import os
import sys
import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from app.embeddings import get_backend

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED_PATH = os.path.join(DATA_DIR, "movies_seed.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
META_PATH = os.path.join(DATA_DIR, "movies_meta.json")


def movie_to_text(m: dict) -> str:
    """Builds the text blob that gets embedded. Overview carries the most semantic weight; 
    genres/keywords/director/actors are appended so the vector also captures tone, cast and 
    thematic tags rather than plot text alone"""

    overview = m.get("overview") or ""
    genres = m.get("genres") or []
    keywords = m.get("keywords") or []
    director = m.get("director") or ""
    actors = m.get("actors") or []

    parts = [
        overview,
        "Genres: " + ", ".join(genres),
        "Themes: " + ", ".join(keywords),
        "Director: " + director,
        "Starring: " + ", ".join(actors),
    ]
    return ". ".join(p for p in parts if p.strip())




def main(prefer_offline: bool = False):
    with open(SEED_PATH, encoding = "utf-8") as f: movies = json.load(f)

    texts = [movie_to_text(m) for m in movies]

    backend = get_backend(corpus_for_fallback_fit=texts, prefer_offline=prefer_offline)
    print(f"[build_index] using embedding backend: {backend.name} (dim={backend.dim})")

    vectors = backend.encode(texts)  # (N, dim), L2-normalized float32
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
    index.add(vectors)

    os.makedirs(DATA_DIR, exist_ok = True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding = "utf-8") as f:
        json.dump({"movies": movies, "backend": backend.name, "dim": dim}, f, indent=2)

    print(f"[build_index] indexed {len(movies)} movies -> {INDEX_PATH}")


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    main(prefer_offline=offline)
