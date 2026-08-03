"""Loads data/movies_seed.json into the configured database (Postgres or
the SQLite default). Optional — search works off FAISS regardless."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from app.database import init_db, SessionLocal, Movie

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    init_db()
    with open(os.path.join(DATA_DIR, "movies_seed.json")) as f:
        movies = json.load(f)

    db = SessionLocal()
    try:
        db.query(Movie).delete()
        for m in movies:
            db.add(Movie(
                id=m["id"], title=m["title"], year=m.get("year"),
                director=m.get("director"), overview=m.get("overview"),
                genres=m.get("genres", []), actors=m.get("actors", []),
                keywords=m.get("keywords", []),
            ))
        db.commit()
        print(f"Loaded {len(movies)} movies into the database.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
