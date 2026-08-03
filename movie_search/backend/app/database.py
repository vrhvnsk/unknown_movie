"""
Optional PostgreSQL layer.

The search engine itself runs entirely off the FAISS index + the JSON
metadata file — Postgres is NOT required to use semantic search. It's
here so you have a real system-of-record for movie metadata (e.g. if you
later want to add user accounts, saved searches, or ratings), matching
the "Database: PostgreSQL" part of the spec.

If DATABASE_URL isn't set, the app runs fine without ever touching this
module. To enable Postgres:

    export DATABASE_URL=postgresql://user:password@localhost:5432/moviesearch
    python seed_db.py     # loads data/movies_seed.json into Postgres
"""
import os

from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/movies.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    year = Column(Integer)
    director = Column(String)
    overview = Column(String)
    genres = Column(JSON)      # list[str]
    actors = Column(JSON)      # list[str]
    keywords = Column(JSON)    # list[str]


class SavedSearch(Base):
    """Optional: lets a frontend persist a user's past semantic queries."""
    __tablename__ = "saved_searches"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String)
    mood_sliders = Column(JSON, nullable=True)
    top_result_title = Column(String, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
