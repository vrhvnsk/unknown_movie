import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.search import get_engine

app = FastAPI(title="Semantic Movie Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=20, ge=1, le=100)


class MoodSearchRequest(BaseModel):
    happy: float = 0.0
    dark: float = 0.0
    romantic: float = 0.0
    fast_paced: float = 0.0
    base_query: str = ""
    k: int = Field(default=20, ge=1, le=100)


@app.get("/api/health")
def health():
    engine = get_engine()
    return {"status": "ok", "movies_indexed": len(engine.movies), "backend": engine.backend_name}


@app.get("/api/movies")
def list_movies():
    engine = get_engine()
    return {"movies": engine.movies}


@app.get("/api/movie/{movie_id}")
def get_movie(movie_id: int):
    engine = get_engine()
    movie = engine.get_movie(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


@app.post("/api/search")
def search(req: SearchRequest):
    engine = get_engine()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    results = engine.search(req.query, k=req.k)
    return {"query": req.query, "results": results}


@app.post("/api/mood-search")
def mood_search(req: MoodSearchRequest):
    engine = get_engine()
    sliders = {"happy": req.happy, "dark": req.dark, "romantic": req.romantic, "fast_paced": req.fast_paced}
    results = engine.mood_search(sliders, k=req.k, base_query=req.base_query)
    return {"sliders": sliders, "results": results}


@app.get("/api/graph/{movie_id}")
def graph(movie_id: int, depth: int = 1, k: int = 5):
    engine = get_engine()
    if engine.get_movie(movie_id) is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return engine.graph(movie_id, depth=depth, k=k)


@app.get("/")
def root():
    return {"message": "Semantic Movie Search API. See /docs for interactive API docs."}
