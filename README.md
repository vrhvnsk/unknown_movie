# Semantic Movie Search

A working implementation of the pitch: search movies by *meaning* rather than
keywords, with mood sliders and a similarity graph you can click through.

This has been built and tested end-to-end — the API runs, the frontend
talks to it through a real proxy, search actually returns thematically
correct results without keyword overlap (e.g. "a movie where the
protagonist slowly loses their sanity because reality feels fake" surfaces
*Black Swan*, *The Truman Show*, *The Machinist*, *Shutter Island*).

## What's actually in here vs. what's a stub

| Piece | Status |
|---|---|
| FastAPI backend, all endpoints | ✅ fully working, tested |
| FAISS vector index + search | ✅ fully working, tested |
| "Why" explanations (similarity %, matched genres/themes) | ✅ working |
| Mood sliders | ✅ working, but see caveat below |
| Movie similarity graph (D3 force layout) | ✅ working |
| React frontend | ✅ working, builds clean, tested against live API |
| Dataset | ⚠️ 58 hand-written movies, not real TMDB data (see below) |
| Embeddings | ⚠️ defaults to real sentence-transformers, auto-falls-back to offline TF-IDF/LSA if there's no internet |
| PostgreSQL | ⚠️ wired up and optional — not required for search to work |

### Why the dataset isn't real TMDB data
This was built in a sandboxed environment with no access to the TMDB API,
Kaggle, or Hugging Face. So `backend/data/build_seed.py` contains 58
hand-written movies (title, original short overview, genres, cast,
director, keywords) spanning a wide range of tones — enough to prove the
search actually understands theme and mood, not just vocabulary.

**To swap in the real ~5,000-movie TMDB dataset:** get a free API key at
https://www.themoviedb.org/settings/api, then write a loader that hits
`/discover/movie` or the public TMDB CSV dump and reshapes each row into:

```json
{"title": "...", "year": 2010, "genres": ["..."], "director": "...",
 "actors": ["..."], "keywords": ["..."], "overview": "..."}
```

into a list saved as `backend/data/movies_seed.json`, then run
`python build_index.py` again. Everything downstream is unaffected.

### Why embeddings auto-fall-back
`sentence-transformers` downloads its model weights from Hugging Face on
first use. In an offline/sandboxed environment that fails, so
`app/embeddings.py` catches that and transparently falls back to a
TF-IDF + LSA vectorizer (via scikit-learn, no internet needed) — same
interface, same FAISS index format, works everywhere. **You have real
internet access, so you'll get real sentence embeddings automatically** —
no flag needed. Quality will be noticeably better than the offline
fallback shown in testing, especially for the mood sliders (the offline
LSA vectors have decent thematic pull but weaker abstract-mood signal
than real semantic embeddings do).

## Quick start

### 1. Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

python build_index.py        # builds FAISS index from the seed dataset
                              # (first run downloads the embedding model —
                              #  needs internet; add --offline to force the
                              #  TF-IDF fallback instead)

uvicorn app.main:app --reload --port 8000
```

Check it's alive: `curl http://localhost:8000/api/health`
Interactive API docs: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — the Vite dev server proxies `/api/*` to the
backend on port 8000 (see `vite.config.js`), so both need to be running.

### 3. (Optional) PostgreSQL

Search works entirely off the FAISS index; Postgres is only there if you
want a real system-of-record for movie metadata (e.g. for later adding
user accounts, ratings, or saved searches):

```bash
export DATABASE_URL=postgresql://user:password@localhost:5432/moviesearch
cd backend && python seed_db.py
```

Without `DATABASE_URL` set, it defaults to a local SQLite file and is
simply unused by the search flow.

## How it works

```
movies_seed.json
      │  (title + overview + genres + keywords + cast/director, concatenated)
      ▼
embeddings.py  →  sentence-transformers (or offline TF-IDF/LSA fallback)
      │
      ▼
FAISS IndexFlatIP (cosine similarity via normalized vectors)
      │
      ├── /api/search        free-text query → embed → nearest neighbors
      ├── /api/mood-search    slider weights → composite direction vector → nearest neighbors
      └── /api/graph/{id}     movie → its nearest neighbors → small force-directed graph
```

**Explanations** (`explain()` in `app/search.py`) are intentionally simple
and inspectable: the raw cosine similarity as a percentage, plus which of
the *movie's own* genres/keywords textually overlap with your query words.
This is a heuristic, not another model call — cheap, fast, and honest
about what it's showing you.

**Mood sliders** work by embedding a short canonical phrase per mood
("a bleak, disturbing, grim, unsettling story" for Dark, etc.), then
building a weighted composite vector from your slider positions and
searching FAISS with that. With real sentence-transformer embeddings this
gives a legitimately different result set as you drag the sliders; with
the offline fallback it's weaker since TF-IDF has less vocabulary overlap
with those short mood phrases.

**The graph view** takes a movie, finds its k nearest neighbors, and those
neighbors' neighbors one level out, and renders it as an interactive D3
force-directed graph — click any node to re-center the graph on it and
keep exploring.

## Project structure

```
backend/
  app/
    main.py          FastAPI app + endpoints
    search.py         SearchEngine: search, mood_search, graph, explanations
    embeddings.py      Embedding backends (sentence-transformers + offline fallback)
    database.py        Optional Postgres/SQLAlchemy models
  data/
    build_seed.py       Generates movies_seed.json
    movies_seed.json    The 58-movie demo dataset
    faiss.index         Pre-built index (offline backend) — rebuild for real embeddings
    movies_meta.json    Movie metadata aligned to the index
  build_index.py        Run to (re)build the FAISS index
  seed_db.py             Optional: loads seed data into Postgres/SQLite
  requirements.txt
frontend/
  src/
    App.jsx              Tabs, search box, results grid, graph view
    api.js                Fetch wrapper for the backend
    components/
      MovieCard.jsx        Result card with similarity bar + matched-theme chips
      MoodSliders.jsx        The four mood sliders
      MovieGraph.jsx          D3 force-directed similarity graph
  vite.config.js
  package.json
```

## Extending this

- **Bigger dataset**: swap in real TMDB/MovieLens data as described above — nothing else needs to change.
- **Better explanations**: right now "why" is genre/keyword text overlap; you could add an LLM call that compares the query and movie overview directly for a written rationale.
- **Full movie graph, not just local neighborhoods**: precompute a global 2D UMAP/t-SNE projection of all embeddings and render the whole dataset as one explorable map instead of one-movie-at-a-time neighborhoods.
- **Persisted user history**: the `SavedSearch` Postgres model is there but unused — wire up an endpoint to log queries and their top result.
