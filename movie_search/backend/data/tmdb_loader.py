import requests
import json
import time
import os

API_KEY = "03ed468ef88e1d272050d330343fc416"    # change before releasing ig but i'll forget anyway so
BASE = "https://api.themoviedb.org/3"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "movies_seed.json")

def get_json(url, params=None):
    params = params or {}
    params["api_key"] = API_KEY
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def fetch_movie_ids(pages=50):
    ids = []
    for page in range(1, pages + 1):
        print(f"Fetching page {page}...")
        data = get_json(f"{BASE}/discover/movie", {"page": page})
        ids.extend([m["id"] for m in data["results"]])
        time.sleep(0.25)
    return ids

def fetch_movie_details(movie_id):
    details = get_json(f"{BASE}/movie/{movie_id}")
    credits = get_json(f"{BASE}/movie/{movie_id}/credits")
    keywords = get_json(f"{BASE}/movie/{movie_id}/keywords")

    director = next((c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"), None)
    actors = [c["name"] for c in credits.get("cast", [])[:5]]
    kw = [k["name"] for k in keywords.get("keywords", [])]
    genres = [g["name"] for g in details.get("genres", [])]

    release_date = details.get("release_date") or ""
    year = int(release_date[:4]) if release_date[:4].isdigit() else None

    return {
        "title": details.get("title"),
        "year": year,
        "genres": genres,
        "director": director,
        "actors": actors,
        "keywords": kw,
        "overview": details.get("overview", "")
    }

def main():
    print(f"Saving TMDB dataset to: {OUTPUT_PATH}")

    # add on - checking if directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Fetching movie IDs...")
    ids = fetch_movie_ids(pages=500)    #later change !!
    print(f"Found {len(ids)} movie IDs")

    movies = []
    for i, movie_id in enumerate(ids):

        try: movies.append(fetch_movie_details(movie_id))
        except Exception as e: print(f"Failed {movie_id}: {e}")

        time.sleep(0.25)

        if i % 50 == 0: print(f"Processed {i}/{len(ids)}")

    # note!!: search.py looks movies up by this positional id, i.e. (get_movie, neighbors, and the /api/graph/{id} endpoint all depend on it)
    for i, m in enumerate(movies): m["id"] = i

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(movies)} movies to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()