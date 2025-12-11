
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
import ast
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Base directory for artifacts
PK_DIR = Path(".")

# Debug: Show current directory and files (useful for Render)
try:
    st.write("Current working directory:", os.getcwd())
    st.write("Files in directory:", os.listdir("."))
except Exception:
    pass

# TMDB API settings
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '8265bd1679663a7ea12ac168da84d2e8')
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

# Poster cache file
POSTER_CACHE_FILE = PK_DIR / 'poster_cache.json'

# ----------------------------
# Poster Cache Functions
# ----------------------------
def load_poster_cache():
    try:
        if POSTER_CACHE_FILE.exists():
            import json
            with open(POSTER_CACHE_FILE, 'r', encoding='utf-8') as fh:
                return json.load(fh)
    except Exception:
        return {}
    return {}

def save_poster_cache(cache):
    try:
        import json
        with open(POSTER_CACHE_FILE, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

def fetch_tmdb_poster(movie_id: int, poster_cache=None):
    if poster_cache is None:
        poster_cache = load_poster_cache()
    key = str(movie_id)
    if key in poster_cache and poster_cache[key]:
        return poster_cache[key]

    if not TMDB_API_KEY:
        return None
    try:
        url = f'https://api.themoviedb.org/3/movie/{int(movie_id)}'
    except Exception:
        return None
    params = {'api_key': TMDB_API_KEY, 'language': 'en-US'}
    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code != 200:
            poster_cache[key] = None
            save_poster_cache(poster_cache)
            return None
        data = resp.json()
        poster = data.get('poster_path')
        if poster:
            full = f"{TMDB_IMAGE_BASE}{poster}"
            poster_cache[key] = full
            save_poster_cache(poster_cache)
            return full
    except Exception:
        poster_cache[key] = None
        save_poster_cache(poster_cache)
        return None
    poster_cache[key] = None
    save_poster_cache(poster_cache)
    return None

# ----------------------------
# Artifact Loading
# ----------------------------
@st.cache_data
def load_pickle(name):
    p = PK_DIR / name
    if not p.exists():
        return None
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def load_artifacts():
    artifacts = {}
    artifacts['movies_df'] = load_pickle('movies_metadata.pkl') or load_pickle('movies_dict.pkl')
    artifacts['tfidf_vectors'] = load_pickle('tfidf_vectors.pkl')
    artifacts['tfidf_similarity'] = load_pickle('tfidf_similarity.pkl') or load_pickle('similarity.pkl')
    artifacts['tfidf_vectorizer'] = load_pickle('tfidf_vectorizer.pkl')
    artifacts['count_vectors'] = load_pickle('count_vectors.pkl')
    artifacts['count_vectorizer'] = load_pickle('count_vectorizer.pkl')
    artifacts['knn_model'] = load_pickle('knn_model.pkl')
    return artifacts

art = load_artifacts()

# Debug: Print artifact summary
def _artifact_brief(obj):
    if obj is None:
        return 'MISSING'
    if hasattr(obj, 'shape'):
        return f'{type(obj).__name__} shape={obj.shape}'
    return type(obj).__name__

print('ARTIFACT SUMMARY:')
for k in ['movies_df','tfidf_vectors','tfidf_similarity','tfidf_vectorizer','count_vectors','knn_model']:
    print(f' - {k}: {_artifact_brief(art.get(k))}')

# ----------------------------
# UI Setup
# ----------------------------
st.title("Hybrid Movie Recommendation — TF-IDF vs KNN")

if art['movies_df'] is None:
    st.error("Required pickles not found in the project directory. Run the notebook to generate artifacts first.")
    st.stop()

movies_df = art['movies_df']
if not isinstance(movies_df, pd.DataFrame):
    movies_df = pd.DataFrame(movies_df)

for c in ['title', 'movie_id', 'tags']:
    if c not in movies_df.columns:
        movies_df[c] = None

# Sidebar controls
st.sidebar.header("Controls")
method = st.sidebar.selectbox("Recommendation method", ["TF-IDF", "KNN", "Compare"])
search_input = st.sidebar.text_input("Search movie (partial or full)")
matches = []
if search_input:
    matches = movies_df[movies_df['title'].str.contains(search_input, case=False, na=False)]['title'].tolist()
selection = st.sidebar.selectbox("Matches (pick one)", options=matches[:50]) if matches else None
query = selection if selection else search_input
num = st.sidebar.slider("Number of recommendations", 1, 20, 5)

# ----------------------------
# Recommendation Functions
# ----------------------------
def find_movie_index(title):
    try:
        return int(movies_df[movies_df['title'] == title].index[0])
    except Exception:
        tmp = movies_df[movies_df['title'].str.contains(title, case=False, na=False)]
        return int(tmp.index[0]) if not tmp.empty else None

def recommend_tfidf(title, top_n=5):
    idx = find_movie_index(title)
    if idx is None:
        return []
    if art['tfidf_similarity'] is None and art['tfidf_vectors'] is not None:
        st.info('Computing TF-IDF similarity matrix (one-time, may be slow)...')
        art['tfidf_similarity'] = cosine_similarity(art['tfidf_vectors'])
    if art['tfidf_similarity'] is None:
        return []
    sim = art['tfidf_similarity'][idx]
    inds = np.argsort(sim)[::-1][1: top_n+1]
    return list(zip(movies_df.iloc[inds]['title'].values, sim[inds]))

def recommend_knn(title, top_n=5):
    idx = find_movie_index(title)
    if idx is None or art['knn_model'] is None or art['count_vectors'] is None:
        return []
    distances, indices = art['knn_model'].kneighbors(art['count_vectors'][idx], n_neighbors=top_n+1)
    inds = indices[0][1:]
    sims = 1 / (1 + distances[0][1:])
    return list(zip(movies_df.iloc[inds]['title'].values, sims))

# ----------------------------
# Poster and Display
# ----------------------------
def get_poster_url_for_movie(row, movie_id=None):
    candidate_ids = [movie_id] if movie_id else []
    for col in ['id', 'tmdb_id', 'tmdbId', 'movieId', 'movie_id']:
        if col in row.index and pd.notna(row[col]):
            candidate_ids.append(row[col])
    for cid in candidate_ids:
        try:
            cid_int = int(cid)
            cache = load_poster_cache()
            tmdb_url = fetch_tmdb_poster(cid_int, poster_cache=cache)
            if tmdb_url:
                return tmdb_url
        except Exception:
            continue
    return f"https://via.placeholder.com/300x450?text={row.get('title','Poster')}"

def show_recommendations_with_posters(results):
    if not results:
        st.warning('No recommendations found.')
        return
    cols = st.columns(len(results))
    for i, ((title, score), c) in enumerate(zip(results, cols), 1):
        row = movies_df[movies_df['title'] == title].iloc[0] if not movies_df.empty else pd.Series({'title': title})
        poster_url = get_poster_url_for_movie(row, movie_id=row.get('movie_id', None))
        c.image(poster_url, use_container_width=True)
        c.markdown(f"**{i}. {title}**")
        c.markdown(f"Score: {score:.4f}")

# ----------------------------
# Main Display
# ----------------------------
st.header("Recommendations")
if not query:
    st.info("Enter a movie title in the sidebar to get recommendations")
else:
    if method == 'TF-IDF':
        res = recommend_tfidf(query, top_n=num)
        show_recommendations_with_posters(res)
    elif method == 'KNN':
        res = recommend_knn(query, top_n=num)
        show_recommendations_with_posters(res)
    else:
        st.subheader("TF-IDF")
        res1 = recommend_tfidf(query, top_n=num)
        show_recommendations_with_posters(res1)
        st.subheader("KNN")
        res2 = recommend_knn(query, top_n=num)
        show_recommendations_with_posters(res2)
        st.markdown(f"**Overlap:** {len({t for t,_ in res1} & {t for t,_ in res2})}/{num}")
