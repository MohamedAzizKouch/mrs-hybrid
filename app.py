import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
import json
import ast
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

st.set_page_config(page_title="Movie Recommender", layout="wide")

PK_DIR = Path(".")

# TMDB configuration: prefer env var, fallback to a provided key if present
TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '8265bd1679663a7ea12ac168da84d2e8')
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

# Poster cache file (simple JSON mapping movie_id -> poster_url)
POSTER_CACHE_FILE = PK_DIR / 'poster_cache.json'

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
    """Fetch poster URL from TMDB, using file-backed cache when possible.
    Returns full image URL or None.
    """
    if poster_cache is None:
        poster_cache = load_poster_cache()
    key = str(movie_id)
    # cache hit
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
        try:
            save_poster_cache(poster_cache)
        except Exception:
            pass
        return None
    poster_cache[key] = None
    try:
        save_poster_cache(poster_cache)
    except Exception:
        pass
    return None

@st.cache_data
def load_pickle(name):
    p = PK_DIR / name
    if not p.exists():
        return None
    # Read the first bytes to decide whether this file looks like a pickle
    try:
        with open(p, 'rb') as f:
            head = f.read(4)
            # Common pickle protocol starts with 0x80
            if len(head) > 0 and head[0] == 0x80:
                f.seek(0)
                try:
                    return pickle.load(f)
                except Exception as e:
                    # fall through to tolerant loaders
                    pickle_err = e
            else:
                pickle_err = None
    except Exception as e:
        pickle_err = e

    # If we reach here, either file didn't look like a pickle or unpickling failed.
    # Try tolerant fallbacks: JSON, python literal, CSV, or DataFrame reconstruction.
    text = None
    try:
        text = p.read_text(encoding='utf-8')
    except Exception:
        text = None

    if text:
        # Try JSON
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return obj
            if isinstance(obj, list):
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return obj
            return obj
        except Exception:
            pass

        # Try python literal repr
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return obj
            if isinstance(obj, list):
                try:
                    return pd.DataFrame(obj)
                except Exception:
                    return obj
            return obj
        except Exception:
            pass

    # Try reading as CSV
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        pass

    # Nothing worked — warn and return None
    try:
        msg = f"Failed to load '{name}' as pickle/JSON/CSV."
        if pickle_err is not None:
            msg += f" Pickle error: {pickle_err}"
        st.warning(msg)
    except Exception:
        pass
    return None

@st.cache_data
def load_artifacts():
    artifacts = {}
    # Movies metadata: prefer `movies_metadata.pkl`, fallback to `movies_dict.pkl`
    _a = load_pickle('movies_metadata.pkl')
    _b = load_pickle('movies_dict.pkl')
    artifacts['movies_df'] = _a if _a is not None else _b
    # TF-IDF artifacts
    artifacts['tfidf_vectors'] = load_pickle('tfidf_vectors.pkl')
    _a = load_pickle('tfidf_similarity.pkl')
    _b = load_pickle('similarity.pkl')
    artifacts['tfidf_similarity'] = _a if _a is not None else _b
    artifacts['tfidf_vectorizer'] = load_pickle('tfidf_vectorizer.pkl')
    # Count/KNN artifacts
    artifacts['count_vectors'] = load_pickle('count_vectors.pkl')
    artifacts['count_vectorizer'] = load_pickle('count_vectorizer.pkl')
    artifacts['knn_model'] = load_pickle('knn_model.pkl')
    # Models and metadata (try alternate names)
    artifacts['classifier_rf'] = load_pickle('classifier_rf.pkl')
    artifacts['classifier_xgb'] = load_pickle('classifier_xgb.pkl')
    _a = load_pickle('regressor_rating.pkl')
    _b = load_pickle('regressor.pkl')
    artifacts['regressor'] = _a if _a is not None else _b
    _a = load_pickle('feature_columns.pkl')
    _b = load_pickle('columns.pkl')
    artifacts['feature_columns'] = _a if _a is not None else _b
    _a = load_pickle('movie_data_ml.pkl')
    _b = load_pickle('movie_ml.pkl')
    artifacts['movie_data_ml'] = _a if _a is not None else _b
    artifacts['genre_classes'] = load_pickle('genre_classes.pkl')
    return artifacts

art = load_artifacts()

# Print a concise artifact summary to stdout (useful for Render logs)
def _artifact_brief(obj):
    try:
        if obj is None:
            return 'MISSING'
        t = type(obj)
        # pandas DataFrame
        if hasattr(obj, 'shape') and hasattr(obj, 'columns'):
            return f'DataFrame shape={obj.shape}'
        # numpy / sparse
        if hasattr(obj, 'shape'):
            return f'{t.__name__} shape={getattr(obj, "shape", None)}'
        # sklearn estimators
        return t.__name__
    except Exception:
        return str(type(obj))

try:
    print('ARTIFACT SUMMARY:')
    for k in ['movies_df','tfidf_vectors','tfidf_similarity','tfidf_vectorizer','count_vectors','knn_model']:
        v = art.get(k)
        print(f' - {k}: {_artifact_brief(v)}')
except Exception:
    pass

st.title("Hybrid Movie Recommendation — TF-IDF vs KNN")

if art['movies_df'] is None:
    st.error("Required pickles not found in the project directory. Run the notebook to generate artifacts first.")
    st.stop()

# Ensure movies_df is DataFrame and normalized
raw_movies = art['movies_df']
if raw_movies is None:
    movies_df = None
elif isinstance(raw_movies, pd.DataFrame):
    movies_df = raw_movies.copy()
else:
    # If it's a dict saved via df[['movie_id','title','tags']].to_dict(), reconstruct
    try:
        movies_df = pd.DataFrame(raw_movies)
        # If keys are columns and values are dicts of index->value, transpose
        if 'title' not in movies_df.columns and movies_df.shape[0] > 0:
            movies_df = movies_df.T
    except Exception:
        movies_df = pd.DataFrame(raw_movies)

# Ensure expected columns
if movies_df is not None:
    for c in ['title', 'movie_id', 'tags']:
        if c not in movies_df.columns:
            movies_df[c] = None

st.sidebar.header("Controls")
method = st.sidebar.selectbox("Recommendation method", ["TF-IDF", "KNN", "Compare"])
# Autocomplete: free-text input + live matches selectbox
search_input = st.sidebar.text_input("Search movie (partial or full)")
matches = []
if search_input:
    try:
        matches = movies_df[movies_df['title'].str.contains(search_input, case=False, na=False)]['title'].tolist()
    except Exception:
        matches = []

selection = None
if matches:
    # show top 50 matches to keep the UI responsive
    selection = st.sidebar.selectbox("Matches (pick one)", options=matches[:50])

# Final query: prefer selection (exact pick), else typed input
query = selection if selection else search_input

num = st.sidebar.slider("Number of recommendations", 1, 20, 5)

# TMDB fetching is always enabled by default (controls removed from UI)
enable_tmdb = True
use_cache_only = False

col1 = st.container()

def find_movie_index(title):
    df = movies_df
    try:
        return int(df[df['title'] == title].index[0])
    except Exception:
        tmp = df[df['title'].str.contains(title, case=False, na=False)]
        if tmp.empty:
            return None
        return int(tmp.index[0])

def recommend_tfidf(title, top_n=5):
    idx = find_movie_index(title)
    if idx is None:
        return []
    # ensure similarity matrix exists; compute if only vectors present
    if art.get('tfidf_similarity') is None and art.get('tfidf_vectors') is not None:
        try:
            st.info('Computing TF-IDF similarity matrix (one-time, may be slow)...')
            art['tfidf_similarity'] = cosine_similarity(art['tfidf_vectors'])
        except Exception:
            return []
    if art.get('tfidf_similarity') is None:
        return []
    sim = art['tfidf_similarity'][idx]
    inds = np.argsort(sim)[::-1][1: top_n+1]
    return list(zip(movies_df.iloc[inds]['title'].values, sim[inds]))

def recommend_knn(title, top_n=5):
    idx = find_movie_index(title)
    if idx is None:
        return []
    if art.get('knn_model') is None or art.get('count_vectors') is None:
        return []
    # kneighbors expects array-like
    distances, indices = art['knn_model'].kneighbors(art['count_vectors'][idx], n_neighbors=top_n+1)
    inds = indices[0][1:]
    dists = distances[0][1:]
    sims = 1 / (1 + dists)
    return list(zip(movies_df.iloc[inds]['title'].values, sims))

def get_poster_url_for_movie(row, movie_id=None):
    # 1) Prefer TMDB API lookup if a valid TMDB id is available
    candidate_ids = []
    if movie_id is not None:
        candidate_ids.append(movie_id)
    # try common id column names
    for col in ['id', 'tmdb_id', 'tmdbId', 'movieId', 'movie_id']:
        try:
            if col in row.index and pd.notna(row[col]):
                candidate_ids.append(row[col])
        except Exception:
            continue

    for cid in candidate_ids:
        try:
            # ensure numeric
            cid_int = int(cid)
        except Exception:
            continue
        try:
            # respect sidebar toggles: if disabled, skip HTTP fetch
            if not enable_tmdb:
                tmdb_url = None
            elif use_cache_only:
                # check cache without making network calls
                cache = load_poster_cache()
                tmdb_url = cache.get(str(cid_int))
            else:
                cache = load_poster_cache()
                tmdb_url = fetch_tmdb_poster(cid_int, poster_cache=cache)
            if tmdb_url:
                return tmdb_url
        except Exception:
            continue

    # 2) Try common fields that may contain a poster URL or path
    for col in ['poster_link', 'poster', 'poster_path', 'poster_url']:
        try:
            if col in row.index and pd.notna(row[col]) and row[col]:
                val = str(row[col])
                if val.startswith('/'):
                    return f"{TMDB_IMAGE_BASE}{val}"
                if val.startswith('http'):
                    return val
        except Exception:
            continue

    # 3) Fallback: generate a placeholder image with the movie title
    title = None
    try:
        title = row.get('title') if 'title' in row.index else None
    except Exception:
        title = None
    if title is None and movie_id is not None:
        title = str(movie_id)
    if title is None:
        title = 'Poster'
    import urllib.parse
    text = urllib.parse.quote_plus(title)
    return f"https://via.placeholder.com/300x450?text={text}"

def show_recommendations_with_posters(results, container):
    if not results:
        container.warning('No recommendations found (movie not in dataset or artifacts missing)')
        return
    # display as horizontal list of posters
    cols = st.columns(len(results))
    for i, ((title, score), c) in enumerate(zip(results, cols), 1):
        try:
            # find movie row
            row = movies_df[movies_df['title'] == title].iloc[0]
        except Exception:
            row = pd.Series({'title': title})
        poster_url = get_poster_url_for_movie(row, movie_id=row.get('movie_id', None))
        c.image(poster_url, use_container_width=True)
        c.markdown(f"**{i}. {title}**")
        c.markdown(f"Score: {score:.4f}")

with col1:
    st.header("Recommendations")
    if not query:
        st.info("Enter a movie title in the sidebar to get recommendations")
    else:
        if method == 'TF-IDF':
            res = recommend_tfidf(query, top_n=num)
            show_recommendations_with_posters(res, st)
        elif method == 'KNN':
            res = recommend_knn(query, top_n=num)
            show_recommendations_with_posters(res, st)
        else:  # Compare
            st.subheader("TF-IDF")
            res1 = recommend_tfidf(query, top_n=num)
            show_recommendations_with_posters(res1, st)
            st.subheader("KNN")
            res2 = recommend_knn(query, top_n=num)
            show_recommendations_with_posters(res2, st)
            # overlap
            titles1 = {t for t, _ in res1}
            titles2 = {t for t, _ in res2}
            st.markdown(f"**Overlap:** {len(titles1 & titles2)}/{num}")

