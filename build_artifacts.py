"""
build_artifacts.py

Small script to (re)build the content-based artifacts required by the Streamlit app.
Run this in the project root; it reads the TMDB CSVs and writes pickles:
 - movies_metadata.pkl (movie_id, title, tags)
 - tfidf_vectors.pkl
 - tfidf_similarity.pkl
 - tfidf_vectorizer.pkl
 - count_vectors.pkl
 - count_vectorizer.pkl
 - knn_model.pkl

This is intended to run on deploy (Render build command) so you don't need to commit large .pkl files.
"""
import ast
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

PK_DIR = Path('.')
MOVIES_CSV = PK_DIR / 'tmdb_5000_movies.csv'
CREDITS_CSV = PK_DIR / 'tmdb_5000_credits.csv'
OUT_DIR = PK_DIR


def safe_parse_list(txt):
    try:
        return [i.get('name','') for i in ast.literal_eval(txt)]
    except Exception:
        return []


def build():
    print('Building artifacts from CSVs...')
    if not MOVIES_CSV.exists():
        raise FileNotFoundError(f"{MOVIES_CSV} not found")
    if not CREDITS_CSV.exists():
        print('Warning: credits CSV not found; continuing with movies CSV only')

    movies = pd.read_csv(MOVIES_CSV)
    # Keep necessary columns
    keep = ['movie_id','title','overview','genres','keywords']
    for c in keep:
        if c not in movies.columns:
            movies[c] = ''

    # Build tags: combine genres, keywords, and overview
    def genres_to_text(g):
        try:
            return ' '.join([i.get('name','') for i in ast.literal_eval(g)])
        except Exception:
            return ''

    def keywords_to_text(k):
        try:
            return ' '.join([i.get('name','') for i in ast.literal_eval(k)])
        except Exception:
            return ''

    movies['tags'] = movies.apply(lambda r: ' '.join([str(r.get('overview','') or ''), genres_to_text(r.get('genres','')), keywords_to_text(r.get('keywords',''))]), axis=1)
    movies_metadata = movies[['movie_id','title','tags']].copy()

    # TF-IDF
    print('Vectorizing TF-IDF...')
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_vectors = tfidf.fit_transform(movies_metadata['tags'].fillna(''))
    print('Computing TF-IDF similarity (this may take some memory)...')
    tfidf_similarity = cosine_similarity(tfidf_vectors)

    # Count + KNN (for KNN approach)
    print('Vectorizing CountVectorizer + training KNN...')
    count_vec = CountVectorizer(max_features=5000, stop_words='english')
    count_vectors = count_vec.fit_transform(movies_metadata['tags'].fillna(''))
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(count_vectors)

    # Save artifacts
    print('Saving pickles...')
    pickle.dump(movies_metadata, open(OUT_DIR / 'movies_metadata.pkl', 'wb'))
    pickle.dump(tfidf_vectors, open(OUT_DIR / 'tfidf_vectors.pkl', 'wb'))
    pickle.dump(tfidf_similarity, open(OUT_DIR / 'tfidf_similarity.pkl', 'wb'))
    pickle.dump(tfidf, open(OUT_DIR / 'tfidf_vectorizer.pkl', 'wb'))
    pickle.dump(count_vectors, open(OUT_DIR / 'count_vectors.pkl', 'wb'))
    pickle.dump(count_vec, open(OUT_DIR / 'count_vectorizer.pkl', 'wb'))
    pickle.dump(knn, open(OUT_DIR / 'knn_model.pkl', 'wb'))

    print('Artifacts built successfully.')

if __name__ == '__main__':
    build()
