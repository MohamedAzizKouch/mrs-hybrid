# Movie Recommender Streamlit App

This repository contains a Streamlit app (`app.py`) that loads artifacts produced by the `MRS.ipynb` notebook and provides:

- Content-based recommendations using TF-IDF + cosine similarity
- Recommendations using KNN (CountVectorizer + NearestNeighbors)
- Comparison view between methods
- Simple ML predictions (hit classification and rating regression) using saved models

Required artifacts (pickles) — the app will try multiple common names:
- movies_metadata.pkl or movies_dict.pkl
- tfidf_vectors.pkl and/or tfidf_similarity.pkl (or similarity.pkl)
- count_vectors.pkl and count_vectorizer.pkl
- knn_model.pkl
- classifier_rf.pkl, classifier_xgb.pkl
- regressor_rating.pkl or regressor.pkl
- feature_columns.pkl or columns.pkl
- movie_data_ml.pkl or movie_ml.pkl
- genre_classes.pkl

Quick start

1. Create a venv and install packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit app:

```powershell
streamlit run .\app.py
```

If the app reports missing artifacts, run `MRS.ipynb` to regenerate the pickles. If you prefer, you can run cells to generate only the pickles (cells near the end of the notebook).

If you want fuzzy search/autocomplete enhancements or hybrid blending UI, open an issue or ask me to add them.

Deploying on Render (recommended)

To ensure the app has the required TF-IDF / KNN artifacts on deploy, you have two options:

1) Commit the generated `.pkl` artifacts to the repo (quick, but large files in Git).

2) Regenerate artifacts during the build step (recommended). We provide `build_artifacts.py` to build the content-based pickles from the CSV files.

Example Render settings:
- Build Command: `python build_artifacts.py`
- Start Command: `streamlit run app.py`

This ensures the required files (`movies_metadata.pkl`, `tfidf_vectors.pkl`, `tfidf_similarity.pkl`, `count_vectors.pkl`, `knn_model.pkl`, etc.) are present in the environment at runtime.
