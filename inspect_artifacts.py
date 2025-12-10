import pathlib
import pickle
import json
import ast
import pandas as pd

FILES = [
    'movies_metadata.pkl','movies_dict.pkl',
    'tfidf_vectors.pkl','tfidf_similarity.pkl','similarity.pkl',
    'tfidf_vectorizer.pkl','count_vectors.pkl','count_vectorizer.pkl',
    'knn_model.pkl','classifier_rf.pkl','classifier_xgb.pkl',
    'regressor_rating.pkl','regressor.pkl','feature_columns.pkl','columns.pkl',
    'movie_data_ml.pkl','movie_ml.pkl','genre_classes.pkl'
]

p = pathlib.Path('.')

def preview_file(fp: pathlib.Path):
    out = {'path': str(fp), 'exists': fp.exists()}
    if not fp.exists():
        return out
    try:
        with open(fp, 'rb') as f:
            head = f.read(64)
        out['first_bytes'] = head[:32]
    except Exception as e:
        out['first_bytes_error'] = str(e)
    # Try reading as text
    try:
        text = fp.read_text(encoding='utf-8')
        out['text_preview'] = text.strip().replace('\n',' ')[:300]
    except Exception:
        out['text_preview'] = None
    # Try pickle load
    try:
        with open(fp, 'rb') as f:
            obj = pickle.load(f)
        out['pickle_load'] = True
        try:
            if isinstance(obj, (list, tuple)):
                out['pickle_type'] = f'list/tuple len={len(obj)}'
            elif hasattr(obj, 'shape'):
                out['pickle_type'] = f'{type(obj)} shape={getattr(obj, "shape", None)}'
            elif isinstance(obj, dict):
                out['pickle_type'] = f'dict keys={list(obj.keys())[:10]}'
            else:
                out['pickle_type'] = str(type(obj))
        except Exception:
            out['pickle_type'] = str(type(obj))
    except Exception as e:
        out['pickle_load'] = False
        out['pickle_error'] = str(e)
    # Try JSON
    try:
        text = fp.read_text(encoding='utf-8')
        obj = json.loads(text)
        out['json_load'] = True
        out['json_type'] = type(obj).__name__
    except Exception:
        out['json_load'] = False
    # Try ast.literal_eval
    try:
        text = fp.read_text(encoding='utf-8')
        obj = ast.literal_eval(text)
        out['literal_eval'] = True
        out['literal_type'] = type(obj).__name__
    except Exception:
        out['literal_eval'] = False
    # Try CSV
    try:
        df = pd.read_csv(fp)
        out['csv_read'] = True
        out['csv_shape'] = getattr(df, 'shape', None)
    except Exception:
        out['csv_read'] = False
    return out


def main():
    print('Inspecting artifact files in repo root...')
    results = []
    for fn in FILES:
        fp = p / fn
        r = preview_file(fp)
        results.append(r)
    # Print summary
    for r in results:
        print('\n---')
        print('file:', r.get('path'))
        print('exists:', r.get('exists'))
        if r.get('first_bytes') is not None:
            fb = r['first_bytes']
            try:
                print('first_bytes:', fb[:32])
            except Exception:
                print('first_bytes: (binary)')
        if r.get('text_preview'):
            print('text_preview (start):', r['text_preview'][:200])
        print('pickle_load:', r.get('pickle_load'))
        if not r.get('pickle_load'):
            print('pickle_error:', r.get('pickle_error'))
        else:
            print('pickle_type:', r.get('pickle_type'))
        print('json_load:', r.get('json_load'), 'literal_eval:', r.get('literal_eval'), 'csv_read:', r.get('csv_read'))

if __name__ == '__main__':
    main()
