import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from utils.feature_engineering import compute_tfidf, derive_numeric_features, save_tfidf_artifacts
import numpy as np
from scipy import sparse


def test_tfidf_and_numeric_derivation(tmp_path):
    df = pd.DataFrame([
        {"training_id": "t1", "title": "Intro to Python", "description": "Learn Python basics", "hours": 3, "rating": 4.5, "skill": "python"},
        {"training_id": "t2", "title": "Advanced Python", "description": "Deep dive", "hours": 10, "rating": 4.7, "skill": "python,advanced"},
    ])
    df2 = derive_numeric_features(df)
    assert 'title_len' in df2.columns and 'description_len' in df2.columns
    X, vec = compute_tfidf(df)
    assert sparse.issparse(X)
    out_prefix = str(tmp_path / 'art')
    artifacts = save_tfidf_artifacts(X, vec, out_prefix)
    assert os.path.exists(artifacts['vectorizer'])
    assert os.path.exists(artifacts['matrix'])