import os
import sys
import pandas as pd
# ensure repo root is on path for imports when pytest runs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_cleaner import run_full_clean, _load_provider_map
from utils.data_cleaner import _impute_hours, _impute_rating, _encode_embedding_as_b64, _decode_embedding_b64
import numpy as np

def test_provider_map_loads():
    pm = _load_provider_map()
    assert isinstance(pm, dict)

def test_run_full_clean_training():
    path = 'build_training_dataset/training_database.csv'
    assert os.path.exists(path)
    res = run_full_clean(path, dataset_type='training', out_clean_path=None)
    assert 'clean_report' in res

def test_run_full_clean_jd():
    path = 'build_jd_dataset/jd_database.csv'
    assert os.path.exists(path)
    res = run_full_clean(path, dataset_type='jd', out_clean_path=None)
    assert 'clean_report' in res


def test_imputation_and_embedding_codec():
    # small df
    df = pd.DataFrame([
        {"training_id": "t1", "title": "A", "description": "X", "provider": "Coursera", "hours": None, "rating": None},
        {"training_id": "t2", "title": "B", "description": "Y", "provider": "Coursera", "hours": 3.0, "rating": 4.5},
        {"training_id": "t3", "title": "C", "description": "Z", "provider": None, "hours": None, "rating": None},
    ])
    # normalize provider via map load
    pm = _load_provider_map()
    df['provider_normalized'] = df['provider'].apply(lambda p: (pm.get(str(p).strip().lower()) if p else None))
    df2 = _impute_hours(df, strategy='median_by_provider')
    df3 = _impute_rating(df2, strategy='median_by_provider')
    # after imputation, rows with same provider should get median
    assert not df3['hours'].isna().all()
    assert not df3['rating'].isna().all()

    vec = [0.1, 0.2, 0.3]
    b64 = _encode_embedding_as_b64(vec)
    assert isinstance(b64, str)
    arr = _decode_embedding_b64(b64)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 3
