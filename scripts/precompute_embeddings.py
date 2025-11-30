"""scripts/precompute_embeddings.py

Compute embeddings for a dataset (training CSV) and store results as .npy and
base64-encoded strings in a new `embedding` column in the cleaned CSV.
"""
from __future__ import annotations
import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List

from ai.openai_client import openai_rank_courses  # used for embeddings via client
from ai.openai_client import _cosine


def _get_embeddings_texts(df: pd.DataFrame) -> List[str]:
    texts = []
    for _, r in df.iterrows():
        title = str(r.get('title') or '')
        desc = str(r.get('description') or '')
        texts.append((title + '\n' + desc).strip())
    return texts


def compute_embeddings_openai(texts: List[str]) -> List[List[float]]:
    """Use the existing openai client to compute embeddings in batches.
    This function expects OPENAI_API_KEY to be configured in env.
    """
    from openai import OpenAI
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set for embedding compute')
    client = OpenAI(api_key=api_key)
    model = 'text-embedding-3-small'
    # Create in a single call if small, otherwise batch
    resp = client.embeddings.create(model=model, input=texts)
    return [r.embedding for r in resp.data]


def save_embeddings(np_array: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, np_array)


def run(path: str, out_csv: str, out_npy: str):
    df = pd.read_csv(path)
    texts = _get_embeddings_texts(df)
    embs = compute_embeddings_openai(texts)
    arr = np.asarray(embs, dtype=np.float32)
    # attach base64 to df
    import base64
    def to_b64(vec):
        if vec is None:
            return ''
        b = np.asarray(vec, dtype=np.float32).tobytes()
        return base64.b64encode(b).decode('ascii')
    df['embedding'] = [to_b64(v) for v in embs]
    save_embeddings(arr, out_npy)
    df.to_csv(out_csv, index=False)
    print('Wrote', out_csv, out_npy, 'rows=', len(df))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_npy', required=True)
    args = ap.parse_args()
    run(args.path, args.out_csv, args.out_npy)
