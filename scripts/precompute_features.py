"""scripts/precompute_features.py

Run the `utils.feature_engineering` pipeline to precompute TF-IDF and derived features.
"""
from __future__ import annotations
import argparse
from utils.feature_engineering import run_pipeline

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--out_prefix', required=True)
    args = ap.parse_args()
    res = run_pipeline(args.path, args.out_csv, args.out_prefix)
    print(res)
