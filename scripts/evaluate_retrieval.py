#!/usr/bin/env python3
"""CLI to compute retrieval/ranking/classification and embedding-health metrics.

Example usage:
  python scripts/evaluate_retrieval.py \
    --gt ground_truth.json \
    --retrieved retrieved.json \
    --embeddings embeddings.npy \
    --out eval_report.json

ground_truth.json format: { "query_id": ["doc1", "doc2", ...], ... }
retrieved.json format: { "query_id": ["doc5", "doc2", ...], ... }

embeddings.npy: optional; numpy array of shape (n, d) for embedding health checks.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set
import numpy as np

from utils.eval_metrics import (
    compute_precision_recall_at_k_for_all,
    mean_reciprocal_rank,
    ndcg_at_k,
    classification_metrics,
    mean_cosine_similarity,
)


def load_json_map(path: Path) -> Dict[str, List[str]]:
    with open(path, 'r') as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt', required=True, help='Ground truth JSON: query_id -> [relevant_doc_ids]')
    p.add_argument('--retrieved', required=True, help='Retrieved JSON: query_id -> [retrieved_doc_ids]')
    p.add_argument('--ks', nargs='+', type=int, default=[1,5,10], help='k values for Precision/Recall/NDCG')
    p.add_argument('--embeddings', help='Optional numpy .npy file with embeddings for health check')
    p.add_argument('--out', required=True, help='Output JSON file to write evaluation report')
    args = p.parse_args()

    gt_map_raw = load_json_map(Path(args.gt))
    retrieved_raw = load_json_map(Path(args.retrieved))

    # normalize to sets/lists
    all_relevant = {qid: set(v) for qid, v in gt_map_raw.items()}
    all_retrieved = {qid: v for qid, v in retrieved_raw.items()}

    ks = args.ks
    pr = compute_precision_recall_at_k_for_all(all_relevant, all_retrieved, ks)
    mrr = mean_reciprocal_rank(all_relevant, all_retrieved, max(ks))
    ndcg = ndcg_at_k(all_relevant, all_retrieved, max(ks))

    report = {
        'precision_recall_at_k': pr,
        'mrr_at_k': float(mrr),
        'ndcg_at_k': float(ndcg),
    }

    if args.embeddings:
        embs = np.load(args.embeddings)
        health = mean_cosine_similarity(embs)
        report['embedding_health'] = health

    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Wrote evaluation report to {args.out}')


if __name__ == '__main__':
    main()
