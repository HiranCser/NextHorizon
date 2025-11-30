"""Evaluation metrics for retrieval, ranking, classification, and embedding health.

Provides:
- precision_recall_at_k
- mean_reciprocal_rank (MRR)
- ndcg_at_k
- classification_metrics (accuracy, F1, confusion matrix)
- embedding_health (mean cosine similarity, zero-norm rate)
"""
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
from math import log2

def precision_recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> Tuple[float, Optional[float]]:
    """Compute precision@k and recall@k for a single query.

    relevant: set of relevant document ids
    retrieved: ranked list of document ids
    Returns (precision, recall) where recall is None when no relevant items.
    """
    if k <= 0:
        return 0.0, None
    # dedupe retrieved while preserving order to avoid counting duplicates
    seen = set()
    dedup = []
    for d in retrieved:
        if d in seen:
            continue
        seen.add(d)
        dedup.append(d)
    topk = dedup[:k]
    hits = sum(1 for d in topk if d in relevant)
    precision = hits / float(k)
    recall = hits / float(len(relevant)) if len(relevant) > 0 else None
    return precision, recall

def mean_reciprocal_rank(all_relevant: Dict[str, Set[str]], all_retrieved: Dict[str, List[str]], k: int) -> float:
    """Compute MRR@k over a set of queries.
    all_relevant: mapping query_id -> set(relevant_doc_ids)
    all_retrieved: mapping query_id -> list(retrieved_doc_ids ordered by score)
    """
    rr_sum = 0.0
    n = 0
    for qid, rel in all_relevant.items():
        retrieved = all_retrieved.get(qid, [])
        # dedupe retrieved
        seen = set()
        dedup = []
        for d in retrieved:
            if d in seen:
                continue
            seen.add(d)
            dedup.append(d)
        retrieved = dedup
        found = False
        for rank, doc in enumerate(retrieved[:k], start=1):
            if doc in rel:
                rr_sum += 1.0 / rank
                found = True
                break
        if not found:
            rr_sum += 0.0
        n += 1
    return rr_sum / max(1, n)

def ndcg_at_k(all_relevant: Dict[str, Set[str]], all_retrieved: Dict[str, List[str]], k: int) -> float:
    """Compute average NDCG@k (binary relevance) over queries.

    This uses the binary relevance formulation where gain = rel (0/1) and
    DCG = sum(rel_i / log2(rank+1)). IDCG is the DCG of an ideal list
    with all relevant items at the top (up to k). Returns average over queries.
    """
    ndcg_sum = 0.0
    n = 0
    for qid, rel in all_relevant.items():
        retrieved = all_retrieved.get(qid, [])
        # dedupe retrieved while preserving order
        seen = set()
        dedup = []
        for d in retrieved:
            if d in seen:
                continue
            seen.add(d)
            dedup.append(d)
        retrieved = dedup[:k]
        dcg = 0.0
        for i, doc in enumerate(retrieved, start=1):
            rel_i = 1.0 if doc in rel else 0.0
            dcg += rel_i / log2(i + 1)
        # ideal DCG: place as many relevant items as possible at top
        ideal_rels = min(len(rel), k)
        idcg = 0.0
        for i in range(1, ideal_rels + 1):
            idcg += 1.0 / log2(i + 1)
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        ndcg_sum += ndcg
        n += 1
    return ndcg_sum / max(1, n)

def compute_precision_recall_at_k_for_all(all_relevant: Dict[str, Set[str]], all_retrieved: Dict[str, List[str]], ks: List[int]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for k in ks:
        precisions = []
        recalls = []
        for qid, rel in all_relevant.items():
            prec, rec = precision_recall_at_k(rel, all_retrieved.get(qid, []), k)
            precisions.append(prec)
            if rec is not None:
                recalls.append(rec)
        out[k] = {
            'precision_mean': float(np.mean(precisions)) if precisions else 0.0,
            'recall_mean': float(np.mean(recalls)) if recalls else None,
        }
    return out

def classification_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, Any]:
    """Compute accuracy, F1 (macro), and confusion matrix using sklearn if available."""
    try:
        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    except Exception:
        raise RuntimeError('sklearn is required for classification_metrics')
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average='macro'))
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {'accuracy': acc, 'f1_macro': f1, 'confusion_matrix': cm}

def mean_cosine_similarity(embs: np.ndarray, sample_pairs: int = 10000, random_state: Optional[int] = 42) -> Dict[str, Any]:
    """Compute mean cosine similarity over randomly sampled pairs and zero-norm rate.

    embs: numpy array shape (n, d)
    Returns dict with mean_cosine, std_cosine, zero_norm_rate, sample_pairs_used
    """
    rng = np.random.default_rng(random_state)
    n = embs.shape[0]
    norms = np.linalg.norm(embs, axis=1)
    zero_norm_rate = float(np.count_nonzero(norms == 0.0) / max(1, n))
    # prepare normalized embeddings for cosine
    nonzero = norms > 0
    if nonzero.sum() < 2:
        return {'mean_cosine': 0.0, 'std_cosine': 0.0, 'zero_norm_rate': zero_norm_rate, 'sample_pairs': 0}
    normed = embs[nonzero] / norms[nonzero][:, None]
    m = normed.shape[0]
    pairs = min(sample_pairs, m * (m - 1) // 2)
    if pairs <= 0:
        return {'mean_cosine': 0.0, 'std_cosine': 0.0, 'zero_norm_rate': zero_norm_rate, 'sample_pairs': 0}
    # sample random pairs
    idx_a = rng.integers(0, m, size=pairs)
    idx_b = rng.integers(0, m, size=pairs)
    mask = idx_a != idx_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]
    if len(idx_a) == 0:
        return {'mean_cosine': 0.0, 'std_cosine': 0.0, 'zero_norm_rate': zero_norm_rate, 'sample_pairs': 0}
    sims = np.sum(normed[idx_a] * normed[idx_b], axis=1)
    return {'mean_cosine': float(np.mean(sims)), 'std_cosine': float(np.std(sims)), 'zero_norm_rate': zero_norm_rate, 'sample_pairs': int(len(sims))}

__all__ = [
    'precision_recall_at_k', 'compute_precision_recall_at_k_for_all', 'mean_reciprocal_rank', 'ndcg_at_k',
    'classification_metrics', 'mean_cosine_similarity'
]
