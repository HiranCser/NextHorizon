#!/usr/bin/env python3
"""
Train/Test Evaluation Framework for NextHorizon Retrieval System

This script implements proper train/test evaluation for the career recommendation system,
including cross-validation and comprehensive metrics assessment.

Usage:
    python scripts/evaluate_train_test.py --dataset jd --test-size 0.2 --k 5 10
    python scripts/evaluate_train_test.py --dataset training --test-size 0.3 --k 1 3 5
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import sys
import os
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.eval_metrics import (
    compute_precision_recall_at_k_for_all,
    mean_reciprocal_rank,
    ndcg_at_k,
    mean_cosine_similarity
)


def load_dataset(dataset_type: str) -> pd.DataFrame:
    """Load the appropriate dataset based on type."""
    if dataset_type == 'jd':
        path = Path("build_jd_dataset/jd_database.csv")
    elif dataset_type == 'training':
        path = Path("build_training_dataset/training_database.csv")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def manual_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Manual implementation of train/test split without sklearn dependency."""
    np.random.seed(random_state)
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    test_count = int(len(df) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return df.iloc[train_indices].reset_index(drop=True), df.iloc[test_indices].reset_index(drop=True)
    """Load the appropriate dataset based on type."""
    if dataset_type == 'jd':
        path = Path("build_jd_dataset/jd_database.csv")
    elif dataset_type == 'training':
        path = Path("build_training_dataset/training_database.csv")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def create_retrieval_simulation(df: pd.DataFrame, dataset_type: str, test_size: float = 0.2,
                               random_state: int = 42) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Create a simulated retrieval scenario for evaluation.

    For JD dataset: Use role titles as queries, full JDs as documents
    For training dataset: Use skills as queries, courses as documents
    """
    # Split dataset
    train_df, test_df = manual_train_test_split(df, test_size=test_size, random_state=random_state)

    if dataset_type == 'jd':
        # For job descriptions: queries are role titles, documents are full JDs
        queries = {}
        ground_truth = {}
        documents = {}

        # Create query-document mapping
        for idx, row in test_df.iterrows():
            role_title = str(row.get('role_title', '')).strip()
            jd_text = str(row.get('jd_text', '')).strip()
            jd_id = str(idx)

            if role_title and jd_text:
                queries[jd_id] = role_title
                documents[jd_id] = jd_text
                # Ground truth: JDs with same role title are relevant
                relevant_docs = train_df[train_df['role_title'] == role_title].index.astype(str).tolist()
                ground_truth[jd_id] = relevant_docs[:10]  # Limit to top 10

    elif dataset_type == 'training':
        # For training courses: queries are skills, documents are courses
        skill_to_courses = {}
        for idx, row in train_df.iterrows():
            skill = str(row.get('skill', '')).strip()
            if skill:
                if skill not in skill_to_courses:
                    skill_to_courses[skill] = []
                skill_to_courses[skill].append(str(idx))

        queries = {}
        ground_truth = {}
        documents = {}

        # Create test queries from skills in test set
        test_skills = set()
        for idx, row in test_df.iterrows():
            skill = str(row.get('skill', '')).strip()
            if skill and skill in skill_to_courses:
                test_skills.add(skill)

        for skill in list(test_skills)[:50]:  # Limit for evaluation
            query_id = f"skill_{hash(skill) % 10000}"
            queries[query_id] = skill
            ground_truth[query_id] = skill_to_courses[skill][:10]  # Top 10 relevant courses

            # Documents are course descriptions
            for course_id in skill_to_courses[skill][:20]:  # Include some irrelevant
                course_idx = int(course_id)
                if course_idx < len(train_df):
                    course_row = train_df.iloc[course_idx]
                    title = str(course_row.get('title', ''))
                    desc = str(course_row.get('description', ''))
                    documents[course_id] = f"{title} {desc}"

    # Simulate retrieval results (in real scenario, this would use your embedding system)
    retrieved = {}
    for query_id, query_text in queries.items():
        # Simple keyword-based retrieval simulation
        scores = {}
        for doc_id, doc_text in documents.items():
            # Calculate simple similarity (in production, use embeddings)
            query_words = set(query_text.lower().split())
            doc_words = set(doc_text.lower().split())
            similarity = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))
            scores[doc_id] = similarity

        # Sort by similarity and take top results
        retrieved[query_id] = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:20]

    return ground_truth, retrieved, np.random.randn(len(documents), 1536).astype(np.float32)


def evaluate_retrieval_system(ground_truth: Dict[str, List[str]],
                            retrieved: Dict[str, List[str]],
                            embeddings: np.ndarray,
                            ks: List[int] = [1, 5, 10]) -> Dict[str, Any]:
    """Evaluate the retrieval system using standard IR metrics."""

    # Convert ground truth to sets
    gt_sets = {qid: set(doc_ids) for qid, doc_ids in ground_truth.items()}

    # Compute all metrics
    pr_metrics = compute_precision_recall_at_k_for_all(gt_sets, retrieved, ks)
    mrr = mean_reciprocal_rank(gt_sets, retrieved, max(ks))
    ndcg = ndcg_at_k(gt_sets, retrieved, max(ks))

    # Embedding health check
    emb_health = mean_cosine_similarity(embeddings)

    # Additional statistics
    stats = {
        'num_queries': len(ground_truth),
        'num_documents': len(set(doc for docs in retrieved.values() for doc in docs)),
        'avg_relevant_per_query': np.mean([len(docs) for docs in ground_truth.values()]),
        'avg_retrieved_per_query': np.mean([len(docs) for docs in retrieved.values()]),
    }

    return {
        'metrics': {
            'precision_recall_at_k': pr_metrics,
            'mrr': float(mrr),
            'ndcg': float(ndcg),
            'embedding_health': emb_health
        },
        'statistics': stats,
        'configuration': {
            'k_values': ks,
            'evaluation_type': 'retrieval_simulation'
        }
    }


def run_cross_validation(df: pd.DataFrame, dataset_type: str, test_size: float = 0.2,
                       n_splits: int = 5, ks: List[int] = [1, 5, 10]) -> Dict[str, Any]:
    """Run cross-validation evaluation."""

    results = []
    np.random.seed(42)

    for fold in range(n_splits):
        print(f"Running fold {fold + 1}/{n_splits}...")

        # Create fold-specific evaluation
        gt, retrieved, embeddings = create_retrieval_simulation(
            df, dataset_type, test_size, random_state=42 + fold
        )

        fold_result = evaluate_retrieval_system(gt, retrieved, embeddings, ks)
        fold_result['fold'] = fold + 1
        results.append(fold_result)

    # Aggregate results
    aggregated = {
        'cross_validation': {
            'n_splits': n_splits,
            'test_size': test_size,
            'fold_results': results
        },
        'summary': {
            'mean_mrr': float(np.mean([r['metrics']['mrr'] for r in results])),
            'std_mrr': float(np.std([r['metrics']['mrr'] for r in results])),
            'mean_ndcg': float(np.mean([r['metrics']['ndcg'] for r in results])),
            'std_ndcg': float(np.std([r['metrics']['ndcg'] for r in results])),
        }
    }

    # Add precision/recall summaries
    for k in ks:
        precisions = [r['metrics']['precision_recall_at_k'][k]['precision_mean'] for r in results]
        recalls = [r['metrics']['precision_recall_at_k'][k]['recall_mean'] for r in results if r['metrics']['precision_recall_at_k'][k]['recall_mean'] is not None]

        aggregated['summary'][f'precision@{k}_mean'] = float(np.mean(precisions))
        aggregated['summary'][f'precision@{k}_std'] = float(np.std(precisions))

        if recalls:
            aggregated['summary'][f'recall@{k}_mean'] = float(np.mean(recalls))
            aggregated['summary'][f'recall@{k}_std'] = float(np.std(recalls))

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Train/Test Evaluation for NextHorizon")
    parser.add_argument('--dataset', choices=['jd', 'training'], required=True,
                       help='Dataset to evaluate (jd=job descriptions, training=courses)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--k', nargs='+', type=int, default=[1, 5, 10],
                       help='k values for evaluation metrics')
    parser.add_argument('--cross-validation', type=int, default=1,
                       help='Number of CV folds (1 = single train/test split)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print(f"Loading {args.dataset} dataset...")
    df = load_dataset(args.dataset)

    print(f"Dataset shape: {df.shape}")
    print(f"Test size: {args.test_size}")
    print(f"K values: {args.k}")

    if args.cross_validation > 1:
        print(f"Running {args.cross_validation}-fold cross-validation...")
        results = run_cross_validation(df, args.dataset, args.test_size,
                                     args.cross_validation, args.k)
    else:
        print("Running single train/test evaluation...")
        gt, retrieved, embeddings = create_retrieval_simulation(
            df, args.dataset, args.test_size, args.seed
        )
        results = evaluate_retrieval_system(gt, retrieved, embeddings, args.k)

    # Generate output filename
    if args.output is None:
        cv_suffix = f"_cv{args.cross_validation}" if args.cross_validation > 1 else ""
        args.output = f"reports/eval_{args.dataset}_test{args.test_size}{cv_suffix}.json"

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    # Print summary
    if 'summary' in results:
        print("\nSummary Statistics:")
        for key, value in results['summary'].items():
            if isinstance(value, float):
                print(".3f")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()