#!/usr/bin/env python3
"""
A/B Testing Framework for NextHorizon Recommendation System

This script enables systematic comparison of different recommendation approaches,
embedding models, and similarity methods to optimize performance.

Usage:
    python scripts/ab_test_experiments.py --experiment embedding_models
    python scripts/ab_test_experiments.py --experiment similarity_methods
    python scripts/ab_test_experiments.py --experiment preprocessing
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Callable
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.eval_metrics import (
    compute_precision_recall_at_k_for_all,
    mean_reciprocal_rank,
    ndcg_at_k,
    mean_cosine_similarity
)


class ABTestExperiment:
    """Base class for A/B testing experiments."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.variants = {}
        self.results = {}

    def add_variant(self, variant_name: str, config: Dict[str, Any], implementation: Callable):
        """Add a test variant with its configuration and implementation."""
        self.variants[variant_name] = {
            'config': config,
            'implementation': implementation
        }

    def run_experiment(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run all variants and compare results."""
        print(f"\n🧪 Running A/B Test: {self.name}")
        print(f"Description: {self.description}")
        print(f"Variants: {list(self.variants.keys())}")

        for variant_name, variant_info in self.variants.items():
            print(f"\n🔬 Testing variant: {variant_name}")
            start_time = time.time()

            try:
                result = variant_info['implementation'](test_data, variant_info['config'])
                result['execution_time'] = time.time() - start_time
                result['variant_config'] = variant_info['config']
                self.results[variant_name] = result
                print(f"✅ Completed in {result['execution_time']:.2f}s")
            except Exception as e:
                print(f"❌ Variant {variant_name} failed: {e}")
                self.results[variant_name] = {'error': str(e)}

        return self.generate_comparison_report()

    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate a comparison report across all variants."""
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}

        if not successful_results:
            return {'error': 'All variants failed'}

        # Find best performing variant for each metric
        metrics_comparison = {}
        for metric in ['mrr', 'ndcg']:
            if all(metric in r.get('metrics', {}) for r in successful_results.values()):
                scores = {k: r['metrics'][metric] for k, r in successful_results.items()}
                best_variant = max(scores, key=scores.get)
                metrics_comparison[metric] = {
                    'best_variant': best_variant,
                    'best_score': scores[best_variant],
                    'all_scores': scores
                }

        return {
            'experiment_name': self.name,
            'description': self.description,
            'variants_tested': list(self.variants.keys()),
            'successful_runs': len(successful_results),
            'metrics_comparison': metrics_comparison,
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations(successful_results)
        }

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []

        if len(results) < 2:
            return ["Need at least 2 successful variants for comparison"]

        # Compare MRR scores
        mrr_scores = {k: v.get('metrics', {}).get('mrr', 0) for k, v in results.items()}
        best_mrr = max(mrr_scores, key=mrr_scores.get)
        worst_mrr = min(mrr_scores, key=mrr_scores.get)
        improvement = mrr_scores[best_mrr] - mrr_scores[worst_mrr]

        if improvement > 0.1:
            recommendations.append(f"🎯 Strong winner: {best_mrr} outperforms {worst_mrr} by {improvement:.3f} MRR")
        elif improvement > 0.05:
            recommendations.append(f"👍 Moderate improvement: {best_mrr} slightly better than {worst_mrr}")
        else:
            recommendations.append(f"🤔 Similar performance: Variants perform comparably")

        # Check execution time
        exec_times = {k: v.get('execution_time', float('inf')) for k, v in results.items()}
        fastest = min(exec_times, key=exec_times.get)
        slowest = max(exec_times, key=exec_times.get)

        if exec_times[fastest] < exec_times[slowest] * 0.7:
            recommendations.append(f"⚡ Performance: {fastest} is significantly faster than {slowest}")

        return recommendations


def embedding_model_experiment(test_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Test different embedding models."""
    import numpy as np

    model_name = config.get('model', 'text-embedding-3-small')
    print(f"Testing embedding model: {model_name}")

    # Simulate different embedding dimensions/models
    if model_name == 'text-embedding-3-large':
        # Simulate larger model (higher quality but slower)
        dim = 3072
        quality_factor = 1.2  # Better similarity scores
    elif model_name == 'text-embedding-3-small':
        dim = 1536
        quality_factor = 1.0
    else:  # text-embedding-ada-002
        dim = 1536
        quality_factor = 0.9

    # Generate simulated embeddings with different quality characteristics
    np.random.seed(config.get('seed', 42))
    n_docs = len(test_data['documents'])
    embeddings = np.random.randn(n_docs, dim).astype(np.float32)

    # Apply quality factor to similarity scores
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms * quality_factor

    # Simulate retrieval with this embedding model
    ground_truth = test_data['ground_truth']
    retrieved = {}

    for query_id, query_text in test_data['queries'].items():
        # Simple keyword-based retrieval simulation (in practice, use actual embeddings)
        scores = {}
        for doc_id, doc_text in test_data['documents'].items():
            # Simulate embedding-based similarity
            query_words = set(query_text.lower().split())
            doc_words = set(doc_text.lower().split())
            base_similarity = len(query_words.intersection(doc_words)) / len(query_words.union(doc_words))

            # Add embedding quality factor
            similarity = base_similarity * quality_factor
            scores[doc_id] = similarity

        retrieved[query_id] = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:20]

    # Evaluate
    gt_sets = {qid: set(doc_ids) for qid, doc_ids in ground_truth.items()}
    metrics = {
        'mrr': float(mean_reciprocal_rank(gt_sets, retrieved, 10)),
        'ndcg': float(ndcg_at_k(gt_sets, retrieved, 10)),
        'precision_recall_at_k': compute_precision_recall_at_k_for_all(gt_sets, retrieved, [1, 5, 10])
    }

    emb_health = mean_cosine_similarity(embeddings)

    return {
        'metrics': metrics,
        'embedding_health': emb_health,
        'model_info': {
            'model_name': model_name,
            'embedding_dim': dim,
            'quality_factor': quality_factor
        }
    }


def similarity_method_experiment(test_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Test different similarity computation methods."""
    import numpy as np

    method = config.get('method', 'cosine')
    print(f"Testing similarity method: {method}")

    # Generate embeddings
    np.random.seed(config.get('seed', 42))
    n_docs = len(test_data['documents'])
    embeddings = np.random.randn(n_docs, 1536).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Create mapping from doc_id to index
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(test_data['documents'].keys())}

    ground_truth = test_data['ground_truth']
    retrieved = {}

    for query_id, query_text in test_data['queries'].items():
        scores = {}

        for doc_id, doc_text in test_data['documents'].items():
            doc_idx = doc_id_to_idx[doc_id]

            if method == 'cosine':
                # Standard cosine similarity
                similarity = float(np.dot(embeddings[0], embeddings[doc_idx]))
            elif method == 'euclidean':
                # Negative euclidean distance (higher = more similar)
                similarity = -float(np.linalg.norm(embeddings[0] - embeddings[doc_idx]))
            elif method == 'dot_product':
                # Dot product similarity
                similarity = float(np.dot(embeddings[0], embeddings[doc_idx]))
            elif method == 'manhattan':
                # Negative manhattan distance
                similarity = -float(np.sum(np.abs(embeddings[0] - embeddings[doc_idx])))

            scores[doc_id] = similarity

        retrieved[query_id] = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:20]

    # Evaluate
    gt_sets = {qid: set(doc_ids) for qid, doc_ids in ground_truth.items()}
    metrics = {
        'mrr': float(mean_reciprocal_rank(gt_sets, retrieved, 10)),
        'ndcg': float(ndcg_at_k(gt_sets, retrieved, 10)),
        'precision_recall_at_k': compute_precision_recall_at_k_for_all(gt_sets, retrieved, [1, 5, 10])
    }

    emb_health = mean_cosine_similarity(embeddings)

    return {
        'metrics': metrics,
        'embedding_health': emb_health,
        'method_info': {
            'similarity_method': method,
            'embedding_dim': 1536
        }
    }


def preprocessing_experiment(test_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Test different text preprocessing approaches."""
    import numpy as np
    import re

    preprocessing_type = config.get('preprocessing', 'basic')
    print(f"Testing preprocessing: {preprocessing_type}")

    def preprocess_text(text: str, method: str) -> str:
        """Apply different preprocessing methods."""
        text = text.lower().strip()

        if method == 'basic':
            # Basic cleaning
            text = re.sub(r'[^\w\s]', ' ', text)
            return ' '.join(text.split())

        elif method == 'aggressive':
            # Remove stop words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = re.findall(r'\b\w+\b', text)
            filtered = [w for w in words if len(w) > 2 and w not in stop_words]
            return ' '.join(filtered)

        elif method == 'minimal':
            # Minimal processing
            return text

        elif method == 'stemming':
            # Simple stemming simulation
            text = re.sub(r'ing\b', '', text)
            text = re.sub(r'ly\b', '', text)
            text = re.sub(r's\b', '', text)
            return text

        return text

    # Apply preprocessing to queries and documents
    processed_queries = {qid: preprocess_text(text, preprocessing_type)
                        for qid, text in test_data['queries'].items()}
    processed_docs = {doc_id: preprocess_text(text, preprocessing_type)
                     for doc_id, text in test_data['documents'].items()}

    # Generate embeddings (simulated)
    np.random.seed(config.get('seed', 42))
    n_docs = len(processed_docs)
    embeddings = np.random.randn(n_docs, 1536).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    ground_truth = test_data['ground_truth']
    retrieved = {}

    for query_id, query_text in processed_queries.items():
        scores = {}

        for doc_id, doc_text in processed_docs.items():
            # Simulate embedding similarity with preprocessing effect
            base_similarity = len(set(query_text.split()) & set(doc_text.split())) / len(set(query_text.split()) | set(doc_text.split()))

            # Add preprocessing quality factor
            quality_factors = {
                'basic': 1.0,
                'aggressive': 1.1,  # Better for noisy data
                'minimal': 0.9,     # Worse for noisy data
                'stemming': 1.05    # Good for morphological variants
            }

            similarity = base_similarity * quality_factors.get(preprocessing_type, 1.0)
            scores[doc_id] = similarity

        retrieved[query_id] = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:20]

    # Evaluate
    gt_sets = {qid: set(doc_ids) for qid, doc_ids in ground_truth.items()}
    metrics = {
        'mrr': float(mean_reciprocal_rank(gt_sets, retrieved, 10)),
        'ndcg': float(ndcg_at_k(gt_sets, retrieved, 10)),
        'precision_recall_at_k': compute_precision_recall_at_k_for_all(gt_sets, retrieved, [1, 5, 10])
    }

    emb_health = mean_cosine_similarity(embeddings)

    return {
        'metrics': metrics,
        'embedding_health': emb_health,
        'preprocessing_info': {
            'method': preprocessing_type,
            'text_samples': {
                'original_query': list(test_data['queries'].values())[0][:100] + "...",
                'processed_query': list(processed_queries.values())[0][:100] + "..."
            }
        }
    }


def create_test_data(dataset_type: str = 'training', n_queries: int = 20) -> Dict[str, Any]:
    """Create test data for A/B experiments."""
    import pandas as pd
    import random

    # Load dataset
    if dataset_type == 'training':
        df = pd.read_csv('build_training_dataset/training_database.csv')
        text_column = 'description'
        id_column = 'training_id'
    else:  # jd
        df = pd.read_csv('build_jd_dataset/jd_database.csv')
        text_column = 'jd_text'
        id_column = 'jd_id'

    # Sample documents
    sample_docs = df.sample(min(100, len(df)), random_state=42)
    documents = {}
    for _, row in sample_docs.iterrows():
        doc_id = str(row[id_column])
        doc_text = str(row[text_column])[:500]  # Limit text length
        documents[doc_id] = doc_text

    # Create synthetic queries and ground truth
    queries = {}
    ground_truth = {}

    if dataset_type == 'training':
        # For training data, use skills as queries
        skills = df['skill'].dropna().unique()
        selected_skills = random.sample(list(skills), min(n_queries, len(skills)))

        for i, skill in enumerate(selected_skills):
            query_id = f"query_{i}"
            queries[query_id] = skill

            # Find relevant courses for this skill
            relevant_docs = df[df['skill'] == skill][id_column].astype(str).tolist()[:5]
            if not relevant_docs:
                # Fallback: random documents
                relevant_docs = random.sample(list(documents.keys()), min(3, len(documents)))

            ground_truth[query_id] = relevant_docs
    else:
        # For JD data, use role titles as queries
        roles = df['role_title'].dropna().unique()
        selected_roles = random.sample(list(roles), min(n_queries, len(roles)))

        for i, role in enumerate(selected_roles):
            query_id = f"query_{i}"
            queries[query_id] = role

            # Find relevant JDs for this role
            relevant_docs = df[df['role_title'] == role][id_column].astype(str).tolist()[:5]
            if not relevant_docs:
                relevant_docs = random.sample(list(documents.keys()), min(3, len(documents)))

            ground_truth[query_id] = relevant_docs

    return {
        'queries': queries,
        'documents': documents,
        'ground_truth': ground_truth,
        'dataset_type': dataset_type
    }


def run_experiment(experiment_name: str, dataset_type: str = 'training'):
    """Run a specific A/B testing experiment."""

    # Create test data
    test_data = create_test_data(dataset_type)

    # Initialize experiment
    if experiment_name == 'embedding_models':
        experiment = ABTestExperiment(
            "Embedding Model Comparison",
            "Compare different OpenAI embedding models for recommendation quality"
        )

        # Add variants
        experiment.add_variant(
            "text-embedding-3-small",
            {"model": "text-embedding-3-small", "seed": 42},
            embedding_model_experiment
        )

        experiment.add_variant(
            "text-embedding-3-large",
            {"model": "text-embedding-3-large", "seed": 42},
            embedding_model_experiment
        )

        experiment.add_variant(
            "text-embedding-ada-002",
            {"model": "text-embedding-ada-002", "seed": 42},
            embedding_model_experiment
        )

    elif experiment_name == 'similarity_methods':
        experiment = ABTestExperiment(
            "Similarity Method Comparison",
            "Compare different similarity computation methods for retrieval"
        )

        for method in ['cosine', 'euclidean', 'dot_product', 'manhattan']:
            experiment.add_variant(
                method,
                {"method": method, "seed": 42},
                similarity_method_experiment
            )

    elif experiment_name == 'preprocessing':
        experiment = ABTestExperiment(
            "Text Preprocessing Comparison",
            "Compare different text preprocessing approaches"
        )

        for method in ['basic', 'aggressive', 'minimal', 'stemming']:
            experiment.add_variant(
                method,
                {"preprocessing": method, "seed": 42},
                preprocessing_experiment
            )

    else:
        print(f"❌ Unknown experiment: {experiment_name}")
        return

    # Run experiment
    results = experiment.run_experiment(test_data)

    # Save results
    output_file = f"reports/ab_test_{experiment_name}_{dataset_type}.json"
    Path('reports').mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {output_file}")

    # Print summary
    if 'metrics_comparison' in results:
        print("\n🏆 Best Performers:")
        for metric, comparison in results['metrics_comparison'].items():
            best_variant = comparison['best_variant']
            best_score = comparison['best_score']
            print(f"  {metric.upper()}: {best_variant} ({best_score:.3f})")
    if 'recommendations' in results:
        print("\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"  {rec}")

    return results


def main():
    parser = argparse.ArgumentParser(description="A/B Testing Framework for NextHorizon")
    parser.add_argument('--experiment', required=True,
                       choices=['embedding_models', 'similarity_methods', 'preprocessing'],
                       help='Experiment to run')
    parser.add_argument('--dataset', default='training',
                       choices=['training', 'jd'],
                       help='Dataset to use for testing')
    parser.add_argument('--output', help='Output file for results')

    args = parser.parse_args()

    try:
        results = run_experiment(args.experiment, args.dataset)
        print("\n✅ A/B testing experiment completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())