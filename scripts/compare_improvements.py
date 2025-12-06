#!/usr/bin/env python3
"""
Compare Old vs New JD Recommendation Performance

This script evaluates the performance improvement from our A/B testing enhancements
by comparing the original implementation vs the enhanced version.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.eval_metrics import mean_reciprocal_rank, ndcg_at_k, compute_precision_recall_at_k_for_all

def create_test_dataset(dataset_type: str = 'jd', n_queries: int = 50) -> Dict[str, Any]:
    """Create a test dataset for evaluation."""
    # Load the appropriate dataset
    if dataset_type == 'jd':
        df = pd.read_csv('build_jd_dataset/jd_database.csv')
        text_col = 'jd_text'
        id_col = 'jd_id'
        group_col = 'role_title'
    else:  # training
        df = pd.read_csv('build_training_dataset/training_database.csv')
        text_col = 'description'
        id_col = 'training_id'
        group_col = 'skill'

    # Sample queries (ground truth)
    if dataset_type == 'jd':
        # For JD, use role titles as queries
        unique_roles = df['role_title'].dropna().unique()
        selected_queries = np.random.choice(unique_roles, min(n_queries, len(unique_roles)), replace=False)
    else:
        # For training, use skills as queries
        unique_skills = df['skill'].dropna().unique()
        selected_queries = np.random.choice(unique_skills, min(n_queries, len(unique_skills)), replace=False)

    # Create ground truth mappings
    ground_truth = {}
    documents = {}

    for query in selected_queries:
        if dataset_type == 'jd':
            relevant_docs = df[df['role_title'] == query][id_col].astype(str).tolist()[:3]  # Top 3 matches
        else:
            relevant_docs = df[df['skill'] == query][id_col].astype(str).tolist()[:3]  # Top 3 matches

        if relevant_docs:
            query_id = f"query_{len(ground_truth)}"
            ground_truth[query_id] = relevant_docs

            # Add query text
            documents[query_id] = query

    # Create document corpus (what we're searching through)
    corpus_docs = {}
    sample_size = min(200, len(df))  # Reasonable corpus size
    sampled_df = df.sample(sample_size, random_state=42)

    for _, row in sampled_df.iterrows():
        doc_id = str(row[id_col])
        doc_text = str(row[text_col])[:500]  # Limit text length
        corpus_docs[doc_id] = doc_text

    return {
        'queries': documents,
        'documents': corpus_docs,
        'ground_truth': ground_truth,
        'dataset_type': dataset_type
    }

def evaluate_implementation(implementation_name: str, rank_function, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a specific implementation."""
    print(f"🔬 Evaluating {implementation_name}...")

    retrieved = {}
    query_texts = list(test_data['queries'].values())

    # Simulate ranking for each query
    for i, (query_id, query_text) in enumerate(test_data['queries'].items()):
        # Create snippets for ranking (simulate the format expected by the functions)
        snippets = []
        for doc_id, doc_text in test_data['documents'].items():
            snippets.append({
                'title': doc_id,
                'snippet': doc_text,
                'link': '',
                'source': 'test_db'
            })

        try:
            # Call the ranking function
            if 'enhanced' in implementation_name.lower():
                ranked_results = rank_function(query_text, snippets, top_k=20,
                                             similarity_method="manhattan", preprocess_text=True)
            else:
                ranked_results = rank_function(query_text, snippets, top_k=20)

            # Extract ranked document IDs
            retrieved[query_id] = [result['title'] for result in ranked_results]

        except Exception as e:
            print(f"❌ Error in {implementation_name}: {e}")
            # Fallback: random ranking
            doc_ids = list(test_data['documents'].keys())
            np.random.shuffle(doc_ids)
            retrieved[query_id] = doc_ids[:20]

    # Calculate metrics
    gt_sets = {qid: set(doc_ids) for qid, doc_ids in test_data['ground_truth'].items()}

    metrics = {
        'mrr': float(mean_reciprocal_rank(gt_sets, retrieved, 10)),
        'ndcg': float(ndcg_at_k(gt_sets, retrieved, 10)),
        'precision_recall_at_k': compute_precision_recall_at_k_for_all(gt_sets, retrieved, [1, 5, 10])
    }

    return {
        'metrics': metrics,
        'retrieved': retrieved,
        'num_queries': len(retrieved),
        'avg_relevant_per_query': np.mean([len(gt) for gt in gt_sets.values()]),
        'avg_retrieved_per_query': np.mean([len(ret) for ret in retrieved.values()])
    }

def run_comparison_test():
    """Run comprehensive comparison between old and new implementations."""
    print("🚀 JD Recommendation Improvement Comparison Test")
    print("=" * 60)

    # Create test datasets
    jd_test_data = create_test_dataset('jd', n_queries=30)
    training_test_data = create_test_dataset('training', n_queries=30)

    print(f"📊 Test Datasets Created:")
    print(f"  JD Dataset: {len(jd_test_data['queries'])} queries, {len(jd_test_data['documents'])} documents")
    print(f"  Training Dataset: {len(training_test_data['queries'])} queries, {len(training_test_data['documents'])} documents")

    # Import the functions we want to compare
    try:
        from ai.openai_client import openai_rank_roles, openai_rank_roles_enhanced
        from ai.openai_client import openai_rank_jds, openai_rank_jds_enhanced
    except ImportError as e:
        print(f"❌ Failed to import functions: {e}")
        return False

    results = {}

    # Test configurations
    test_configs = [
        ('JD Roles - Original', openai_rank_roles, jd_test_data),
        ('JD Roles - Enhanced', openai_rank_roles_enhanced, jd_test_data),
        ('JD Details - Original', openai_rank_jds, jd_test_data),
        ('JD Details - Enhanced', openai_rank_jds_enhanced, jd_test_data),
        ('Training - Original', openai_rank_roles, training_test_data),
        ('Training - Enhanced', openai_rank_roles_enhanced, training_test_data),
    ]

    for test_name, rank_func, test_data in test_configs:
        try:
            result = evaluate_implementation(test_name, rank_func, test_data)
            results[test_name] = result
            print(f"✅ {test_name}: MRR={result['metrics']['mrr']:.3f}, NDCG={result['metrics']['ndcg']:.3f}")
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results[test_name] = {'error': str(e)}

    # Analyze and display results
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)

    # Group results by dataset
    jd_results = {k: v for k, v in results.items() if 'JD' in k and 'error' not in v}
    training_results = {k: v for k, v in results.items() if 'Training' in k and 'error' not in v}

    def display_comparison(dataset_name: str, results_dict: Dict[str, Any]):
        print(f"\n🎯 {dataset_name} Dataset:")

        if not results_dict:
            print("  No valid results to compare")
            return

        # Extract metrics
        metrics_data = []
        for test_name, result in results_dict.items():
            if 'error' not in result:
                metrics = result['metrics']
                is_enhanced = 'Enhanced' in test_name
                metrics_data.append({
                    'name': test_name,
                    'enhanced': is_enhanced,
                    'mrr': metrics['mrr'],
                    'ndcg': metrics['ndcg'],
                    'precision@5': metrics['precision_recall_at_k']['5']['precision_mean']
                })

        if not metrics_data:
            print("  No metrics data available")
            return

        # Display comparison table
        print("  | Implementation | MRR | NDCG | Precision@5 |")
        print("  |----------------|-----|------|-------------|")

        for data in metrics_data:
            enhanced_marker = "🆕" if data['enhanced'] else "📊"
            print(f"  | {enhanced_marker} {data['name'].replace(dataset_name + ' ', '')} | {data['mrr']:.3f} | {data['ndcg']:.3f} | {data['precision@5']:.1%} |")

        # Calculate improvements
        if len(metrics_data) >= 2:
            original = next((d for d in metrics_data if not d['enhanced']), None)
            enhanced = next((d for d in metrics_data if d['enhanced']), None)

            if original and enhanced:
                mrr_improvement = enhanced['mrr'] - original['mrr']
                ndcg_improvement = enhanced['ndcg'] - original['ndcg']
                prec_improvement = enhanced['precision@5'] - original['precision@5']

                print("\n  📈 Improvements:")
                print(f"    • MRR: {mrr_improvement:+.1%} ({'📈' if mrr_improvement > 0 else '📉'} {abs(mrr_improvement)*100:.1f}%)")
                print(f"    • NDCG: {ndcg_improvement:+.1%} ({'📈' if ndcg_improvement > 0 else '📉'} {abs(ndcg_improvement)*100:.1f}%)")
                print(f"    • Precision@5: {prec_improvement:+.1%} ({'📈' if prec_improvement > 0 else '📉'} {abs(prec_improvement)*100:.1f}%)")
    # Display results by dataset
    display_comparison("JD Roles", {k: v for k, v in jd_results.items() if 'Roles' in k})
    display_comparison("JD Details", {k: v for k, v in jd_results.items() if 'Details' in k})
    display_comparison("Training", training_results)

    # Overall summary
    print("\n" + "=" * 60)
    print("🏆 OVERALL SUMMARY")
    print("=" * 60)

    # Calculate average improvements
    improvements = []
    for dataset_name, results_dict in [("JD", jd_results), ("Training", training_results)]:
        dataset_improvements = []
        for test_name, result in results_dict.items():
            if 'Enhanced' in test_name and 'error' not in result:
                base_name = test_name.replace(' - Enhanced', ' - Original')
                if base_name in results_dict and 'error' not in results_dict[base_name]:
                    orig_mrr = results_dict[base_name]['metrics']['mrr']
                    enh_mrr = result['metrics']['mrr']
                    improvement = enh_mrr - orig_mrr
                    dataset_improvements.append(improvement)

        if dataset_improvements:
            avg_improvement = np.mean(dataset_improvements)
            improvements.append((dataset_name, avg_improvement))

    if improvements:
        print("🎯 Average MRR Improvements:")
        for dataset, improvement in improvements:
            status = "📈 IMPROVED" if improvement > 0 else "📉 DECLINED" if improvement < 0 else "➡️ NO CHANGE"
            print(f"  {dataset}: {improvement:+.1%} {status}")
    # Save detailed results
    output_file = "reports/improvement_comparison.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for k, v in results.items():
            if 'error' not in v:
                json_results[k] = {
                    'metrics': {
                        'mrr': float(v['metrics']['mrr']),
                        'ndcg': float(v['metrics']['ndcg']),
                        'precision_recall_at_k': {
                            str(k_val): {
                                'precision_mean': float(data['precision_mean']),
                                'recall_mean': float(data['recall_mean'])
                            }
                            for k_val, data in v['metrics']['precision_recall_at_k'].items()
                        }
                    },
                    'num_queries': v['num_queries'],
                    'avg_relevant_per_query': float(v['avg_relevant_per_query']),
                    'avg_retrieved_per_query': float(v['avg_retrieved_per_query'])
                }
            else:
                json_results[k] = v

        json.dump(json_results, f, indent=2)

    print(f"\n📄 Detailed results saved to: {output_file}")

    return True

if __name__ == "__main__":
    try:
        success = run_comparison_test()
        if success:
            print("\n✅ Comparison test completed successfully!")
        else:
            print("\n❌ Comparison test failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)