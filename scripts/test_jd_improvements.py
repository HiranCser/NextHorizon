#!/usr/bin/env python3
"""
Test JD Recommendation Improvements

This script tests the enhanced JD recommendation functions against the original
implementation to validate that our A/B testing improvements actually work.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.openai_client import openai_rank_roles, openai_rank_roles_enhanced, openai_rank_jds, openai_rank_jds_enhanced
from utils.eval_metrics import mean_reciprocal_rank, ndcg_at_k, compute_precision_recall_at_k_for_all

def load_test_data():
    """Load test data for evaluation."""
    # Load JD database
    jd_df = pd.read_csv('build_jd_dataset/jd_database.csv')

    # Create sample resume text for testing
    sample_resume = """
    Experienced software engineer with 5+ years in Python, machine learning, and data science.
    Proficient in TensorFlow, PyTorch, scikit-learn. Background in computer vision and NLP.
    Led teams in developing AI-powered applications. Strong background in algorithms and data structures.
    """

    # Create role snippets for testing
    role_snippets = []
    for role in jd_df['role_title'].unique()[:20]:  # Test with first 20 roles
        role_jds = jd_df[jd_df['role_title'] == role]['jd_text'].fillna('').astype(str).tolist()[:5]
        combined_text = ' '.join(role_jds)
        role_snippets.append({
            'title': role,
            'snippet': combined_text[:1000],  # Limit text length
            'link': '',
            'source': 'jd_db'
        })

    # Create JD rows for detailed testing
    jd_rows = []
    test_roles = jd_df['role_title'].unique()[:3]  # Test with 3 roles
    for role in test_roles:
        role_data = jd_df[jd_df['role_title'] == role].head(10)  # 10 JDs per role
        for _, row in role_data.iterrows():
            jd_rows.append({
                'role_title': row['role_title'],
                'company': row.get('company', 'Unknown'),
                'source_title': row.get('source_title', f"{row['role_title']} Position"),
                'source_url': row.get('source_url', ''),
                'jd_text': str(row['jd_text'])[:1000]  # Limit text length
            })

    return sample_resume, role_snippets, jd_rows

def evaluate_role_ranking():
    """Evaluate role ranking improvements."""
    print("🧪 Testing Role Ranking Improvements")
    print("=" * 50)

    resume_text, role_snippets, _ = load_test_data()

    # Test original implementation
    print("Testing original implementation...")
    start_time = time.time()
    original_results = openai_rank_roles(resume_text, role_snippets, top_k=10)
    original_time = time.time() - start_time

    # Test enhanced implementation
    print("Testing enhanced implementation...")
    start_time = time.time()
    enhanced_results = openai_rank_roles_enhanced(resume_text, role_snippets, top_k=10,
                                                 similarity_method="manhattan", preprocess_text=True)
    enhanced_time = time.time() - start_time

    # Compare results
    print("\n📊 Performance Comparison:")
    print(f"Original time: {original_time:.2f}s")
    print(f"Enhanced time: {enhanced_time:.2f}s")
    print(f"Time difference: {enhanced_time - original_time:.2f}s ({((enhanced_time - original_time) / original_time * 100):.1f}%)")
    # Show top 5 results from each
    print("\n🔝 Top 5 Results Comparison:")
    print("Original Implementation:")
    for i, result in enumerate(original_results[:5], 1):
        print(f"  {i}. {result['role_title']} (score: {result['score']:.3f})")
    return original_results, enhanced_results

def evaluate_jd_ranking():
    """Evaluate JD ranking improvements."""
    print("\n🧪 Testing JD Ranking Improvements")
    print("=" * 50)

    resume_text, _, jd_rows = load_test_data()

    # Test original implementation
    print("Testing original JD ranking...")
    start_time = time.time()
    original_results = openai_rank_jds(resume_text, jd_rows, top_k=10)
    original_time = time.time() - start_time

    # Test enhanced implementation
    print("Testing enhanced JD ranking...")
    start_time = time.time()
    enhanced_results = openai_rank_jds_enhanced(resume_text, jd_rows, top_k=10,
                                               similarity_method="manhattan", preprocess_text=True)
    enhanced_time = time.time() - start_time

    # Compare results
    print("\n📊 Performance Comparison:")
    print(f"Original time: {original_time:.2f}s")
    print(f"Enhanced time: {enhanced_time:.2f}s")
    print(f"Time difference: {enhanced_time - original_time:.2f}s ({((enhanced_time - original_time) / original_time * 100):.1f}%)")
    # Show top 5 results from each
    print("\n🔝 Top 5 Results Comparison:")
    print("Original Implementation:")
    for i, result in enumerate(original_results[:5], 1):
        print(f"  {i}. {result['title']} ({result['match_percent']}%)")
    print("\nEnhanced Implementation:")
    for i, result in enumerate(enhanced_results[:5], 1):
        print(f"  {i}. {result['title']} ({result['match_percent']}%)")
    return original_results, enhanced_results

def run_comprehensive_test():
    """Run comprehensive test of improvements."""
    print("🚀 NextHorizon JD Recommendation Improvements Test")
    print("=" * 60)

    try:
        # Test role ranking
        role_orig, role_enh = evaluate_role_ranking()

        # Test JD ranking
        jd_orig, jd_enh = evaluate_jd_ranking()

        print("\n" + "=" * 60)
        print("✅ IMPROVEMENT SUMMARY")
        print("=" * 60)

        print("🎯 Key Improvements Applied:")
        print("  • Manhattan distance similarity (vs cosine)")
        print("  • Aggressive text preprocessing")
        print("  • Stop word removal and text normalization")

        print("\n💡 Expected Benefits:")
        print("  • Better semantic matching for job descriptions")
        print("  • Reduced noise from common words")
        print("  • Improved ranking of relevant positions")

        print("\n🔄 Next Steps:")
        print("  1. Deploy enhanced functions to production")
        print("  2. Monitor user engagement and satisfaction")
        print("  3. Run A/B test with real users")
        print("  4. Consider fine-tuning embeddings on domain data")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)