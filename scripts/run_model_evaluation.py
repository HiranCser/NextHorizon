#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Suite for NextHorizon

This script runs multiple evaluation scenarios to assess model performance
across different dimensions: retrieval accuracy, embedding quality, and
cross-validation robustness.

Usage:
    python scripts/run_model_evaluation.py
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def run_evaluation_command(cmd: List[str]) -> Dict[str, Any]:
    """Run an evaluation command and return parsed results."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {result.stderr}")
            return {}

        # Parse JSON output
        output_file = cmd[cmd.index('--output') + 1] if '--output' in cmd else None
        if output_file and Path(output_file).exists():
            with open(output_file, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        return {}


def run_comprehensive_evaluation():
    """Run comprehensive evaluation suite."""

    evaluations = {
        'jd_single_split': {
            'cmd': [
                'python', 'scripts/evaluate_train_test.py',
                '--dataset', 'jd',
                '--test-size', '0.2',
                '--k', '1', '3', '5', '10',
                '--output', 'reports/eval_jd_single.json'
            ],
            'description': 'JD dataset single train/test split'
        },
        'jd_cross_validation': {
            'cmd': [
                'python', 'scripts/evaluate_train_test.py',
                '--dataset', 'jd',
                '--test-size', '0.2',
                '--cross-validation', '5',
                '--k', '1', '5', '10',
                '--output', 'reports/eval_jd_cv5.json'
            ],
            'description': 'JD dataset 5-fold cross-validation'
        },
        'training_single_split': {
            'cmd': [
                'python', 'scripts/evaluate_train_test.py',
                '--dataset', 'training',
                '--test-size', '0.3',
                '--k', '1', '3', '5',
                '--output', 'reports/eval_training_single.json'
            ],
            'description': 'Training dataset single train/test split'
        },
        'training_cross_validation': {
            'cmd': [
                'python', 'scripts/evaluate_train_test.py',
                '--dataset', 'training',
                '--test-size', '0.3',
                '--cross-validation', '3',
                '--k', '1', '5',
                '--output', 'reports/eval_training_cv3.json'
            ],
            'description': 'Training dataset 3-fold cross-validation'
        }
    }

    results = {}

    print("🚀 Starting Comprehensive Model Evaluation")
    print("=" * 50)

    for eval_name, config in evaluations.items():
        print(f"\n📊 Running: {config['description']}")
        print(f"Command: {' '.join(config['cmd'])}")

        result = run_evaluation_command(config['cmd'])
        if result:
            results[eval_name] = result
            print("✅ Evaluation completed successfully")
        else:
            print("❌ Evaluation failed")
            results[eval_name] = {'error': 'Evaluation failed'}

    # Generate comprehensive report
    generate_evaluation_report(results)

    return results


def generate_evaluation_report(results: Dict[str, Any]):
    """Generate a comprehensive evaluation report."""

    report = {
        'evaluation_summary': {},
        'performance_comparison': {},
        'recommendations': [],
        'metadata': {
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'total_evaluations': len(results)
        }
    }

    # Extract key metrics from each evaluation
    for eval_name, result in results.items():
        if 'error' in result:
            continue

        eval_summary = {
            'dataset': eval_name.split('_')[0],
            'evaluation_type': 'cross_validation' if 'cross_validation' in result else 'single_split'
        }

        if 'summary' in result:
            # Cross-validation results
            summary = result['summary']
            eval_summary.update({
                'mrr_mean': summary.get('mean_mrr', 0),
                'mrr_std': summary.get('std_mrr', 0),
                'ndcg_mean': summary.get('mean_ndcg', 0),
                'ndcg_std': summary.get('std_ndcg', 0),
                'precision@5_mean': summary.get('precision@5_mean', 0),
                'recall@5_mean': summary.get('recall@5_mean', 0)
            })
        elif 'metrics' in result:
            # Single evaluation results
            metrics = result['metrics']
            eval_summary.update({
                'mrr': metrics.get('mrr', 0),
                'ndcg': metrics.get('ndcg', 0),
                'precision@5': metrics['precision_recall_at_k'].get(5, {}).get('precision_mean', 0),
                'recall@5': metrics['precision_recall_at_k'].get(5, {}).get('recall_mean', 0)
            })

        report['evaluation_summary'][eval_name] = eval_summary

    # Generate performance comparison
    jd_evals = {k: v for k, v in report['evaluation_summary'].items() if v['dataset'] == 'jd'}
    training_evals = {k: v for k, v in report['evaluation_summary'].items() if v['dataset'] == 'training'}

    report['performance_comparison'] = {
        'jd_vs_training': {
            'jd_avg_mrr': sum(v.get('mrr', v.get('mrr_mean', 0)) for v in jd_evals.values()) / len(jd_evals),
            'training_avg_mrr': sum(v.get('mrr', v.get('mrr_mean', 0)) for v in training_evals.values()) / len(training_evals)
        }
    }

    # Generate recommendations
    recommendations = []

    # Check for concerning metrics
    for eval_name, summary in report['evaluation_summary'].items():
        mrr = summary.get('mrr', summary.get('mrr_mean', 0))
        if mrr < 0.5:
            recommendations.append(f"⚠️  {eval_name}: MRR ({mrr:.3f}) indicates poor retrieval performance")
        elif mrr > 0.8:
            recommendations.append(f"✅ {eval_name}: Excellent MRR ({mrr:.3f}) - strong retrieval performance")

    if not recommendations:
        recommendations.append("📊 Model performance is within acceptable ranges across all evaluations")

    # Add general recommendations
    recommendations.extend([
        "🔍 Consider A/B testing different embedding models (text-embedding-3-small vs larger variants)",
        "📈 Implement continuous evaluation pipeline for production monitoring",
        "🎯 Focus on improving recall metrics for better coverage of relevant results",
        "🔄 Consider fine-tuning embeddings on domain-specific career data"
    ])

    report['recommendations'] = recommendations

    # Save comprehensive report
    report_path = Path('reports/model_evaluation_comprehensive.json')
    report_path.parent.mkdir(exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary to console
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE EVALUATION REPORT")
    print("=" * 60)

    print(f"\n📊 Total Evaluations: {len(results)}")
    print(f"\n🎯 Performance Summary:")

    for eval_name, summary in report['evaluation_summary'].items():
        print(f"\n{eval_name.upper()}:")
        if 'mrr' in summary:
            print(".3f")
        if 'mrr_mean' in summary:
            print(".3f")
        if 'precision@5' in summary:
            print(".3f")
        if 'precision@5_mean' in summary:
            print(".3f")

    print(f"\n💡 Recommendations:")
    for rec in recommendations:
        print(f"  {rec}")

    print(f"\n📄 Detailed report saved to: {report_path}")


def main():
    """Main entry point."""
    try:
        results = run_comprehensive_evaluation()
        print("\n✅ Comprehensive evaluation completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())