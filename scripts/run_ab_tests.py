#!/usr/bin/env python3
"""
Quick A/B Testing Runner for NextHorizon

This script provides easy commands to run different A/B testing experiments
and compare recommendation system improvements.

Usage:
    python scripts/run_ab_tests.py embedding_models
    python scripts/run_ab_tests.py similarity_methods
    python scripts/run_ab_tests.py preprocessing
    python scripts/run_ab_tests.py all  # Run all experiments
"""

import subprocess
import sys
import os
from pathlib import Path

def run_experiment(experiment_name: str, dataset: str = 'training'):
    """Run a single A/B testing experiment."""
    print(f"\n🧪 Running A/B Test: {experiment_name} on {dataset} dataset")

    cmd = [
        sys.executable,
        'scripts/ab_test_experiments.py',
        '--experiment', experiment_name,
        '--dataset', dataset
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run experiment: {e}")
        return False

def run_all_experiments():
    """Run all A/B testing experiments."""
    experiments = ['embedding_models', 'similarity_methods', 'preprocessing']
    datasets = ['training', 'jd']

    results = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING ON {dataset.upper()} DATASET")
        print(f"{'='*60}")

        for experiment in experiments:
            success = run_experiment(experiment, dataset)
            results[f"{experiment}_{dataset}"] = success

    print(f"\n{'='*60}")
    print("📊 A/B TESTING SUMMARY")
    print(f"{'='*60}")

    successful = sum(results.values())
    total = len(results)

    print(f"✅ Successful experiments: {successful}/{total}")

    if successful == total:
        print("🎉 All experiments completed successfully!")
    else:
        print("⚠️  Some experiments failed. Check output above.")

    print("\n📄 Results saved in reports/ directory:")
    for key, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {key}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_ab_tests.py <experiment> [dataset]")
        print("\nExperiments:")
        print("  embedding_models  - Compare different embedding models")
        print("  similarity_methods - Compare similarity computation methods")
        print("  preprocessing      - Compare text preprocessing approaches")
        print("  all               - Run all experiments")
        print("\nDatasets:")
        print("  training (default) - Test on training course data")
        print("  jd                 - Test on job description data")
        return 1

    experiment = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else 'training'

    if experiment == 'all':
        run_all_experiments()
    elif experiment in ['embedding_models', 'similarity_methods', 'preprocessing']:
        success = run_experiment(experiment, dataset)
        if success:
            print(f"\n✅ Experiment '{experiment}' completed successfully!")
        else:
            print(f"\n❌ Experiment '{experiment}' failed!")
            return 1
    else:
        print(f"❌ Unknown experiment: {experiment}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())