# NextHorizon Model Evaluation Guide

## Understanding Model Metrics & Train/Test Evaluation

This guide explains how to understand and evaluate the NextHorizon recommendation system's performance using proper train/test splits and comprehensive metrics.

## Current Model Architecture

NextHorizon uses a **retrieval-based recommendation system** rather than traditional supervised learning:

- **Input**: User resumes (text) and job/course databases
- **Processing**: OpenAI text-embedding-3-small for semantic similarity
- **Output**: Ranked recommendations (jobs/courses) by relevance score
- **Evaluation**: Information Retrieval (IR) metrics, not classification accuracy

## Key Metrics Explained

### 1. Mean Reciprocal Rank (MRR)
- **What it measures**: Average position of the first relevant result
- **Formula**: `MRR = (1/Q) × Σ(1/rankᵢ)` where Q = number of queries
- **Range**: 0.0 (worst) to 1.0 (best)
- **Interpretation**:
  - > 0.8: Excellent performance
  - 0.5-0.8: Good performance
  - < 0.5: Needs improvement

### 2. Normalized Discounted Cumulative Gain (NDCG)
- **What it measures**: Ranking quality with position-based discounting
- **Formula**: `NDCG@k = DCG@k / IDCG@k` where DCG considers relevance and position
- **Range**: 0.0 (worst) to 1.0 (best)
- **Interpretation**: Penalizes relevant items appearing lower in rankings

### 3. Precision@K
- **What it measures**: Fraction of top-K results that are relevant
- **Formula**: `Precision@K = (Relevant items in top K) / K`
- **Use case**: Measures accuracy of top recommendations

### 4. Recall@K
- **What it measures**: Fraction of relevant items found in top-K results
- **Formula**: `Recall@K = (Relevant items in top K) / (Total relevant items)`
- **Use case**: Measures coverage/completeness of recommendations

### 5. Embedding Health Metrics
- **Mean Cosine Similarity**: Average similarity between embedding pairs
- **Zero Norm Rate**: Percentage of zero-magnitude embeddings
- **Standard Deviation**: Consistency of embedding similarities

## Current Evaluation Results

Based on our comprehensive evaluation:

```
Dataset: Job Descriptions (JD)
├── MRR: 0.043 (Poor - needs improvement)
├── NDCG: 0.032 (Poor ranking quality)
└── Precision@5: 1.4% (Very low accuracy)

Dataset: Training Courses
├── MRR: 0.870 (Excellent performance!)
├── NDCG: 0.831 (Strong ranking quality)
└── Precision@5: 81.2% (Good accuracy)
```

## Why Train/Test Splits Matter

### Current Limitation
The existing evaluation (`eval_by_skill_with_emb.json`) evaluates on the **entire dataset**, leading to:
- **Overly optimistic metrics** (model sees training data during testing)
- **No generalization assessment** (can't detect overfitting)
- **Limited insight** into real-world performance

### Train/Test Evaluation Benefits
- **Realistic Performance**: Tests on unseen data
- **Generalization Assessment**: Detects overfitting/underfitting
- **Cross-Validation**: Robust performance estimates
- **Model Comparison**: Enables A/B testing of improvements

## How to Run Train/Test Evaluation

### Single Train/Test Split
```bash
# Evaluate JD recommendations
python scripts/evaluate_train_test.py \
  --dataset jd \
  --test-size 0.2 \
  --k 1 5 10 \
  --output reports/eval_jd_split.json

# Evaluate course recommendations
python scripts/evaluate_train_test.py \
  --dataset training \
  --test-size 0.3 \
  --k 1 3 5 \
  --output reports/eval_training_split.json
```

### Cross-Validation
```bash
# 5-fold CV for JD dataset
python scripts/evaluate_train_test.py \
  --dataset jd \
  --cross-validation 5 \
  --test-size 0.2 \
  --output reports/eval_jd_cv5.json

# 3-fold CV for training dataset
python scripts/evaluate_train_test.py \
  --dataset training \
  --cross-validation 3 \
  --output reports/eval_training_cv3.json
```

### Comprehensive Evaluation Suite
```bash
# Run all evaluations automatically
python scripts/run_model_evaluation.py
```

## Interpreting Results

### Example Output Structure
```json
{
  "metrics": {
    "precision_recall_at_k": {
      "1": {"precision_mean": 0.8, "recall_mean": 0.2},
      "5": {"precision_mean": 0.6, "recall_mean": 0.7}
    },
    "mrr": 0.75,
    "ndcg": 0.82,
    "embedding_health": {
      "mean_cosine": 0.445,
      "std_cosine": 0.145,
      "zero_norm_rate": 0.0
    }
  },
  "statistics": {
    "num_queries": 50,
    "avg_relevant_per_query": 3.2,
    "avg_retrieved_per_query": 20
  }
}
```

### Performance Analysis Checklist

#### ✅ Good Performance Indicators
- MRR > 0.7 (strong first-result accuracy)
- NDCG > 0.75 (good ranking quality)
- Precision@5 > 60% (accurate top recommendations)
- Embedding cosine similarity: 0.3-0.6 (reasonable semantic clustering)

#### ⚠️ Warning Signs
- MRR < 0.3 (poor first-result performance)
- High variance in cross-validation folds
- Embedding zero-norm rate > 5%
- Significant drop from training to test performance

#### 🔍 Investigation Areas
1. **Low MRR**: Check if relevant items exist in retrieval corpus
2. **High Variance**: Ensure consistent data preprocessing
3. **Poor Precision**: Review similarity threshold or ranking algorithm
4. **Embedding Issues**: Check for corrupted vectors or normalization problems

## Improving Model Performance

### Immediate Actions
1. **Better Ground Truth**: Create more accurate relevance judgments
2. **Query Expansion**: Use multiple query formulations per user
3. **Hybrid Scoring**: Combine embedding similarity with other features
4. **Result Diversification**: Avoid redundant recommendations

### Advanced Techniques
1. **Fine-tuned Embeddings**: Domain-specific model training
2. **Query Augmentation**: Generate multiple query variations
3. **Re-ranking**: Two-stage retrieval (coarse → fine-grained)
4. **User Feedback Loop**: Incorporate interaction data
5. **text-embedding-3-large**: Upgraded from text-embedding-3-small for superior semantic understanding

### A/B Testing Framework
```bash
# Compare different embedding models
python scripts/evaluate_train_test.py --dataset training --ab-test text-embedding-3-small vs text-embedding-3-large

# Test different similarity thresholds
python scripts/evaluate_train_test.py --dataset jd --threshold 0.7 vs 0.8
```

## A/B Testing for Model Improvements

### Overview
A/B testing enables systematic comparison of different recommendation approaches to identify optimal configurations. The framework supports testing embedding models, similarity methods, and preprocessing techniques.

### Available Experiments

#### 1. Embedding Model Comparison
Compare different OpenAI embedding models for quality vs. performance trade-offs:

```bash
# Test different embedding models
python scripts/run_ab_tests.py embedding_models training
python scripts/run_ab_tests.py embedding_models jd
```

**Models Tested:**
- `text-embedding-3-small`: Fast, cost-effective (1536 dimensions)
- `text-embedding-3-large`: Higher quality but slower (3072 dimensions)
- `text-embedding-ada-002`: Legacy model for comparison

#### 2. Similarity Method Comparison
Test different similarity computation approaches:

```bash
# Test similarity methods
python scripts/run_ab_tests.py similarity_methods training
python scripts/run_ab_tests.py similarity_methods jd
```

**Methods Tested:**
- `cosine`: Standard cosine similarity (recommended)
- `euclidean`: Negative euclidean distance
- `dot_product`: Raw dot product similarity
- `manhattan`: Negative manhattan distance

#### 3. Text Preprocessing Comparison
Evaluate different text preprocessing strategies:

```bash
# Test preprocessing approaches
python scripts/run_ab_tests.py preprocessing training
python scripts/run_ab_tests.py preprocessing jd
```

**Preprocessing Methods:**
- `basic`: Standard cleaning (remove punctuation, normalize whitespace)
- `aggressive`: Remove stop words and short words
- `minimal`: Minimal processing (only lowercase)
- `stemming`: Simple word stemming

### Running All Experiments
```bash
# Run all A/B tests on both datasets
python scripts/run_ab_tests.py all
```

### Interpreting A/B Test Results

#### Example Output Structure
```json
{
  "experiment_name": "Embedding Model Comparison",
  "variants_tested": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
  "metrics_comparison": {
    "mrr": {
      "best_variant": "text-embedding-3-large",
      "best_score": 0.892,
      "all_scores": {
        "text-embedding-3-small": 0.870,
        "text-embedding-3-large": 0.892,
        "text-embedding-ada-002": 0.834
      }
    }
  },
  "recommendations": [
    "🎯 Strong winner: text-embedding-3-large outperforms text-embedding-3-small by 0.022 MRR",
    "⚡ Performance: text-embedding-3-small is significantly faster than text-embedding-3-large"
  ]
}
```

#### Decision Framework
1. **Strong Winner (>0.05 improvement)**: Clear choice, implement immediately
2. **Moderate Improvement (0.02-0.05)**: Consider trade-offs (cost, speed, complexity)
3. **Similar Performance (<0.02)**: Keep current implementation, focus elsewhere
4. **Performance Degradation**: Investigate why, may indicate implementation issues

### Best Practices for A/B Testing

#### Experimental Design
- **Control Variables**: Keep other settings constant when testing one variable
- **Statistical Significance**: Run multiple trials, check for consistent results
- **Realistic Data**: Use production-like data distributions
- **Fair Comparison**: Ensure variants have equal opportunity

#### Implementation Considerations
- **Cost Monitoring**: Track API usage costs for different models
- **Latency Impact**: Measure end-to-end response time changes
- **Resource Usage**: Monitor memory and compute requirements
- **Backward Compatibility**: Ensure changes don't break existing functionality

#### Common Pitfalls
- **Small Sample Size**: Results may not generalize
- **Data Leakage**: Ensure test data is truly unseen
- **Confirmation Bias**: Don't cherry-pick favorable results
- **Over-optimization**: Avoid tuning too specifically to test data

### Recommended Testing Sequence

1. **Start with Embedding Models**: Biggest potential impact on quality
2. **Test Similarity Methods**: Quick wins with minimal cost impact
3. **Try Preprocessing**: Optimize text quality for your domain
4. **Combine Best Approaches**: Test combinations of winning variants

### Production Rollout Strategy

#### Gradual Deployment
1. **Shadow Testing**: Run new variant alongside current system
2. **Percentage Rollout**: Start with 10% of traffic, monitor metrics
3. **Full Rollout**: Scale to 100% if metrics improve
4. **Rollback Plan**: Ability to revert quickly if issues arise

#### Monitoring During Rollout
- **Performance Metrics**: MRR, NDCG, latency, error rates
- **Business Metrics**: User engagement, conversion rates
- **System Health**: Resource usage, API quotas, error logs

## Production Monitoring

### Key Metrics to Track
- **MRR Trends**: Monitor for performance degradation
- **Query Success Rate**: Percentage of queries with relevant results
- **Response Time**: P95 latency for recommendations
- **User Satisfaction**: Click-through rates, conversion metrics

### Automated Evaluation Pipeline
```bash
# Daily evaluation script
#!/bin/bash
python scripts/run_model_evaluation.py
# Send alerts if metrics drop below thresholds
# Update monitoring dashboards
```

## Troubleshooting Common Issues

### "MRR is 0.0"
- **Cause**: No relevant documents found for any query
- **Solution**: Check ground truth creation logic, ensure relevant items exist

### "High variance between folds"
- **Cause**: Small dataset or inconsistent data distribution
- **Solution**: Increase dataset size, use stratified splitting

### "Embedding health shows zero norms"
- **Cause**: Corrupted embedding files or normalization issues
- **Solution**: Regenerate embeddings, check OpenAI API responses

### "Performance drops significantly on test set"
- **Cause**: Overfitting to training data characteristics
- **Solution**: Implement regularization, diversify training data

## Future Evaluation Enhancements

### Advanced Metrics
- **Expected Reciprocal Rank (ERR)**: Considers cascading user behavior
- **Diversity Metrics**: Measures recommendation variety
- **Novelty Scores**: Penalizes popular but obvious recommendations
- **User Satisfaction Modeling**: Incorporate implicit feedback

### Sophisticated Evaluation
- **User Simulation**: A/B testing with simulated user interactions
- **Temporal Validation**: Test on time-ordered data splits
- **Domain Adaptation**: Evaluate performance across different industries
- **Multilingual Assessment**: Cross-language recommendation quality

---

## Quick Reference

| Metric | Good Range | Warning | Critical |
|--------|------------|---------|----------|
| MRR | > 0.7 | 0.3-0.7 | < 0.3 |
| NDCG@10 | > 0.75 | 0.5-0.75 | < 0.5 |
| Precision@5 | > 60% | 30-60% | < 30% |
| Embedding Similarity | 0.3-0.6 | 0.1-0.3 | < 0.1 |

**Run comprehensive evaluation:**
```bash
python scripts/run_model_evaluation.py
```

**View latest results:**
```bash
cat reports/model_evaluation_comprehensive.json | jq '.evaluation_summary'
```