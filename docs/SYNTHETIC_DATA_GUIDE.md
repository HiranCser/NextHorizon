# 🤖 Synthetic Data Generation for NextHorizon

## Overview

This guide explains how synthetic data helps improve **resume-to-JD matching** - our weakest area.

## 📊 What We Generated

We created **150 training examples** with:
- **50 Resumes** - Realistic user profiles with skills, experience, and education
- **50 JDs** - Realistic job descriptions with requirements and seniority levels  
- **50 Courses** - Training courses matched to skills
- **50 Matching Pairs** - Ground truth data for evaluation

## 🎯 Why This Helps Resume-to-JD Matching

### Current Problem
- **Real JD Database**: Only ~400 jobs
- **Real Resume Data**: Very few examples to learn from
- **Result**: AI model is poorly trained, can't match resumes to jobs well (MRR = 0.043)

### Solution with Synthetic Data
- **Training Examples**: 150 perfect resume-JD pairs
- **Better AI Training**: Model learns what makes a good match
- **Evaluation Data**: Test if our improvements actually work
- **Expected Improvement**: 50% better matching (MRR: 0.043 → 0.065)

## 🚀 Quick Start

### Generate All Synthetic Data
```bash
python scripts/generate_synthetic_data.py --type all --count 100 --output synthetic_data/
```

### Generate Specific Data Types
```bash
# Generate 50 synthetic resumes
python scripts/generate_synthetic_data.py --type resume --count 50 --output synthetic_resumes.csv

# Generate 50 synthetic JDs
python scripts/generate_synthetic_data.py --type jd --count 50 --output synthetic_jds.csv

# Generate 50 synthetic courses
python scripts/generate_synthetic_data.py --type course --count 50 --output synthetic_courses.csv

# Generate matching pairs (for evaluation)
python scripts/generate_synthetic_data.py --type matching_pairs --count 50 --output matching_pairs.json
```

## 📋 Generated Data Structure

### Synthetic Resumes
Each resume contains:
- Professional role (Backend Engineer, Frontend Engineer, ML Engineer, etc.)
- Years of experience (2-15 years)
- Relevant skills (8-10 skills per role type)
- Work experience (realistic job progression)
- Education and certifications
- Full resume text for embedding matching

**Example:**
```
Resume: resume_0001
- Primary Role: Backend Engineer
- Level: Senior
- Experience: 8 years
- Skills: Python, Kubernetes, PostgreSQL, Docker, Node.js, AWS, Microservices, API Design
- Companies: TechCorp, DataSystems, CloudLabs
```

### Synthetic Job Descriptions
Each JD contains:
- Job title and company
- Role type (backend, frontend, ML, data, etc.)
- Seniority level (Junior, Mid, Senior, Staff)
- Required experience (2-10 years)
- Skills required
- Full JD text for embedding matching

**Example:**
```
JD: jd_0001
- Title: Senior Backend Engineer - Amazon
- Role Type: Backend
- Seniority: Senior
- Years Required: 5+
- Skills: Python, Microservices, Docker, Kubernetes, PostgreSQL
```

### Synthetic Courses
Each course contains:
- Title and provider (Coursera, Udemy, edX, etc.)
- Skill taught
- Difficulty level (Beginner, Intermediate, Advanced, Expert)
- Duration (5-100 hours)
- Rating (3.5-5.0)
- Price ($0-$299)

**Example:**
```
Course: course_0001
- Title: Advanced Python Course - Coursera
- Skill: Python
- Level: Advanced
- Hours: 45
- Rating: 4.8/5.0
- Price: $99
```

### Matching Pairs (Ground Truth)
For each resume-JD pair:
- **Matching JD**: Resume matches this job (ground truth = 1)
- **Non-matching JDs**: Resume doesn't match (ground truth = 0)

**Structure:**
```json
{
  "pair_id": "pair_0000",
  "resume": { /* resume object */ },
  "matching_jds": ["jd_0000"],
  "non_matching_jds": ["jd_0050", "jd_0051"],
  "ground_truth": {
    "jd_0000": 1,
    "jd_0050": 0,
    "jd_0051": 0
  }
}
```

## 🔬 Using Synthetic Data for Testing

### Test 1: Evaluate Resume-to-JD Matching
```bash
python scripts/evaluate_train_test.py \
  --dataset synthetic \
  --test-size 0.2 \
  --k 1 5 10 \
  --output reports/synthetic_eval.json
```

### Test 2: Benchmark Improvements
```bash
# Use synthetic matching pairs as ground truth
python scripts/ab_test_experiments.py \
  --experiment embedding_models \
  --test-data synthetic_data/matching_pairs.json
```

## 🛠️ Customization

### Generate More Data
```bash
# Generate 500 resumes instead of 50
python scripts/generate_synthetic_data.py --type resume --count 500 --output synthetic_resumes_large.csv
```

### Change Random Seed (for Different Variants)
```bash
# Generate different data with different seed
python scripts/generate_synthetic_data.py --type all --count 100 --seed 123 --output synthetic_data_v2/
```

### Mix Real and Synthetic Data
```bash
# Combine real JDs with synthetic resumes
cat build_jd_dataset/jd_database.csv synthetic_data/synthetic_jds.csv > combined_jds.csv

# Use for evaluation
python scripts/evaluate_train_test.py --dataset combined_jds.csv
```

## 📊 Performance Expectations

With synthetic training data:
- **Before**: MRR = 0.043 (very poor matching)
- **Expected After**: MRR = 0.065-0.080 (better but still needs work)
- **Role Type Accuracy**: 85%+ correct role suggestions
- **Top-10 Recall**: 70%+ of relevant jobs in top 10

## ⚠️ Important Notes

### Synthetic Data is NOT a Replacement
- Synthetic data helps **test and improve** the system
- Real user feedback is still essential
- Use synthetic data for **rapid iteration** and **testing improvements**
- Validate improvements with **real data** when possible

### Data Diversity
Generated data includes:
- **7 role types**: Backend, Frontend, Full Stack, ML, DevOps, Data, QA
- **7 skill levels**: Junior, Mid, Senior, Staff
- **Experience ranges**: 2-15 years (realistic distribution)
- **Multiple companies**: Real tech companies + synthetic ones

### Reproducibility
- Set `--seed` for reproducible results
- Default seed is 42 (same data each time)
- Different seeds generate different variants

## 🔄 Next Steps

1. **Generate Synthetic Data** ✅
   ```bash
   python scripts/generate_synthetic_data.py --type all --count 100
   ```

2. **Evaluate Current System**
   ```bash
   python scripts/evaluate_train_test.py --dataset synthetic_data/
   ```

3. **Test Improvements**
   ```bash
   python scripts/run_ab_tests.py all
   ```

4. **Monitor Progress**
   - Track MRR improvement from 0.043 → target 0.065+
   - Monitor accuracy of role type matching
   - Check precision@5 for top recommendations

5. **Validate with Real Data**
   - Once synthetic tests show improvement
   - Run evaluation on real user resumes
   - Measure actual business impact (MRR improvement)

## 💾 File Locations

Generated files are stored in `synthetic_data/`:
```
synthetic_data/
├── synthetic_resumes.csv       # 50 synthetic resumes
├── synthetic_jds.csv           # 50 synthetic JDs
├── synthetic_courses.csv       # 50 synthetic courses
└── matching_pairs.json         # 50 resume-JD matching pairs
```

Each CSV can be used with existing evaluation scripts.

---

**Goal**: Use synthetic data to iterate quickly and improve resume-to-JD matching from 0.043 → 0.065+ MRR before testing on real data.
