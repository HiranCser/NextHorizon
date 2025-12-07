# Resume Database Builder - Quick Start Guide

## 📋 What is the Resume Database Builder?

The Resume Database Builder creates **synthetic resume data** for testing the NextHorizon resume-to-JD matching system. It generates realistic resumes with:
- Different seniority levels (Junior, Mid-Level, Senior, Lead)
- Various technical roles (Software Engineer, Data Scientist, etc.)
- Domain-specific skills
- Experience levels
- Company backgrounds

## 🚀 Quick Start

### 1. Generate Resume Database

```bash
cd /home/rkumar/nextHorizon

# Generate with default settings (3 resumes per level per role)
python build_resume_dataset/build_resume_database.py \
  --roles build_resume_dataset/sample_roles.csv \
  --out build_resume_dataset/resume_database.csv

# Generate with custom settings
python build_resume_dataset/build_resume_database.py \
  --roles build_resume_dataset/sample_roles.csv \
  --out build_resume_dataset/resume_database.csv \
  --levels 5 \
  --per-level 5 \
  --workers 4
```

### 2. Arguments

```
--roles            CSV file with role_title column (required)
--out              Output CSV file (default: resume_database.csv)
--levels           Number of seniority levels to generate (1-5, default: 3)
--per-level        Resumes per level per role (default: 3)
--workers          Parallel workers for faster generation (default: 4)
--http-proxy       HTTP proxy (optional)
--https-proxy      HTTPS proxy (optional)
--verify           SSL verification mode (certifi, system, insecure, path)
--ca-bundle        Path to CA certificate bundle
```

## 📊 Output Format

Generated `resume_database.csv` contains:

```csv
resume_id,role_target,level,years_experience,skills,skills_count,companies_worked,resume_text,source,date_generated
R64C27492,Software Engineer,Junior,2.0,Java; IntelliJ; JIRA; Ruby,4,"Google, Amazon, Microsoft","Software Engineer with 2 years...",synthetic,2025-12-07
```

### Columns Explained

| Column | Description |
|--------|-------------|
| `resume_id` | Unique resume ID (e.g., R64C27492) |
| `role_target` | Target job role |
| `level` | Seniority level (Junior, Mid-Level, Senior, Lead) |
| `years_experience` | Total years of experience |
| `skills` | Comma-separated technical skills |
| `skills_count` | Number of unique skills |
| `companies_worked` | Companies in resume |
| `resume_text` | Full resume text (700+ chars) |
| `source` | Source type (always "synthetic" for generated) |
| `date_generated` | Generation date |

## 🧪 Test the Generator

```bash
# Test resume generation functions
python scripts/test_resume_generation.py
```

This will:
- Generate sample resumes for 4 roles
- Test skill extraction
- Test seniority inference
- Display sample resume output

## 💡 Use Cases

### 1. Training Data for ML Models
```python
import pandas as pd

# Load resume database
resumes = pd.read_csv('build_resume_dataset/resume_database.csv')

# Use for training resume matching models
for idx, resume in resumes.iterrows():
    print(f"Role: {resume['role_target']}")
    print(f"Experience: {resume['years_experience']} years")
    print(f"Skills: {resume['skills']}")
```

### 2. Testing Resume-to-JD Matching
```python
# Test if system can match a Junior Python Developer resume
# to appropriate Job Descriptions

junior_python = resumes[
    (resumes['role_target'].str.contains('Python', case=False)) &
    (resumes['level'] == 'Junior')
].iloc[0]

# Use this resume to test matching against JD database
```

### 3. Evaluation & Benchmarking
```python
# Create test set for evaluation
test_resumes = resumes.sample(frac=0.2, random_state=42)

# Evaluate recommendation quality
for idx, resume in test_resumes.iterrows():
    predictions = recommendation_system.predict(resume)
    # Validate predictions against expected job roles
```

## 🎯 Examples

### Generate Resumes for Specific Roles

```bash
# Create custom roles file
cat > my_roles.csv << EOF
role_title
Python Engineer
Go Developer
Rust Developer
EOF

# Generate database
python build_resume_dataset/build_resume_database.py \
  --roles my_roles.csv \
  --out custom_resumes.csv \
  --levels 3 \
  --per-level 2
```

### Generate Large Dataset

```bash
# Generate 10 levels x 10 resumes per level per role = 1,000 resumes per role
python build_resume_dataset/build_resume_database.py \
  --roles build_resume_dataset/sample_roles.csv \
  --out large_resume_database.csv \
  --levels 5 \
  --per-level 10 \
  --workers 8
```

## 📈 What Gets Generated

For each role and seniority level, the system generates:

1. **Experience Years**: Based on seniority level
   - Junior: 1-3 years
   - Mid-Level: 3-6 years
   - Senior: 6-10 years
   - Lead: 8-15 years

2. **Skills**: Role-specific technical skills
   - Python Developer → Python, Django, Flask, FastAPI, etc.
   - DevOps Engineer → Docker, Kubernetes, AWS, Terraform, etc.
   - Data Scientist → Pandas, NumPy, TensorFlow, PyTorch, etc.

3. **Companies**: Random selection from major tech companies
   - Google, Amazon, Microsoft, Meta, Apple, etc.

4. **Resume Text**: Formatted resume with:
   - Title and experience summary
   - Skills list
   - Work experience timeline
   - Achievements and metrics

## 🔗 Integration with NextHorizon

### Using with Resume Parsing

```python
from utils.resume_processor import process_resume

# Load synthetic resume
resume_db = pd.read_csv('build_resume_dataset/resume_database.csv')
sample_resume = resume_db.iloc[0]

# Process resume
structured = process_resume(sample_resume['resume_text'])

# Use for matching
matches = recommendation_system.match_resume_to_jds(structured)
```

### Using with JD Database

```python
import pandas as pd

# Load both databases
resumes = pd.read_csv('build_resume_dataset/resume_database.csv')
jds = pd.read_csv('build_jd_dataset/jd_database.csv')

# Create training pairs for resume-to-JD matching
for idx, resume in resumes.iterrows():
    matching_jds = jds[
        (jds['exp_min_years'] <= resume['years_experience']) &
        (jds['seniority_level'] == resume['level'])
    ]
    
    # Positive examples: matched JDs
    # Negative examples: non-matched JDs
```

## 📊 Statistics

Default generation creates:
- **Total resumes**: 120 (10 roles × 4 levels × 3 per level)
- **Roles covered**: 10
- **Seniority levels**: 4
- **Skills per resume**: 8
- **Companies per resume**: 2-4

## ✅ Verification

After generation, verify the output:

```bash
# Check file size and row count
wc -l build_resume_dataset/resume_database.csv

# Preview first few rows
head -5 build_resume_dataset/resume_database.csv

# Verify columns
head -1 build_resume_dataset/resume_database.csv | tr ',' '\n' | nl
```

## 🚀 Next Steps

1. **Test Resume-to-JD Matching**: Use generated resumes with existing JD database
2. **Evaluate Recommendation Quality**: Measure MRR/NDCG for matching accuracy
3. **Create Ground Truth Pairs**: Link resumes to appropriate JDs
4. **Train Models**: Use as training data for embedding models
5. **A/B Testing**: Compare different matching algorithms

## 💡 Tips

- **Start small**: Generate a small dataset first to test your pipeline
- **Use parallel workers**: Speed up generation with `--workers 8`
- **Vary seniority levels**: Include all levels for comprehensive testing
- **Create realistic scenarios**: Match resume generation to your use cases

## 🔗 Related Files

- **Resume Parser**: `utils/resume_processor.py`
- **JD Database**: `build_jd_dataset/jd_database.csv`
- **Evaluation Scripts**: `scripts/evaluate_train_test.py`
- **Test Scripts**: `scripts/test_resume_generation.py`
