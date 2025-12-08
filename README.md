NextHorizon: AI-Powered Career Recommendation System
NextHorizon is an end-to-end machine learning pipeline designed to provide personalized career recommendations, skill gap analysis, and learning path suggestions. It leverages natural language processing, embeddings, and retrieval-augmented generation to match users' resumes with relevant job roles and training courses.

Features
Resume Parsing: Extract structured information from resume text using advanced LLMs.
Role Recommendations: Suggest next career roles based on resume analysis and job market data.
Skill Gap Analysis: Identify missing skills and recommend targeted courses.
Course Recommendations: Rank and suggest training courses from various providers.
Data Pipeline: Automated data collection, cleaning, preprocessing, and feature engineering.
Embeddings & Retrieval: Use state-of-the-art embeddings for semantic search and similarity matching.
Evaluation Metrics: Comprehensive evaluation including Precision@k, Recall@k, MRR, NDCG, and embedding health.
Interactive UI: Streamlit-based web interface for easy interaction.
Scalable Architecture: Modular design supporting batch processing and real-time inference.
Table of Contents
Installation
Setup
Usage
Data Pipeline
Configuration
API Reference
Contributing
License
Installation
Prerequisites
Python 3.8+
Node.js 14+ (for PDF generation)
Git
OpenAI API key (for LLM and embedding features)
Clone the Repository
git clone https://github.com/HiranCser/NextHorizon.git
cd NextHorizon
Install Dependencies
Python Dependencies
pip install -r requirements.txt
Node.js Dependencies (for PDF generation)
npm install
Optional: SpaCy Model
For enhanced NLP features:

python -m spacy download en_core_web_sm
Setup
Environment Variables
Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_api_key_here
Data Setup
The project uses pre-built datasets. To generate or update them:

Training Dataset:

python scripts/precompute_features.py --input build_training_dataset/training_database.csv --out-prefix build_training_dataset/training_database
Embeddings:

python scripts/precompute_embeddings.py --input build_training_dataset/training_database.clean.csv --out build_training_dataset/training_database
Job Descriptions:

python build_jd_dataset/build_jd_database.py --roles role_list.csv --out jd_database.csv
Database Setup
No traditional database is required. Data is stored in CSV and NumPy files for simplicity and portability.

Usage
Running the Application
Start the Streamlit app:

streamlit run app.py
Navigate to http://localhost:8501 in your browser.

Key Features Usage
1. Resume Upload and Parsing
Upload your resume (PDF or text).
The system parses it using OpenAI's GPT model to extract skills, experience, and career level.
2. Role Recommendations
Based on your resume, get personalized job role suggestions.
Uses embedding similarity to match your profile with job descriptions.
3. Skill Gap Analysis
Compare your skills against target roles.
Receive recommendations for skills to develop.
4. Course Recommendations
Get ranked course suggestions from various providers.
Courses are matched based on skill gaps and your current level.
Command Line Usage
Precompute Pipeline
# Full pipeline: clean -> features -> embeddings
make precompute

# Individual steps
python scripts/precompute_features.py --input data.csv --out-prefix output_prefix
python scripts/precompute_embeddings.py --input cleaned_data.csv --out output_prefix
Evaluation
# Evaluate retrieval metrics
python scripts/evaluate_retrieval.py --gt ground_truth.json --retrieved retrieved.json --ks 1 5 10 --out reports/eval_report.json
PDF Generation
# Generate documentation PDF
python scripts/md_to_pdf.py
node scripts/html_to_pdf.js
Docker Usage
Build and run with Docker:

docker build -t nexthorizon .
docker run -p 8501:8501 nexthorizon
Data Pipeline
Overview
The ML pipeline consists of several stages:

Data Collection: Gather training data and job descriptions from various sources.
Data Cleaning: Remove duplicates, handle missing values, normalize text.
Preprocessing: Feature engineering, TF-IDF vectorization, NLP enrichment.
Embeddings: Generate dense vector representations using OpenAI embeddings.
Indexing: Build vector indexes for fast retrieval.
Training/Evaluation: Train models and evaluate performance.
Serving: Deploy models for real-time inference.
Data Sources
Training Data: Course catalogs from educational platforms.
Job Descriptions: Scraped from job boards and company career pages.
User Data: Resume text and interaction data.
Feature Engineering
Text normalization and tokenization
TF-IDF vectorization for sparse features
Named Entity Recognition (NER) using spaCy
Readability scores and linguistic features
Categorical encoding for provider and level information
Embeddings
Uses OpenAI's text-embedding-3-small model
Batched processing for efficiency
Cached results to avoid redundant API calls
Configuration
Config Files
config/azure_config.py: Azure-related settings
config/session_config.py: Session management
config/__init__.py: General configuration
Environment Variables
OPENAI_API_KEY: Required for LLM and embedding features
STREAMLIT_SERVER_PORT: Port for Streamlit app (default: 8501)
Makefile Targets
make precompute: Run full preprocessing pipeline
make clean: Remove generated files
make test: Run unit tests
make docker-build: Build Docker image
API Reference
Core Functions
Resume Processing
from utils.resume_processor import process_resume

result = process_resume(resume_text)
Role Recommendations
from ui.role_recommendations import get_role_recommendations

recommendations = get_role_recommendations(resume_data)
Course Recommendations
from ui.course_recommendations import get_course_recommendations

courses = get_course_recommendations(skill_gaps, resume_text)
OpenAI Client
from ai.openai_client import openai_rank_courses, openai_parse_resume

# Rank courses
ranked_courses = openai_rank_courses(skill_gaps, resume_text, course_data)

# Parse resume
parsed_resume = openai_parse_resume(resume_text)
Troubleshooting
Common Issues
OpenAI API Errors: Ensure OPENAI_API_KEY is set correctly.
SpaCy Model Missing: Run python -m spacy download en_core_web_sm.
Memory Issues: For large datasets, increase system memory or use batch processing.
PDF Generation Fails: Ensure Node.js dependencies are installed.
Logs
Check logs in the logs/ directory for detailed error information.

Contributing
Fork the repository
Create a feature branch: git checkout -b feature-name
Make your changes and add tests
Run tests: make test
Commit your changes: git commit -am 'Add feature'
Push to the branch: git push origin feature-name
Submit a pull request
Development Setup
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .
isort .
License
This project is licensed under the MIT License - see the LICENSE file for details.

Support
For questions or issues, please open an issue on GitHub or contact the maintainers.

