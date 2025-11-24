# NextHorizon - Complete Code Review & End-to-End Summary

## 🎯 Project Overview

**NextHorizon** is an AI-powered career development platform that helps professionals:
1. **Parse and analyze resumes** using AI
2. **Match profiles to job roles** using vector similarity search
3. **Identify skill gaps** for target roles
4. **Recommend personalized courses** to bridge those gaps

**Tech Stack**: Streamlit (UI) + OpenAI GPT-4o-mini & Embeddings (AI) + Pandas (Data)

---

## 📁 Project Structure

```
NextHorizon/
├── app.py                              # Main Streamlit entry point
├── requirements.txt                    # Dependencies (12 packages)
├── Dockerfile                          # Container build definition
│
├── config/                             # Configuration & Session Management
│   ├── __init__.py
│   └── session_config.py               # Streamlit session state initialization
│
├── ai/                                 # OpenAI Integration Layer
│   ├── __init__.py
│   └── openai_client.py                # Vector embeddings & GPT-4o-mini API calls
│
├── utils/                              # Core Business Logic
│   ├── __init__.py
│   ├── resume_processor.py             # Resume parsing (PDF/DOCX/TXT)
│   ├── skill_analysis.py               # Skill extraction & gap analysis
│   ├── skill_clarification.py          # Interactive Q&A for skill refinement
│   ├── session_helpers.py              # Session state validation helpers
│   └── data_enhancer.py                # Resume data enhancement
│
├── ui/                                 # Streamlit UI Components (4 tabs)
│   ├── __init__.py
│   ├── resume_parsing.py               # Tab 1: Resume upload & parsing
│   ├── role_recommendations.py         # Tab 2: Job role matching
│   ├── skill_gaps.py                   # Tab 3: Gap analysis
│   ├── course_recommendations.py       # Tab 4: Course suggestions
│   └── sidebar.py                      # Database upload sidebar
│
├── build_jd_dataset/                   # Job Description Database
│   ├── build_jd_database.py            # JD database builder script
│   ├── jd_database.csv                 # 294 job descriptions (scraped data)
│   └── role_list.csv                   # 66 career roles
│
└── build_training_dataset/             # Course Database
    ├── build_training_database.py      # Training dataset builder
    ├── skill_list.csv                  # Comprehensive skills list
    └── training_database.csv           # Course recommendations data
```

---

## 🔄 End-to-End User Flow

### **Tab 1: Resume Parsing** 📄

**User Actions:**
1. Upload resume (PDF/DOCX/TXT)
2. Click "Run Extraction & Parsing"
3. Review and edit parsed data
4. Save structured profile

**Backend Flow:**
```python
# 1. File Upload → Text Extraction
resume_file → extract_text_from_file() → raw_text

# 2. Text Cleaning
raw_text → clean_resume_text() → cleaned_text

# 3. AI Parsing (OpenAI GPT-4o-mini)
cleaned_text → openai_parse_resume() → structured_json
{
  "professional_summary": "...",
  "current_role": {"role": "...", "company": "..."},
  "technical_skills": ["Python", "SQL", ...],
  "work_experience": [{...}],
  "education": [{...}],
  "certifications": [...],
  ...
}

# 4. Data Enhancement (optional)
structured_json → backfill_from_text() → enhanced_json

# 5. Store in Session
st.session_state.structured_json = enhanced_json
st.session_state.cleaned_text = cleaned_text
```

**Key Files:**
- `ui/resume_parsing.py` - UI component
- `utils/resume_processor.py` - Processing pipeline
- `ai/openai_client.py` - OpenAI API integration

**Output:** Structured JSON profile stored in session state

---

### **Tab 2: Job Role Recommendation** 🎯

**User Actions:**
1. Enter career aspirations (optional text)
2. Select Top-K roles (slider 1-10)
3. View ranked role recommendations
4. Select a specific role to see top 5 matching JDs
5. Save selected role for skill analysis

**Backend Flow:**
```python
# 1. Build Resume Text
structured_json → build_resume_text() → resume_text
# Includes: summary, current role, skills, experience, education

# 2. Append Aspirations
resume_text += user_aspirations

# 3. Create Role Snippets from JD Database
jd_df.groupby("role_title") → role_snippets
# Each role: {title, snippet (concatenated JDs), link}

# 4. Vector Similarity Matching (OpenAI Embeddings)
openai_rank_roles(resume_text, role_snippets, top_k=5)
↓
text-embedding-3-small model
↓
Cosine similarity scores
↓
Ranked roles with match %

# 5. Detailed JD Matching
Selected role → filter JD database → openai_rank_jds()
↓
Returns top 5 most relevant job descriptions
```

**Key Files:**
- `ui/role_recommendations.py` - UI component
- `ai/openai_client.py` - `openai_rank_roles()`, `openai_rank_jds()`
- `build_jd_dataset/jd_database.csv` - 294 job descriptions

**Data Flow:**
```
Resume Text + Aspirations
        ↓
Vector Embedding (OpenAI)
        ↓
Compare with JD Embeddings
        ↓
Ranked Roles (Top 5-10)
        ↓
Selected Role saved to session
```

**Output:** Selected role title stored in `st.session_state.chosen_role_title`

---

### **Tab 3: Skill Gap Analysis** 🔍

**User Actions:**
1. View required skills for selected role
2. See matched skills (green) vs gaps (red)
3. Answer clarification questions (optional)
4. Save refined skill assessment

**Backend Flow:**
```python
# 1. Validate Prerequisites
if not st.session_state.chosen_role_title:
    show error message
    return

# 2. Extract Required Skills from JD Database
get_required_skills_for_role(role_title, jd_df)
↓
Filter JDs for selected role
↓
Extract skills using regex patterns:
  - Programming languages (python, java, etc.)
  - Frameworks (react, django, etc.)
  - Tools (docker, git, aws, etc.)
  - Databases (sql, mongodb, etc.)
↓
Count frequency across JDs
↓
Return common skills (appear in >20% of JDs)

# 3. Get Candidate Skills
technical_skills from structured_json
+ soft_skills
+ skills from aspirations text

# 4. Calculate Gaps
calculate_skill_gaps(candidate_skills, required_skills)
↓
For each required skill:
  - Check if exact or partial match in candidate skills
  - Add to "matched" or "gaps" list
↓
Return (gaps, matched_skills)

# 5. Generate Clarification Questions (AI-powered)
generate_clarification_questions(structured_json, gaps[:10])
↓
GPT-4o-mini generates 3-5 yes/no questions
↓
User answers → incorporate_clarification_answers()
↓
Updates skill gaps based on answers

# 6. Store Results
st.session_state.skill_gaps = gaps
st.session_state.matched_skills = matched_skills
```

**Key Files:**
- `ui/skill_gaps.py` - UI component
- `utils/skill_analysis.py` - Gap calculation logic
- `utils/skill_clarification.py` - AI Q&A generation
- `utils/session_helpers.py` - Validation helpers

**Skill Extraction Patterns:**
```python
# Regex patterns used:
- Programming: python|java|javascript|c++|go|rust
- Web: react|angular|vue|django|flask|spring
- Cloud: aws|azure|gcp|docker|kubernetes
- Data: pandas|numpy|tensorflow|pytorch|sql
- Tools: git|jenkins|jira|linux|agile
```

**Output:** 
- Matched skills list
- Skill gaps list (stored in session for course recommendations)

---

### **Tab 4: Course Recommendation** 📚

**User Actions:**
1. View identified skill gaps
2. Click "Find Courses"
3. Browse personalized course recommendations per skill
4. Access course links

**Backend Flow:**
```python
# 1. Validate Prerequisites
if not st.session_state.skill_gaps:
    redirect to Tab 3
    return

# 2. Get Skill Gaps
gaps = get_skill_gaps()
# Retrieved from session state (already calculated in Tab 3)

# 3. Load Training Dataset
training_df = get_training_dataframe()
# Columns: training_id, skill, title, description, provider, 
#          hours, price, rating, link

# 4. For Each Skill Gap, Find Relevant Courses
for gap in gaps:
    # Filter training dataset
    relevant_courses = training_df[
        (title contains gap) OR 
        (description contains gap) OR 
        (skill contains gap)
    ]
    
    # Vector similarity ranking
    openai_rank_courses(
        [gap],                    # Single skill gap
        resume_text,              # Full resume context
        relevant_courses,         # Candidate courses
        top_k=5                   # Top 5 courses
    )
    ↓
    text-embedding-3-small model
    ↓
    Embed: (gap + resume_text)
    Embed: Each course description
    ↓
    Cosine similarity
    ↓
    Ranked courses with match %
    
# 5. Display Results
For each skill gap:
  Show top 5 courses with:
    - Title
    - Provider (Coursera, Udemy, etc.)
    - Hours (if available)
    - Price
    - Rating
    - Match percentage
    - Link to course
```

**Key Files:**
- `ui/course_recommendations.py` - UI component
- `ai/openai_client.py` - `openai_rank_courses()`
- `build_training_dataset/training_database.csv` - Course database
- `utils/skill_analysis.py` - Helper functions

**Course Matching Algorithm:**
1. **Filtering**: Find courses mentioning the skill gap
2. **Embedding**: Create vector for (skill_gap + resume_text)
3. **Similarity**: Compare with course description vectors
4. **Ranking**: Sort by cosine similarity score
5. **Top-K**: Return top 5 most relevant courses

**Output:** 
- Personalized course list for each skill gap
- Direct links to course platforms

---

## 🤖 AI Integration Details

### **OpenAI Models Used**

#### 1. **GPT-4o-mini** (Text Generation)
**Used in:**
- Resume parsing (`openai_parse_resume`)
- Skill clarification question generation
- Data enhancement

**Prompt Structure Example:**
```python
prompt = f"""
Extract structured information from this resume:

{resume_text}

Return a JSON object with these fields:
- professional_summary
- current_role (role, company)
- technical_skills (list)
- work_experience (list of dicts)
- education (list of dicts)
...
"""
```

#### 2. **text-embedding-3-small** (Vector Embeddings)
**Used in:**
- Role matching (`openai_rank_roles`)
- JD matching (`openai_rank_jds`)
- Course matching (`openai_rank_courses`)

**How it works:**
```python
# 1. Create embeddings
resume_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=resume_text
).data[0].embedding

course_embeddings = client.embeddings.create(
    model="text-embedding-3-small",
    input=[course1_desc, course2_desc, ...]
).data

# 2. Calculate cosine similarity
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine(resume_embedding, course_emb) for course_emb in course_embeddings]

# 3. Rank by similarity
ranked_courses = sorted(zip(courses, scores), key=lambda x: x[1], reverse=True)
```

**Why Vector Search?**
- Semantic understanding (not just keyword matching)
- Captures context and relevance
- More accurate than traditional search
- Handles synonyms and related concepts

---

## 📊 Database Schema

### **1. Job Description Database** (`jd_database.csv`)
```
Columns (13):
- jd_id: Unique identifier
- role_title: Job role (e.g., "Data Scientist")
- company: Company name
- source_title: Original job posting title
- source_url: Link to job posting
- source_domain: Website domain
- jd_text: Full job description text
- date_scraped: Scraping date
- exp_min_years: Minimum experience
- exp_max_years: Maximum experience
- exp_evidence: Text indicating experience
- seniority_level: Junior/Mid/Senior/Unspecified
- seniority_evidence: Text indicating seniority

Total Records: 294 job descriptions
Unique Roles: 66 roles (from role_list.csv)
```

### **2. Training Database** (`training_database.csv`)
```
Columns (9):
- training_id: Unique identifier
- skill: Skill name (e.g., "python", "java")
- title: Course title
- description: Course description
- provider: Platform (Coursera, Udemy, Codecademy, etc.)
- hours: Course duration
- price: Course price
- rating: User rating
- link: Course URL

Source Platforms:
- Coursera
- Udemy
- Codecademy
- LinkedIn Learning
- Pluralsight
- And more...
```

### **3. Session State Schema** (Streamlit)
```python
st.session_state = {
    # Tab 1: Resume Parsing
    "cleaned_text": str,              # Extracted resume text
    "structured_json": dict,          # Parsed resume data
    "validation_report": str,         # Parsing validation status
    
    # Tab 2: Role Recommendations
    "chosen_role_title": str,         # Selected target role
    "user_aspirations": str,          # Career goals text
    
    # Tab 3: Skill Gap Analysis
    "skill_gaps": list[str],          # Missing skills
    "matched_skills": list[str],      # Skills user has
    
    # Databases
    "jd_df": pd.DataFrame,            # Job description database
    "training_df": pd.DataFrame,      # Course database
}
```

---

## 🔑 Key Technical Features

### **1. Multi-Format Resume Parsing**
```python
# Supports: PDF, DOCX, TXT
def extract_text_from_file(file_path):
    if suffix == ".pdf":
        # Use PyPDF2
    elif suffix == ".docx":
        # Use python-docx or docx2txt
    else:
        # Plain text
```

### **2. Skill Extraction (Regex + AI)**
```python
# Regex patterns for common skills
patterns = [
    r'\b(python|java|javascript)\b',
    r'\b(react|angular|vue)\b',
    r'\b(aws|azure|gcp)\b',
    # ... 50+ patterns
]

# AI enhancement
- Clarification questions (GPT-4o-mini)
- Context understanding from aspirations
```

### **3. Vector Similarity Search**
```python
# OpenAI text-embedding-3-small
# Embedding dimension: 1536
# Similarity metric: Cosine similarity

score = cosine(resume_vector, job_vector)
# Range: -1 to 1 (higher = more similar)
# Display as percentage: score * 100
```

### **4. Session State Management**
```python
# Centralized validation helpers
def validate_role_selected():
    if not st.session_state.get("chosen_role_title"):
        st.warning("Please select a role in Tab 2 first")
        return False
    return True
```

### **5. Error Handling**
```python
try:
    result = openai_parse_resume(text)
except Exception as e:
    st.error(f"Failed: {str(e)}")
    # Graceful degradation
    # User can still edit manually
```

---

## 🚀 Deployment Architecture

### **Current Setup**
- **Local Development**: Streamlit + OpenAI API
- **Environment Variables**: `.env` file for API keys
- **Docker Ready**: Dockerfile provided

### **Recommended Production Setup**
```
┌─────────────────┐
│   User Browser  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Azure App      │
│  Service /      │
│  Streamlit      │
│  Cloud          │
└────────┬────────┘
         │
         ├──────────┐
         │          │
         ▼          ▼
  ┌──────────┐ ┌──────────┐
  │ OpenAI   │ │ Blob     │
  │ API      │ │ Storage  │
  │ (GPT +   │ │ (CSV     │
  │ Embeddings)│ │ files)  │
  └──────────┘ └──────────┘
```

### **Environment Variables Required**
```bash
OPENAI_API_KEY=sk-...
CREWAI_DISABLE_TELEMETRY=true
OTEL_SDK_DISABLED=true
```

---

## 💡 Strengths & Innovations

### ✅ **Strengths**

1. **AI-First Design**: No complex ML models, just reliable APIs
2. **Clean Architecture**: Modular, maintainable code
3. **User-Friendly**: Streamlit provides intuitive interface
4. **Semantic Search**: Vector embeddings > keyword matching
5. **Interactive Refinement**: Clarification Q&A improves accuracy
6. **End-to-End Flow**: Complete career development pipeline
7. **Extensible**: Easy to add new features/data sources

### 🎯 **Innovations**

1. **Hybrid Skill Detection**: Regex patterns + AI clarification
2. **Context-Aware Matching**: Resume + aspirations for role matching
3. **Personalized Course Paths**: Per-skill recommendations
4. **Real-Time Parsing**: No pre-processing, parse on upload
5. **Session-Based Workflow**: Linear flow with validation gates

---

## ⚠️ Current Limitations & Recommendations

### **Limitations**

1. **Static Databases**: 
   - 294 JDs might not cover all roles
   - Training data needs updates
   
2. **No User Accounts**: 
   - Session-based (lost on refresh)
   - No history or saved profiles
   
3. **Single-User**: 
   - No multi-tenancy
   - No collaboration features
   
4. **Cost**: 
   - OpenAI API calls for every operation
   - Can be expensive at scale
   
5. **Limited Error Recovery**:
   - If parsing fails, user must re-upload
   - No partial saves

### **Recommendations for Improvement**

#### **1. Add User Authentication & Persistence**
```python
# Implement:
- Azure AD / Auth0 integration
- Azure Cosmos DB for user profiles
- Session resumption across devices
```

#### **2. Optimize API Costs**
```python
# Strategies:
- Cache embeddings in Cosmos DB
- Batch API calls
- Use cheaper models for simple tasks
- Add rate limiting
```

#### **3. Enhance Data Sources**
```python
# Improvements:
- Real-time job scraping (LinkedIn, Indeed)
- Course API integrations (Coursera, Udemy APIs)
- Regular database updates (monthly)
- User-contributed courses
```

#### **4. Add Analytics Dashboard**
```python
# Track:
- Most recommended roles
- Common skill gaps
- Popular courses
- Conversion metrics (course enrollments)
```

#### **5. Implement Feedback Loop**
```python
# Collect:
- User ratings on recommendations
- Course completion status
- Skill improvement tracking
- Model performance metrics
```

#### **6. Add Azure Cosmos DB Integration**
```python
# Benefits per your instruction file:
- Global distribution
- Low latency
- Elastic scaling
- Perfect for:
  * User profiles (with chat/context)
  * Skill assessments
  * Course progress tracking
  * RAG pattern for recommendations
```

**Example Cosmos DB Schema:**
```json
{
  "id": "user_123",
  "partitionKey": "userId",
  "type": "user_profile",
  "profile": {
    "resume": {...},
    "target_role": "Data Scientist",
    "skill_gaps": ["tensorflow", "pytorch"],
    "enrolled_courses": [...]
  },
  "chat_history": [
    {"role": "user", "message": "I want to be a data scientist"},
    {"role": "assistant", "message": "Based on your profile..."}
  ],
  "recommendations": [...]
}
```

#### **7. Add LLM-Powered Chat Assistant**
```python
# Features:
- Natural language career counseling
- Interactive skill assessment
- Course Q&A
- Progress tracking reminders
- Using GPT-4 with Cosmos DB for context
```

---

## 🔧 Quick Setup & Running

### **Local Development**
```bash
# 1. Clone & navigate
cd ~/NextHorizon

# 2. Create virtual environment
python3 -m venv nh
source nh/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
echo "OPENAI_API_KEY=your_key_here" > .env

# 5. Run application
streamlit run app.py

# Access at: http://localhost:8501
```

### **Docker Deployment**
```bash
# Build
docker build -t nexthorizon:latest .

# Run
docker run -d -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  --name nexthorizon \
  nexthorizon:latest

# Access at: http://localhost:8501
```

---

## 📈 Usage Statistics (Current Data)

- **Job Descriptions**: 294
- **Unique Roles**: 66
- **Training Courses**: ~800+ (estimated from file)
- **Skills Tracked**: 100+ (regex patterns + AI extraction)
- **API Models**: 2 (GPT-4o-mini + text-embedding-3-small)
- **Supported Resume Formats**: 3 (PDF, DOCX, TXT)

---

## 🎓 Code Quality Assessment

### **Architecture: ⭐⭐⭐⭐½ (4.5/5)**
- Clean separation of concerns
- Modular design
- Good naming conventions
- Minor improvement: Add type hints everywhere

### **Error Handling: ⭐⭐⭐½ (3.5/5)**
- Basic try-catch blocks present
- Could use more graceful degradation
- Add retry logic for API calls
- Better user error messages

### **Documentation: ⭐⭐⭐⭐ (4/5)**
- Good README and PROJECT_OVERVIEW
- Code comments present
- Could add API documentation
- Add docstrings for all functions

### **Scalability: ⭐⭐⭐ (3/5)**
- Session-based design limits scale
- No caching mechanism
- Works for MVP/demo
- Needs database for production

### **Security: ⭐⭐⭐ (3/5)**
- API keys in .env (good)
- No authentication (needs improvement)
- No input validation for resume content
- Add rate limiting

---

## 🏁 Conclusion

**NextHorizon** is a well-architected, AI-powered career development platform with:

✅ **Strong Foundation**: Clean code, modular design, AI-first approach  
✅ **Complete Workflow**: Resume → Roles → Gaps → Courses  
✅ **Modern Tech**: OpenAI embeddings, Streamlit, Docker-ready  
✅ **Good UX**: Interactive, step-by-step guidance  

**Ready for**: MVP, Demo, Proof-of-Concept  
**Needs for Production**: User auth, database persistence, cost optimization, monitoring

**Overall Rating**: ⭐⭐⭐⭐ (4/5) - Excellent MVP with clear production path

---

**Report Generated**: November 24, 2025  
**Review Scope**: Complete codebase (24 files)  
**Lines of Code**: ~3,500+ (estimated)
