# FILE: build_resume_dataset/build_resume_database.py

"""
Resume Database Builder - Synthetic Resume Generator

This script creates a synthetic resume database by scraping resume examples
from GitHub, portfolio sites, and resume template repositories using DuckDuckGo.

Features:
- Searches for resume examples and GitHub profiles
- Extracts resume information (skills, experience, projects)
- Generates synthetic resumes from templates
- Creates ground truth resume-JD pairs for evaluation
- Parallel processing for efficient data collection

Install:
    pip install ddgs beautifulsoup4 tldextract html5lib requests pandas certifi

Usage:
    python build_resume_dataset.py --roles role_list.csv --out resume_database.csv --per_role 5
"""

from __future__ import annotations
import argparse, csv, os, re, time, uuid, tldextract, html, logging, sys
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

try:
    import certifi
    CERTIFI_PATH = certifi.where()
except Exception:
    CERTIFI_PATH = None

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# GitHub and portfolio domains to search
RESUME_DOMAINS = [
    "github.com",
    "gitlab.com", 
    "linkedin.com",
    "behance.net",
    "dribbble.com",
    "portfolio",
    "resume",
]

# Common skills to include in synthetic resumes
SKILL_CATEGORIES = {
    "Programming Languages": [
        "Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "Ruby", "Kotlin"
    ],
    "Web Technologies": [
        "React", "Angular", "Vue.js", "Node.js", "Express", "Django", "Flask", "Spring Boot",
        "HTML", "CSS", "REST APIs", "GraphQL", "Webpack"
    ],
    "Data & AI": [
        "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Jupyter", 
        "SQL", "MongoDB", "PostgreSQL", "Data Analysis", "Machine Learning"
    ],
    "DevOps & Cloud": [
        "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Jenkins", "GitLab CI/CD",
        "Terraform", "Ansible", "Linux", "Shell Scripting"
    ],
    "Databases": [
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "DynamoDB", "Firebase"
    ],
    "Tools & Platforms": [
        "Git", "GitHub", "JIRA", "Confluence", "Slack", "Figma", "VS Code", "IntelliJ"
    ]
}

COMPANIES = [
    "Google", "Amazon", "Microsoft", "Apple", "Meta", "Tesla", "Netflix", "Uber",
    "Airbnb", "Stripe", "Spotify", "Adobe", "Salesforce", "IBM", "Oracle",
    "GitHub", "GitLab", "HashiCorp", "Elastic", "MongoDB", "DataDog"
]

ROLES_EXPERIENCE = {
    "Junior": (1, 3),
    "Mid-Level": (3, 6),
    "Senior": (6, 10),
    "Lead": (8, 15),
    "Principal": (12, 20)
}

# -------- Search backends --------
def _import_ddg():
    """Import DuckDuckGo search library"""
    try:
        from ddgs import DDGS
        return "ddgs", DDGS
    except Exception:
        pass
    try:
        from duckduckgo_search import DDGS
        return "duckduckgo_search", DDGS
    except Exception:
        return None, None

def search_ddg(query: str, k: int = 15) -> List[Dict[str, Any]]:
    """Search using DuckDuckGo"""
    pkg, DDGS = _import_ddg()
    if DDGS is None:
        logging.warning("ddgs/duckduckgo_search not installed; ddg search disabled.")
        return []
    
    results = []
    try:
        with DDGS() as ddgs:
            kwargs = dict(max_results=int(k))
            try:
                it = ddgs.text(query, **kwargs)
            except TypeError:
                it = ddgs.text(keywords=query, **kwargs)
            for x in it:
                results.append({
                    "title": (x.get("title") or ""),
                    "link": (x.get("href") or x.get("url") or ""),
                    "snippet": (x.get("body") or ""),
                    "source": pkg
                })
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.warning(f"ddg search error: {e}")
        return []
    return results

def search_resumes(query: str, k: int = 15) -> List[Dict[str, Any]]:
    """Search for resume examples and GitHub profiles"""
    return search_ddg(query, k)

# -------- HTTP helpers --------
def make_session(http_proxy: Optional[str], https_proxy: Optional[str], verify_mode: str, ca_bundle: Optional[str]) -> requests.Session:
    """Create HTTP session with retries and configuration"""
    s = requests.Session()
    retry = Retry(
        total=2, read=2, connect=2,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD","GET","OPTIONS"]
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))

    if http_proxy or https_proxy:
        s.proxies.update({k:v for k,v in {"http":http_proxy, "https":https_proxy}.items() if v})

    if verify_mode == "insecure":
        s.verify = False
    elif verify_mode == "certifi":
        s.verify = CERTIFI_PATH
    elif verify_mode == "system":
        s.verify = True
    elif verify_mode == "path":
        s.verify = ca_bundle
    else:
        s.verify = CERTIFI_PATH

    s.headers.update({"User-Agent": UA})
    return s

def fetch(session: requests.Session, url: str, connect_timeout: float, read_timeout: float) -> str:
    """Fetch page content"""
    try:
        r = session.get(url, timeout=(float(connect_timeout), float(read_timeout)))
        if r.status_code != 200:
            return ""
        return r.text
    except Exception as e:
        logging.debug(f"Request failed for {url}: {e}")
        return ""

# -------- Parsing helpers --------
def visible_text_from_html(html_text: str, limit_chars: int = 8000) -> str:
    """Extract visible text from HTML"""
    soup = BeautifulSoup(html_text or "", "html5lib")
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:limit_chars]

def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from resume text"""
    skills_found = set()
    text_lower = text.lower()
    
    for category, skill_list in SKILL_CATEGORIES.items():
        for skill in skill_list:
            if skill.lower() in text_lower:
                skills_found.add(skill)
    
    return sorted(list(skills_found))

def extract_experience_years(text: str) -> Tuple[Optional[float], Optional[float], str]:
    """Extract experience from resume text"""
    patterns = [
        (r"(\d{1,2})\s*(?:-|to)\s*(\d{1,2})\s*years?", True),
        (r"(\d{1,2})\s*\+\s*years?", False),
        (r"(\d{1,2})\s*years?(?:\s+of)?(?:\s+experience)?", False),
    ]
    
    for pattern, is_range in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if is_range and len(match.groups()) >= 2:
                min_exp = float(match.group(1))
                max_exp = float(match.group(2))
                return min_exp, max_exp, match.group(0)
            else:
                exp = float(match.group(1))
                return exp, exp, match.group(0)
    
    return None, None, ""

def extract_companies_from_text(text: str) -> List[str]:
    """Extract company names from resume text"""
    companies_found = []
    text_lower = text.lower()
    
    for company in COMPANIES:
        if company.lower() in text_lower:
            companies_found.append(company)
    
    return companies_found[:3]  # Top 3 companies

def infer_seniority_from_experience(min_years: Optional[float], max_years: Optional[float]) -> str:
    """Infer seniority level from years of experience"""
    if min_years is None:
        return "Unspecified"
    
    years = min_years
    
    for level, (level_min, level_max) in ROLES_EXPERIENCE.items():
        if level_min <= years <= level_max:
            return level
    
    if years > 20:
        return "Principal"
    elif years > 15:
        return "Lead"
    elif years > 10:
        return "Senior"
    else:
        return "Mid-Level"

def generate_synthetic_resume(role: str, level: str, num_resumes: int = 1) -> List[Dict[str, Any]]:
    """Generate synthetic resumes for testing"""
    import random
    
    resumes = []
    
    for i in range(num_resumes):
        # Get experience range for level
        min_exp, max_exp = ROLES_EXPERIENCE.get(level, (1, 3))
        years = random.randint(min_exp, max_exp)
        
        # Select skills based on role
        role_lower = role.lower()
        selected_skills = []
        
        # Add role-specific skills
        if "python" in role_lower or "data" in role_lower:
            selected_skills.extend(SKILL_CATEGORIES["Data & AI"][:3])
        if "frontend" in role_lower or "ui" in role_lower:
            selected_skills.extend(SKILL_CATEGORIES["Web Technologies"][:3])
        if "backend" in role_lower or "api" in role_lower:
            selected_skills.extend(SKILL_CATEGORIES["Web Technologies"][3:6])
        if "devops" in role_lower or "cloud" in role_lower:
            selected_skills.extend(SKILL_CATEGORIES["DevOps & Cloud"][:3])
        
        # Add general skills
        selected_skills.extend(random.sample(SKILL_CATEGORIES["Programming Languages"], 2))
        selected_skills.extend(random.sample(SKILL_CATEGORIES["Tools & Platforms"], 2))
        
        selected_skills = list(set(selected_skills))[:8]  # Unique, max 8
        
        # Select companies
        num_companies = random.randint(2, 4)
        worked_companies = random.sample(COMPANIES, min(num_companies, len(COMPANIES)))
        
        # Build experience section
        experience_section = f"- {level} {role} at {worked_companies[0]} ({random.randint(1, 4)} years)\n"
        if len(worked_companies) > 1:
            experience_section += f"        - {role} at {worked_companies[1]} ({random.randint(1, 3)} years)\n"
        if len(worked_companies) > 2:
            experience_section += f"        - Junior {role} at {worked_companies[2]} ({random.randint(1, 2)} years)"
        
        resume_text = f"""
        {role} with {years} years of experience
        
        Skills: {', '.join(selected_skills)}
        
        Experience:
        {experience_section}
        
        
        Proficiencies in: {', '.join(selected_skills)}
        
        Achievements:
        - Led projects using {random.choice(selected_skills)}
        - Improved system performance by {random.randint(10, 50)}%
        - Mentored {random.randint(2, 5)} junior developers
        - Deployed solutions impacting {random.randint(100, 10000)} users
        """
        
        worked_companies_str = ", ".join(worked_companies)
        
        resume_data = {
            "resume_id": f"R{uuid.uuid4().hex[:8].upper()}",
            "role_target": role,
            "level": level,
            "years_experience": float(years),
            "skills": ", ".join(selected_skills),
            "skills_count": len(selected_skills),
            "companies_worked": worked_companies_str,
            "resume_text": resume_text.strip(),
            "source": "synthetic",
            "date_generated": time.strftime("%Y-%m-%d"),
        }
        
        resumes.append(resume_data)
    
    return resumes

def process_role(role: str, num_levels: int, resumes_per_level: int) -> List[Dict[str, Any]]:
    """Generate synthetic resumes for a role across different seniority levels"""
    logging.info(f"Generating resumes for role: {role}")
    
    all_resumes = []
    levels = list(ROLES_EXPERIENCE.keys())[:num_levels]
    
    for level in levels:
        logging.info(f"  Generating {resumes_per_level} {level} {role} resumes...")
        synthetic_resumes = generate_synthetic_resume(role, level, resumes_per_level)
        all_resumes.extend(synthetic_resumes)
    
    logging.info(f"Generated {len(all_resumes)} resumes for role: {role}")
    return all_resumes

def main():
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Build synthetic resume database")
    parser.add_argument("--roles", required=True, help="CSV file with role_title column")
    parser.add_argument("--out", default="resume_database.csv", help="Output CSV file")
    parser.add_argument("--levels", type=int, default=3, help="Number of seniority levels to generate (1-5)")
    parser.add_argument("--per-level", type=int, default=3, help="Resumes per level per role")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--http-proxy", type=str, default=None)
    parser.add_argument("--https-proxy", type=str, default=None)
    parser.add_argument("--verify", choices=["certifi","system","insecure","path"], default="certifi")
    parser.add_argument("--ca-bundle", type=str, default=None)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Load roles
    try:
        roles_df = pd.read_csv(args.roles)
        roles = [r.strip() for r in roles_df["role_title"].astype(str).tolist() if r.strip()]
        logging.info(f"Loaded {len(roles)} roles from {args.roles}")
    except Exception as e:
        logging.error(f"Failed to load roles from {args.roles}: {e}")
        return
    
    # Validate levels
    levels = min(int(args.levels), 5)
    per_level = int(args.per_level)
    
    all_resumes = []
    
    # Process roles in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_role, role, levels, per_level): role
            for role in roles
        }
        
        for future in as_completed(futures):
            role = futures[future]
            try:
                resumes = future.result()
                all_resumes.extend(resumes)
            except Exception as e:
                logging.error(f"Failed to process role {role}: {e}")
    
    # Save results
    if all_resumes:
        df = pd.DataFrame(all_resumes)
        df.to_csv(args.out, index=False)
        logging.info(f"Saved {len(all_resumes)} resumes to {args.out}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Resume Database Generated Successfully!")
        print(f"{'='*60}")
        print(f"Total resumes: {len(all_resumes)}")
        print(f"Roles covered: {df['role_target'].nunique()}")
        print(f"Seniority levels: {df['level'].nunique()}")
        print(f"Output file: {args.out}")
        print(f"\nBreakdown by level:")
        print(df['level'].value_counts().to_string())
        print(f"\nBreakdown by role:")
        print(df['role_target'].value_counts().to_string())
    else:
        logging.warning("No resumes generated!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
