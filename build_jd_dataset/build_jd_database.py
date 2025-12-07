
# FILE: build_jd_database.py

"""
Hardened JD collector + metadata extraction (experience & seniority) with quality filters.

New columns added:
- exp_min_years        (float | None)   # minimum years of experience required (parsed/inferred)
- exp_max_years        (float | None)   # maximum years if a range is specified; None for "3+ years"
- exp_evidence         (str)            # short text span that triggered extraction
- seniority_level      (str)            # Intern | Junior | Mid | Senior | Staff | Principal | Lead | Director | Executive | Unspecified
- seniority_evidence   (str)            # matched phrase (usually from the title)
- matched_skills       (str)            # comma-separated list of matched technical skills

Quality filters:
- JD text must be minimum 2000 characters
- Must contain technical skills, soft skills, and required experience in years
- Must contain at least one skill from the required technical skills list
- Minimum 5 valid JDs per role

Other features:
- Search engine: SerpAPI (if SERPAPI_API_KEY) or ddgs (DuckDuckGo, no key). Force with --engine.
- Robust fetch with retries, timeouts, optional proxies, and TLS verification controls.
- Parallel fetching per role; per-role progress logs.

Install
    pip install ddgs beautifulsoup4 tldextract html5lib requests pandas certifi

Usage
    python build_jd_database.py --roles role_list.csv --out jd_database.csv --per_role 5
    # proxy / TLS options are available; run with -h for details.
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

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

GOOD_DOMAINS = [
    "boards.greenhouse.io", "jobs.lever.co", "smartrecruiters.com",
    "ashbyhq.com", "myworkdayjobs.com", "jobs.ashbyhq.com"
]

# Technical skills list for filtering
REQUIRED_TECHNICAL_SKILLS = {
    "Python": [r"\bpython\b"],
    "SQL": [r"\bsql\b", r"\bsql\s+database\b"],
    "Machine Learning": [r"\bmachine\s+learning\b", r"\bml\b", r"\bscikit\s*-?\s*learn\b", r"\bxgboost\b"],
    "Deep Learning": [r"\bdeep\s+learning\b", r"\bpytorch\b", r"\btensorflow\b", r"\btf\b"],
    "Cloud Platforms": [r"\baws\b", r"\bamazon\s+web\s+services\b", r"\bazure\b", r"\bgcp\b", r"\bgoogle\s+cloud\b"],
    "MLOps Tools": [r"\bmlflow\b", r"\bkubeflow\b", r"\bsagemaker\b"],
    "Docker": [r"\bdocker\b", r"\bcontainers?\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "Data Engineering": [r"\bspark\b", r"\bapache\s+spark\b", r"\bkafka\b", r"\bairflow\b"],
    "APIs & Microservices": [r"\bapi\b", r"\brest\b", r"\brestful\b", r"\bfastapi\b", r"\bflask\b", r"\bgrpc\b", r"\bmicroservices?\b"],
    "Linux / Bash": [r"\blinux\b", r"\bbash\b", r"\bshell\s+script\b"],
    "Version Control": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "Databases": [r"\bpostgresql\b", r"\bpostgres\b", r"\bmongodb\b", r"\bmongo\b", r"\bsql\s+database\b", r"\bnosql\b"],
    "Computer Vision": [r"\bcomputer\s+vision\b", r"\bopencv\b", r"\byolo\b", r"\bvision\s+transformer\b"],
    "NLP": [r"\bnlp\b", r"\bnatural\s+language\b", r"\btransformers?\b", r"\bhuggingface\b", r"\bllm\b", r"\blarge\s+language\s+model\b"],
    "CI/CD": [r"\bci\s*/?\s*cd\b", r"\bcicd\b", r"\bjenkins\b", r"\bgithub\s+actions?\b", r"\bgitlab\s+ci\b"],
    "Data Visualization": [r"\bdata\s+visualization\b", r"\btableau\b", r"\bpower\s+bi\b", r"\bplotly\b", r"\bmatplotlib\b"],
    "Distributed Systems": [r"\bdistributed\s+systems?\b", r"\bscalability\b", r"\bcaching\b", r"\bmessage\s+queue\b"],
    "Generative AI": [r"\bgenerative\s+ai\b", r"\bllm\b", r"\bembedding\b", r"\brag\b", r"\bvector\s+db\b", r"\bvector\s+database\b"],
    "System Design": [r"\bsystem\s+design\b", r"\bscalability\b", r"\bcaching\b", r"\bload\s+balanc\b"],
}

# Soft skills patterns
SOFT_SKILLS_PATTERNS = [
    r"\bcommunication\b", r"\bteamwork\b", r"\bproblem\s+solving\b",
    r"\bleadership\b", r"\badaptability\b", r"\bcritical\s+thinking\b",
    r"\banalytical\b", r"\bcollaboration\b"
]

# Role-specific high-priority technical skills mapping
ROLE_SPECIFIC_SKILLS = {
    "Machine Learning Engineer": [
        "Python", "Machine Learning", "Deep Learning", "Cloud Platforms",
        "MLOps Tools", "Kubernetes", "APIs & Microservices", "Generative AI"
    ],
    "Data Scientist": [
        "Python", "SQL", "Machine Learning", "Deep Learning",
        "Data Visualization", "Generative AI"
    ],
    "Data Engineer": [
        "Python", "SQL", "Cloud Platforms", "Data Engineering",
        "Linux / Bash", "Databases", "Distributed Systems"
    ],
    "Cloud Engineer": [
        "Cloud Platforms", "Docker", "Kubernetes", "APIs & Microservices",
        "Linux / Bash", "CI/CD", "Distributed Systems"
    ],
    "DevOps Engineer": [
        "Docker", "Kubernetes", "Linux / Bash", "Version Control",
        "CI/CD", "Distributed Systems"
    ],
    "MLOps Engineer": [
        "Cloud Platforms", "MLOps Tools", "Docker", "Kubernetes",
        "Version Control", "CI/CD", "Generative AI"
    ],
    "Computer Vision Engineer": [
        "Python", "Deep Learning", "APIs & Microservices",
        "Computer Vision", "Generative AI"
    ],
    "NLP Engineer": [
        "Python", "Deep Learning", "APIs & Microservices",
        "NLP", "Generative AI"
    ],
    "AI Product Manager": [
        "Python", "Machine Learning", "Cloud Platforms",
        "Data Visualization", "Generative AI", "System Design"
    ],
    "Backend Software Engineer": [
        "Python", "APIs & Microservices", "Linux / Bash",
        "Version Control", "Distributed Systems", "System Design"
    ],
}

# ----------------- Search backends -----------------
def _import_ddg():
    try:
        from ddgs import DDGS  # new package
        return "ddgs", DDGS
    except Exception:
        pass
    try:
        from duckduckgo_search import DDGS  # legacy
        return "duckduckgo_search", DDGS
    except Exception:
        return None, None

def search_ddg(query: str, k: int = 15) -> List[Dict[str, Any]]:
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
                    "link":  (x.get("href") or x.get("url") or ""),
                    "snippet": (x.get("body") or ""),
                    "source": pkg
                })
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logging.warning(f"ddg search error: {e}")
        return []
    return results

def search_serpapi(query: str, k: int = 15) -> List[Dict[str, Any]]:
    out = []
    if not SERPAPI_KEY:
        return out
    try:
        params = {"engine":"google","q":query,"num":int(k),"api_key":SERPAPI_KEY}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        for it in (data.get("organic_results") or [])[:k]:
            out.append({
                "title": it.get("title",""),
                "link": it.get("link",""),
                "snippet": it.get("snippet",""),
                "source": "serpapi",
            })
    except Exception as e:
        logging.warning(f"serpapi search error: {e}")
        return []
    return out

def search_web(query: str, k: int, engine: str) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []
    if engine == "serpapi":
        out = search_serpapi(query, k)
        return out or search_ddg(query, k)
    if engine == "ddg":
        out = search_ddg(query, k)
        return out or search_serpapi(query, k)
    # auto
    if SERPAPI_KEY:
        out = search_serpapi(query, k)
        if out: return out
    return search_ddg(query, k)

# ----------------- HTTP helpers -----------------
def is_good(url: str) -> bool:
    try:
        h = urlparse(url).hostname or ""
        return any(h.endswith(d) for d in GOOD_DOMAINS)
    except Exception:
        return False

def make_session(http_proxy: Optional[str], https_proxy: Optional[str], verify_mode: str, ca_bundle: Optional[str]) -> requests.Session:
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

    # TLS verify
    if verify_mode == "insecure":
        s.verify = False
    elif verify_mode == "certifi":
        s.verify = ca_bundle or CERTIFI_PATH
    elif verify_mode == "system":
        s.verify = True
    elif verify_mode == "path":
        s.verify = ca_bundle
    else:
        s.verify = ca_bundle or CERTIFI_PATH

    s.headers.update({"User-Agent": UA})
    return s

def fetch(session: requests.Session, url: str, connect_timeout: float, read_timeout: float) -> str:
    try:
        r = session.get(url, timeout=(float(connect_timeout), float(read_timeout)))
        if r.status_code != 200:
            return ""
        return r.text
    except requests.exceptions.SSLError as e:
        logging.warning(f"SSL error for {url}: {e}")
        return ""
    except requests.exceptions.RequestException as e:
        logging.debug(f"Request failed for {url}: {e}")
        return ""

# ----------------- Parsing helpers (experience & seniority) -----------------
_WORD2NUM = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15
}

def _num_from_word(w: str) -> Optional[int]:
    return _WORD2NUM.get(w.lower().strip())

def extract_experience(text: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Extracts years of experience from text.
    Returns (min_years, max_years, evidence).
    """
    if not text: 
        return None, None, ""
    t = " ".join(text.split())  # collapse whitespace
    # Range: 3-5 years / 3 to 5 years
    m = re.search(r"\b(\d{1,2})\s*(?:-|to|–)\s*(\d{1,2})\s*(?:\+)?\s*(?:years?|yrs?)\b", t, re.IGNORECASE)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ev = m.group(0)
        return float(min(a,b)), float(max(a,b)), ev
    # Number+: 3+ years
    m = re.search(r"\b(\d{1,2})\s*\+\s*(?:years?|yrs?)\b", t, re.IGNORECASE)
    if m:
        a = int(m.group(1)); ev = m.group(0)
        return float(a), None, ev
    # Minimum of X years
    m = re.search(r"\b(?:min(?:imum)?(?:\s+of)?|at\s+least)\s*(\d{1,2})\s*(?:years?|yrs?)\b", t, re.IGNORECASE)
    if m:
        a = int(m.group(1)); ev = m.group(0)
        return float(a), None, ev
    # Simple X years of experience
    m = re.search(r"\b(\d{1,2})\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)\b", t, re.IGNORECASE)
    if m:
        a = int(m.group(1)); ev = m.group(0)
        return float(a), float(a), ev
    # Word numbers
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:\+)?\s*(?:years?|yrs?)\b", t, re.IGNORECASE)
    if m:
        a = _num_from_word(m.group(1)); ev = m.group(0)
        if a is not None:
            return float(a), float(a), ev
    return None, None, ""

_SENIORITY_PATTERNS = [
    ("Executive",  [r"\bvice\s+president\b", r"\bvp\b", r"\bchief\b", r"\bcto\b", r"\bcxo\b"]),
    ("Director",   [r"\bdirector\b", r"\bhead\s+of\b"]),
    ("Principal",  [r"\bprincipal\b"]),
    ("Staff",      [r"\bstaff\b"]),
    ("Lead",       [r"\blead\b", r"\blead[-\s]?engineer\b", r"\btech(?:nical)?\s+lead\b"]),
    ("Senior",     [r"\bsenior\b", r"\bsr\.?\b"]),
    ("Mid",        [r"\bmid[-\s]?level\b", r"\blevel\s*ii\b", r"\blevel\s*2\b"]),
    ("Junior",     [r"\bjunior\b", r"\bentry[-\s]?level\b", r"\bassociate\b"]),
    ("Intern",     [r"\bintern(?:ship)?\b"]),
]

def infer_seniority(title: str, text: str) -> Tuple[str, str]:
    """Return (level, evidence)."""
    blob = f"{title or ''} {text or ''}".lower()
    for level, patterns in _SENIORITY_PATTERNS:
        for pat in patterns:
            m = re.search(pat, blob, re.IGNORECASE)
            if m:
                return level, m.group(0)
    return "Unspecified", ""

def visible_text_from_html(html_text: str, limit_chars: int = 50000) -> str:
    """Extract visible text from HTML, preserving more content for validation."""
    soup = BeautifulSoup(html_text or "", "html5lib")
    for t in soup(["script", "style", "noscript"]):
        t.extract()
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text[:limit_chars]

def has_soft_skills(text: str) -> bool:
    """Check if text contains soft skills."""
    for pattern in SOFT_SKILLS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def extract_matched_skills(text: str) -> List[str]:
    """Extract and return matched technical skills from text."""
    matched = []
    for skill_name, patterns in REQUIRED_TECHNICAL_SKILLS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(skill_name)
                break  # Don't duplicate same skill name
    return matched

def has_required_technical_skills(text: str) -> bool:
    """Check if text contains at least one required technical skill."""
    return len(extract_matched_skills(text)) > 0

def is_valid_jd(text: str, role: str, exp_min: Optional[float], exp_max: Optional[float], exp_evidence: str) -> Tuple[bool, List[str]]:
    """
    Validate if JD meets quality criteria:
    a) jd_text minimum 2000 characters
    b) Contains technical skills, soft skills, and required experience in years
    c) Contains at least one skill from role-specific high-priority technical skills
    
    Returns (is_valid, matched_skills)
    """
    # Check a) - minimum 2000 characters
    if len(text) < 2000:
        return False, []
    
    # Check for experience in years
    has_experience = (exp_min is not None or exp_max is not None) and exp_evidence
    if not has_experience:
        return False, []
    
    # Check for soft skills
    if not has_soft_skills(text):
        return False, []
    
    # Check c) - role-specific required technical skills
    all_matched_skills = extract_matched_skills(text)
    
    # Get role-specific required skills
    role_key = None
    for rkey in ROLE_SPECIFIC_SKILLS.keys():
        if rkey.lower() in role.lower() or role.lower() in rkey.lower():
            role_key = rkey
            break
    
    if role_key:
        required_skills = ROLE_SPECIFIC_SKILLS[role_key]
        matched_role_skills = [s for s in all_matched_skills if s in required_skills]
        if not matched_role_skills:
            return False, []
        return True, matched_role_skills
    else:
        # Fallback: if role not in mapping, require at least one skill from general list
        if not all_matched_skills:
            return False, []
        return True, all_matched_skills

# ----------------- Main logic -----------------
def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", required=True, help="CSV with role_title column")
    ap.add_argument("--out", default="jd_database.csv")
    ap.add_argument("--per_role", type=int, default=5)
    ap.add_argument("--engine", choices=["auto","serpapi","ddg"], default="auto")
    ap.add_argument("--sleep", type=float, default=0.7, help="seconds between roles")
    ap.add_argument("--connect-timeout", type=float, default=6.0)
    ap.add_argument("--read-timeout", type=float, default=18.0)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--http-proxy", type=str, default=None)
    ap.add_argument("--https-proxy", type=str, default=None)
    ap.add_argument("--verify", choices=["certifi","system","insecure","path"], default="certifi")
    ap.add_argument("--ca-bundle", type=str, default=None, help="Path to CA bundle when --verify path")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.verify == "certifi" and CERTIFI_PATH:
        os.environ.setdefault("REQUESTS_CA_BUNDLE", CERTIFI_PATH)

    roles_df = pd.read_csv(args.roles)
    roles = [r.strip() for r in roles_df["role_title"].astype(str).tolist() if r.strip()]

    session = make_session(args.http_proxy, args.https_proxy, args.verify, args.ca_bundle)

    base_fields = ["jd_id","role_title","company","source_title","source_url","source_domain","jd_text","date_scraped"]
    extra_fields = ["exp_min_years","exp_max_years","exp_evidence","seniority_level","seniority_evidence","matched_skills"]
    fieldnames = base_fields + extra_fields
    rows_all = []

    for ridx, role in enumerate(roles, 1):
        logging.info(f"[{ridx}/{len(roles)}] Searching postings for role: {role}")
        needed = int(args.per_role)
        valid_count = 0
        q = f'site:({ " OR ".join(GOOD_DOMAINS) }) "{role}"'
        
        attempt = 0
        max_attempts = 3
        all_candidates = []

        # Keep searching until we have enough valid JDs or exhaust attempts
        while valid_count < needed and attempt < max_attempts:
            attempt += 1
            logging.info(f"  Attempt {attempt}/{max_attempts} to find {needed - valid_count} more valid JDs")
            
            results = search_web(q, k=40, engine=args.engine)
            
            cand_urls = []
            seen = set()
            for r in results:
                url = r.get("link","")
                if not url or not is_good(url): continue
                if url in seen or url in [c[1] for c in all_candidates]: continue
                seen.add(url)
                cand_urls.append((r.get("title",""), url))

            if not cand_urls:
                logging.warning(f"  No more candidates found for {role}")
                break

            def work(item):
                title, url = item
                html_raw = fetch(session, url, args.connect_timeout, args.read_timeout)
                return (title, url, html_raw)

            fetched = []
            with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
                futs = [ex.submit(work, it) for it in cand_urls]
                for fut in as_completed(futs):
                    try:
                        title, url, html_raw = fut.result()
                    except Exception as e:
                        logging.debug(f"worker error: {e}")
                        continue
                    if not html_raw:
                        continue
                    fetched.append((title, url, html_raw))

            # Process fetched pages and validate
            for (title, url, html_raw) in fetched:
                vis = visible_text_from_html(html_raw, limit_chars=50000)
                
                # Extract metadata
                exp_min, exp_max, exp_ev = extract_experience(f"{title} {vis}")
                level, level_ev = infer_seniority(title, vis)
                
                # Validate JD quality (including role-specific skills)
                is_valid, matched_skills = is_valid_jd(vis, role, exp_min, exp_max, exp_ev)
                
                if not is_valid:
                    all_candidates.append((title, url, html_raw, None))
                    continue

                # Create JD summary from meta or page content
                soup = BeautifulSoup(html_raw, "html5lib")
                meta = soup.find("meta", attrs={"name":"description"})
                if meta and meta.get("content"):
                    jd_summary = meta["content"]
                else:
                    jd_summary = " ".join([t.get_text(" ", strip=True) for t in soup.find_all(["p","li"])][:50])
                jd_summary = re.sub(r"\s+", " ", jd_summary).strip()

                company = extract_company(url, title)
                dom = tldextract.extract(url)
                
                rows_all.append({
                    "jd_id": str(uuid.uuid4()),
                    "role_title": role,
                    "company": company,
                    "source_title": (title or "").strip(),
                    "source_url": url,
                    "source_domain": f"{dom.subdomain}.{dom.domain}.{dom.suffix}".strip("."),
                    "jd_text": vis[:5000],  # Use first 5000 chars as summary
                    "date_scraped": time.strftime("%Y-%m-%d"),
                    "exp_min_years": exp_min,
                    "exp_max_years": exp_max,
                    "exp_evidence": exp_ev,
                    "seniority_level": level,
                    "seniority_evidence": level_ev,
                    "matched_skills": "; ".join(matched_skills),
                })
                valid_count += 1
                
                if valid_count >= needed:
                    break
            
            time.sleep(max(0.1, float(args.sleep)))

        logging.info(f"Successfully collected {valid_count}/{needed} valid JDs for role: {role}")
        if valid_count < needed:
            logging.warning(f"Warning: Only {valid_count} valid JDs found for {role} (required {needed})")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_all)

    logging.info(f"Done. Wrote {len(rows_all)} rows to {args.out}")

# ---- shared helper functions ----
def make_session(http_proxy: Optional[str], https_proxy: Optional[str], verify_mode: str, ca_bundle: Optional[str]) -> requests.Session:
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

# Company extraction helper
from urllib.parse import urlparse
def extract_company(url: str, title: str) -> str:
    try:
        ex = tldextract.extract(url)
        sub = ex.subdomain.split(".")
        if "boards" in sub and "greenhouse" in (ex.domain or ""):
            path = urlparse(url).path.strip("/").split("/")
            if path: return path[0].replace("-", " ").title()
        if "jobs" in sub and "lever" in (ex.domain or ""):
            path = urlparse(url).path.strip("/").split("/")
            if path: return path[0].replace("-", " ").title()
        m = re.search(r"at ([^-|–:]+)", title, re.IGNORECASE)
        if m: return m.group(1).strip()
    except Exception:
        pass
    return ""

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
