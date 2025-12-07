# FILE: ai/openai_client.py - OpenAI Integration
from __future__ import annotations
from typing import Any, Dict, List
import os
import re
import json
import numpy as np

def _cosine(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a.dot(b) / (na * nb))

def _manhattan(a, b):
    """Calculate negative manhattan distance (higher = more similar)"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return -float(np.sum(np.abs(a - b)))

def _euclidean(a, b):
    """Calculate negative euclidean distance (higher = more similar)"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return -float(np.linalg.norm(a - b))

def _dot_product(a, b):
    """Calculate dot product similarity"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a.dot(b))

def _preprocess_aggressive(text: str) -> str:
    """Aggressive text preprocessing - remove stop words and short words"""
    if not text:
        return ""

    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text.lower())

    # Remove stop words and short words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'can', 'shall', 'will', 'would', 'could', 'should', 'may', 'might', 'must'
    }

    words = text.split()
    filtered = [w for w in words if len(w) > 2 and w not in stop_words]

    return ' '.join(filtered)

def _norm(s: str) -> str:
    """Normalize text string"""
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)

def openai_rank_roles(resume_text: str, role_snippets: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank job roles using OpenAI embeddings"""
    resume_text = _norm(resume_text)
    if not role_snippets:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

    texts = [str(s.get("snippet", "")) for s in role_snippets]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Upgraded to text-embedding-3-large for better semantic understanding
        model = "text-embedding-3-large"
        emb_resume = client.embeddings.create(model=model, input=resume_text).data[0].embedding
        emb_snips = client.embeddings.create(model=model, input=texts).data
        scores = [_cosine(emb_resume, emb_s.embedding) for emb_s in emb_snips]
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")

    ranked = list(zip(role_snippets, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    out = []
    for r, sc in ranked[:max(1, int(top_k))]:
        out.append({
            "role_title": r.get("title", ""),
            "score": float(sc),
            "link": r.get("link", "")
        })
    return out

def openai_rank_roles_enhanced(resume_text: str, role_snippets: List[Dict[str, Any]], top_k: int = 5,
                              similarity_method: str = "manhattan", preprocess_text: bool = True) -> List[Dict[str, Any]]:
    """
    Enhanced role ranking using A/B testing insights:
    - Manhattan distance for better performance
    - Aggressive preprocessing to remove noise
    - text-embedding-3-large for superior semantic understanding
    """
    # Apply preprocessing if enabled
    if preprocess_text:
        resume_text = _preprocess_aggressive(resume_text)
        role_snippets = [{**s, "snippet": _preprocess_aggressive(str(s.get("snippet", "")))} for s in role_snippets]
    else:
        resume_text = _norm(resume_text)

    if not role_snippets:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

    texts = [str(s.get("snippet", "")) for s in role_snippets]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Use text-embedding-3-large for superior semantic understanding
        model = "text-embedding-3-large"
        emb_resume = client.embeddings.create(model=model, input=resume_text).data[0].embedding
        emb_snips = client.embeddings.create(model=model, input=texts).data

        # Use different similarity methods based on A/B testing results
        if similarity_method == "manhattan":
            scores = [_manhattan(emb_resume, emb_s.embedding) for emb_s in emb_snips]
        elif similarity_method == "euclidean":
            scores = [_euclidean(emb_resume, emb_s.embedding) for emb_s in emb_snips]
        elif similarity_method == "dot_product":
            scores = [_dot_product(emb_resume, emb_s.embedding) for emb_s in emb_snips]
        else:  # default to cosine
            scores = [_cosine(emb_resume, emb_s.embedding) for emb_s in emb_snips]

    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")

    ranked = list(zip(role_snippets, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    out = []
    for r, sc in ranked[:max(1, int(top_k))]:
        out.append({
            "role_title": r.get("title", ""),
            "score": float(sc),
            "link": r.get("link", "")
        })
    return out

def openai_rank_jds(resume_text: str, jd_rows: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank job descriptions using OpenAI embeddings"""
    resume_text = _norm(resume_text)
    if not jd_rows:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

    texts = [str(x.get("jd_text", "")) for x in jd_rows]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Upgraded to text-embedding-3-large for better semantic understanding
        model = "text-embedding-3-large"
        emb_resume = client.embeddings.create(model=model, input=resume_text).data[0].embedding
        emb_snips = client.embeddings.create(model=model, input=texts).data
        scores = [_cosine(emb_resume, emb_s.embedding) for emb_s in emb_snips]
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")

    ranked = list(zip(jd_rows, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    out = []
    for row, sc in ranked[:max(1, int(top_k))]:
        out.append({
            "role_title": row.get("role_title", ""),
            "company": row.get("company", ""),
            "title": row.get("source_title", "") or row.get("title", ""),
            "link": row.get("source_url", "") or row.get("link", ""),
            "match_percent": round(float(sc * 100.0), 1),
        })
    return out

def openai_rank_jds_enhanced(resume_text: str, jd_rows: List[Dict[str, Any]], top_k: int = 5,
                           similarity_method: str = "manhattan", preprocess_text: bool = True) -> List[Dict[str, Any]]:
    """
    Enhanced JD ranking using A/B testing insights:
    - Manhattan distance for better performance on JD data
    - Aggressive preprocessing to remove noise and improve matching
    - text-embedding-3-large for superior semantic understanding
    """
    # Apply preprocessing if enabled
    if preprocess_text:
        resume_text = _preprocess_aggressive(resume_text)
        jd_rows = [{**row, "jd_text": _preprocess_aggressive(str(row.get("jd_text", "")))} for row in jd_rows]
    else:
        resume_text = _norm(resume_text)

    if not jd_rows:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

    texts = [str(x.get("jd_text", "")) for x in jd_rows]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Use text-embedding-3-large for superior semantic understanding
        model = "text-embedding-3-large"
        emb_resume = client.embeddings.create(model=model, input=resume_text).data[0].embedding
        emb_snips = client.embeddings.create(model=model, input=texts).data

        # Use different similarity methods based on A/B testing results
        if similarity_method == "manhattan":
            scores = [_manhattan(emb_resume, emb_s.embedding) for emb_s in emb_snips]
        elif similarity_method == "euclidean":
            scores = [_euclidean(emb_resume, emb_s.embedding) for emb_s in emb_snips]
        elif similarity_method == "dot_product":
            scores = [_dot_product(emb_resume, emb_s.embedding) for emb_s in emb_snips]
        else:  # default to cosine
            scores = [_cosine(emb_resume, emb_s.embedding) for emb_s in emb_snips]

    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")

    ranked = list(zip(jd_rows, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    out = []
    seen_titles = set()  # Track titles we've already added to avoid duplicates
    
    for row, sc in ranked[:max(1, int(top_k * 2))]:  # Get 2x items to account for duplicates
        # Convert similarity score back to 0-1 range for display
        if similarity_method == "manhattan":
            # Manhattan returns negative distances, convert to positive similarity
            display_score = max(0, min(1, (sc + 2.0) / 4.0))  # Rough normalization
        elif similarity_method == "euclidean":
            # Euclidean returns negative distances, convert to positive similarity
            display_score = max(0, min(1, (sc + 2.0) / 4.0))  # Rough normalization
        elif similarity_method == "dot_product":
            # Dot product can be any range, normalize roughly
            display_score = max(0, min(1, (sc + 1.0) / 2.0))  # Rough normalization
        else:
            display_score = sc

        title = row.get("source_title", "") or row.get("title", "")
        
        # Skip if we've already seen this title
        if title in seen_titles:
            continue
        
        seen_titles.add(title)
        
        out.append({
            "role_title": row.get("role_title", ""),
            "company": row.get("company", ""),
            "title": title,
            "link": row.get("source_url", "") or row.get("link", ""),
            "match_percent": round(float(display_score * 100.0), 1),
        })
        
        # Stop once we have enough unique items
        if len(out) >= top_k:
            break
    
    return out

def openai_rank_courses(gaps, resume_text: str, snippets: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank course snippets against the user's gaps + resume using OpenAI embeddings"""
    gaps = [str(g) for g in (gaps or []) if str(g).strip()]
    bundle = " ".join(gaps + [resume_text or ""]).strip()
    
    # Handle both training course format and general snippet format
    docs = []
    for s in (snippets or []):
        if isinstance(s, dict):
            # For training courses, use description as the text to embed
            snippet_text = s.get("description", "") or s.get("snippet", "") or s.get("title", "")
            docs.append(str(snippet_text))
        else:
            docs.append(str(s))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = "text-embedding-3-small"
        emb_q = client.embeddings.create(model=model, input=bundle).data[0].embedding
        emb_docs = client.embeddings.create(model=model, input=docs).data
        scores = [_cosine(emb_q, d.embedding) for d in emb_docs]
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")

    pairs = list(zip(snippets or [], scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    out = []
    for r, sc in pairs[:max(1, int(top_k))]:
        # Handle both training course format and general snippet format
        title = r.get("title", "Course")
        link = r.get("link", "")
        provider = r.get("provider", "") or r.get("source", "")
        hours = r.get("hours")
        price = r.get("price")
        rating = r.get("rating")
        
        course_data = {
            "title": title,
            "provider": provider,
            "link": link,
            "match_percent": round(float(sc * 100.0), 1)
        }
        
        # Add optional fields if available
        if hours is not None and not (isinstance(hours, float) and np.isnan(hours)):
            course_data["hours"] = hours
        if price is not None and price != 'unknown':
            course_data["price"] = price
        if rating is not None and not (isinstance(rating, float) and np.isnan(rating)):
            course_data["rating"] = rating
            
        out.append(course_data)
    return out

def openai_parse_resume(resume_text: str) -> Dict[str, Any]:
    """Parse resume text into structured data using OpenAI"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    # Define the expected schema
    schema = {
        "professional_summary": "",
        "current_role": {"role": "", "company": ""},
        "technical_skills": [""],
        "career_level": "",
        "industry_focus": "",
        "work_experience": [
            {"title": "", "company": "", "start_date": "", "end_date": "", "responsibilities": ""}
        ],
        "key_achievements": [""],
        "soft_skills": [""],
        "location": "",
        "projects": [""],
        "education": [
            {"degree": "", "institution": "", "graduation_date": ""}
        ],
        "certifications": [""],
    }
    
    prompt = f"""
    Analyze the following resume text and extract structured professional information.
    Return ONLY valid JSON matching EXACTLY this schema (keys & nesting):
    
    {json.dumps(schema, indent=2)}
    
    IMPORTANT: Do NOT extract personal identifying information like name, email, or phone number.
    Only extract location (country, state, city if available).
    Use empty strings/lists if information is unknown. No commentary outside JSON.
    
    Resume text:
    {resume_text}
    """
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        structured_data = json.loads(response.choices[0].message.content)
        return structured_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract structured data: {str(e)}")
