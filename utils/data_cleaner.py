"""
utils/data_cleaner.py

Provides: schema validation (pandera), provider normalization, cleaning pipeline,
embedding-health checks, and reporting. Produces before/after reports and writes
cleaned CSV files.
"""
from __future__ import annotations
import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import base64
from typing import Tuple

PROVIDER_MAP_PATH = os.path.join(os.path.dirname(__file__), 'provider_map.json')

def _load_provider_map() -> Dict[str, str]:
    try:
        with open(PROVIDER_MAP_PATH, 'r', encoding='utf-8') as f:
            m = json.load(f)
    except Exception:
        m = {}
    # normalize keys
    return {k.strip().lower(): v for k, v in m.items()}


TRAINING_SCHEMA = DataFrameSchema({
    "training_id": Column(pa.String, nullable=False),
    "skill": Column(pa.String, nullable=True),
    "title": Column(pa.String, nullable=True),
    "description": Column(pa.String, nullable=True),
    "provider": Column(pa.String, nullable=True),
    "hours": Column(pa.Float, nullable=True, checks=Check(lambda s: pd.isna(s) or (s >= 0), element_wise=True)),
    "price": Column(pa.String, nullable=True),
    "rating": Column(pa.Float, nullable=True, checks=Check(lambda s: pd.isna(s) or ((s >= 0) and (s <= 5)), element_wise=True)),
    "link": Column(pa.String, nullable=True)
})

JD_SCHEMA = DataFrameSchema({
    "jd_id": Column(pa.String, nullable=False),
    "role_title": Column(pa.String, nullable=False),
    "company": Column(pa.String, nullable=True),
    "source_title": Column(pa.String, nullable=True),
    "source_url": Column(pa.String, nullable=True),
    "source_domain": Column(pa.String, nullable=True),
    "jd_text": Column(pa.String, nullable=True),
    "date_scraped": Column(pa.String, nullable=True),
    "exp_min_years": Column(pa.Float, nullable=True),
    "exp_max_years": Column(pa.Float, nullable=True),
    "exp_evidence": Column(pa.String, nullable=True),
    "seniority_level": Column(pa.String, nullable=True),
    "seniority_evidence": Column(pa.String, nullable=True),
})


def _normalize_provider(val: Optional[str], mapping: Dict[str, str]) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    key = s.lower().strip()
    if key in mapping:
        return mapping[key]
    # fuzzy fallback: pick closest mapping key
    try:
        from difflib import get_close_matches
        candidates = get_close_matches(key, mapping.keys(), n=1, cutoff=0.75)
        if candidates:
            return mapping[candidates[0]]
    except Exception:
        pass
    return s


def _impute_hours(df: pd.DataFrame, strategy: str = "median_by_provider") -> pd.DataFrame:
    df = df.copy()
    if "hours" not in df.columns:
        return df
    # coerce numeric first
    df["hours"] = _coerce_numeric(df, "hours")
    if strategy == "median_by_provider":
        medians = df.groupby("provider_normalized")["hours"].median()
        def impute_row(r):
            if pd.notna(r.get("hours")):
                return r.get("hours")
            prov = r.get("provider_normalized")
            if pd.isna(prov) or prov == '':
                return df["hours"].median()
            m = medians.get(prov, np.nan)
            return float(m) if pd.notna(m) else float(df["hours"].median())
        df["hours"] = df.apply(impute_row, axis=1)
    elif strategy == "global_median":
        df["hours"] = df["hours"].fillna(df["hours"].median())
    return df


def _impute_rating(df: pd.DataFrame, strategy: str = "median_by_provider") -> pd.DataFrame:
    df = df.copy()
    if "rating" not in df.columns:
        return df
    df["rating"] = _coerce_numeric(df, "rating")
    if strategy == "median_by_provider":
        medians = df.groupby("provider_normalized")["rating"].median()
        global_median = df["rating"].median()
        # If global median is NaN (all ratings missing), fall back to 0.0
        if pd.isna(global_median):
            global_median = 0.0

        def impute_row(r):
            if pd.notna(r.get("rating")):
                return r.get("rating")
            prov = r.get("provider_normalized")
            if pd.isna(prov) or prov == '':
                return float(global_median)
            m = medians.get(prov, np.nan)
            return float(m) if pd.notna(m) else float(global_median)

        df["rating"] = df.apply(impute_row, axis=1)
    elif strategy == "global_median":
        df["rating"] = df["rating"].fillna(df["rating"].median())
    return df


def _clamp_numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["rating"] = df["rating"].clip(lower=0.0, upper=5.0)
    if "hours" in df.columns:
        df["hours"] = pd.to_numeric(df["hours"], errors="coerce")
        # Replace negative hours with absolute value or NaN -> abs is conservative
        df["hours"] = df["hours"].apply(lambda x: abs(x) if pd.notna(x) and x < 0 else x)
    return df


def _encode_embedding_as_b64(vec: Optional[List[float]]) -> Optional[str]:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32)
    b = arr.tobytes()
    return base64.b64encode(b).decode('ascii')


def _decode_embedding_b64(s: Optional[str], dim: Optional[int] = None) -> Optional[np.ndarray]:
    if s is None or pd.isna(s):
        return None
    try:
        b = base64.b64decode(s)
        arr = np.frombuffer(b, dtype=np.float32)
        if dim and arr.size != dim:
            return None
        return arr
    except Exception:
        return None


def _normalize_domain(dom: Optional[str]) -> Optional[str]:
    if not dom or pd.isna(dom):
        return None
    s = str(dom).lower().strip()
    if s.startswith('www.'):
        s = s[4:]
    return s


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors='coerce')
    return s


def embedding_health_check(df: pd.DataFrame, embedding_col: str = 'embedding') -> Dict[str, Any]:
    """Check for presence of embedding vectors (if precomputed)"""
    if embedding_col not in df.columns:
        return {"present": False, "missing_count": len(df)}
    missing = df[embedding_col].isna().sum()
    return {"present": True, "missing_count": int(missing), "total": len(df)}


def clean_training_dataframe(df: pd.DataFrame, provider_map: Dict[str, str]) -> (pd.DataFrame, Dict[str, Any]):
    report_before = {
        "rows": len(df),
        "providers_raw": df.get('provider', pd.Series(dtype=str)).fillna('').unique().tolist(),
        "missing_hours": int(df['hours'].isna().sum()) if 'hours' in df.columns else None,
        "missing_rating": int(df['rating'].isna().sum()) if 'rating' in df.columns else None,
    }

    df2 = df.copy()
    # Trim strings
    for c in ['title', 'description', 'provider', 'skill', 'link']:
        if c in df2.columns:
            df2[c] = df2[c].astype(str).str.strip().replace({'nan':'', 'None':''})

    # Normalize provider
    if 'provider' in df2.columns:
        df2['provider_normalized'] = df2['provider'].apply(lambda x: _normalize_provider(x, provider_map))
    else:
        df2['provider_normalized'] = None

    # Coerce numeric
    for col in ['hours', 'rating']:
        if col in df2.columns:
            df2[col] = _coerce_numeric(df2, col)

    # Impute and clamp numeric ranges
    df2 = _impute_hours(df2, strategy='median_by_provider')
    df2 = _impute_rating(df2, strategy='median_by_provider')
    df2 = _clamp_numeric_ranges(df2)

    # Create simple text length
    df2['description_len'] = df2['description'].fillna('').astype(str).str.len()

    report_after = {
        "rows": len(df2),
        "providers_normalized": df2['provider_normalized'].dropna().unique().tolist(),
        "missing_hours": int(df2['hours'].isna().sum()) if 'hours' in df2.columns else None,
        "missing_rating": int(df2['rating'].isna().sum()) if 'rating' in df2.columns else None,
    }

    return df2, {"before": report_before, "after": report_after}


def clean_jd_dataframe(df: pd.DataFrame) -> (pd.DataFrame, Dict[str, Any]):
    report_before = {"rows": len(df), "sample_domains": df.get('source_domain', pd.Series(dtype=str)).fillna('').unique().tolist()[:10]}
    df2 = df.copy()
    # Normalize domain
    if 'source_domain' in df2.columns:
        df2['source_domain_norm'] = df2['source_domain'].apply(_normalize_domain)
    else:
        df2['source_domain_norm'] = None

    # Coerce numeric experience
    for col in ['exp_min_years', 'exp_max_years']:
        if col in df2.columns:
            df2[col] = _coerce_numeric(df2, col)

    # Improve company extraction: fallback to extracting from source_url if empty
    if 'company' in df2.columns and 'source_url' in df2.columns:
        def _extract_company(row):
            c = row.get('company')
            if c and str(c).strip():
                return str(c).strip()
            url = row.get('source_url') or ''
            try:
                from urllib.parse import urlparse
                p = urlparse(url)
                host = p.hostname or ''
                if host:
                    parts = host.split('.')
                    if len(parts) >= 2:
                        return parts[-2].title()
            except Exception:
                pass
            return ''
        df2['company'] = df2.apply(_extract_company, axis=1)

    report_after = {"rows": len(df2), "sample_domains_norm": df2['source_domain_norm'].dropna().unique().tolist()[:10]}
    return df2, {"before": report_before, "after": report_after}


def validate_with_schema(df: pd.DataFrame, dataset_type: str = 'training') -> Dict[str, Any]:
    try:
        if dataset_type == 'training':
            TRAINING_SCHEMA.validate(df, lazy=True)
        else:
            JD_SCHEMA.validate(df, lazy=True)
        return {"valid": True, "errors": []}
    except pa.errors.SchemaErrors as e:
        # Collect readable errors
        err_list = []
        for er in e.failure_cases.to_dict(orient='records'):
            err_list.append(er)
        return {"valid": False, "errors": err_list}


def run_full_clean(path: str, dataset_type: str = 'training', out_clean_path: Optional[str] = None) -> Dict[str, Any]:
    df = pd.read_csv(path)
    provider_map = _load_provider_map()
    res = {"path": path}
    if dataset_type == 'training':
        res['schema_before'] = validate_with_schema(df, 'training')
        df_clean, rpt = clean_training_dataframe(df, provider_map)
        res['clean_report'] = rpt
        res['schema_after'] = validate_with_schema(df_clean, 'training')
        if out_clean_path:
            df_clean.to_csv(out_clean_path, index=False)
            res['clean_path'] = out_clean_path
    else:
        res['schema_before'] = validate_with_schema(df, 'jd')
        df_clean, rpt = clean_jd_dataframe(df)
        res['clean_report'] = rpt
        res['schema_after'] = validate_with_schema(df_clean, 'jd')
        if out_clean_path:
            df_clean.to_csv(out_clean_path, index=False)
            res['clean_path'] = out_clean_path
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--type', choices=['training','jd'], default='training')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    r = run_full_clean(args.path, dataset_type=args.type, out_clean_path=args.out)
    print(json.dumps(r, indent=2)[:20000])
