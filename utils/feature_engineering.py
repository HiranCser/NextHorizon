"""utils/feature_engineering.py

Compute features for training/course and JD datasets.
- TF-IDF for combined title+description
- Numeric buckets for hours and rating
- Derived features: title_len, description_len, skill_count
- Save artifacts (TF-IDF vectorizer as pickle, sparse matrix as npz)
"""
from __future__ import annotations
import os
import pickle
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import json

try:
    import textstat
except Exception:
    textstat = None

try:
    import spacy
    _spacy_nlp = None
    try:
        _spacy_nlp = spacy.load('en_core_web_sm')
    except Exception:
        # model not available; will attempt lazy load later
        _spacy_nlp = None
except Exception:
    spacy = None
    _spacy_nlp = None


def _combine_text(df: pd.DataFrame) -> pd.Series:
    return (df.get('title', '').fillna('') + ' \n ' + df.get('description', '').fillna('')).astype(str)


def compute_tfidf(df: pd.DataFrame, max_features: int = 2048) -> Tuple[sparse.spmatrix, TfidfVectorizer]:
    texts = _combine_text(df)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words='english')
    X = vec.fit_transform(texts)
    return X, vec


def derive_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # title/description length
    df['title_len'] = df.get('title', '').fillna('').astype(str).str.len()
    df['description_len'] = df.get('description', '').fillna('').astype(str).str.len()
    # skill count heuristic
    if 'skill' in df.columns:
        df['skill_count'] = df['skill'].fillna('').astype(str).apply(lambda s: len([x for x in s.split(',') if x.strip()]))
    else:
        df['skill_count'] = 0
    # hours_bucket
    if 'hours' in df.columns:
        df['hours_numeric'] = pd.to_numeric(df['hours'], errors='coerce')
        df['hours_bucket'] = pd.cut(df['hours_numeric'].fillna(0), bins=[-1,1,5,10,20,50,9999], labels=['<1','1-5','5-10','10-20','20-50','50+'])
    else:
        df['hours_numeric'] = pd.NA
        df['hours_bucket'] = 'unknown'
    # rating_bucket
    if 'rating' in df.columns:
        df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating_bucket'] = pd.cut(df['rating_numeric'].fillna(0), bins=[-1,2,3,4,5], labels=['<=2','2-3','3-4','4-5'])
    else:
        df['rating_numeric'] = pd.NA
        df['rating_bucket'] = 'unknown'
    return df


def _readability_score(text: str) -> float:
    if not text or text.strip() == '':
        return float('nan')
    if textstat is None:
        # fallback: approx by average sentence length heuristic
        s = text.strip()
        sentences = s.count('.') + s.count('!') + s.count('?')
        words = len(s.split())
        if sentences == 0:
            return max(0.0, 206.835 - 1.015 * (words / max(1, len(s.split('.')))))
        return max(0.0, 206.835 - 1.015 * (words / sentences))
    try:
        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return float('nan')


def _spacy_entities_and_pos(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {'ents_total': 0, 'ents_person': 0, 'ents_org': 0, 'ents_gpe': 0, 'noun_pct': 0.0, 'verb_pct': 0.0}
    if not text or text.strip() == '':
        return out
    nlp = None
    if _spacy_nlp is not None:
        nlp = _spacy_nlp
    elif spacy is not None:
        try:
            nlp = spacy.load('en_core_web_sm')
        except Exception:
            nlp = None
    if nlp is None:
        # fallback heuristics
        words = text.split()
        out['noun_pct'] = 0.2
        out['verb_pct'] = 0.1
        out['ents_total'] = 0
        return out
    doc = nlp(text)
    ents = list(doc.ents)
    out['ents_total'] = len(ents)
    out['ents_person'] = sum(1 for e in ents if e.label_ == 'PERSON')
    out['ents_org'] = sum(1 for e in ents if e.label_ == 'ORG')
    out['ents_gpe'] = sum(1 for e in ents if e.label_ == 'GPE')
    # POS proportions
    pos_counts = {}
    for t in doc:
        pos_counts[t.pos_] = pos_counts.get(t.pos_, 0) + 1
    total = sum(pos_counts.values()) or 1
    out['noun_pct'] = pos_counts.get('NOUN', 0) / total
    out['verb_pct'] = pos_counts.get('VERB', 0) / total
    return out


def enrich_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute readability, entity counts, and POS heuristics for text fields."""
    df = df.copy()
    df['readability'] = df.get('description', '').fillna('').astype(str).apply(_readability_score)
    ent_counts = df.get('description', '').fillna('').astype(str).apply(_spacy_entities_and_pos)
    df['ents_total'] = ent_counts.apply(lambda d: d.get('ents_total', 0))
    df['ents_person'] = ent_counts.apply(lambda d: d.get('ents_person', 0))
    df['ents_org'] = ent_counts.apply(lambda d: d.get('ents_org', 0))
    df['ents_gpe'] = ent_counts.apply(lambda d: d.get('ents_gpe', 0))
    df['noun_pct'] = ent_counts.apply(lambda d: d.get('noun_pct', 0.0))
    df['verb_pct'] = ent_counts.apply(lambda d: d.get('verb_pct', 0.0))
    return df


def spacy_model_available() -> bool:
    """Return True if spaCy and the small English model are available."""
    try:
        return _spacy_nlp is not None
    except Exception:
        return False


def run_pipeline(path: str, out_csv: str, out_prefix: str, enrich: bool = True) -> Dict[str, Any]:
    """Run end-to-end feature pipeline. When `enrich` is True, compute readability and NER/POS features.

    Writes TF-IDF artifacts and a JSON report at `out_prefix + '.report.json'`.
    """
    df = pd.read_csv(path)
    df = derive_numeric_features(df)
    if enrich:
        df = enrich_text_features(df)
    X, vec = compute_tfidf(df)
    artifacts = save_tfidf_artifacts(X, vec, out_prefix)
    # attach some summary stats
    df['tfidf_rows'] = X.shape[0]
    df.to_csv(out_csv, index=False)

    report = {
        'out_csv': out_csv,
        'artifacts': artifacts,
        'rows': len(df),
        'tfidf_shape': X.shape,
        'missingness': {
            'hours_missing': int(df['hours'].isna().sum()) if 'hours' in df.columns else None,
            'rating_missing': int(df['rating'].isna().sum()) if 'rating' in df.columns else None,
        },
        'spacy_model_available': spacy_model_available()
    }
    # include TF-IDF vocab size if vectorizer available
    try:
        report['tfidf_vocab_size'] = len(vec.vocabulary_)
    except Exception:
        report['tfidf_vocab_size'] = None

    # per-column missingness summary (top columns)
    try:
        miss = df.isna().sum().to_dict()
        report['per_column_missing'] = {k: int(v) for k, v in miss.items()}
    except Exception:
        report['per_column_missing'] = {}

    # top N entities aggregated across dataset (requires spaCy)
    report['top_entities'] = []
    try:
        if spacy_model_available():
            from collections import Counter
            ctr = Counter()
            nlp = _spacy_nlp
            for txt in df.get('description', '').fillna('').astype(str):
                if not txt:
                    continue
                doc = nlp(txt)
                for e in doc.ents:
                    ctr[(e.label_, e.text.strip())] += 1
            # return top 20 entities
            top = ctr.most_common(20)
            report['top_entities'] = [{'label': l[0][0], 'text': l[0][1], 'count': l[1]} for l in top]
    except Exception:
        report['top_entities'] = []
    report_path = out_prefix + '.report.json'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

    return report


def save_tfidf_artifacts(X: sparse.spmatrix, vec: TfidfVectorizer, out_prefix: str) -> Dict[str, str]:
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    vec_path = out_prefix + '.tfidf.pkl'
    mat_path = out_prefix + '.tfidf.npz'
    with open(vec_path, 'wb') as f:
        pickle.dump(vec, f)
    sparse.save_npz(mat_path, X)
    return {'vectorizer': vec_path, 'matrix': mat_path}

