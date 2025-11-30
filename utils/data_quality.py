"""
utils/data_quality.py

Simple data quality and outlier detection helpers for JD and training CSVs.

Usage examples:
    from utils.data_quality import run_quality_checks
    run_quality_checks('build_jd_dataset/jd_database.csv', 'jd')

This is intentionally lightweight and uses pandas only (already in requirements).
"""
from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np


def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}: {e}")


def numeric_outliers(series: pd.Series, method: str = "iqr") -> Dict[str, Any]:
    """Return basic outlier stats for a numeric series.

    Methods supported: 'iqr' (default), 'zscore'.
    """
    s = pd.to_numeric(series, errors='coerce').dropna()
    if s.empty:
        return {"count": 0, "outliers": [], "method": method}

    if method == "zscore":
        mean = s.mean()
        std = s.std()
        z = (s - mean) / std
        out = s[np.abs(z) > 3]
        return {"count": int(s.size), "outliers": out.tolist(), "method": method}

    # default: IQR
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    out = s[(s < lower) | (s > upper)]
    return {"count": int(s.size), "outliers": out.tolist(), "method": "iqr", "bounds": [float(lower), float(upper)]}


def text_length_outliers(series: pd.Series, lower_pct: float = 0.01) -> Dict[str, Any]:
    """Detect very short or very long text entries by length distribution.

    Returns lists of shortest and longest examples (top/bottom `lower_pct` fraction).
    """
    txt = series.fillna("").astype(str)
    lens = txt.str.len()
    n = len(lens)
    if n == 0:
        return {"count": 0, "short_examples": [], "long_examples": []}
    k = max(1, int(n * lower_pct))
    short_idx = lens.nsmallest(k).index.tolist()
    long_idx = lens.nlargest(k).index.tolist()
    return {
        "count": n,
        "short_examples": [(i, txt.loc[i]) for i in short_idx],
        "long_examples": [(i, txt.loc[i][:500]) for i in long_idx]
    }


def category_frequency(series: pd.Series, top_n: int = 10) -> Dict[str, Any]:
    s = series.fillna("").astype(str)
    vc = s.value_counts().head(top_n)
    return {"top": vc.to_dict(), "unique_count": int(s.nunique())}


def run_quality_checks(path: str, dataset_type: str = "jd", out_path: Optional[str] = None) -> Dict[str, Any]:
    """Run a set of quality checks and simple outlier detection.

    dataset_type: 'jd' or 'training' to select heuristics.
    Returns a dict with findings. Optionally writes JSON to `out_path`.
    """
    df = _safe_read_csv(path)
    report: Dict[str, Any] = {"rows": len(df), "columns": list(df.columns)}

    if dataset_type == "jd":
        # Check experience fields
        for col in ["exp_min_years", "exp_max_years"]:
            if col in df.columns:
                report[col] = numeric_outliers(df[col])

        # Text length on jd_text
        if "jd_text" in df.columns:
            report["jd_text_length"] = text_length_outliers(df["jd_text"])

        # Domain frequency
        if "source_domain" in df.columns:
            report["source_domain"] = category_frequency(df["source_domain"], top_n=20)

    elif dataset_type == "training":
        # hours, rating checks
        for col in ["hours", "rating"]:
            if col in df.columns:
                report[col] = numeric_outliers(df[col])

        # provider frequency
        if "provider" in df.columns:
            report["provider"] = category_frequency(df["provider"], top_n=50)

        # text length checks on description/title
        for col in ["description", "title"]:
            if col in df.columns:
                report[f"{col}_length"] = text_length_outliers(df[col])

    # General checks
    # Duplicates by URL or ID
    if "source_url" in df.columns:
        report["duplicate_source_url_count"] = int(df["source_url"].duplicated().sum())
    if "training_id" in df.columns:
        report["duplicate_training_id_count"] = int(df["training_id"].duplicated().sum())

    # Missingness summary
    missing = df.isna().sum().to_dict()
    report["missing_counts"] = {k: int(v) for k, v in missing.items()}

    if out_path:
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to write report to {out_path}: {e}")

    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run data quality checks for NextHorizon datasets")
    ap.add_argument("--path", required=True, help="Path to CSV file")
    ap.add_argument("--type", choices=["jd", "training"], default="jd", help="Dataset type")
    ap.add_argument("--out", default=None, help="Optional output JSON path for report")
    args = ap.parse_args()

    rpt = run_quality_checks(args.path, dataset_type=args.type, out_path=args.out)
    print(json.dumps(rpt, indent=2)[:10000])
