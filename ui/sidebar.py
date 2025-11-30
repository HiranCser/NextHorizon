# FILE: ui/sidebar.py - Sidebar UI Components
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Dict

from utils.data_quality import run_quality_checks

def render_sidebar():
    """Render the sidebar with database management"""
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Database Management")
    
    with st.sidebar.expander("Upload Databases", expanded=False):
        # Job Description Database Upload
        st.markdown("**Job Description Database**")
        jd_csv = st.file_uploader(
            "Upload JD Database (CSV)", 
            type=["csv"], 
            key="jd_upload",
            help="CSV with columns: role_title, jd_text, company, source_url, etc."
        )
        if jd_csv:
            try:
                df = pd.read_csv(jd_csv)
                st.session_state.jd_df = df
                st.success(f"✅ JD Database loaded: {len(df)} entries")
                st.write(f"**Columns:** {', '.join(df.columns)}")
                # Run quick data quality checks
                try:
                    rpt = run_quality_checks(path=st.session_state.get('jd_upload').name if hasattr(st.session_state.get('jd_upload'), 'name') else '', dataset_type='jd')
                except Exception:
                    # fallback: run on in-memory df by writing tempfile
                    import tempfile, json
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                    df.to_csv(tmp.name, index=False)
                    tmp.close()
                    rpt = run_quality_checks(tmp.name, dataset_type='jd')
                st.write("**Data Quality Summary (preview):**")
                st.json({k: rpt[k] for k in rpt.keys() & {'rows','missing_counts','source_domain','exp_min_years'} if k in rpt})
                if st.button("💾 Save JD Quality Report", key="save_jd_report"):
                    import json, os
                    outp = 'build_jd_dataset/jd_quality_report.json'
                    with open(outp, 'w', encoding='utf-8') as f:
                        json.dump(rpt, f, indent=2)
                    st.success(f"Saved report to {outp}")
            except Exception as e:
                st.error(f"Error loading JD database: {e}")
        
        st.markdown("---")
        
        # Course Database Upload  
        st.markdown("**Course Database**")
        course_csv = st.file_uploader(
            "Upload Course Database (CSV)", 
            type=["csv"], 
            key="course_upload",
            help="CSV with columns: course_title, course_url, skills, description, etc."
        )
        if course_csv:
            try:
                df = pd.read_csv(course_csv)
                st.session_state.training_df = df
                st.success(f"✅ Course Database loaded: {len(df)} entries")
                st.write(f"**Columns:** {', '.join(df.columns)}")
                # Quick data quality checks for training set
                try:
                    rpt = run_quality_checks(path=st.session_state.get('course_upload').name if hasattr(st.session_state.get('course_upload'), 'name') else '', dataset_type='training')
                except Exception:
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                    df.to_csv(tmp.name, index=False)
                    tmp.close()
                    rpt = run_quality_checks(tmp.name, dataset_type='training')
                st.write("**Data Quality Summary (preview):**")
                st.json({k: rpt[k] for k in rpt.keys() & {'rows','missing_counts','provider','hours'} if k in rpt})
                if st.button("💾 Save Training Quality Report", key="save_training_report"):
                    import json
                    outp = 'build_training_dataset/training_quality_report.json'
                    with open(outp, 'w', encoding='utf-8') as f:
                        json.dump(rpt, f, indent=2)
                    st.success(f"Saved report to {outp}")
            except Exception as e:
                st.error(f"Error loading course database: {e}")
        
        # Show current database status
        st.markdown("---")
        st.markdown("**Current Status:**")
        jd_count = len(st.session_state.get("jd_df", pd.DataFrame()))
        course_count = len(st.session_state.get("training_df", pd.DataFrame()))
        st.write(f"• JD Database: {jd_count} entries")
        st.write(f"• Course Database: {course_count} entries")
    
    st.sidebar.caption("Tip: Keep this structure; extend tabs independently without touching the main app.")
