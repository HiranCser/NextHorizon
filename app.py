# FILE: app.py - Main Application Entry Point
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

# Disable telemetry
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")

# Import modules
from config.session_config import initialize_session_state, check_environment
from ui.resume_parsing import render as tab1_render
from ui.role_recommendations import render as tab2_render  
from ui.skill_gaps import render as tab3_render
from ui.course_recommendations import render as tab4_render
from ui.diagnostics import render as diagnostics_render

def load_databases():
    """Pre-load databases on startup"""
    try:
        # Load JD database
        jd_path = Path("build_jd_dataset/jd_database.csv")
        if jd_path.exists():
            st.session_state.jd_df = pd.read_csv(jd_path)
        
        # Load training database
        training_path = Path("build_training_dataset/training_database.csv")
        if training_path.exists():
            st.session_state.training_df = pd.read_csv(training_path)
    except Exception as e:
        pass  # Silent fail, will show warning in UI if needed


def get_pipeline_status():
    """Read pipeline artifacts (report, embeddings) and return a small status dict."""
    out = {
        'tfidf_vocab_size': None,
        'tfidf_rows': None,
        'embeddings_present': False,
        'emb_shape': None,
        'spacy_enrichment': False,
    }
    try:
        rpt_path = 'build_training_dataset/training_database.report.json'
        if os.path.exists(rpt_path):
            import json
            with open(rpt_path, 'r', encoding='utf-8') as f:
                rp = json.load(f)
            out['tfidf_vocab_size'] = rp.get('tfidf_vocab_size')
            out['tfidf_rows'] = rp.get('rows')
            out['spacy_enrichment'] = rp.get('spacy_model_available', False)
    except Exception:
        pass
    try:
        if os.path.exists('build_training_dataset/training_database.emb.npy'):
            import numpy as np
            arr = np.load('build_training_dataset/training_database.emb.npy')
            out['embeddings_present'] = True
            out['emb_shape'] = list(arr.shape)
    except Exception:
        out['embeddings_present'] = False
    return out

def apply_custom_css():
    """Apply custom CSS for enhanced UI"""
    st.markdown("""
    <style>
        /* Hide sidebar */
        [data-testid="stSidebar"] {
            display: none;
        }
        
        /* Main container */
        .main > div {
            padding-top: 2rem;
        }
        
        /* Header styling */
        h1 {
            color: #1f77b4;
            font-weight: 700;
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Subheader styling */
        h2, h3 {
            color: #2c3e50;
            margin-top: 1.5rem;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            background-color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e3f2fd;
            border-color: #2196f3;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        
        /* File uploader styling */
        .uploadedFile {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 20px;
            background-color: #f8f9fa;
        }
        
        /* Success/Info/Warning boxes */
        .stSuccess, .stInfo, .stWarning {
            border-radius: 8px;
            padding: 15px;
        }
        
        /* Text inputs and areas */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            transition: border-color 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
        }
        
        /* Slider styling */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Progress bars */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Markdown links */
        a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        a:hover {
            color: #764ba2;
            text-decoration: underline;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    # Configure Streamlit
    st.set_page_config(
        page_title="NextHorizon - Your Personalized Career Guide", 
        page_icon="🧭", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load environment
    load_dotenv(override=True)
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize application
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🧭 NextHorizon</h1>
        <p style='font-size: 1.2rem; color: #666; font-weight: 500;'>Your AI-Powered Career Development Partner</p>
        <p style='color: #999; font-size: 0.9rem;'>Upload • Analyze • Match • Learn • Grow</p>
    </div>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Pre-load databases on first run
    if 'databases_loaded' not in st.session_state:
        load_databases()
        st.session_state.databases_loaded = True
    
    # Environment checks (silent)
    check_environment()
    
    # Main content tabs with enhanced styling
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 1-Resume Analysis",
        "🎯 2-Role Matching", 
        "🔍 3-Skill Gap Analysis",
        "📚 4-Learning Path"
    ])
    
    with tab1:
        tab1_render()
    with tab2:
        tab2_render()
    with tab3:
        tab3_render()
    with tab4:
        tab4_render()
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0 1rem 0; color: #999; font-size: 0.9rem;'>
        <hr style='margin-bottom: 1rem; border: none; border-top: 1px solid #e0e0e0;'>
        <p>🚀 Powered by OpenAI GPT-4o-mini & Vector Embeddings | Built with ❤️ for Career Growth</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
