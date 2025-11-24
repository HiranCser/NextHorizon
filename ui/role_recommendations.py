
# FILE: ui/role_recommendations.py
from __future__ import annotations
import streamlit as st
import pandas as pd

from ai.openai_client import openai_rank_roles, openai_rank_jds
from utils.resume_processor import build_resume_text

def _get_resume_text() -> str:
    """Get resume text using the utility function"""
    return build_resume_text()

def _get_jd_df() -> pd.DataFrame:
    df = st.session_state.get("jd_df")
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def render():
    # Hero section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
        <h2 style='color: white; margin: 0;'>🎯 Step 2: Find Your Perfect Role</h2>
        <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem;'>
            AI-powered role matching based on your skills, experience, and career aspirations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if resume is uploaded and parsed (structured_json exists)
    if not st.session_state.get("structured_json"):
        st.warning("⚠️ **Please complete Step 1 first!**")
        st.info("👈 Go to the **Resume Analysis** tab to upload and parse your resume.")
        return
    
    # Career aspirations section
    st.markdown("### 💭 Tell Us About Your Career Goals")
    user_aspirations = st.text_area(
        "What are your career aspirations?", 
        value=st.session_state.get("user_aspirations", ""),
        height=120,
        help="Share your career goals, desired roles, target industries, or skills you want to develop. This helps us find the best matches for you!",
        key="aspirations_input",
        placeholder="Example: I want to transition into a Data Science role in the healthcare industry, focusing on machine learning and predictive analytics..."
    )
    
    # Save aspirations to session state
    st.session_state.user_aspirations = user_aspirations
    
    
    resume_text = _get_resume_text()
    
    # Include user aspirations in the resume text for matching
    if user_aspirations.strip():
        resume_text += f" Career Aspirations: {user_aspirations}"
    
    jd_df = _get_jd_df()

    if not resume_text:
        st.info("📝 Complete the Resume Analysis section for better role matching.")
        return
    if jd_df.empty:
        st.error("❌ Job database not found. Please ensure the database is loaded.")
        return

    st.markdown("### 🎯 Personalized Role Recommendations")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Adjust the number of role suggestions:**")
    with col2:
        k = st.slider("Top matches", 1, 10, 5, key="asp_k_web", label_visibility="collapsed")
    
    # Validate JD database structure
    required_cols = ['role_title', 'jd_text']
    missing_cols = [col for col in required_cols if col not in jd_df.columns]
    if missing_cols:
        st.error(f"❌ Database error: Missing columns {missing_cols}")
        return
        
    grp = jd_df.groupby("role_title")["jd_text"].apply(lambda s: " ".join(s.astype(str).fillna("").tolist()[:20]))
    snippets = [{"title": r, "link": "", "snippet": txt, "source": "jd_db"} for r, txt in grp.items() if r and txt]
    
    with st.spinner("🔍 Analyzing roles and matching with your profile..."):
        ranked_roles = openai_rank_roles(resume_text, snippets, top_k=k)

    st.markdown("#### 📊 Your Top Matching Roles")
    
    for i, p in enumerate(ranked_roles, 1):
        match_score = int(round(p['score']*100, 0))
        
        # Color code based on match percentage
        if match_score >= 80:
            badge_color = "#4caf50"
            emoji = "🌟"
        elif match_score >= 60:
            badge_color = "#2196f3"
            emoji = "⭐"
        else:
            badge_color = "#ff9800"
            emoji = "💡"
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid {badge_color};'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <strong style='font-size: 1.1rem;'>{emoji} {i}. {p['role_title']}</strong>
                </div>
                <div style='background: {badge_color}; color: white; padding: 0.3rem 1rem; 
                            border-radius: 20px; font-weight: bold;'>
                    {match_score}% Match
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💼 Explore Specific Job Openings")
    st.markdown("Select a role to see the top 5 job descriptions that match your profile:")
    
    roles = [p["role_title"] for p in ranked_roles]
    sel = st.selectbox("Choose a role", roles, key="asp_sel_role_openai")
    
    if st.button("🔎 Find Job Openings", use_container_width=True):
        with st.spinner("🔍 Finding the best job matches for you..."):
            rows = jd_df[jd_df["role_title"]==sel]
            jd_rows = rows[["role_title","company","source_title","source_url","jd_text"]].to_dict(orient="records")
            items = openai_rank_jds(resume_text, jd_rows, top_k=5)
            
        if items:
            st.markdown(f"#### 📋 Top Job Openings for **{sel}**")
            for idx, it in enumerate(items, 1):
                st.markdown(f"""
                <div style='background: white; padding: 1.2rem; border-radius: 10px; 
                            margin: 1rem 0; border: 2px solid #e0e0e0;'>
                    <div style='display: flex; justify-content: space-between; align-items: start;'>
                        <div style='flex: 1;'>
                            <h4 style='margin: 0; color: #2c3e50;'>{idx}. {it['title']}</h4>
                            <p style='color: #666; margin: 0.3rem 0;'>🏢 {it['company']}</p>
                        </div>
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 0.4rem 1rem; border-radius: 20px; 
                                    font-weight: bold; white-space: nowrap;'>
                            {it['match_percent']}% Match
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if it.get("link"):
                    st.markdown(f"  → [🔗 View Job Posting]({it['link']})")
        else:
            st.info("No job openings found for this role.")
    else:
        st.info("No JDs found for that role in your JD database.")

    # Save the selected role to session state for skill gaps analysis
    if sel and st.button("Save Selected Role for Skill Analysis", key="save_role_for_skills"):
        st.session_state.chosen_role_title = sel
        # Clear previous skill gaps to ensure fresh analysis for new role
        if "skill_gaps" in st.session_state:
            del st.session_state.skill_gaps
        if "matched_skills" in st.session_state:
            del st.session_state.matched_skills
        st.success(f"Selected role '{sel}' saved for skill gap analysis.")
