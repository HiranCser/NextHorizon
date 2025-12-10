
# FILE: ui/role_recommendations.py
from __future__ import annotations
from typing import Any, cast
import streamlit as st
import pandas as pd

from ai.openai_client import openai_rank_roles_enhanced, openai_rank_jds_enhanced
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
    
    # Clear previous selections when coming back to this page
    if "last_rendered_tab" not in st.session_state:
        st.session_state.last_rendered_tab = "role_recommendations"
    elif st.session_state.last_rendered_tab != "role_recommendations":
        # User has navigated away and come back, clear selections
        if "available_jd_items" in st.session_state:
            del st.session_state.available_jd_items
        # Clear individual checkbox states
        for key in list(st.session_state.keys()):
            if key.startswith("jd_select_"):
                del st.session_state[key]
    
    st.session_state.last_rendered_tab = "role_recommendations"
    
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
    if user_aspirations and user_aspirations.strip():
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
        ranked_roles = openai_rank_roles_enhanced(resume_text, snippets, top_k=k,
                                                 similarity_method="cosine", preprocess_text=True)

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
        # Clear previous selections when finding new job openings
        st.session_state.jd_selections = {}
        
        rows = jd_df[jd_df["role_title"]==sel]
        jd_rows: list[dict[str, Any]] = cast(list[dict[str, Any]], rows[["role_title","company","source_title","source_url","jd_text"]].to_dict(orient="records"))
        items = openai_rank_jds_enhanced(resume_text, jd_rows, top_k=5,
                                       similarity_method="cosine", preprocess_text=True)
        
        # Store items AND original jd_rows in session state so we have access to full jd_text
        st.session_state.current_jd_items = items
        st.session_state.current_jd_rows = jd_rows  # Keep original rows with jd_text
        st.session_state.current_selected_role = sel
    
    # Display items if they exist in session state
    if st.session_state.get("current_jd_items"):
        items = st.session_state.current_jd_items
        jd_rows = st.session_state.get("current_jd_rows", [])
        sel = st.session_state.get("current_selected_role", "")
        
        st.markdown(f"#### 📋 Top Job Openings for **{sel}**")
        st.markdown("💡 **Tip:** Select the job openings you want to consider for skill gap analysis below:")
        
        # Store items in session state for selection
        st.session_state.available_jd_items = items
        
        # Initialize selection tracking in session state
        if "jd_selections" not in st.session_state:
            st.session_state.jd_selections = {}
        
        # Create checkboxes for each job opening - use index as unique identifier
        selected_items = []
        for idx, it in enumerate(items, 1):
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                is_selected = st.checkbox(
                    f"Select", 
                    key=f"jd_select_{idx}",
                    value=st.session_state.jd_selections.get(f"jd_select_{idx}", False)
                )
                # Track selection without modifying after creation
                st.session_state.jd_selections[f"jd_select_{idx}"] = is_selected
            with col2:
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
            
            if is_selected:
                # Find the corresponding jd_row to get full jd_text
                matching_row = next((row for row in jd_rows if row.get("source_title") == it.get("title")), None)
                if matching_row:
                    selected_items.append({
                        **it,
                        "jd_text": matching_row.get("jd_text", "")
                    })
        
        # Save selected items and the role for skill gaps analysis
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✅ Select All", use_container_width=True, key="select_all_jds"):
                # Update tracking dict before rerun - do it all at once
                st.session_state.jd_selections = {f"jd_select_{idx}": True for idx in range(1, len(items) + 1)}
                st.rerun()
        
        with col2:
            if st.button("❌ Clear All", use_container_width=True, key="clear_all_jds"):
                # Clear all selections
                st.session_state.jd_selections = {f"jd_select_{idx}": False for idx in range(1, len(items) + 1)}
                st.rerun()
        
        with col3:
            if st.button("💾 Save Selected", use_container_width=True, key="save_selected_jds_button"):
                if selected_items:
                    st.session_state.chosen_role_title = sel
                    st.session_state.selected_jd_items = selected_items
                    st.session_state.selected_jd_count = len(selected_items)
                    # Clear previous skill gaps to ensure fresh analysis for new role
                    if "skill_gaps" in st.session_state:
                        del st.session_state.skill_gaps
                    if "matched_skills" in st.session_state:
                        del st.session_state.matched_skills
                    st.success(f"✅ Saved role '{sel}' with {len(selected_items)} selected job opening(s) for skill gap analysis.")
                else:
                    st.warning("⚠️ Please select at least one job opening before saving.")
        
        # Display selected job openings summary
        if selected_items:
            st.markdown("---")
            st.markdown("### ✅ Selected Job Openings Summary")
            st.markdown(f"**Total Selected:** {len(selected_items)} out of {len(items)} job opening(s)")
            
            selected_col1, selected_col2 = st.columns([1, 2])
            
            with selected_col1:
                st.markdown("**Selected Positions:**")
                for idx, item in enumerate(selected_items, 1):
                    st.markdown(f"- {idx}. {item['title']}")
            
            with selected_col2:
                st.markdown("**Match Scores:**")
                for item in selected_items:
                    match_percent = item.get('match_percent', 'N/A')
                    st.markdown(f"- **{item['title']}**: {match_percent}%")
            
            # Show average match score
            try:
                match_scores = [int(item.get('match_percent', 0)) for item in selected_items]
                avg_match = sum(match_scores) / len(match_scores) if match_scores else 0
                st.markdown(f"**Average Match Score:** {int(avg_match)}%")
            except:
                pass
    else:
        st.info("Click 'Find Job Openings' to explore available positions for the selected role.")
