
# FILE: ui/course_recommendations.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Dict, List, Any
from utils.skill_analysis import get_required_skills_for_role, calculate_skill_gaps
from utils.session_helpers import (
    validate_skill_gaps_completed, 
    get_jd_dataframe, 
    get_training_dataframe,
    get_resume_text,
    get_candidate_skills,
    get_current_role,
    get_skill_gaps
)
from ai.openai_client import openai_rank_courses

def render():
    # Hero section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
        <h2 style='color: white; margin: 0;'>📚 Step 4: Your Personalized Learning Path</h2>
        <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem;'>
            Discover curated courses to bridge your skill gaps and achieve your career goals.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have necessary data
    if not validate_skill_gaps_completed():
        st.warning("⚠️ **Please complete Steps 1, 2, and 3 first!**")
        st.info("👈 Complete the Skill Gap Analysis to see personalized course recommendations.")
        return
    
    jd_df = get_jd_dataframe()
    if jd_df.empty:
        st.error("❌ Job database not found.")
        return
    
    role_title = get_current_role()
    candidate_skills = get_candidate_skills()
    
    # Use stored skill gaps from skill gaps tab (if available) or calculate fresh
    gaps = get_skill_gaps()
    if not gaps:
        # Fallback: calculate gaps if not available in session
        required_skills = get_required_skills_for_role(role_title, jd_df)
        gaps, _ = calculate_skill_gaps(candidate_skills, required_skills)
        st.info("⚠️ Calculating fresh skill gaps. Visit 'Skill Gap Analysis' tab first for better results.")
    
    if not gaps:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
            <h2 style='margin: 0;'>🎉 Congratulations!</h2>
            <p style='margin: 1rem 0 0 0; font-size: 1.1rem;'>
                No major skill gaps detected for <strong>{role_title}</strong>!
            </p>
            <p style='margin: 0.5rem 0 0 0;'>
                You appear well-prepared for this role. Consider exploring advanced or specialized courses for career growth.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1.5rem;'>
        <p style='margin: 0; color: #666;'>Building learning path for:</p>
        <h3 style='margin: 0.5rem 0 0 0; color: #2c3e50;'>🎯 {role_title}</h3>
        <p style='margin: 0.5rem 0 0 0; color: #666; font-size: 0.9rem;'>
            {len(gaps)} skills identified for development
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show information about skill gaps source
    if st.session_state.get("skill_gaps") is not None:
        st.markdown(f"""
        <div style='background: #e3f2fd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <p style='margin: 0;'>
                <strong>🎯 Priority Skills:</strong> {', '.join(gaps[:8])}
                {f' <em>...and {len(gaps) - 8} more</em>' if len(gaps) > 8 else ''}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background: #fff3cd; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
            <p style='margin: 0;'>
                <strong>🎯 Skills to Develop:</strong> {', '.join(gaps[:8])}
                {f' <em>...and {len(gaps) - 8} more</em>' if len(gaps) > 8 else ''}
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #856404;'>
                💡 Complete the Skill Gap Analysis for more accurate recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check for training dataset
    training_df = get_training_dataframe()
    has_training_dataset = not training_df.empty
    
    # Use Vector Search for course recommendations
    if has_training_dataset:
        st.markdown("### 🔍 Find Your Perfect Courses")
        
        if st.button("🚀 Generate Learning Path", key="btn_openai_training", use_container_width=True):
            with st.spinner("🤖 AI is analyzing thousands of courses to find the best matches for you..."):
                resume_text = get_resume_text()
                
                # Filter courses relevant to skill gaps
                relevant_courses = training_df.copy()
                course_count = 0
                
                for idx, gap in enumerate(gaps[:10], 1):  # Show top 10 skill gaps
                    gap_lower = str(gap).lower().strip()
                    if gap_lower:
                        mask = (
                            training_df['skill'].str.lower().str.contains(gap_lower, na=False) |
                            training_df['title'].str.lower().str.contains(gap_lower, na=False) |
                            training_df['description'].str.lower().str.contains(gap_lower, na=False)
                        )
                        if mask.any():
                            gap_courses = training_df[mask].copy()
                            gap_courses_list = gap_courses.to_dict('records')
                            recs = openai_rank_courses([gap], resume_text, gap_courses_list, top_k=5)
                            
                            if recs:
                                # Skill header with enhanced design
                                st.markdown(f"""
                                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                            padding: 1rem; border-radius: 10px; margin: 1.5rem 0 1rem 0;'>
                                    <h3 style='color: white; margin: 0;'>
                                        {idx}. 📘 {gap} <span style='font-size: 0.9rem; opacity: 0.9;'>
                                        ({len(recs)} recommended courses)</span>
                                    </h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for course_idx, course in enumerate(recs, 1):
                                    title = course.get('title', 'Unknown Course')
                                    provider = course.get('provider', 'Unknown')
                                    link = course.get('link', '')
                                    hours = course.get('hours')
                                    price = course.get('price')
                                    rating = course.get('rating')
                                    match_percent = course.get('match_percent', 0)
                                    course_count += 1
                                    
                                    # Build additional info string
                                    info_parts = [f"🏢 <strong>{provider}</strong>"]
                                    if hours:
                                        info_parts.append(f"⏱️ {hours} hours")
                                    if price and price != 'unknown':
                                        info_parts.append(f"💰 {price}")
                                    if rating:
                                        info_parts.append(f"⭐ {rating}/5")
                                    info_parts.append(f"🎯 {match_percent}% match")
                                    
                                    info_text = " • ".join(info_parts)
                                    
                                    # Course card with enhanced design
                                    st.markdown(f"""
                                    <div style='background: white; padding: 1.2rem; margin: 0.8rem 0; 
                                                border-radius: 10px; border: 2px solid #e0e0e0;
                                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                                        <h4 style='margin: 0 0 0.5rem 0; color: #2c3e50;'>
                                            {course_idx}. {title}
                                        </h4>
                                        <p style='color: #666; margin: 0; font-size: 0.9rem;'>
                                            {info_text}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    if link:
                                        st.markdown(f"  [🔗 **Enroll Now**]({link})")
                                    
                                    st.markdown("<br>", unsafe_allow_html=True)
                
                if course_count > 0:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); 
                                padding: 1.5rem; border-radius: 10px; text-align: center; 
                                color: white; margin-top: 2rem;'>
                        <h3 style='margin: 0;'>🎓 {course_count} Courses Found!</h3>
                        <p style='margin: 0.5rem 0 0 0;'>
                            Start learning today and bridge your skill gaps to reach your career goals!
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No specific courses found. Try refining your skill gaps or check back later for updated course database.")
    else:
        st.warning("⚠️ **No training database available.**")
        st.info("Please ensure the training database is properly loaded.")
