
# FILE: ui/skill_gaps.py
from __future__ import annotations
import streamlit as st
import pandas as pd
from utils.skill_analysis import (
    extract_skills_from_jd_text, 
    extract_skills_from_aspirations, 
    get_required_skills_for_role, 
    calculate_skill_gaps
)
from utils.session_helpers import (
    validate_role_selected,
    get_jd_dataframe,
    get_candidate_skills,
    get_current_role,
    get_user_aspirations
)
from utils.skill_clarification import (
    generate_clarification_questions,
    incorporate_clarification_answers
)

def render():
    # Hero section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center;'>
        <h2 style='color: white; margin: 0;'>🔍 Step 3: Identify Your Skill Gaps</h2>
        <p style='color: rgba(255,255,255,0.9); margin-top: 0.5rem;'>
            Discover what skills you need to develop to reach your target role.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we have necessary data
    if not validate_role_selected():
        st.warning("⚠️ **Please complete Steps 1 & 2 first!**")
        st.info("👈 Upload your resume and select a target role from previous steps.")
        return
    
    jd_df = get_jd_dataframe()
    if jd_df.empty:
        st.error("❌ Job database not found.")
        return
    
    role_title = get_current_role()
    st.markdown(f"""
    <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1.5rem;'>
        <p style='margin: 0; color: #666;'>Analyzing skill requirements for:</p>
        <h3 style='margin: 0.5rem 0 0 0; color: #2c3e50;'>🎯 {role_title}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get required skills for the selected role from JD database
    required_skills = get_required_skills_for_role(role_title, jd_df)
    
    if not required_skills:
        st.warning(f"⚠️ No job descriptions found for role '{role_title}' in the database.")
        return
    
    # Get candidate's current skills
    candidate_skills = get_candidate_skills()
    
    # Also get skills from aspirations
    user_aspirations = get_user_aspirations()
    if user_aspirations:
        aspirations_skills = extract_skills_from_aspirations(user_aspirations)
        candidate_skills.extend(aspirations_skills)
        candidate_skills = list(set(candidate_skills))  # Remove duplicates
    
    # Calculate skill gaps using the shared utility function
    gaps, matched_skills = calculate_skill_gaps(candidate_skills, required_skills)
    
    # Store skill gaps in session state for use in course recommendations
    st.session_state.skill_gaps = gaps
    st.session_state.matched_skills = matched_skills
    
    # Display results with enhanced UI
    st.markdown("### 📊 Skill Analysis Results")
    
    # Summary cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin: 0; font-size: 2.5rem;'>{len(matched_skills)}</h2>
            <p style='margin: 0.5rem 0 0 0;'>✅ Skills Matched</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin: 0; font-size: 2.5rem;'>{len(gaps)}</h2>
            <p style='margin: 0.5rem 0 0 0;'>📈 Skills to Learn</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_required = len(matched_skills) + len(gaps)
        match_percent = int((len(matched_skills) / total_required * 100)) if total_required > 0 else 0
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
            <h2 style='margin: 0; font-size: 2.5rem;'>{match_percent}%</h2>
            <p style='margin: 0.5rem 0 0 0;'>🎯 Role Readiness</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h3 style='margin: 0; color: #4caf50;'>✅ Your Current Skills</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if matched_skills:
            for skill in matched_skills[:15]:
                st.markdown(f"""
                <div style='background: white; padding: 0.7rem; margin: 0.3rem 0; 
                            border-radius: 8px; border-left: 4px solid #4caf50;'>
                    • {skill}
                </div>
                """, unsafe_allow_html=True)
            if len(matched_skills) > 15:
                st.info(f"💡 Plus {len(matched_skills) - 15} more skills!")
        else:
            st.info("No matching skills detected from job requirements.")
    
    with col2:
        st.markdown("""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
            <h3 style='margin: 0; color: #ff9800;'>📈 Skills to Develop</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if gaps:
            for skill in gaps[:15]:
                st.markdown(f"""
                <div style='background: white; padding: 0.7rem; margin: 0.3rem 0; 
                            border-radius: 8px; border-left: 4px solid #ff9800;'>
                    • {skill}
                </div>
                """, unsafe_allow_html=True)
            if len(gaps) > 15:
                st.info(f"💡 Plus {len(gaps) - 15} more skills to explore!")
        else:
            st.success("🎉 Excellent! No major skill gaps detected!")
    
    # Clarification questions (if applicable)
    if gaps:
        st.markdown("---")
        st.markdown("### 🤔 Clarification Questions")
        
        # Generate questions based on the skill gaps
        questions = generate_clarification_questions(st.session_state.structured_json, gaps[:10])  # Limit to top 10 gaps
        
        # Debug information
        with st.expander("🔍 Debug Info", expanded=False):
            st.write(f"Number of skill gaps: {len(gaps)}")
            st.write(f"Current technical skills: {len(st.session_state.structured_json.get('technical_skills', []))}")
            st.write(f"Questions generated: {len(questions)}")
        
        if questions:
            with st.form("clarify_skill_gaps"):
                answers = {}
                for i, q in enumerate(questions):
                    if getattr(q, "options", None):
                        val = st.multiselect(q.text, q.options, default=[])
                    else:
                        val = st.text_input(q.text, value="")
                    answers[q.id] = val
                
                submitted = st.form_submit_button("✅ Apply Answers")
                
            if submitted:
                # Apply the clarification answers
                new_json = incorporate_clarification_answers({k:v for k,v in answers.items() if v}, st.session_state.structured_json)
                st.session_state.structured_json = new_json
                st.success("Applied clarification answers successfully!")
                
                # Recalculate skill gaps with updated information
                updated_candidate_skills = new_json.get("technical_skills", [])
                
                # Also include aspirations skills in updated calculation
                if user_aspirations:
                    updated_aspirations_skills = extract_skills_from_aspirations(user_aspirations)
                    updated_candidate_skills.extend(updated_aspirations_skills)
                    updated_candidate_skills = list(set(updated_candidate_skills))  # Remove duplicates
                
                # Recalculate gaps using shared utility
                updated_gaps, updated_matched_skills = calculate_skill_gaps(updated_candidate_skills, required_skills)
                
                st.info(f"🔄 Updated technical skills: {', '.join(updated_candidate_skills)}")
                
                # Update session state with new gaps
                st.session_state.skill_gaps = updated_gaps
                st.session_state.matched_skills = updated_matched_skills
                
                if len(updated_gaps) < len(gaps):
                    st.success(f"🎉 Great! Reduced skill gaps from {len(gaps)} to {len(updated_gaps)}")
                    if updated_matched_skills:
                        st.info(f"✅ Now matched: {', '.join(updated_matched_skills[-3:])}")  # Show last 3 matched
                else:
                    st.info("Skills updated but gaps remain the same. Make sure you selected the skills you actually have.")
                
                # Refresh the page to show updated analysis
                st.rerun()
        else:
            st.info("No clarification questions needed at this time.")
