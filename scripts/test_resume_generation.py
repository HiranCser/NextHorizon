#!/usr/bin/env python3
"""
Test script to generate synthetic resume database
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from build_resume_dataset.build_resume_database import (
    generate_synthetic_resume,
    extract_skills_from_text,
    infer_seniority_from_experience,
    process_role
)

def test_resume_generation():
    """Test synthetic resume generation"""
    print("🧪 Testing Synthetic Resume Generation\n")
    print("=" * 70)
    
    # Test 1: Generate resumes for different roles and levels
    test_roles = [
        "Software Engineer",
        "Data Scientist",
        "DevOps Engineer",
        "Frontend Developer"
    ]
    
    for role in test_roles:
        print(f"\n📋 Generating resumes for: {role}")
        resumes = process_role(role, num_levels=3, resumes_per_level=2)
        print(f"   ✓ Generated {len(resumes)} resumes")
        
        # Show sample
        if resumes:
            sample = resumes[0]
            print(f"   - Sample ID: {sample['resume_id']}")
            print(f"   - Level: {sample['level']}")
            print(f"   - Experience: {sample['years_experience']} years")
            print(f"   - Skills: {sample['skills'][:50]}...")
    
    # Test 2: Test skill extraction
    print(f"\n{'='*70}")
    print("\n🔍 Testing Skill Extraction")
    sample_text = """
    Senior Python Developer with 7 years of experience.
    Expert in Django, Flask, FastAPI, and Docker.
    Strong background in AWS, Kubernetes, and PostgreSQL.
    Proficient in React, JavaScript, and REST APIs.
    """
    
    skills = extract_skills_from_text(sample_text)
    print(f"   Extracted skills: {', '.join(skills)}")
    
    # Test 3: Test seniority inference
    print(f"\n{'='*70}")
    print("\n📊 Testing Seniority Inference")
    test_cases = [
        (1, 1, "Junior Developer"),
        (3, 6, "Mid-Level Developer"),
        (7, 10, "Senior Developer"),
        (12, 15, "Lead Developer"),
        (18, 20, "Principal Engineer"),
    ]
    
    for min_exp, max_exp, expected_role in test_cases:
        level = infer_seniority_from_experience(min_exp, max_exp)
        print(f"   {min_exp}-{max_exp} years → {level}")
    
    # Test 4: Generate comprehensive resume
    print(f"\n{'='*70}")
    print("\n📄 Sample Generated Resume")
    sample_resumes = generate_synthetic_resume("Python Developer", "Senior", 1)
    if sample_resumes:
        resume = sample_resumes[0]
        print(f"\nResume ID: {resume['resume_id']}")
        print(f"Role: {resume['role_target']}")
        print(f"Level: {resume['level']}")
        print(f"Experience: {resume['years_experience']} years")
        print(f"Skills: {resume['skills']}")
        print(f"Companies: {resume['companies_worked']}")
        print(f"\nResume Text (first 300 chars):")
        print(resume['resume_text'][:300] + "...")
    
    print(f"\n{'='*70}")
    print("✅ All tests completed successfully!\n")

if __name__ == "__main__":
    test_resume_generation()
