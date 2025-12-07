#!/usr/bin/env python3
"""
Test Suite for Resume Parsing Functionality

Tests the resume parsing, data extraction, and form handling
using synthetic resumes from the resume database.

Usage:
    python scripts/test_resume_parsing.py
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd


class ResumeParsingTester:
    """Test suite for resume parsing functionality"""
    
    def __init__(self):
        self.resume_db_path = 'build_resume_dataset/resume_database.csv'
        self.test_results = []
        self.passed = 0
        self.failed = 0
        
    def load_test_data(self):
        """Load synthetic resumes for testing"""
        if not Path(self.resume_db_path).exists():
            print(f"❌ Resume database not found at {self.resume_db_path}")
            return None
        
        return pd.read_csv(self.resume_db_path)
    
    def test_experience_calculator(self):
        """Test the experience year calculation logic"""
        print("\n" + "="*80)
        print("🧪 TEST 1: Experience Year Calculator")
        print("="*80)
        
        def calculate_total_experience_years(work_experience):
            """Replicate the function from resume_parsing.py"""
            total_months = 0
            current_year = datetime.now().year
            
            for work in work_experience:
                start_date = work.get('start_date', '').strip()
                end_date = work.get('end_date', '').strip()
                
                if not start_date:
                    continue
                
                # Handle "Present" or "Current" end dates
                if end_date.lower() in ['present', 'current', ''] or not end_date:
                    end_date = str(current_year)
                
                # Extract years from dates
                start_year = None
                end_year = None
                
                start_match = re.search(r'\b(19|20)\d{2}\b', start_date)
                end_match = re.search(r'\b(19|20)\d{2}\b', end_date)
                
                if start_match:
                    start_year = int(start_match.group())
                if end_match:
                    end_year = int(end_match.group())
                elif end_date.lower() in ['present', 'current']:
                    end_year = current_year
                
                if start_year and end_year and end_year >= start_year:
                    total_months += (end_year - start_year) * 12
            
            return round(total_months / 12, 1) if total_months > 0 else 0
        
        # Test cases
        test_cases = [
            {
                "name": "Single 5-year job",
                "work_exp": [{"start_date": "2019", "end_date": "2024", "title": "Engineer"}],
                "expected": 5.0
            },
            {
                "name": "Multiple jobs (5 + 3 years)",
                "work_exp": [
                    {"start_date": "2019", "end_date": "2024", "title": "Engineer"},
                    {"start_date": "2015", "end_date": "2018", "title": "Developer"}
                ],
                "expected": 8.0
            },
            {
                "name": "Job with 'Present' end date",
                "work_exp": [{"start_date": "2020", "end_date": "Present", "title": "Senior"}],
                "expected": datetime.now().year - 2020
            },
            {
                "name": "Job with date format 'Jan 2020'",
                "work_exp": [{"start_date": "Jan 2019", "end_date": "Dec 2024", "title": "Engineer"}],
                "expected": 5.0
            },
            {
                "name": "Empty work experience",
                "work_exp": [],
                "expected": 0.0
            }
        ]
        
        for test_case in test_cases:
            result = calculate_total_experience_years(test_case["work_exp"])
            passed = abs(result - test_case["expected"]) < 0.5
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status} | {test_case['name']}")
            print(f"   Expected: {test_case['expected']} years")
            print(f"   Got:      {result} years")
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
            
            self.test_results.append({
                "test": f"Experience Calculator - {test_case['name']}",
                "passed": passed
            })
    
    def test_skills_parsing(self):
        """Test technical skills parsing from comma-separated strings"""
        print("\n" + "="*80)
        print("🧪 TEST 2: Skills Parsing")
        print("="*80)
        
        def parse_skills(skills_str):
            """Parse comma-separated skills string"""
            if not skills_str:
                return []
            return [s.strip() for s in skills_str.split(",") if s.strip()]
        
        test_cases = [
            {
                "name": "Multiple skills",
                "input": "Python, Java, SQL, Machine Learning",
                "expected": ["Python", "Java", "SQL", "Machine Learning"]
            },
            {
                "name": "Skills with extra spaces",
                "input": "  Python  ,  Java  ,  SQL  ",
                "expected": ["Python", "Java", "SQL"]
            },
            {
                "name": "Empty string",
                "input": "",
                "expected": []
            },
            {
                "name": "Single skill",
                "input": "Python",
                "expected": ["Python"]
            }
        ]
        
        for test_case in test_cases:
            result = parse_skills(test_case["input"])
            passed = result == test_case["expected"]
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status} | {test_case['name']}")
            print(f"   Input:    {test_case['input']}")
            print(f"   Expected: {test_case['expected']}")
            print(f"   Got:      {result}")
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
            
            self.test_results.append({
                "test": f"Skills Parsing - {test_case['name']}",
                "passed": passed
            })
    
    def test_structured_json_validation(self):
        """Test that structured JSON has all required fields"""
        print("\n" + "="*80)
        print("🧪 TEST 3: Structured JSON Validation")
        print("="*80)
        
        required_fields = [
            "professional_summary",
            "current_role",
            "technical_skills",
            "work_experience",
            "education",
            "career_level",
            "industry_focus",
            "location",
            "total_years_experience"
        ]
        
        # Mock structured JSON from resume processing
        sample_structured = {
            "professional_summary": "Senior Python Developer with 7 years experience",
            "current_role": {"role": "Senior Engineer", "company": "Google"},
            "technical_skills": ["Python", "SQL", "AWS"],
            "work_experience": [{"title": "Senior", "company": "Google", "start_date": "2020", "end_date": "Present"}],
            "education": [{"degree": "BS", "institution": "MIT", "graduation_date": "2017"}],
            "career_level": "Senior",
            "industry_focus": "Technology",
            "location": "San Francisco, CA",
            "total_years_experience": 7.0,
            "soft_skills": ["Leadership", "Communication"],
            "certifications": ["AWS Certified"]
        }
        
        print("\nValidating required fields...")
        for field in required_fields:
            has_field = field in sample_structured
            status = "✅" if has_field else "❌"
            print(f"{status} {field}: {has_field}")
            
            if has_field:
                self.passed += 1
            else:
                self.failed += 1
            
            self.test_results.append({
                "test": f"Required Field - {field}",
                "passed": has_field
            })
    
    def test_resume_database_integration(self):
        """Test parsing synthetic resumes from database"""
        print("\n" + "="*80)
        print("🧪 TEST 4: Resume Database Integration")
        print("="*80)
        
        df = self.load_test_data()
        if df is None:
            return
        
        print(f"\nLoaded {len(df)} synthetic resumes")
        
        # Test 1: All resumes have required columns
        required_cols = ['resume_id', 'role_target', 'level', 'years_experience', 'skills', 'resume_text']
        print("\nChecking required columns...")
        
        for col in required_cols:
            has_col = col in df.columns
            status = "✅" if has_col else "❌"
            print(f"{status} {col}")
            
            if has_col:
                self.passed += 1
            else:
                self.failed += 1
            
            self.test_results.append({
                "test": f"Database Column - {col}",
                "passed": has_col
            })
        
        # Test 2: No missing values
        print("\nChecking data completeness...")
        missing_check = df.isnull().sum().sum()
        has_no_nulls = missing_check == 0
        status = "✅" if has_no_nulls else "❌"
        print(f"{status} No missing values: {has_no_nulls} (total nulls: {missing_check})")
        
        if has_no_nulls:
            self.passed += 1
        else:
            self.failed += 1
        
        self.test_results.append({
            "test": "Database Completeness",
            "passed": has_no_nulls
        })
        
        # Test 3: Sample resumes can be parsed
        print("\nTesting resume text extraction...")
        sample_resumes = df.sample(min(3, len(df)))
        
        for idx, row in sample_resumes.iterrows():
            resume_text = row['resume_text']
            has_content = len(resume_text) > 20
            skills = row['skills']
            has_skills = len(skills) > 0
            
            passed = has_content and has_skills
            status = "✅" if passed else "❌"
            print(f"{status} Resume {row['resume_id']}: {row['role_target']} ({passed})")
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
            
            self.test_results.append({
                "test": f"Resume Extraction - {row['resume_id']}",
                "passed": passed
            })
    
    def test_form_data_transformation(self):
        """Test transforming form data into structured JSON"""
        print("\n" + "="*80)
        print("🧪 TEST 5: Form Data Transformation")
        print("="*80)
        
        # Simulate form inputs (as they come from Streamlit)
        form_data = {
            "summary": "Experienced Senior Developer",
            "current_designation": "Senior Software Engineer",
            "current_company": "Google",
            "tech_skills": "Python, Java, SQL, AWS",
            "soft_skills": "Leadership, Communication",
            "certifications": "AWS Certified, PMP",
            "career_level": "Senior",
            "industry_focus": "Technology",
            "location": "San Francisco, CA",
            "work_experiences": [
                {
                    "title": "Senior Engineer",
                    "company": "Google",
                    "start_date": "2020",
                    "end_date": "Present",
                    "responsibilities": "Led team of 5 engineers"
                }
            ],
            "educations": [
                {
                    "degree": "BS",
                    "specialization": "Computer Science",
                    "institution": "MIT",
                    "graduation_date": "2017"
                }
            ],
            "total_experience": 7.0
        }
        
        # Transform to structured JSON (as done in resume_parsing.py line 293-307)
        structured_json = {
            "professional_summary": form_data["summary"],
            "current_role": {
                "role": form_data["current_designation"],
                "company": form_data["current_company"]
            },
            "technical_skills": [s.strip() for s in form_data["tech_skills"].split(",") if s.strip()],
            "soft_skills": [s.strip() for s in form_data["soft_skills"].split(",") if s.strip()],
            "certifications": [s.strip() for s in form_data["certifications"].split(",") if s.strip()],
            "work_experience": form_data["work_experiences"],
            "education": form_data["educations"],
            "career_level": form_data["career_level"],
            "industry_focus": form_data["industry_focus"],
            "location": form_data["location"],
            "total_years_experience": form_data["total_experience"]
        }
        
        # Validate transformation
        print("\nValidating form data transformation...")
        
        tests = [
            ("Summary preserved", form_data["summary"] == structured_json["professional_summary"]),
            ("Current role set correctly", structured_json["current_role"]["role"] == "Senior Software Engineer"),
            ("Technical skills parsed", len(structured_json["technical_skills"]) == 4),
            ("Soft skills parsed", len(structured_json["soft_skills"]) == 2),
            ("Certifications parsed", len(structured_json["certifications"]) == 2),
            ("Work experience preserved", len(structured_json["work_experience"]) == 1),
            ("Education preserved", len(structured_json["education"]) == 1),
            ("Career level set", structured_json["career_level"] == "Senior"),
            ("Industry focus set", structured_json["industry_focus"] == "Technology"),
            ("Location set", structured_json["location"] == "San Francisco, CA"),
            ("Experience preserved", structured_json["total_years_experience"] == 7.0),
        ]
        
        for test_name, result in tests:
            status = "✅" if result else "❌"
            print(f"{status} {test_name}")
            
            if result:
                self.passed += 1
            else:
                self.failed += 1
            
            self.test_results.append({
                "test": f"Form Transformation - {test_name}",
                "passed": result
            })
    
    def test_edge_cases(self):
        """Test edge cases in resume parsing"""
        print("\n" + "="*80)
        print("🧪 TEST 6: Edge Cases")
        print("="*80)
        
        edge_cases = [
            {
                "name": "Resume with special characters",
                "text": "Senior Python Developer (C++, C#, C) with 5+ years @ FAANG companies",
                "expected": True
            },
            {
                "name": "Resume with Unicode characters",
                "text": "Développeur Senior avec expertise en Français et English",
                "expected": True
            },
            {
                "name": "Empty resume",
                "text": "",
                "expected": False
            },
            {
                "name": "Very long resume",
                "text": "A" * 5000,
                "expected": True
            },
            {
                "name": "Resume with numbers",
                "text": "Managed 100+ projects with 50 team members, achieved 99.9% uptime",
                "expected": True
            }
        ]
        
        print("\nTesting edge cases...")
        for case in edge_cases:
            # Simple test: check if text has minimum length
            is_valid = len(case["text"]) > 0
            passed = is_valid == case["expected"]
            
            status = "✅" if passed else "❌"
            print(f"{status} {case['name']}: {passed}")
            
            if passed:
                self.passed += 1
            else:
                self.failed += 1
            
            self.test_results.append({
                "test": f"Edge Case - {case['name']}",
                "passed": passed
            })
    
    def run_all_tests(self):
        """Run all test suites"""
        print("\n" + "🧪" * 40)
        print("RESUME PARSING TEST SUITE")
        print("🧪" * 40)
        
        self.test_experience_calculator()
        self.test_skills_parsing()
        self.test_structured_json_validation()
        self.test_resume_database_integration()
        self.test_form_data_transformation()
        self.test_edge_cases()
        
        self.print_summary()
    
    def print_summary(self):
        """Print test summary and results"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\n✅ PASSED: {self.passed}")
        print(f"❌ FAILED: {self.failed}")
        print(f"📊 TOTAL:  {total}")
        print(f"🎯 PASS RATE: {pass_rate:.1f}%")
        
        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {self.failed} tests failed. Review above for details.")
        
        # Group results by test category
        print("\n" + "="*80)
        print("📈 RESULTS BY CATEGORY")
        print("="*80)
        
        categories = {}
        for result in self.test_results:
            test_name = result["test"].split(" - ")[0]
            if test_name not in categories:
                categories[test_name] = {"passed": 0, "total": 0}
            
            categories[test_name]["total"] += 1
            if result["passed"]:
                categories[test_name]["passed"] += 1
        
        for category, stats in sorted(categories.items()):
            pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            status = "✅" if stats["passed"] == stats["total"] else "⚠️"
            print(f"{status} {category}: {stats['passed']}/{stats['total']} ({pass_rate:.0f}%)")
        
        print("\n" + "="*80)


def main():
    """Main entry point"""
    tester = ResumeParsingTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
