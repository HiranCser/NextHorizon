"""
Script to Extract JD Requirements
This script reads jd_database.csv and extracts structured requirements:
JD ID, Role Title, Expected Technical Skills, Expected Soft Skills, Expected Seniority, Expected Years Of Experience
Output is stored in jd_requirements.csv
"""

import pandas as pd
import json
import os
from pathlib import Path
import warnings
from openai import OpenAI

warnings.filterwarnings('ignore')

# File paths
SCRIPT_DIR = Path(__file__).parent
JD_DATABASE = Path(f"{SCRIPT_DIR}/../build_jd_dataset/jd_database.csv")
OUTPUT_FILE = SCRIPT_DIR / "jd_requirements.csv"

def load_jd_database():
    """Load JD database"""
    print("Loading JD database...")
    if not JD_DATABASE.exists():
        raise FileNotFoundError(f"JD database not found at {JD_DATABASE}")
    
    jd_df = pd.read_csv(JD_DATABASE)
    print(f"✓ Loaded {len(jd_df)} job descriptions\n")
    return jd_df

def extract_requirements_with_openai(jd_text, role_title, seniority_level, exp_min, exp_max):
    """
    Use OpenAI to extract structured requirements from JD text
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Analyze the following job description and extract the requirements in JSON format.
    
    Job Description:
    Role Title: {role_title}
    Seniority Level: {seniority_level}
    Minimum Experience: {exp_min} years
    Maximum Experience: {exp_max} years
    
    Description Text:
    {jd_text if jd_text else "No description available"}
    
    Extract and return ONLY valid JSON (no markdown, no extra text) with this structure:
    {{
        "technical_skills": ["skill1", "skill2", ...],
        "soft_skills": ["skill1", "skill2", ...],
        "seniority_level": "Junior|Mid-Level|Senior|Lead|Director|Executive|Unspecified",
        "years_of_experience": {{
            "minimum": <number>,
            "maximum": <number>
        }}
    }}
    
    Guidelines:
    - Extract actual technical skills mentioned (programming languages, frameworks, tools, databases, etc.)
    - Extract soft skills mentioned (communication, leadership, problem-solving, teamwork, etc.)
    - Normalize seniority level to one of the specified values
    - Use provided experience min/max if available, otherwise infer from description
    - Return empty arrays if no specific skills are found
    - Be strict about data types - numbers for experience years, strings for skills
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"✗ Error extracting requirements: {str(e)}")
        return {
            "technical_skills": [],
            "soft_skills": [],
            "seniority_level": seniority_level or "Unspecified",
            "years_of_experience": {
                "minimum": exp_min if pd.notna(exp_min) else None,
                "maximum": exp_max if pd.notna(exp_max) else None
            }
        }

def process_jd_database(jd_df):
    """
    Process each JD and extract requirements
    """
    print("Extracting requirements from JDs...")
    print("="*80 + "\n")
    
    results = []
    total = len(jd_df)
    
    for idx, row in jd_df.iterrows():
        progress = f"[{idx+1}/{total}]"
        
        try:
            jd_id = row.get('jd_id', '')
            role_title = row.get('role_title', '')
            jd_text = row.get('jd_text', '')
            seniority = row.get('seniority_level', '')
            exp_min = row.get('exp_min_years')
            exp_max = row.get('exp_max_years')
            
            print(f"{progress} Processing: {role_title}")
            
            # Extract requirements using OpenAI
            requirements = extract_requirements_with_openai(jd_text, role_title, seniority, exp_min, exp_max)
            
            # Format result
            result = {
                'jd_id': jd_id,
                'role_title': role_title,
                'expected_technical_skills': '|'.join(requirements.get('technical_skills', [])),
                'expected_soft_skills': '|'.join(requirements.get('soft_skills', [])),
                'expected_seniority': requirements.get('seniority_level', 'Unspecified'),
                'expected_years_of_experience_min': requirements.get('years_of_experience', {}).get('minimum'),
                'expected_years_of_experience_max': requirements.get('years_of_experience', {}).get('maximum')
            }
            
            results.append(result)
            print(f"   ✓ Tech Skills: {len(requirements.get('technical_skills', []))} | " 
                  f"Soft Skills: {len(requirements.get('soft_skills', []))} | "
                  f"Seniority: {requirements.get('seniority_level', 'Unspecified')}\n")
            
        except Exception as e:
            print(f"   ✗ Error: {str(e)}\n")
            # Add record with empty fields on error
            results.append({
                'jd_id': row.get('jd_id', ''),
                'role_title': row.get('role_title', ''),
                'expected_technical_skills': '',
                'expected_soft_skills': '',
                'expected_seniority': row.get('seniority_level', 'Unspecified'),
                'expected_years_of_experience_min': row.get('exp_min_years'),
                'expected_years_of_experience_max': row.get('exp_max_years')
            })
    
    return results

def save_results(results):
    """Save results to CSV"""
    print("="*80)
    print(f"\nSaving results to {OUTPUT_FILE}...")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✓ Saved {len(results_df)} records to jd_requirements.csv\n")
    print("Sample of extracted requirements:")
    print(results_df.head(10).to_string())
    
    return results_df

def print_summary_statistics(results_df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal JDs Processed: {len(results_df)}")
    print(f"Total Unique Roles: {results_df['role_title'].nunique()}")
    
    # Technical Skills Summary
    all_tech_skills = []
    for skills_str in results_df['expected_technical_skills']:
        if pd.notna(skills_str) and skills_str.strip():
            all_tech_skills.extend(skills_str.split('|'))
    
    print(f"\nTechnical Skills Extracted: {len(set(all_tech_skills))}")
    if all_tech_skills:
        print(f"Top 10 Technical Skills:")
        from collections import Counter
        skill_counts = Counter(all_tech_skills)
        for skill, count in skill_counts.most_common(10):
            print(f"  - {skill}: {count}")
    
    # Soft Skills Summary
    all_soft_skills = []
    for skills_str in results_df['expected_soft_skills']:
        if pd.notna(skills_str) and skills_str.strip():
            all_soft_skills.extend(skills_str.split('|'))
    
    print(f"\nSoft Skills Extracted: {len(set(all_soft_skills))}")
    if all_soft_skills:
        print(f"Top 10 Soft Skills:")
        from collections import Counter
        skill_counts = Counter(all_soft_skills)
        for skill, count in skill_counts.most_common(10):
            print(f"  - {skill}: {count}")
    
    # Seniority Summary
    print(f"\nSeniority Level Distribution:")
    seniority_counts = results_df['expected_seniority'].value_counts()
    for level, count in seniority_counts.items():
        percentage = count / len(results_df) * 100
        print(f"  - {level}: {count} ({percentage:.1f}%)")
    
    # Experience Summary
    print(f"\nExperience Range Summary:")
    min_exp = results_df['expected_years_of_experience_min'].dropna()
    max_exp = results_df['expected_years_of_experience_max'].dropna()
    
    if len(min_exp) > 0:
        print(f"  Minimum Experience: {min_exp.min():.0f} - {min_exp.max():.0f} years (avg: {min_exp.mean():.1f})")
    if len(max_exp) > 0:
        print(f"  Maximum Experience: {max_exp.min():.0f} - {max_exp.max():.0f} years (avg: {max_exp.mean():.1f})")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("JD REQUIREMENTS EXTRACTION SCRIPT")
    print("="*80 + "\n")
    
    try:
        # Load JD database
        jd_df = load_jd_database()
        
        # Process each JD
        results = process_jd_database(jd_df)
        
        # Save results
        results_df = save_results(results)
        
        # Print statistics
        print_summary_statistics(results_df)
        
        print("\n" + "="*80)
        print("EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Fatal Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
