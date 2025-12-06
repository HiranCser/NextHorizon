"""
Script to Generate Skill List from JD Database
This script reads jd_database.csv, extracts all technical and soft skills,
and generates a comprehensive skill_list.csv for later use in training database generation.
"""

import pandas as pd
import json
import os
from pathlib import Path
import warnings
from collections import Counter
from openai import OpenAI

warnings.filterwarnings('ignore')

# File paths
SCRIPT_DIR = Path(__file__).parent
JD_DATABASE = Path(f"{SCRIPT_DIR}/../build_jd_dataset/jd_database.csv")
OUTPUT_FILE = SCRIPT_DIR / "generated_skill_list.csv"

def load_jd_database():
    """Load JD database"""
    print("Loading JD database...")
    if not JD_DATABASE.exists():
        raise FileNotFoundError(f"JD database not found at {JD_DATABASE}")
    
    jd_df = pd.read_csv(JD_DATABASE)
    print(f"✓ Loaded {len(jd_df)} job descriptions\n")
    return jd_df

def extract_skills_with_openai(jd_texts, batch_size=10):
    """
    Use OpenAI to extract all unique skills from JD texts in batches
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    client = OpenAI(api_key=api_key)
    all_skills = set()
    
    # Process in batches
    total = len(jd_texts)
    print(f"Processing {total} job descriptions in batches of {batch_size}...\n")
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_texts = jd_texts[batch_start:batch_end]
        
        print(f"[{batch_end}/{total}] Processing batch {batch_start//batch_size + 1}...")
        
        combined_text = "\n".join([f"JD {i}: {text}" for i, text in enumerate(batch_texts)])
        
        prompt = f"""
        Analyze the following job descriptions and extract ALL unique technical and soft skills mentioned.
        Return a JSON object with two arrays: technical_skills and soft_skills.
        
        Job Descriptions:
        {combined_text}
        
        Return ONLY valid JSON (no markdown, no extra text) with this structure:
        {{
            "technical_skills": ["skill1", "skill2", ...],
            "soft_skills": ["skill1", "skill2", ...]
        }}
        
        Guidelines:
        - Include programming languages, frameworks, libraries, tools, databases, platforms, methodologies
        - Include soft skills like communication, leadership, problem-solving, teamwork, etc.
        - Each skill should be concise and standardized (e.g., "Python", "Machine Learning", "Project Management")
        - Avoid duplicates
        - Return unique skills only
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            all_skills.update(result.get('technical_skills', []))
            all_skills.update(result.get('soft_skills', []))
            
            print(f"   ✓ Extracted {len(result.get('technical_skills', []))} technical skills")
            print(f"   ✓ Extracted {len(result.get('soft_skills', []))} soft skills\n")
            
        except Exception as e:
            print(f"   ✗ Error processing batch: {str(e)}\n")
    
    return list(all_skills)

def extract_all_skills_comprehensive(jd_df):
    """
    Extract all skills from JD database
    """
    print("="*80)
    print("EXTRACTING SKILLS FROM JD DATABASE")
    print("="*80 + "\n")
    
    # Combine all JD texts
    jd_texts = []
    for idx, row in jd_df.iterrows():
        text = str(row.get('jd_text', '')) if pd.notna(row.get('jd_text')) else ''
        if text.strip():
            jd_texts.append(text[:2000])  # Limit text length for API efficiency
    
    # Extract skills using OpenAI
    skills = extract_skills_with_openai(jd_texts)
    
    return skills

def deduplicate_and_normalize_skills(skills):
    """
    Deduplicate and normalize skills
    """
    print("="*80)
    print("DEDUPLICATING AND NORMALIZING SKILLS")
    print("="*80 + "\n")
    
    # Remove duplicates (case-insensitive)
    normalized_skills = {}
    for skill in skills:
        skill_lower = skill.lower().strip()
        if skill_lower and len(skill_lower) > 1:  # Skip very short skills
            if skill_lower not in normalized_skills:
                normalized_skills[skill_lower] = skill
    
    # Sort alphabetically
    unique_skills = sorted(normalized_skills.values(), key=lambda x: x.lower())
    
    print(f"Total Unique Skills: {len(unique_skills)}\n")
    
    return unique_skills

def save_skill_list(skills):
    """Save skill list to CSV"""
    print("="*80)
    print("SAVING SKILL LIST")
    print("="*80 + "\n")
    
    skills_df = pd.DataFrame({
        'skill_name': skills
    })
    
    skills_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✓ Saved {len(skills_df)} unique skills to generated_skill_list.csv\n")
    print("Sample of extracted skills:")
    print(skills_df.head(30).to_string(index=False))
    
    return skills_df

def print_summary_statistics(skills_df, original_jd_count):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    
    print(f"Original JD Count: {original_jd_count}")
    print(f"Unique Skills Generated: {len(skills_df)}")
    
    # Sample skills
    print(f"\nAll {len(skills_df)} Skills (first 50):")
    for i, skill in enumerate(skills_df['skill_name'].head(50).values, 1):
        print(f"  {i:2d}. {skill}")

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("SKILL LIST GENERATION FROM JD DATABASE")
    print("="*80 + "\n")
    
    try:
        # Load JD database
        jd_df = load_jd_database()
        original_count = len(jd_df)
        
        # Extract all skills from JDs
        skills = extract_all_skills_comprehensive(jd_df)
        
        # Deduplicate and normalize
        unique_skills = deduplicate_and_normalize_skills(skills)
        
        # Save skill list
        skills_df = save_skill_list(unique_skills)
        
        # Print statistics
        print_summary_statistics(skills_df, original_count)
        
        print("\n" + "="*80)
        print("SKILL LIST GENERATION COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Fatal Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
