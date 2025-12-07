#!/usr/bin/env python3
"""
Synthetic Data Generator for NextHorizon

Generates realistic synthetic:
- Resumes (user profiles)
- Job Descriptions (JD database)
- Courses (training recommendations)

This helps improve resume-to-JD matching by providing more training examples.

Usage:
    python scripts/generate_synthetic_data.py --type resume --count 100 --output synthetic_resumes.csv
    python scripts/generate_synthetic_data.py --type jd --count 50 --output synthetic_jds.csv
    python scripts/generate_synthetic_data.py --type all --count 100 --output synthetic_data/
"""

from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SyntheticDataGenerator:
    """Generate realistic synthetic data for NextHorizon."""

    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        self.random_seed = random_seed

    # ===================== RESUME DATA GENERATION =====================

    SKILL_SETS = {
        'backend': ['Python', 'Java', 'Node.js', 'C#', 'Go', 'Rust', 'PostgreSQL', 'MongoDB', 'API Design', 'Microservices', 'AWS', 'Docker', 'Kubernetes'],
        'frontend': ['React', 'Angular', 'Vue.js', 'TypeScript', 'CSS', 'HTML', 'JavaScript', 'Redux', 'REST APIs', 'GraphQL', 'Material-UI', 'Responsive Design'],
        'fullstack': ['React', 'Node.js', 'MongoDB', 'Express.js', 'JavaScript', 'TypeScript', 'AWS', 'Docker', 'PostgreSQL', 'REST APIs', 'Git', 'Agile'],
        'ml': ['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy', 'SQL', 'Jupyter', 'Statistics', 'Deep Learning', 'NLP', 'Computer Vision'],
        'devops': ['Docker', 'Kubernetes', 'AWS', 'Azure', 'CI/CD', 'Jenkins', 'GitLab', 'Terraform', 'Ansible', 'Linux', 'Monitoring', 'Infrastructure'],
        'data': ['SQL', 'Python', 'Tableau', 'Power BI', 'Excel', 'Statistics', 'ETL', 'Data Warehousing', 'Spark', 'Hadoop', 'Pandas', 'R'],
        'qa': ['Testing', 'Selenium', 'Jest', 'Pytest', 'SQL', 'Automation', 'JIRA', 'Test Planning', 'API Testing', 'Load Testing', 'Agile', 'Python'],
    }

    ROLES = {
        'backend': 'Backend Engineer',
        'frontend': 'Frontend Engineer',
        'fullstack': 'Full Stack Developer',
        'ml': 'Machine Learning Engineer',
        'devops': 'DevOps Engineer',
        'data': 'Data Analyst',
        'qa': 'QA Engineer',
    }

    COMPANIES = [
        'Google', 'Amazon', 'Meta', 'Apple', 'Microsoft', 'Netflix', 'Uber', 'Airbnb',
        'Stripe', 'Shopify', 'Slack', 'Discord', 'GitHub', 'GitLab', 'Figma', 'Notion',
        'TechCorp', 'DataSystems', 'CloudLabs', 'FintechXYZ', 'MedTech Inc'
    ]

    COMPANIES_EXPERIENCE = {
        'startup': ['StartupXYZ', 'TechBootstrap', 'InnovateCo', 'RocketLabs', 'QuickStart'],
        'scale': ['ScaleUp Tech', 'GrowthCo', 'ExpandNow', 'RisingStars', 'NextGen'],
        'enterprise': ['Fortune500', 'EnterpriseGlobal', 'BigTech', 'LargeScale', 'Corporate Inc'],
    }

    def generate_resume(self, resume_id: int) -> Dict[str, Any]:
        """Generate a synthetic resume."""
        role_type = random.choice(list(self.SKILL_SETS.keys()))
        years_exp = random.randint(1, 15)
        level = 'Junior' if years_exp < 3 else 'Mid' if years_exp < 8 else 'Senior'

        # Select relevant skills
        skills = random.sample(self.SKILL_SETS[role_type], k=random.randint(5, 8))

        # Add some random other skills (realistic cross-skilling)
        other_skills = []
        for sk_type in random.sample(list(self.SKILL_SETS.keys()), k=2):
            other_skills.extend(random.sample(self.SKILL_SETS[sk_type], k=1))
        skills.extend(other_skills)

        # Generate work experience
        experiences = []
        remaining_years = years_exp
        company_types = ['startup', 'scale', 'enterprise']

        for i in range(max(1, years_exp // 3)):
            max_exp = max(1, min(5, remaining_years))
            exp_years = random.randint(1, max_exp)
            remaining_years -= exp_years
            company = random.choice(self.COMPANIES_EXPERIENCE.get(random.choice(company_types), self.COMPANIES))
            title = f"{random.choice(['Senior', 'Lead', 'Principal'])} {self.ROLES[role_type]}" if i == 0 else self.ROLES[role_type]
            experiences.append(f"{exp_years} years as {title} at {company}")
            if remaining_years <= 0:
                break

        # Generate education
        education = random.choice([
            "BS in Computer Science",
            "MS in Computer Science",
            "BS in Engineering",
            "Bootcamp Graduate - Full Stack Development",
            "Self-taught with online certifications"
        ])

        resume_text = f"""
        Professional Role: {level} {self.ROLES[role_type]}
        Years of Experience: {years_exp}
        
        Skills: {', '.join(skills)}
        
        Work Experience:
        {chr(10).join(experiences)}
        
        Education: {education}
        
        Certifications: AWS Certified Solutions Architect, {random.choice(['Kubernetes Certified', 'Google Cloud Certified', 'Azure Certified'])}
        """

        return {
            'id': f"resume_{resume_id:04d}",
            'role_type': role_type,
            'level': level,
            'years_experience': years_exp,
            'skills': '; '.join(skills),
            'resume_text': resume_text.strip(),
            'primary_role': self.ROLES[role_type],
        }

    # ===================== JOB DESCRIPTION GENERATION =====================

    JD_TEMPLATES = {
        'backend': {
            'title_templates': [
                'Senior Backend Engineer - {company}',
                'Backend Software Engineer - {company}',
                'Python/Go Backend Developer - {company}',
                'API & Microservices Engineer - {company}',
            ],
            'description': 'Design and build scalable backend systems. Work with {tools} technologies. {team_size} engineer team.',
            'requirements': ['5+ years backend development', 'Proficiency in {language}', 'Experience with microservices', 'Strong database knowledge']
        },
        'frontend': {
            'title_templates': [
                'Senior Frontend Engineer - {company}',
                'React Developer - {company}',
                'UI/UX Frontend Developer - {company}',
                'JavaScript Engineer - {company}',
            ],
            'description': 'Build beautiful user interfaces with {tools}. Work on high-traffic web applications. {team_size} engineer team.',
            'requirements': ['3+ years frontend development', 'Expert in {framework}', 'CSS/HTML proficiency', 'Performance optimization']
        },
        'fullstack': {
            'title_templates': [
                'Full Stack Engineer - {company}',
                'Full Stack Developer - {company}',
                'Senior Full Stack Engineer - {company}',
            ],
            'description': 'Own features end-to-end with {tools}. Build scalable applications. {team_size} engineer team.',
            'requirements': ['4+ years full stack experience', 'Full stack expertise', 'Database design', 'DevOps fundamentals']
        },
        'ml': {
            'title_templates': [
                'Machine Learning Engineer - {company}',
                'ML/AI Engineer - {company}',
                'Senior ML Engineer - {company}',
            ],
            'description': 'Build ML models and AI systems. Work with {tools} stack. {team_size} ML/AI team.',
            'requirements': ['3+ years ML experience', 'Python + TensorFlow', 'Statistics knowledge', 'Deep learning experience']
        },
        'devops': {
            'title_templates': [
                'DevOps Engineer - {company}',
                'Senior DevOps Engineer - {company}',
                'Infrastructure Engineer - {company}',
            ],
            'description': 'Manage cloud infrastructure with {tools}. Build CI/CD pipelines. {team_size} infrastructure team.',
            'requirements': ['5+ years DevOps', 'Kubernetes expertise', 'Infrastructure as Code', 'AWS/Azure/GCP']
        },
        'data': {
            'title_templates': [
                'Data Analyst - {company}',
                'Senior Data Analyst - {company}',
                'Analytics Engineer - {company}',
            ],
            'description': 'Analyze data with {tools}. Build dashboards and reports. {team_size} data team.',
            'requirements': ['3+ years data analysis', 'SQL mastery', 'Tableau/Power BI', 'Statistical analysis']
        },
    }

    def generate_jd(self, jd_id: int) -> Dict[str, Any]:
        """Generate a synthetic job description."""
        role_type = random.choice(list(self.JD_TEMPLATES.keys()))
        template = self.JD_TEMPLATES[role_type]
        company = random.choice(self.COMPANIES)
        seniority = random.choice(['Junior', 'Mid', 'Senior', 'Staff'])
        exp_required = random.randint(2, 10)

        # Generate title
        title = random.choice(template['title_templates']).format(company=company)

        # Generate description
        tools = ', '.join(random.sample(self.SKILL_SETS[role_type], k=3))
        team_size = random.choice(['5-10', '10-20', '20+'])
        description = template['description'].format(tools=tools, team_size=team_size)

        # Generate requirements
        requirements = template['requirements'].copy()
        requirements[0] = f"{exp_required}+ years {role_type} development"

        # Add experience/seniority level
        experience_text = f"\nSeniority Level: {seniority}\nYears Required: {exp_required}+"

        jd_text = f"""
        Job Title: {title}
        
        Company: {company}
        
        Description:
        {description}
        
        Requirements:
        {chr(10).join(f'- {req}' for req in requirements)}
        {experience_text}
        
        Benefits: Competitive salary, health insurance, remote work, professional development
        """

        return {
            'id': f"jd_{jd_id:04d}",
            'role_title': title,
            'company': company,
            'role_type': role_type,
            'seniority_level': seniority,
            'years_required': exp_required,
            'jd_text': jd_text.strip(),
            'skills_required': '; '.join(random.sample(self.SKILL_SETS[role_type], k=5)),
        }

    # ===================== COURSE DATA GENERATION =====================

    COURSE_PROVIDERS = ['Coursera', 'Udemy', 'edX', 'LinkedIn Learning', 'Pluralsight', 'DataCamp', 'Codecademy']
    COURSE_LEVELS = ['Beginner', 'Intermediate', 'Advanced', 'Expert']

    def generate_course(self, course_id: int) -> Dict[str, Any]:
        """Generate a synthetic course."""
        skill = random.choice(list(self.SKILL_SETS.keys()))
        skill_name = random.choice(self.SKILL_SETS[skill])
        provider = random.choice(self.COURSE_PROVIDERS)
        level = random.choice(self.COURSE_LEVELS)
        hours = random.randint(5, 100)
        rating = round(random.uniform(3.5, 5.0), 1)
        price = random.choice([0, 49, 99, 199, 299])

        title = f"{level} {skill_name} Course - {provider}"
        description = f"""
        Learn {skill_name} with this comprehensive {level.lower()} course on {provider}.
        
        This course covers:
        - {skill_name} fundamentals and advanced concepts
        - Best practices and design patterns
        - Real-world project examples
        - Hands-on exercises and assignments
        - Industry-standard tools and workflows
        
        Duration: {hours} hours
        Level: {level}
        Rating: {rating}/5.0
        Price: ${price}
        """

        return {
            'id': f"course_{course_id:04d}",
            'title': title,
            'skill': skill_name,
            'skill_category': skill,
            'provider': provider,
            'level': level,
            'hours': hours,
            'rating': rating,
            'price': price,
            'description': description.strip(),
            'url': f"https://{provider.lower().replace(' ', '')}.com/courses/{skill_name.lower().replace(' ', '-')}",
        }

    # ===================== MATCHING PAIRS GENERATION =====================

    def generate_matching_pairs(self, num_pairs: int = 50) -> List[Dict[str, Any]]:
        """Generate resume-JD matching pairs for evaluation."""
        pairs = []

        for i in range(num_pairs):
            # Choose a role type
            role_type = random.choice(list(self.SKILL_SETS.keys()))

            # Generate a resume for this role
            resume = self.generate_resume(i)
            resume['role_type'] = role_type

            # Generate matching JDs (same role type)
            jd_match = self.generate_jd(i)
            jd_match['role_type'] = role_type

            # Generate non-matching JDs
            other_role_types = [rt for rt in self.SKILL_SETS.keys() if rt != role_type]
            jd_nomatch_1 = self.generate_jd(i + num_pairs)
            jd_nomatch_1['role_type'] = random.choice(other_role_types)

            jd_nomatch_2 = self.generate_jd(i + 2 * num_pairs)
            jd_nomatch_2['role_type'] = random.choice(other_role_types)

            pairs.append({
                'pair_id': f"pair_{i:04d}",
                'resume': resume,
                'matching_jds': [jd_match['id']],
                'non_matching_jds': [jd_nomatch_1['id'], jd_nomatch_2['id']],
                'ground_truth': {jd_match['id']: 1, jd_nomatch_1['id']: 0, jd_nomatch_2['id']: 0}
            })

        return pairs

    # ===================== FILE OUTPUT =====================

    def save_resumes(self, count: int, output_path: str):
        """Save synthetic resumes to CSV."""
        resumes = [self.generate_resume(i) for i in range(count)]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=resumes[0].keys())
            writer.writeheader()
            writer.writerows(resumes)

        print(f"✅ Generated {count} synthetic resumes → {output_path}")
        return resumes

    def save_jds(self, count: int, output_path: str):
        """Save synthetic JDs to CSV."""
        jds = [self.generate_jd(i) for i in range(count)]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=jds[0].keys())
            writer.writeheader()
            writer.writerows(jds)

        print(f"✅ Generated {count} synthetic JDs → {output_path}")
        return jds

    def save_courses(self, count: int, output_path: str):
        """Save synthetic courses to CSV."""
        courses = [self.generate_course(i) for i in range(count)]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=courses[0].keys())
            writer.writeheader()
            writer.writerows(courses)

        print(f"✅ Generated {count} synthetic courses → {output_path}")
        return courses

    def save_matching_pairs(self, count: int, output_path: str):
        """Save matching pairs as JSON for evaluation."""
        pairs = self.generate_matching_pairs(count)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2)

        print(f"✅ Generated {count} matching pairs → {output_path}")
        return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for NextHorizon")
    parser.add_argument('--type', choices=['resume', 'jd', 'course', 'matching_pairs', 'all'],
                       default='all', help='Type of data to generate')
    parser.add_argument('--count', type=int, default=100, help='Number of items to generate')
    parser.add_argument('--output', type=str, default='synthetic_data/',
                       help='Output path (directory for --type=all, file otherwise)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    generator = SyntheticDataGenerator(random_seed=args.seed)

    # Create output directory if needed
    output_path = Path(args.output)
    if args.type == 'all':
        output_path.mkdir(parents=True, exist_ok=True)

    print("🔄 Generating synthetic data...")
    print(f"   Type: {args.type}, Count: {args.count}, Seed: {args.seed}")
    print()

    if args.type in ['resume', 'all']:
        resume_path = output_path / 'synthetic_resumes.csv' if args.type == 'all' else Path(args.output)
        generator.save_resumes(args.count, str(resume_path))

    if args.type in ['jd', 'all']:
        jd_path = output_path / 'synthetic_jds.csv' if args.type == 'all' else Path(args.output)
        generator.save_jds(args.count, str(jd_path))

    if args.type in ['course', 'all']:
        course_path = output_path / 'synthetic_courses.csv' if args.type == 'all' else Path(args.output)
        generator.save_courses(args.count, str(course_path))

    if args.type in ['matching_pairs', 'all']:
        pairs_path = output_path / 'matching_pairs.json' if args.type == 'all' else Path(args.output)
        generator.save_matching_pairs(args.count, str(pairs_path))

    print("\n✅ Synthetic data generation complete!")
    print(f"📁 Output directory: {output_path}")


if __name__ == '__main__':
    main()
