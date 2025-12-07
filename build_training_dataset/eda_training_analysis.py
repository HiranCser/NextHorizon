"""
EDA Script: Correlating Training Database and Skill List
This script performs exploratory data analysis on training_database.csv and skill_list.csv
to understand skill coverage, training providers, course distribution, and data quality metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# File paths
DATA_DIR = Path(__file__).parent
TRAINING_DATABASE = DATA_DIR / 'training_database.clean.csv'
SKILL_LIST = DATA_DIR / 'skill_list.csv'
OUTPUT_DIR = DATA_DIR / 'eda_outputs'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load CSV files"""
    print("Loading data files...")
    training_df = pd.read_csv(TRAINING_DATABASE)
    skill_df = pd.read_csv(SKILL_LIST)
    print(f"✓ Training Database loaded: {len(training_df)} records")
    print(f"✓ Skill List loaded: {len(skill_df)} records")
    return training_df, skill_df

def print_summary_statistics(training_df, skill_df):
    """Print basic summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTraining Database Shape: {training_df.shape}")
    print(f"Skill List Shape: {skill_df.shape}")
    
    print(f"\nTraining Database Columns:\n{training_df.columns.tolist()}")
    print(f"\nSkill List Columns:\n{skill_df.columns.tolist()}")
    
    print(f"\nUnique Skills in Training Database: {training_df['skill'].nunique()}")
    print(f"Total Skills in Skill List: {len(skill_df)}")
    
    print(f"\nUnique Providers: {training_df['provider'].nunique()}")
    print(f"Unique Courses: {training_df['title'].nunique()}")
    
    print("\n" + "-"*80)
    print("Data Quality:")
    print("-"*80)
    print(training_df.isnull().sum())

def analyze_skill_coverage(training_df, skill_df):
    """Analyze skill coverage and correlation"""
    print("\n" + "="*80)
    print("SKILL COVERAGE ANALYSIS")
    print("="*80)
    
    skills_in_training = set(training_df['skill'].unique())
    skills_in_list = set(skill_df['skill_name'].unique())
    
    matched_skills = skills_in_training.intersection(skills_in_list)
    unmatched_in_training = skills_in_training - skills_in_list
    unmatched_in_list = skills_in_list - skills_in_training
    
    print(f"\n✓ Matched Skills (in both DB and Skill List): {len(matched_skills)}")
    print(f"✗ Unmatched in Training Database: {len(unmatched_in_training)}")
    print(f"✗ Unmatched in Skill List: {len(unmatched_in_list)}")
    
    print(f"\nSkill Coverage Rate: {len(matched_skills) / len(skill_df) * 100:.1f}%")
    
    if unmatched_in_training:
        print(f"\nSkills in Training Database but not in Skill List:")
        for skill in sorted(unmatched_in_training):
            print(f"  - {skill}")
    
    if unmatched_in_list:
        print(f"\nSkills in Skill List but not in Training Database:")
        for skill in sorted(unmatched_in_list):
            count = (training_df['skill'] == skill).sum()
            print(f"  - {skill}")
    
    return matched_skills, unmatched_in_training, unmatched_in_list

def analyze_skill_distribution(training_df):
    """Analyze distribution of training courses by skill"""
    print("\n" + "="*80)
    print("SKILL DISTRIBUTION ANALYSIS")
    print("="*80)
    
    skill_counts = training_df['skill'].value_counts()
    print(f"\nTop 15 Skills by Course Count:")
    print(skill_counts.head(15))
    
    print(f"\nTotal Skills: {len(skill_counts)}")
    print(f"Average Courses per Skill: {skill_counts.mean():.2f}")
    print(f"Median Courses per Skill: {skill_counts.median():.2f}")
    print(f"Min Courses: {skill_counts.min()}")
    print(f"Max Courses: {skill_counts.max()}")
    
    return skill_counts

def analyze_provider_distribution(training_df):
    """Analyze training provider distribution"""
    print("\n" + "="*80)
    print("TRAINING PROVIDER ANALYSIS")
    print("="*80)
    
    provider_counts = training_df['provider'].value_counts()
    print(f"\nTraining Providers:")
    print(provider_counts)
    
    print(f"\nTotal Providers: {len(provider_counts)}")
    print(f"Average Courses per Provider: {provider_counts.mean():.2f}")
    
    return provider_counts

def analyze_skill_provider_distribution(training_df):
    """Analyze which providers offer which skills"""
    print("\n" + "="*80)
    print("SKILL-PROVIDER DISTRIBUTION ANALYSIS")
    print("="*80)
    
    skill_provider = training_df.groupby(['skill', 'provider']).size().unstack(fill_value=0)
    print(f"\nSkill-Provider Matrix (Top 10 Skills):")
    print(skill_provider.head(10))

def analyze_course_ratings(training_df):
    """Analyze course ratings availability and distribution"""
    print("\n" + "="*80)
    print("COURSE RATINGS ANALYSIS")
    print("="*80)
    
    rating_available = training_df['rating'].notna().sum()
    rating_missing = training_df['rating'].isna().sum()
    
    print(f"\nCourses with Rating Data: {rating_available}")
    print(f"Courses with Missing Rating: {rating_missing}")
    print(f"Rating Data Completeness: {rating_available / len(training_df) * 100:.1f}%")
    
    if rating_available > 0:
        print(f"\nRating Statistics:")
        print(training_df['rating'].describe())
    
    return rating_available

def analyze_course_duration(training_df):
    """Analyze course duration information"""
    print("\n" + "="*80)
    print("COURSE DURATION ANALYSIS")
    print("="*80)
    
    hours_available = training_df['hours'].notna().sum()
    hours_missing = training_df['hours'].isna().sum()
    
    print(f"\nCourses with Duration Data: {hours_available}")
    print(f"Courses with Missing Duration: {hours_missing}")
    print(f"Duration Data Completeness: {hours_available / len(training_df) * 100:.1f}%")
    
    if hours_available > 0:
        print(f"\nDuration Statistics (in hours):")
        print(training_df['hours'].describe())
    
    return hours_available

def analyze_course_pricing(training_df):
    """Analyze course pricing information"""
    print("\n" + "="*80)
    print("COURSE PRICING ANALYSIS")
    print("="*80)
    
    price_data = training_df[training_df['price'] != 'unknown']['price']
    price_missing = (training_df['price'] == 'unknown').sum()
    
    print(f"\nCourses with Pricing Data: {len(price_data)}")
    print(f"Courses with Missing/Unknown Pricing: {price_missing}")
    print(f"Pricing Data Completeness: {len(price_data) / len(training_df) * 100:.1f}%")

def create_skill_distribution_chart(skill_counts):
    """Create bar chart of skills by course count"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    top_skills = skill_counts.head(15)
    colors = sns.color_palette("husl", len(top_skills))
    
    ax.barh(range(len(top_skills)), top_skills.values, color=colors)
    ax.set_yticks(range(len(top_skills)))
    ax.set_yticklabels(top_skills.index)
    ax.set_xlabel('Number of Training Courses', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Skills by Training Course Availability', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_skills.values):
        ax.text(v + 0.2, i, str(int(v)), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'skill_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: skill_distribution.png")
    plt.close()

def create_provider_distribution_chart(provider_counts):
    """Create bar chart of training providers"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("Set2", len(provider_counts))
    bars = ax.bar(range(len(provider_counts)), provider_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(provider_counts)))
    ax.set_xticklabels(provider_counts.index, rotation=45, ha='right', fontweight='bold')
    ax.set_ylabel('Number of Courses', fontsize=12, fontweight='bold')
    ax.set_title('Training Courses by Provider', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for bar, value in zip(bars, provider_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'provider_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: provider_distribution.png")
    plt.close()

def create_skill_provider_heatmap(training_df):
    """Create heatmap of skills vs providers"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get top skills
    top_skills = training_df['skill'].value_counts().head(12).index
    skill_provider_df = training_df[training_df['skill'].isin(top_skills)].copy()
    
    # Create pivot table
    pivot_table = pd.crosstab(skill_provider_df['skill'], skill_provider_df['provider'])
    
    # Create heatmap
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Skill vs Training Provider Heatmap (Top 12 Skills)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Training Provider', fontsize=12, fontweight='bold')
    ax.set_ylabel('Skill', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'skill_provider_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: skill_provider_heatmap.png")
    plt.close()

def create_skill_matching_summary(matched_skills, unmatched_in_training, unmatched_in_list):
    """Create visualization of skill matching"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Matched Skills', 'Unmatched in Training DB', 'Unmatched in Skill List']
    values = [len(matched_skills), len(unmatched_in_training), len(unmatched_in_list)]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Skill Matching Summary: Training Database vs Skill List', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, max(values) * 1.15])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'skill_matching_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: skill_matching_summary.png")
    plt.close()

def create_data_completeness_chart(training_df):
    """Create chart showing data completeness"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    completeness = (training_df.notna().sum() / len(training_df) * 100).sort_values(ascending=True)
    
    colors = ['green' if x > 80 else 'orange' if x > 50 else 'red' for x in completeness.values]
    ax.barh(range(len(completeness)), completeness.values, color=colors)
    ax.set_yticks(range(len(completeness)))
    ax.set_yticklabels(completeness.index)
    ax.set_xlabel('Data Completeness (%)', fontsize=12, fontweight='bold')
    ax.set_title('Data Completeness by Column', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim([0, 105])
    
    # Add percentage labels
    for i, v in enumerate(completeness.values):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_completeness.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: data_completeness.png")
    plt.close()

def create_course_duration_chart(training_df):
    """Create histogram of course duration"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    duration_data = training_df[training_df['hours'].notna()]['hours']
    
    if len(duration_data) > 0:
        ax.hist(duration_data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Course Duration (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Training Course Duration', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f"Mean: {duration_data.mean():.1f} hrs\nMedian: {duration_data.median():.1f} hrs\nStd Dev: {duration_data.std():.1f} hrs"
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'course_duration_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: course_duration_distribution.png")
    plt.close()

def create_skill_coverage_pie_chart(matched_skills, unmatched_in_list):
    """Create pie chart of skill coverage"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sizes = [len(matched_skills), len(unmatched_in_list)]
    labels = [f'Covered\n({len(matched_skills)})', f'Not Covered\n({len(unmatched_in_list)})']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, explode=explode, startangle=90,
                                        textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('Skill Coverage Rate\n(Skill List vs Training Database)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'skill_coverage_pie.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: skill_coverage_pie.png")
    plt.close()

def create_detailed_report(training_df, skill_df, matched_skills, unmatched_in_training, unmatched_in_list):
    """Create a detailed text report"""
    report_path = OUTPUT_DIR / 'eda_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("Training Database and Skill List Correlation\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Training Courses: {len(training_df)}\n")
        f.write(f"Total Skills in Skill List: {len(skill_df)}\n")
        f.write(f"Unique Skills in Training Database: {training_df['skill'].nunique()}\n")
        f.write(f"Unique Training Providers: {training_df['provider'].nunique()}\n")
        f.write(f"Unique Courses: {training_df['title'].nunique()}\n\n")
        
        f.write("2. SKILL COVERAGE ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Matched Skills: {len(matched_skills)}\n")
        f.write(f"Unmatched in Training Database: {len(unmatched_in_training)}\n")
        f.write(f"Unmatched in Skill List: {len(unmatched_in_list)}\n")
        f.write(f"Coverage Rate: {len(matched_skills) / len(skill_df) * 100:.1f}%\n\n")
        
        if unmatched_in_list:
            f.write("Skills in Skill List not found in Training Database:\n")
            for skill in sorted(unmatched_in_list):
                f.write(f"  - {skill}\n")
            f.write("\n")
        
        f.write("3. TOP 15 SKILLS BY TRAINING COURSE AVAILABILITY\n")
        f.write("-"*80 + "\n")
        skill_counts = training_df['skill'].value_counts()
        for i, (skill, count) in enumerate(skill_counts.head(15).items(), 1):
            f.write(f"{i:2d}. {skill}: {count} courses\n")
        f.write("\n")
        
        f.write("4. TRAINING PROVIDER DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        provider_counts = training_df['provider'].value_counts()
        for provider, count in provider_counts.items():
            percentage = count / len(training_df) * 100
            f.write(f"{provider}: {count} courses ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("5. COURSE RATINGS ANALYSIS\n")
        f.write("-"*80 + "\n")
        rating_available = training_df['rating'].notna().sum()
        f.write(f"Courses with Rating Data: {rating_available}\n")
        f.write(f"Courses with Missing Rating: {training_df['rating'].isna().sum()}\n")
        f.write(f"Data Completeness: {rating_available / len(training_df) * 100:.1f}%\n\n")
        
        f.write("6. COURSE DURATION ANALYSIS\n")
        f.write("-"*80 + "\n")
        hours_available = training_df['hours'].notna().sum()
        f.write(f"Courses with Duration Data: {hours_available}\n")
        f.write(f"Courses with Missing Duration: {training_df['hours'].isna().sum()}\n")
        f.write(f"Data Completeness: {hours_available / len(training_df) * 100:.1f}%\n")
        if hours_available > 0:
            f.write(f"Average Duration: {training_df['hours'].mean():.1f} hours\n")
            f.write(f"Duration Range: {training_df['hours'].min():.1f} - {training_df['hours'].max():.1f} hours\n\n")
        
        f.write("7. DATA QUALITY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Records: {len(training_df)}\n")
        f.write(f"Missing Values per Column:\n")
        for col, missing_count in training_df.isnull().sum().items():
            percentage = missing_count / len(training_df) * 100
            f.write(f"  {col}: {missing_count} ({percentage:.1f}%)\n")
    
    print(f"✓ Saved: eda_report.txt")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("STARTING EDA: TRAINING DATABASE AND SKILL LIST CORRELATION")
    print("="*80 + "\n")
    
    try:
        # Load data
        training_df, skill_df = load_data()
        
        # Print summaries
        print_summary_statistics(training_df, skill_df)
        
        # Analyze skills
        matched_skills, unmatched_in_training, unmatched_in_list = analyze_skill_coverage(training_df, skill_df)
        
        # Analyze skill distribution
        skill_counts = analyze_skill_distribution(training_df)
        
        # Analyze providers
        provider_counts = analyze_provider_distribution(training_df)
        
        # Analyze skill-provider distribution
        analyze_skill_provider_distribution(training_df)
        
        # Analyze ratings
        analyze_course_ratings(training_df)
        
        # Analyze duration
        analyze_course_duration(training_df)
        
        # Analyze pricing
        analyze_course_pricing(training_df)
        
        # Create visualizations
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        create_skill_distribution_chart(skill_counts)
        create_provider_distribution_chart(provider_counts)
        create_skill_provider_heatmap(training_df)
        create_skill_matching_summary(matched_skills, unmatched_in_training, unmatched_in_list)
        create_data_completeness_chart(training_df)
        create_course_duration_chart(training_df)
        create_skill_coverage_pie_chart(matched_skills, unmatched_in_list)
        
        # Create report
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80 + "\n")
        
        create_detailed_report(training_df, skill_df, matched_skills, unmatched_in_training, unmatched_in_list)
        
        print("\n" + "="*80)
        print("EDA COMPLETED SUCCESSFULLY!")
        print(f"Output files saved to: {OUTPUT_DIR}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during EDA: {str(e)}")
        raise

if __name__ == "__main__":
    main()
