"""
EDA Script: Correlating JD Database and Role List
This script performs exploratory data analysis on jd_database.csv and role_list.csv
to understand role distribution, seniority levels, and data quality metrics.
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
JD_DATABASE = DATA_DIR / 'jd_database.clean.csv'
ROLE_LIST = DATA_DIR / 'role_list.csv'
OUTPUT_DIR = DATA_DIR / 'eda_outputs'

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    """Load CSV files"""
    print("Loading data files...")
    jd_df = pd.read_csv(JD_DATABASE)
    role_df = pd.read_csv(ROLE_LIST)
    print(f"✓ JD Database loaded: {len(jd_df)} records")
    print(f"✓ Role List loaded: {len(role_df)} records")
    return jd_df, role_df

def print_summary_statistics(jd_df, role_df):
    """Print basic summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nJD Database Shape: {jd_df.shape}")
    print(f"Role List Shape: {role_df.shape}")
    
    print(f"\nJD Database Columns:\n{jd_df.columns.tolist()}")
    print(f"\nRole List Columns:\n{role_df.columns.tolist()}")
    
    print(f"\nUnique Roles in JD Database: {jd_df['role_title'].nunique()}")
    print(f"Total Roles in Role List: {len(role_df)}")
    
    print("\n" + "-"*80)
    print("Data Quality:")
    print("-"*80)
    print(jd_df.isnull().sum())

def analyze_role_distribution(jd_df, role_df):
    """Analyze distribution of roles"""
    print("\n" + "="*80)
    print("ROLE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    role_counts = jd_df['role_title'].value_counts()
    print(f"\nTop 15 Most Frequent Roles:")
    print(role_counts.head(15))
    
    # Correlation: Roles in DB vs Role List
    roles_in_jd = set(jd_df['role_title'].unique())
    roles_in_list = set(role_df['role_title'].unique())
    
    matched_roles = roles_in_jd.intersection(roles_in_list)
    unmatched_in_jd = roles_in_jd - roles_in_list
    unmatched_in_list = roles_in_list - roles_in_jd
    
    print(f"\n✓ Matched Roles (in both DB and Role List): {len(matched_roles)}")
    print(f"✗ Unmatched in JD Database: {len(unmatched_in_jd)}")
    print(f"✗ Unmatched in Role List: {len(unmatched_in_list)}")
    
    if unmatched_in_list:
        print(f"\nRoles in Role List but not in JD Database:")
        for role in sorted(unmatched_in_list):
            print(f"  - {role}")
    
    return role_counts, matched_roles, unmatched_in_jd, unmatched_in_list

def analyze_seniority_distribution(jd_df):
    """Analyze seniority level distribution"""
    print("\n" + "="*80)
    print("SENIORITY LEVEL ANALYSIS")
    print("="*80)
    
    seniority_counts = jd_df['seniority_level'].value_counts()
    print(f"\nSeniority Level Distribution:")
    print(seniority_counts)
    
    print(f"\nMissing Seniority Data: {jd_df['seniority_level'].isnull().sum()}")
    
    return seniority_counts

def analyze_source_domain(jd_df):
    """Analyze source domain distribution"""
    print("\n" + "="*80)
    print("SOURCE DOMAIN ANALYSIS")
    print("="*80)
    
    source_counts = jd_df['source_domain'].value_counts()
    print(f"\nTop 10 Source Domains:")
    print(source_counts.head(10))
    
    return source_counts

def analyze_experience_range(jd_df):
    """Analyze experience requirements"""
    print("\n" + "="*80)
    print("EXPERIENCE RANGE ANALYSIS")
    print("="*80)
    
    min_exp_available = jd_df['exp_min_years'].notna().sum()
    max_exp_available = jd_df['exp_max_years'].notna().sum()
    
    print(f"\nJobs with Minimum Experience Specified: {min_exp_available}")
    print(f"Jobs with Maximum Experience Specified: {max_exp_available}")
    
    if min_exp_available > 0:
        print(f"\nMinimum Experience Statistics:")
        print(jd_df['exp_min_years'].describe())
    
    if max_exp_available > 0:
        print(f"\nMaximum Experience Statistics:")
        print(jd_df['exp_max_years'].describe())
    
    return min_exp_available, max_exp_available

def create_role_distribution_chart(role_counts):
    """Create bar chart of top roles"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    top_roles = role_counts.head(15)
    colors = sns.color_palette("husl", len(top_roles))
    
    ax.barh(range(len(top_roles)), top_roles.values, color=colors)
    ax.set_yticks(range(len(top_roles)))
    ax.set_yticklabels(top_roles.index)
    ax.set_xlabel('Number of Job Postings', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Frequent Roles in JD Database', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_roles.values):
        ax.text(v + 0.5, i, str(v), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'role_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: role_distribution.png")
    plt.close()

def create_seniority_distribution_chart(seniority_counts):
    """Create pie chart of seniority levels"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = sns.color_palette("Set2", len(seniority_counts))
    wedges, texts, autotexts = ax.pie(
        seniority_counts.values,
        labels=seniority_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    ax.set_title('Job Distribution by Seniority Level', fontsize=14, fontweight='bold', pad=20)
    
    # Add count labels
    for i, (label, count) in enumerate(seniority_counts.items()):
        texts[i].set_text(f'{label}\n(n={count})')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'seniority_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: seniority_distribution.png")
    plt.close()

def create_source_domain_chart(source_counts):
    """Create bar chart of top source domains"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    top_sources = source_counts.head(10)
    colors = sns.color_palette("coolwarm", len(top_sources))
    
    ax.bar(range(len(top_sources)), top_sources.values, color=colors)
    ax.set_xticks(range(len(top_sources)))
    ax.set_xticklabels(top_sources.index, rotation=45, ha='right')
    ax.set_ylabel('Number of Job Postings', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Source Domains for Job Postings', fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for i, v in enumerate(top_sources.values):
        ax.text(i, v + 20, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'source_domain_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: source_domain_distribution.png")
    plt.close()

def create_role_seniority_heatmap(jd_df):
    """Create heatmap of roles vs seniority levels"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get top roles
    top_roles = jd_df['role_title'].value_counts().head(10).index
    role_seniority_df = jd_df[jd_df['role_title'].isin(top_roles)].copy()
    
    # Create pivot table
    pivot_table = pd.crosstab(role_seniority_df['role_title'], role_seniority_df['seniority_level'])
    
    # Create heatmap
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Role vs Seniority Level Distribution (Top 10 Roles)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Seniority Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Role Title', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'role_seniority_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: role_seniority_heatmap.png")
    plt.close()

def create_experience_distribution_chart(jd_df):
    """Create visualization of experience requirements"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Minimum experience
    min_exp_data = jd_df['exp_min_years'].dropna()
    if len(min_exp_data) > 0:
        axes[0].hist(min_exp_data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Minimum Experience Requirements', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
    
    # Maximum experience
    max_exp_data = jd_df['exp_max_years'].dropna()
    if len(max_exp_data) > 0:
        axes[1].hist(max_exp_data, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Maximum Experience Requirements', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'experience_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: experience_distribution.png")
    plt.close()

def create_data_completeness_chart(jd_df):
    """Create chart showing data completeness"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    completeness = (jd_df.notna().sum() / len(jd_df) * 100).sort_values(ascending=True)
    
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

def create_role_matching_summary(matched_roles, unmatched_in_jd, unmatched_in_list):
    """Create visualization of role matching"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Matched Roles', 'Unmatched in JD DB', 'Unmatched in Role List']
    values = [len(matched_roles), len(unmatched_in_jd), len(unmatched_in_list)]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Role Matching Summary: JD Database vs Role List', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, max(values) * 1.15])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'role_matching_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: role_matching_summary.png")
    plt.close()

def create_detailed_report(jd_df, role_df, matched_roles, unmatched_in_jd, unmatched_in_list):
    """Create a detailed text report"""
    report_path = OUTPUT_DIR / 'eda_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS REPORT\n")
        f.write("JD Database and Role List Correlation\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total JD Records: {len(jd_df)}\n")
        f.write(f"Total Role List Entries: {len(role_df)}\n")
        f.write(f"Unique Roles in JD Database: {jd_df['role_title'].nunique()}\n\n")
        
        f.write("2. ROLE MATCHING ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Matched Roles: {len(matched_roles)}\n")
        f.write(f"Unmatched in JD Database: {len(unmatched_in_jd)}\n")
        f.write(f"Unmatched in Role List: {len(unmatched_in_list)}\n")
        f.write(f"Match Rate: {len(matched_roles) / len(role_df) * 100:.1f}%\n\n")
        
        if unmatched_in_list:
            f.write("Roles in Role List not found in JD Database:\n")
            for role in sorted(unmatched_in_list):
                f.write(f"  - {role}\n")
            f.write("\n")
        
        f.write("3. TOP 15 ROLES BY FREQUENCY\n")
        f.write("-"*80 + "\n")
        role_counts = jd_df['role_title'].value_counts()
        for i, (role, count) in enumerate(role_counts.head(15).items(), 1):
            f.write(f"{i:2d}. {role}: {count} postings\n")
        f.write("\n")
        
        f.write("4. SENIORITY LEVEL DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        seniority_counts = jd_df['seniority_level'].value_counts()
        for level, count in seniority_counts.items():
            percentage = count / len(jd_df) * 100
            f.write(f"{level}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("5. DATA QUALITY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Records: {len(jd_df)}\n")
        f.write(f"Missing Values per Column:\n")
        for col, missing_count in jd_df.isnull().sum().items():
            percentage = missing_count / len(jd_df) * 100
            f.write(f"  {col}: {missing_count} ({percentage:.1f}%)\n")
    
    print(f"✓ Saved: eda_report.txt")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("STARTING EDA: JD DATABASE AND ROLE LIST CORRELATION")
    print("="*80 + "\n")
    
    try:
        # Load data
        jd_df, role_df = load_data()
        
        # Print summaries
        print_summary_statistics(jd_df, role_df)
        
        # Analyze roles
        role_counts, matched_roles, unmatched_in_jd, unmatched_in_list = analyze_role_distribution(jd_df, role_df)
        
        # Analyze seniority
        seniority_counts = analyze_seniority_distribution(jd_df)
        
        # Analyze source domains
        source_counts = analyze_source_domain(jd_df)
        
        # Analyze experience
        min_exp, max_exp = analyze_experience_range(jd_df)
        
        # Create visualizations
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        create_role_distribution_chart(role_counts)
        create_seniority_distribution_chart(seniority_counts)
        create_source_domain_chart(source_counts)
        create_role_seniority_heatmap(jd_df)
        create_experience_distribution_chart(jd_df)
        create_data_completeness_chart(jd_df)
        create_role_matching_summary(matched_roles, unmatched_in_jd, unmatched_in_list)
        
        # Create report
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80 + "\n")
        
        create_detailed_report(jd_df, role_df, matched_roles, unmatched_in_jd, unmatched_in_list)
        
        print("\n" + "="*80)
        print("EDA COMPLETED SUCCESSFULLY!")
        print(f"Output files saved to: {OUTPUT_DIR}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during EDA: {str(e)}")
        raise

if __name__ == "__main__":
    main()
