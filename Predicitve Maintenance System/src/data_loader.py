"""
Data Loader Module
Handles loading and basic data exploration
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load dataset from CSV"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def explore_data(df):
    """Explore and print basic data information"""
    print("\n========== DATA EXPLORATION ==========")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    # Check for null values
    print(f"\nNull values:\n{df.isnull().sum()}")
    if df.isnull().sum().sum() == 0:
        print("✓ No null values found!")
    
    # Class distribution
    print(f"\n========== CLASS DISTRIBUTION ==========")
    class_dist = df['Machine_Failure'].value_counts()
    print(f"\nMachine_Failure distribution:")
    print(f"  Class 0 (Healthy): {class_dist[0]} ({class_dist[0]/len(df)*100:.1f}%)")
    print(f"  Class 1 (Failure): {class_dist[1]} ({class_dist[1]/len(df)*100:.1f}%)")
    
    return class_dist

def plot_class_distribution(df, save_path):
    """Plot and save class distribution"""
    plt.figure(figsize=(8, 6))
    class_counts = df['Machine_Failure'].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # Green for healthy, red for failure
    
    bars = plt.bar(['Healthy (0)', 'Failure (1)'], 
                    class_counts.values, 
                    color=colors, 
                    alpha=0.8, 
                    edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Machine Status', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title('Machine Failure Distribution', fontsize=14, fontweight='bold')
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Class distribution plot saved to {save_path}")
    plt.close()

def plot_correlation_heatmap(df, save_path):
    """Plot and save correlation heatmap"""
    plt.figure(figsize=(12, 10))
    
    # Drop the target variable and Split column for correlation analysis
    feature_cols = [col for col in df.columns if col not in ['Machine_Failure', 'Split']]
    correlation_matrix = df[feature_cols + ['Machine_Failure']].corr()
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Correlation heatmap saved to {save_path}")
    plt.close()
