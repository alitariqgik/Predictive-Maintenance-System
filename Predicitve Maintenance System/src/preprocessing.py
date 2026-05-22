"""
Preprocessing Module
Handles data preprocessing, scaling, and class imbalance
"""
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Feature columns (sensors)
FEATURE_COLUMNS = [
    'CPU_Temp', 'GPU_Temp', 'Ambient_Temp', 'Fan_Speed',
    'CPU_Usage', 'RAM_Usage', 'Disk_IO', 'PSU_Voltage',
    'Vibration', 'Uptime'
]

TARGET_COLUMN = 'Machine_Failure'

def preprocess_data(df, models_path):
    """
    Preprocess data: use pre-split column, apply SMOTE, and scale
    Returns: X_train, X_test, y_train, y_test, and saves scaler
    """
    print("\n========== DATA PREPROCESSING ==========")
    
    # Use pre-split data from 'Split' column
    print("\nUsing pre-split data from CSV (400 train, 100 test)...")
    train_df = df[df['Split'] == 'train'].copy()
    test_df = df[df['Split'] == 'test'].copy()
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Apply SMOTE to handle class imbalance
    print("\nApplying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"  After SMOTE - X_train: {X_train_smote.shape}")
    print(f"  Class distribution after SMOTE:")
    unique, counts = np.unique(y_train_smote, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    Class {u}: {c} ({c/len(y_train_smote)*100:.1f}%)")
    
    # Apply StandardScaler
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = f"{models_path}\\scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Convert back to DataFrame for compatibility
    X_train_scaled = np.array(X_train_scaled)
    X_test_scaled = np.array(X_test_scaled)
    
    return X_train_scaled, X_test_scaled, y_train_smote, y_test, scaler

def get_feature_columns():
    """Return feature column names"""
    return FEATURE_COLUMNS

def get_target_column():
    """Return target column name"""
    return TARGET_COLUMN
