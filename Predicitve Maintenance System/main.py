"""
Main Pipeline
Orchestrates the entire Predictive Maintenance System workflow
"""
import os
import sys

# Add src directory to path
sys.path.insert(0, 'D:\\ds211-p\\src')

from data_loader import load_data, explore_data, plot_class_distribution, plot_correlation_heatmap
from preprocessing import preprocess_data
from train import train_all_models
from evaluate import (
    evaluate_all_models, plot_roc_curves, plot_f1_comparison,
    plot_best_model_confusion_matrix, get_best_model
)
from predict import predict_failures
import joblib

# Paths
DATA_FILE = "D:\\ds211-p\\data\\raw\\pre_data.csv"
PROJECT_DIR = "D:\\ds211-p"
MODELS_DIR = f"{PROJECT_DIR}\\models"
RESULTS_DIR = f"{PROJECT_DIR}\\results"

def main():
    """Main pipeline execution"""
    
    # Ensure directories exist to prevent errors
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("\n")
    print("=" * 80)
    print("PREDICTIVE MAINTENANCE SYSTEM - DS211 PROJECT".center(80))
    print("=" * 80)
    print()
    
    # ========== STEP 0: Load Fixed Dataset ==========
    print("\n========== STEP 0: LOADING FIXED DATASET ==========")
    print(f"\n[OK] Using pre-split dataset: {DATA_FILE}")
    print("  (400 training samples + 100 testing samples in one CSV file)")
    
    # ========== STEP 1: Load and Explore Data ==========
    print("\n========== STEP 1: LOADING AND EXPLORING DATA ==========")
    df = load_data(DATA_FILE)
    explore_data(df)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_class_distribution(df, f"{RESULTS_DIR}\\class_distribution.png")
    plot_correlation_heatmap(df, f"{RESULTS_DIR}\\correlation_heatmap.png")
    
    # ========== STEP 2: Preprocessing ==========
    print("\n========== STEP 2: PREPROCESSING DATA ==========")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, MODELS_DIR)
    
    # ========== STEP 3: Train Models ==========
    print("\n========== STEP 3: TRAINING ALL MODELS ==========")
    models = train_all_models(X_train, y_train, MODELS_DIR)
    
    # ========== STEP 4: Evaluate Models ==========
    print("\n========== STEP 4: EVALUATING ALL MODELS ==========")
    results_df, model_predictions, models_dict = evaluate_all_models(models, X_test, y_test, RESULTS_DIR)
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)
    
    # Generate plots
    plot_roc_curves(models_dict, model_predictions, X_test, y_test, RESULTS_DIR)
    plot_f1_comparison(results_df, RESULTS_DIR)
    
    # ========== STEP 5: Select Best Model ==========
    print("\n========== STEP 5: SELECTING BEST MODEL ==========")
    best_model_name, best_f1_score = get_best_model(results_df, MODELS_DIR)
    best_threshold = results_df.loc[results_df['Model'] == best_model_name, 'Optimal_Threshold'].values[0]
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1-Score: {best_f1_score:.4f}")
    print(f"   Optimal Threshold: {best_threshold:.4f}")
    
    # Save best model
    best_model = models_dict[best_model_name]
    best_model_path = f"{MODELS_DIR}\\best_model.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"✓ Best model saved to {best_model_path}")
    
    # Plot confusion matrix for best model
    plot_best_model_confusion_matrix(best_model_name, model_predictions, y_test, RESULTS_DIR)
    
    # ========== STEP 6 & 7: Predictions on Full Dataset ==========
    print("\n========== STEP 6 & 7: MAKING PREDICTIONS ON FULL DATASET ==========")
    print(f"\nLoading data from {DATA_FILE} for predictions...")
    predict_failures(
        DATA_FILE,
        best_model_path,
        f"{MODELS_DIR}\\scaler.pkl",
        f"{RESULTS_DIR}\\predictions_output.csv",
        optimal_threshold=best_threshold
    )
    
    # ========== COMPLETION ==========
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!".center(80))
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  • Models: {MODELS_DIR}")
    print(f"  • Results & Plots: {RESULTS_DIR}")
    print(f"  • Best Model: {best_model_path}")
    print(f"  • Predictions: {RESULTS_DIR}\\predictions_output.csv")
    print("\n")

if __name__ == "__main__":
    main()
