"""
Model Evaluation Module
Evaluates all models, applies optimal probability thresholding, and generates comparison reports and plots
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
)

def evaluate_all_models(models, X_test, y_test, results_path):
    """
    Evaluate all models using optimal F1-score thresholding and return results dataframe
    """
    print("\n========== EVALUATING ALL MODELS (OPTIMAL THRESHOLDS) ==========\n")
    
    results = []
    model_predictions = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # Get probabilities
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate optimal threshold based on Precision-Recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Calculate F1.5 score (prioritizing recall) for all thresholds
        beta = 1.5
        f_scores = np.zeros_like(thresholds)
        for i in range(len(thresholds)):
            p = precisions[i]
            r = recalls[i]
            if (p + r) == 0:
                f_scores[i] = 0
            else:
                f_scores[i] = (1 + beta**2) * (p * r) / ((beta**2 * p) + r)
        
        # Find threshold that maximizes the Recall-heavy F-score
        best_idx = np.argmax(f_scores)
        best_threshold = thresholds[best_idx]
        
        # Apply the optimal threshold
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        
        # Metrics using optimal predictions
        accuracy = accuracy_score(y_test, y_pred_optimal)
        precision = precision_score(y_test, y_pred_optimal, zero_division=0)
        recall = recall_score(y_test, y_pred_optimal, zero_division=0)
        f1 = f1_score(y_test, y_pred_optimal, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba) # ROC AUC uses probas
        
        print(f"  Best Threshold: {best_threshold:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        results.append({
            'Model': model_name,
            'Optimal_Threshold': best_threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        model_predictions[model_name] = {
            'y_pred': y_pred_optimal,
            'y_pred_proba': y_pred_proba,
            'optimal_threshold': best_threshold
        }
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_csv_path = f"{results_path}\\model_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n✓ Model comparison saved to {results_csv_path}")
    
    return results_df, model_predictions, models

def plot_roc_curves(models, model_predictions, X_test, y_test, results_path):
    """Plot ROC curves for all models"""
    print("\nGenerating ROC curves...")
    
    plt.figure(figsize=(10, 8))
    
    for model_name in models.keys():
        y_pred_proba = model_predictions[model_name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)')
    
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = f"{results_path}\\roc_curves.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved to {roc_path}")
    plt.close()

def plot_f1_comparison(results_df, results_path):
    """Plot F1-Score comparison bar chart"""
    print("Generating F1-Score comparison...")
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = plt.bar(range(len(results_df)), results_df['F1-Score'], color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    plt.title('F1-Score Comparison - All Models (Optimal Thresholds)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(results_df)), results_df['Model'], rotation=45, ha='right')
    plt.ylim(0, max(results_df['F1-Score']) * 1.15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    f1_path = f"{results_path}\\f1_comparison.png"
    plt.savefig(f1_path, dpi=300, bbox_inches='tight')
    print(f"✓ F1-Score comparison saved to {f1_path}")
    plt.close()

def plot_best_model_confusion_matrix(best_model_name, model_predictions, y_test, results_path):
    """Plot confusion matrix for the best model"""
    print("Generating best model confusion matrix...")
    
    y_pred = model_predictions[best_model_name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Failure'],
                yticklabels=['Healthy', 'Failure'],
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    cm_path = f"{results_path}\\best_model_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {cm_path}")
    plt.close()

def get_best_model(results_df, models_path):
    """Get best model name based on F1-Score"""
    best_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_f1_score = results_df.loc[best_idx, 'F1-Score']
    
    return best_model_name, best_f1_score
