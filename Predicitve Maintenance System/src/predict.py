"""
Prediction Module
Makes predictions on new data and generates formatted output
"""
import pandas as pd
import joblib
import numpy as np

def predict_failures(input_csv_path, best_model_path, scaler_path, output_csv_path, optimal_threshold=0.5):
    """
    Load best model and scaler, make predictions on input CSV
    Format and display results, save to output CSV
    """
    print("\n========== MAKING PREDICTIONS ==========\n")
    
    # Load model and scaler
    best_model = joblib.load(best_model_path)
    scaler = joblib.load(scaler_path)
    
    # Load input data
    df_input = pd.read_csv(input_csv_path)
    
    # Feature columns
    feature_cols = [
        'CPU_Temp', 'GPU_Temp', 'Ambient_Temp', 'Fan_Speed',
        'CPU_Usage', 'RAM_Usage', 'Disk_IO', 'PSU_Voltage',
        'Vibration', 'Uptime'
    ]
    
    X_input = df_input[feature_cols]
    X_input_scaled = scaler.transform(X_input)
    
    # Make predictions
    y_pred_proba = best_model.predict_proba(X_input_scaled)[:, 1]
    
    # Prepare output
    results = []
    for idx in range(len(df_input)):
        failure_prob = y_pred_proba[idx]
        cpu_temp = df_input.iloc[idx]['CPU_Temp']
        gpu_temp = df_input.iloc[idx]['GPU_Temp']
        
        # Determine status and action based on the optimal threshold
        if failure_prob < optimal_threshold * 0.7:
            status = "✅ Healthy"
            action = "No action needed"
        elif failure_prob < optimal_threshold * 1.2:
            status = "⚠️  Warning"
            action = "Schedule maintenance soon"
        else:
            status = "🔴 Critical"
            action = "Immediate attention required!"
        
        results.append({
            'Machine': f"Machine {idx + 1}",
            'CPU_Temp': f"{cpu_temp:.1f}°C",
            'GPU_Temp': f"{gpu_temp:.1f}°C",
            'Status': status,
            'Failure_Probability': f"{failure_prob*100:.1f}%",
            'Action': action
        })
    
    results_df = pd.DataFrame(results)
    
    # Display formatted table
    print("=" * 110)
    print(f"{'Machine':<15} {'CPU Temp':<12} {'GPU Temp':<12} {'Status':<18} {'Failure Prob':<16} {'Action':<30}")
    print("=" * 110)
    
    for _, row in results_df.iterrows():
        print(f"{row['Machine']:<15} {row['CPU_Temp']:<12} {row['GPU_Temp']:<12} {row['Status']:<18} {row['Failure_Probability']:<16} {row['Action']:<30}")
    
    print("=" * 110)
    
    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    print(f"\n✓ Prediction results saved to {output_csv_path}")
    
    return results_df
