"""
Model Training Module
Trains all 5 models using GridSearchCV and saves them
"""
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import numpy as np

def train_all_models(X_train, y_train, models_path):
    """
    Train all 5 models with GridSearchCV and save them
    Returns: dictionary of trained models
    """
    print("\n========== TRAINING ALL MODELS (WITH GRID SEARCH) ==========\n")
    
    models = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'class_weight': ['balanced']
    }
    lr = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), lr_param_grid, cv=3, scoring='f1', n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_best = lr.best_estimator_
    joblib.dump(lr_best, f"{models_path}\\logistic_regression.pkl")
    models['Logistic Regression'] = lr_best
    print(f"  ✓ Saved! Best params: {lr.best_params_}")
    
    # 2. Decision Tree
    print("\nTraining Decision Tree Classifier...")
    dt_param_grid = {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    }
    dt = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_param_grid, cv=3, scoring='f1', n_jobs=-1)
    dt.fit(X_train, y_train)
    dt_best = dt.best_estimator_
    joblib.dump(dt_best, f"{models_path}\\decision_tree.pkl")
    models['Decision Tree'] = dt_best
    print(f"  ✓ Saved! Best params: {dt.best_params_}")
    
    # 3. Random Forest
    print("\nTraining Random Forest Classifier...")
    rf_param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [15, 25, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_best = rf.best_estimator_
    joblib.dump(rf_best, f"{models_path}\\random_forest.pkl")
    models['Random Forest'] = rf_best
    print(f"  ✓ Saved! Best params: {rf.best_params_}")
    
    # Calculate scale_pos_weight for XGBoost
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos = num_neg / max(1, num_pos)

    # 4. XGBoost
    print("\nTraining XGBoost Classifier...")
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'scale_pos_weight': [scale_pos]
    }
    xgb = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, cv=3, scoring='f1', n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_best = xgb.best_estimator_
    joblib.dump(xgb_best, f"{models_path}\\xgboost.pkl")
    models['XGBoost'] = xgb_best
    print(f"  ✓ Saved! Best params: {xgb.best_params_}")
    
    # 5. Support Vector Machine
    print("\nTraining Support Vector Machine...")
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced']
    }
    svm = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=3, scoring='f1', n_jobs=-1)
    svm.fit(X_train, y_train)
    svm_best = svm.best_estimator_
    joblib.dump(svm_best, f"{models_path}\\svm.pkl")
    models['Support Vector Machine'] = svm_best
    print(f"  ✓ Saved! Best params: {svm.best_params_}")
    
    print("\n✓ All 5 models trained and saved!")
    
    return models
