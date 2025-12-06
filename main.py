import pandas as pd
import joblib
import os
import sys

# Ensure src is in python path if running locally without install
sys.path.append(os.path.dirname(__file__))

from src.config import DATA_PATH, MODELS_PATH, RANDOM_STATE
from src.data import load_data, clean_data
from src.preprocessing import get_train_test_split, get_preprocessor
from src.evaluation import eval_cls, plot_class_distribution, plot_amount_distribution
from src.training import train_lr, train_rf, train_xgb_full_pipeline

def main():
    print("Starting Fraud Detection Pipeline")
    
    # Load and Clean Data
    df = load_data(DATA_PATH)
    
    # EDA Plots (saved to images folder)
    plot_class_distribution(df)
    plot_amount_distribution(df)
    
    X, y = clean_data(df)
    
    # Split Data
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    # Calculate scale_pos_weight for XGB
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight: {scale_pos_weight}")
    
    # Preprocessor
    preprocessor = get_preprocessor()
    
    # Storage for comparison
    results_list = []
    
    # Logistic Regression
    best_lr = train_lr(X_train, y_train, preprocessor)
    y_pred_lr = best_lr.predict(X_test)
    y_proba_lr = best_lr.predict_proba(X_test)[:, 1]
    lr_metrics = eval_cls(y_test, y_pred_lr, y_proba_lr, model_name="Logistic Regression")
    results_list.append({"Model": "Logistic Regression", **lr_metrics})
    
    # Random Forest
    best_rf = train_rf(X_train, y_train, preprocessor)
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
    rf_metrics = eval_cls(y_test, y_pred_rf, y_proba_rf, model_name="Random Forest")
    results_list.append({"Model": "Random Forest", **rf_metrics})
    
    # XGBoost (Full Pipeline)
    final_xgb, best_threshold = train_xgb_full_pipeline(X_train, y_train, preprocessor, scale_pos_weight, X_test, y_test)
    
    # Evaluate on Test
    y_proba_xgb = final_xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_proba_xgb >= best_threshold).astype(int)
    
    xgb_metrics = eval_cls(y_test, y_pred_xgb, y_proba_xgb, model_name="XGBoost Final")
    results_list.append({"Model": "XGBoost Final", **xgb_metrics})
    
    # Comparison
    print("\nModel Comparison")
    compare_df = pd.DataFrame(results_list).set_index("Model")
    print(compare_df)
    
    # Save Model
    model_save_path = os.path.join(MODELS_PATH, "xgb_final_model.pkl")
    joblib.dump(final_xgb, model_save_path)
    print(f"\nFinal model saved to {model_save_path}")
    
    # Save threshold info
    threshold_path = os.path.join(MODELS_PATH, "xgb_threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(str(best_threshold))
    print(f"Optimal threshold saved to {threshold_path}")

if __name__ == "__main__":
    main()
