import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_curve
from src.config import RANDOM_STATE
from src.models import (
    get_logistic_regression_pipeline, LR_PARAM_GRID,
    get_random_forest_pipeline, RF_PARAM_DIST,
    get_xgboost_pipeline, XGB_PARAM_DIST
)
from src.evaluation import eval_cls

def train_lr(X_train, y_train, preprocessor):
    # Trains and tunes Logistic Regression
    print("\nTraining Logistic Regression")
    pipeline_lr = get_logistic_regression_pipeline(preprocessor)
    
    grid_lr = GridSearchCV(
        estimator=pipeline_lr,
        param_grid=LR_PARAM_GRID,
        scoring="roc_auc",
        n_jobs=-1
    )
    grid_lr.fit(X_train, y_train)
    print(f"Best params (LR): {grid_lr.best_params_}")
    return grid_lr.best_estimator_

def train_rf(X_train, y_train, preprocessor):
    # Trains and tunes Random Forest
    print("\nTraining Random Forest")
    pipeline_rf = get_random_forest_pipeline(preprocessor)
    
    # Using 3-fold CV for RF
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    rnd_rf = RandomizedSearchCV(
        estimator=pipeline_rf,
        param_distributions=RF_PARAM_DIST,
        n_iter=10,
        scoring="roc_auc",
        refit=True,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    rnd_rf.fit(X_train, y_train)
    print(f"Best params (RF): {rnd_rf.best_params_}")
    return rnd_rf.best_estimator_

def train_xgb_full_pipeline(X_train, y_train, preprocessor, scale_pos_weight, X_test, y_test):
    # Runs the full XGBoost training pipeline
    print("\nTraining XGBoost Pipeline")
    
    # Baseline
    print("Training Baseline XGBoost")
    xgb_baseline = get_xgboost_pipeline(preprocessor, scale_pos_weight)

    # RandomizedSearch
    print("Running RandomizedSearchCV for XGBoost")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    rnd_search = RandomizedSearchCV(
        estimator=xgb_baseline,
        param_distributions=XGB_PARAM_DIST,
        n_iter=40,
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    rnd_search.fit(X_train, y_train)
    print(f"Best params (Randomized): {rnd_search.best_params_}")
    print(f"Best CV PR-AUC: {rnd_search.best_score_}")
    
    # GridSearch
    print("Running GridSearchCV for XGBoost based on best Randomized params")
    best_rnd = rnd_search.best_params_
    
    # Construct grid around best params
    grid_params = {
        "xgbclassifier__max_depth": [2, 3, 4],
        "xgbclassifier__min_child_weight": [1, 2],
        "xgbclassifier__learning_rate": [0.05, 0.07, 0.09],
        "xgbclassifier__subsample": [0.7, 0.8, 0.9],
        "xgbclassifier__colsample_bytree": [best_rnd["xgbclassifier__colsample_bytree"]],
        "xgbclassifier__gamma": [best_rnd["xgbclassifier__gamma"]],
        "xgbclassifier__reg_lambda": [best_rnd["xgbclassifier__reg_lambda"]],
        "xgbclassifier__n_estimators": [best_rnd["xgbclassifier__n_estimators"]],
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_baseline,
        param_grid=grid_params,
        scoring="average_precision",
        cv=cv,
        refit=True,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params (Grid): {grid_search.best_params_}")
    print(f"Best CV PR-AUC: {grid_search.best_score_}")
    
    # Final Fit with Early Stopping
    print("Training Final XGBoost with Early Stopping")
    # Split training set for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    
    # Clone and fit preprocessor
    prep = clone(preprocessor)
    prep.fit(X_tr, y_tr)
    X_tr_p = prep.transform(X_tr)
    X_val_p = prep.transform(X_val)
    
    # Get the XGB model with best params
    xgb_final_clf = clone(grid_search.best_estimator_.named_steps["xgbclassifier"])
    xgb_final_clf.set_params(
        n_estimators=2000,
        early_stopping_rounds=50,
        verbosity=1,
        eval_metric="aucpr"
    )
    
    xgb_final_clf.fit(
        X_tr_p, y_tr,
        eval_set=[(X_val_p, y_val)],
        verbose=False
    )
    
    print(f"Best iteration: {xgb_final_clf.best_iteration}")
    print(f"Best validation aucpr: {xgb_final_clf.best_score}")
    
    # Create final pipeline
    final_pipeline = make_pipeline(prep, xgb_final_clf)
    
    # Threshold Tuning
    print("Tuning Threshold on Validation Set")
    y_val_proba = xgb_final_clf.predict_proba(X_val_p)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Validation F1 at best threshold: {f1_scores[best_idx]:.3f}")
    
    return final_pipeline, best_threshold
