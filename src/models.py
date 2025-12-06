from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from scipy.stats import randint, uniform
from src.config import RANDOM_STATE

# Hyperparameter Grids/Distributions

LR_PARAM_GRID = {
    "logisticregression__penalty": ["l1", "l2"],
    "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    "logisticregression__solver": ["liblinear", "saga"]
}

RF_PARAM_DIST = {
    "randomforestclassifier__n_estimators": randint(200, 500),
    "randomforestclassifier__max_depth": randint(10, 31),
    "randomforestclassifier__min_samples_leaf": randint(2, 6),
    "randomforestclassifier__max_features": ["sqrt", "log2"],
    "randomforestclassifier__max_samples": [0.5, 0.75, None]
}

XGB_PARAM_DIST = {
    "xgbclassifier__n_estimators": randint(400, 1601),
    "xgbclassifier__max_depth": randint(3, 10),
    "xgbclassifier__min_child_weight": randint(1, 11),
    "xgbclassifier__learning_rate": uniform(0.02, 0.18),
    "xgbclassifier__subsample": uniform(0.6, 0.4),
    "xgbclassifier__colsample_bytree": uniform(0.6, 0.4),
    "xgbclassifier__gamma": uniform(0.0, 5.0),
    "xgbclassifier__reg_lambda": uniform(0.0, 10.0)
}

def get_logistic_regression_pipeline(preprocessor):
    # Returns a pipeline with preprocessor and LogisticRegression.
    return make_pipeline(
        preprocessor,
        LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    )

def get_random_forest_pipeline(preprocessor):
    # Returns a pipeline with preprocessor and RandomForestClassifier.
    return make_pipeline(
        preprocessor,
        RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

def get_xgboost_pipeline(preprocessor, scale_pos_weight, n_estimators=500, learning_rate=0.05, 
                         max_depth=4, subsample=0.8, colsample_bytree=0.8):
    # Returns a pipeline with preprocessor and XGBClassifier.
    return make_pipeline(
        preprocessor,
        XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
