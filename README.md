# Credit Card Fraud Detection — Notebook & Models


This README documents the **fraud_detection.ipynb** notebook: from quick EDA to training and evaluating classifiers for fraud detection.

- **Dataset**: `creditcard.csv` (anonymized transactions via PCA-like components, target = `Class`).
- **Size**: 284807 rows × 31 columns.
- **Imbalance**: class 1 (fraud) ≈ 0.173% (492) vs class 0 (legit) ≈ 99.827% (284315).

## 1) Quick EDA
- No missing values detected in the initial overview.
- **Amount** is heavily skewed (skewness ≈ **16.978**), which is expected.

**Target distribution**

![Class distribution](images/class_distribution.png)


## 2) Preprocessing

- **Standardize `Amount`** only (`StandardScaler`); variables `V1..V28` are already scaled (PCA-like).
- **Split**: `train_test_split(..., stratify=y)` to preserve class proportions.
- **scikit-learn Pipeline**: `ColumnTransformer` + estimator to ensure reproducible inference.

## 3) Models & tuning

- **Logistic Regression**: `class_weight='balanced'`, grid over (`C`, `penalty`, `solver`), `scoring='roc_auc'`.
- **Random Forest**: baseline with `class_weight='balanced'`, then `RandomizedSearchCV` over (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `max_samples`), `scoring='roc_auc'`.
- **XGBoost**: `RandomizedSearchCV` → `GridSearchCV` targeting **`average_precision` (PR-AUC)**, then final training with **early stopping** on a validation set (`eval_metric='aucpr'`).
**Validation curve (RandomForest / max_depth → Recall)**

![Validation Curve RF](images/validation_curve_rf.png)


## 4) Evaluation & results

**Test-set comparison**

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9724 | 0.6929 | 0.8862 | 0.0637 | 0.1188 |
| Random Forest (tuned) | 0.9738 | 0.8321 | 0.7724 | 0.9048 | 0.8333 |
| Logistic Regression | 0.9724 | 0.6929 | 0.8862 | 0.0637 | 0.1188 |
| Random Forest (tuned) | 0.9738 | 0.8321 | 0.7724 | 0.9048 | 0.8333 |
| XGBoost (baseline) | 0.9831 | 0.8608 | 0.8374 | 0.8306 | 0.8340 |

**Logistic Regression — diagnostics**

- Confusion matrix:

![CM LogReg](images/confusion_matrix_logreg.png)

- ROC:

![ROC LogReg](images/roc_curve_logreg.png)

- Precision–Recall:

![PR LogReg](images/pr_curve_logreg.png)

**XGBoost (final model)**

- **Best iteration** (early stopping): **954**
- **Best validation PR-AUC**: **0.8397**
- **Optimized threshold**: **0.855**
- **At this threshold** → Precision **0.967**, Recall **0.797**, F1 **0.874**

_Note_: on highly imbalanced data, **PR-AUC** is more informative than ROC-AUC.


## 5) Model export & inference

The final pipeline is serialized with `joblib`: `models/xgb_final_model.pkl`.

**Load and predict**

```python
import joblib
import pandas as pd
import json

model = joblib.load("models/xgb_final_model.pkl")  # path from repo root
X_new = pd.DataFrame([...])  # same columns as training
probas = model.predict_proba(X_new)[:, 1]

# Optional: read decision threshold from JSON
thr = 0.855  # replace by reading models/threshold.json if you add it
preds = (probas >= thr).astype(int)
```

## 6) Reproducibility

- Fixed seeds (`random_state=42`).
- Pipelines ensure the same preprocessing at train and inference time.
- Recommended repo layout:
  - `data/` (not versioned) with `creditcard.csv`
  - `images/`, `models/`, `notebooks/`, `reports/`
  - `requirements.txt`
  - optional: `train.py` to re-run training outside notebooks

## 7) Limitations & improvement ideas (80/20)

- **Decision threshold**: optimize vs **business costs** (false positives vs true positives). Provide a Precision–Recall curve and `Precision@K` table.
- **Quick explainability**: feature importances (XGBoost) and optionally SHAP.
- **Calibration**: check probability calibration (e.g., `CalibratedClassifierCV`).
- **Drift monitoring**: track basic statistics & fraud rate over time.

## 8) How to run

1. Put `creditcard.csv` in `data/` and adjust its path in the loading cell.
2. Install dependencies:
   ```bash
   pip install -U pandas scikit-learn xgboost matplotlib seaborn joblib
   ```
3. Run all cells. Key figures are saved to `images/` and referenced above.

## 9) To‑do / Proposals

- [ ] Save **all figures** to `images/` (DPI ≥ 200) via a small helper (see below).
- [ ] Write out a **metrics report** automatically at the end (e.g., `reports/metrics.json` + `reports/figures.md`).
- [ ] Add `models/threshold.json` with the chosen operating threshold.
- [ ] Add a minimal **inference script** (`predict.py`) that loads the model and applies the threshold.

### (Optional) Helper to systematically save figures

```python
from pathlib import Path
import matplotlib.pyplot as plt

IMAGES_DIR = Path("images"); IMAGES_DIR.mkdir(parents=True, exist_ok=True)
def savefig(name):
    plt.savefig(IMAGES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
```
Use `savefig("roc_curve_logreg")` right before `plt.show()`.