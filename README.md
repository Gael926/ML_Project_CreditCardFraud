# Détection de fraudes par carte bancaire — Notebook & Modèles


Ce README documente le notebook **fraud_detection.ipynb** : de l'exploration des données jusqu'à l'entraînement et l'évaluation de modèles de classification pour la détection de fraudes.

- **Jeu de données** : `creditcard.csv` (transactions anonymisées par PCA, cible `Class`).
- **Taille** : 284807 lignes × 31 colonnes.
- **Déséquilibre** : classe 1 (fraude) ≈ 0.173% (492 cas) vs classe 0 (légitime) ≈ 99.827% (284315 cas).

## 1) Exploration rapide (EDA)
- Aucune valeur manquante détectée selon l'aperçu initial.
- **Montant** très asymétrique (skewness ≈ **16.978**). Normal.

**Distribution de la cible**

![Distribution de la classe](assets/class_distribution_1670778c.png)


## 2) Prétraitement

- **Standardisation de `Amount`** uniquement (`StandardScaler`) ; les variables `V1..V28` sont déjà « normalisées » (PCA).
- **Découpage**: `train_test_split(..., stratify=y)` pour conserver les proportions.
- **Pipeline scikit-learn**: `ColumnTransformer` + modèle, afin d'assurer une inférence reproductible.

## 3) Modèles & réglages

- **Régression Logistique** : `class_weight="balanced"`, grille (`C`, `penalty`, `solver`), `scoring="roc_auc"`.
- **Random Forest** : baseline `class_weight="balanced"`, puis `RandomizedSearchCV` (n_estimators, max_depth, min_samples_leaf, max_features, max_samples), `scoring="roc_auc"`.
- **XGBoost** : `RandomizedSearchCV` puis `GridSearchCV` ciblant **`average_precision` (PR-AUC)**, puis entraînement final avec **early stopping** sur un jeu de validation (métrique `aucpr`).
**Courbe de validation (RandomForest / max_depth → Recall)**

![Validation Curve RF](assets/rf_validation_curve_c04977ff.png)


## 4) Évaluation & résultats

**Comparatif sur le jeu de test**

| Modèle | ROC-AUC | PR-AUC | Rappel | Précision | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9724 | 0.6929 | 0.8862 | 0.0637 | 0.1188 |
| Random Forest (tuned) | 0.9738 | 0.8321 | 0.7724 | 0.9048 | 0.8333 |
| Logistic Regression | 0.9724 | 0.6929 | 0.8862 | 0.0637 | 0.1188 |
| Random Forest (tuned) | 0.9738 | 0.8321 | 0.7724 | 0.9048 | 0.8333 |
| XGBoost (baseline) | 0.9831 | 0.8608 | 0.8374 | 0.8306 | 0.8340 |

| Modèle | ROC-AUC | PR-AUC | Rappel | Précision | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9724 | 0.6929 | 0.8862 | 0.0637 | 0.1188 |
| Random Forest (tuned) | 0.9738 | 0.8321 | 0.7724 | 0.9048 | 0.8333 |
| Logistic Regression | 0.9724 | 0.6929 | 0.8862 | 0.0637 | 0.1188 |
| Random Forest (tuned) | 0.9738 | 0.8321 | 0.7724 | 0.9048 | 0.8333 |
| XGBoost (baseline) | 0.9831 | 0.8608 | 0.8374 | 0.8306 | 0.8340 |

**Régression Logistique — diagnostics**

- Matrice de confusion :

![CM LogReg](assets/confusion_matrix_logreg_9bf69f09.png)

**XGBoost (modèle final)**

- **Itération optimale** (early stopping): **954**
- **Meilleure PR-AUC (val.)**: **0.8397**
- **Seuil optimisé**: **0.855**
- **À ce seuil** → Précision **0.967**, Rappel **0.797**, F1 **0.874**

_Remarque_: sur données très déséquilibrées, **PR-AUC** est plus informative que ROC-AUC.


## 5) Export du modèle & inférence

Le pipeline final est sérialisé avec `joblib` : `../models/xgb_final_model.pkl`.

**Charger et prédire**

```python
import joblib
import pandas as pd

model = joblib.load("models/xgb_final_model.pkl")  # chemin relatif depuis la racine du projet
X_new = pd.DataFrame([...])  # mêmes colonnes que l'entraînement
probas = model.predict_proba(X_new)[:, 1]
preds = (probas >= 0.855).astype(int)  # seuil issu du notebook (à ajuster selon coût métier)
```

## 6) Reproductibilité

- Fixation des graines (`random_state=42`).
- Pipelines scikit-learn garantissant le même prétraitement en entraînement et inférence.
- **À prévoir dans le repo** (recommandé) :
  - `requirements.txt` ou `environment.yml`,
  - dossier `data/` (non versionné) avec `creditcard.csv`,
  - dossiers `images/`, `models/`, `notebooks/`, `reports/`,
  - script `train.py` (optionnel) pour rejouer l’entraînement hors notebook.

## 7) Limites & pistes d'amélioration (80/20)

- **Seuil décisionnel**: optimiser vs. **coûts** (faux positifs vs. vrais positifs). Fournir une courbe précision–rappel + table `Precision@K`.
- **Explicabilité rapide**: importance des variables (XGBoost `feature_importances_`), SHAP (optionnel).
- **Calibration**: vérifier la calibration des probabilités (ex. `CalibratedClassifierCV`).
- **Drift**: prévoir suivi des dérives (statistiques de base & taux de fraude au fil du temps).

## 8) Comment rejouer le notebook

1. Placer `creditcard.csv` dans `data/` et ajuster le chemin dans la cellule de chargement.
2. Installer les dépendances minimales :
   ```bash
   pip install -U pandas scikit-learn xgboost matplotlib seaborn joblib
   ```
3. Exécuter toutes les cellules. Les figures clés sont sauvegardées et également exportées ci-dessus.

## 9) À faire / Propositions

- [ ] Exporter systématiquement **toutes les figures** en `.png` (DPI ≥ 200) vers `images/` (tu as déjà des `plt.savefig(...)` pour la LogReg 👍).
- [ ] Générer automatiquement un **rapport** (ex. `reports/metrics.json` + `reports/figures.md`) en fin de notebook.
- [ ] Ajouter `requirements.txt` (je peux te le déduire depuis le notebook si tu veux).
- [ ] Ajouter un **script d'inférence** minimal (`predict.py`) qui charge le modèle et applique le seuil.