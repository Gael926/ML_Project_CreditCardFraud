import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)
from src.config import IMAGES_PATH

def eval_cls(y_true, y_pred, y_proba, model_name="Model"):
    # Evaluate classification model performance.

    print(f"\nEvaluation for {model_name}")
    
    # Calculate metrics
    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }
    
    # Print metrics
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5), dpi=100)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    cm_path = os.path.join(IMAGES_PATH, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

    # Plot Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f'PR-AUC: {metrics["PR-AUC"]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pr_path = os.path.join(IMAGES_PATH, f"precision_recall_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PR Curve to {pr_path}")

    return metrics

def plot_class_distribution(df):
    # Plots and saves the class distribution.
    class_counts = df["Class"].value_counts()
    class_pct = df["Class"].value_counts(normalize=True) * 100

    print(f"Counts:\n {class_counts}")
    print(f"\nPercentages (%):\n{class_pct.round(3)}")

    plt.figure(figsize=(6,4))
    sns.countplot(x="Class", data=df)
    plt.title("Target distribution (0=Legit, 1=Fraud)")
    plt.tight_layout()
    save_path = os.path.join(IMAGES_PATH, "class_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution plot to {save_path}")

def plot_amount_distribution(df):
    # Plots and saves the transaction amount distribution.
    print("Amount skewness:", df["Amount"].skew().round(3))

    plt.figure(figsize=(6,4), dpi=120)
    sns.histplot(df["Amount"], bins=60, kde=False)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Count")
    plt.tight_layout()
    save_path = os.path.join(IMAGES_PATH, "amount_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved amount distribution plot to {save_path}")
