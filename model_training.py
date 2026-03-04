"""
model_training.py  -  Influential User Identification via Social Media Mining
==============================================================================
Trains and evaluates four classifiers to identify influential social media users:
  1. Random Forest
  2. Gradient Boosting (sklearn GBM)
  3. Support Vector Machine (RBF kernel)
  4. Logistic Regression

Outputs:
  models/results.csv         - per-model classification metrics
  models/feature_importance.csv - RF feature importance ranking
  outputs/confusion_matrix.png
  outputs/roc_curves.png
  outputs/feature_importance.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     roc_auc_score, roc_curve, accuracy_score,
                                     f1_score, precision_score, recall_score)
from sklearn.pipeline        import Pipeline
import joblib

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "verified","followers_count","following_count","tweet_count","listed_count",
    "account_age_days","follower_following_ratio","tweets_per_day",
    "listed_per_follower","in_degree","out_degree","degree_ratio",
    "pagerank","hits_authority","hits_hub","betweenness","clustering_coef",
    "tweet_count_actual","avg_retweets","avg_likes","avg_replies",
    "avg_hashtags","avg_mentions","total_engagement",
]
TARGET_COL = "influential"


def load_features():
    path = os.path.join(DATA_DIR, "features.csv")
    df   = pd.read_csv(path)
    # Drop cols that are not in FEATURE_COLS
    present = [c for c in FEATURE_COLS if c in df.columns]
    X = df[present].fillna(0)
    y = df[TARGET_COL]
    print(f"  Feature matrix: {X.shape}  |  Class balance: {y.value_counts().to_dict()}")
    return X, y, df


def build_models():
    return {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=12,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=150, max_depth=5,
                learning_rate=0.1, random_state=42)),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=1.0, probability=True,
                           class_weight="balanced", random_state=42)),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=500, class_weight="balanced", random_state=42)),
        ]),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"  {name:20s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    return {
        "model": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "roc_auc": auc,
        "y_pred": y_pred, "y_proba": y_proba,
    }


def plot_confusion_matrices(results, y_test, labels):
    n   = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=labels, yticklabels=labels)
        axes[i].set_title(f"{name}\nF1={res['f1']:.3f}  AUC={res['roc_auc']:.3f}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(8, 6))
    colors = ["#2196F3","#4CAF50","#FF5722","#9C27B0"]
    for (name, res), color in zip(results.items(), colors):
        if res["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, lw=2)
    plt.plot([0,1],[0,1],"k--", lw=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names):
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=True)
    top = importances.tail(20)
    plt.figure(figsize=(9, 7))
    colors = ["#1976D2" if v < top.median() else "#D32F2F" for v in top.values]
    top.plot(kind="barh", color=colors)
    plt.xlabel("Importance Score")
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    imp_df = importances.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df.sort_values("importance", ascending=False).to_csv(
        os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)


def train_and_evaluate():
    print("=" * 55)
    print(" MODEL TRAINING & EVALUATION")
    print("=" * 55)

    print("[1/4] Loading feature matrix...")
    X, y, df = load_features()

    print("[2/4] Train/test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"  Train={len(X_train)}, Test={len(X_test)}")

    print("[3/4] Training models...")
    models  = build_models()
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_train, X_test, y_train, y_test)

    print("[4/4] Generating visualisations...")
    plot_confusion_matrices(results, y_test, labels=["Non-Influential", "Influential"])
    plot_roc_curves(results, y_test)
    plot_feature_importance(models["RandomForest"], X.columns.tolist())

    # Save best model
    best_name  = max(results, key=lambda n: results[n]["f1"])
    best_model = models[best_name]
    best_model.fit(X, y)          # refit on full data
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    print(f"  Best model: {best_name} (F1={results[best_name]['f1']:.3f}) saved.")

    # Save metrics table
    metrics = pd.DataFrame([{
        "Model": r["model"], "Accuracy": r["accuracy"],
        "Precision": r["precision"], "Recall": r["recall"],
        "F1": r["f1"], "ROC_AUC": r["roc_auc"],
    } for r in results.values()])
    metrics.to_csv(os.path.join(MODEL_DIR, "results.csv"), index=False)
    print("\n", metrics.to_string(index=False))
    return results, models, X, y


if __name__ == "__main__":
    train_and_evaluate()