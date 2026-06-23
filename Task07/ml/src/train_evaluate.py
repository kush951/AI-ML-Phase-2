"""
PlaceMux · AI/ML · Multi-Model Training & Evaluation Pipeline
Trains multiple classifiers, selects the best, and saves the winner.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from data_generator import generate_dataset
from feature_engineering import build_features, feature_names

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run_pipeline():
    print("=" * 60)
    print("PlaceMux · Matching Pipeline")
    print("=" * 60)

    # ── 1. Generate / Load Data ───────────────────────────────────────
    print("\n[1/6] Generating dataset …")
    df, _, _ = generate_dataset(n_students=600, n_jobs=120, pairs_per_student=10)
    print(f"     Records: {len(df):,}  |  Positive rate: {df['match_label'].mean():.1%}")

    # ── 2. Feature Engineering ────────────────────────────────────────
    print("\n[2/6] Engineering features …")
    X = build_features(df)
    y = df["match_label"]
    feat_names = feature_names()
    print(f"     Features: {X.shape[1]}  |  Shape: {X.shape}")

    # ── 3. Train / Val / Test Split ───────────────────────────────────
    print("\n[3/6] Splitting data (60/20/20) …")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    print(f"     Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── 4. Define Candidate Models ────────────────────────────────────
    print("\n[4/6] Training candidate models …")
    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.8, random_state=42
            ))
        ]),
        "AdaBoost": Pipeline([
            ("clf", AdaBoostClassifier(
                n_estimators=100, learning_rate=0.5, random_state=42
            ))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=5, gamma="scale",
                        probability=True, random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        results[name] = {
            "model": model,
            "cv_auc": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "val_precision": precision_score(y_val, y_pred),
            "val_recall": recall_score(y_val, y_pred),
            "val_f1": f1_score(y_val, y_pred),
            "val_auc": roc_auc_score(y_val, y_prob),
            "val_ap": average_precision_score(y_val, y_prob),
        }
        print(f"     {name:<22} CV-AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
              f"Val-F1={results[name]['val_f1']:.4f}  Val-AUC={results[name]['val_auc']:.4f}")

    # ── 5. Ensemble Best 3 Models ─────────────────────────────────────
    print("\n     Building ensemble from top-3 models …")
    sorted_models = sorted(results.items(), key=lambda x: x[1]["val_auc"], reverse=True)
    top3 = [(name, info["model"]) for name, info in sorted_models[:3]]
    ensemble = VotingClassifier(
        estimators=[(n.replace(" ", "_"), m) for n, m in top3],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_val)
    y_prob_ens = ensemble.predict_proba(X_val)[:, 1]
    cv_ens = cross_val_score(ensemble, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    results["Ensemble (Top-3)"] = {
        "model": ensemble,
        "cv_auc": cv_ens.mean(),
        "cv_std": cv_ens.std(),
        "val_precision": precision_score(y_val, y_pred_ens),
        "val_recall": recall_score(y_val, y_pred_ens),
        "val_f1": f1_score(y_val, y_pred_ens),
        "val_auc": roc_auc_score(y_val, y_prob_ens),
        "val_ap": average_precision_score(y_val, y_prob_ens),
    }
    print(f"     {'Ensemble (Top-3)':<22} CV-AUC={cv_ens.mean():.4f}±{cv_ens.std():.4f}  "
          f"Val-F1={results['Ensemble (Top-3)']['val_f1']:.4f}  "
          f"Val-AUC={results['Ensemble (Top-3)']['val_auc']:.4f}")

    # ── 6. Select Best & Evaluate on Test Set ─────────────────────────
    print("\n[5/6] Selecting best model and evaluating on held-out test set …")
    best_name = max(results, key=lambda n: results[n]["val_auc"])
    best = results[best_name]
    best_model = best["model"]

    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "precision": precision_score(y_test, y_pred_test),
        "recall": recall_score(y_test, y_pred_test),
        "f1": f1_score(y_test, y_pred_test),
        "auc_roc": roc_auc_score(y_test, y_prob_test),
        "avg_precision": average_precision_score(y_test, y_prob_test),
        "false_positive_rate": (confusion_matrix(y_test, y_pred_test)[0][1]
                                / max(confusion_matrix(y_test, y_pred_test)[0].sum(), 1)),
    }

    print(f"\n  Best model: {best_name}")
    print(f"  ─────────────────────────────")
    for k, v in test_metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred_test,
                                target_names=["No Match", "Match"], digits=4))

    # ── 7. Baseline Comparison ────────────────────────────────────────
    # Baseline: rank by composite_score threshold
    baseline_pred = (X_test["composite_score"] >= 0.5).astype(int)
    baseline_metrics = {
        "precision": precision_score(y_test, baseline_pred),
        "recall": recall_score(y_test, baseline_pred),
        "f1": f1_score(y_test, baseline_pred),
        "auc_roc": roc_auc_score(y_test, X_test["composite_score"]),
    }
    print(f"\n  Baseline (Skill-Overlap Heuristic):")
    for k, v in baseline_metrics.items():
        print(f"  {k:<22}: {v:.4f}")

    improvement = {k: test_metrics.get(k, 0) - v for k, v in baseline_metrics.items()}
    print(f"\n  Improvement over Baseline:")
    for k, v in improvement.items():
        print(f"  {k:<22}: {v:+.4f}")

    # ── 8. Save Artefacts ─────────────────────────────────────────────
    print("\n[6/6] Saving model & metadata …")
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    # Save all model metrics for report
    metrics_export = {}
    for name, info in results.items():
        metrics_export[name] = {
            k: round(v, 6) for k, v in info.items() if k != "model"
        }
    metrics_export["__best__"] = best_name
    metrics_export["__test_metrics__"] = {k: round(v, 6) for k, v in test_metrics.items()}
    metrics_export["__baseline__"] = {k: round(v, 6) for k, v in baseline_metrics.items()}
    metrics_export["__improvement__"] = {k: round(v, 6) for k, v in improvement.items()}

    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_export, f, indent=2)

    print(f"     Saved model → {MODELS_DIR / 'best_model.pkl'}")
    print(f"     Saved metrics → {MODELS_DIR / 'metrics.json'}")
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete. Best: {best_name}  |  Test AUC: {test_metrics['auc_roc']:.4f}")
    print(f"{'=' * 60}\n")

    return best_model, metrics_export, X_test, y_test


if __name__ == "__main__":
    run_pipeline()
