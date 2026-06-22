"""
PlaceMux · Model Zoo & Selection
Trains, evaluates and selects the best matching model.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                               VotingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score,
                              average_precision_score, confusion_matrix,
                              classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ─── NDCG helper ───────────────────────────────────────────────────────────────

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)
    ideal_gains = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_gains / discounts[:len(ideal_gains)])
    return dcg / idcg if idcg > 0 else 0.0


def mean_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    yt_sorted = y_true[order]
    cumsum = np.cumsum(yt_sorted)
    precision_at_k = cumsum / (np.arange(len(yt_sorted)) + 1)
    return np.mean(precision_at_k[yt_sorted == 1]) if yt_sorted.sum() > 0 else 0.0


# ─── Model definitions ─────────────────────────────────────────────────────────

def get_models():
    """Return dict of all candidate models (unfitted)."""
    return {
        "Baseline_SkillOverlap": None,  # handled separately
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                       random_state=42))
        ]),
        "Random_Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8,
                                            class_weight="balanced", random_state=42))
        ]),
        "Gradient_Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                max_depth=4, subsample=0.8,
                                                random_state=42))
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=4,  # handles class imbalance
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            num_leaves=31, class_weight="balanced",
            random_state=42, verbose=-1
        ),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
                cv=3
            ))
        ]),
        "MLP_NeuralNet": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu", max_iter=300,
                early_stopping=True, validation_fraction=0.1,
                random_state=42
            ))
        ]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", AdaBoostClassifier(n_estimators=100, learning_rate=0.1,
                                        random_state=42))
        ]),
    }


# ─── Baseline (no model) ───────────────────────────────────────────────────────

def baseline_predict_proba(X: np.ndarray) -> np.ndarray:
    """Skill overlap ratio is column 0 (skill_match_ratio)."""
    proba = X[:, 0]  # skill_match_ratio
    return np.column_stack([1 - proba, proba])


# ─── Evaluation on one split ───────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy":   round(accuracy_score(y_true, y_pred), 4),
        "precision":  round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":    round(roc_auc_score(y_true, y_proba), 4),
        "avg_precision": round(average_precision_score(y_true, y_proba), 4),
        "ndcg_5":     round(ndcg_at_k(y_true, y_proba, k=5), 4),
        "map":        round(mean_average_precision(y_true, y_proba), 4),
        "fpr":        round(fp / max(fp + tn, 1), 4),
        "support":    int(y_true.sum()),
    }


# ─── Full training & comparison ────────────────────────────────────────────────

def train_and_compare(X_train, y_train, X_test, y_test, feature_names):
    """Train every model, evaluate, return sorted results."""
    models = get_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        t0 = time.time()

        if name == "Baseline_SkillOverlap":
            y_proba_test = baseline_predict_proba(X_test)[:, 1]
            y_pred_test = (y_proba_test >= 0.35).astype(int)
            metrics = evaluate(y_test, y_pred_test, y_proba_test)
            metrics["train_time_s"] = 0.0

        else:
            model.fit(X_train, y_train)
            y_proba_test = model.predict_proba(X_test)[:, 1]
            y_pred_test = model.predict(X_test)
            metrics = evaluate(y_test, y_pred_test, y_proba_test)
            metrics["train_time_s"] = round(time.time() - t0, 3)
            trained_models[name] = model

        results[name] = metrics
        print(f"  {name:<28} F1={metrics['f1']:.4f}  AUC={metrics['roc_auc']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"FPR={metrics['fpr']:.4f}")

    return results, trained_models


def select_best(results: dict) -> str:
    """Choose best model by harmonic mean of F1 + AUC-ROC."""
    def composite(m):
        return 2 * m["f1"] * m["roc_auc"] / max(m["f1"] + m["roc_auc"], 1e-6)

    candidates = {k: v for k, v in results.items() if k != "Baseline_SkillOverlap"}
    return max(candidates, key=lambda k: composite(candidates[k]))


def get_feature_importances(model, feature_names: list[str]) -> dict:
    """Extract feature importances from the best model (if available)."""
    clf = model
    # Unwrap pipeline
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)

    if hasattr(clf, "feature_importances_"):
        fi = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        fi = np.abs(clf.coef_[0])
    else:
        return {}

    total = fi.sum()
    return {
        name: round(float(val / total), 4)
        for name, val in sorted(zip(feature_names, fi),
                                 key=lambda x: x[1], reverse=True)
    }
