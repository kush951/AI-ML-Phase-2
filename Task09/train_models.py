"""
train_models.py
PlaceMux · Task 9 — Multi-model selection & evaluation
Models: Baseline, Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM
Metrics: Precision, Recall, F1, FPR, AUC-ROC
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR
MODEL_DIR = BASE_DIR
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "skill_coverage",
    "avg_score_on_required",
    "n_skills_matched",
    "nice_to_have_coverage",
    "edu_match",
    "exp_match",
    "loc_match",
    "student_avg_skill",
    "years_experience",
    "n_student_skills",
    "n_job_required",
]


# ─────────────────────────────────────────────────────────────
# BASELINE MODEL
# ─────────────────────────────────────────────────────────────
class SkillOverlapBaseline:
    """
    Simple rule-based baseline model.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.name = "Baseline (Skill Overlap)"

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return (X["skill_coverage"] >= self.threshold).astype(int).values

        return (X[:, 0] >= self.threshold).astype(int)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            scores = X["skill_coverage"].values
        else:
            scores = X[:, 0]

        return np.column_stack([1 - scores, scores])


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_proba=None):

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    fpr = fp / max(fp + tn, 1)

    metrics = {
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
        "fpr": round(fpr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    if y_proba is not None:
        metrics["auc_roc"] = round(
            roc_auc_score(y_true, y_proba[:, 1]), 4
        )

    return metrics


# ─────────────────────────────────────────────────────────────
# CONVERSION QUALITY CHECK
# ─────────────────────────────────────────────────────────────
def conversion_quality_check(df, y_true, y_pred):

    df = df.copy()

    df["_pred"] = y_pred
    df["_true"] = y_true

    results = {}

    for tier in ["paid", "free"]:

        mask = df["payment_tier"] == tier

        sub_true = df.loc[mask, "_true"]
        sub_pred = df.loc[mask, "_pred"]

        results[tier] = compute_metrics(
            sub_true,
            sub_pred
        )

    precision_gap = abs(
        results["paid"]["precision"] -
        results["free"]["precision"]
    )

    recall_gap = abs(
        results["paid"]["recall"] -
        results["free"]["recall"]
    )

    THRESHOLD = 0.08

    return {
        "paid_metrics": results["paid"],
        "free_metrics": results["free"],
        "precision_gap": round(precision_gap, 4),
        "recall_gap": round(recall_gap, 4),
        "threshold": THRESHOLD,
        "relevance_regression_detected":
            precision_gap > THRESHOLD or
            recall_gap > THRESHOLD,
    }


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def run_pipeline():

    print("=" * 60)
    print("PlaceMux · Task 9 · Multi-Model Training Pipeline")
    print("=" * 60)

    # LOAD DATA
    csv_path = DATA_DIR / "matches.csv"

    print(f"\nLoading dataset from:\n{csv_path}")

    df = pd.read_csv(csv_path)

    print(
        f"\n[DATA] {len(df)} pairs | "
        f"{df['label'].mean():.1%} positive rate"
    )

    X = df[FEATURE_COLS]
    y = df["label"]

    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )

    print(
        f"[SPLIT] "
        f"Train={len(X_train)} | "
        f"Val={len(X_val)} | "
        f"Test={len(X_test)}"
    )

    # SCALE
    scaler = StandardScaler()

    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # MODELS
    models = {

        "Baseline (Skill Overlap)": SkillOverlapBaseline(
            threshold=0.5
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42
        ),
    }

    all_results = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    print("\n[TRAINING] Fitting models...\n")

    for name, model in models.items():

        print(f"→ {name}")

        if name == "Baseline (Skill Overlap)":

            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)

            cv_f1 = None

        else:

            model.fit(X_train_s, y_train)

            y_pred_test = model.predict(X_test_s)
            y_proba_test = model.predict_proba(X_test_s)

            cv_scores = cross_val_score(
                model,
                X_train_s,
                y_train,
                cv=cv,
                scoring="f1",
                n_jobs=-1
            )

            cv_f1 = round(cv_scores.mean(), 4)

        test_metrics = compute_metrics(
            y_test,
            y_pred_test,
            y_proba_test
        )

        all_results[name] = {
            "test": test_metrics,
            "cv_f1": cv_f1,
        }

        print(
            f"F1={test_metrics['f1']} | "
            f"AUC={test_metrics.get('auc_roc', 'N/A')}"
        )

    # BEST MODEL
    ml_models = {
        k: v for k, v in all_results.items()
        if k != "Baseline (Skill Overlap)"
    }

    best_name = max(
        ml_models,
        key=lambda k: ml_models[k]["test"]["f1"]
    )

    best_model = models[best_name]

    print(
        f"\n[BEST MODEL] "
        f"{best_name} | "
        f"F1={all_results[best_name]['test']['f1']}"
    )

    # CONVERSION CHECK
    y_pred_all = best_model.predict(
        scaler.transform(X)
    )

    conv_check = conversion_quality_check(
        df,
        y,
        y_pred_all
    )

    print("\n[CONVERSION CHECK]")
    print(conv_check)

    # SAVE MODELS
    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.pkl"
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    joblib.dump(
        models,
        MODEL_DIR / "all_models.pkl"
    )

    # LOG
    experiment_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "best_model": best_name,
        "results": all_results,
        "conversion_quality": conv_check,
    }

    log_path = LOG_DIR / "experiment_log.json"

    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)

    print(f"\n[LOG SAVED] {log_path}")

    print("\nTraining completed successfully!")

    return experiment_log


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()