"""
PlaceMux · Full Training Pipeline
Trains Logistic Regression → Random Forest → XGBoost → LightGBM
with hyperparameter tuning, SMOTE, and MLflow tracking.
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from utils.features import FEATURE_COLS
from models.baseline import SkillOverlapBaseline

MODELS_DIR = Path(__file__).parent
DATA_DIR   = Path(__file__).parent.parent / "data"
CV         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

LR_GRID = {
    "clf__C":        [0.01, 0.1, 1.0, 10.0],
    "clf__penalty":  ["l1", "l2"],
    "clf__solver":   ["liblinear"],
}

RF_GRID = {
    "n_estimators":    [100, 200],
    "max_depth":       [5, 10, None],
    "min_samples_leaf":[1, 3],
    "class_weight":    ["balanced"],
}

XGB_GRID = {
    "n_estimators":  [100, 200, 300],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample":     [0.7, 0.9],
    "colsample_bytree": [0.7, 1.0],
    "scale_pos_weight": [1, 2],
}

LGB_GRID = {
    "n_estimators":  [100, 200],
    "max_depth":     [4, 6, -1],
    "learning_rate": [0.05, 0.1],
    "num_leaves":    [15, 31],
    "class_weight":  ["balanced"],
}


def _metrics(y_true, y_pred, y_prob) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fpr = fp / max(fp + tn, 1)
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "fpr":       round(fpr, 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_true, y_prob), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def load_or_generate_data():
    csv_path = DATA_DIR / "sample_data.csv"
    if csv_path.exists():
        print(f"Loading data from {csv_path}")
        return pd.read_csv(csv_path)
    print("Generating synthetic data...")
    from data.generate_data import generate_dataset
    df = generate_dataset()
    df.to_csv(csv_path, index=False)
    return df


def train_logistic_regression(X_train, y_train, X_val, y_val) -> tuple:
    print("\n[1/4] Training Logistic Regression...")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
    gs   = GridSearchCV(pipe, LR_GRID, cv=CV, scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    m = _metrics(y_val, y_pred, y_prob)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Val F1={m['f1']}  Precision={m['precision']}  Recall={m['recall']}  FPR={m['fpr']}")
    return model, m


def train_random_forest(X_train, y_train, X_val, y_val) -> tuple:
    print("\n[2/4] Training Random Forest...")
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42),
        RF_GRID, cv=CV, scoring="f1", n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    m = _metrics(y_val, y_pred, y_prob)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Val F1={m['f1']}  Precision={m['precision']}  Recall={m['recall']}  FPR={m['fpr']}")
    return model, m


def train_xgboost(X_train, y_train, X_val, y_val) -> tuple:
    if not XGB_AVAILABLE:
        print("\n[3/4] XGBoost not installed — skipping.")
        return None, {}
    print("\n[3/4] Training XGBoost...")
    pos_weight = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    gs = GridSearchCV(
        XGBClassifier(
            random_state=42, use_label_encoder=False,
            eval_metric="logloss", verbosity=0,
            early_stopping_rounds=20,
        ),
        XGB_GRID, cv=CV, scoring="f1", n_jobs=-1, verbose=0,
    )
    gs.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    model = gs.best_estimator_
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    m = _metrics(y_val, y_pred, y_prob)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Val F1={m['f1']}  Precision={m['precision']}  Recall={m['recall']}  FPR={m['fpr']}")
    return model, m


def train_lightgbm(X_train, y_train, X_val, y_val) -> tuple:
    if not LGB_AVAILABLE:
        print("\n[4/4] LightGBM not installed — skipping.")
        return None, {}
    print("\n[4/4] Training LightGBM...")
    gs = GridSearchCV(
        LGBMClassifier(random_state=42, verbosity=-1),
        LGB_GRID, cv=CV, scoring="f1", n_jobs=-1, verbose=0,
    )
    gs.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[],
    )
    model = gs.best_estimator_
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    m = _metrics(y_val, y_pred, y_prob)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Val F1={m['f1']}  Precision={m['precision']}  Recall={m['recall']}  FPR={m['fpr']}")
    return model, m


def run_smote(X_train, y_train):
    if not SMOTE_AVAILABLE:
        print("  SMOTE not available — using class_weight balancing instead.")
        return X_train, y_train
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  SMOTE: {len(y_train)} → {len(y_res)} samples (balanced)")
    return X_res, y_res


def main():
    print("=" * 60)
    print("PlaceMux · Training Pipeline")
    print("=" * 60)

    # ---- Data ----
    df = load_or_generate_data()
    X  = df[FEATURE_COLS]
    y  = df["label"]

    # 60 / 20 / 20 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ---- Class balancing ----
    print("\nApplying SMOTE on training set...")
    X_train_bal, y_train_bal = run_smote(X_train, y_train)

    # ---- Baseline ----
    print("\n[0/4] Recording baseline (skill-overlap only)...")
    baseline = SkillOverlapBaseline()
    b_metrics = baseline.evaluate(X_test, y_test)
    print(f"  Baseline — F1={b_metrics['f1']}  Precision={b_metrics['precision']}"
          f"  Recall={b_metrics['recall']}  FPR={b_metrics['fpr']}")

    # ---- Train all models ----
    lr_model,  lr_m  = train_logistic_regression(X_train_bal, y_train_bal, X_val, y_val)
    rf_model,  rf_m  = train_random_forest(X_train_bal, y_train_bal, X_val, y_val)
    xgb_model, xgb_m = train_xgboost(X_train_bal, y_train_bal, X_val, y_val)
    lgb_model, lgb_m = train_lightgbm(X_train_bal, y_train_bal, X_val, y_val)

    # ---- Pick best by val F1 ----
    candidates = [
        ("LogisticRegression", lr_model,  lr_m),
        ("RandomForest",       rf_model,  rf_m),
        ("XGBoost",            xgb_model, xgb_m),
        ("LightGBM",           lgb_model, lgb_m),
    ]
    candidates = [(n, m, s) for n, m, s in candidates if m is not None and s]
    best_name, best_model, _ = max(candidates, key=lambda x: x[2].get("f1", 0))
    print(f"\n★ Best model on validation: {best_name}")

    # ---- Save all models ----
    artifact = {
        "best_model_name": best_name,
        "baseline_metrics": b_metrics,
        "val_metrics": {n: s for n, _, s in candidates},
        "feature_cols": FEATURE_COLS,
    }
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(lr_model,   MODELS_DIR / "lr_model.pkl")
    if rf_model:  joblib.dump(rf_model,  MODELS_DIR / "rf_model.pkl")
    if xgb_model: joblib.dump(xgb_model, MODELS_DIR / "xgb_model.pkl")
    if lgb_model: joblib.dump(lgb_model, MODELS_DIR / "lgb_model.pkl")

    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(artifact, f, indent=2)

    # Save test split for evaluate.py
    X_test.to_csv(MODELS_DIR / "X_test.csv", index=False)
    y_test.to_csv(MODELS_DIR / "y_test.csv", index=False)
    X_train.to_csv(MODELS_DIR / "X_train.csv", index=False)

    print(f"\nModels saved to {MODELS_DIR}")
    print("Run `python models/evaluate.py` for full test-set evaluation.")


if __name__ == "__main__":
    main()
