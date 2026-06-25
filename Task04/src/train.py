"""
PlaceMux — Explainability Task
Stage B.1-B.3 — Design baseline, implement & train multiple models, measure on
held-out real-shaped data. Stage C — pick a winner with numbers, not vibes.

Run:  python3 src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, accuracy_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(ROOT / "data" / "match_pairs.csv")
MODELS_DIR = str(ROOT / "models")
REPORTS_DIR = str(ROOT / "reports")
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "domain_match",
    "required_overlap_count",
    "required_overlap_ratio",
    "nice_overlap_count",
    "nice_overlap_ratio",
    "avg_required_skill_score",
    "min_required_skill_score",
    "meets_min_score_threshold",
    "years_experience",
    "min_years_required",
    "years_gap",
    "verified_skill_breadth",
]
TARGET_COL = "is_good_match"


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def baseline_predict(df, threshold=0.5):
    """Stage B.1 baseline: rank purely by required-skill overlap ratio.
    No verified score, no domain match, no experience — the thing every
    later models must beat."""
    return (df["required_overlap_ratio"] >= threshold).astype(int)


def eval_block(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "false_positive_rate": round(false_positive_rate(y_true, y_pred), 4),
    }
    if y_proba is not None:
        out["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
    return out


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # 60/20/20 train/val/test — test is touched exactly once, at the end.
    X_train, X_temp, y_train, y_temp, df_train, df_temp = train_test_split(
        X, y, df, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, df_val, df_test = train_test_split(
        X_temp, y_temp, df_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # ---------------- Baseline ----------------
    base_val_pred = baseline_predict(df_val)
    baseline_metrics = eval_block(y_val, base_val_pred)
    print("\n[Baseline: skill-overlap-only rule] validation metrics:", baseline_metrics)

    # ---------------- Candidate models (multiple models families) ----------------
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42, class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "svm_rbf": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=25),
    }

    experiment_log = []
    fitted = {}
    for name, model in candidates.items():
        t0 = time.time()
        if name in ("logistic_regression", "svm_rbf", "knn"):
            model.fit(X_train_s, y_train)
            val_pred = model.predict(X_val_s)
            val_proba = model.predict_proba(X_val_s)[:, 1]
        else:
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            val_proba = model.predict_proba(X_val)[:, 1]
        train_time = round(time.time() - t0, 3)

        metrics = eval_block(y_val, val_pred, val_proba)
        metrics.update({"models": name, "train_seconds": train_time})
        experiment_log.append(metrics)
        fitted[name] = model
        print(f"[{name}] val:", metrics)

    log_df = pd.DataFrame(experiment_log).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    log_df.to_csv(f"{REPORTS_DIR}/experiment_log.csv", index=False)

    # ---------------- Model selection ----------------
    # Selection rule (declared up front, not picked after peeking at test):
    # rank by ROC-AUC first (overall ranking quality), tie-break by F1
    # (balance of precision/recall), then prefer the models that also beats
    # the baseline's false-positive rate — a hiring product punishes FPs hard.
    log_df["selection_score"] = log_df["roc_auc"] * 0.6 + log_df["f1"] * 0.4
    best_name = log_df.sort_values("selection_score", ascending=False).iloc[0]["models"]
    best_model = fitted[best_name]
    print(f"\n>>> Selected best models: {best_name}")

    # ---------------- Final, single-touch test evaluation ----------------
    if best_name in ("logistic_regression", "svm_rbf", "knn"):
        test_pred = best_model.predict(X_test_s)
        test_proba = best_model.predict_proba(X_test_s)[:, 1]
    else:
        test_pred = best_model.predict(X_test)
        test_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = eval_block(y_test, test_pred, test_proba)
    base_test_pred = baseline_predict(df_test)
    base_test_metrics = eval_block(y_test, base_test_pred)

    print("\n[FINAL — held-out test] best models:", test_metrics)
    print("[FINAL — held-out test] baseline   :", base_test_metrics)

    # ---------------- Segment breakdown (no single-number-no-baseline pitfall) ----------------
    df_test = df_test.copy()
    df_test["pred"] = test_pred
    domain_lookup = pd.read_json(
        str(ROOT / "data" / "jobs.json")
    )[["job_id", "domain"]]
    df_test_m = df_test.merge(domain_lookup, on="job_id", how="left")
    seg_rows = []
    for dom, g in df_test_m.groupby("domain"):
        seg_rows.append({
            "domain": dom,
            "n": int(len(g)),
            **eval_block(g[TARGET_COL].values, g["pred"].values),
        })
    seg_df = pd.DataFrame(seg_rows)
    seg_df.to_csv(f"{REPORTS_DIR}/segment_breakdown.csv", index=False)
    print("\nSegment breakdown (held-out test):\n", seg_df)

    # ---------------- Persist everything ----------------
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.joblib")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.joblib")
    with open(f"{MODELS_DIR}/model_meta.json", "w") as f:
        json.dump({
            "best_model_name": best_name,
            "needs_scaling": best_name in ("logistic_regression", "svm_rbf", "knn"),
            "feature_cols": FEATURE_COLS,
            "validation_metrics": log_df.to_dict(orient="records"),
            "baseline_validation_metrics": baseline_metrics,
            "test_metrics_best_model": test_metrics,
            "test_metrics_baseline": base_test_metrics,
            "selection_rule": "0.6*ROC_AUC + 0.4*F1 on validation set, test set touched once at the end",
        }, f, indent=2)

    print("\nSaved: best_model.joblib, scaler.joblib, model_meta.json, experiment_log.csv, segment_breakdown.csv")


if __name__ == "__main__":
    main()
